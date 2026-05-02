"""Shared SQL helpers for Nsight Systems profile queries.

Extracted from nsys-ai sql_cli.py and sql_shared.py.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing, contextmanager
from collections.abc import Iterator

_COPY_KIND_NAMES = {0: "Unknown", 1: "H2D", 2: "D2H", 4: "H2H", 8: "D2D", 10: "P2P"}
_NCCL_KEYWORDS = ("nccl", "allreduce", "allgather", "reducescatter", "broadcast", "sendrecv", "reduce")


def _find_table(all_tables: set[str], prefix: str) -> str | None:
    if prefix in all_tables:
        return prefix
    for t in sorted(all_tables):
        if t.startswith(prefix):
            return t
    return None


def _get_cols(conn: sqlite3.Connection, tbl: str) -> list[str]:
    try:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({tbl})").fetchall()]
    except sqlite3.Error:
        return []


def _load_tables(conn: sqlite3.Connection) -> tuple[set[str], bool]:
    all_tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    return all_tables, "StringIds" in all_tables


@contextmanager
def _open_db(sqlite_path: str) -> Iterator[tuple[sqlite3.Connection, set[str], bool]]:
    with closing(sqlite3.connect(sqlite_path)) as conn:
        conn.row_factory = sqlite3.Row
        all_tables, has_strings = _load_tables(conn)
        yield conn, all_tables, has_strings


def _has_columns(conn: sqlite3.Connection, tbl: str, *cols: str) -> bool:
    existing = set(_get_cols(conn, tbl))
    return all(c in existing for c in cols)


def _table_bounds(conn: sqlite3.Connection, tbl: str, *, include_total: bool = False) -> tuple[int, int, int]:
    select_total = ", SUM([end] - start)" if include_total else ""
    try:
        row = conn.execute(f"SELECT MIN(start), MAX([end]){select_total} FROM {tbl}").fetchone()
    except sqlite3.Error:
        return 0, 0, 0
    if not row or row[0] is None:
        return 0, 0, 0
    total = int(row[2] or 0) if include_total else 0
    return int(row[0]), int(row[1]), total


def _kernel_name_expr(conn: sqlite3.Connection, kernel_tbl: str, has_strings: bool, alias: str = "k") -> tuple[str, str]:
    cols = _get_cols(conn, kernel_tbl)
    short_col = "shortName" if "shortName" in cols else ""
    demangled_col = "demangledName" if "demangledName" in cols else ""

    if has_strings and short_col:
        ref = f"{alias}.{short_col}"
        if demangled_col:
            dref = f"{alias}.{demangled_col}"
            return (
                f"COALESCE(d.value, s.value, CAST({ref} AS TEXT))",
                f"LEFT JOIN StringIds s ON {ref}=s.id LEFT JOIN StringIds d ON {dref}=d.id",
            )
        return (f"COALESCE(s.value, CAST({ref} AS TEXT))", f"LEFT JOIN StringIds s ON {ref}=s.id")

    if demangled_col:
        return f"CAST({alias}.{demangled_col} AS TEXT)", ""
    if short_col:
        return f"CAST({alias}.{short_col} AS TEXT)", ""
    return "'unknown'", ""


def _kernel_rows_query(conn: sqlite3.Connection, kernel_tbl: str, has_strings: bool) -> str:
    name_expr, join_clause = _kernel_name_expr(conn, kernel_tbl, has_strings, alias="k")
    return f"""SELECT k.streamId, k.start, k.[end], {name_expr} AS kernel_name
FROM {kernel_tbl} k {join_clause}"""


def _nccl_name_sql(column: str = "kernel_name") -> str:
    norm = f"LOWER(COALESCE({column}, ''))"
    return " OR ".join(f"{norm} LIKE '%{kw}%'" for kw in _NCCL_KEYWORDS)


def _kernel_name_lookup(conn: sqlite3.Connection, kernel_tbl: str, has_strings: bool,
                        where: str, order: str) -> str:
    name_expr, join_clause = _kernel_name_expr(conn, kernel_tbl, has_strings, alias="k")
    return f"""SELECT {name_expr} AS kernel_name
FROM {kernel_tbl} k {join_clause}
WHERE {where} {order} LIMIT 1"""


def _stream_span_stats(conn: sqlite3.Connection, kernel_tbl: str, stream_ids: set[int]) -> tuple[int, int, int, int]:
    if not stream_ids:
        return 0, 0, 0, 0
    placeholders = ",".join("?" * len(stream_ids))
    try:
        row = conn.execute(f"""SELECT MIN(start), MAX([end]), SUM([end]-start), COUNT(*)
FROM {kernel_tbl} WHERE streamId IN ({placeholders})""", list(stream_ids)).fetchone()
    except sqlite3.Error:
        return 0, 0, 0, 0
    if not row or row[0] is None:
        return 0, 0, 0, 0
    return int(row[0]), int(row[1]), int(row[2] or 0), int(row[3] or 0)


def _stream_class_query(conn: sqlite3.Connection, kernel_tbl: str, has_strings: bool) -> str:
    nccl_sql = _nccl_name_sql()
    return f"""SELECT streamId,
  SUM(CASE WHEN {nccl_sql} THEN 1 ELSE 0 END) AS nccl_count,
  SUM(CASE WHEN NOT ({nccl_sql}) THEN 1 ELSE 0 END) AS compute_count
FROM ({_kernel_rows_query(conn, kernel_tbl, has_strings)})
GROUP BY streamId"""


def _union_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    sorted_iv = sorted(intervals)
    total = 0
    cur_start, cur_end = sorted_iv[0]
    for s, e in sorted_iv[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = s, e
    total += cur_end - cur_start
    return total
