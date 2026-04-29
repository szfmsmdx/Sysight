"""Shared helpers for deep Nsight Systems SQL analyzers."""

from __future__ import annotations

import sqlite3

from .text import format_table

_COPY_KIND_NAMES = {0: "Unknown", 1: "H2D", 2: "D2H", 4: "H2H", 8: "D2D", 10: "P2P"}

_NCCL_KEYWORDS = ("nccl", "allreduce", "allgather", "reducescatter", "broadcast", "sendrecv", "reduce")


def _find_table(all_tables: set[str], prefix: str) -> str | None:
    if prefix in all_tables:
        return prefix
    for table_name in sorted(all_tables):
        if table_name.startswith(prefix):
            return table_name
    return None


def _get_cols(conn: sqlite3.Connection, tbl: str) -> list[str]:
    try:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({tbl})").fetchall()]
    except sqlite3.Error:
        return []


def _is_nccl(name: str) -> bool:
    low = name.lower()
    return any(keyword in low for keyword in _NCCL_KEYWORDS)


def _fmt_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """将表头+数据行格式化为等宽对齐字符串列表。"""
    return format_table(headers, rows)


def _kernel_name_expr(
    conn: sqlite3.Connection,
    kernel_tbl: str,
    has_strings: bool,
    alias: str = "k",
) -> tuple[str, str]:
    """Return (name_expr, join_clause) for kernel names across schema variants."""
    cols = _get_cols(conn, kernel_tbl)
    short_col = "shortName" if "shortName" in cols else ""
    demangled_col = "demangledName" if "demangledName" in cols else ""

    if has_strings and short_col:
        short_ref = f"{alias}.{short_col}"
        if demangled_col:
            demangled_ref = f"{alias}.{demangled_col}"
            return (
                f"COALESCE(d.value, s.value, CAST({short_ref} AS TEXT))",
                f"LEFT JOIN StringIds s ON {short_ref} = s.id "
                f"LEFT JOIN StringIds d ON {demangled_ref} = d.id",
            )
        return (
            f"COALESCE(s.value, CAST({short_ref} AS TEXT))",
            f"LEFT JOIN StringIds s ON {short_ref} = s.id",
        )

    if demangled_col:
        return f"CAST({alias}.{demangled_col} AS TEXT)", ""
    if short_col:
        return f"CAST({alias}.{short_col} AS TEXT)", ""
    return "'unknown_kernel'", ""
