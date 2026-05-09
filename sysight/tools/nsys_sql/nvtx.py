"""nsys_sql.nvtx — NVTX range aggregation for Nsight Systems SQLite files."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from sysight.tools.registry import ToolDef
from sysight.tools.nsys_sql._helpers import _find_table, _open_db


@dataclass
class NvtxRangeInfo:
    name: str
    count: int
    total_ns: int
    avg_ns: float
    max_ns: int
    pct_of_trace: float
    kernel_count: int = 0
    runtime_count: int = 0
    memcpy_count: int = 0
    sync_count: int = 0


@dataclass
class NvtxResult:
    ranges: list[NvtxRangeInfo] = field(default_factory=list)
    trace_duration_ns: int = 0


def nvtx(sqlite: str, limit: int = 50, include_counts: bool = True) -> NvtxResult:
    """Aggregate completed NVTX ranges by name and count contained CUDA events.

    Set include_counts=False to skip the expensive contained-event counting
    (useful for large profiles where the JOIN queries are too slow).
    """
    result = NvtxResult()

    with _open_db(sqlite) as (conn, all_tables, has_strings):
        nvtx_tbl = _find_table(all_tables, "NVTX_EVENTS")
        if not nvtx_tbl:
            return result

        name_expr = "n.text"
        join = ""
        if has_strings:
            str_tbl = _find_table(all_tables, "StringIds")
            if str_tbl:
                join = f" LEFT JOIN {str_tbl} s ON n.textId=s.id"
                name_expr = "COALESCE(n.text, s.value)"

        where = "n.[end] IS NOT NULL AND n.[end] > n.start"
        try:
            bounds = conn.execute(
                f"SELECT MIN(start), MAX([end]) FROM {nvtx_tbl} n WHERE {where}"
            ).fetchone()
            if bounds and bounds[0] is not None and bounds[1] is not None:
                result.trace_duration_ns = int(bounds[1]) - int(bounds[0])
        except sqlite3.Error:
            result.trace_duration_ns = 0

        sql = f"""
            SELECT {name_expr} AS range_name,
                   COUNT(*) AS cnt,
                   SUM(n.[end]-n.start) AS total_ns,
                   AVG(n.[end]-n.start) AS avg_ns,
                   MAX(n.[end]-n.start) AS max_ns
            FROM {nvtx_tbl} n {join}
            WHERE {where}
            GROUP BY range_name
            ORDER BY total_ns DESC
            LIMIT ?
        """

        rows = []
        try:
            rows = conn.execute(sql, (limit,)).fetchall()
        except sqlite3.Error:
            return result

        for row in rows:
            name = row["range_name"] or "unknown"
            total_ns = int(row["total_ns"] or 0)
            counts: dict[str, int] = {}
            if include_counts:
                kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
                runtime_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_RUNTIME")
                memcpy_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_MEMCPY")
                sync_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION")
                counts = _contained_counts_batched(
                    conn, nvtx_tbl, name_expr, join, where, name,
                    kernel_tbl, runtime_tbl, memcpy_tbl, sync_tbl,
                )
            result.ranges.append(NvtxRangeInfo(
                name=name,
                count=int(row["cnt"] or 0),
                total_ns=total_ns,
                avg_ns=float(row["avg_ns"] or 0),
                max_ns=int(row["max_ns"] or 0),
                pct_of_trace=round(total_ns / result.trace_duration_ns * 100, 1)
                if result.trace_duration_ns else 0.0,
                kernel_count=counts.get("kernel", 0),
                runtime_count=counts.get("runtime", 0),
                memcpy_count=counts.get("memcpy", 0),
                sync_count=counts.get("sync", 0),
            ))

    return result


def _contained_counts_batched(
    conn: sqlite3.Connection,
    nvtx_tbl: str,
    name_expr: str,
    join: str,
    where: str,
    range_name: str,
    kernel_tbl: str | None,
    runtime_tbl: str | None,
    memcpy_tbl: str | None,
    sync_tbl: str | None,
) -> dict[str, int]:
    """Count contained CUDA events for a given NVTX range name using batched JOINs.

    Instead of looping over every NVTX instance and running per-interval COUNT
    queries (which is O(instances × tables)), we use a single SQL query per
    event table that JOINs the NVTX ranges with the event table on the time
    containment condition.
    """
    counts: dict[str, int] = {}

    table_map = {
        "kernel": kernel_tbl,
        "runtime": runtime_tbl,
        "memcpy": memcpy_tbl,
        "sync": sync_tbl,
    }

    for label, event_tbl in table_map.items():
        if not event_tbl:
            continue
        try:
            row = conn.execute(
                f"SELECT COUNT(*) FROM {nvtx_tbl} n {join} "
                f"INNER JOIN {event_tbl} e "
                f"ON e.start >= n.start AND e.[end] <= n.[end] "
                f"WHERE {where} AND {name_expr} = ?",
                (range_name,),
            ).fetchone()
            counts[label] = int(row[0] or 0) if row else 0
        except sqlite3.Error:
            counts[label] = 0

    return counts


NVTX_TOOL = ToolDef(
    name="nsys_sql_nvtx",
    description="Aggregate NVTX ranges from an Nsight Systems SQLite profile and count contained CUDA events",
    parameters={
        "type": "object",
        "properties": {
            "sqlite": {"type": "string", "description": "Path to .sqlite file"},
            "limit": {"type": "integer", "default": 50},
        },
        "required": ["sqlite"],
    },
    fn=nvtx,
    read_only=True,
)