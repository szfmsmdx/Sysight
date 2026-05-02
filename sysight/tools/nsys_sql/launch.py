"""nsys_sql.launch — Kernel launch overhead (API → GPU latency)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from sysight.tools.registry import ToolDef
from sysight.tools.nsys_sql._helpers import _open_db, _find_table, _has_columns, _kernel_name_expr


@dataclass
class KernelLaunchEntry:
    kernel_name: str
    api_ms: float
    kernel_ms: float
    overhead_us: float


@dataclass
class LaunchResult:
    entries: list[KernelLaunchEntry] = field(default_factory=list)
    avg_overhead_us: float = 0.0
    max_overhead_us: float = 0.0


def launch(sqlite: str, limit: int = 20) -> LaunchResult:
    """Query kernel launch overhead — gap between CUDA API call and GPU execution start."""
    result = LaunchResult()

    with _open_db(sqlite) as (conn, all_tables, has_strings):
        kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
        runtime_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_RUNTIME")
        if not kernel_tbl or not runtime_tbl:
            return result
        if not _has_columns(conn, kernel_tbl, "correlationId"):
            return result
        if not _has_columns(conn, runtime_tbl, "correlationId"):
            return result

        name_expr, kernel_join = _kernel_name_expr(conn, kernel_tbl, has_strings, alias="k")
        sql = f"""SELECT {name_expr} AS kernel_name,
                   ROUND((r.[end]-r.start)/1e6,3) AS api_ms,
                   ROUND((k.[end]-k.start)/1e6,3) AS kernel_ms,
                   ROUND((k.start-r.start)/1e3,1) AS overhead_us
            FROM {runtime_tbl} r JOIN {kernel_tbl} k ON r.correlationId=k.correlationId
            {kernel_join}
            WHERE k.start>=r.start ORDER BY overhead_us DESC LIMIT {limit}"""

        try:
            for row in conn.execute(sql):
                result.entries.append(KernelLaunchEntry(
                    kernel_name=row["kernel_name"] or "unknown",
                    api_ms=float(row["api_ms"] or 0),
                    kernel_ms=float(row["kernel_ms"] or 0),
                    overhead_us=float(row["overhead_us"] or 0),
                ))
        except sqlite3.Error:
            pass

    if result.entries:
        result.avg_overhead_us = sum(e.overhead_us for e in result.entries) / len(result.entries)
        result.max_overhead_us = max(e.overhead_us for e in result.entries)

    return result


LAUNCH_TOOL = ToolDef(
    name="nsys_sql_launch",
    description="Query kernel launch overhead — the gap between CUDA runtime API call and kernel execution start on GPU",
    parameters={
        "type": "object",
        "properties": {"sqlite": {"type": "string"}, "limit": {"type": "integer", "default": 20}},
        "required": ["sqlite"],
    },
    fn=launch, read_only=True,
)
