"""nsys_sql.kernels — Top-N GPU kernels by total time."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from sysight.tools.registry import ToolDef
from sysight.tools.nsys_sql._helpers import _open_db, _find_table, _kernel_name_expr, _table_bounds


@dataclass
class KernelInfo:
    name: str
    count: int
    total_ns: int
    avg_ns: float
    max_ns: int


@dataclass
class KernelsResult:
    kernels: list[KernelInfo] = field(default_factory=list)
    total_kernel_ns: int = 0
    trace_duration_ns: int = 0


def kernels(sqlite: str, limit: int = 20) -> KernelsResult:
    """Query top-N GPU kernels by total time from an Nsight Systems SQLite profile."""
    result = KernelsResult()

    with _open_db(sqlite) as (conn, all_tables, has_strings):
        kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
        if not kernel_tbl:
            return result

        name_expr, join_clause = _kernel_name_expr(conn, kernel_tbl, has_strings, alias="k")

        trace_start, trace_end, total_kernel_ns = _table_bounds(conn, kernel_tbl, include_total=True)
        if trace_end > trace_start:
            result.trace_duration_ns = trace_end - trace_start
        result.total_kernel_ns = total_kernel_ns

        sql = f"""
            SELECT {name_expr} AS kernel_name,
                   COUNT(*) AS cnt, SUM(k.[end]-k.start) AS total_ns,
                   AVG(k.[end]-k.start) AS avg_ns, MAX(k.[end]-k.start) AS max_ns
            FROM {kernel_tbl} k {join_clause}
            GROUP BY kernel_name ORDER BY total_ns DESC LIMIT {limit}
        """
        try:
            for row in conn.execute(sql).fetchall():
                result.kernels.append(KernelInfo(
                    name=row["kernel_name"] or "unknown",
                    count=int(row["cnt"]), total_ns=int(row["total_ns"] or 0),
                    avg_ns=float(row["avg_ns"] or 0), max_ns=int(row["max_ns"] or 0),
                ))
        except sqlite3.Error:
            pass

    return result


KERNELS_TOOL = ToolDef(
    name="nsys_sql_kernels",
    description="Query top-N GPU kernels by total execution time from an Nsight Systems SQLite profile",
    parameters={
        "type": "object",
        "properties": {
            "sqlite": {"type": "string", "description": "Path to .sqlite file"},
            "limit": {"type": "integer", "default": 20},
        },
        "required": ["sqlite"],
    },
    fn=kernels,
    read_only=True,
)
