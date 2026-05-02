"""nsys_sql.nccl — NCCL communication breakdown per stream."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from sysight.tools.registry import ToolDef
from sysight.tools.nsys_sql._helpers import _open_db, _find_table, _kernel_rows_query, _nccl_name_sql


@dataclass
class NcclStreamInfo:
    stream_id: int
    op_count: int
    total_ns: int
    avg_ns: float


@dataclass
class NcclResult:
    streams: list[NcclStreamInfo] = field(default_factory=list)
    total_nccl_ns: int = 0
    total_ops: int = 0


def nccl(sqlite: str, limit: int = 20) -> NcclResult:
    """Query NCCL collective communication operations per stream."""
    result = NcclResult()

    with _open_db(sqlite) as (conn, all_tables, has_strings):
        kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
        if not kernel_tbl:
            return result

        nccl_sql = _nccl_name_sql()
        sql = f"""SELECT streamId, COUNT(*) AS op_count,
                   SUM([end]-start) AS total_ns, AVG([end]-start) AS avg_ns
            FROM ({_kernel_rows_query(conn, kernel_tbl, has_strings)})
            WHERE {nccl_sql}
            GROUP BY streamId ORDER BY total_ns DESC LIMIT {limit}"""

        try:
            for row in conn.execute(sql):
                ops = int(row["op_count"]); ns = int(row["total_ns"] or 0)
                result.total_ops += ops; result.total_nccl_ns += ns
                result.streams.append(NcclStreamInfo(
                    stream_id=int(row["streamId"]), op_count=ops,
                    total_ns=ns, avg_ns=float(row["avg_ns"] or 0),
                ))
        except sqlite3.Error:
            pass

    return result


NCCL_TOOL = ToolDef(
    name="nsys_sql_nccl",
    description="Query NCCL collective communication operations (all-reduce, all-gather, etc.) per stream",
    parameters={
        "type": "object",
        "properties": {"sqlite": {"type": "string"}, "limit": {"type": "integer", "default": 20}},
        "required": ["sqlite"],
    },
    fn=nccl, read_only=True,
)
