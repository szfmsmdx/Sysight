"""nsys_sql.memcpy — Memory bandwidth analysis (H2D, D2H, D2D, P2P)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from sysight.tools.registry import ToolDef
from sysight.tools.nsys_sql._helpers import _open_db, _find_table, _has_columns, _COPY_KIND_NAMES


@dataclass
class MemcpyInfo:
    direction: str
    count: int
    total_bytes: int
    total_ns: int
    avg_bw_gbps: float


@dataclass
class MemcpyResult:
    memcpy_ops: list[MemcpyInfo] = field(default_factory=list)
    total_bytes: int = 0
    total_ns: int = 0


def memcpy(sqlite: str) -> MemcpyResult:
    """Query memory copy operations with direction, bytes, and estimated bandwidth."""
    result = MemcpyResult()

    with _open_db(sqlite) as (conn, all_tables, _):
        memcpy_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_MEMCPY")
        if not memcpy_tbl:
            return result
        if not _has_columns(conn, memcpy_tbl, "bytes", "copyKind"):
            return result

        sql = f"""SELECT copyKind, COUNT(*) AS cnt, SUM(bytes) AS total_bytes,
                   SUM([end]-start) AS total_ns,
                   CASE WHEN SUM([end]-start)>0 THEN SUM(bytes)/(SUM([end]-start)/1e9)/1e9 ELSE 0 END AS avg_bw
            FROM {memcpy_tbl} GROUP BY copyKind ORDER BY total_bytes DESC"""

        try:
            for row in conn.execute(sql):
                b = int(row["total_bytes"] or 0)
                ns = int(row["total_ns"] or 0)
                result.total_bytes += b
                result.total_ns += ns
                kind = int(row["copyKind"] or 0)
                result.memcpy_ops.append(MemcpyInfo(
                    direction=_COPY_KIND_NAMES.get(kind, f"Kind{kind}"),
                    count=int(row["cnt"]), total_bytes=b, total_ns=ns,
                    avg_bw_gbps=float(row["avg_bw"] or 0),
                ))
        except sqlite3.Error:
            pass

    return result


MEMCPY_TOOL = ToolDef(
    name="nsys_sql_memcpy",
    description="Query memory copy operations (H2D, D2H, D2D, P2P) with direction, bytes, and estimated bandwidth",
    parameters={"type": "object", "properties": {"sqlite": {"type": "string"}}, "required": ["sqlite"]},
    fn=memcpy, read_only=True,
)
