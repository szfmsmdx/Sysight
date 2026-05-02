"""nsys_sql.sync — CUDA synchronization event cost analysis."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from sysight.tools.registry import ToolDef
from sysight.tools.nsys_sql._helpers import _open_db, _find_table, _table_bounds, _union_ns


@dataclass
class SyncEventInfo:
    sync_type: str
    count: int
    total_ns: int
    avg_ns: float
    max_ns: int


@dataclass
class SyncResult:
    sync_events: list[SyncEventInfo] = field(default_factory=list)
    total_sync_ns: int = 0
    sync_wall_pct: float = 0.0


def sync(sqlite: str) -> SyncResult:
    """Query CUDA synchronization events and their wall-clock cost."""
    result = SyncResult()

    with _open_db(sqlite) as (conn, all_tables, has_strings):
        sync_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION")
        if not sync_tbl:
            return result

        kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
        trace_duration_ns = 0
        if kernel_tbl:
            s, e, _ = _table_bounds(conn, kernel_tbl)
            if e > s:
                trace_duration_ns = e - s

        # Build query
        enum_tbl = _find_table(all_tables, "ENUM_CUPTI_SYNC_TYPE")
        if enum_tbl and has_strings:
            sql = f"""SELECT COALESCE(e.name, 'sync_'||s.syncType) AS sync_name,
                       COUNT(*) AS cnt, SUM(s.[end]-s.start) AS total_ns,
                       AVG(s.[end]-s.start) AS avg_ns, MAX(s.[end]-s.start) AS max_ns
                FROM {sync_tbl} s LEFT JOIN {enum_tbl} e ON s.syncType=e.id
                GROUP BY sync_name ORDER BY total_ns DESC"""
        else:
            sql = f"""SELECT 'sync_'||CAST(syncType AS TEXT) AS sync_name,
                       COUNT(*) AS cnt, SUM([end]-start) AS total_ns,
                       AVG([end]-start) AS avg_ns, MAX([end]-start) AS max_ns
                FROM {sync_tbl} GROUP BY sync_name ORDER BY total_ns DESC"""

        try:
            for row in conn.execute(sql):
                ns = int(row["total_ns"] or 0)
                result.total_sync_ns += ns
                result.sync_events.append(SyncEventInfo(
                    sync_type=row["sync_name"] or "unknown", count=int(row["cnt"]),
                    total_ns=ns, avg_ns=float(row["avg_ns"] or 0),
                    max_ns=int(row["max_ns"] or 0),
                ))
        except sqlite3.Error:
            pass

        if trace_duration_ns > 0:
            try:
                intervals = [(int(r["start"]), int(r["end"])) for r in
                             conn.execute(f"SELECT start,[end] FROM {sync_tbl} WHERE [end]>start").fetchall()]
                wall_ns = min(_union_ns(intervals), trace_duration_ns)
                result.sync_wall_pct = round(wall_ns / trace_duration_ns * 100, 1)
            except sqlite3.Error:
                pass

    return result


SYNC_TOOL = ToolDef(
    name="nsys_sql_sync",
    description="Query CUDA synchronization events (cudaDeviceSynchronize, cudaStreamSynchronize, etc.) and their wall-clock cost",
    parameters={"type": "object", "properties": {"sqlite": {"type": "string"}}, "required": ["sqlite"]},
    fn=sync, read_only=True,
)
