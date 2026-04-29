"""Synchronization-oriented deep Nsight Systems SQL analyzers."""

from __future__ import annotations

import logging
import sqlite3

from .extract import union_ns
from .models import NsysFinding
from .sql_shared import _find_table, _fmt_table

logger = logging.getLogger(__name__)


def _sql_sync_cost(
    findings: list[NsysFinding],
    conn: sqlite3.Connection,
    sync_tbl: str,
    total_ns: int,
) -> None:
    """分析 CUPTI 同步事件的 wall-clock 代价。"""
    sync_enum_tbl = _find_table(
        {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()},
        "ENUM_CUPTI_SYNC_TYPE",
    )

    if sync_enum_tbl:
        sql = f"""
            SELECT COALESCE(e.name, 'Unknown') AS sync_type_name,
                   COUNT(*) AS call_count,
                   ROUND(SUM(s.[end] - s.start) / 1e6, 2) AS total_ms,
                   ROUND(AVG(s.[end] - s.start) / 1e6, 3) AS avg_ms,
                   ROUND(MAX(s.[end] - s.start) / 1e6, 3) AS max_ms
            FROM {sync_tbl} s
            LEFT JOIN {sync_enum_tbl} e ON s.syncType = e.id
            GROUP BY sync_type_name ORDER BY total_ms DESC
        """
    else:
        sql = f"""
            SELECT CAST(syncType AS TEXT) AS sync_type_name,
                   COUNT(*) AS call_count,
                   ROUND(SUM([end] - start) / 1e6, 2) AS total_ms,
                   ROUND(AVG([end] - start) / 1e6, 3) AS avg_ms,
                   ROUND(MAX([end] - start) / 1e6, 3) AS max_ms
            FROM {sync_tbl}
            GROUP BY syncType ORDER BY total_ms DESC
        """
    try:
        rows = conn.execute(sql).fetchall()
    except sqlite3.Error as e:
        logger.debug("sql_sync_cost 失败：%s", e)
        return
    if not rows:
        return

    total_sync_ms = sum(float(row["total_ms"] or 0) for row in rows)
    total_calls = sum(int(row["call_count"] or 0) for row in rows)
    total_trace_ms = max(total_ns / 1e6, 0.001)
    sync_inclusive_pct = total_sync_ms / total_trace_ms * 100

    sync_wall_ms = 0.0
    sync_wall_pct = 0.0
    try:
        interval_rows = conn.execute(
            f"SELECT start, [end] FROM {sync_tbl} WHERE [end] > start"
        ).fetchall()
        intervals = [(int(row["start"]), int(row["end"])) for row in interval_rows]
        sync_wall_ns = min(union_ns(intervals), int(total_ns or 0))
        sync_wall_ms = sync_wall_ns / 1e6
        sync_wall_pct = sync_wall_ms / total_trace_ms * 100
    except sqlite3.Error:
        sync_wall_ms = total_sync_ms
        sync_wall_pct = sync_inclusive_pct

    tbl_rows = [
        [
            row["sync_type_name"] or "Unknown",
            str(row["call_count"]),
            f"{row['total_ms']:.2f}",
            f"{row['avg_ms']:.3f}",
            f"{row['max_ms']:.3f}",
        ]
        for row in rows
    ]
    ev_lines = _fmt_table(["同步类型", "次数", "总(ms)", "均(ms)", "最大(ms)"], tbl_rows)

    severity = "warning" if sync_wall_pct > 10 else "info"
    findings.append(NsysFinding(
        category="sql_sync_cost",
        severity=severity,
        title=f"同步代价（CUPTI）：{sync_wall_ms:.1f}ms wall，占 trace {sync_wall_pct:.1f}%",
        description=(
            f"CUPTI 同步事件共 {total_calls} 次，union wall 代价 {sync_wall_ms:.1f}ms，"
            f"占 trace 时长的 {sync_wall_pct:.1f}%。"
            f"inclusive 总和为 {total_sync_ms:.1f}ms（{sync_inclusive_pct:.1f}%），"
            "可因多线程/多 stream 重叠而超过 100%。"
        ),
        evidence=ev_lines[:7] + [
            f"同步 wall-union：{sync_wall_ms:.1f}ms（{sync_wall_pct:.1f}% trace）",
            f"同步 inclusive：{total_sync_ms:.1f}ms（{sync_inclusive_pct:.1f}% trace，可重叠）",
        ],
        next_step=(
            "减少 cudaDeviceSynchronize 调用（使用流级别同步代替）。"
            "对于必要的同步，尽量将其安排在 GPU 空闲时执行以降低影响。"
        ),
    ))
