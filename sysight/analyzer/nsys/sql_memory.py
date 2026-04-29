"""Memory-oriented deep Nsight Systems SQL analyzers."""

from __future__ import annotations

import logging
import sqlite3

from .models import NsysFinding
from .sql_shared import _COPY_KIND_NAMES, _fmt_table, _get_cols

logger = logging.getLogger(__name__)


def _sql_memory_bandwidth(
    findings: list[NsysFinding],
    conn: sqlite3.Connection,
    memcpy_tbl: str,
) -> None:
    """分析各方向内存带宽（H2D/D2H/D2D/P2P）。"""
    cols = _get_cols(conn, memcpy_tbl)
    if "bytes" not in cols or "copyKind" not in cols:
        return

    sql = f"""
        SELECT copyKind,
               COUNT(*) AS op_count,
               ROUND(SUM(bytes) / 1e6, 2) AS total_mb,
               ROUND(AVG(bytes) / 1e3, 1) AS avg_kb,
               ROUND(SUM([end] - start) / 1e6, 2) AS total_dur_ms,
               ROUND(AVG([end] - start) / 1e3, 1) AS avg_dur_us,
               CASE WHEN SUM([end] - start) > 0
                    THEN ROUND(SUM(bytes) / (SUM([end] - start) / 1e9) / 1e9, 2)
                    ELSE 0 END AS avg_bw_gbps,
               COALESCE(ROUND(MAX(CASE WHEN ([end] - start) > 0
                    THEN bytes / (([end] - start) / 1e9) / 1e9 END), 2), 0) AS peak_bw_gbps
        FROM {memcpy_tbl}
        GROUP BY copyKind ORDER BY total_mb DESC
    """
    try:
        rows = conn.execute(sql).fetchall()
    except sqlite3.Error as e:
        logger.debug("sql_memory_bandwidth 失败：%s", e)
        return
    if not rows:
        return

    total_mb = sum(float(row["total_mb"] or 0) for row in rows)
    total_dur_ms = sum(float(row["total_dur_ms"] or 0) for row in rows)

    low_bw_warn: list[str] = []
    tbl_rows = []
    for row in rows:
        kind = _COPY_KIND_NAMES.get(row["copyKind"], f"Kind{row['copyKind']}")
        avg_bw = float(row["avg_bw_gbps"] or 0)
        peak_bw = float(row["peak_bw_gbps"] or 0)
        tbl_rows.append([
            kind,
            str(row["op_count"]),
            f"{row['total_mb']:.2f}",
            f"{row['avg_kb']:.1f}",
            f"{row['total_dur_ms']:.2f}",
            f"{avg_bw:.2f}",
            f"{peak_bw:.2f}",
        ])
        if kind == "H2D" and avg_bw > 0 and avg_bw < 5.0:
            low_bw_warn.append(f"HtoD 均带宽仅 {avg_bw:.1f}GB/s（<5GB/s，可能未使用 pin memory）")
        elif kind == "D2D" and avg_bw > 0 and avg_bw < 50.0:
            low_bw_warn.append(f"D2D 均带宽仅 {avg_bw:.1f}GB/s（<50GB/s，可能跨 NUMA）")
    ev_lines = _fmt_table(["方向", "次数", "总MB", "均KB", "总时长(ms)", "均带宽(GB/s)", "峰值(GB/s)"], tbl_rows)
    evidence = ev_lines[:8] + low_bw_warn

    severity = "warning" if total_dur_ms > 100 else "info"

    findings.append(NsysFinding(
        category="sql_memory_bandwidth",
        severity=severity,
        title=f"内存带宽分析：总计 {total_mb:.1f}MB，{total_dur_ms:.1f}ms",
        description=(
            f"CUDA 内存拷贝操作总计 {total_mb:.1f}MB，耗时 {total_dur_ms:.1f}ms。"
            + (f" 检测到低带宽警告：{'; '.join(low_bw_warn)}" if low_bw_warn else "")
        ),
        evidence=evidence,
        next_step=(
            "若 H2D 带宽低，检查是否使用了 cudaHostAlloc (pin memory)。"
            "若 D2D 带宽低，检查 NUMA 拓扑和 NVLink 连接。"
            "使用 cudaMemcpyAsync 并行化内存拷贝与计算。"
        ),
    ))
