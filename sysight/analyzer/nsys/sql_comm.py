"""Communication-oriented deep Nsight Systems SQL analyzers."""

from __future__ import annotations

import logging
import sqlite3

from .models import NsysFinding
from .sql_shared import _NCCL_KEYWORDS, _fmt_table

logger = logging.getLogger(__name__)


def _sql_nccl_breakdown(
    findings: list[NsysFinding],
    conn: sqlite3.Connection,
    kernel_tbl: str,
) -> None:
    """NCCL 集合通信操作按 stream 分解分析。"""
    like_clauses = " OR ".join(f"LOWER(s.value) LIKE '%{keyword}%'" for keyword in _NCCL_KEYWORDS)

    sql = f"""
        SELECT k.streamId,
               COUNT(*) AS op_count,
               ROUND(SUM(k.[end] - k.start) / 1e6, 2) AS total_ms,
               ROUND(AVG(k.[end] - k.start) / 1e6, 3) AS avg_ms,
               ROUND(MAX(k.[end] - k.start) / 1e6, 3) AS max_ms,
               ROUND(MIN(k.[end] - k.start) / 1e6, 3) AS min_ms
        FROM {kernel_tbl} k
        JOIN StringIds s ON k.shortName = s.id
        WHERE ({like_clauses})
        GROUP BY k.streamId
        ORDER BY total_ms DESC
        LIMIT 20
    """
    try:
        rows = conn.execute(sql).fetchall()
    except sqlite3.Error as e:
        logger.debug("sql_nccl_breakdown 失败：%s", e)
        return
    if not rows:
        return

    total_nccl_ms = sum(float(row["total_ms"] or 0) for row in rows)
    total_ops = sum(int(row["op_count"] or 0) for row in rows)

    tbl_rows = [
        [
            str(row["streamId"]),
            str(row["op_count"]),
            f"{row['total_ms']:.2f}",
            f"{row['avg_ms']:.3f}",
            f"{row['max_ms']:.3f}",
            f"{row['min_ms']:.3f}",
        ]
        for row in rows
    ]
    ev_lines = _fmt_table(["Stream", "次数", "总(ms)", "均(ms)", "最大(ms)", "最小(ms)"], tbl_rows)

    if len(rows) > 1:
        max_stream_ms = float(rows[0]["total_ms"] or 0)
        min_stream_ms = float(rows[-1]["total_ms"] or 0)
        imbalance_pct = (max_stream_ms - min_stream_ms) / max_stream_ms * 100 if max_stream_ms > 0 else 0
        if imbalance_pct > 30:
            ev_lines.append(
                f"⚠️ Stream 负载不均衡：最大 {max_stream_ms:.1f}ms vs 最小 {min_stream_ms:.1f}ms"
                f"（差异 {imbalance_pct:.0f}%）"
            )

    findings.append(NsysFinding(
        category="sql_nccl_breakdown",
        severity="info",
        title=f"NCCL 通信分解：{total_ops} 次操作，{total_nccl_ms:.1f}ms，{len(rows)} 个 stream",
        description=(
            f"NCCL 集合操作（AllReduce/AllGather/ReduceScatter 等）共 {total_ops} 次，"
            f"总计 {total_nccl_ms:.1f}ms，分布在 {len(rows)} 个 CUDA stream 上。"
            "各 stream 通常对应不同的并行维度（DP/TP/PP）。"
        ),
        evidence=ev_lines[:8],
        next_step=(
            "检查各 stream 的 NCCL 时间是否均衡。不均衡可能意味着某个并行维度的通信量过大。"
            "使用 overlap_breakdown 分析 NCCL 与计算的重叠情况。"
        ),
    ))
