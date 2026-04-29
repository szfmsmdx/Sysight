"""Profile-health summaries for deep Nsight Systems SQL analysis."""

from __future__ import annotations

import logging
import sqlite3

from .extract import find_gaps, intersect_total, union_ns
from .models import NsysFinding
from .sql_shared import _get_cols, _is_nccl, _kernel_name_expr

logger = logging.getLogger(__name__)


def _sql_profile_health(
    findings: list[NsysFinding],
    conn: sqlite3.Connection,
    kernel_tbl: str,
    runtime_tbl: str | None,
    memcpy_tbl: str | None,
    nvtx_tbl: str | None,
    has_strings: bool,
    total_ns: int,
) -> None:
    """汇总 profile 关键指标并推断主要瓶颈。"""
    del nvtx_tbl
    denominator_ns = max(int(total_ns or 0), 1)

    top_kernel_name = ""
    top_kernel_ms = 0.0
    total_kernel_ms = 0.0
    compute_intervals: list[tuple[int, int]] = []
    nccl_intervals: list[tuple[int, int]] = []
    all_gpu_intervals: list[tuple[int, int]] = []

    try:
        name_expr, join_clause = _kernel_name_expr(conn, kernel_tbl, has_strings, alias="k")
        agg_rows = conn.execute(
            f"""
            SELECT {name_expr} AS kname,
                   ROUND(SUM(k.[end] - k.start) / 1e6, 2) AS total_ms
            FROM {kernel_tbl} k {join_clause}
            GROUP BY kname ORDER BY total_ms DESC LIMIT 5
            """
        ).fetchall()
        if agg_rows:
            top_kernel_name = agg_rows[0][0] or "?"
            top_kernel_ms = float(agg_rows[0][1] or 0)
        total_row = conn.execute(
            f"SELECT ROUND(SUM([end] - start) / 1e6, 2) FROM {kernel_tbl}"
        ).fetchone()
        total_kernel_ms = float(total_row[0] or 0) if total_row else 0.0

        interval_rows = conn.execute(
            f"""
            SELECT k.start, k.[end], {name_expr} AS kname
            FROM {kernel_tbl} k {join_clause}
            WHERE k.[end] > k.start
            """
        ).fetchall()
        for row in interval_rows:
            interval = (int(row["start"]), int(row["end"]))
            name = str(row["kname"] or "")
            all_gpu_intervals.append(interval)
            if _is_nccl(name):
                nccl_intervals.append(interval)
            else:
                compute_intervals.append(interval)
    except sqlite3.Error as exc:
        logger.debug("_sql_profile_health (top_kernels): %s", exc)

    if memcpy_tbl:
        try:
            for row in conn.execute(
                f"SELECT start, [end] FROM {memcpy_tbl} WHERE [end] > start"
            ).fetchall():
                all_gpu_intervals.append((int(row["start"]), int(row["end"])))
        except sqlite3.Error as exc:
            logger.debug("_sql_profile_health (memcpy intervals): %s", exc)

    trace_start_ns = min((start for start, _ in all_gpu_intervals), default=0)
    trace_end_ns = max((end for _, end in all_gpu_intervals), default=denominator_ns)
    observed_span_ns = max(0, trace_end_ns - trace_start_ns)
    denominator_ns = max(denominator_ns, observed_span_ns, 1)

    compute_union_ns = union_ns(compute_intervals)
    nccl_union_ns = union_ns(nccl_intervals)
    overlap_ns = intersect_total(compute_intervals, nccl_intervals)
    nccl_only_ms = max(0, nccl_union_ns - overlap_ns) / 1e6
    compute_only_ms = max(0, compute_union_ns - overlap_ns) / 1e6
    overlap_pct = 100.0 if nccl_union_ns <= 0 else round(overlap_ns / nccl_union_ns * 100, 1)

    idle_pct = 0.0
    gap_count = 0
    if all_gpu_intervals:
        gpu_active_ns = min(union_ns(all_gpu_intervals), denominator_ns)
        idle_ns = max(0, denominator_ns - gpu_active_ns)
        idle_pct = round(idle_ns / denominator_ns * 100, 1)
        gap_end_ns = trace_start_ns + denominator_ns
        gap_count = len(find_gaps(all_gpu_intervals, trace_start_ns, gap_end_ns, min_gap_ns=1_000_000))

    sync_density_pct = 0.0
    sync_inclusive_pct = 0.0
    if runtime_tbl and has_strings:
        try:
            if "nameId" in _get_cols(conn, runtime_tbl):
                sync_names = conn.execute(
                    """
                    SELECT id FROM StringIds
                    WHERE value LIKE 'cudaDeviceSynchronize%'
                       OR value LIKE 'cudaStreamSynchronize%'
                       OR value LIKE 'cudaEventSynchronize%'
                       OR value LIKE 'cudaStreamWaitEvent%'
                    """
                ).fetchall()
                if sync_names:
                    ph = ",".join(str(row[0]) for row in sync_names)
                    sync_rows = conn.execute(
                        f"""
                        SELECT start, [end] FROM {runtime_tbl}
                        WHERE nameId IN ({ph}) AND [end] > start
                        """
                    ).fetchall()
                    sync_intervals = [(int(row["start"]), int(row["end"])) for row in sync_rows]
                    sync_union = min(union_ns(sync_intervals), denominator_ns)
                    sync_inclusive = sum(end - start for start, end in sync_intervals)
                    sync_density_pct = round(sync_union / denominator_ns * 100, 1)
                    sync_inclusive_pct = round(sync_inclusive / denominator_ns * 100, 1)
        except sqlite3.Error as exc:
            logger.debug("_sql_profile_health (sync_density): %s", exc)

    if sync_density_pct > 20.0:
        suspected_bottleneck = f"高 CPU 同步阻塞（{sync_density_pct:.1f}% wall）"
    elif nccl_union_ns > 0 and overlap_pct < 30.0:
        suspected_bottleneck = f"NCCL 序列化（union 重叠率仅 {overlap_pct:.1f}%）"
    elif idle_pct > 15.0:
        suspected_bottleneck = f"GPU 空闲气泡（{idle_pct:.1f}% wall，{gap_count} 个全局间隙）"
    elif total_kernel_ms > 0 and top_kernel_ms / total_kernel_ms > 0.6:
        suspected_bottleneck = f"Kernel 热点：{top_kernel_name[:40]}（{top_kernel_ms / total_kernel_ms * 100:.0f}%）"
    elif nccl_only_ms > compute_only_ms and compute_only_ms > 0:
        suspected_bottleneck = "通信主导（NCCL 独占 > 计算独占）"
    else:
        suspected_bottleneck = "计算主导（无明显序列化瓶颈）"

    trace_ms = denominator_ns / 1e6
    evidence = [
        f"Trace wall：{trace_ms:.1f}ms，GPU kernel 总计(incl)：{total_kernel_ms:.1f}ms",
        (
            f"Top kernel：{top_kernel_name[:55]}（{top_kernel_ms:.1f}ms，{top_kernel_ms / total_kernel_ms * 100:.1f}%）"
            if total_kernel_ms > 0
            else f"Top kernel：{top_kernel_name[:55]}（{top_kernel_ms:.1f}ms）"
        ),
        f"NCCL 独占(union)：{nccl_only_ms:.1f}ms，计算独占(union)：{compute_only_ms:.1f}ms，NCCL 重叠率(union)：{overlap_pct:.1f}%",
        f"GPU 空闲(global union)：{idle_pct:.1f}%（{gap_count} 个 >1ms 全局间隙）",
        f"同步等待(runtime union)：{sync_density_pct:.1f}% wall"
        + (f"；inclusive {sync_inclusive_pct:.1f}%" if sync_inclusive_pct else ""),
        f"→ 疑似主要瓶颈：{suspected_bottleneck}",
    ]

    severity = "warning" if idle_pct > 10 or sync_density_pct > 15 or overlap_pct < 30 else "info"
    findings.append(NsysFinding(
        category="sql_profile_health",
        severity=severity,
        title=f"Profile 健康摘要：{suspected_bottleneck}",
        description=(
            f"Profile 全局健康概览（trace {trace_ms:.1f}ms）。"
            f"GPU 空闲 {idle_pct:.1f}% wall，NCCL 重叠率 {overlap_pct:.1f}% union，"
            f"同步等待 {sync_density_pct:.1f}% wall。"
            f"疑似主要瓶颈：{suspected_bottleneck}。"
        ),
        evidence=evidence,
        next_step=(
            "以 suspected_bottleneck 为首要优化方向。"
            "查看 sql_nvtx_layer_breakdown 获取精确代码层级归因，"
            "查看 sql_root_cause_analysis 获取具体反模式修复建议。"
        ),
    ))
