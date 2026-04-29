"""Compute-oriented deep Nsight Systems SQL analyzers."""

from __future__ import annotations

import logging
import sqlite3

from .models import NsysFinding
from .sql_shared import _fmt_table, _get_cols

logger = logging.getLogger(__name__)


def _sql_top_kernels(
    findings: list[NsysFinding],
    conn: sqlite3.Connection,
    kernel_tbl: str,
    has_strings: bool,
    total_ns: int,
    limit: int = 15,
) -> None:
    """按总执行时间列出 Top-N GPU 内核。"""
    cols = _get_cols(conn, kernel_tbl)
    has_demangled = "demangledName" in cols

    if has_strings and has_demangled:
        name_expr = "COALESCE(d.value, s.value, 'kernel_' || CAST(k.shortName AS TEXT)) AS kernel_name"
        join_clause = "LEFT JOIN StringIds s ON k.shortName = s.id LEFT JOIN StringIds d ON k.demangledName = d.id"
    elif has_strings:
        name_expr = "COALESCE(s.value, 'kernel_' || CAST(k.shortName AS TEXT)) AS kernel_name"
        join_clause = "LEFT JOIN StringIds s ON k.shortName = s.id"
    else:
        name_expr = "CAST(k.shortName AS TEXT) AS kernel_name"
        join_clause = ""

    sql = f"""
        SELECT {name_expr},
               COUNT(*) AS invocations,
               ROUND(SUM(k.[end] - k.start) / 1e6, 2) AS total_ms,
               ROUND(AVG(k.[end] - k.start) / 1e6, 3) AS avg_ms,
               ROUND(MIN(k.[end] - k.start) / 1e6, 3) AS min_ms,
               ROUND(MAX(k.[end] - k.start) / 1e6, 3) AS max_ms
        FROM {kernel_tbl} k {join_clause}
        GROUP BY kernel_name ORDER BY total_ms DESC LIMIT {limit}
    """
    try:
        rows = conn.execute(sql).fetchall()
    except sqlite3.Error as e:
        logger.debug("sql_top_kernels 失败：%s", e)
        return
    if not rows:
        return

    total_ms = max(total_ns / 1e6, 0.001)
    tbl_rows = []
    for row in rows:
        name = row["kernel_name"] or "unknown"
        short = name[:58] + ".." if len(name) > 60 else name
        tbl_rows.append([
            short,
            str(row["invocations"]),
            f"{row['total_ms']:.2f}",
            f"{row['avg_ms']:.3f}",
            f"{row['max_ms']:.3f}",
        ])
    ev_lines = _fmt_table(["内核名称", "次数", "总(ms)", "均(ms)", "最大(ms)"], tbl_rows)

    top_name = rows[0]["kernel_name"] or "unknown"
    top_ms = float(rows[0]["total_ms"])
    top_pct = top_ms / total_ms * 100

    findings.append(NsysFinding(
        category="sql_top_kernels",
        severity="info",
        title=f"SQL Top 内核：{top_name[:55]}（{top_ms:.1f}ms，{top_pct:.1f}%）",
        description=(
            f"按总执行时间排列的 Top-{min(limit, len(rows))} GPU 内核。"
            f"最热内核 '{top_name[:40]}' 累计 {top_ms:.1f}ms，"
            f"占 trace {top_pct:.1f}%，共调用 {rows[0]['invocations']} 次。"
        ),
        evidence=ev_lines[:9],
        related_hotspots=[row["kernel_name"] or "" for row in rows[:5]],
        next_step=(
            "关注累计时间最高的内核。使用 Nsight Compute (ncu) 对 Top-3 内核做指标分析，"
            "检查 SM 利用率、内存带宽效率和 Tensor Core 使用率。"
        ),
    ))


def _sql_gpu_idle_gaps(
    findings: list[NsysFinding],
    conn: sqlite3.Connection,
    kernel_tbl: str,
    runtime_tbl: str | None,
    has_strings: bool,
    min_gap_ns: int = 1_000_000,
    limit: int = 15,
    nvtx_tbl: str | None = None,
) -> None:
    """查找 GPU stream 内空闲间隙（气泡），含 CPU API 归因和 NVTX 重叠标注。"""
    agg_sql = f"""
        WITH ordered AS (
            SELECT k.streamId, k.start, k.[end],
                   LAG(k.[end]) OVER (PARTITION BY k.deviceId, k.streamId ORDER BY k.start) AS prev_end
            FROM {kernel_tbl} k
        ), gaps AS (
            SELECT streamId, prev_end AS gs, start AS ge, (start - prev_end) AS gn
            FROM ordered WHERE prev_end IS NOT NULL AND (start - prev_end) > {min_gap_ns}
        )
        SELECT COUNT(*) AS gap_count,
               ROUND(SUM(gn) / 1e6, 2) AS total_idle_ms,
               SUM(CASE WHEN gn <= 5000000 THEN 1 ELSE 0 END) AS gaps_tiny,
               SUM(CASE WHEN gn BETWEEN 5000001 AND 50000000 THEN 1 ELSE 0 END) AS gaps_medium,
               SUM(CASE WHEN gn > 50000000 THEN 1 ELSE 0 END) AS gaps_large
        FROM gaps
    """
    try:
        agg_row = conn.execute(agg_sql).fetchone()
        if not agg_row:
            return
        agg = dict(agg_row)
    except sqlite3.Error as e:
        logger.debug("sql_gpu_idle_gaps agg 失败：%s", e)
        return

    if not agg.get("gap_count"):
        return

    if has_strings:
        top_sql = f"""
            WITH ordered AS (
                SELECT k.streamId, k.deviceId, k.start, k.[end], s.value AS kn,
                       LAG(k.[end]) OVER (PARTITION BY k.deviceId, k.streamId ORDER BY k.start) AS prev_end,
                       LAG(s.value) OVER (PARTITION BY k.deviceId, k.streamId ORDER BY k.start) AS pk
                FROM {kernel_tbl} k JOIN StringIds s ON k.shortName = s.id
            )
            SELECT streamId, deviceId, prev_end AS gap_start_ns, start AS gap_end_ns,
                   (start - prev_end) AS gap_ns, pk AS before_kernel, kn AS after_kernel
            FROM ordered WHERE prev_end IS NOT NULL AND (start - prev_end) > {min_gap_ns}
            ORDER BY gap_ns DESC LIMIT {limit}
        """
    else:
        top_sql = f"""
            WITH ordered AS (
                SELECT k.streamId, k.deviceId, k.start, k.[end],
                       LAG(k.[end]) OVER (PARTITION BY k.deviceId, k.streamId ORDER BY k.start) AS prev_end
                FROM {kernel_tbl} k
            )
            SELECT streamId, deviceId, prev_end AS gap_start_ns, start AS gap_end_ns,
                   (start - prev_end) AS gap_ns, NULL AS before_kernel, NULL AS after_kernel
            FROM ordered WHERE prev_end IS NOT NULL AND (start - prev_end) > {min_gap_ns}
            ORDER BY gap_ns DESC LIMIT {limit}
        """
    try:
        top_rows = [dict(row) for row in conn.execute(top_sql).fetchall()]
    except sqlite3.Error as e:
        logger.debug("sql_gpu_idle_gaps top 失败：%s", e)
        top_rows = []

    gap_count = agg.get("gap_count", 0)
    total_idle_ms = float(agg.get("total_idle_ms") or 0)
    gaps_tiny = agg.get("gaps_tiny", 0)
    gaps_medium = agg.get("gaps_medium", 0)
    gaps_large = agg.get("gaps_large", 0)

    gap_evidence = [
        f"共 {gap_count} 个 stream 内间隙，per-stream 总空闲 {total_idle_ms:.1f}ms",
        f"分布：{gaps_tiny} 个 <5ms，{gaps_medium} 个 5-50ms，{gaps_large} 个 >50ms",
    ]

    for gap in top_rows[:5]:
        gap_ms = gap.get("gap_ns", 0) / 1e6
        stream_id = gap.get("streamId", "?")
        before = (gap.get("before_kernel") or "(起始)")[:30]
        attr_info = ""
        nvtx_info = ""
        gs = gap.get("gap_start_ns")
        ge = gap.get("gap_end_ns")
        if gs is not None and ge is not None:
            if runtime_tbl and has_strings:
                try:
                    api_rows = conn.execute(f"""
                        SELECT s.value AS api_name, ROUND(SUM(r.[end] - r.start) / 1e6, 2) AS total_ms
                        FROM {runtime_tbl} r JOIN StringIds s ON r.nameId = s.id
                        WHERE r.start < {ge} AND r.[end] > {gs}
                        GROUP BY s.value ORDER BY total_ms DESC LIMIT 3
                    """).fetchall()
                    if api_rows:
                        top_api = api_rows[0]["api_name"] or ""
                        top_api_ms = float(api_rows[0]["total_ms"] or 0)
                        attr_info = f" [CPU: {top_api} {top_api_ms:.1f}ms]"
                except sqlite3.Error:
                    pass
            if nvtx_tbl:
                try:
                    nvtx_cols = _get_cols(conn, nvtx_tbl)
                    has_textid = "textId" in nvtx_cols
                    if has_strings and has_textid:
                        nvtx_label_expr = "COALESCE(n.text, s.value)"
                        nvtx_join = "LEFT JOIN StringIds s ON n.textId = s.id"
                    else:
                        nvtx_label_expr = "n.text"
                        nvtx_join = ""
                    nvtx_rows = conn.execute(f"""
                        SELECT DISTINCT {nvtx_label_expr} AS label
                        FROM {nvtx_tbl} n {nvtx_join}
                        WHERE n.[end] > {gs} AND n.start < {ge}
                          AND {nvtx_label_expr} IS NOT NULL
                          AND {nvtx_label_expr} != ''
                        LIMIT 4
                    """).fetchall()
                    if nvtx_rows:
                        labels = [str(row["label"]) for row in nvtx_rows if row["label"]]
                        nvtx_info = f" [NVTX: {', '.join(labels)}]"
                except sqlite3.Error:
                    pass
        gap_evidence.append(
            f"Stream {stream_id}：{gap_ms:.2f}ms 空闲 | 前驱内核: {before}{attr_info}{nvtx_info}"
        )

    severity = "warning" if total_idle_ms > 50 or gaps_large > 0 else "info"
    findings.append(NsysFinding(
        category="sql_gpu_idle_gaps",
        severity=severity,
        title=f"GPU stream 空闲气泡：{gap_count} 个间隙，per-stream 总计 {total_idle_ms:.1f}ms",
        description=(
            f"在各 CUDA stream 内共发现 {gap_count} 个 GPU 空闲间隙（>{min_gap_ns/1e6:.0f}ms），"
            f"per-stream 累计 {total_idle_ms:.1f}ms。该值可能超过 trace wall time；"
            f"其中 {gaps_large} 个超过 50ms 的大间隙是主要优化目标。"
        ),
        evidence=gap_evidence,
        next_step=(
            "大间隙（>50ms）通常是 CPU 同步、DataLoader 阻塞或 PCIe 传输导致的。"
            "检查间隙期间的 CUDA Runtime API 活动以定位原因。"
            "考虑使用 CUDA graphs 或流水线重叠来填补这些气泡。"
        ),
    ))
