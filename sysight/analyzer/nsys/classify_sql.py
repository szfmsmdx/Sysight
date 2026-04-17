"""nsys/classify_sql.py — 深度 SQLite 直接分析（移植自 nsys-ai skills）。

本模块由 classify.py 的 classify_bottlenecks() 在有 sqlite_path 时调用。
包含以下分析，对标 nsys-ai 的同名 skill：

  sql_top_kernels       Top-N GPU 内核（按总时间）
  sql_gpu_idle_gaps     GPU 流内的空闲气泡（含 CPU API 归因）
  sql_memory_bandwidth  各方向内存带宽统计（H2D/D2H/D2D/P2P）
  sql_nccl_breakdown    NCCL 集合通信按流分解
  sql_sync_cost         CUPTI 同步事件 wall-clock 代价
  sql_nvtx_hotspots     NVTX 注释段热点分布
"""

from __future__ import annotations

import logging
import sqlite3

from .models import NsysFinding, NsysTrace
from .text import format_table

logger = logging.getLogger(__name__)

_COPY_KIND_NAMES = {0: "Unknown", 1: "H2D", 2: "D2H", 4: "H2H", 8: "D2D", 10: "P2P"}

_NCCL_KEYWORDS = ("nccl", "allreduce", "allgather", "reducescatter", "broadcast", "sendrecv", "reduce")


# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════

def run_deep_sql_analysis(findings: list[NsysFinding], trace: NsysTrace) -> None:
    """深度 SQL 分析入口，由 classify_bottlenecks() 调用。"""
    sqlite_path = trace.sqlite_path
    if not sqlite_path:
        return
    try:
        with sqlite3.connect(sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            all_tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            has_strings = "StringIds" in all_tables

            kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
            runtime_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_RUNTIME")
            memcpy_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_MEMCPY")
            sync_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION")
            nvtx_tbl = _find_table(all_tables, "NVTX_EVENTS")

            if kernel_tbl:
                _sql_top_kernels(findings, conn, kernel_tbl, has_strings, trace.duration_ns)
                _sql_gpu_idle_gaps(findings, conn, kernel_tbl, runtime_tbl, has_strings,
                                   nvtx_tbl=nvtx_tbl)
                if has_strings:
                    _sql_nccl_breakdown(findings, conn, kernel_tbl)

            if memcpy_tbl:
                _sql_memory_bandwidth(findings, conn, memcpy_tbl)

            if sync_tbl:
                _sql_sync_cost(findings, conn, sync_tbl, trace.duration_ns)

            if nvtx_tbl:
                _sql_nvtx_hotspots(findings, conn, nvtx_tbl, has_strings, trace.duration_ns)

    except sqlite3.Error as e:
        logger.warning("深度 SQL 分析失败（非致命）：%s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _find_table(all_tables: set[str], prefix: str) -> str | None:
    if prefix in all_tables:
        return prefix
    for t in sorted(all_tables):
        if t.startswith(prefix):
            return t
    return None


def _get_cols(conn: sqlite3.Connection, tbl: str) -> list[str]:
    try:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({tbl})").fetchall()]
    except sqlite3.Error:
        return []


def _is_nccl(name: str) -> bool:
    low = name.lower()
    return any(kw in low for kw in _NCCL_KEYWORDS)


def _fmt_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """将表头+数据行格式化为等宽对齐字符串列表（第一行是表头，第二行分隔线，后续是数据）。"""
    return format_table(headers, rows)


# ═══════════════════════════════════════════════════════════════════════════════
# sql_top_kernels（移植自 nsys-ai top_kernels skill）
# ═══════════════════════════════════════════════════════════════════════════════

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
    for r in rows:
        name = r["kernel_name"] or "unknown"
        short = name[:58] + ".." if len(name) > 60 else name
        tbl_rows.append([short, str(r["invocations"]), f"{r['total_ms']:.2f}", f"{r['avg_ms']:.3f}", f"{r['max_ms']:.3f}"])
    ev_lines = _fmt_table(["内核名称", "次数", "总(ms)", "均(ms)", "最大(ms)"], tbl_rows)

    top_name = rows[0]["kernel_name"] or "unknown"
    top_ms = float(rows[0]["total_ms"])
    top_pct = top_ms / total_ms * 100

    findings.append(NsysFinding(
        category="sql_top_kernels",
        severity="info",
        confidence=0.95,
        title=f"SQL Top 内核：{top_name[:55]}（{top_ms:.1f}ms，{top_pct:.1f}%）",
        description=(
            f"按总执行时间排列的 Top-{min(limit, len(rows))} GPU 内核。"
            f"最热内核 '{top_name[:40]}' 累计 {top_ms:.1f}ms，"
            f"占 trace {top_pct:.1f}%，共调用 {rows[0]['invocations']} 次。"
        ),
        evidence=ev_lines[:9],
        related_hotspots=[r["kernel_name"] or "" for r in rows[:5]],
        next_step=(
            "关注累计时间最高的内核。使用 Nsight Compute (ncu) 对 Top-3 内核做指标分析，"
            "检查 SM 利用率、内存带宽效率和 Tensor Core 使用率。"
        ),
    ))


# ═══════════════════════════════════════════════════════════════════════════════
# sql_gpu_idle_gaps（移植自 nsys-ai gpu_idle_gaps skill）
# ═══════════════════════════════════════════════════════════════════════════════

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

    # 获取 Top 间隙
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
        top_rows = [dict(r) for r in conn.execute(top_sql).fetchall()]
    except sqlite3.Error as e:
        logger.debug("sql_gpu_idle_gaps top 失败：%s", e)
        top_rows = []

    gap_count = agg.get("gap_count", 0)
    total_idle_ms = float(agg.get("total_idle_ms") or 0)
    gaps_tiny = agg.get("gaps_tiny", 0)
    gaps_medium = agg.get("gaps_medium", 0)
    gaps_large = agg.get("gaps_large", 0)

    gap_evidence = [
        f"共 {gap_count} 个间隙，总空闲 {total_idle_ms:.1f}ms",
        f"分布：{gaps_tiny} 个 <5ms，{gaps_medium} 个 5-50ms，{gaps_large} 个 >50ms",
    ]

    # CPU API 归因 + NVTX 重叠：为 Top-5 间隙查询 CUDA runtime API 和 NVTX 范围
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
            # NVTX ranges overlapping this gap
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
                        labels = [str(r["label"]) for r in nvtx_rows if r["label"]]
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
        confidence=0.90,
        title=f"GPU 空闲气泡：{gap_count} 个间隙，总计 {total_idle_ms:.1f}ms",
        description=(
            f"在各 CUDA stream 内共发现 {gap_count} 个 GPU 空闲间隙（>{min_gap_ns/1e6:.0f}ms），"
            f"总计 {total_idle_ms:.1f}ms。其中 {gaps_large} 个超过 50ms 的大间隙是主要优化目标。"
        ),
        evidence=gap_evidence,
        next_step=(
            "大间隙（>50ms）通常是 CPU 同步、DataLoader 阻塞或 PCIe 传输导致的。"
            "检查间隙期间的 CUDA Runtime API 活动以定位原因。"
            "考虑使用 CUDA graphs 或流水线重叠来填补这些气泡。"
        ),
    ))


# ═══════════════════════════════════════════════════════════════════════════════
# sql_memory_bandwidth（移植自 nsys-ai memory_bandwidth skill）
# ═══════════════════════════════════════════════════════════════════════════════

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

    total_mb = sum(float(r["total_mb"] or 0) for r in rows)
    total_dur_ms = sum(float(r["total_dur_ms"] or 0) for r in rows)

    low_bw_warn = []
    tbl_rows = []
    for r in rows:
        kind = _COPY_KIND_NAMES.get(r["copyKind"], f"Kind{r['copyKind']}")
        avg_bw = float(r["avg_bw_gbps"] or 0)
        peak_bw = float(r["peak_bw_gbps"] or 0)
        tbl_rows.append([kind, str(r["op_count"]), f"{r['total_mb']:.2f}", f"{r['avg_kb']:.1f}",
                          f"{r['total_dur_ms']:.2f}", f"{avg_bw:.2f}", f"{peak_bw:.2f}"])
        if kind == "H2D" and avg_bw > 0 and avg_bw < 5.0:
            low_bw_warn.append(f"HtoD 均带宽仅 {avg_bw:.1f}GB/s（<5GB/s，可能未使用 pin memory）")
        elif kind == "D2D" and avg_bw > 0 and avg_bw < 50.0:
            low_bw_warn.append(f"D2D 均带宽仅 {avg_bw:.1f}GB/s（<50GB/s，可能跨 NUMA）")
    ev_lines = _fmt_table(["方向", "次数", "总MB", "均KB", "总时长(ms)", "均带宽(GB/s)", "峰值(GB/s)"], tbl_rows)
    evidence = ev_lines[:8] + low_bw_warn

    # 是否有大量拷贝
    severity = "warning" if total_dur_ms > 100 else "info"

    findings.append(NsysFinding(
        category="sql_memory_bandwidth",
        severity=severity,
        confidence=0.90,
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


# ═══════════════════════════════════════════════════════════════════════════════
# sql_nccl_breakdown（移植自 nsys-ai nccl_breakdown skill）
# ═══════════════════════════════════════════════════════════════════════════════

def _sql_nccl_breakdown(
    findings: list[NsysFinding],
    conn: sqlite3.Connection,
    kernel_tbl: str,
) -> None:
    """NCCL 集合通信操作按 stream 分解分析。"""
    # 构建 NCCL 关键字的 LIKE 条件
    like_clauses = " OR ".join(f"LOWER(s.value) LIKE '%{kw}%'" for kw in _NCCL_KEYWORDS)

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

    total_nccl_ms = sum(float(r["total_ms"] or 0) for r in rows)
    total_ops = sum(int(r["op_count"] or 0) for r in rows)

    tbl_rows = [[str(r["streamId"]), str(r["op_count"]), f"{r['total_ms']:.2f}",
                  f"{r['avg_ms']:.3f}", f"{r['max_ms']:.3f}", f"{r['min_ms']:.3f}"]
                 for r in rows]
    ev_lines = _fmt_table(["Stream", "次数", "总(ms)", "均(ms)", "最大(ms)", "最小(ms)"], tbl_rows)

    # 检测 stream 间不均衡
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
        confidence=0.90,
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


# ═══════════════════════════════════════════════════════════════════════════════
# sql_sync_cost（移植自 nsys-ai sync_cost_analysis skill）
# ═══════════════════════════════════════════════════════════════════════════════

def _sql_sync_cost(
    findings: list[NsysFinding],
    conn: sqlite3.Connection,
    sync_tbl: str,
    total_ns: int,
) -> None:
    """分析 CUPTI 同步事件的 wall-clock 代价。"""
    # 检查是否有 syncType 列及其枚举表
    sync_enum_tbl = _find_table(
        {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()},
        "ENUM_CUPTI_SYNC_TYPE"
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

    total_sync_ms = sum(float(r["total_ms"] or 0) for r in rows)
    total_calls = sum(int(r["call_count"] or 0) for r in rows)
    total_trace_ms = max(total_ns / 1e6, 0.001)
    sync_pct = total_sync_ms / total_trace_ms * 100

    tbl_rows = [[r["sync_type_name"] or "Unknown", str(r["call_count"]),
                  f"{r['total_ms']:.2f}", f"{r['avg_ms']:.3f}", f"{r['max_ms']:.3f}"]
                 for r in rows]
    ev_lines = _fmt_table(["同步类型", "次数", "总(ms)", "均(ms)", "最大(ms)"], tbl_rows)

    severity = "warning" if sync_pct > 10 else "info"
    findings.append(NsysFinding(
        category="sql_sync_cost",
        severity=severity,
        confidence=0.90,
        title=f"同步代价（CUPTI）：{total_sync_ms:.1f}ms，占 trace {sync_pct:.1f}%",
        description=(
            f"CUPTI 同步事件共 {total_calls} 次，wall-clock 代价 {total_sync_ms:.1f}ms，"
            f"占 trace 时长的 {sync_pct:.1f}%。"
            "这代表 CPU 真正被 CUDA 同步阻塞的时间（含不同 stream 间自然重叠的扣除）。"
        ),
        evidence=ev_lines[:7] + [f"同步密度：{sync_pct:.1f}% 的 trace 时间"],
        next_step=(
            "减少 cudaDeviceSynchronize 调用（使用流级别同步代替）。"
            "对于必要的同步，尽量将其安排在 GPU 空闲时执行以降低影响。"
        ),
    ))


# ═══════════════════════════════════════════════════════════════════════════════
# sql_nvtx_hotspots（移植自 nsys-ai nvtx-based analysis）
# ═══════════════════════════════════════════════════════════════════════════════

def _sql_nvtx_hotspots(
    findings: list[NsysFinding],
    conn: sqlite3.Connection,
    nvtx_tbl: str,
    has_strings: bool,
    total_ns: int,
    limit: int = 15,
) -> None:
    """NVTX 注释段热点分析：按总耗时排列最重要的标注区段。"""
    cols = _get_cols(conn, nvtx_tbl)
    has_textid = "textId" in cols

    if has_textid and has_strings:
        name_expr = "COALESCE(n.text, s.value) AS label"
        join_clause = "LEFT JOIN StringIds s ON n.textId = s.id"
    else:
        name_expr = "n.text AS label"
        join_clause = ""

    sql = f"""
        SELECT {name_expr},
               COUNT(*) AS call_count,
               ROUND(SUM(n.[end] - n.start) / 1e6, 2) AS total_ms,
               ROUND(AVG(n.[end] - n.start) / 1e6, 3) AS avg_ms,
               ROUND(MAX(n.[end] - n.start) / 1e6, 3) AS max_ms
        FROM {nvtx_tbl} n {join_clause}
        WHERE label IS NOT NULL AND label != '' AND n.[end] > n.start
        GROUP BY label ORDER BY total_ms DESC LIMIT {limit}
    """
    try:
        rows = conn.execute(sql).fetchall()
    except sqlite3.Error as e:
        logger.debug("sql_nvtx_hotspots 失败：%s", e)
        return
    if not rows:
        return

    total_trace_ms = max(total_ns / 1e6, 0.001)
    tbl_rows = []
    for r in rows:
        label = (r["label"] or "")[:46]
        pct = float(r["total_ms"] or 0) / total_trace_ms * 100
        tbl_rows.append([label, str(r["call_count"]), f"{r['total_ms']:.2f}",
                          f"{r['avg_ms']:.3f}", f"{r['max_ms']:.3f}", f"({pct:.1f}%)"])
    ev_lines = _fmt_table(["NVTX 标注", "次数", "总(ms)", "均(ms)", "最大(ms)", "占比"], tbl_rows)

    top_label = rows[0]["label"] or "unknown"
    top_ms = float(rows[0]["total_ms"] or 0)
    top_pct = top_ms / total_trace_ms * 100

    findings.append(NsysFinding(
        category="sql_nvtx_hotspots",
        severity="info",
        confidence=0.85,
        title=f"NVTX 热点：{top_label[:50]}（{top_ms:.1f}ms，{top_pct:.1f}%）",
        description=(
            f"Top NVTX 注释段 '{top_label[:40]}' 总计 {top_ms:.1f}ms，"
            f"占 trace {top_pct:.1f}%。NVTX 注释可帮助定位代码级热点。"
        ),
        evidence=ev_lines[:8],
        related_hotspots=[r["label"] or "" for r in rows[:5]],
        next_step=(
            "关注耗时最多的 NVTX 区段，这通常对应模型中的关键计算步骤。"
            "在热点区段内进一步细化 NVTX 注释以定位具体瓶颈。"
        ),
    ))
