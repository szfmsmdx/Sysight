"""nsys/classify_sql.py — 深度 SQLite 直接分析（移植自 nsys-ai skills）。

本模块由 classify.py 的 classify_bottlenecks() 在有 sqlite_path 时调用。
包含以下分析，对标 nsys-ai 的同名 skill：

  sql_top_kernels       Top-N GPU 内核（按总时间）
  sql_gpu_idle_gaps     GPU 流内的空闲气泡（含 CPU API 归因 + NVTX 重叠）
  sql_memory_bandwidth  各方向内存带宽统计（H2D/D2H/D2D/P2P）
  sql_nccl_breakdown    NCCL 集合通信按流分解
  sql_sync_cost         CUPTI 同步事件 wall-clock 代价
  sql_nvtx_hotspots     NVTX 注释段热点分布
  sql_nvtx_layer_breakdown  每个 NVTX region 的 GPU 时间分解（移植自 nsys-ai nvtx_layer_breakdown）
  sql_root_cause_analysis   已知性能反模式的程序化检测（移植自 nsys-ai root_cause_matcher）
  sql_profile_health        一次调用获取 profile 健康摘要（移植自 nsys-ai profile_health_manifest）

NVTX→Kernel 归因算法（attribute_kernels_to_nvtx）：
  通过 correlationId 链精确绑定每个 GPU kernel 到发起它的 Runtime API，
  再通过 O(N+M) sort-merge stack sweep 在线程维度找到包裹该 API 的最内层 NVTX range。
  这是比字符串匹配更精确的代码定位方式。
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing

from .classify_sql_nvtx import _sql_nvtx_hotspots, _sql_nvtx_layer_breakdown
from .extract import find_gaps, intersect_total, union_ns
from .models import NsysFinding, NsysTrace
from .text import format_table

logger = logging.getLogger(__name__)

_COPY_KIND_NAMES = {0: "Unknown", 1: "H2D", 2: "D2H", 4: "H2H", 8: "D2D", 10: "P2P"}

_NCCL_KEYWORDS = ("nccl", "allreduce", "allgather", "reducescatter", "broadcast", "sendrecv", "reduce")


# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════

def run_deep_sql_analysis(findings: list[NsysFinding], trace: NsysTrace) -> None:
    """深度 SQL 分析入口，由 classify_bottlenecks() 调用。

    分析顺序（各项均非致命，失败时记录 debug 日志后继续）：
      1. sql_top_kernels          — Top-N GPU kernel 统计
      2. sql_gpu_idle_gaps        — GPU stream 内空闲气泡
      3. sql_nccl_breakdown       — NCCL 集合通信分解
      4. sql_memory_bandwidth     — 各方向内存带宽
      5. sql_sync_cost            — 同步代价
      6. sql_nvtx_hotspots        — NVTX 注释段热点
      7. sql_nvtx_layer_breakdown — NVTX region 级 GPU 时间分解（需 kernel+runtime+nvtx）
      8. sql_root_cause_analysis  — 已知反模式程序化检测
      9. sql_profile_health       — Profile 全局健康摘要
    """
    sqlite_path = trace.sqlite_path
    if not sqlite_path:
        return
    try:
        with closing(sqlite3.connect(sqlite_path)) as conn:
            conn.row_factory = sqlite3.Row
            all_tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            has_strings = "StringIds" in all_tables

            kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
            runtime_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_RUNTIME")
            memcpy_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_MEMCPY")
            memset_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_MEMSET")
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

            # ── 新增：NVTX→Kernel 精确归因 & 层级时间分解 ─────────────────────
            if kernel_tbl and runtime_tbl and nvtx_tbl:
                try:
                    _sql_nvtx_layer_breakdown(
                        findings, conn, kernel_tbl, runtime_tbl, nvtx_tbl,
                        has_strings, trace.duration_ns,
                    )
                except Exception as e:
                    logger.debug("sql_nvtx_layer_breakdown 失败（非致命）：%s", e)

            # ── 新增：已知反模式程序化检测（root_cause_matcher 移植）────────
            if kernel_tbl:
                try:
                    _sql_root_cause_analysis(
                        findings, conn, kernel_tbl, runtime_tbl,
                        memcpy_tbl, memset_tbl, sync_tbl, nvtx_tbl,
                        has_strings, trace.duration_ns,
                    )
                except Exception as e:
                    logger.debug("sql_root_cause_analysis 失败（非致命）：%s", e)

            # ── 新增：profile 健康摘要（profile_health_manifest 移植）────────
            if kernel_tbl:
                try:
                    _sql_profile_health(
                        findings, conn, kernel_tbl, runtime_tbl,
                        memcpy_tbl, nvtx_tbl, has_strings, trace.duration_ns,
                    )
                except Exception as e:
                    logger.debug("sql_profile_health 失败（非致命）：%s", e)

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


def _kernel_name_expr(
    conn: sqlite3.Connection,
    kernel_tbl: str,
    has_strings: bool,
    alias: str = "k",
) -> tuple[str, str]:
    """Return (name_expr, join_clause) for kernel names across schema variants."""
    cols = _get_cols(conn, kernel_tbl)
    short_col = "shortName" if "shortName" in cols else ""
    demangled_col = "demangledName" if "demangledName" in cols else ""

    if has_strings and short_col:
        short_ref = f"{alias}.{short_col}"
        if demangled_col:
            demangled_ref = f"{alias}.{demangled_col}"
            return (
                f"COALESCE(d.value, s.value, CAST({short_ref} AS TEXT))",
                f"LEFT JOIN StringIds s ON {short_ref} = s.id "
                f"LEFT JOIN StringIds d ON {demangled_ref} = d.id",
            )
        return (
            f"COALESCE(s.value, CAST({short_ref} AS TEXT))",
            f"LEFT JOIN StringIds s ON {short_ref} = s.id",
        )

    if demangled_col:
        return f"CAST({alias}.{demangled_col} AS TEXT)", ""
    if short_col:
        return f"CAST({alias}.{short_col} AS TEXT)", ""
    return "'unknown_kernel'", ""


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
        f"共 {gap_count} 个 stream 内间隙，per-stream 总空闲 {total_idle_ms:.1f}ms",
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
    sync_inclusive_pct = total_sync_ms / total_trace_ms * 100

    sync_wall_ms = 0.0
    sync_wall_pct = 0.0
    try:
        interval_rows = conn.execute(
            f"SELECT start, [end] FROM {sync_tbl} WHERE [end] > start"
        ).fetchall()
        intervals = [(int(r["start"]), int(r["end"])) for r in interval_rows]
        sync_wall_ns = min(union_ns(intervals), int(total_ns or 0))
        sync_wall_ms = sync_wall_ns / 1e6
        sync_wall_pct = sync_wall_ms / total_trace_ms * 100
    except sqlite3.Error:
        sync_wall_ms = total_sync_ms
        sync_wall_pct = sync_inclusive_pct

    tbl_rows = [[r["sync_type_name"] or "Unknown", str(r["call_count"]),
                  f"{r['total_ms']:.2f}", f"{r['avg_ms']:.3f}", f"{r['max_ms']:.3f}"]
                 for r in rows]
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




# ═══════════════════════════════════════════════════════════════════════════════
# sql_root_cause_analysis（移植自 nsys-ai root_cause_matcher）
# ═══════════════════════════════════════════════════════════════════════════════

def _sql_root_cause_analysis(
    findings: list[NsysFinding],
    conn: sqlite3.Connection,
    kernel_tbl: str,
    runtime_tbl: str | None,
    memcpy_tbl: str | None,
    memset_tbl: str | None,
    sync_tbl: str | None,
    nvtx_tbl: str | None,
    has_strings: bool,
    total_ns: int,
) -> None:
    """程序化检测已知 GPU 性能反模式（移植自 nsys-ai root_cause_matcher）。

    检测项目：
      1. 过度同步：cudaDeviceSynchronize/StreamSynchronize 占比
      2. 同步 Memcpy：阻塞 host 的 cudaMemcpy（非 Async）
      3. Pageable memory：async memcpy 实为同步（srcKind/dstKind=1）
      4. 同步 Memset：阻塞 host 的 cudaMemset（非 Async）
      5. NCCL 同 stream 序列化：NCCL 与计算在同一 stream
      6. Sync-After-NCCL：NCCL kernel 后 1ms 内有同步调用
    """
    _rc_patterns: list[dict] = []

    # ── 1. 过度同步 API ───────────────────────────────────────────────────────
    if runtime_tbl and has_strings:
        try:
            sync_name_rows = conn.execute(
                """
                SELECT id, value FROM StringIds
                WHERE value LIKE 'cudaDeviceSynchronize%'
                   OR value LIKE 'cudaStreamSynchronize%'
                   OR value LIKE 'cudaEventSynchronize%'
                   OR value LIKE 'cudaStreamWaitEvent%'
                """
            ).fetchall()
            if sync_name_rows:
                name_ids = [r[0] for r in sync_name_rows]
                ph = ",".join(str(nid) for nid in name_ids)
                sync_stat = conn.execute(
                    f"""
                    SELECT COUNT(*) AS call_count, COALESCE(SUM([end] - start), 0) AS total_ns
                    FROM {runtime_tbl}
                    WHERE nameId IN ({ph})
                    """
                ).fetchone()
                if sync_stat and sync_stat[0]:
                    total_gpu_ns_row = conn.execute(
                        f"SELECT COALESCE(SUM([end] - start), 0) FROM {kernel_tbl}"
                    ).fetchone()
                    total_gpu_ns = int(total_gpu_ns_row[0] or 0) if total_gpu_ns_row else 0
                    sync_ns = int(sync_stat[1] or 0)
                    sync_ms = sync_ns / 1e6
                    sync_pct = (sync_ns / total_gpu_ns * 100) if total_gpu_ns > 0 else 100.0
                    if sync_ms >= 1.0 and sync_pct >= 2.0:
                        api_names = ", ".join(sorted({r[1].split("_v")[0] for r in sync_name_rows}))
                        _rc_patterns.append({
                            "pattern": "过度同步（Excessive Synchronization）",
                            "severity": "warning",
                            "evidence": (
                                f"{sync_stat[0]} 次同步调用共 {sync_ms:.1f}ms"
                                f"（GPU 时间的 {sync_pct:.1f}%）。API：{api_names}"
                            ),
                            "recommendation": (
                                "从训练循环中移除 .item()/.cpu() 隐式同步。"
                                "用 torch.cuda.set_sync_debug_mode(1) 定位隐式同步。"
                                "用 CUDA events 替换 cudaDeviceSynchronize。"
                            ),
                        })
        except sqlite3.Error as e:
            logger.debug("_sql_root_cause (sync_api): %s", e)

    # ── 2. 同步 Memcpy ────────────────────────────────────────────────────────
    if runtime_tbl and memcpy_tbl and has_strings:
        try:
            sync_memcpy_names = conn.execute(
                """
                SELECT id FROM StringIds
                WHERE value LIKE 'cudaMemcpy%'
                  AND value NOT LIKE 'cudaMemcpyAsync%'
                """
            ).fetchall()
            if sync_memcpy_names:
                ph = ",".join(str(r[0]) for r in sync_memcpy_names)
                row = conn.execute(
                    f"""
                    SELECT COUNT(*) AS cnt,
                           COALESCE(SUM(m.bytes), 0) AS total_bytes,
                           COALESCE(SUM(m.[end] - m.start), 0) AS total_ns
                    FROM {runtime_tbl} r
                    JOIN {memcpy_tbl} m ON r.correlationId = m.correlationId
                    WHERE r.nameId IN ({ph})
                    """
                ).fetchone()
                if row and row[0]:
                    cnt = row[0]
                    total_bytes = int(row[1] or 0)
                    mem_ns = int(row[2] or 0)
                    _rc_patterns.append({
                        "pattern": "同步 Memcpy（Synchronous Memcpy）",
                        "severity": "warning",
                        "evidence": (
                            f"{cnt} 次同步 cudaMemcpy：{total_bytes/1e6:.1f}MB，"
                            f"{mem_ns/1e6:.1f}ms。这些调用阻塞 host 线程。"
                        ),
                        "recommendation": (
                            "改用 cudaMemcpyAsync + pinned memory。"
                            "DataLoader 使用 pin_memory=True，张量 .to(device) 加 non_blocking=True。"
                        ),
                    })
        except sqlite3.Error as e:
            logger.debug("_sql_root_cause (sync_memcpy): %s", e)

    # ── 3. Pageable memory（async memcpy 实为同步）────────────────────────────
    if memcpy_tbl:
        try:
            cols = _get_cols(conn, memcpy_tbl)
            if "srcKind" in cols and "dstKind" in cols:
                row = conn.execute(
                    f"""
                    SELECT COUNT(*) AS cnt,
                           COALESCE(SUM(bytes), 0) AS total_bytes,
                           COALESCE(SUM([end] - start), 0) AS total_ns
                    FROM {memcpy_tbl}
                    WHERE srcKind = 1 OR dstKind = 1
                    """
                ).fetchone()
                if row and row[0]:
                    cnt = row[0]
                    total_bytes = int(row[1] or 0)
                    mem_ns = int(row[2] or 0)
                    _rc_patterns.append({
                        "pattern": "Pageable Memory 异步拷贝（实为同步）",
                        "severity": "warning",
                        "evidence": (
                            f"{cnt} 次 memcpy 使用 pageable（非 pinned）内存："
                            f"{total_bytes/1e6:.1f}MB，{mem_ns/1e6:.1f}ms。"
                            "Pageable memory 导致 async memcpy 静默降级为同步。"
                        ),
                        "recommendation": (
                            "使用 cudaMallocHost() / DataLoader pin_memory=True。"
                            "Pinned memory 才能实现真正的异步 H2D 重叠。"
                        ),
                    })
        except sqlite3.Error as e:
            logger.debug("_sql_root_cause (pageable): %s", e)

    # ── 4. 同步 Memset ────────────────────────────────────────────────────────
    if runtime_tbl and memset_tbl and has_strings:
        try:
            sync_memset_names = conn.execute(
                """
                SELECT id FROM StringIds
                WHERE value LIKE 'cudaMemset%'
                  AND value NOT LIKE 'cudaMemsetAsync%'
                """
            ).fetchall()
            if sync_memset_names:
                ph = ",".join(str(r[0]) for r in sync_memset_names)
                row = conn.execute(
                    f"""
                    SELECT COUNT(*) AS cnt,
                           COALESCE(SUM(ms.[end] - ms.start), 0) AS total_ns
                    FROM {runtime_tbl} r
                    JOIN {memset_tbl} ms ON r.correlationId = ms.correlationId
                    WHERE r.nameId IN ({ph})
                    """
                ).fetchone()
                if row and row[0]:
                    cnt = row[0]
                    mem_ns = int(row[1] or 0)
                    _rc_patterns.append({
                        "pattern": "同步 Memset（Synchronous Memset）",
                        "severity": "info",
                        "evidence": (
                            f"{cnt} 次同步 cudaMemset，{mem_ns/1e6:.2f}ms 阻塞 host 线程。"
                        ),
                        "recommendation": "改用 cudaMemsetAsync 在对应 stream 上执行。",
                    })
        except sqlite3.Error as e:
            logger.debug("_sql_root_cause (sync_memset): %s", e)

    # ── 5. NCCL 同 stream 序列化 ──────────────────────────────────────────────
    if has_strings:
        try:
            like_clauses = " OR ".join(
                f"LOWER(s.value) LIKE '%{kw}%'" for kw in _NCCL_KEYWORDS
            )
            same_stream_rows = conn.execute(
                f"""
                SELECT k.streamId,
                    SUM(CASE WHEN {like_clauses} THEN 1 ELSE 0 END) AS nccl_count,
                    SUM(CASE WHEN NOT ({like_clauses}) THEN 1 ELSE 0 END) AS compute_count
                FROM {kernel_tbl} k
                JOIN StringIds s ON k.shortName = s.id
                GROUP BY k.streamId
                HAVING nccl_count > 0 AND compute_count > 0
                """
            ).fetchall()
            if same_stream_rows:
                streams_list = [str(r[0]) for r in same_stream_rows]
                _rc_patterns.append({
                    "pattern": "NCCL 序列化——同 Stream（Same-Stream Serialization）",
                    "severity": "critical",
                    "evidence": (
                        f"Stream [{', '.join(streams_list[:5])}] 上 NCCL 与计算 kernel 共存，"
                        "被 CUDA stream 顺序强制序列化。"
                    ),
                    "recommendation": (
                        "将 AllReduce 移到独立的 NCCL stream。"
                        "PyTorch DDP：检查 find_unused_parameters=True 是否强制同步。"
                        "使用 bucket_cap_mb 调整 gradient bucketing 粒度。"
                    ),
                })
        except sqlite3.Error as e:
            logger.debug("_sql_root_cause (nccl same_stream): %s", e)

    # ── 6. Sync-After-NCCL ────────────────────────────────────────────────────
    if runtime_tbl and has_strings:
        try:
            sync_names_for_nccl = conn.execute(
                """
                SELECT id FROM StringIds
                WHERE value LIKE 'cudaStreamSynchronize%'
                   OR value LIKE 'cudaDeviceSynchronize%'
                """
            ).fetchall()
            if sync_names_for_nccl:
                like_clauses_nccl = " OR ".join(
                    f"LOWER(sn.value) LIKE '%{kw}%'" for kw in _NCCL_KEYWORDS
                )
                ph = ",".join(str(r[0]) for r in sync_names_for_nccl)
                found = conn.execute(
                    f"""
                    SELECT 1 FROM {kernel_tbl} k
                    JOIN StringIds sn ON k.shortName = sn.id
                    WHERE ({like_clauses_nccl})
                      AND EXISTS (
                          SELECT 1 FROM {runtime_tbl} r
                          WHERE r.nameId IN ({ph})
                            AND r.start >= k.[end]
                            AND r.start <= k.[end] + 1000000
                      )
                    LIMIT 1
                    """
                ).fetchone()
                if found:
                    _rc_patterns.append({
                        "pattern": "NCCL 后立即同步（Sync-After-NCCL）",
                        "severity": "warning",
                        "evidence": (
                            "NCCL kernel 完成后 1ms 内检测到 "
                            "cudaStreamSynchronize / cudaDeviceSynchronize 调用。"
                        ),
                        "recommendation": (
                            "移除 NCCL 操作后的显式同步。"
                            "通信/计算传输使用 non_blocking=True。"
                        ),
                    })
        except sqlite3.Error as e:
            logger.debug("_sql_root_cause (sync_after_nccl): %s", e)

    if not _rc_patterns:
        return

    # 按 severity 排序并生成 Finding
    sev_rank = {"critical": 0, "warning": 1, "info": 2}
    _rc_patterns.sort(key=lambda p: sev_rank.get(p["severity"], 3))

    top_sev = _rc_patterns[0]["severity"]
    evidence_lines: list[str] = []
    for p in _rc_patterns:
        icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(p["severity"], "⚪")
        evidence_lines.append(f"{icon} {p['pattern']}：{p['evidence']}")
        evidence_lines.append(f"   建议：{p['recommendation']}")

    pattern_names = [p["pattern"] for p in _rc_patterns]
    findings.append(NsysFinding(
        category="sql_root_cause_analysis",
        severity=top_sev,
        title=f"根因分析：{len(_rc_patterns)} 个反模式（最严重：{top_sev}）",
        description=(
            f"程序化检测发现 {len(_rc_patterns)} 个已知 GPU 性能反模式："
            + "；".join(p["pattern"] for p in _rc_patterns[:3]) + "。"
        ),
        evidence=evidence_lines[:12],
        related_hotspots=pattern_names[:5],
        next_step=(
            "按 severity 顺序逐一修复。CRITICAL 级（同 stream 序列化）通常收益最大。"
            "每次修复后重新 profile 验证效果。"
        ),
    ))


# ═══════════════════════════════════════════════════════════════════════════════
# sql_profile_health（移植自 nsys-ai profile_health_manifest）
# ═══════════════════════════════════════════════════════════════════════════════

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
    """Profile 健康摘要——汇总关键指标并推断主要瓶颈。

    移植自 nsys-ai profile_health_manifest，作为 LLM/agent 快速概况入口。
    采集：top kernel、NCCL overlap 粗估、GPU 空闲率、同步密度。
    输出 NsysFinding（category="sql_profile_health"），内含 suspected_bottleneck。
    """
    denominator_ns = max(int(total_ns or 0), 1)

    # ── Kernel intervals, Top kernels & inclusive kernel time ───────────────
    top_kernel_name: str = ""
    top_kernel_ms: float = 0.0
    total_kernel_ms: float = 0.0
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
        total_kernel_ms = float((total_row[0] or 0)) if total_row else 0.0

        interval_rows = conn.execute(
            f"""
            SELECT k.start, k.[end], {name_expr} AS kname
            FROM {kernel_tbl} k {join_clause}
            WHERE k.[end] > k.start
            """
        ).fetchall()
        for row in interval_rows:
            start_ns, end_ns = int(row["start"]), int(row["end"])
            name = str(row["kname"] or "")
            iv = (start_ns, end_ns)
            all_gpu_intervals.append(iv)
            if _is_nccl(name):
                nccl_intervals.append(iv)
            else:
                compute_intervals.append(iv)
    except sqlite3.Error as e:
        logger.debug("_sql_profile_health (top_kernels): %s", e)

    if memcpy_tbl:
        try:
            for row in conn.execute(
                f"SELECT start, [end] FROM {memcpy_tbl} WHERE [end] > start"
            ).fetchall():
                all_gpu_intervals.append((int(row["start"]), int(row["end"])))
        except sqlite3.Error as e:
            logger.debug("_sql_profile_health (memcpy intervals): %s", e)

    # If memcpy/runtime extends beyond kernel span, use the wider observed span
    # so profile health never reports physically impossible >100% idle.
    trace_start_ns = min((s for s, _ in all_gpu_intervals), default=0)
    trace_end_ns = max((e for _, e in all_gpu_intervals), default=denominator_ns)
    observed_span_ns = max(0, trace_end_ns - trace_start_ns)
    denominator_ns = max(denominator_ns, observed_span_ns, 1)

    # ── NCCL overlap (union / wall-clock semantics) ─────────────────────────
    compute_union_ns = union_ns(compute_intervals)
    nccl_union_ns = union_ns(nccl_intervals)
    overlap_ns = intersect_total(compute_intervals, nccl_intervals)
    nccl_only_ms = max(0, nccl_union_ns - overlap_ns) / 1e6
    compute_only_ms = max(0, compute_union_ns - overlap_ns) / 1e6
    overlap_pct: float = 100.0
    if nccl_union_ns > 0:
        overlap_pct = round(overlap_ns / nccl_union_ns * 100, 1)

    # ── GPU 空闲率（global union, not per-stream gap sum）───────────────────
    idle_pct: float = 0.0
    gap_count: int = 0
    if all_gpu_intervals:
        gpu_active_ns = min(union_ns(all_gpu_intervals), denominator_ns)
        idle_ns = max(0, denominator_ns - gpu_active_ns)
        idle_pct = round(idle_ns / denominator_ns * 100, 1)
        gap_end_ns = trace_start_ns + denominator_ns
        gap_count = len(find_gaps(all_gpu_intervals, trace_start_ns, gap_end_ns, min_gap_ns=1_000_000))

    # ── 同步密度（runtime sync interval union + inclusive note）──────────────
    sync_density_pct: float = 0.0
    sync_inclusive_pct: float = 0.0
    if runtime_tbl and has_strings:
        try:
            runtime_cols = _get_cols(conn, runtime_tbl)
            if "nameId" in runtime_cols:
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
                    ph = ",".join(str(r[0]) for r in sync_names)
                    sync_rows = conn.execute(
                        f"""
                        SELECT start, [end] FROM {runtime_tbl}
                        WHERE nameId IN ({ph}) AND [end] > start
                        """
                    ).fetchall()
                    sync_intervals = [
                        (int(r["start"]), int(r["end"])) for r in sync_rows
                    ]
                    sync_union = min(union_ns(sync_intervals), denominator_ns)
                    sync_inclusive = sum(e - s for s, e in sync_intervals)
                    sync_density_pct = round(sync_union / denominator_ns * 100, 1)
                    sync_inclusive_pct = round(sync_inclusive / denominator_ns * 100, 1)
        except sqlite3.Error as e:
            logger.debug("_sql_profile_health (sync_density): %s", e)

    # ── 推断主要瓶颈（启发式规则） ─────────────────────────────────────────
    suspected_bottleneck = ""
    if sync_density_pct > 20.0:
        suspected_bottleneck = f"高 CPU 同步阻塞（{sync_density_pct:.1f}% wall）"
    elif nccl_union_ns > 0 and overlap_pct < 30.0:
        suspected_bottleneck = f"NCCL 序列化（union 重叠率仅 {overlap_pct:.1f}%）"
    elif idle_pct > 15.0:
        suspected_bottleneck = f"GPU 空闲气泡（{idle_pct:.1f}% wall，{gap_count} 个全局间隙）"
    elif total_kernel_ms > 0 and top_kernel_ms / total_kernel_ms > 0.6:
        top_pct = top_kernel_ms / total_kernel_ms * 100
        suspected_bottleneck = f"Kernel 热点：{top_kernel_name[:40]}（{top_pct:.0f}%）"
    elif nccl_only_ms > compute_only_ms and compute_only_ms > 0:
        suspected_bottleneck = "通信主导（NCCL 独占 > 计算独占）"
    else:
        suspected_bottleneck = "计算主导（无明显序列化瓶颈）"

    trace_ms = denominator_ns / 1e6
    evidence = [
        f"Trace wall：{trace_ms:.1f}ms，GPU kernel 总计(incl)：{total_kernel_ms:.1f}ms",
        f"Top kernel：{top_kernel_name[:55]}（{top_kernel_ms:.1f}ms，"
        f"{top_kernel_ms/total_kernel_ms*100:.1f}%）" if total_kernel_ms > 0 else
        f"Top kernel：{top_kernel_name[:55]}（{top_kernel_ms:.1f}ms）",
        f"NCCL 独占(union)：{nccl_only_ms:.1f}ms，计算独占(union)：{compute_only_ms:.1f}ms，"
        f"NCCL 重叠率(union)：{overlap_pct:.1f}%",
        f"GPU 空闲(global union)：{idle_pct:.1f}%（{gap_count} 个 >1ms 全局间隙）",
        f"同步等待(runtime union)：{sync_density_pct:.1f}% wall"
        + (f"；inclusive {sync_inclusive_pct:.1f}%" if sync_inclusive_pct else ""),
        f"→ 疑似主要瓶颈：{suspected_bottleneck}",
    ]

    severity = "warning" if (idle_pct > 10 or sync_density_pct > 15 or overlap_pct < 30) else "info"
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
