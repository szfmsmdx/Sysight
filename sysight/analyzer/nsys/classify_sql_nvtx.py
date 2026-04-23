"""NVTX-focused deep SQL analysis helpers for nsys.

This module isolates NVTX hotspot analysis, GIL anchor extraction, NVTX-to-
kernel attribution, and NVTX layer breakdown from the broader classify_sql
module so SQL analysis can be split by responsibility.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict

from .models import NsysFinding, NvtxKernelAttribution
from .text import format_table

logger = logging.getLogger(__name__)

_NCCL_KEYWORDS = ("nccl", "allreduce", "allgather", "reducescatter", "broadcast", "sendrecv", "reduce")


def _get_cols(conn: sqlite3.Connection, tbl: str) -> list[str]:
    try:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({tbl})").fetchall()]
    except sqlite3.Error:
        return []


def _is_nccl(name: str) -> bool:
    low = name.lower()
    return any(kw in low for kw in _NCCL_KEYWORDS)


def _fmt_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return format_table(headers, rows)


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
        label_expr = "COALESCE(n.text, s.value)"
        join_clause = "LEFT JOIN StringIds s ON n.textId = s.id"
    else:
        label_expr = "n.text"
        join_clause = ""
    name_expr = f"{label_expr} AS label"

    sql = f"""
        SELECT {name_expr},
               COUNT(*) AS call_count,
               ROUND(SUM(n.[end] - n.start) / 1e6, 2) AS total_ms,
               ROUND(AVG(n.[end] - n.start) / 1e6, 3) AS avg_ms,
               ROUND(MAX(n.[end] - n.start) / 1e6, 3) AS max_ms
        FROM {nvtx_tbl} n {join_clause}
        WHERE {label_expr} IS NOT NULL AND {label_expr} != '' AND n.[end] > n.start
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

    _sql_nvtx_gil_anchor(findings, conn, nvtx_tbl, label_expr, join_clause, total_ns)


def _sql_nvtx_gil_anchor(
    findings: list[NsysFinding],
    conn: sqlite3.Connection,
    nvtx_tbl: str,
    label_expr: str,
    join_clause: str,
    total_ns: int,
) -> None:
    """Create a Host-side profile anchor from NVTX GIL ranges."""
    try:
        rows = conn.execute(
            f"""
            SELECT n.start, n.[end], {label_expr} AS label
            FROM {nvtx_tbl} n {join_clause}
            WHERE n.[end] > n.start
              AND {label_expr} IN ('Holding GIL', 'Waiting for GIL')
            """
        ).fetchall()
    except sqlite3.Error as e:
        logger.debug("sql_nvtx_gil_anchor 失败：%s", e)
        return

    if not rows:
        return

    by_label: dict[str, list[tuple[int, int]]] = defaultdict(list)
    all_intervals: list[tuple[int, int]] = []
    for row in rows:
        label = str(row["label"] or "")
        start_ns, end_ns = int(row["start"]), int(row["end"])
        if not label or end_ns <= start_ns:
            continue
        iv = (start_ns, end_ns)
        by_label[label].append(iv)
        all_intervals.append(iv)

    if not all_intervals:
        return

    observed_span = max(e for _, e in all_intervals) - min(s for s, _ in all_intervals)
    denominator_ns = max(int(total_ns or 0), observed_span, 1)
    wall_ns = min(_union_ns(all_intervals), denominator_ns)
    wall_pct = wall_ns / denominator_ns * 100
    inclusive_ns = sum(e - s for s, e in all_intervals)
    inclusive_pct = inclusive_ns / denominator_ns * 100

    evidence = [
        f"GIL NVTX union {wall_ns / 1e6:.1f}ms（占 NVTX 观测窗口 {wall_pct:.1f}%），"
        f"inclusive {inclusive_ns / 1e6:.1f}ms（{inclusive_pct:.1f}%）",
        "inclusive 可超过 100%，表示多个线程上的 GIL NVTX range 并行重叠。",
    ]
    for label in ("Waiting for GIL", "Holding GIL"):
        intervals = by_label.get(label, [])
        if not intervals:
            continue
        label_wall_ns = min(_union_ns(intervals), denominator_ns)
        label_incl_ns = sum(e - s for s, e in intervals)
        evidence.append(
            f"{label}: {len(intervals)} 次，"
            f"{label_wall_ns / 1e6:.1f}ms union，{label_incl_ns / 1e6:.1f}ms inclusive"
        )

    severity = "warning" if wall_pct >= 10.0 or by_label.get("Waiting for GIL") else "info"
    findings.append(NsysFinding(
        category="host_gil_contention",
        severity=severity,
        title=f"Host GIL NVTX 活动覆盖观测窗口 {wall_pct:.1f}%",
        description=(
            "NVTX 标注显示 Python GIL 相关区间覆盖了显著 wall 时间。"
            "这不是 CPU 采样，但可作为 Host 侧 profile anchor，提示检查 Python 调度、"
            "DataLoader、多线程回调或阻塞式同步。"
        ),
        evidence=evidence,
        related_hotspots=[label for label in ("Waiting for GIL", "Holding GIL") if label in by_label],
        next_step=(
            "若 CPU 采样缺失，建议重新采集并启用 CPU backtrace。"
            "同时围绕这些 GIL 区间检查 DataLoader、Python 回调、日志/metric 聚合和同步调用。"
        ),
    ))


def attribute_kernels_to_nvtx(
    conn: sqlite3.Connection,
    kernel_tbl: str,
    runtime_tbl: str,
    nvtx_tbl: str,
    has_strings: bool,
    trim: tuple[int, int] | None = None,
    limit: int | None = None,
) -> list[NvtxKernelAttribution]:
    """将每个 GPU kernel 归因到包裹它的 NVTX range（innermost）。"""
    trim_sql = ""
    trim_params: list = []
    if trim:
        trim_sql = "AND k.start >= ? AND k.[end] <= ?"
        trim_params = [trim[0], trim[1]]

    if has_strings:
        name_expr = (
            "COALESCE(d.value, s.value, 'kernel_' || CAST(k.shortName AS TEXT))"
        )
        kernel_join = (
            "LEFT JOIN StringIds s ON k.shortName = s.id "
            "LEFT JOIN StringIds d ON k.demangledName = d.id"
        )
    else:
        cols_k = _get_cols(conn, kernel_tbl)
        if "demangledName" in cols_k:
            name_expr = "COALESCE(CAST(k.demangledName AS TEXT), CAST(k.shortName AS TEXT))"
        else:
            name_expr = "CAST(k.shortName AS TEXT)"
        kernel_join = ""

    try:
        kr_rows = conn.execute(
            f"""
            SELECT r.globalTid, r.start AS r_start, r.[end] AS r_end,
                   k.start AS k_start, k.[end] AS k_end,
                   {name_expr} AS kernel_name
            FROM {kernel_tbl} k
            JOIN {runtime_tbl} r ON r.correlationId = k.correlationId
            {kernel_join}
            WHERE 1=1 {trim_sql}
            """,
            trim_params,
        ).fetchall()
    except sqlite3.Error as e:
        logger.debug("attribute_kernels_to_nvtx (kr join) 失败：%s", e)
        return []

    if not kr_rows:
        return []

    cols_nvtx = _get_cols(conn, nvtx_tbl)
    has_textid = "textId" in cols_nvtx

    if has_strings and has_textid:
        text_expr = "COALESCE(n.text, s.value)"
        text_join = "LEFT JOIN StringIds s ON n.textId = s.id"
    else:
        text_expr = "n.text"
        text_join = ""

    tids = sorted({int(r[0]) for r in kr_rows if r[0] is not None})
    if not tids:
        return []
    min_r_start = min(int(r[1]) for r in kr_rows)
    max_r_end = max(int(r[2]) for r in kr_rows)

    if len(tids) <= 900:
        tid_clause = f"AND n.globalTid IN ({','.join('?' for _ in tids)})"
        nvtx_params: list = list(tids) + [max_r_end, min_r_start]
    else:
        tid_clause = ""
        nvtx_params = [max_r_end, min_r_start]

    try:
        nvtx_rows = conn.execute(
            f"""
            SELECT n.globalTid, n.start, n.[end], {text_expr} AS text
            FROM {nvtx_tbl} n
            {text_join}
            WHERE n.eventType = 59
              AND n.[end] > n.start
              {tid_clause}
              AND n.start <= ?
              AND n.[end] >= ?
            ORDER BY n.globalTid, n.start
            """,
            nvtx_params,
        ).fetchall()
    except sqlite3.Error as e:
        logger.debug("attribute_kernels_to_nvtx (nvtx load) 失败：%s", e)
        return []

    nvtx_by_tid: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for row in nvtx_rows:
        tid, ns, ne, text = int(row[0]), int(row[1]), int(row[2]), row[3]
        if text:
            nvtx_by_tid[tid].append((ns, ne, str(text)))

    kr_by_tid: dict[int, list[tuple[int, int, int, int, str]]] = defaultdict(list)
    for row in kr_rows:
        gtid = int(row[0]) if row[0] is not None else 0
        r_start, r_end = int(row[1]), int(row[2])
        k_start, k_end = int(row[3]), int(row[4])
        k_name = row[5] or "unknown"
        kr_by_tid[gtid].append((r_start, r_end, k_start, k_end, k_name))

    results: list[NvtxKernelAttribution] = []

    for tid, kr_list in kr_by_tid.items():
        if tid not in nvtx_by_tid:
            continue

        nvtx_list = nvtx_by_tid[tid]
        kr_list.sort(key=lambda x: x[0])

        nvtx_idx = 0
        open_stack: list[tuple[int, int, str]] = []

        for r_start, r_end, k_start, k_end, k_name in kr_list:
            while open_stack and open_stack[-1][1] < r_start:
                open_stack.pop()

            while nvtx_idx < len(nvtx_list) and nvtx_list[nvtx_idx][0] <= r_start:
                ns, ne, nt = nvtx_list[nvtx_idx]
                if ne >= r_start:
                    open_stack.append((ns, ne, nt))
                nvtx_idx += 1

            best_idx = -1
            for i in range(len(open_stack) - 1, -1, -1):
                ns, ne, _ = open_stack[i]
                if ns <= r_start and ne >= r_end:
                    best_idx = i
                    break

            if best_idx < 0:
                continue

            enclosing = [
                e for e in open_stack[: best_idx + 1]
                if e[0] <= r_start and e[1] >= r_end
            ]
            if not enclosing:
                continue

            nvtx_text = enclosing[-1][2]
            nvtx_path = " > ".join(e[2] for e in enclosing)
            nvtx_depth = len(enclosing) - 1

            results.append(NvtxKernelAttribution(
                kernel_name=k_name,
                k_start_ns=k_start,
                k_end_ns=k_end,
                k_dur_ns=k_end - k_start,
                nvtx_text=nvtx_text,
                nvtx_path=nvtx_path,
                nvtx_depth=nvtx_depth,
            ))

    results.sort(key=lambda r: r.k_start_ns)
    if limit is not None:
        results = results[:limit]
    return results


def _sql_nvtx_layer_breakdown(
    findings: list[NsysFinding],
    conn: sqlite3.Connection,
    kernel_tbl: str,
    runtime_tbl: str,
    nvtx_tbl: str,
    has_strings: bool,
    total_ns: int,
    limit: int = 20,
) -> None:
    """每个 NVTX region 的 GPU 时间分解（compute vs NCCL + outlier 检测）。"""
    attributions = attribute_kernels_to_nvtx(
        conn, kernel_tbl, runtime_tbl, nvtx_tbl, has_strings,
    )
    if not attributions:
        return

    groups: dict[str, dict] = {}
    for a in attributions:
        key = a.nvtx_path
        if key not in groups:
            groups[key] = {
                "total_ns": 0, "nccl_ns": 0, "compute_ns": 0,
                "count": 0, "max_ns": 0,
                "nvtx_depth": a.nvtx_depth, "nvtx_text": a.nvtx_text,
                "k_times": defaultdict(int),
            }
        g = groups[key]
        g["total_ns"] += a.k_dur_ns
        g["count"] += 1
        if a.k_dur_ns > g["max_ns"]:
            g["max_ns"] = a.k_dur_ns
        if _is_nccl(a.kernel_name):
            g["nccl_ns"] += a.k_dur_ns
        else:
            g["compute_ns"] += a.k_dur_ns
        g["k_times"][a.kernel_name] += a.k_dur_ns

    if not groups:
        return

    sorted_groups = sorted(groups.items(), key=lambda x: -x[1]["total_ns"])

    all_times_ms = sorted(g["total_ns"] / 1e6 for _, g in sorted_groups if g["total_ns"] > 0)
    count = len(all_times_ms)
    if count >= 4:
        def _q(p: float) -> float:
            pos = p * (count - 1)
            lo = int(pos)
            frac = pos - lo
            if frac == 0:
                return all_times_ms[lo]
            return all_times_ms[lo] + frac * (all_times_ms[min(lo + 1, count - 1)] - all_times_ms[lo])
        med = _q(0.5)
        q1, q3 = _q(0.25), _q(0.75)
        iqr = q3 - q1
        fence = (q3 + 1.5 * iqr) if iqr > 0 else (med * 2.0)
    elif count >= 2:
        mid = count // 2
        med = all_times_ms[mid] if count % 2 else (all_times_ms[mid - 1] + all_times_ms[mid]) / 2
        fence = med * 2.0
    else:
        med, fence = 0.0, float("inf")

    total_trace_ms = max(total_ns / 1e6, 0.001)
    tbl_rows = []
    outlier_info: list[str] = []

    for nvtx_path, group in sorted_groups[:limit]:
        total_ms = group["total_ns"] / 1e6
        nccl_ms = group["nccl_ns"] / 1e6
        compute_ms = group["compute_ns"] / 1e6
        nccl_pct = 100.0 * group["nccl_ns"] / group["total_ns"] if group["total_ns"] > 0 else 0.0
        pct_of_trace = total_ms / total_trace_ms * 100
        is_outlier = count >= 2 and total_ms > fence and total_ms > med * 1.5

        display_path = nvtx_path if len(nvtx_path) <= 40 else "..." + nvtx_path[-37:]
        outlier_flag = "⚠️" if is_outlier else ""

        top_k = sorted(group["k_times"].items(), key=lambda x: -x[1])[:3]
        top_k_str = ", ".join(f"{kn[:25]}({kd/1e6:.1f}ms)" for kn, kd in top_k)

        tbl_rows.append([
            display_path,
            str(group["nvtx_depth"]),
            str(group["count"]),
            f"{total_ms:.2f}",
            f"{compute_ms:.2f}",
            f"{nccl_ms:.2f}",
            f"{nccl_pct:.1f}%",
            f"{pct_of_trace:.1f}%",
            outlier_flag,
        ])

        if is_outlier:
            ratio = total_ms / med if med > 0 else 0.0
            outlier_info.append(
                f"⚠️ 离群层 '{display_path}' {total_ms:.1f}ms "
                f"（{ratio:.1f}× 中位数 {med:.1f}ms），热点：{top_k_str}"
            )

    ev_lines = _fmt_table(
        ["NVTX Region", "深度", "内核数", "总GPU(ms)", "计算(ms)", "NCCL(ms)", "NCCL%", "占trace%", "异常"],
        tbl_rows,
    )

    all_evidence = ev_lines[:12]
    if outlier_info:
        all_evidence += outlier_info[:3]

    top_path, top_group = sorted_groups[0]
    top_ms = top_group["total_ns"] / 1e6
    top_pct = top_ms / total_trace_ms * 100
    n_outliers = len(outlier_info)

    severity = "warning" if n_outliers > 0 else "info"
    findings.append(NsysFinding(
        category="sql_nvtx_layer_breakdown",
        severity=severity,
        title=(
            f"NVTX 层 GPU 时间分解：{len(sorted_groups)} 个 region，"
            f"热点 '{top_path[:40]}' {top_ms:.1f}ms（{top_pct:.1f}%）"
            + (f"，{n_outliers} 个离群层" if n_outliers else "")
        ),
        description=(
            f"通过 correlationId 链将 {len(attributions)} 个 GPU kernel 归因到 "
            f"{len(sorted_groups)} 个 NVTX region，精确到具体代码路径。"
            f"热点区域 '{top_path[:50]}' 占 trace {top_pct:.1f}%。"
            + (f" 检测到 {n_outliers} 个统计离群层。" if n_outliers else "")
        ),
        evidence=all_evidence,
        related_hotspots=[p for p, _ in sorted_groups[:5]],
        next_step=(
            "关注耗时最多且 NCCL% 高的层——这是通信优化的直接目标。"
            "离群层（⚠️）显著慢于中位数，优先排查其 top kernel。"
            "使用 Nsight Compute 对热点 kernel 进行深层分析。"
        ),
    ))


def _union_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    merged = sorted(intervals)
    total = 0
    cur_start, cur_end = merged[0]
    for start, end in merged[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
            continue
        total += max(0, cur_end - cur_start)
        cur_start, cur_end = start, end
    total += max(0, cur_end - cur_start)
    return total


__all__ = [
    "_sql_nvtx_hotspots",
    "attribute_kernels_to_nvtx",
    "_sql_nvtx_layer_breakdown",
]
