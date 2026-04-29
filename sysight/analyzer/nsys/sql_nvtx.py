"""NVTX-oriented deep Nsight Systems SQL analyzers."""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict

from .extract import union_ns
from .models import NsysFinding, NvtxKernelAttribution
from .sql_shared import _fmt_table, _get_cols, _is_nccl, _kernel_name_expr

logger = logging.getLogger(__name__)


def _sql_nvtx_hotspots(
    findings: list[NsysFinding],
    conn: sqlite3.Connection,
    nvtx_tbl: str,
    has_strings: bool,
    total_ns: int,
    limit: int = 15,
) -> None:
    """按总耗时列出最重要的 NVTX 标注区段。"""
    cols = _get_cols(conn, nvtx_tbl)
    has_textid = "textId" in cols

    if has_textid and has_strings:
        label_expr = "COALESCE(n.text, s.value)"
        join_clause = "LEFT JOIN StringIds s ON n.textId = s.id"
    else:
        label_expr = "n.text"
        join_clause = ""

    sql = f"""
        SELECT {label_expr} AS label,
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
    except sqlite3.Error as exc:
        logger.debug("sql_nvtx_hotspots 失败：%s", exc)
        return
    if not rows:
        return

    total_trace_ms = max(total_ns / 1e6, 0.001)
    tbl_rows = []
    for row in rows:
        label = (row["label"] or "")[:46]
        pct = float(row["total_ms"] or 0) / total_trace_ms * 100
        tbl_rows.append([
            label,
            str(row["call_count"]),
            f"{row['total_ms']:.2f}",
            f"{row['avg_ms']:.3f}",
            f"{row['max_ms']:.3f}",
            f"({pct:.1f}%)",
        ])
    evidence = _fmt_table(["NVTX 标注", "次数", "总(ms)", "均(ms)", "最大(ms)", "占比"], tbl_rows)

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
        evidence=evidence[:8],
        related_hotspots=[row["label"] or "" for row in rows[:5]],
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
    except sqlite3.Error as exc:
        logger.debug("sql_nvtx_gil_anchor 失败：%s", exc)
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
        interval = (start_ns, end_ns)
        by_label[label].append(interval)
        all_intervals.append(interval)
    if not all_intervals:
        return

    observed_span = max(end for _, end in all_intervals) - min(start for start, _ in all_intervals)
    denominator_ns = max(int(total_ns or 0), observed_span, 1)
    wall_ns = min(union_ns(all_intervals), denominator_ns)
    inclusive_ns = sum(end - start for start, end in all_intervals)
    evidence = [
        f"GIL NVTX union {wall_ns / 1e6:.1f}ms（占 NVTX 观测窗口 {wall_ns / denominator_ns * 100:.1f}%），inclusive {inclusive_ns / 1e6:.1f}ms（{inclusive_ns / denominator_ns * 100:.1f}%）",
        "inclusive 可超过 100%，表示多个线程上的 GIL NVTX range 并行重叠。",
    ]
    for label in ("Waiting for GIL", "Holding GIL"):
        intervals = by_label.get(label, [])
        if not intervals:
            continue
        evidence.append(
            f"{label}: {len(intervals)} 次，{min(union_ns(intervals), denominator_ns) / 1e6:.1f}ms union，{sum(end - start for start, end in intervals) / 1e6:.1f}ms inclusive"
        )

    findings.append(NsysFinding(
        category="host_gil_contention",
        severity="warning" if wall_ns / denominator_ns * 100 >= 10.0 or by_label.get("Waiting for GIL") else "info",
        title=f"Host GIL NVTX 活动覆盖观测窗口 {wall_ns / denominator_ns * 100:.1f}%",
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
    """将每个 GPU kernel 归因到包裹它的最内层 NVTX range。"""
    trim_sql = ""
    trim_params: list[int] = []
    if trim:
        trim_sql = "AND k.start >= ? AND k.[end] <= ?"
        trim_params = [trim[0], trim[1]]

    name_expr, kernel_join = _kernel_name_expr(conn, kernel_tbl, has_strings, alias="k")
    try:
        kernel_rows = conn.execute(
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
    except sqlite3.Error as exc:
        logger.debug("attribute_kernels_to_nvtx (kr join) 失败：%s", exc)
        return []
    if not kernel_rows:
        return []

    has_textid = "textId" in _get_cols(conn, nvtx_tbl)
    if has_strings and has_textid:
        text_expr = "COALESCE(n.text, s.value)"
        text_join = "LEFT JOIN StringIds s ON n.textId = s.id"
    else:
        text_expr = "n.text"
        text_join = ""

    tids = sorted({int(row[0]) for row in kernel_rows if row[0] is not None})
    if not tids:
        return []
    min_r_start = min(int(row[1]) for row in kernel_rows)
    max_r_end = max(int(row[2]) for row in kernel_rows)

    if len(tids) <= 900:
        tid_clause = f"AND n.globalTid IN ({','.join('?' for _ in tids)})"
        nvtx_params: list[int] = list(tids) + [max_r_end, min_r_start]
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
    except sqlite3.Error as exc:
        logger.debug("attribute_kernels_to_nvtx (nvtx load) 失败：%s", exc)
        return []

    nvtx_by_tid: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for row in nvtx_rows:
        tid = int(row[0])
        text = row[3]
        if text:
            nvtx_by_tid[tid].append((int(row[1]), int(row[2]), str(text)))

    kernels_by_tid: dict[int, list[tuple[int, int, int, int, str]]] = defaultdict(list)
    for row in kernel_rows:
        tid = int(row[0]) if row[0] is not None else 0
        kernels_by_tid[tid].append((
            int(row[1]),
            int(row[2]),
            int(row[3]),
            int(row[4]),
            row[5] or "unknown",
        ))

    results: list[NvtxKernelAttribution] = []
    for tid, kernel_list in kernels_by_tid.items():
        nvtx_list = nvtx_by_tid.get(tid)
        if not nvtx_list:
            continue
        kernel_list.sort(key=lambda item: item[0])

        open_stack: list[tuple[int, int, str]] = []
        nvtx_idx = 0
        for r_start, r_end, k_start, k_end, kernel_name in kernel_list:
            while open_stack and open_stack[-1][1] < r_start:
                open_stack.pop()
            while nvtx_idx < len(nvtx_list) and nvtx_list[nvtx_idx][0] <= r_start:
                start_ns, end_ns, text = nvtx_list[nvtx_idx]
                if end_ns >= r_start:
                    open_stack.append((start_ns, end_ns, text))
                nvtx_idx += 1

            best_idx = -1
            for idx in range(len(open_stack) - 1, -1, -1):
                start_ns, end_ns, _ = open_stack[idx]
                if start_ns <= r_start and end_ns >= r_end:
                    best_idx = idx
                    break
            if best_idx < 0:
                continue

            enclosing = [
                entry for entry in open_stack[: best_idx + 1]
                if entry[0] <= r_start and entry[1] >= r_end
            ]
            if not enclosing:
                continue

            results.append(NvtxKernelAttribution(
                kernel_name=kernel_name,
                k_start_ns=k_start,
                k_end_ns=k_end,
                k_dur_ns=k_end - k_start,
                nvtx_text=enclosing[-1][2],
                nvtx_path=" > ".join(entry[2] for entry in enclosing),
                nvtx_depth=len(enclosing) - 1,
            ))

    results.sort(key=lambda row: row.k_start_ns)
    return results[:limit] if limit is not None else results


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
    """按 NVTX region 汇总 GPU 时间，并标出离群热点。"""
    attributions = attribute_kernels_to_nvtx(conn, kernel_tbl, runtime_tbl, nvtx_tbl, has_strings)
    if not attributions:
        return

    groups: dict[str, dict[str, object]] = {}
    for attr in attributions:
        group = groups.setdefault(attr.nvtx_path, {
            "total_ns": 0,
            "nccl_ns": 0,
            "compute_ns": 0,
            "count": 0,
            "nvtx_depth": attr.nvtx_depth,
            "k_times": defaultdict(int),
        })
        group["total_ns"] = int(group["total_ns"]) + attr.k_dur_ns
        group["count"] = int(group["count"]) + 1
        if _is_nccl(attr.kernel_name):
            group["nccl_ns"] = int(group["nccl_ns"]) + attr.k_dur_ns
        else:
            group["compute_ns"] = int(group["compute_ns"]) + attr.k_dur_ns
        group["k_times"][attr.kernel_name] += attr.k_dur_ns

    sorted_groups = sorted(groups.items(), key=lambda item: -int(item[1]["total_ns"]))
    if not sorted_groups:
        return

    all_times_ms = sorted(int(group["total_ns"]) / 1e6 for _, group in sorted_groups if int(group["total_ns"]) > 0)
    count = len(all_times_ms)
    if count >= 4:
        def _quantile(p: float) -> float:
            pos = p * (count - 1)
            lo = int(pos)
            frac = pos - lo
            if frac == 0:
                return all_times_ms[lo]
            hi = min(lo + 1, count - 1)
            return all_times_ms[lo] + frac * (all_times_ms[hi] - all_times_ms[lo])
        median = _quantile(0.5)
        q1 = _quantile(0.25)
        q3 = _quantile(0.75)
        iqr = q3 - q1
        fence = q3 + 1.5 * iqr if iqr > 0 else median * 2.0
    elif count >= 2:
        mid = count // 2
        median = all_times_ms[mid] if count % 2 else (all_times_ms[mid - 1] + all_times_ms[mid]) / 2
        fence = median * 2.0
    else:
        median = 0.0
        fence = float("inf")

    total_trace_ms = max(total_ns / 1e6, 0.001)
    tbl_rows = []
    outliers: list[str] = []
    for nvtx_path, group in sorted_groups[:limit]:
        total_ms = int(group["total_ns"]) / 1e6
        nccl_ms = int(group["nccl_ns"]) / 1e6
        compute_ms = int(group["compute_ns"]) / 1e6
        nccl_pct = 100.0 * int(group["nccl_ns"]) / int(group["total_ns"]) if int(group["total_ns"]) > 0 else 0.0
        pct_of_trace = total_ms / total_trace_ms * 100
        is_outlier = count >= 2 and total_ms > fence and total_ms > median * 1.5
        display_path = nvtx_path if len(nvtx_path) <= 40 else "..." + nvtx_path[-37:]
        top_k = sorted(group["k_times"].items(), key=lambda item: -item[1])[:3]
        tbl_rows.append([
            display_path,
            str(group["nvtx_depth"]),
            str(group["count"]),
            f"{total_ms:.2f}",
            f"{compute_ms:.2f}",
            f"{nccl_ms:.2f}",
            f"{nccl_pct:.1f}%",
            f"{pct_of_trace:.1f}%",
            "⚠️" if is_outlier else "",
        ])
        if is_outlier:
            ratio = total_ms / median if median > 0 else 0.0
            outliers.append(
                f"⚠️ 离群层 '{display_path}' {total_ms:.1f}ms（{ratio:.1f}× 中位数 {median:.1f}ms），热点："
                + ", ".join(f"{name[:25]}({dur / 1e6:.1f}ms)" for name, dur in top_k)
            )

    evidence = _fmt_table(
        ["NVTX Region", "深度", "内核数", "总GPU(ms)", "计算(ms)", "NCCL(ms)", "NCCL%", "占trace%", "异常"],
        tbl_rows,
    )[:12]
    if outliers:
        evidence += outliers[:3]

    top_path, top_group = sorted_groups[0]
    top_ms = int(top_group["total_ns"]) / 1e6
    top_pct = top_ms / total_trace_ms * 100
    findings.append(NsysFinding(
        category="sql_nvtx_layer_breakdown",
        severity="warning" if outliers else "info",
        title=(
            f"NVTX 层 GPU 时间分解：{len(sorted_groups)} 个 region，"
            f"热点 '{top_path[:40]}' {top_ms:.1f}ms（{top_pct:.1f}%）"
            + (f"，{len(outliers)} 个离群层" if outliers else "")
        ),
        description=(
            f"通过 correlationId 链将 {len(attributions)} 个 GPU kernel 归因到 {len(sorted_groups)} 个 NVTX region，"
            f"热点区域 '{top_path[:50]}' 占 trace {top_pct:.1f}%。"
            + (f" 检测到 {len(outliers)} 个统计离群层。" if outliers else "")
        ),
        evidence=evidence,
        related_hotspots=[path for path, _ in sorted_groups[:5]],
        next_step=(
            "关注耗时最多且 NCCL% 高的层——这是通信优化的直接目标。"
            "离群层（⚠️）显著慢于中位数，优先排查其 top kernel。"
            "使用 Nsight Compute 对热点 kernel 进行深层分析。"
        ),
    ))
