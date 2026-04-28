"""nsys/render.py — NsysDiag 终端报告渲染器。

设计目标：
- 纯静态文本报告，无 TUI / curses / rich 依赖。
- 仅用标准库：textwrap、shutil、datetime。
- 信息层次：
    头部 → 采集质量 → 总览（任意/平均/最差 GPU）
    → 建议排查顺序 → 瓶颈分布
    → 分域诊断（主要 finding 合并；SQL 作辅助证据）
    → CPU 热点 → 指标语义 / 警告
- 机器可读 JSON 保留在 cli.py (_nsys_diag_to_dict)。

指标语义：
- "Union %"    = interval-union(events) / trace wall time  （≤ 100%）
- "Inclusive %" = 所有事件时长之和 / trace wall time（多流/设备/线程重叠时可超 100%）
- BottleneckLabel.pct_of_trace 对 gpu_idle/sync_wait/GPU 内核是 Union %；
  SQL 类 finding 是 Inclusive %，渲染时会加标注。
"""

from __future__ import annotations

import shutil
import textwrap
from datetime import datetime
from typing import Sequence

from .models import (
    BottleneckSummary,
    NsysDiag,
    NsysFinding,
    SampleHotspot,
)
from .text import format_table, pad_display

try:
    from .sql_cli import run_sql_nvtx, run_sql_sync, run_sql_memcpy
    _SQL_AVAILABLE = True
except Exception:
    _SQL_AVAILABLE = False


# ── 终端宽度 ───────────────────────────────────────────────────────────────────

def _term_width() -> int:
    return min(shutil.get_terminal_size((100, 40)).columns, 120)


# ── 基础渲染原语 ───────────────────────────────────────────────────────────────

def _rule(width: int, char: str = "─") -> str:
    return char * width


def _section(title: str, width: int) -> str:
    bar = _rule(width)
    return f"{bar}\n  {title}\n{bar}"


def _kv_block(rows: list[tuple[str, str]], key_width: int = 14) -> list[str]:
    lines = []
    for k, v in rows:
        lines.append(f"  {pad_display(k, key_width)}  {v}")
    return lines


def _bar(pct: float, width: int = 20) -> str:
    """ASCII 进度条 [████░░░░░░]，pct 限制在 [0, 1] 范围内显示。"""
    clamped = max(0.0, min(1.0, pct))
    filled = round(clamped * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _fmt_ms(ns: int | float) -> str:
    ms = ns / 1e6
    if ms >= 1000:
        return f"{ms / 1000:.2f}s "
    return f"{ms:.1f}ms"


def _fmt_pct(ratio: float) -> str:
    return f"{ratio * 100:5.1f}%"


def _fmt_bytes_gib(value: int | None) -> str:
    if value is None:
        return "—"
    gib = value / (1024 ** 3)
    rounded = int(round(gib / 5.0) * 5) if gib >= 20 else int(round(gib))
    return f"{max(rounded, 1)}GB"


def _fmt_bandwidth_gbps(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{value / 1e9:.0f}GB/s"


def _compact_gpu_label(name: str) -> str:
    normalized = " ".join(name.split())
    for prefix in ("NVIDIA ", "NVIDIA"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].strip()
    return normalized or name


def _render_gpu_inventory(diag: NsysDiag) -> str | None:
    if not diag.gpu_devices:
        return None

    by_name: dict[str, list] = {}
    for device in diag.gpu_devices:
        key = _compact_gpu_label(device.name)
        by_name.setdefault(key, []).append(device)

    parts: list[str] = []
    for name, devices in sorted(by_name.items(), key=lambda item: (-len(item[1]), item[0])):
        first = devices[0]
        detail = [
            f"HBM {_fmt_bytes_gib(first.total_memory_bytes)}",
        ]
        if first.sm_count is not None:
            detail.append(f"SM {first.sm_count}")
        if first.compute_capability:
            detail.append(f"CC {first.compute_capability}")
        if first.memory_bandwidth_bytes_per_s is not None:
            detail.append(_fmt_bandwidth_gbps(first.memory_bandwidth_bytes_per_s))
        parts.append(f"{name}*{len(devices)} ({', '.join(detail)})")
    return "； ".join(parts)


def _question_lookup(diag: NsysDiag) -> dict[str, object]:
    if not diag.localization:
        return {}
    return {question.question_id: question for question in diag.localization.questions}


def _anchor_lookup(diag: NsysDiag) -> dict[str, object]:
    if not diag.localization:
        return {}
    return {anchor.window_id: anchor for anchor in diag.localization.anchors}


def _window_lookup(diag: NsysDiag) -> dict[str, object]:
    return {
        f"W{idx}": window
        for idx, window in enumerate(diag.windows, start=1)
    }


def _callsite_from_result(item) -> str:
    parts = [item.file_path] if getattr(item, "file_path", "") else []
    if getattr(item, "line", None) is not None:
        parts.append(str(item.line))
    if getattr(item, "function", ""):
        parts.append(item.function)
    return ":".join(parts)


def _window_identity_label(window) -> str:
    iter_label = ""
    for label in getattr(window, "nvtx_labels", []) or []:
        text = (label or "").strip()
        if text.lower().startswith("iter_"):
            iter_label = text
            break
    rank = getattr(window, "window_rank_in_iter", None)
    if not iter_label or rank is None:
        return ""

    category = getattr(window, "event_category", "")
    runtime_name = (getattr(window, "runtime_api", None) or getattr(window, "event_name", "") or "").lower()
    kind = "窗口"
    if category == "sync_wait":
        kind = "同步"
    elif category == "cuda_api" or "launch" in runtime_name:
        kind = "launch"
    elif category == "gpu_memcpy" or "memcpy" in runtime_name:
        kind = "memcpy"
    return f"{iter_label} 中第 {rank} 个{kind}"



def _window_location_label(window, localization_status: str | None, question=None, anchor=None) -> tuple[str, str]:
    if question and getattr(question, "file_path", ""):
        return "细定位", _callsite_from_result(question)
    if anchor and getattr(anchor, "file_path", ""):
        return "细定位", _callsite_from_result(anchor)
    if question and getattr(question, "rationale", ""):
        return "细定位", question.rationale
    if anchor and getattr(anchor, "rationale", ""):
        return "细定位", anchor.rationale
    if localization_status == "ok":
        return "细定位", "见下方 Codex 调查结果"
    if localization_status == "running":
        return "定位", "Codex 调查进行中；当前无稳定代码锚点，请查看工件目录中的结果文件"
    if window.coarse_location:
        return "粗定位", window.coarse_location
    return "定位", "当前 profile 缺少稳定代码锚点；建议补充 repo-root 或使用 Codex 细定位"


def _table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    col_widths: Sequence[int] | None = None,
    indent: int = 2,
) -> list[str]:
    """渲染固定宽度文本表格。"""
    pad = " " * indent
    return [pad + ln for ln in format_table(headers, rows, col_widths)]


def _wrap(text: str, indent: int = 4, width: int | None = None) -> str:
    w = width or (_term_width() - indent)
    prefix = " " * indent
    return textwrap.fill(text, width=w, initial_indent=prefix, subsequent_indent=prefix)


def _render_evidence_block(evidence: list[str], limit: int, indent: int = 6) -> list[str]:
    """将 SQL finding 的 evidence 列表加缩进原样输出。"""
    if not evidence:
        return []

    pad = " " * indent
    # 判断是否是 _fmt_table 输出（分隔线行含 ─）
    is_table = len(evidence) >= 2 and "─" in evidence[1]

    if is_table:
        # 表头 + 分隔线 + 至多 limit 条数据
        show = evidence[:limit + 2]  # +2 for header + divider
        return [pad + ln for ln in show]
    else:
        # 自由文本，加 · 前缀
        show = evidence[:limit]
        return [f"{pad[:-2]}· {ln}" for ln in show]


# ── Domain 分类 ────────────────────────────────────────────────────────────────
# SQL 类 finding (sql_*) 是其所属 domain 的辅助证据，不单独占 Qn 编号。
# 渲染时作为 "辅助证据" 块合并展示。

_DOMAIN_ORDER = [
    "triage", "scheduling", "compute", "runtime", "memory",
    "sync", "communication", "host", "other",
]

_CATEGORY_TO_DOMAIN: dict[str, str] = {
    # 全局诊断 / 根因
    "sql_profile_health":       "triage",
    "sql_root_cause_analysis":  "triage",
    # 调度 / GPU 利用率
    "gpu_idle":               "scheduling",
    "sql_gpu_idle_gaps":      "scheduling",
    # 计算
    "gpu_compute_hotspot":    "compute",
    "many_tiny_kernels":      "compute",
    "sql_top_kernels":        "compute",
    "sql_nvtx_layer_breakdown": "compute",
    # 运行时 / 内核启动开销
    "host_launch_overhead":   "runtime",
    # 内存传输
    "gpu_memcpy_hotspot":     "memory",
    "gpu_memcpy_h2d":         "memory",
    "gpu_memcpy_d2h":         "memory",
    "gpu_memset":             "memory",
    "low_memcpy_throughput":  "memory",
    "sql_memory_bandwidth":   "memory",
    # 同步
    "sync_wait":              "sync",
    "sql_sync_cost":          "sync",
    # 通信
    "gpu_comm_hotspot":       "communication",
    "comm_not_overlapped":    "communication",
    "sql_nccl_breakdown":     "communication",
    # Host / CPU
    "cpu_hotspot":            "host",
    "host_gil_contention":     "host",
    "sql_nvtx_hotspots":      "host",
}

_DOMAIN_LABEL: dict[str, str] = {
    "triage":        "全局诊断",
    "scheduling":    "调度 / GPU 利用率",
    "compute":       "计算",
    "runtime":       "运行时 / 内核启动开销",
    "memory":        "内存传输",
    "sync":          "同步",
    "communication": "通信",
    "host":          "Host / CPU",
    "other":         "其他",
}

# SQL 类 finding 作辅助证据，不单独成 Qn。
# 注意：sql_nvtx_hotspots 是 Host/CPU 领域唯一可能的主 finding（无 CPU 采样时），
# 因此不归入辅助证据，让它作为主 Qn 展示。
_SQL_CATEGORIES = frozenset({
    "sql_profile_health",
    "sql_gpu_idle_gaps", "sql_memory_bandwidth", "sql_sync_cost",
    "sql_nccl_breakdown", "sql_top_kernels",
    "sql_nvtx_hotspots", "sql_nvtx_layer_breakdown",
})

_SEV_ICON: dict[str, str] = {
    "critical": "🔴",
    "warning":  "🟡",
    "info":     "🔵",
}
_SEV_LABEL: dict[str, str] = {
    "critical": "严重",
    "warning":  "警告",
    "info":     "提示",
}

# 这些类别的 pct_of_trace 是 inclusive sum（可超 100%）。
_INCLUSIVE_CATEGORIES = frozenset({
    "sync_wait", "cuda_api", "sql_sync_cost",
    "host_launch_overhead",
})


def _pct_tag(category: str, ratio: float) -> str:
    if category in _INCLUSIVE_CATEGORIES or ratio > 1.0:
        return f"{ratio * 100:5.1f}% incl"
    return f"{ratio * 100:5.1f}% wall"


def _group_findings(
    findings: list[NsysFinding],
) -> dict[str, tuple[list[NsysFinding], list[NsysFinding]]]:
    """按 domain 分组 → (主要 findings, SQL 辅助证据 findings)。"""
    primary: dict[str, list[NsysFinding]] = {d: [] for d in _DOMAIN_ORDER}
    sql_ev: dict[str, list[NsysFinding]] = {d: [] for d in _DOMAIN_ORDER}
    for f in findings:
        domain = _CATEGORY_TO_DOMAIN.get(f.category, "other")
        bucket = sql_ev if f.category in _SQL_CATEGORIES else primary
        bucket.setdefault(domain, []).append(f)

    # Promote selected SQL categories only when their domain lacks a direct primary finding.
    for domain in _DOMAIN_ORDER:
        direct_primary = primary.get(domain, [])
        aux = sql_ev.get(domain, [])
        promoted: list[NsysFinding] = []
        kept_aux: list[NsysFinding] = []
        for finding in aux:
            if _sql_can_promote_without_primary(finding.category) and not direct_primary:
                promoted.append(finding)
            else:
                kept_aux.append(finding)
        primary[domain] = direct_primary + promoted
        sql_ev[domain] = kept_aux

    return {d: (primary.get(d, []), sql_ev.get(d, [])) for d in _DOMAIN_ORDER}



def _sql_can_promote_without_primary(category: str) -> bool:
    return category in {"sql_root_cause_analysis", "sql_nvtx_hotspots", "sql_nvtx_layer_breakdown"}


# ── 排查队列优先级 ─────────────────────────────────────────────────────────────

_DOMAIN_PRIORITY = {
    "triage":        0,
    "scheduling":    1,
    "runtime":       2,
    "sync":          3,
    "memory":        4,
    "compute":       5,
    "communication": 6,
    "host":          7,
    "other":         8,
}

# 每个 domain 的一行排查提示。
_DOMAIN_HINT: dict[str, str] = {
    "triage":        "先按 suspected bottleneck 和 root-cause 反模式确定排查主线。",
    "scheduling":    "检查各设备活跃率、stream 空闲气泡、step bubble。",
    "runtime":       "检查 cudaLaunchKernel 调用次数、小内核融合机会、CUDA Graph 可行性。",
    "sync":          "确认同步调用的 wall-clock 实际影响；判断是否可改为异步或批量同步。",
    "memory":        "定位 .to/.cuda/copy_/pin_memory/non_blocking 调用点；区分 pageable vs pinned。",
    "compute":       "分析 top kernel 的 occupancy、寄存器溢出、warp 分歧情况。",
    "communication": "确认 exposed（未被重叠）通信时间及各 rank/stream 负载均衡情况。",
    "host":          "检查 CPU 采样热点、GIL 竞争、DataLoader 阻塞、OSRT syscall 开销。",
    "other":         "手动检查相关 findings。",
}


def _build_localization_queue(
    findings: list[NsysFinding],
    bottlenecks: BottleneckSummary | None,
) -> list[tuple[str, str, str]]:
    """返回 [(domain 标签, 摘要, 提示)]，按优先级排序，仅包含有 finding 的 domain。"""
    groups = _group_findings(findings)
    seen_domains: list[str] = []
    for domain in _DOMAIN_ORDER:
        primary, sql = groups[domain]
        if primary or sql:
            seen_domains.append(domain)

    seen_domains.sort(key=lambda d: _DOMAIN_PRIORITY.get(d, 99))

    queue = []
    for domain in seen_domains:
        primary, sql = groups[domain]
        all_f = primary + sql
        # 取严重性最高的 finding 作摘要
        top = sorted(all_f, key=lambda f: {"critical": 0, "warning": 1, "info": 2}.get(f.severity, 3))[0]
        summary = top.title
        hint = _DOMAIN_HINT.get(domain, "")
        queue.append((_DOMAIN_LABEL.get(domain, domain), summary, hint))

    return queue


# ── 采集质量 ───────────────────────────────────────────────────────────────────

def _render_capture_quality(diag: NsysDiag, width: int) -> list[str]:
    """展示各数据源的采集状态。"""
    lines: list[str] = []
    lines.append("")
    lines.append(_section("采集质量", width))
    lines.append("")

    b = diag.bottlenecks
    has_kernels = b is not None and b.gpu_active_ns > 0
    has_memcpy  = b is not None and any(lb.category == "gpu_memcpy" for lb in b.labels)
    has_nvtx    = any(f.category in ("sql_nvtx_hotspots",) for f in diag.findings)
    has_cpu     = bool(diag.hotspots)
    has_nccl    = any(f.category in ("gpu_comm_hotspot", "sql_nccl_breakdown") for f in diag.findings)
    has_per_dev = b is not None and bool(b.per_device)

    def _tick(present: bool) -> str:
        return "✅" if present else "⬜"

    rows = [
        ("GPU 内核",    f"{_tick(has_kernels)}  {'已采集' if has_kernels else '未找到'}"),
        ("GPU memcpy",  f"{_tick(has_memcpy)}  {'已采集' if has_memcpy else '未找到'}"),
        ("NCCL 通信",   f"{_tick(has_nccl)}  {'已采集' if has_nccl else '未找到'}"),
        ("NVTX 标注",   f"{_tick(has_nvtx)}  {'已采集' if has_nvtx else '未采集 — 请添加 torch.profiler NVTX 或 nvtx.range()'}"),
        ("CPU 采样",    f"{_tick(has_cpu)}  {'已采集' if has_cpu else '未采集 — nsys 默认已开启，若缺失请检查采集命令是否含 --backtrace cpu'}"),
        ("多设备数据",  f"{_tick(has_per_dev)}  {'已采集（' + str(len(b.per_device)) + ' 个 GPU）' if has_per_dev else '单设备或未解析'}"),
    ]
    lines.extend(_kv_block(rows, key_width=16))
    return lines


# ── 各 Section 渲染器 ─────────────────────────────────────────────────────────

def _render_header(diag: NsysDiag, width: int) -> list[str]:
    lines: list[str] = []
    lines.append(_rule(width, "═"))
    lines.append("  🔍  Sysight  ·  nsys Profile 报告")
    lines.append(_rule(width, "═"))
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    status_map = {
        "ok":               "✅  正常",
        "error":            "❌  错误",
        "action_required":  "⚠️  需要操作",
    }
    rows = [
        ("Profile 文件",  diag.profile_path or "—"),
        ("SQLite",        diag.sqlite_path or "—"),
        ("状态",          status_map.get(diag.status, diag.status)),
        ("生成时间",      now),
    ]
    lines.extend(_kv_block(rows, key_width=12))
    return lines


def _render_action_required(diag: NsysDiag, width: int) -> list[str]:
    lines: list[str] = []
    lines.append("")
    lines.append(_section("需要操作", width))
    lines.append("")
    if diag.required_action:
        for ln in diag.required_action.splitlines():
            lines.append(f"    {ln}")
    lines.append("")
    lines.append("  完成操作后重新运行：")
    profile = diag.profile_path
    lines.append(f"    PYTHONPATH=src python3 -m sysight.analyzer nsys {profile}")
    return lines


def _render_error(diag: NsysDiag, width: int) -> list[str]:
    lines: list[str] = []
    lines.append("")
    lines.append(_section("错误详情", width))
    lines.append("")
    for w in diag.warnings:
        lines.append(f"  ❌  {w}")
    if diag.summary:
        lines.append("")
        lines.append(_wrap(diag.summary, indent=4))
    return lines


def _render_summary(diag: NsysDiag, width: int) -> list[str]:
    lines: list[str] = []
    lines.append("")
    lines.append(_section("总览", width))
    lines.append("")

    b = diag.bottlenecks
    if b:
        any_active_r = b.gpu_active_ns / b.total_ns if b.total_ns else 0.0
        idle_r = 1.0 - any_active_r

        rows: list[tuple[str, str]] = [
            ("Trace 时长",    _fmt_ms(b.total_ns)),
            ("GPU 信息",      _render_gpu_inventory(diag) or "—"),
            ("任意 GPU 活跃", f"{_fmt_ms(b.gpu_active_ns)}  {_fmt_pct(any_active_r)} wall  {_bar(any_active_r)}"),
            ("任意 GPU 空闲", f"{_fmt_ms(b.gpu_idle_ns)}   {_fmt_pct(idle_r)} wall  {_bar(idle_r)}"),
        ]

        if b.per_device:
            actives = [dev.pct_active for dev in b.per_device]
            avg_r = sum(actives) / len(actives)
            worst_dev = min(b.per_device, key=lambda d: d.pct_active)
            best_dev  = max(b.per_device, key=lambda d: d.pct_active)
            skew_pp = (best_dev.pct_active - worst_dev.pct_active) * 100
            rows.append(("平均 GPU 活跃",   f"{_fmt_pct(avg_r)} wall  {_bar(avg_r)}"))
            rows.append(("最差 GPU 活跃",   f"{_fmt_pct(worst_dev.pct_active)} wall  GPU {worst_dev.device_id}"))
            rows.append(("最好 GPU 活跃",   f"{_fmt_pct(best_dev.pct_active)} wall  GPU {best_dev.device_id}"))
            rows.append(("设备负载偏差",    f"{skew_pp:.1f} pp"))

        lines.extend(_kv_block(rows, key_width=16))

    if diag.summary:
        lines.append("")
        lines.append(_wrap(diag.summary, indent=2, width=width - 4))

    return lines


def _render_localization_queue(diag: NsysDiag, width: int) -> list[str]:
    if not diag.findings:
        return []

    queue = _build_localization_queue(diag.findings, diag.bottlenecks)
    if not queue:
        return []

    lines: list[str] = []
    lines.append("")
    lines.append(_section("建议排查顺序", width))
    lines.append("")

    for i, (domain_label, summary, hint) in enumerate(queue, 1):
        lines.append(f"  {i}. {domain_label}")
        lines.append(f"     发现：  {summary}")
        if hint:
            lines.append(f"     排查：  {hint}")
        lines.append("")

    return lines



def _render_localization_result(diag: NsysDiag, width: int, verbose: bool = False) -> list[str]:
    inv = diag.localization
    if inv is None:
        return []
    if inv.status == "ok" and inv.questions and not verbose:
        return ["", _section("Codex 调查结果", width), "", "  （暂停使用）逐题明细暂不打印；Codex 已回填到下方优化建议。"]

    lines: list[str] = []
    lines.append("")
    lines.append(_section("Codex 调查结果", width))
    lines.append("")
    lines.extend(_kv_block([
        ("Backend", inv.backend),
        ("状态", inv.status),
        ("命令", " ".join(inv.command) if inv.command else "—"),
        ("PID", str(inv.pid) if inv.pid else "—"),
        ("工件目录", inv.artifact_dir or "—"),
        ("Prompt", inv.prompt_path or "—"),
        ("Stdout", inv.stdout_path or "—"),
        ("Stderr", inv.stderr_path or "—"),
        ("结果文件", inv.output_path or "—"),
    ], key_width=12))

    if inv.error:
        lines.append("")
        lines.append(_wrap(f"错误：{inv.error}", indent=2, width=width - 4))

    if inv.summary:
        lines.append("")
        lines.append(_wrap(f"总结：{inv.summary}", indent=2, width=width - 4))

    if inv.questions:
        lines.append("")
        lines.append("  （暂停使用）逐题明细暂不打印；Codex 已回填到下方优化建议。")
    elif inv.anchors:
        lines.append("")
        lines.append("  结构化结果（窗口级回退）：")
        limit = len(inv.anchors) if verbose else min(len(inv.anchors), 8)
        for anchor in inv.anchors[:limit]:
            status = anchor.status or "unknown"
            location = _callsite_from_result(anchor) or "未定位到代码文件"
            lines.append(_wrap(f"{anchor.window_id}. [{status}] {location}", indent=4, width=width - 6))
            if anchor.rationale:
                lines.append(_wrap(f"原因：{anchor.rationale}", indent=6, width=width - 8))
            if anchor.suggestion:
                lines.append(_wrap(f"建议：{anchor.suggestion}", indent=6, width=width - 8))
        if len(inv.anchors) > limit:
            lines.append(f"    … 还有 {len(inv.anchors) - limit} 条结构化结果（使用 --report full 展开）")

    return lines



def _priority_from_severity(severity: str) -> str:
    return {
        "critical": "P0",
        "warning": "P1",
        "info": "P2",
    }.get(severity, "P1")


def _render_recommendations(diag: NsysDiag, width: int, verbose: bool = False) -> list[str]:
    if diag.localization is None or not diag.localization.questions:
        return []

    lines: list[str] = []
    lines.append("")
    lines.append(_section("优化建议", width))
    lines.append("")

    localization_status = diag.localization.status
    anchor_by_id = _anchor_lookup(diag)
    window_by_id = _window_lookup(diag)
    finding_by_problem = {
        (finding.stable_id or finding.category): finding
        for finding in diag.findings
    }

    for idx, question in enumerate(diag.localization.questions[:12], start=1):
        finding = finding_by_problem.get(question.problem_id)
        priority = _priority_from_severity(finding.severity if finding else "warning")
        title = question.title or (finding.title if finding else question.category or question.problem_id)
        lines.append(f"  {idx}. [{question.question_id}/{priority}] {title}")
        lines.append(f"     问题：  {question.category or (finding.category if finding else 'unknown')}")

        emitted_locations: set[str] = set()
        matched_windows = [window_by_id[window_id] for window_id in question.window_ids if window_id in window_by_id]
        if matched_windows:
            win_desc = ", ".join(
                f"{w.event_name}@{w.start_ns/1e6:.3f}-{w.end_ns/1e6:.3f}ms"
                for w in matched_windows[:3]
            )
            lines.append(f"     窗口：  {win_desc}")
            for window_id in question.window_ids[:2]:
                window = window_by_id.get(window_id)
                if window is None:
                    continue
                anchor = anchor_by_id.get(window_id)
                loc_label, loc_value = _window_location_label(window, localization_status, question=question, anchor=anchor)
                identity = _window_identity_label(window)
                prefix = f"{identity} / " if identity else ""
                lines.append(_wrap(f"{loc_label}：  {prefix}{window.event_name}@{window.start_ns/1e6:.3f}ms -> {loc_value}", indent=5, width=width - 6))
                if loc_label == "细定位":
                    emitted_locations.add(loc_value)

        callsite = _callsite_from_result(question)
        if callsite and callsite not in emitted_locations:
            lines.append(f"     细定位：  {callsite}")
        if question.rationale:
            lines.append(_wrap(f"原因：  {question.rationale}", indent=5, width=width - 6))
        if question.suggestion:
            lines.append(_wrap(f"动作：  {question.suggestion}", indent=5, width=width - 6))
        if finding and finding.next_step and finding.next_step != question.suggestion:
            lines.append(_wrap(f"补充建议：  {finding.next_step}", indent=5, width=width - 6))
        lines.append("")

    return lines


def _render_bottlenecks(diag: NsysDiag, width: int) -> list[str]:
    b = diag.bottlenecks
    if not b:
        return []

    lines: list[str] = []
    lines.append("")
    lines.append(_section("瓶颈分布  （wall/union 时间维度）", width))
    lines.append("")

    headers = ["类别", "Union %", "Inclusive %", "Union 时长", "GPU 活跃 %"]
    rows = []
    for lb in b.labels:
        incl_pct = lb.inclusive_ns / b.total_ns if b.total_ns else 0.0
        gpu_pct = (
            f"{lb.pct_of_gpu_active * 100:5.1f}%"
            if lb.pct_of_gpu_active is not None else "  —   "
        )
        union_tag = f"{lb.pct_of_trace * 100:5.1f}%"
        incl_tag = f"{incl_pct * 100:6.1f}%" + (" ⚠️" if incl_pct > 1.0 else "  ")
        rows.append([
            lb.category,
            union_tag,
            incl_tag,
            _fmt_ms(lb.active_ns),
            gpu_pct,
        ])
    lines.extend(_table(headers, rows, col_widths=[22, 8, 11, 10, 11]))
    lines.append("")
    lines.append("  ⚠️  Inclusive % > 100% 表示事件在多个 stream/线程/设备间重叠，属正常现象。")
    lines.append("    排优先级时请使用 Union %（wall-clock 维度）。")

    if b.top_events:
        lines.append("")
        lines.append(f"  Top {min(len(b.top_events), 10)} 事件  （inclusive sum / wall time）：")
        ev_headers = ["名称", "类别", "次数", "总耗时(incl)", "平均", "Incl %"]
        ev_rows = []
        for ev in b.top_events[:10]:
            incl_flag = " ⚠️" if ev.inclusive_pct > 1.0 else "  "
            ev_rows.append([
                ev.name[:46],
                ev.category,
                str(ev.count),
                _fmt_ms(ev.total_ns),
                _fmt_ms(ev.avg_ns),
                f"{ev.inclusive_pct * 100:5.1f}%{incl_flag}",
            ])
        lines.extend(_table(ev_headers, ev_rows, col_widths=[46, 14, 7, 12, 10, 9]))

    return lines


def _render_findings(diag: NsysDiag, width: int, verbose: bool = False) -> list[str]:
    """渲染分域诊断 findings 紧凑列表（仅终端展示，不包含在 Codex profile_report_text 里）。"""
    if not diag.findings:
        return []

    groups = _group_findings(diag.findings)
    lines: list[str] = []
    lines.append("")
    lines.append(_section("分域诊断", width))

    for domain in _DOMAIN_ORDER:
        primary, sql_aux = groups[domain]
        all_findings = primary + sql_aux
        if not all_findings:
            continue
        domain_label = _DOMAIN_LABEL.get(domain, domain)
        lines.append("")
        lines.append(f"  ── {domain_label} ──")
        for f in all_findings:
            icon = _SEV_ICON.get(f.severity, "  ")
            sev_label = _SEV_LABEL.get(f.severity, f.severity)
            lines.append(f"  {icon} [{sev_label}] {f.title}")
            for ev in (f.evidence or []):
                if ev.strip():
                    lines.append(f"       {ev.strip()}")

    lines.append("")
    return lines


def _render_cpu_hotspots(diag: NsysDiag, width: int, verbose: bool = False) -> list[str]:
    """Render CPU sample hotspots (profile-only, no repo mapping)."""
    if not diag.hotspots:
        return []

    lines: list[str] = []
    lines.append("")
    lines.append(_section("CPU 采样热点", width))
    lines.append("")

    headers = ["热点位置", "采样次数", "占比"]
    display_hotspots = _merge_hotspots_for_display(diag.hotspots)
    rows = []
    for h in display_hotspots[:10]:
        summary = h.coarse_location or h.frame.raw or h.frame.symbol or "—"
        rows.append([summary[:64], str(h.count), f"{h.pct * 100:.1f}%"])
    lines.extend(_table(headers, rows, col_widths=[64, 8, 8]))

    lines.append("")
    lines.append("  排查建议：优先看已归一化后的等待原因或用户态热点，不输出低信号原生采样栈。")

    if len(display_hotspots) > 10:
        lines.append(f"  … 还有 {len(display_hotspots) - 10} 个（使用 --json 查看完整列表）")

    return lines


def _merge_hotspots_for_display(hotspots: list[SampleHotspot]) -> list[SampleHotspot]:
    merged: dict[str, SampleHotspot] = {}
    for hotspot in hotspots:
        key = hotspot.coarse_location or hotspot.frame.raw or hotspot.frame.symbol or "—"
        existing = merged.get(key)
        if existing is None:
            merged[key] = hotspot
            continue
        existing.count += hotspot.count
        existing.pct += hotspot.pct
        if len(hotspot.callstack) > len(existing.callstack):
            existing.callstack = hotspot.callstack
            existing.frame = hotspot.frame
            existing.event_window_ns = hotspot.event_window_ns
    return sorted(merged.values(), key=lambda item: item.count, reverse=True)



def _callstack_triage_hint(callstack: list[str]) -> str:
    text = " <- ".join(callstack).lower()
    if any(token in text for token in ("watchdog", "cond_wait", "cond_timedwait", "gil", "restorethread")):
        return "等待/阻塞"
    if any(token in text for token in ("synchronize", "memcpy", "copy_device", "copy_") ):
        return "同步/拷贝"
    if any(token in text for token in ("launch", "cuda", "kernel")):
        return "launch/runtime"
    return "用户态热点"



def _render_sql_overview(diag: NsysDiag, width: int) -> list[str]:
    """NVTX / sync / memcpy 运行时概览（仅终端输出）。"""
    if not _SQL_AVAILABLE or not diag.sqlite_path:
        return []

    lines: list[str] = []
    lines.append("")
    lines.append(_section("SQL 概览（NVTX / 同步 / 内存拷贝）", width))

    try:
        nvtx = run_sql_nvtx(diag.sqlite_path, limit=15)
        if nvtx.nvtx_ranges:
            lines.append("")
            lines.append("  NVTX range 耗时（top 15，按总耗时排序）：")
            headers = ["标签", "次数", "总耗时", "平均"]
            rows = []
            for r in nvtx.nvtx_ranges:
                rows.append([
                    r.text[:40],
                    str(r.count),
                    _fmt_ms(r.total_ns),
                    _fmt_ms(r.avg_ns),
                ])
            lines.extend(_table(headers, rows, col_widths=[40, 6, 10, 10]))
    except Exception:
        pass

    try:
        sync = run_sql_sync(diag.sqlite_path)
        if sync.sync_events:
            lines.append("")
            lines.append(f"  CUDA 同步事件（wall pct={sync.sync_wall_pct}%）：")
            headers = ["类型", "次数", "总耗时", "avg"]
            rows = []
            for s in sync.sync_events:
                rows.append([
                    s.sync_type[-40:],
                    str(s.count),
                    _fmt_ms(s.total_ns),
                    f"{s.avg_ns/1e3:.0f}us",
                ])
            lines.extend(_table(headers, rows, col_widths=[40, 6, 10, 10]))
    except Exception:
        pass

    try:
        memcpy = run_sql_memcpy(diag.sqlite_path)
        if memcpy.memcpy_ops:
            lines.append("")
            lines.append("  内存拷贝统计：")
            headers = ["方向", "次数", "总量(MB)", "耗时", "BW(GB/s)"]
            rows = []
            for m in memcpy.memcpy_ops:
                rows.append([
                    m.direction,
                    str(m.count),
                    f"{m.total_bytes / 1e6:.2f}",
                    _fmt_ms(m.total_ns),
                    f"{m.avg_bw_gbps:.1f}",
                ])
            lines.extend(_table(headers, rows, col_widths=[8, 6, 10, 10, 10]))
    except Exception:
        pass

    lines.append("")
    return lines


def _render_warnings(diag: NsysDiag, width: int) -> list[str]:
    lines: list[str] = []

    has_warnings = bool(diag.warnings)
    has_semantics = diag.bottlenecks is not None

    if not has_warnings and not has_semantics:
        return []

    lines.append("")
    lines.append(_section("指标语义 & 说明", width))
    lines.append("")
    lines.append("  百分比定义：")
    lines.append("    Union %     = interval-union(事件) / trace wall time  [≤ 100%]")
    lines.append("    Inclusive % = 所有事件时长之和 / trace wall time  [多流重叠时可超 100%]")
    lines.append("    Inclusive > 100% 在多 stream/线程/设备重叠时属正常，不代表实际 wall 耗时。")
    lines.append("    排优先级请用 Union %；分析单线程/单流负载时再参考 Inclusive %。")

    if diag.warnings:
        lines.append("")
    for w in diag.warnings:
        lines.append(f"  💡  {w}")

    return lines


# ── 公共入口 ───────────────────────────────────────────────────────────────────

def render_nsys_profile_report(diag: NsysDiag, verbose: bool = False) -> str:
    """Render the profile-side report only, without Codex localization output.

    Section order (compact):
      1. Header
      2. Summary
      3. Bottleneck distribution
      4. CPU hotspots
      5. Metric semantics / warnings

    Sections intentionally omitted (noise for Codex / downstream consumers):
      - Capture quality  (metadata, not signal)
      - Localization queue / suggested order  (analyzer heuristics, not ground truth)
      - Domain findings  (heuristic layer; Codex should reason from raw numbers)
    """
    width = _term_width()
    parts: list[list[str]] = []

    parts.append(_render_header(diag, width))

    if diag.status == "action_required":
        parts.append(_render_action_required(diag, width))
        parts.append([[_rule(width, "═")]])  # type: ignore[list-item]
        return "\n".join(ln for block in parts for ln in block)

    if diag.status == "error":
        parts.append(_render_error(diag, width))
        parts.append([[_rule(width, "═")]])  # type: ignore[list-item]
        return "\n".join(ln for block in parts for ln in block)

    parts.append(_render_summary(diag, width))
    parts.append(_render_bottlenecks(diag, width))
    parts.append(_render_cpu_hotspots(diag, width, verbose=verbose))
    parts.append(_render_warnings(diag, width))

    lines_flat = [ln for block in parts for ln in block]
    lines_flat.append("")
    lines_flat.append(_rule(width, "═"))

    return "\n".join(lines_flat)


def render_nsys_terminal(diag: NsysDiag, verbose: bool = False) -> str:
    """将 NsysDiag 渲染为结构化终端报告。

    Section 顺序（compact）：
      1. 头部
      2. 总览  （任意 / 平均 / 最差 GPU）
      3. 瓶颈分布  （Union % + Inclusive % + ⚠ 标注）
      4. CPU 采样热点
      5. Codex 调查结果（有调查时）
      6. 优化建议（有调查时）
      7. 指标语义 & 说明

    以下 section 不再输出（交由 Codex 自主推断）：
      - 采集质量
      - 建议排查顺序
      - 分域诊断
    """
    width = _term_width()
    parts: list[list[str]] = []

    parts.append(_render_header(diag, width))

    if diag.status == "action_required":
        parts.append(_render_action_required(diag, width))
        parts.append([[_rule(width, "═")]])  # type: ignore[list-item]
        return "\n".join(ln for block in parts for ln in block)

    if diag.status == "error":
        parts.append(_render_error(diag, width))
        parts.append([[_rule(width, "═")]])  # type: ignore[list-item]
        return "\n".join(ln for block in parts for ln in block)

    parts.append(_render_summary(diag, width))
    parts.append(_render_bottlenecks(diag, width))
    parts.append(_render_cpu_hotspots(diag, width, verbose=verbose))
    parts.append(_render_sql_overview(diag, width))
    if diag.localization is not None:
        parts.append(_render_localization_result(diag, width, verbose=verbose))
        parts.append(_render_recommendations(diag, width, verbose=verbose))
    parts.append(_render_warnings(diag, width))

    lines_flat = [ln for block in parts for ln in block]
    lines_flat.append("")
    lines_flat.append(_rule(width, "═"))

    return "\n".join(lines_flat)
