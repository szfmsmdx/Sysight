"""nsys/render.py — NsysDiag 终端报告渲染器。

设计目标：
- 纯静态文本报告，无 TUI / curses / rich 依赖。
- 仅用标准库：textwrap、shutil、datetime。
- 信息层次：
    头部 → 采集质量 → 总览（任意/平均/最差 GPU）
    → 建议排查顺序 → 瓶颈分布
    → 分域诊断（主要 finding 合并；SQL 作辅助证据）
    → 代码定位 → 任务草案 → 指标语义 / 警告
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
    NsysDiag, NsysFinding, BottleneckSummary,
    MappedHotspot, TaskDraft,
)
from .text import format_table, pad_display


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
    """将 SQL finding 的 evidence 列表加缩进原样输出。

    classify_sql.py 已通过 _fmt_table() 保证对齐，这里只负责缩进。
    evidence[0] 是表头，evidence[1] 是分隔线，evidence[2:] 是数据行。
    非表格类 evidence（降级分支）逐行加 · 前缀输出。
    """
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

_DOMAIN_ORDER = ["scheduling", "compute", "runtime", "memory", "sync", "communication", "host", "other"]

_CATEGORY_TO_DOMAIN: dict[str, str] = {
    # 调度 / GPU 利用率
    "gpu_idle":               "scheduling",
    "sql_gpu_idle_gaps":      "scheduling",
    # 计算
    "gpu_compute_hotspot":    "compute",
    "many_tiny_kernels":      "compute",
    "sql_top_kernels":        "compute",
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
    "sql_nvtx_hotspots":      "host",
}

_DOMAIN_LABEL: dict[str, str] = {
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
    "sql_gpu_idle_gaps", "sql_memory_bandwidth", "sql_sync_cost",
    "sql_nccl_breakdown", "sql_top_kernels",
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
        if f.category in _SQL_CATEGORIES:
            sql_ev.setdefault(domain, []).append(f)
        else:
            primary.setdefault(domain, []).append(f)
    return {d: (primary.get(d, []), sql_ev.get(d, [])) for d in _DOMAIN_ORDER}


# ── 排查队列优先级 ─────────────────────────────────────────────────────────────
# 数字越小优先级越高。
# 原则：先修阻塞性问题（调度/启动/同步），再看内存/计算，最后看通信（通常已隐藏）。

_DOMAIN_PRIORITY = {
    "scheduling":    1,
    "runtime":       2,
    "sync":          3,
    "memory":        4,
    "compute":       5,
    "communication": 6,
    "host":          7,
    "other":         8,
}

# 每个 domain 的一行排查提示（仅 profile 模式，不含代码修改建议）。
_DOMAIN_HINT: dict[str, str] = {
    "scheduling":    "检查各设备活跃率、stream 空闲气泡、step bubble。",
    "runtime":       "检查 cudaLaunchKernel 调用次数、小内核融合机会、CUDA Graph 可行性。",
    "sync":          "确认同步调用的 wall-clock 实际影响；判断是否可改为异步或批量同步。",
    "memory":        "定位 .to/.cuda/copy_/pin_memory/non_blocking 调用点；区分 pageable vs pinned。",
    "compute":       "分析 top kernel 的 occupancy、寄存器溢出、warp 分歧情况。",
    "communication": "确认 exposed（未被重叠）通信时间及各 rank/stream 负载均衡情况。",
    "host":          "检查 CPU 采样热点、GIL 竞争、DataLoader 阻塞、OSRT syscall 开销。",
    "other":         "手动检查相关 findings。",
}


def _build_investigation_queue(
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
    has_cpu     = any(f.category == "cpu_hotspot" for f in diag.findings)
    has_nccl    = any(f.category in ("gpu_comm_hotspot", "sql_nccl_breakdown") for f in diag.findings)
    has_repo    = bool(diag.hotspots)
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
        ("代码仓库映射", f"{_tick(has_repo)}  {'已启用' if has_repo else '未启用（--no-repo）'}"),
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
    repo = diag.profile_path.rsplit("/", 1)[0] if "/" in (diag.profile_path or "") else "."
    lines.append(f"    PYTHONPATH=src python3 -m sysight.analyzer {repo} nsys {diag.profile_path}")
    return lines


def _render_error(diag: NsysDiag, width: int) -> list[str]:
    lines: list[str] = []
    lines.append("")
    lines.append(_section("错误详情", width))
    lines.append("")
    for w in diag.repo_warnings:
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


def _render_investigation_queue(diag: NsysDiag, width: int) -> list[str]:
    if not diag.findings:
        return []

    queue = _build_investigation_queue(diag.findings, diag.bottlenecks)
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


def _render_bottlenecks(diag: NsysDiag, width: int) -> list[str]:
    b = diag.bottlenecks
    if not b:
        return []

    lines: list[str] = []
    lines.append("")
    lines.append(_section("瓶颈分布  （wall/union 时间维度）", width))
    lines.append("")

    headers = ["类别", "Union %", "Inclusive %", "Union 时长", "GPU 活跃 %", "置信度"]
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
            f"{lb.confidence:.2f}",
        ])
    lines.extend(_table(headers, rows, col_widths=[22, 8, 11, 10, 11, 6]))
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
    if not diag.findings:
        return []

    lines: list[str] = []
    lines.append("")
    lines.append(_section("分域诊断", width))

    groups = _group_findings(diag.findings)
    q_idx = 0

    for domain in _DOMAIN_ORDER:
        primary, sql_evidence = groups[domain]
        if not primary and not sql_evidence:
            continue

        domain_label = _DOMAIN_LABEL.get(domain, domain.title())
        lines.append("")
        lines.append(f"  ── {domain_label} ──")

        # 主要 findings 作 Qn 条目
        max_show = len(primary) if verbose else min(len(primary), 3)
        for f in primary[:max_show]:
            q_idx += 1
            icon  = _SEV_ICON.get(f.severity, "·")
            label = _SEV_LABEL.get(f.severity, f.severity.upper())
            lines.append("")
            lines.append(f"    Q{q_idx}. {icon} [{label}]  {f.title}")

            if f.evidence:
                ev_limit = 5 if verbose else 3
                # sql_* finding 的 evidence 第一行是表头，走表格渲染
                if f.category.startswith("sql_"):
                    lines.append("")
                    for ln in _render_evidence_block(f.evidence, ev_limit, indent=7):
                        lines.append(ln)
                    # _fmt_table 输出：[header, divider, data0, data1, ...]
                    # evidence[0]=表头, evidence[1]=分隔线, evidence[2:]=数据
                    is_table = len(f.evidence) >= 2 and "─" in f.evidence[1]
                    data_count = len(f.evidence) - 2 if is_table else len(f.evidence)
                    if data_count > ev_limit:
                        lines.append(f"       … （还有 {data_count - ev_limit} 条）")
                else:
                    for ev in f.evidence[:ev_limit]:
                        lines.append(f"       证据  {ev}")
                    if len(f.evidence) > ev_limit:
                        lines.append(f"       证据  … （还有 {len(f.evidence) - ev_limit} 条）")

            if f.next_step:
                lines.append(_wrap(f"排查：  {f.next_step}", indent=7, width=width - 8))

        if not verbose and len(primary) > max_show:
            lines.append(f"    … 还有 {len(primary) - max_show} 条（使用 --report full 展开）")

        # SQL 证据合并块（每个最多展示 3/5 条）
        if sql_evidence:
            lines.append("")
            lines.append("    辅助证据：")
            for sf in sql_evidence:
                ev_limit = 5 if verbose else 3
                lines.extend(_render_evidence_block(sf.evidence, ev_limit))
                is_table = len(sf.evidence) >= 2 and "─" in sf.evidence[1]
                data_count = len(sf.evidence) - 2 if is_table else len(sf.evidence)
                if data_count > ev_limit:
                    lines.append(f"      · … （还有 {data_count - ev_limit} 条）")

    return lines


def _render_code_localization(diag: NsysDiag, width: int) -> list[str]:
    lines: list[str] = []
    lines.append("")
    lines.append(_section("代码定位", width))
    lines.append("")

    if not diag.hotspots:
        profile = diag.sqlite_path or diag.profile_path or "<profile.sqlite>"

        lines.append("  代码定位已跳过（--no-repo）。")
        lines.append("")
        lines.append("  启用代码仓库映射请运行：")
        lines.append(f"    PYTHONPATH=src python3 -m sysight.analyzer <repo> nsys {profile}")
        lines.append("")

        if diag.task_drafts:
            lines.append("  各任务草案搜索入口（--no-repo 模式，详情见任务草案章节）：")
            for td in diag.task_drafts:
                lines.append(f"    {td.id}：{td.hypothesis[:70]}")
                for spec in td.search_specs[:1]:
                    pat = spec.get("pattern", "")
                    lines.append(f'      rg -n "{pat}" <repo>')
        return lines

    cpu_hotspots = [
        h for h in diag.hotspots
        if h.repo_file and h.match_confidence > 0 and h.match_reason != "nvtx_region"
    ]
    nvtx_hotspots = [
        h for h in diag.hotspots
        if h.repo_file and h.match_confidence > 0 and h.match_reason == "nvtx_region"
    ]

    if not cpu_hotspots and not nvtx_hotspots:
        lines.append("  未在代码仓库中找到匹配热点（置信度均为 0 或符号无法映射）。")
        lines.append("  建议：确认 nsys 采集时携带了 --backtrace cpu，或使用 --json 查看原始热点数据。")
        return lines

    # ── CPU sample 热点 ───────────────────────────────────────────────────────
    if cpu_hotspots:
        lines.append("  CPU 采样热点：")
        lines.append("")
        headers = ["文件", "函数", "行号", "置信度"]
        rows = []
        for h in cpu_hotspots[:10]:
            fn = (h.function or "—").split("::")[-1] if h.function else "—"
            line_str = str(h.sample.frame.source_line) if h.sample.frame.source_line else "—"
            rows.append([h.repo_file[:48], fn[:28], line_str, f"{h.match_confidence:.2f}"])
        lines.extend(_table(headers, rows, col_widths=[48, 28, 6, 6]))
        if len(cpu_hotspots) > 10:
            lines.append(f"  … 还有 {len(cpu_hotspots) - 10} 个（使用 --json 查看完整列表）")

    # ── NVTX region 调用点 ────────────────────────────────────────────────────
    if nvtx_hotspots:
        if cpu_hotspots:
            lines.append("")
        lines.append("  NVTX region 调用点：")
        lines.append("")
        headers = ["文件", "函数", "行号", "NVTX region"]
        rows = []
        for h in nvtx_hotspots[:10]:
            fn = (h.function or "—").split("::")[-1] if h.function else "—"
            line_str = str(h.sample.frame.source_line) if h.sample.frame.source_line else "—"
            region = (h.sample.frame.symbol or "?")[:36]
            rows.append([h.repo_file[:40], fn[:24], line_str, region])
        lines.extend(_table(headers, rows, col_widths=[40, 24, 6, 36]))
        if len(nvtx_hotspots) > 10:
            lines.append(f"  … 还有 {len(nvtx_hotspots) - 10} 个（使用 --json 查看完整列表）")

    return lines


# 各 finding 类别对应的代码调用点搜索模式（--no-repo 时提示用）。
_CALLSITE_HINTS: dict[str, list[str]] = {
    "gpu_memcpy":         [".to(device)", ".cuda()", "copy_()", "pin_memory", "non_blocking"],
    "gpu_memcpy_hotspot": [".to(device)", ".cuda()", "copy_()", "pin_memory", "non_blocking"],
    "sync_wait":          ["synchronize()", ".item()", ".cpu()", "barrier()", "wait()"],
    "gpu_idle":           ["DataLoader", "__iter__", "collate_fn", "cudaStreamSynchronize"],
    "gpu_comm":           ["all_reduce()", "backward()", "dist.barrier()"],
    "gpu_comm_hotspot":   ["all_reduce()", "backward()", "dist.barrier()"],
    "host_launch":        ["cudaLaunchKernel", "torch.compile", "CUDA Graphs"],
}


def _render_task_drafts(diag: NsysDiag, width: int) -> list[str]:
    if not diag.task_drafts:
        return []

    lines: list[str] = []
    lines.append("")
    lines.append(_section("任务草案  （供 LLM 调查员 / Optimizer 使用）", width))
    lines.append("")

    for i, td in enumerate(diag.task_drafts, 1):
        lines.append(f"  T{i}.  {td.hypothesis}")
        lines.append(f"       验证指标：  {td.verification_metric}")
        lines.append(f"       ID：        {td.id}   推断方式={td.inferred_by}")
        if td.candidate_callsites:
            lines.append("")
            lines.append("       候选调用点（确定性定位，按相关度排序）：")
            for cs_id in td.candidate_callsites[:5]:
                # id format: "<path>:<line>:<col>:<call_name>"
                parts = cs_id.split(":")
                if len(parts) >= 4:
                    cs_path, cs_line, _col, cs_call = parts[0], parts[1], parts[2], parts[3]
                    # Strip .runfiles/ prefix for readability
                    disp_path = cs_path
                    if ".runfiles/" in cs_path:
                        disp_path = cs_path[cs_path.index(".runfiles/") + len(".runfiles/"):]
                    lines.append(f"         · {disp_path}:{cs_line}  .{cs_call}()")
                else:
                    lines.append(f"         · {cs_id}")

        # 搜索入口（rg 命令），明确标注"非定论"
        if td.search_specs:
            lines.append("")
            lines.append("       搜索入口（需人工核实，非定论）：")
            for spec in td.search_specs:
                pat = spec.get("pattern", "")
                kind = spec.get("kind", "rg")
                if kind == "rg":
                    lines.append(f'         {kind} -n "{pat}" <repo>')
                else:
                    lines.append(f"         {kind}: {pat}")

        # Top 证据窗口（来自 profile，no-repo 时最接近"定位"的信息）
        if td.evidence_windows:
            lines.append("")
            lines.append("       Top 证据窗口：")
            for j, w in enumerate(td.evidence_windows[:3], 1):
                start = w.get("start_ms", 0)
                end = w.get("end_ms", 0)
                dur = w.get("duration_ms", end - start)
                dev = w.get("device")
                stream = w.get("stream")
                evt = w.get("event", "?")
                loc_parts = []
                if dev is not None:
                    loc_parts.append(f"device={dev}")
                if stream is not None:
                    loc_parts.append(f"stream={stream}")
                loc_str = "  " + "  ".join(loc_parts) if loc_parts else ""
                lines.append(f"         {j}. {start:.1f}→{end:.1f}ms ({dur:.1f}ms){loc_str}  [{evt}]")
                bk = w.get("before_kernel")
                ak = w.get("after_kernel")
                if bk:
                    lines.append(f"            前序内核：{bk[:60]}")
                if ak:
                    lines.append(f"            后序内核：{ak[:60]}")
                nvtx = w.get("overlap_nvtx", [])
                if nvtx:
                    lines.append(f"            NVTX 重叠：{', '.join(str(n) for n in nvtx[:4])}")
            if len(td.evidence_windows) > 3:
                lines.append(f"         … 还有 {len(td.evidence_windows) - 3} 个窗口（使用 --json 查看）")

        lines.append("")

    return lines


def _render_warnings(diag: NsysDiag, width: int) -> list[str]:
    lines: list[str] = []

    has_warnings = bool(diag.repo_warnings)
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

    if diag.repo_warnings:
        lines.append("")
    for w in diag.repo_warnings:
        lines.append(f"  💡  {w}")

    return lines


# ── 公共入口 ───────────────────────────────────────────────────────────────────

def render_nsys_terminal(diag: NsysDiag, verbose: bool = False) -> str:
    """将 NsysDiag 渲染为结构化终端报告。

    Section 顺序：
      1. 头部
      2. 采集质量
      3. 总览  （任意 / 平均 / 最差 GPU）
      4. 建议排查顺序
      5. 瓶颈分布  （Union % + Inclusive % + ⚠ 标注）
      6. 分域诊断  （主要 finding 合并；SQL finding 作辅助证据）
      7. 代码定位  （--no-repo 时给出各任务的调用点搜索提示）
      8. 任务草案
      9. 指标语义 & 说明
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

    parts.append(_render_capture_quality(diag, width))
    parts.append(_render_summary(diag, width))
    parts.append(_render_investigation_queue(diag, width))
    parts.append(_render_bottlenecks(diag, width))
    parts.append(_render_findings(diag, width, verbose=verbose))
    parts.append(_render_code_localization(diag, width))
    parts.append(_render_task_drafts(diag, width))
    parts.append(_render_warnings(diag, width))

    lines_flat = [ln for block in parts for ln in block]
    lines_flat.append("")
    lines_flat.append(_rule(width, "═"))

    return "\n".join(lines_flat)
