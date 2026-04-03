"""Report assembly and formatting for the analysis MVP."""

from __future__ import annotations

from datetime import datetime

from sysight.analysis.code_location import format_code_locations, locate_code_for_gaps, locate_code_for_windows
from sysight.analysis.iterations import detect_iterations, iteration_summary
from sysight.analysis.nvtx import nvtx_kernel_map, nvtx_layer_breakdown
from sysight.analysis.queries import (
    build_evidence_report as _build_evidence_report,
    detect_idle_gaps,
    h2d_distribution,
    kernel_grid_block_summary,
    kernel_launch_overhead,
    large_h2d_transfers,
    match_root_causes,
    memory_bandwidth_summary,
    memory_transfer_summary,
    nccl_anomalies,
    nccl_breakdown,
    overlap_analysis,
    pageable_memcpy_summary,
    sync_api_summary,
    tensor_core_summary,
    top_kernel_summary,
)
from sysight.analysis.summary import auto_commentary, format_summary as _format_gpu_summary, gpu_summary
from sysight.profile import Profile

_COPY_KIND_NAMES = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}


def format_info(prof: Profile, profile_path: str) -> str:
    """Format profile metadata."""
    lines = [f"Profile: {profile_path}"]
    if prof.schema.version:
        lines.append(f"  Nsight version (heuristic): {prof.schema.version}")
    lines.append(f"  GPUs: {prof.meta.devices}")
    lines.append(f"  Kernels: {prof.meta.kernel_count}  |  NVTX: {prof.meta.nvtx_count}")
    lines.append(f"  Time: {prof.meta.time_range[0] / 1e9:.3f}s - {prof.meta.time_range[1] / 1e9:.3f}s")
    lines.append("")
    for device in prof.meta.devices:
        info = prof.meta.gpu_info.get(device)
        if not info:
            continue
        lines.append(
            f"  GPU {device}: {info.name} | PCI={info.pci_bus} | SMs={info.sm_count} | "
            f"Mem={info.memory_bytes / 1e9:.0f}GB | Kernels={info.kernel_count} | Streams={info.streams}"
        )
    return "\n".join(lines)


def format_summary(prof: Profile, device: int | None = None, trim: tuple[int, int] | None = None) -> str:
    """Format one or all GPU summaries."""
    devices = [device] if device is not None else prof.meta.devices
    chunks = []
    for dev in devices:
        summary = gpu_summary(prof, dev, trim)
        chunks.append(_format_gpu_summary(summary))
        chunks.append("")
        chunks.append(auto_commentary(summary))
        chunks.append("")
    return "\n".join(chunks).rstrip()


def run_analysis(
    prof: Profile, device: int | None = None, trim: tuple[int, int] | None = None
) -> dict:
    """Run the lightweight analysis pipeline."""
    summaries = {dev: gpu_summary(prof, dev, trim) for dev in prof.meta.devices}
    target_device = device if device is not None else (prof.meta.devices[0] if prof.meta.devices else 0)
    target_summary = summaries[target_device]
    target_range = trim or prof.meta.time_range

    data = {
        "profile": prof,
        "trim": trim,
        "target_device": target_device,
        "target_range": target_range,
        "fleet_summaries": summaries,
        "target_summary": target_summary,
        "top_kernels_detail": top_kernel_summary(prof, target_device, trim),
        "idle_gaps": detect_idle_gaps(prof, target_device, trim),
        "memory_transfers": memory_transfer_summary(prof, target_device, trim),
        "memory_bandwidth": memory_bandwidth_summary(prof, target_device, trim),
        "h2d_distribution": h2d_distribution(prof, target_device, trim),
        "large_h2d": large_h2d_transfers(prof, target_device, trim),
        "nccl_breakdown": nccl_breakdown(prof, target_device, trim),
        "nccl_anomalies": nccl_anomalies(prof, target_device, trim),
        "launch_overhead": kernel_launch_overhead(prof, target_device, trim),
        "overlap": overlap_analysis(prof, target_device, trim),
        "sync_summary": sync_api_summary(prof, target_device, trim),
        "pageable_memcpy": pageable_memcpy_summary(prof, target_device, trim),
    }
    data["iterations"] = detect_iterations(prof, target_device, trim)
    data["iterations_summary"] = iteration_summary(data["iterations"])
    data["nvtx_regions"] = nvtx_layer_breakdown(prof, target_device, trim, limit=20)
    data["nvtx_kernel_map"] = nvtx_kernel_map(prof, target_device, trim, limit=30)
    data["tensor_core"] = tensor_core_summary(prof, target_device, trim)
    data["kernel_grid_block"] = kernel_grid_block_summary(prof, target_device, trim, limit=15)
    data["root_causes"] = match_root_causes(data)
    data["code_locations"] = _attach_issue_contexts(prof, data["root_causes"])
    if not data["code_locations"]:
        data["code_locations"] = locate_code_for_gaps(prof, target_device, trim)
    return data


def build_evidence_report(data: dict):
    """Build an evidence report from analysis output."""
    return _build_evidence_report(data)


def _attach_issue_contexts(prof: Profile, findings: list[dict]) -> list[dict]:
    contexts: list[dict] = []
    for finding in findings:
        windows = finding.get("windows") or []
        if not windows:
            finding["contexts"] = []
            continue
        finding_contexts = locate_code_for_windows(prof, windows)
        finding["contexts"] = finding_contexts
        contexts.extend(finding_contexts)
    return contexts


def _sorted_findings(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=_finding_sort_key)


def _fleet_imbalance_summary(data: dict) -> str | None:
    summaries = data.get("fleet_summaries") or {}
    if len(summaries) < 2:
        return None

    util_rows = [
        (device, summary["timing"]["utilization_pct"], summary["timing"]["idle_ms"])
        for device, summary in summaries.items()
    ]
    nccl_rows = [
        (device, summary["nccl_ms"])
        for device, summary in summaries.items()
    ]

    min_util = min(util_rows, key=lambda row: row[1])
    max_util = max(util_rows, key=lambda row: row[1])
    min_nccl = min(nccl_rows, key=lambda row: row[1])
    max_nccl = max(nccl_rows, key=lambda row: row[1])

    util_spread = max_util[1] - min_util[1]
    nccl_spread = max_nccl[1] - min_nccl[1]
    if util_spread < 10 and nccl_spread < 150:
        return None

    return (
        f"多卡表现存在差异：GPU {min_util[0]} 利用率 {min_util[1]:.1f}% / idle {min_util[2]:.1f}ms，"
        f"GPU {max_nccl[0]} NCCL 时间 {max_nccl[1]:.1f}ms；建议优先排查 rank 间负载或同步不均衡。"
    )


def _conclusion_items(data: dict, findings: list[dict]) -> list[str]:
    items = [auto_commentary(data["target_summary"])]
    workflow = data.get("workflow_route") or {}
    if workflow.get("mode"):
        items.append(f"当前 agent 工作流模式为 `{workflow['mode']}`：{workflow.get('summary', '')}")
    if findings:
        top = findings[0]
        items.append(f"首要问题模式是“{top['pattern']}”，当前证据为：{top['evidence']}")
    imbalance = _fleet_imbalance_summary(data)
    if imbalance:
        items.append(imbalance)

    requested_profile = data.get("requested_profile_path")
    resolved_profile = data.get("resolved_profile_path")
    if requested_profile and resolved_profile and data.get("input_was_converted"):
        items.append(f"输入文件已先转换为 SQLite，再执行完整分析：`{resolved_profile}`。")
    return items[:4]


def format_analysis_report(data: dict) -> str:
    """Format the analysis as a conversation-style terminal summary."""
    findings = _sorted_findings(data.get("root_causes", []))
    workflow = data.get("workflow_route") or {}
    lines = ["结论"]
    for item in _conclusion_items(data, findings):
        lines.append(f"- {item}")

    if workflow:
        lines.extend(["", "工作流上下文"])
        for item in _workflow_context_items(workflow):
            lines.append(f"- {item}")

    lines.extend(["", "问题"])
    if findings:
        for finding in findings[:5]:
            severity_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(finding["severity"], "")
            lines.append(f"- {severity_icon} [{finding['severity']}] {finding['pattern']}: {finding['evidence']}")
            loc_hint = _action_short_location(data, finding)
            if loc_hint:
                lines.append(f"  定位：{loc_hint}")
    else:
        lines.append("- 未匹配到明确的问题模式。")

    lines.extend(["", "下一步行动建议"])
    for index, action in enumerate(_action_items(data, findings), start=1):
        lines.append(f"{index}. {action}")

    requested_profile = data.get("requested_profile_path")
    resolved_profile = data.get("resolved_profile_path") or getattr(data.get("profile"), "path", "")
    markdown_path = data.get("markdown_path")
    findings_path = data.get("findings_path")

    lines.append("")
    if requested_profile and data.get("input_was_converted"):
        lines.append(f"输入文件: {requested_profile}")
        lines.append(f"分析 SQLite: {resolved_profile}")
    elif resolved_profile:
        lines.append(f"分析 SQLite: {resolved_profile}")
    if markdown_path:
        lines.append(f"完整报告: {markdown_path}")
    if findings_path:
        lines.append(f"Findings JSON: {findings_path}")
    return "\n".join(lines)


def format_analysis_markdown(data: dict, profile_path: str | None = None) -> str:
    """Format the analysis result as a Markdown report."""
    requested_profile = profile_path or data.get("requested_profile_path") or getattr(data.get("profile"), "path", "")
    resolved_profile = data.get("resolved_profile_path") or getattr(data.get("profile"), "path", "")
    target = data["target_device"]
    target_summary = data["target_summary"]
    timing = target_summary.get("timing", {})
    trim = data.get("trim")
    findings = _sorted_findings(data.get("root_causes", []))
    trim_text = (
        f"{trim[0] / 1e9:.3f}s - {trim[1] / 1e9:.3f}s"
        if trim
        else f"{data['target_range'][0] / 1e9:.3f}s - {data['target_range'][1] / 1e9:.3f}s"
    )

    workflow = data.get("workflow_route") or {}

    num_gpus = len(data.get("fleet_summaries", {}))
    scope_note = (
        f"包含 {num_gpus} 张 GPU（GPU {', '.join(str(d) for d in sorted(data.get('fleet_summaries', {})))})；"
        f"问题分析针对目标 GPU {target}，全局概览示全色卡情况。"
        if num_gpus > 1
        else f"单卡模式（GPU {target}）。"
    )

    lines = [
        "# Sysight 分析报告",
        "",
        f"- 生成时间: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- 输入文件: `{requested_profile}`",
        f"- 分析 SQLite: `{resolved_profile}`",
        f"- 目标 GPU: `{target}`",
        f"- 时间窗口: `{trim_text}`",
        f"- 分析范围: {scope_note}",
        "",
        "## 1. 结论",
        "",
        "### 1.1 摘要结论",
        "",
    ]
    for item in _conclusion_items(data, findings):
        lines.append(f"- {item}")

    if workflow:
        lines.extend(["", "### 1.2 工作流上下文", ""])
        for item in _workflow_context_items(workflow):
            lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "### 1.3 全局概览",
            "",
        ]
    )

    fleet_rows = []
    for device in sorted(data["fleet_summaries"]):
        summary = data["fleet_summaries"][device]
        fleet_rows.append(
            [
                str(device),
                f"{summary['timing']['utilization_pct']:.1f}%",
                f"{summary['timing']['idle_ms']:.1f}",
                f"{summary['nccl_ms']:.1f}",
                summary["top_kernels"][0]["name"] if summary["top_kernels"] else "n/a",
            ]
        )
    lines.extend(
        _markdown_table(
            ["GPU", "Util", "Idle(ms)", "NCCL(ms)", "Top Kernel"],
            fleet_rows,
        )
    )
    lines.extend(_fleet_analysis_commentary(data))

    lines.extend(
        [
            "",
            "### 1.4 目标 GPU 摘要",
            "",
            f"- 设备: `{target_summary['hardware']['name']}` / PCI `{target_summary['hardware']['pci_bus']}`",
            f"- 时间跨度: `{timing.get('span_ms', 0):.1f}ms`，Kernel 时间 `{timing.get('compute_ms', 0):.1f}ms`，Idle `{timing.get('idle_ms', 0):.1f}ms`",
            f"- 汇总判断: {auto_commentary(target_summary)}",
            "",
            "### 1.5 Kernel 热点",
            "",
        ]
    )
    top_kernel_rows = [
        [
            _truncate_kernel_name(row["kernel_name"]),
            str(row["invocations"]),
            f"{row['total_ms']:.2f}",
            f"{row['avg_ms']:.3f}",
            f"{row['max_ms']:.3f}",
        ]
        for row in data.get("top_kernels_detail", [])[:10]
    ]
    if top_kernel_rows:
        lines.extend(
            _markdown_table(
                ["Kernel", "Count", "Total(ms)", "Avg(ms)", "Max(ms)"],
                top_kernel_rows,
            )
        )
        lines.extend(_kernel_hotspot_commentary(data))
    else:
        lines.append("未发现可汇总的 kernel。")

    lines.extend(["", "### 1.6 NVTX / 代码区域热点", ""])
    nvtx_rows = [
        [
            _truncate_nvtx_path(row.get("nvtx_path") or row.get("nvtx_region") or "(unnamed)", max_parts=3, max_len=60),
            str(row["kernel_count"]),
            f"{row['total_gpu_ms']:.2f}",
            f"{row['compute_ms']:.2f}",
            f"{row['nccl_ms']:.2f}",
            f"{row['nccl_pct']:.1f}%",
        ]
        for row in data.get("nvtx_regions", [])[:10]
    ]
    if nvtx_rows:
        lines.extend(
            _markdown_table(
                ["NVTX Region", "Kernels", "Total(ms)", "Compute(ms)", "NCCL(ms)", "NCCL%"],
                nvtx_rows,
            )
        )
        lines.extend(_nvtx_hotspot_commentary(data))
    else:
        lines.append("未检测到可归因的 NVTX 区域，当前代码定位将主要依赖 runtime + sampled stack。")

    lines.extend(["", "### 1.7 通信、传输与同步", ""])
    if data["memory_transfers"]:
        transfer_rows = [
            [
                _COPY_KIND_NAMES.get(row["copyKind"], row["copyKind"]),
                str(row["count"]),
                f"{row['total_mb']:.1f}",
                f"{row['total_ms']:.1f}",
            ]
            for row in data["memory_transfers"]
        ]
        lines.extend(_markdown_table(["Copy", "Ops", "Total(MB)", "Total(ms)"], transfer_rows))
        pattern = data["h2d_distribution"]["pattern"]
        if pattern:
            lines.append("")
            pattern_zh = _H2D_PATTERN_ZH.get(pattern["type"], pattern["type"])
            lines.append(f"- H2D 模式: **{pattern_zh}**")
    if data["nccl_breakdown"]:
        lines.append("")
        lines.append("NCCL 细分：")
        for row in data["nccl_breakdown"]:
            lines.append(
                f"- `{row['type']}`：{row['count']} 次操作，共 {row['total_ms']:.1f}ms，平均 {row['avg_ms']:.2f}ms"
            )
    overlap = data["overlap"]
    if "error" not in overlap:
        lines.append("")
        lines.append(
            f"- 计算/通信 overlap：纯计算 `{overlap['compute_only_ms']:.1f}ms`，"
            f"纯 NCCL `{overlap['nccl_only_ms']:.1f}ms`，重叠 `{overlap['overlap_ms']:.1f}ms`，"
            f"NCCL 隐藏比例 {overlap['overlap_pct']:.1f}%"
        )
    if data["sync_summary"]:
        sync = data["sync_summary"]
        api_str = ", ".join(f"`{a}`" for a in sync["api_names"])
        lines.append(
            f"- 同步 API：{sync['call_count']} 次调用，耗时 {sync['total_ms']:.1f}ms，"
            f"占 GPU kernel 时间 {sync['pct_of_gpu_time']:.1f}%，涉及 {api_str}"
        )

    lines.extend(["", "### 1.8 Iteration 节奏", ""])
    if data["iterations_summary"]:
        summary = data["iterations_summary"]
        lines.append(
            f"- 检测到 `{summary['count']}` 个 iteration，平均 `{summary['avg_ms']:.1f}ms`，"
            f"中位数 `{summary['median_ms']:.1f}ms`，范围 `{summary['min_ms']:.1f}ms` ~ `{summary['max_ms']:.1f}ms`。"
        )
        if summary["heuristic"]:
            lines.append("- 当前使用启发式切分，因为没有命中 iteration NVTX marker。")
        lines.append("")
        iter_rows = [
            [
                str(row["iteration"]),
                row["text"] or "-",
                f"{row['duration_ms']:.1f}",
                str(row["kernel_count"]),
                str(row["nccl_count"]),
            ]
            for row in data["iterations"][:10]
        ]
        lines.extend(_markdown_table(["Iter", "Marker", "Duration(ms)", "Kernels", "NCCL"], iter_rows))
    else:
        lines.append("未检测到稳定 iteration；后续可通过 NVTX iteration marker 提高这部分分析质量。")

    # ── Section 1.9: Tensor Core + Grid/Block ────────────────────────────────
    lines.extend(["", "### 1.9 Tensor Core 利用率 & Kernel 维度", ""])
    tc = data.get("tensor_core") or {}
    if tc.get("total_count", 0) > 0:
        tc_pct = tc.get("tc_time_pct", 0.0)
        non_tc_pct = round(100 - tc_pct, 1)
        lines += [
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| 总 Kernel 时间 | `{tc['total_ms']:.1f}` ms（{tc['total_count']} 次调用）|",
            f"| Tensor Core 时间 | `{tc['tc_ms']:.1f}` ms（**{tc_pct:.1f}%**）|",
            f"| 非 Tensor Core 时间 | `{tc['non_tc_ms']:.1f}` ms（{non_tc_pct:.1f}%）|",
            f"| TC Kernel 调用次数 | `{tc['tc_kernel_count']}` 次（{tc['tc_count_pct']:.1f}%）|",
            "",
        ]
        if tc.get("top_tc_kernels"):
            lines.append("**主要 Tensor Core Kernels：**")
            lines.append("")
            lines.extend(_markdown_table(
                ["Kernel", "调用次数", "总时(ms)", "占比"],
                [
                    [
                        _truncate_kernel_name(r["name"]),
                        str(r["count"]),
                        f"{r['total_ms']:.1f}",
                        f"{r['pct']:.1f}%",
                    ]
                    for r in tc["top_tc_kernels"]
                ],
            ))
        lines.extend(_tensor_core_commentary(tc))
    else:
        lines.append("未能获取 Tensor Core 数据（kernel 数量为 0 或未初始化）。")

    lines.extend(["", "**Kernel Grid/Block 维度（Top 15，按总时间排序）：**", ""])
    gb_rows = data.get("kernel_grid_block") or []
    if gb_rows:
        lines.extend(_markdown_table(
            ["Kernel", "次数", "总时(ms)", "Grid(xyz)", "Block(xyz)", "Thr/Blk", "Reg", "sShMem(B)", "dShMem(B)"],
            [
                [
                    _truncate_kernel_name(row.get("kernel_name") or ""),
                    str(row["invocations"]),
                    f"{row['total_ms']:.2f}",
                    _fmt_grid_dim(row.get("avg_grid_x"), row.get("avg_grid_y"), row.get("avg_grid_z")),
                    _fmt_grid_dim(row.get("avg_block_x"), row.get("avg_block_y"), row.get("avg_block_z")),
                    f"{int(row.get('threads_per_block') or 0)}" + (" ⚠" if row.get("low_occupancy_hint") else ""),
                    str(int(row.get("registers_per_thread") or 0)),
                    str(int(row.get("static_smem_bytes") or 0)),
                    str(int(row.get("dynamic_smem_bytes") or 0)),
                ]
                for row in gb_rows
            ],
        ))
        lines.extend(_grid_block_commentary(gb_rows))
    else:
        lines.append("未获取到 Kernel Grid/Block 数据。")

    lines.extend(["", "## 2. 问题", ""])
    for index, finding in enumerate(findings, start=1):
        severity_label = {"critical": "🔴 严重", "warning": "🟡 警告", "info": "🔵 提示"}.get(
            finding["severity"], finding["severity"]
        )
        lines.append(f"### 2.{index} {severity_label} {finding['pattern']}")
        lines.append("")
        lines.append(f"**证据**：{finding['evidence']}")
        lines.append("")
        locations = _issue_location_candidates(data, finding)
        if locations:
            lines.append("ℹ️ **代码定位**：")
            lines.append("")
            for location in locations:
                # Each location block already contains newlines from _format_window_location
                for loc_line in location.split("\n"):
                    lines.append(loc_line)
                lines.append("")
        lines.append(f"> **建议**：{finding['recommendation']}")
        lines.append("")

    extra_index = len(findings) + 1
    lines.extend([f"### 2.{extra_index} 关键代码定位窗口", ""])
    if data["code_locations"]:
        for window in data["code_locations"][:3]:
            for loc_line in _format_window_location(window).split("\n"):
                lines.append(loc_line)
            lines.append("")
    else:
        lines.append("未生成额外代码定位窗口。")

    if data["nvtx_kernel_map"]:
        extra_index += 1
        lines.extend(["", f"### 2.{extra_index} NVTX 到 Kernel 示例映射", ""])
        for row in data["nvtx_kernel_map"][:10]:
            label = _truncate_nvtx_path(row["nvtx_path"] or row["nvtx_text"] or "(unnamed)")
            kernel = _truncate_kernel_name(row["kernel_name"])
            lines.append(
                f"- `{label}` → `{kernel}` "
                f"({row['start_ms']:.3f}ms - {row['end_ms']:.3f}ms, stream {row['stream']})"
            )

    lines.extend(["", "## 3. 下一步行动建议", ""])
    for index, action in enumerate(_action_items(data, findings), start=1):
        lines.append(f"{index}. {action}")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_H2D_PATTERN_ZH = {
    "init_heavy": "集中在启动阶段（前 2 秒内占大多数）",
    "spread_out": "持续分散在整个 profile 中，疑似热路径上的 H2D",
    "spike": "存在明显脉冲峰值，某些 step 单次拷贝量异常高",
    "unknown": "分布模式未知",
}


def _truncate_kernel_name(name: str, max_len: int = 55) -> str:
    """Shorten a C++ mangled kernel name to a readable length."""
    if not name or len(name) <= max_len:
        return name
    # Try to keep a meaningful prefix before template arguments
    for stop in ("<", "(", " "):
        idx = name.find(stop)
        if 0 < idx <= max_len:
            return name[:idx] + "..."
    return name[:max_len - 3] + "..."


def _fmt_grid_dim(x, y, z) -> str:
    """Format 3D grid/block dims as compact string, dropping trailing 1s."""
    dims = [int(x or 1), int(y or 1), int(z or 1)]
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()
    return "×".join(str(d) for d in dims)


def _tensor_core_commentary(tc: dict) -> list[str]:
    """Generate Markdown commentary for tensor core utilization."""
    lines = [""]
    tc_pct = tc.get("tc_time_pct", 0.0)
    non_tc_ms = tc.get("non_tc_ms", 0.0)
    total_ms = tc.get("total_ms", 0.0)

    if tc_pct >= 60:
        lines.append(
            f"> 💡 **Tensor Core 占比较高（{tc_pct:.1f}%）**："
            f"计算密集型 Kernel（矩阵乘、注意力机制、卷积）占主导，模型计算效率总体良好。"
            f"若整体 MFU 仍偏低，可重点排查 batch size / sequence length 是否足够填满 TC。"
        )
    elif tc_pct >= 20:
        non_pct = round(100 - tc_pct, 1)
        lines.append(
            f"> ⚠️ **Tensor Core 占比中等（{tc_pct:.1f}%）**："
            f"有 {non_pct:.1f}% 的 Kernel 时间（约 {non_tc_ms:.1f}ms）未被识别为 TC 操作，"
            f"可能存在较多逐元素运算（elementwise）、归约、Softmax 等非 TC Kernel，"
            f"或部分 TC Kernel 名称未被当前模式覆盖。"
        )
    else:
        lines.append(
            f"> 🔴 **Tensor Core 占比较低（{tc_pct:.1f}%）**："
            f"在总 Kernel 时间 {total_ms:.1f}ms 中，TC Kernel 仅 {tc.get('tc_ms', 0.0):.1f}ms，"
            f"可能存在过多非矩阵运算（如逐元素激活、归约、数据搬运），"
            f"或 model 存在大量小 batch / 短序列，导致 TC 利用不充分。"
        )
    lines.append(
        "> 注：此处基于 kernel 名称模式匹配（cuBLAS HMMA、FlashAttention、cutlass 等），"
        "若使用自定义 Triton/CUDA Kernel 可能存在漏判。"
    )
    return lines


def _grid_block_commentary(rows: list[dict]) -> list[str]:
    """Generate Markdown commentary for grid/block dimension table."""
    if not rows:
        return []

    lines = [""]
    low_occ = [r for r in rows if r.get("low_occupancy_hint")]
    high_reg = [r for r in rows if (r.get("registers_per_thread") or 0) >= 64]
    high_dyn_smem = [r for r in rows if (r.get("dynamic_smem_bytes") or 0) >= 32 * 1024]

    notes = []
    if low_occ:
        names = "、".join(_truncate_kernel_name(r.get("kernel_name") or "", 30) for r in low_occ[:3])
        notes.append(
            f"⚠️ **低 occupancy 风险**：{names} 等 kernel 每块线程数 ≤ 64，"
            f"可能导致 SM warp 利用率不足。建议调大 block size 或检查其设计意图。"
        )
    if high_reg:
        names = "、".join(_truncate_kernel_name(r.get("kernel_name") or "", 30) for r in high_reg[:2])
        notes.append(
            f"🔵 **高寄存器使用**：{names} 每线程寄存器 ≥ 64，"
            f"这会限制每 SM 并发 warp 数量（register pressure），若 occupancy 偏低可考虑 `--maxrregcount` 调优。"
        )
    if high_dyn_smem:
        names = "、".join(_truncate_kernel_name(r.get("kernel_name") or "", 30) for r in high_dyn_smem[:2])
        notes.append(
            f"🔵 **大动态共享内存**：{names} 动态共享内存 ≥ 32KB，"
            f"可能是 FlashAttention 或分块矩阵乘的正常使用，也需确认未因 bank conflict 损耗性能。"
        )

    if notes:
        for note in notes:
            lines.append(f"> {note}")
            lines.append("")
    else:
        lines.append(
            "> Grid/Block 配置未发现明显异常。所有热点 kernel 的每块线程数均 > 64，"
            "寄存器和共享内存使用在合理范围内。"
        )
    return lines


def _truncate_nvtx_path(path: str, max_parts: int = 3, max_len: int = 80) -> str:
    """Show only the last N components of a ' > ' separated NVTX path."""
    if not path:
        return path
    parts = [p.strip() for p in path.split(" > ") if p.strip()]
    if len(parts) > max_parts:
        shown = " > ".join(parts[-max_parts:])
        return "...> " + shown
    result = " > ".join(parts)
    if len(result) > max_len:
        return result[:max_len - 3] + "..."
    return result


def _short_display_path(path: str, max_len: int = 60) -> str:
    """Keep only the last meaningful parts of a long file path."""
    if not path or len(path) <= max_len:
        return path
    # Already a display_path (relative), just truncate from the right
    parts = path.replace("\\", "/").split("/")
    # Walk backwards accumulating parts until max_len
    result_parts: list[str] = []
    acc = 0
    for part in reversed(parts):
        if acc + len(part) + 1 > max_len and result_parts:
            result_parts.insert(0, "...")
            break
        result_parts.insert(0, part)
        acc += len(part) + 1
    return "/".join(result_parts)


def _short_symbol(symbol: str, max_len: int = 60) -> str:
    """Truncate a long C++ symbol for display."""
    if not symbol or len(symbol) <= max_len:
        return symbol
    for stop in ("(", "<", " "):
        idx = symbol.find(stop)
        if 0 < idx <= max_len:
            return symbol[:idx] + "..."
    return symbol[:max_len - 3] + "..."


def _fleet_analysis_commentary(data: dict) -> list[str]:
    """Generate a short multi-card analysis paragraph after the fleet overview table."""
    summaries = data.get("fleet_summaries") or {}
    if len(summaries) < 2:
        return []

    lines = [""]
    util_rows = sorted(
        [(dev, s["timing"]["utilization_pct"], s["timing"]["idle_ms"]) for dev, s in summaries.items()],
        key=lambda r: r[1],
    )
    nccl_rows = sorted(
        [(dev, s["nccl_ms"]) for dev, s in summaries.items()],
        key=lambda r: r[1],
    )
    avg_util = sum(r[1] for r in util_rows) / len(util_rows)

    # Find outlier GPUs (util < avg - 15 or idle > 3x median)
    outliers = [r for r in util_rows if r[1] < avg_util - 15]
    # NCCL distribution
    max_nccl = nccl_rows[-1]
    min_nccl = nccl_rows[0]
    nccl_spread = max_nccl[1] - min_nccl[1]

    observations = []
    if outliers:
        for dev, util, idle in outliers:
            nccl_for_dev = summaries[dev]["nccl_ms"]
            observations.append(
                f"**GPU {dev} 利用率明显偏低（{util:.1f}%，idle {idle:.0f}ms），"
                f"NCCL 仅 {nccl_for_dev:.0f}ms**，与其他卡差异显著——"
                f"可能原因：该 rank 的计算量分配不均、进入了不同代码分支，或数据并行切分偏斜；"
                f"建议单独抓取该 GPU 的 timeline 确认。"
            )
    else:
        observations.append(
            f"各卡利用率相近（均值 {avg_util:.1f}%），未检测到明显的 rank 间负载不均。"
        )

    if nccl_spread > 200 and len(summaries) > 2:
        observations.append(
            f"NCCL 时间分布差异较大（最高 GPU {max_nccl[0]}：{max_nccl[1]:.0f}ms，"
            f"最低 GPU {min_nccl[0]}：{min_nccl[1]:.0f}ms），"
            f"可能存在 AllReduce 慢节点拖慢整体 barrier 的情况，建议对比各卡 NCCL 调用时序。"
        )

    for obs in observations:
        lines.append(f"> {obs}")
    return lines


def _kernel_hotspot_commentary(data: dict) -> list[str]:
    """Generate a short interpretation paragraph after the kernel hotspot table."""
    kernels = data.get("top_kernels_detail") or []
    target_summary = data.get("target_summary") or {}
    if not kernels:
        return []

    total_compute_ms = target_summary.get("timing", {}).get("compute_ms", 0) or 1
    top = kernels[0]
    top_pct = round(100 * top["total_ms"] / total_compute_ms, 1)

    is_nccl_top = "nccl" in (top["kernel_name"] or "").lower()
    nccl_ms = target_summary.get("nccl_ms", 0)
    compute_only_ms = target_summary.get("compute_only_ms", 0)

    lines = [""]
    if is_nccl_top:
        lines.append(
            f"> **分析**：耗时最高的 kernel 是 NCCL 通信算子（占 GPU 时间 {top_pct:.1f}%），"
            f"说明当前 profile 的主要瓶颈**不在计算，而在通信**。"
            f"纯计算 kernel 合计 {compute_only_ms:.0f}ms，纯 NCCL {nccl_ms:.0f}ms，"
            f"两者比值为 {compute_only_ms/max(nccl_ms,1):.2f}，通信远重于计算。"
            f"在通信/计算 overlap 得到改善之前，优化计算 kernel 本身收益有限。"
        )
    else:
        # Check fragmentation: many small kernels vs one dominator
        if len(kernels) >= 3:
            top3_ms = sum(k["total_ms"] for k in kernels[:3])
            top3_pct = round(100 * top3_ms / total_compute_ms, 1)
            lines.append(
                f"> **分析**：top kernel `{_truncate_kernel_name(top['kernel_name'])}` 占 GPU 时间 {top_pct:.1f}%，"
                f"前 3 个 kernel 合计 {top3_pct:.1f}%——"
                + (
                    "热点高度集中，优先针对该 kernel 做 profiling（ncu）。"
                    if top_pct >= 20
                    else "热点较为分散，建议先通过 NVTX 定位最耗时的代码区域，再逐 kernel 排查。"
                )
            )
        else:
            lines.append(
                f"> **分析**：top kernel `{_truncate_kernel_name(top['kernel_name'])}` 占 GPU 时间 {top_pct:.1f}%。"
            )
    return lines


def _nvtx_hotspot_commentary(data: dict) -> list[str]:
    """Generate a short interpretation paragraph after the NVTX region table."""
    nvtx_rows = data.get("nvtx_regions") or []
    if not nvtx_rows:
        return []

    # Detect duplicate-looking NVTX paths (same tail component)
    tail_counter: dict[str, int] = {}
    for row in nvtx_rows[:10]:
        path = row.get("nvtx_path") or row.get("nvtx_region") or ""
        parts = [p.strip() for p in path.split(" > ") if p.strip()]
        tail = parts[-1] if parts else path
        tail_counter[tail] = tail_counter.get(tail, 0) + 1

    repeated_tails = [t for t, cnt in tail_counter.items() if cnt >= 2]
    high_nccl = [r for r in nvtx_rows[:10] if r.get("nccl_pct", 0) >= 80]

    lines = [""]
    observations = []

    if repeated_tails:
        examples = "、".join(f"`{t}`" for t in repeated_tails[:3])
        observations.append(
            f"表中出现多行 {examples} 等同名 NVTX 尾节点——"
            f"这些**不是重复记录**，而是同一类结构在不同 NVTX 嵌套路径下的多次执行（例如多层 Transformer 中每层都有 MultiHeadAttention）；"
            f"每行代表不同的调用上下文，合并来看即为该模块的总耗时。"
        )

    if high_nccl:
        labels = "、".join(
            f"`{_truncate_nvtx_path(r.get('nvtx_path') or r.get('nvtx_region', ''), max_parts=2)}`"
            for r in high_nccl[:2]
        )
        observations.append(
            f"NCCL% 接近 100% 的区域（{labels}）表明该代码路径**几乎全部时间都在等通信**，"
            f"计算内核极少，是通信/计算 overlap 优化的核心目标区域。"
        )

    for obs in observations:
        lines.append(f"> {obs}")
    return lines if len(lines) > 1 else []


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["(empty)"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        safe = [str(cell).replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(safe) + " |")
    return lines


def _finding_sort_key(finding: dict) -> tuple[int, float, str]:
    order = {"critical": 0, "warning": 1, "info": 2}
    return (
        order.get(finding.get("severity", "info"), 9),
        -float(finding.get("score") or 0.0),
        finding.get("pattern", ""),
    )


def _issue_location_candidates(data: dict, finding: dict) -> list[str]:
    pattern = finding["pattern"]
    candidates: list[str] = []

    if finding.get("contexts"):
        candidates.extend(_format_window_location(window, pattern=pattern) for window in finding["contexts"])

    if pattern in {"NCCL Hotspot", "Layer NCCL Hotspot", "Compute-Communication Imbalance"}:
        candidates.extend(_top_nvtx_region_candidates(data, metric="nccl_ms"))
    if pattern in {"Pipeline Imbalance", "Kernel Hotspot"}:
        candidates.extend(_top_nvtx_region_candidates(data, metric="compute_ms"))

    if pattern in {
        "GPU Bubbles (Pipeline Stalls)",
        "Excessive Synchronization",
        "Excessive H2D Transfers",
        "Continuous H2D Transfers",
        "Pageable Memory in Memcpy",
        "Small Kernel Overhead",
        "Iteration Variance",
    }:
        candidates.extend(_matching_window_candidates(data, pattern))

    if not candidates and data.get("code_locations"):
        candidates.append(_format_window_location(data["code_locations"][0], pattern=pattern))

    seen = set()
    deduped = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped[:3]


def _top_nvtx_region_candidates(data: dict, *, metric: str, limit: int = 2) -> list[str]:
    rows = [row for row in data.get("nvtx_regions", []) if row.get(metric, 0) > 0]
    rows.sort(key=lambda row: (-row.get(metric, 0), -(row.get("total_gpu_ms", 0))))
    candidates = []
    for row in rows[:limit]:
        label = _truncate_nvtx_path(row.get("nvtx_path") or row.get("nvtx_region") or "(unnamed)")
        lines = [
            f"**NVTX 热点**",
            f"  - 区域: `{label}`",
            f"  - 总耗时 {row['total_gpu_ms']:.1f}ms，其中 compute {row['compute_ms']:.1f}ms，NCCL {row['nccl_ms']:.1f}ms",
        ]
        candidates.append("\n".join(lines))
    return candidates


def _matching_window_candidates(data: dict, pattern: str) -> list[str]:
    matched = []
    for window in data.get("code_locations", []):
        attr = (window.get("attribution") or {}).get("category", "")
        thread_text = " ".join(
            " ".join(
                [
                    frame.get("symbol", ""),
                    frame.get("module", ""),
                    frame.get("reason", ""),
                ]
            )
            for thread in window.get("threads", [])
            for frame in thread.get("frames", [])
        ).lower()
        if pattern in {"GPU Bubbles (Pipeline Stalls)", "Excessive Synchronization"}:
            if attr == "synchronization":
                matched.append(_format_window_location(window, pattern=pattern))
        elif pattern in {"Excessive H2D Transfers", "Continuous H2D Transfers", "Pageable Memory in Memcpy"}:
            if attr == "memory_transfer" or "copy" in thread_text:
                matched.append(_format_window_location(window, pattern=pattern))
        elif pattern == "Small Kernel Overhead":
            if attr == "kernel_launch" or any(key in thread_text for key in ("launch", "dynamo", "triton")):
                matched.append(_format_window_location(window, pattern=pattern))
        elif pattern == "Iteration Variance":
            matched.append(_format_window_location(window, pattern=pattern))
    return matched[:2]


def _format_window_location(window: dict, *, pattern: str | None = None) -> str:
    """Format a suspicious window as a structured, readable multi-line Markdown block."""
    attr = window.get("attribution") or {}
    thread = _select_representative_thread(window, pattern)
    frame = _select_representative_frame(thread.get("frames", []), pattern)
    stack = _select_representative_stack(thread, pattern)
    nvtx_raw = _format_nvtx_labels(window.get("nvtx_ranges", []) or thread.get("nvtx_ranges", []))
    nvtx = _truncate_nvtx_path(nvtx_raw) if nvtx_raw else ""

    # --- headline --------------------------------------------------------
    category_label = {
        "synchronization": "同步阻塞",
        "memory_transfer": "内存拷贝",
        "kernel_launch": "Kernel 启动开销",
        "communication": "通信",
        "iteration": "慢 iteration",
        "cpu_stall": "CPU stall",
    }.get(attr.get("category", ""), attr.get("category", "") or "未分类")

    dur_ms = window.get("gap_ms") or window.get("duration_ms")
    dur_str = f"{dur_ms:.1f}ms" if dur_ms is not None else ""
    stream_str = f"stream {window['stream']}" if window.get("stream") is not None else ""
    headline_parts = [p for p in [category_label, dur_str, stream_str] if p]
    headline = "、".join(headline_parts)

    lines: list[str] = []
    lines.append(f"**{headline}**")

    # --- where in the timeline -------------------------------------------
    if window.get("before_kernel"):
        bk = _truncate_kernel_name(window["before_kernel"])
        lines.append(f"  - 前序 kernel: `{bk}`")
    if window.get("after_kernel"):
        ak = _truncate_kernel_name(window["after_kernel"])
        lines.append(f"  - 后续 kernel: `{ak}`")

    # --- NVTX context ----------------------------------------------------
    if nvtx:
        lines.append(f"  - NVTX 上下文: `{nvtx}`")

    # --- Python code location -------------------------------------------
    if window.get("python_locations"):
        loc = window["python_locations"][0]
        short_path = _short_display_path(loc.get("display_path") or loc.get("path") or "")
        func = loc.get("function") or ""
        if func:
            lines.append(f"  - 函数定位: `{short_path}` ➜ `{func}`")
        else:
            lines.append(f"  - 文件定位: `{short_path}`")
        # Second candidate if different file
        if len(window["python_locations"]) > 1:
            loc2 = window["python_locations"][1]
            p2 = _short_display_path(loc2.get("display_path") or loc2.get("path") or "")
            f2 = loc2.get("function") or ""
            if p2 != short_path:
                lines.append(f"  - 备选定位: `{p2}`{(' ➜ `' + f2 + '`') if f2 else ''}")
    elif window.get("project_files"):
        p = _short_display_path(window["project_files"][0]["display_path"])
        lines.append(f"  - 文件候选: `{p}`")

    # --- call stack -----------------------------------------------------
    if stack.get("preview"):
        source_label = {"cuda_callchain": "CUDA callchain", "osrt_callchain": "OS callchain", "sampling": "采样栈"}.get(
            stack.get("source", ""), stack.get("source", "调用栈")
        )
        lines.append(f"  - {source_label}: `{stack['preview']}`")
    elif frame.get("symbol"):
        sym = _short_symbol(frame["symbol"])
        lines.append(f"  - 采样帧: `{sym}`")

    return "\n".join(lines)


def _select_representative_thread(window: dict, pattern: str | None) -> dict:
    threads = window.get("threads", [])
    if not threads:
        return {}
    scored = sorted(
        threads,
        key=lambda thread: (
            _best_stack_score(thread.get("stacks", []), pattern),
            _best_frame_score(thread.get("frames", []), pattern),
            thread.get("total_runtime_ms", 0),
        ),
        reverse=True,
    )
    return scored[0]


def _select_representative_frame(frames: list[dict], pattern: str | None) -> dict:
    if not frames:
        return {}
    if not pattern:
        return frames[0]

    keywords = {
        "Excessive H2D Transfers": ("copy", "_to_copy", "memcpy", "pinned"),
        "Continuous H2D Transfers": ("copy", "_to_copy", "memcpy", "pinned"),
        "Pageable Memory in Memcpy": ("copy", "_to_copy", "memcpy", "pinned"),
        "Small Kernel Overhead": ("dynamo", "launch", "triton", "compile"),
        "GPU Bubbles (Pipeline Stalls)": ("synchronize", "copy", "thpfunction"),
        "Excessive Synchronization": ("synchronize", "event", "wait", "thpfunction"),
        "Iteration Variance": ("thpfunction", "dynamo", "copy", "launch"),
    }.get(pattern, ())

    for frame in frames:
        text = " ".join(
            [
                frame.get("symbol", ""),
                frame.get("module", ""),
                frame.get("reason", ""),
            ]
        ).lower()
        if any(keyword in text for keyword in keywords):
            return frame
    return frames[0]


def _best_frame_score(frames: list[dict], pattern: str | None) -> int:
    if not frames:
        return 0
    selected = _select_representative_frame(frames, pattern)
    if not selected:
        return 0
    text = " ".join(
        [
            selected.get("symbol", ""),
            selected.get("module", ""),
            selected.get("reason", ""),
        ]
    ).lower()
    if any(keyword in text for keyword in ("copy", "_to_copy", "memcpy", "launch", "dynamo", "triton", "synchronize", "thpfunction")):
        return 2
    return 1


def _select_representative_stack(thread: dict, pattern: str | None) -> dict:
    stacks = thread.get("stacks", [])
    if not stacks:
        return {}
    if not pattern:
        return stacks[0]

    keywords = {
        "Excessive H2D Transfers": ("copy", "_to_copy", "memcpy", "pinned"),
        "Continuous H2D Transfers": ("copy", "_to_copy", "memcpy", "pinned"),
        "Pageable Memory in Memcpy": ("copy", "_to_copy", "memcpy", "pinned"),
        "Small Kernel Overhead": ("dynamo", "launch", "triton", "compile"),
        "GPU Bubbles (Pipeline Stalls)": ("synchronize", "backward", "autograd", "copy"),
        "Excessive Synchronization": ("synchronize", "wait", "event", "autograd"),
        "Iteration Variance": ("backward", "forward", "launch", "copy"),
    }.get(pattern, ())

    for stack in stacks:
        text = " ".join(
            [
                stack.get("preview", ""),
                " ".join(frame.get("symbol", "") for frame in stack.get("frames", [])),
            ]
        ).lower()
        if any(keyword in text for keyword in keywords):
            return stack
    return stacks[0]


def _best_stack_score(stacks: list[dict], pattern: str | None) -> int:
    if not stacks:
        return 0
    selected = _select_representative_stack({"stacks": stacks}, pattern)
    if not selected:
        return 0
    text = " ".join(
        [
            selected.get("preview", ""),
            " ".join(frame.get("symbol", "") for frame in selected.get("frames", [])),
        ]
    ).lower()
    if any(keyword in text for keyword in ("copy", "_to_copy", "memcpy", "launch", "dynamo", "triton", "synchronize", "backward", "autograd")):
        return 3
    return 1


def _format_nvtx_labels(rows: list[dict]) -> str:
    labels = []
    generic = {"Holding GIL", "Waiting for GIL", "(unnamed)"}
    for row in rows:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        if text in generic:
            continue
        labels.append(text)
    if not labels:
        labels = [(row.get("text") or "").strip() for row in rows if (row.get("text") or "").strip()]
    return ", ".join(labels[:2])


def _format_python_location(row: dict) -> str:
    label = row.get("display_path") or row.get("path") or ""
    if row.get("function"):
        return f"{label} :: {row['function']}"
    return label


def _workflow_context_items(workflow: dict) -> list[str]:
    items: list[str] = []
    mode = workflow.get("mode")
    if mode:
        items.append(f"模式: `{mode}`")
    if workflow.get("workspace_root"):
        items.append(f"Workspace: `{workflow['workspace_root']}`")
    if workflow.get("program_path"):
        items.append(f"Program 合同: `{workflow['program_path']}`")

    contract = workflow.get("program_contract") or {}
    if contract.get("task"):
        items.append(f"任务: {contract['task']}")
    if contract.get("framework"):
        items.append(f"框架/技术栈: {contract['framework']}")
    if contract.get("entry"):
        items.append(f"启动方式: {contract['entry']}")
    if contract.get("performance_goal"):
        items.append(f"性能目标: {contract['performance_goal']}")
    if contract.get("success_criteria"):
        items.append(f"成功标准: {contract['success_criteria']}")
    if contract.get("important_paths"):
        items.append(
            "关键路径: " + ", ".join(f"`{path}`" for path in contract["important_paths"][:4])
        )
    if contract.get("missing_sections"):
        items.append(
            "program.md 待补字段: " + ", ".join(f"`{field}`" for field in contract["missing_sections"])
        )
    for warning in workflow.get("warnings", [])[:2]:
        items.append(f"提示: {warning}")
    return items[:8]


def _action_short_location(data: dict, finding: dict) -> str:
    """Return a one-line, human-readable location hint for action items."""
    contexts = finding.get("contexts") or []
    if not contexts:
        contexts = data.get("code_locations", [])
    if not contexts:
        return ""
    window = contexts[0]
    loc = (window.get("python_locations") or [])
    if loc:
        short_path = _short_display_path(loc[0].get("display_path") or loc[0].get("path") or "")
        func = loc[0].get("function") or ""
        if func:
            return f"`{short_path}` ➜ `{func}`"
        return f"`{short_path}`"
    files = window.get("project_files") or []
    if files:
        return f"`{_short_display_path(files[0]['display_path'])}`"
    nvtx_raw = _format_nvtx_labels(window.get("nvtx_ranges", []))
    if nvtx_raw:
        return f"NVTX `{_truncate_nvtx_path(nvtx_raw)}`"
    return ""


def _action_items(data: dict, findings: list[dict]) -> list[str]:
    actions = []
    for finding in findings:
        loc_hint = _action_short_location(data, finding)
        action = f"**{finding['pattern']}**：{finding['recommendation']}"
        if loc_hint:
            action += f"\n   优先排查 {loc_hint}"
        actions.append(action)

    if data.get("iterations_summary") and data["iterations_summary"]["heuristic"]:
        actions.append(
            "给训练/推理主循环补充稳定的 NVTX iteration marker，这样 iteration timing、layer breakdown 和问题定位会明显更稳定。"
        )

    workflow = data.get("workflow_route") or {}
    if workflow.get("mode") == "profile-only":
        actions.append("如果后续需要把问题更稳定地映射到项目入口、模块和优化目标，补充 workspace/program.md 后再重跑一次分析。")
    elif workflow.get("mode") == "workspace-aware":
        contract = workflow.get("program_contract") or {}
        if contract.get("entry"):
            actions.append(f"结合 program.md 中的启动方式 `{contract['entry']}`，优先核对热点是否落在同一条主执行路径上。")
        if contract.get("performance_goal"):
            actions.append(f"后续优化阶段建议围绕 program.md 中的目标“{contract['performance_goal']}”来定义 before/after 对比口径。")

    deduped = []
    seen = set()
    for action in actions:
        if action in seen:
            continue
        seen.add(action)
        deduped.append(action)
    return deduped[:6]
