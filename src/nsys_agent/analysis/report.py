"""Report assembly and formatting for the analysis MVP."""

from __future__ import annotations

from datetime import datetime

from nsys_agent.analysis.code_location import format_code_locations, locate_code_for_gaps
from nsys_agent.analysis.iterations import detect_iterations, iteration_summary
from nsys_agent.analysis.nvtx import nvtx_kernel_map, nvtx_layer_breakdown
from nsys_agent.analysis.queries import (
    build_evidence_report as _build_evidence_report,
    detect_idle_gaps,
    h2d_distribution,
    kernel_launch_overhead,
    match_root_causes,
    memory_bandwidth_summary,
    memory_transfer_summary,
    nccl_anomalies,
    nccl_breakdown,
    overlap_analysis,
    pageable_memcpy_summary,
    sync_api_summary,
    top_kernel_summary,
)
from nsys_agent.analysis.summary import auto_commentary, format_summary as _format_gpu_summary, gpu_summary
from nsys_agent.profile import Profile

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
        "nccl_breakdown": nccl_breakdown(prof, target_device, trim),
        "nccl_anomalies": nccl_anomalies(prof, target_device, trim),
        "launch_overhead": kernel_launch_overhead(prof, target_device, trim),
        "overlap": overlap_analysis(prof, target_device, trim),
        "sync_summary": sync_api_summary(prof),
        "pageable_memcpy": pageable_memcpy_summary(prof, target_device),
    }
    data["iterations"] = detect_iterations(prof, target_device, trim)
    data["iterations_summary"] = iteration_summary(data["iterations"])
    data["nvtx_regions"] = nvtx_layer_breakdown(prof, target_device, trim, limit=20)
    data["nvtx_kernel_map"] = nvtx_kernel_map(prof, target_device, trim, limit=30)
    data["code_locations"] = locate_code_for_gaps(prof, target_device, trim)
    data["root_causes"] = match_root_causes(data)
    return data


def build_evidence_report(data: dict):
    """Build an evidence report from analysis output."""
    return _build_evidence_report(data)


def format_analysis_report(data: dict) -> str:
    """Format the lightweight analysis as terminal text."""
    lines = ["=== nsys-agent Analysis Report ===", ""]
    lines.append("Fleet Overview")
    for device in sorted(data["fleet_summaries"]):
        summary = data["fleet_summaries"][device]
        timing = summary["timing"]
        top = summary["top_kernels"][0]["name"] if summary["top_kernels"] else "n/a"
        lines.append(
            f"- GPU {device}: util={timing['utilization_pct']}%, idle={timing['idle_ms']:.1f}ms, "
            f"nccl={summary['nccl_ms']:.1f}ms, top={top}"
        )

    lines.extend(["", f"Target GPU: {data['target_device']}", ""])
    lines.append(_format_gpu_summary(data["target_summary"]))
    lines.extend(["", "Summary", auto_commentary(data["target_summary"]), ""])

    gap_summary = data["idle_gaps"]["summary"]
    if gap_summary:
        lines.append("Idle Gaps")
        lines.append(
            f"- {gap_summary['gap_count']} gaps, {gap_summary['total_idle_ms']:.1f}ms idle "
            f"({gap_summary['pct_of_profile']}% of profile)"
        )
        lines.append(
            f"- Distribution: {gap_summary['gaps_1_5ms']} x 1-5ms, "
            f"{gap_summary['gaps_5_50ms']} x 5-50ms, {gap_summary['gaps_gt50ms']} x >50ms"
        )
        for row in data["idle_gaps"]["rows"][:5]:
            attr = row.get("attribution") or {}
            lines.append(
                f"- Stream {row['streamId']}: gap={row['gap_ns'] / 1e6:.2f}ms after {row['before_kernel']} "
                f"[{attr.get('category', 'unclassified')}]"
            )
        lines.append("")

    if data["memory_transfers"]:
        lines.append("Memory Transfers")
        for row in data["memory_transfers"]:
            lines.append(
                f"- {_COPY_KIND_NAMES.get(row['copyKind'], row['copyKind'])}: {row['count']} ops, "
                f"{row['total_mb']:.1f}MB, {row['total_ms']:.1f}ms"
            )
        pattern = data["h2d_distribution"]["pattern"]
        if pattern:
            lines.append(f"- H2D pattern: {pattern['type']} - {pattern['detail']}")
        lines.append("")

    if data["nccl_breakdown"]:
        lines.append("NCCL Breakdown")
        for row in data["nccl_breakdown"]:
            lines.append(
                f"- {row['type']}: {row['count']} ops, {row['total_ms']:.1f}ms total, "
                f"{row['avg_ms']:.2f}ms avg, {row['pct']:.1f}% of NCCL time"
            )
        lines.append("")

    if data["nccl_anomalies"]:
        lines.append("NCCL Anomalies")
        for row in data["nccl_anomalies"][:5]:
            lines.append(
                f"- {row['op_type']}: {row['dur_ms']:.2f}ms on stream {row['streamId']} "
                f"({row['ratio_to_avg']}x avg)"
            )
        lines.append("")

    overlap = data["overlap"]
    if "error" not in overlap:
        lines.append("Compute / Communication Overlap")
        lines.append(
            f"- Compute only: {overlap['compute_only_ms']:.1f}ms | NCCL only: {overlap['nccl_only_ms']:.1f}ms | "
            f"Overlap: {overlap['overlap_ms']:.1f}ms ({overlap['overlap_pct']}% of NCCL hidden)"
        )
        lines.append(
            f"- Idle: {overlap['idle_ms']:.1f}ms across {overlap['total_ms']:.1f}ms total span"
        )
        lines.append("")

    if data["sync_summary"]:
        sync = data["sync_summary"]
        lines.append("Synchronization")
        lines.append(
            f"- {sync['call_count']} sync calls totalling {sync['total_ms']:.1f}ms "
            f"({sync['pct_of_gpu_time']:.1f}% of GPU time)"
        )
        lines.append(f"- APIs: {', '.join(sync['api_names'])}")
        lines.append("")

    if data["iterations_summary"]:
        summary = data["iterations_summary"]
        lines.append("Iteration Timing")
        lines.append(
            f"- {summary['count']} iterations | avg={summary['avg_ms']:.1f}ms | "
            f"median={summary['median_ms']:.1f}ms | min={summary['min_ms']:.1f}ms | max={summary['max_ms']:.1f}ms"
        )
        if summary["heuristic"]:
            lines.append("- Detection: heuristic fallback (no matching NVTX iteration marker)")
        for row in data["iterations"][:5]:
            lines.append(
                f"- Iter {row['iteration']}: {row['duration_ms']:.1f}ms, "
                f"{row['kernel_count']} kernels, {row['nccl_count']} NCCL"
            )
        lines.append("")

    if data["nvtx_regions"]:
        lines.append("NVTX Region Breakdown")
        for row in data["nvtx_regions"][:8]:
            name = row.get("nvtx_path") or row.get("nvtx_region") or "(unnamed)"
            lines.append(
                f"- {name}: total={row['total_gpu_ms']:.1f}ms, compute={row['compute_ms']:.1f}ms, "
                f"nccl={row['nccl_ms']:.1f}ms ({row['nccl_pct']:.1f}%)"
            )
        lines.append("")

    if data["code_locations"]:
        lines.append(format_code_locations(data["code_locations"]))
        lines.append("")

    lines.append("Root Causes")
    for finding in data["root_causes"]:
        lines.append(f"- [{finding['severity']}] {finding['pattern']}")
        lines.append(f"  Evidence: {finding['evidence']}")
        lines.append(f"  Next step: {finding['recommendation']}")
    return "\n".join(lines)


def format_analysis_markdown(data: dict, profile_path: str | None = None) -> str:
    """Format the analysis result as a Markdown report."""
    profile = profile_path or getattr(data.get("profile"), "path", "")
    target = data["target_device"]
    target_summary = data["target_summary"]
    timing = target_summary.get("timing", {})
    trim = data.get("trim")
    trim_text = (
        f"{trim[0] / 1e9:.3f}s - {trim[1] / 1e9:.3f}s"
        if trim
        else f"{data['target_range'][0] / 1e9:.3f}s - {data['target_range'][1] / 1e9:.3f}s"
    )

    lines = [
        "# nsys-agent 分析报告",
        "",
        f"- 生成时间: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- Profile: `{profile}`",
        f"- 目标 GPU: `{target}`",
        f"- 时间窗口: `{trim_text}`",
        "",
        "## 1. 结果分析",
        "",
        "### 1.1 全局概览",
        "",
    ]

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

    lines.extend(
        [
            "",
            "### 1.2 目标 GPU 摘要",
            "",
            f"- 设备: `{target_summary['hardware']['name']}` / PCI `{target_summary['hardware']['pci_bus']}`",
            f"- 时间跨度: `{timing.get('span_ms', 0):.1f}ms`，Kernel 时间 `{timing.get('compute_ms', 0):.1f}ms`，Idle `{timing.get('idle_ms', 0):.1f}ms`",
            f"- 汇总判断: {auto_commentary(target_summary)}",
            "",
            "### 1.3 Kernel 热点",
            "",
        ]
    )
    top_kernel_rows = [
        [
            row["kernel_name"],
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
    else:
        lines.append("未发现可汇总的 kernel。")

    lines.extend(["", "### 1.4 NVTX / 代码区域热点", ""])
    nvtx_rows = [
        [
            row.get("nvtx_path") or row.get("nvtx_region") or "(unnamed)",
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
    else:
        lines.append("未检测到可归因的 NVTX 区域，当前代码定位将主要依赖 runtime + sampled stack。")

    lines.extend(["", "### 1.5 通信、传输与同步", ""])
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
            lines.append(f"- H2D 模式: `{pattern['type']}`，{pattern['detail']}")
    if data["nccl_breakdown"]:
        lines.append("")
        lines.append("NCCL Breakdown:")
        for row in data["nccl_breakdown"]:
            lines.append(
                f"- `{row['type']}`: {row['count']} ops, {row['total_ms']:.1f}ms total, avg {row['avg_ms']:.2f}ms"
            )
    overlap = data["overlap"]
    if "error" not in overlap:
        lines.append("")
        lines.append(
            f"- Compute/NCCL overlap: compute-only `{overlap['compute_only_ms']:.1f}ms`, "
            f"NCCL-only `{overlap['nccl_only_ms']:.1f}ms`, overlap `{overlap['overlap_ms']:.1f}ms` "
            f"({overlap['overlap_pct']:.1f}% of NCCL hidden)"
        )
    if data["sync_summary"]:
        sync = data["sync_summary"]
        lines.append(
            f"- 同步 API: {sync['call_count']} calls / {sync['total_ms']:.1f}ms "
            f"({sync['pct_of_gpu_time']:.1f}% of GPU kernel time)，API = {', '.join(sync['api_names'])}"
        )

    lines.extend(["", "### 1.6 Iteration 节奏", ""])
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

    lines.extend(["", "## 2. 存在问题", ""])
    findings = sorted(data["root_causes"], key=_finding_sort_key)
    for index, finding in enumerate(findings, start=1):
        lines.append(f"### 2.{index} [{finding['severity']}] {finding['pattern']}")
        lines.append("")
        lines.append(f"- 证据: {finding['evidence']}")
        for location in _issue_location_candidates(data, finding):
            lines.append(f"- 代码定位: {location}")
        lines.append(f"- 建议: {finding['recommendation']}")
        lines.append("")

    lines.extend(["## 3. 下一步行动指南", ""])
    for index, action in enumerate(_action_items(data, findings), start=1):
        lines.append(f"{index}. {action}")

    lines.extend(["", "## 4. 附录", "", "### 4.1 关键代码定位窗口", ""])
    if data["code_locations"]:
        for window in data["code_locations"][:3]:
            lines.append(f"- {_format_window_location(window)}")
    else:
        lines.append("- 未生成额外代码定位窗口。")

    if data["nvtx_kernel_map"]:
        lines.extend(["", "### 4.2 NVTX 到 Kernel 示例映射", ""])
        for row in data["nvtx_kernel_map"][:10]:
            label = row["nvtx_path"] or row["nvtx_text"] or "(unnamed)"
            lines.append(
                f"- `{label}` -> `{row['kernel_name']}` "
                f"({row['start_ms']:.3f}ms - {row['end_ms']:.3f}ms, stream {row['stream']})"
            )

    return "\n".join(lines).rstrip() + "\n"


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


def _finding_sort_key(finding: dict) -> tuple[int, str]:
    order = {"critical": 0, "warning": 1, "info": 2}
    return (order.get(finding.get("severity", "info"), 9), finding.get("pattern", ""))


def _issue_location_candidates(data: dict, finding: dict) -> list[str]:
    pattern = finding["pattern"]
    candidates: list[str] = []

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
        label = row.get("nvtx_path") or row.get("nvtx_region") or "(unnamed)"
        candidates.append(
            f"NVTX `{label}`: total {row['total_gpu_ms']:.1f}ms, compute {row['compute_ms']:.1f}ms, NCCL {row['nccl_ms']:.1f}ms"
        )
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
    attr = window.get("attribution") or {}
    thread = _select_representative_thread(window, pattern)
    frame = _select_representative_frame(thread.get("frames", []), pattern)
    nvtx = _format_nvtx_labels(thread.get("nvtx_ranges", []))
    parts = [
        f"stream {window['stream']}, gap {window['gap_ms']:.2f}ms, after `{window['before_kernel']}`",
    ]
    if attr.get("category"):
        parts.append(f"category `{attr['category']}`")
    if nvtx:
        parts.append(f"NVTX `{nvtx}`")
    if frame.get("symbol"):
        parts.append(f"frame `{frame['symbol']}`")
    return " | ".join(parts)


def _select_representative_thread(window: dict, pattern: str | None) -> dict:
    threads = window.get("threads", [])
    if not threads:
        return {}
    scored = sorted(
        threads,
        key=lambda thread: (_best_frame_score(thread.get("frames", []), pattern), thread.get("total_runtime_ms", 0)),
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


def _format_nvtx_labels(rows: list[dict]) -> str:
    labels = []
    generic = {"Holding GIL", "(unnamed)"}
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


def _action_items(data: dict, findings: list[dict]) -> list[str]:
    actions = []
    for finding in findings:
        locations = _issue_location_candidates(data, finding)
        action = f"{finding['pattern']}: {finding['recommendation']}"
        if locations:
            action += f" 优先查看 {locations[0]}。"
        actions.append(action)

    if data.get("iterations_summary") and data["iterations_summary"]["heuristic"]:
        actions.append(
            "给训练/推理主循环补充稳定的 NVTX iteration marker，这样 iteration timing、layer breakdown 和问题定位会明显更稳定。"
        )

    deduped = []
    seen = set()
    for action in actions:
        if action in seen:
            continue
        seen.add(action)
        deduped.append(action)
    return deduped[:6]
