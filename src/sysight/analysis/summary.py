"""Profile summary helpers."""

from __future__ import annotations

from sysight.profile import Profile


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def gpu_summary(prof: Profile, device: int, trim: tuple[int, int] | None = None) -> dict:
    """Generate a concise summary for a single GPU."""
    info = prof.meta.gpu_info.get(device)
    kernels = prof.kernels(device, trim)
    if not kernels:
        return {"device": device, "error": "no kernels found"}

    from collections import Counter, defaultdict

    starts = [kernel["start"] for kernel in kernels]
    ends = [kernel["end"] for kernel in kernels]
    span_ns = max(ends) - min(starts)
    busy_ns = sum(end - start for start, end in _merge_intervals([(kernel["start"], kernel["end"]) for kernel in kernels]))

    duration_by_name = defaultdict(float)
    count_by_name = Counter()
    stream_kernels = defaultdict(int)
    stream_duration = defaultdict(float)

    for kernel in kernels:
        duration_ms = (kernel["end"] - kernel["start"]) / 1e6
        duration_by_name[kernel["name"]] += duration_ms
        count_by_name[kernel["name"]] += 1
        stream_kernels[kernel["streamId"]] += 1
        stream_duration[kernel["streamId"]] += duration_ms

    top = sorted(duration_by_name.items(), key=lambda item: -item[1])[:10]

    sorted_kernels = sorted(kernels, key=lambda kernel: kernel["start"])
    idle_ns = 0
    max_end = sorted_kernels[0]["end"]
    for kernel in sorted_kernels[1:]:
        if kernel["start"] > max_end:
            idle_ns += kernel["start"] - max_end
        max_end = max(max_end, kernel["end"])

    total_kernel_ns = sum(kernel["end"] - kernel["start"] for kernel in kernels)
    nccl_ms = 0.0
    compute_ms = 0.0
    for name, total_ms in duration_by_name.items():
        if "nccl" in (name or "").lower():
            nccl_ms += total_ms
        else:
            compute_ms += total_ms

    return {
        "device": device,
        "hardware": {
            "name": info.name if info else "unknown",
            "pci_bus": info.pci_bus if info else "",
            "sm_count": info.sm_count if info else 0,
            "memory_gb": round(info.memory_bytes / 1e9, 1) if info else 0,
        },
        "timing": {
            "span_ms": round(span_ns / 1e6, 2),
            "compute_ms": round(busy_ns / 1e6, 2),
            "idle_ms": round(idle_ns / 1e6, 2),
            "utilization_pct": round(100 * busy_ns / span_ns, 1) if span_ns else 0,
        },
        "kernel_count": len(kernels),
        "top_kernels": [
            {
                "name": name,
                "total_ms": round(total_ms, 3),
                "count": count_by_name[name],
                "pct": round(100 * total_ms / (total_kernel_ns / 1e6), 1) if total_kernel_ns else 0,
            }
            for name, total_ms in top
        ],
        "streams": {
            stream: {"kernels": stream_kernels[stream], "total_ms": round(stream_duration[stream], 2)}
            for stream in sorted(stream_kernels)
        },
        "nccl_ms": round(nccl_ms, 3),
        "compute_only_ms": round(compute_ms, 3),
    }


def format_summary(summary: dict) -> str:
    """Format one GPU summary as human-readable text."""
    if "error" in summary:
        return f"GPU {summary['device']}: {summary['error']}"

    hardware = summary["hardware"]
    timing = summary["timing"]
    lines = [
        f"GPU {summary['device']}: {hardware['name']} ({hardware['pci_bus']}) - {hardware['sm_count']} SMs, {hardware['memory_gb']}GB",
        f"  Span: {timing['span_ms']:.1f}ms | Compute: {timing['compute_ms']:.1f}ms | Idle: {timing['idle_ms']:.1f}ms | Util: {timing['utilization_pct']}%",
        f"  Kernels: {summary['kernel_count']}",
        "",
        "  Top kernels:",
    ]
    for kernel in summary["top_kernels"]:
        lines.append(
            f"    {kernel['pct']:5.1f}%  {kernel['total_ms']:8.1f}ms  x{kernel['count']:<4d}  {kernel['name']}"
        )
    lines.append("")
    lines.append("  Streams:")
    for stream, stream_data in summary["streams"].items():
        lines.append(
            f"    Stream {stream}: {stream_data['kernels']} kernels, {stream_data['total_ms']:.1f}ms"
        )
    return "\n".join(lines)


def auto_commentary(summary: dict) -> str:
    """Produce a short narrative for one GPU."""
    if "error" in summary:
        return f"GPU {summary['device']}: {summary['error']}"

    timing = summary["timing"]
    top = summary["top_kernels"]
    parts = [
        f"GPU {summary['device']} ran {summary['kernel_count']} kernels over {timing['span_ms']:.0f}ms with {timing['utilization_pct']}% active timeline coverage."
    ]
    if top:
        parts.append(
            f"Top bottleneck: {top[0]['name']} at {top[0]['pct']}% of compute time ({top[0]['total_ms']:.0f}ms across {top[0]['count']} calls)."
        )
    total_ms = summary["nccl_ms"] + summary["compute_only_ms"]
    if total_ms > 0 and summary["nccl_ms"] > 0:
        nccl_pct = 100 * summary["nccl_ms"] / total_ms
        parts.append(
            f"NCCL accounts for {nccl_pct:.0f}% of kernel time ({summary['nccl_ms']:.0f}ms), compute kernels account for {summary['compute_only_ms']:.0f}ms."
        )
    if timing["idle_ms"] > 10:
        idle_pct = 100 * timing["idle_ms"] / timing["span_ms"] if timing["span_ms"] else 0
        parts.append(f"There are {timing['idle_ms']:.0f}ms of idle gaps ({idle_pct:.0f}% of span).")
    return " ".join(parts)
