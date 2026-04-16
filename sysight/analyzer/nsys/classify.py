"""nsys/classify.py — T4: Bottleneck classification and Finding generation.

Takes NsysTrace events and produces:
  - BottleneckSummary: statistical overview (interval-union active times)
  - list[NsysFinding]: specific diagnosed problems with severity and next steps

The 8 core findings implemented here:
  1. gpu_idle            — GPU has long idle gaps
  2. gpu_compute_hotspot — Non-NCCL compute dominates GPU time
  3. gpu_comm_hotspot    — NCCL/MPI communication dominates GPU time
  4. comm_not_overlapped — Communication is serialized
  5. gpu_memcpy_hotspot  — Memory transfers are a significant cost
  6. sync_wait           — CPU blocked in sync APIs
  7. host_launch_overhead — Kernel launch latency bottleneck
  8. cpu_hotspot         — CPU sampling hot function

Thresholds follow NVIDIA Expert System conventions:
  - GPU idle > 20% of trace duration → warning
  - GPU idle > 40% → critical
  - Comm not overlapped > 50% of comm time → warning
  - Single bottleneck > 60% of GPU active → critical
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

from .extract import (
    find_gaps,
    intersect_intervals,
    intersect_total,
    total_covered,
    union_intervals,
    union_ns,
)
from .models import (
    BottleneckLabel,
    BottleneckSummary,
    EventStat,
    NsysFinding,
    NsysTrace,
    TimelineEvent,
)

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

_GPU_IDLE_WARN_PCT = 0.20       # 20% of trace
_GPU_IDLE_CRIT_PCT = 0.40       # 40% of trace
_COMM_OVERLAP_WARN_PCT = 0.50   # comm overlap < 50% → warning
_HOTSPOT_CRIT_PCT = 0.60        # category > 60% of GPU active → critical
_HOTSPOT_WARN_PCT = 0.30        # category > 30% of GPU active → warning
_SYNC_WARN_PCT = 0.10           # sync time > 10% of trace → warning
_MIN_GAP_NS = 1_000_000         # 1 ms minimum gap to report
_TOP_N_EVENTS = 20
_TOP_N_GAPS = 5
_TINY_KERNEL_THRESHOLD_US = 10  # kernels < 10µs avg are "tiny"
_TINY_KERNEL_COUNT_THRESHOLD = 1000  # more than 1000 tiny kernels


def classify_bottlenecks(trace: NsysTrace) -> tuple[BottleneckSummary, list[NsysFinding]]:
    """T4: Classify bottlenecks and generate findings.

    Returns (BottleneckSummary, list[NsysFinding]).
    """
    findings: list[NsysFinding] = []

    # ── Separate events by category ─────────────────────────────────────────
    by_cat: dict[str, list[TimelineEvent]] = defaultdict(list)
    for ev in trace.events:
        if not ev.is_sample:
            by_cat[ev.category].append(ev)
        else:
            by_cat["cpu_sample"].append(ev)  # samples kept separate

    gpu_compute = by_cat.get("gpu_compute", [])
    gpu_comm = by_cat.get("gpu_comm", [])
    gpu_memcpy = by_cat.get("gpu_memcpy", [])
    cuda_api = by_cat.get("cuda_api", [])
    sync_evs = by_cat.get("sync_wait", [])
    nvtx_evs = by_cat.get("nvtx", [])
    cpu_samples = by_cat.get("cpu_sample", [])
    osrt_evs = by_cat.get("osrt", [])

    total_ns = trace.duration_ns or 1  # guard against zero

    # ── GPU active time via interval union ──────────────────────────────────
    all_gpu_intervals = [
        (ev.start_ns, ev.start_ns + ev.dur_ns)
        for ev in gpu_compute + gpu_comm + gpu_memcpy
        if ev.dur_ns > 0
    ]
    gpu_merged = union_intervals(all_gpu_intervals)
    gpu_active_ns = total_covered(gpu_merged)
    gpu_idle_ns = max(0, total_ns - gpu_active_ns)

    # ── Per-category interval unions ─────────────────────────────────────────
    compute_intervals = [(ev.start_ns, ev.start_ns + ev.dur_ns) for ev in gpu_compute if ev.dur_ns > 0]
    comm_intervals = [(ev.start_ns, ev.start_ns + ev.dur_ns) for ev in gpu_comm if ev.dur_ns > 0]
    memcpy_intervals = [(ev.start_ns, ev.start_ns + ev.dur_ns) for ev in gpu_memcpy if ev.dur_ns > 0]
    sync_intervals = [(ev.start_ns, ev.start_ns + ev.dur_ns) for ev in sync_evs if ev.dur_ns > 0]

    compute_active_ns = union_ns(compute_intervals)
    comm_active_ns = union_ns(comm_intervals)
    memcpy_active_ns = union_ns(memcpy_intervals)
    sync_active_ns = union_ns(sync_intervals)

    # Inclusive sums (sum of raw durations; may overlap)
    compute_inclusive = sum(ev.dur_ns for ev in gpu_compute)
    comm_inclusive = sum(ev.dur_ns for ev in gpu_comm)
    memcpy_inclusive = sum(ev.dur_ns for ev in gpu_memcpy)
    sync_inclusive = sum(ev.dur_ns for ev in sync_evs)

    # ── Build BottleneckLabels ───────────────────────────────────────────────
    labels: list[BottleneckLabel] = []

    def _pct_active(ns: int) -> float | None:
        return (ns / gpu_active_ns) if gpu_active_ns > 0 else None

    if gpu_compute:
        labels.append(BottleneckLabel(
            category="gpu_compute",
            active_ns=compute_active_ns,
            inclusive_ns=compute_inclusive,
            pct_of_trace=compute_active_ns / total_ns,
            pct_of_gpu_active=_pct_active(compute_active_ns),
            confidence=0.9,
            evidence=[f"{len(gpu_compute)} compute kernels, "
                      f"{compute_active_ns / 1e6:.1f}ms active"],
        ))

    if gpu_comm:
        labels.append(BottleneckLabel(
            category="gpu_comm",
            active_ns=comm_active_ns,
            inclusive_ns=comm_inclusive,
            pct_of_trace=comm_active_ns / total_ns,
            pct_of_gpu_active=_pct_active(comm_active_ns),
            confidence=0.9,
            evidence=[f"{len(gpu_comm)} NCCL/comm kernels, "
                      f"{comm_active_ns / 1e6:.1f}ms active"],
        ))

    if gpu_memcpy:
        labels.append(BottleneckLabel(
            category="gpu_memcpy",
            active_ns=memcpy_active_ns,
            inclusive_ns=memcpy_inclusive,
            pct_of_trace=memcpy_active_ns / total_ns,
            pct_of_gpu_active=_pct_active(memcpy_active_ns),
            confidence=0.85,
            evidence=[f"{len(gpu_memcpy)} memcpy/memset ops, "
                      f"{memcpy_active_ns / 1e6:.1f}ms active"],
        ))

    if gpu_idle_ns > 0:
        labels.append(BottleneckLabel(
            category="gpu_idle",
            active_ns=gpu_idle_ns,
            inclusive_ns=gpu_idle_ns,
            pct_of_trace=gpu_idle_ns / total_ns,
            pct_of_gpu_active=None,
            confidence=0.95,
            evidence=[f"GPU idle {gpu_idle_ns / 1e6:.1f}ms of "
                      f"{total_ns / 1e6:.1f}ms trace"],
        ))

    if sync_evs:
        labels.append(BottleneckLabel(
            category="sync_wait",
            active_ns=sync_active_ns,
            inclusive_ns=sync_inclusive,
            pct_of_trace=sync_active_ns / total_ns,
            pct_of_gpu_active=None,
            confidence=0.85,
            evidence=[f"{len(sync_evs)} sync events, "
                      f"{sync_active_ns / 1e6:.1f}ms"],
        ))

    labels.sort(key=lambda lb: lb.pct_of_trace, reverse=True)

    # ── Top events (cross-category) ──────────────────────────────────────────
    top_events = _compute_top_events(
        gpu_compute + gpu_comm + gpu_memcpy + cuda_api + sync_evs,
        total_ns,
        limit=_TOP_N_EVENTS,
    )

    summary = BottleneckSummary(
        total_ns=total_ns,
        gpu_active_ns=gpu_active_ns,
        gpu_idle_ns=gpu_idle_ns,
        labels=labels,
        top_events=top_events,
    )

    # ── Generate Findings ────────────────────────────────────────────────────
    trace_start = _trace_start(trace)

    # 1. GPU Idle
    _finding_gpu_idle(
        findings, gpu_idle_ns, total_ns, gpu_merged, trace_start, trace_start + total_ns
    )

    # 2. GPU Compute hotspot
    if compute_active_ns > 0 and gpu_active_ns > 0:
        _finding_gpu_compute(findings, compute_active_ns, gpu_active_ns, gpu_compute)

    # 3. GPU Comm hotspot
    if comm_active_ns > 0:
        _finding_gpu_comm(findings, comm_active_ns, gpu_active_ns, total_ns, gpu_comm)

    # 4. Comm not overlapped
    if comm_active_ns > 0 and compute_active_ns > 0:
        _finding_comm_overlap(
            findings, comm_intervals, compute_intervals, comm_active_ns
        )

    # 5. GPU Memcpy hotspot
    if memcpy_active_ns > 0:
        _finding_gpu_memcpy(findings, memcpy_active_ns, gpu_active_ns, total_ns, gpu_memcpy)

    # 6. Sync wait
    if sync_active_ns > 0 or _detect_sync_in_runtime(cuda_api):
        _finding_sync_wait(findings, sync_active_ns, total_ns, sync_evs, cuda_api)

    # 7. Host launch overhead
    if cuda_api and gpu_compute:
        _finding_host_launch(findings, cuda_api, gpu_compute, total_ns)

    # 8. CPU hotspot
    if cpu_samples:
        _finding_cpu_hotspot(findings, cpu_samples)

    # 9. Many tiny kernels (bonus finding)
    if gpu_compute:
        _finding_tiny_kernels(findings, gpu_compute)

    findings.sort(key=lambda f: (
        {"critical": 0, "warning": 1, "info": 2}.get(f.severity, 3),
        -f.confidence,
    ))

    return summary, findings


# ── Finding generators ────────────────────────────────────────────────────────

def _finding_gpu_idle(
    findings: list[NsysFinding],
    gpu_idle_ns: int,
    total_ns: int,
    gpu_merged: list[tuple[int, int]],
    trace_start: int,
    trace_end: int,
) -> None:
    idle_pct = gpu_idle_ns / total_ns
    if idle_pct < 0.05:
        return

    severity = "critical" if idle_pct >= _GPU_IDLE_CRIT_PCT else "warning"
    gaps = find_gaps(gpu_merged, trace_start, trace_end, min_gap_ns=_MIN_GAP_NS)
    top_gaps = sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)[:_TOP_N_GAPS]

    gap_strs = [
        f"gap at {g[0] / 1e6:.1f}ms–{g[1] / 1e6:.1f}ms ({(g[1] - g[0]) / 1e6:.1f}ms)"
        for g in top_gaps
    ]

    findings.append(NsysFinding(
        category="gpu_idle",
        severity=severity,
        confidence=0.95,
        title=f"GPU idle {idle_pct * 100:.1f}% of trace ({gpu_idle_ns / 1e6:.1f}ms)",
        description=(
            f"The GPU was idle for {gpu_idle_ns / 1e6:.1f}ms out of "
            f"{total_ns / 1e6:.1f}ms total trace time ({idle_pct * 100:.1f}%). "
            "This may indicate CPU-GPU synchronization stalls, a slow data pipeline, "
            "or excessive kernel launch overhead between steps."
        ),
        time_range_ns=(trace_start, trace_start + total_ns),
        evidence=[f"GPU active {(1 - idle_pct) * 100:.1f}%"] + gap_strs,
        next_step=(
            "Profile CPU activity during GPU idle periods. Check for "
            "cudaDeviceSynchronize, data loading bottlenecks, or Python GIL. "
            "Use NVTX ranges to correlate idle gaps with Python functions."
        ),
    ))


def _finding_gpu_compute(
    findings: list[NsysFinding],
    compute_active_ns: int,
    gpu_active_ns: int,
    gpu_compute: list[TimelineEvent],
) -> None:
    pct = compute_active_ns / gpu_active_ns
    severity = "critical" if pct >= _HOTSPOT_CRIT_PCT else "warning"
    top = _top_kernels_by_time(gpu_compute, n=5)

    findings.append(NsysFinding(
        category="gpu_compute_hotspot",
        severity=severity,
        confidence=0.85,
        title=f"GPU compute dominates {pct * 100:.1f}% of GPU time",
        description=(
            f"Non-NCCL GPU compute kernels occupy {pct * 100:.1f}% of GPU active time "
            f"({compute_active_ns / 1e6:.1f}ms). The profile is compute-bound."
        ),
        evidence=[f"Top kernel: {k}" for k in top],
        related_hotspots=top,
        next_step=(
            "Identify the dominant kernel(s) and optimize them. "
            "Consider using Nsight Compute for kernel-level analysis (SM utilization, "
            "memory bandwidth, register pressure)."
        ),
    ))


def _finding_gpu_comm(
    findings: list[NsysFinding],
    comm_active_ns: int,
    gpu_active_ns: int,
    total_ns: int,
    gpu_comm: list[TimelineEvent],
) -> None:
    pct_of_active = (comm_active_ns / gpu_active_ns) if gpu_active_ns > 0 else 0
    pct_of_trace = comm_active_ns / total_ns

    if pct_of_active < 0.15 and pct_of_trace < 0.10:
        return

    severity = "critical" if pct_of_active >= _HOTSPOT_CRIT_PCT else "warning"
    top = _top_kernels_by_time(gpu_comm, n=5)

    findings.append(NsysFinding(
        category="gpu_comm_hotspot",
        severity=severity,
        confidence=0.85,
        title=f"GPU communication dominates {pct_of_active * 100:.1f}% of GPU time",
        description=(
            f"NCCL/MPI communication kernels occupy {pct_of_active * 100:.1f}% of "
            f"GPU active time ({comm_active_ns / 1e6:.1f}ms). "
            "Communication is a significant cost in this profile."
        ),
        evidence=[
            f"Comm active {comm_active_ns / 1e6:.1f}ms",
            f"Comm/trace {pct_of_trace * 100:.1f}%",
        ] + [f"Top comm: {k}" for k in top],
        related_hotspots=top,
        next_step=(
            "Investigate whether compute-communication overlap is possible (tensor parallelism, "
            "async all-reduce). Check collective sizes and whether NCCL is bandwidth-limited."
        ),
    ))


def _finding_comm_overlap(
    findings: list[NsysFinding],
    comm_intervals: list[tuple[int, int]],
    compute_intervals: list[tuple[int, int]],
    comm_active_ns: int,
) -> None:
    overlap_ns = intersect_total(comm_intervals, compute_intervals)
    overlap_pct = overlap_ns / comm_active_ns if comm_active_ns > 0 else 1.0
    exposed_ns = comm_active_ns - overlap_ns

    if overlap_pct >= 0.80:
        # Well overlapped; info only
        findings.append(NsysFinding(
            category="comm_not_overlapped",
            severity="info",
            confidence=0.8,
            title=f"Communication well overlapped ({overlap_pct * 100:.1f}% of NCCL time hidden)",
            description=(
                f"Compute and NCCL communication overlap {overlap_pct * 100:.1f}% of NCCL time. "
                f"Only {exposed_ns / 1e6:.1f}ms of communication is exposed (serialized)."
            ),
            evidence=[
                f"Overlap {overlap_ns / 1e6:.1f}ms",
                f"Exposed comm {exposed_ns / 1e6:.1f}ms",
            ],
            next_step="Communication overlap is good. No action needed here.",
        ))
        return

    severity = "critical" if overlap_pct < 0.20 else "warning"
    findings.append(NsysFinding(
        category="comm_not_overlapped",
        severity=severity,
        confidence=0.85,
        title=f"Communication poorly overlapped: only {overlap_pct * 100:.1f}% hidden",
        description=(
            f"Only {overlap_pct * 100:.1f}% of NCCL communication time overlaps with compute. "
            f"{exposed_ns / 1e6:.1f}ms of communication is exposed (serialized), "
            "adding directly to the critical path."
        ),
        evidence=[
            f"NCCL active {comm_active_ns / 1e6:.1f}ms",
            f"Overlap {overlap_ns / 1e6:.1f}ms ({overlap_pct * 100:.1f}%)",
            f"Exposed {exposed_ns / 1e6:.1f}ms",
        ],
        next_step=(
            "Enable compute-communication overlap: use async all-reduce with backward pass, "
            "enable ZeRO-1/2 bucketed all-reduce, or switch to tensor/sequence parallelism. "
            "For PyTorch DDP, check gradient_as_bucket_view and overlap_grad_reduction."
        ),
    ))


def _finding_gpu_memcpy(
    findings: list[NsysFinding],
    memcpy_active_ns: int,
    gpu_active_ns: int,
    total_ns: int,
    gpu_memcpy: list[TimelineEvent],
) -> None:
    pct_of_trace = memcpy_active_ns / total_ns
    if pct_of_trace < 0.05:
        return

    pct_of_active = (memcpy_active_ns / gpu_active_ns) if gpu_active_ns > 0 else 0
    severity = "critical" if pct_of_active >= _HOTSPOT_CRIT_PCT else "warning"

    # Count direction breakdown
    htod = sum(1 for ev in gpu_memcpy if "htod" in ev.name.lower() or ev.extra.get("copy_kind") == 1)
    dtoh = sum(1 for ev in gpu_memcpy if "dtoh" in ev.name.lower() or ev.extra.get("copy_kind") == 2)
    total_bytes = sum(int(ev.extra.get("size_bytes", 0)) for ev in gpu_memcpy)

    evidence = [
        f"Memcpy {memcpy_active_ns / 1e6:.1f}ms ({pct_of_trace * 100:.1f}% of trace)",
        f"{len(gpu_memcpy)} ops: {htod} HtoD, {dtoh} DtoH",
    ]
    if total_bytes > 0:
        throughput_gbs = total_bytes / memcpy_active_ns  # GB/s (ns → s implicit)
        evidence.append(f"Total {total_bytes / 1e9:.2f}GB, ~{throughput_gbs:.1f}GB/s")

    findings.append(NsysFinding(
        category="gpu_memcpy_hotspot",
        severity=severity,
        confidence=0.85,
        title=f"Memory transfers consume {pct_of_trace * 100:.1f}% of trace",
        description=(
            f"GPU memory copies/sets occupy {memcpy_active_ns / 1e6:.1f}ms "
            f"({pct_of_trace * 100:.1f}% of trace). "
            "This may indicate excessive host-device transfers or non-async memcpy patterns."
        ),
        evidence=evidence,
        next_step=(
            "Check for synchronous memcpy (cudaMemcpy without Async). "
            "Pin host memory (cudaHostAlloc/cudaMallocHost) for higher HtoD bandwidth. "
            "Prefetch data asynchronously using cudaMemcpyAsync with separate streams. "
            "Minimize redundant CPU→GPU transfers by keeping tensors on device."
        ),
    ))


def _detect_sync_in_runtime(cuda_api: list[TimelineEvent]) -> bool:
    """Return True if any sync API calls are in the runtime event list."""
    for ev in cuda_api:
        if ev.name in _SYNC_API_NAMES:
            return True
    return False


_SYNC_API_NAMES = frozenset({
    "cudaDeviceSynchronize", "cudaStreamSynchronize",
    "cudaEventSynchronize", "cudaEventQuery",
    "cuStreamSynchronize", "cuDeviceSynchronize",
    "cudaMemcpy", "cudaMemcpyToSymbol", "cudaMemcpyFromSymbol",
    "cudaMemset",
})


def _finding_sync_wait(
    findings: list[NsysFinding],
    sync_active_ns: int,
    total_ns: int,
    sync_evs: list[TimelineEvent],
    cuda_api: list[TimelineEvent],
) -> None:
    # Also count sync APIs from runtime table
    sync_api_calls = [ev for ev in cuda_api if ev.name in _SYNC_API_NAMES]
    sync_api_ns = sum(ev.dur_ns for ev in sync_api_calls)
    total_sync_ns = sync_active_ns + sync_api_ns
    pct = total_sync_ns / total_ns if total_ns > 0 else 0

    if pct < 0.03 and not sync_api_calls:
        return

    severity = "warning" if pct >= _SYNC_WARN_PCT else "info"

    # Top sync APIs by time
    api_counts: Counter[str] = Counter()
    api_times: dict[str, int] = {}
    for ev in sync_api_calls:
        api_counts[ev.name] += 1
        api_times[ev.name] = api_times.get(ev.name, 0) + ev.dur_ns
    top_apis = sorted(api_times.items(), key=lambda x: -x[1])[:5]

    evidence = [f"Sync overhead {total_sync_ns / 1e6:.1f}ms ({pct * 100:.1f}% of trace)"]
    for name, ns in top_apis:
        evidence.append(f"{name}: {ns / 1e6:.1f}ms × {api_counts[name]}")

    related = [name for name, _ in top_apis]

    findings.append(NsysFinding(
        category="sync_wait",
        severity=severity,
        confidence=0.85,
        title=f"Synchronization overhead {total_sync_ns / 1e6:.1f}ms ({pct * 100:.1f}% of trace)",
        description=(
            f"CPU is blocked in CUDA synchronization calls for {total_sync_ns / 1e6:.1f}ms. "
            "Excessive synchronization serializes GPU execution and creates idle bubbles."
        ),
        evidence=evidence,
        related_events=related,
        next_step=(
            "Replace cudaDeviceSynchronize with stream-level sync (cudaStreamSynchronize). "
            "Use cudaMemcpyAsync + events instead of synchronous cudaMemcpy. "
            "Reduce sync frequency by batching work or using CUDA events for fine-grained sync."
        ),
    ))


def _finding_host_launch(
    findings: list[NsysFinding],
    cuda_api: list[TimelineEvent],
    gpu_compute: list[TimelineEvent],
    total_ns: int,
) -> None:
    """Detect host launch overhead: many short kernels with high launch-to-kernel ratio."""
    if not gpu_compute:
        return

    kernel_count = len(gpu_compute)
    avg_kernel_ns = sum(ev.dur_ns for ev in gpu_compute) / kernel_count

    # Count correlation-matched runtime calls
    corr_ids = {ev.correlation_id for ev in gpu_compute if ev.correlation_id is not None}
    launch_calls = [ev for ev in cuda_api if ev.correlation_id in corr_ids]

    if not launch_calls:
        return

    avg_launch_ns = sum(ev.dur_ns for ev in launch_calls) / len(launch_calls)
    launch_overhead_ns = sum(ev.dur_ns for ev in launch_calls)
    overhead_pct = launch_overhead_ns / total_ns

    # Only report if avg kernel is short relative to launch latency
    if avg_kernel_ns > 100_000 or overhead_pct < 0.05:
        return

    severity = "warning" if overhead_pct < 0.20 else "critical"

    findings.append(NsysFinding(
        category="host_launch_overhead",
        severity=severity,
        confidence=0.75,
        title=(
            f"Host kernel launch overhead {overhead_pct * 100:.1f}% of trace "
            f"({kernel_count} kernels, avg {avg_kernel_ns / 1e3:.1f}µs)"
        ),
        description=(
            f"There are {kernel_count} GPU kernels with avg duration {avg_kernel_ns / 1e3:.1f}µs, "
            f"but avg launch latency is {avg_launch_ns / 1e3:.1f}µs. "
            "Launch overhead is a significant fraction of kernel execution time."
        ),
        evidence=[
            f"{kernel_count} kernels, avg {avg_kernel_ns / 1e3:.1f}µs each",
            f"Avg launch latency {avg_launch_ns / 1e3:.1f}µs",
            f"Total launch overhead {launch_overhead_ns / 1e6:.1f}ms ({overhead_pct * 100:.1f}%)",
        ],
        next_step=(
            "Fuse small kernels into larger ones using CUDA kernel fusion or torch.compile. "
            "Use CUDA graphs to replay kernel sequences without re-incurring launch latency. "
            "Reduce Python overhead by minimizing tensor operations in the hot loop."
        ),
    ))


def _finding_cpu_hotspot(
    findings: list[NsysFinding],
    cpu_samples: list[TimelineEvent],
) -> None:
    """Find hot CPU functions from sampling data."""
    if not cpu_samples:
        return

    # Count samples per function
    counter: Counter[str] = Counter(ev.name for ev in cpu_samples)
    total = len(cpu_samples)
    top = counter.most_common(10)

    # Filter out generic/uninformative symbols
    _SKIP = {"cpu_sample", "unknown", "", "__GI___libc_start_main",
              "start_thread", "clone", "futex_wait"}
    top_filtered = [(name, cnt) for name, cnt in top if name not in _SKIP]

    if not top_filtered:
        return

    top_name, top_count = top_filtered[0]
    top_pct = top_count / total

    if top_pct < 0.05:
        return

    severity = "warning" if top_pct >= 0.20 else "info"

    evidence = [
        f"{name}: {cnt} samples ({cnt / total * 100:.1f}%)"
        for name, cnt in top_filtered[:5]
    ]

    findings.append(NsysFinding(
        category="cpu_hotspot",
        severity=severity,
        confidence=0.70,
        title=f"CPU hotspot: {top_name!r} ({top_pct * 100:.1f}% of samples)",
        description=(
            f"CPU profiling shows {top_name!r} accounts for "
            f"{top_pct * 100:.1f}% of CPU samples. "
            "This may be contributing to GPU idle time."
        ),
        evidence=evidence,
        related_hotspots=[name for name, _ in top_filtered[:5]],
        next_step=(
            "Investigate the hot CPU function. For Python processes, this may indicate "
            "GIL contention, DataLoader bottlenecks, or excessive Python overhead. "
            "Use cProfile or py-spy for deeper CPU profiling."
        ),
    ))


def _finding_tiny_kernels(
    findings: list[NsysFinding],
    gpu_compute: list[TimelineEvent],
) -> None:
    """Detect many-tiny-kernels pattern."""
    if len(gpu_compute) < _TINY_KERNEL_COUNT_THRESHOLD:
        return

    avg_ns = sum(ev.dur_ns for ev in gpu_compute) / len(gpu_compute)
    if avg_ns >= _TINY_KERNEL_THRESHOLD_US * 1000:
        return

    tiny = [ev for ev in gpu_compute if ev.dur_ns < _TINY_KERNEL_THRESHOLD_US * 1000]
    tiny_pct = len(tiny) / len(gpu_compute)

    # Top tiny kernel names
    counter: Counter[str] = Counter(ev.name for ev in tiny)
    top = counter.most_common(5)

    findings.append(NsysFinding(
        category="many_tiny_kernels",
        severity="warning",
        confidence=0.80,
        title=(
            f"{len(gpu_compute)} kernels, {tiny_pct * 100:.0f}% under {_TINY_KERNEL_THRESHOLD_US}µs "
            f"(avg {avg_ns / 1e3:.1f}µs)"
        ),
        description=(
            f"{len(tiny)} of {len(gpu_compute)} kernels ({tiny_pct * 100:.0f}%) are shorter than "
            f"{_TINY_KERNEL_THRESHOLD_US}µs. Kernel launch overhead may dominate."
        ),
        evidence=[
            f"Total kernels: {len(gpu_compute)}, tiny: {len(tiny)}",
            f"Avg kernel duration: {avg_ns / 1e3:.1f}µs",
        ] + [f"{name}: {cnt}" for name, cnt in top],
        related_hotspots=[name for name, _ in top],
        next_step=(
            "Use CUDA graphs or torch.compile to eliminate per-kernel launch overhead. "
            "Fuse elementwise operations. Increase batch size if possible to amortize launch cost."
        ),
    ))


# ── Statistical helpers ───────────────────────────────────────────────────────

def _compute_top_events(
    events: list[TimelineEvent],
    total_ns: int,
    limit: int = 20,
) -> list[EventStat]:
    """Aggregate events by name and return top-N by total time."""
    by_name: dict[str, list[int]] = defaultdict(list)
    by_cat: dict[str, str] = {}
    for ev in events:
        if ev.dur_ns > 0:
            by_name[ev.name].append(ev.dur_ns)
            by_cat[ev.name] = ev.category

    stats = []
    for name, durs in by_name.items():
        total = sum(durs)
        stats.append(EventStat(
            name=name,
            category=by_cat[name],
            count=len(durs),
            total_ns=total,
            max_ns=max(durs),
            avg_ns=total / len(durs),
            inclusive_pct=total / total_ns if total_ns > 0 else 0.0,
        ))

    stats.sort(key=lambda s: -s.total_ns)
    return stats[:limit]


def _top_kernels_by_time(events: list[TimelineEvent], n: int = 5) -> list[str]:
    """Return top-N event names by total duration."""
    by_name: dict[str, int] = defaultdict(int)
    for ev in events:
        by_name[ev.name] += ev.dur_ns
    return [k for k, _ in sorted(by_name.items(), key=lambda x: -x[1])[:n]]


def _trace_start(trace: NsysTrace) -> int:
    """Find the earliest event start in the trace."""
    starts = [ev.start_ns for ev in trace.events if not ev.is_sample and ev.dur_ns > 0]
    return min(starts) if starts else 0
