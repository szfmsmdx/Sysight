"""nsys/classify.py — 瓶颈分类与 Finding 生成。

核心 Finding（内存事件列表分析）：
  1. gpu_idle             GPU 长时间空闲
  2. gpu_compute_hotspot  非 NCCL 计算占主导
  3. gpu_comm_hotspot     NCCL/MPI 通信占主导
  4. comm_not_overlapped  通信未与计算重叠
  5. gpu_memcpy_hotspot   内存拷贝耗时显著
  6. sync_wait            CPU 阻塞于同步 API
  7. host_launch_overhead 内核启动延迟瓶颈
  8. cpu_hotspot          CPU 采样热点函数
  9. many_tiny_kernels    大量微小内核

深度 SQL 分析（classify_sql.py，移植自 nsys-ai skills）：
  - sql_top_kernels       按总耗时排列的 Top-N GPU 内核
  - sql_gpu_idle_gaps     GPU 流内核间的空闲气泡
  - sql_memory_bandwidth  各方向内存带宽统计
  - sql_nccl_breakdown    NCCL 集合通信按流分解
  - sql_sync_cost         CUPTI 同步表的同步代价
  - sql_nvtx_hotspots     NVTX 注释段热点分布
  - sql_nvtx_layer_breakdown  NVTX region 级 GPU 时间分解
  - sql_root_cause_analysis   已知反模式程序化检测
  - sql_profile_health        Profile 全局健康摘要

对齐 nsys-ai 设计：使用 severity (critical/warning/info) 而非数值 confidence。
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

from .extract import (
    find_gaps,
    intersect_total,
    total_covered,
    union_intervals,
    union_ns,
)
from .models import (
    BottleneckLabel,
    BottleneckSummary,
    DeviceBreakdown,
    EventStat,
    NsysFinding,
    NsysTrace,
    TimelineEvent,
    stable_finding_id,
)

logger = logging.getLogger(__name__)

# ── 阈值 ─────────────────────────────────────────────────────────────────────

_GPU_IDLE_WARN_PCT = 0.20
_GPU_IDLE_CRIT_PCT = 0.40
_HOTSPOT_CRIT_PCT = 0.60
_HOTSPOT_WARN_PCT = 0.30
_SYNC_WARN_PCT = 0.10
_MIN_GAP_NS = 1_000_000
_TOP_N_EVENTS = 20
_TOP_N_GAPS = 5
_TINY_KERNEL_THRESHOLD_US = 10
_TINY_KERNEL_COUNT_THRESHOLD = 1000

_SYNC_API_NAMES = frozenset({
    "cudaDeviceSynchronize", "cudaStreamSynchronize",
    "cudaEventSynchronize", "cudaEventQuery",
    "cuStreamSynchronize", "cuDeviceSynchronize",
    "cudaMemcpy", "cudaMemcpyToSymbol", "cudaMemcpyFromSymbol",
    "cudaMemset",
})


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════

def classify_bottlenecks(
    trace: NsysTrace,
    *,
    include_deep_sql: bool = True,
) -> tuple[BottleneckSummary, list[NsysFinding]]:
    """瓶颈分类，生成 Findings。返回 (BottleneckSummary, list[NsysFinding])。"""
    findings: list[NsysFinding] = []

    by_cat: dict[str, list[TimelineEvent]] = defaultdict(list)
    for ev in trace.events:
        if not ev.is_sample:
            by_cat[ev.category].append(ev)
        else:
            by_cat["cpu_sample"].append(ev)

    gpu_compute = by_cat.get("gpu_compute", [])
    gpu_comm = by_cat.get("gpu_comm", [])
    gpu_memcpy = by_cat.get("gpu_memcpy", [])
    cuda_api = by_cat.get("cuda_api", [])
    sync_evs = by_cat.get("sync_wait", [])
    cpu_samples = by_cat.get("cpu_sample", [])

    total_ns = trace.duration_ns or 1

    all_gpu_intervals = [
        (ev.start_ns, ev.start_ns + ev.dur_ns)
        for ev in gpu_compute + gpu_comm + gpu_memcpy
        if ev.dur_ns > 0
    ]
    gpu_merged = union_intervals(all_gpu_intervals)
    gpu_active_ns = total_covered(gpu_merged)
    gpu_idle_ns = max(0, total_ns - gpu_active_ns)

    compute_intervals = [(ev.start_ns, ev.start_ns + ev.dur_ns) for ev in gpu_compute if ev.dur_ns > 0]
    comm_intervals = [(ev.start_ns, ev.start_ns + ev.dur_ns) for ev in gpu_comm if ev.dur_ns > 0]
    memcpy_intervals = [(ev.start_ns, ev.start_ns + ev.dur_ns) for ev in gpu_memcpy if ev.dur_ns > 0]
    sync_intervals = [(ev.start_ns, ev.start_ns + ev.dur_ns) for ev in sync_evs if ev.dur_ns > 0]

    compute_active_ns = union_ns(compute_intervals)
    comm_active_ns = union_ns(comm_intervals)
    memcpy_active_ns = union_ns(memcpy_intervals)
    sync_active_ns = union_ns(sync_intervals)

    compute_inclusive = sum(ev.dur_ns for ev in gpu_compute)
    comm_inclusive = sum(ev.dur_ns for ev in gpu_comm)
    memcpy_inclusive = sum(ev.dur_ns for ev in gpu_memcpy)
    sync_inclusive = sum(ev.dur_ns for ev in sync_evs)

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
            evidence=[f"{len(gpu_compute)} 个计算内核，{compute_active_ns / 1e6:.1f}ms 活跃"],
        ))

    if gpu_comm:
        labels.append(BottleneckLabel(
            category="gpu_comm",
            active_ns=comm_active_ns,
            inclusive_ns=comm_inclusive,
            pct_of_trace=comm_active_ns / total_ns,
            pct_of_gpu_active=_pct_active(comm_active_ns),
            evidence=[f"{len(gpu_comm)} 个 NCCL/通信内核，{comm_active_ns / 1e6:.1f}ms 活跃"],
        ))

    if gpu_memcpy:
        labels.append(BottleneckLabel(
            category="gpu_memcpy",
            active_ns=memcpy_active_ns,
            inclusive_ns=memcpy_inclusive,
            pct_of_trace=memcpy_active_ns / total_ns,
            pct_of_gpu_active=_pct_active(memcpy_active_ns),
            evidence=[f"{len(gpu_memcpy)} 次 memcpy/memset，{memcpy_active_ns / 1e6:.1f}ms 活跃"],
        ))

    if gpu_idle_ns > 0:
        labels.append(BottleneckLabel(
            category="gpu_idle",
            active_ns=gpu_idle_ns,
            inclusive_ns=gpu_idle_ns,
            pct_of_trace=gpu_idle_ns / total_ns,
            pct_of_gpu_active=None,
            evidence=[f"GPU 空闲 {gpu_idle_ns / 1e6:.1f}ms / 总计 {total_ns / 1e6:.1f}ms"],
        ))

    if sync_evs:
        labels.append(BottleneckLabel(
            category="sync_wait",
            active_ns=sync_active_ns,
            inclusive_ns=sync_inclusive,
            pct_of_trace=sync_active_ns / total_ns,
            pct_of_gpu_active=None,
            evidence=[f"{len(sync_evs)} 个同步事件，{sync_active_ns / 1e6:.1f}ms"],
        ))

    labels.sort(key=lambda lb: lb.pct_of_trace, reverse=True)

    top_events = _compute_top_events(
        gpu_compute + gpu_comm + gpu_memcpy + cuda_api + sync_evs,
        total_ns,
        limit=_TOP_N_EVENTS,
    )

    # Per-device breakdown (non-empty only when multiple device_ids present)
    per_device = _compute_per_device(
        gpu_compute + gpu_comm + gpu_memcpy, total_ns
    )

    summary = BottleneckSummary(
        total_ns=total_ns,
        gpu_active_ns=gpu_active_ns,
        gpu_idle_ns=gpu_idle_ns,
        labels=labels,
        top_events=top_events,
        per_device=per_device,
    )

    trace_start = _trace_start(trace)

    _finding_gpu_idle(findings, gpu_idle_ns, total_ns, gpu_merged, trace_start, trace_start + total_ns)

    if compute_active_ns > 0 and gpu_active_ns > 0:
        _finding_gpu_compute(findings, compute_active_ns, gpu_active_ns, gpu_compute)

    if comm_active_ns > 0:
        _finding_gpu_comm(findings, comm_active_ns, gpu_active_ns, total_ns, gpu_comm)

    if comm_active_ns > 0 and compute_active_ns > 0:
        _finding_comm_overlap(findings, comm_intervals, compute_intervals, comm_active_ns)

    if memcpy_active_ns > 0:
        _finding_gpu_memcpy(findings, memcpy_active_ns, gpu_active_ns, total_ns, gpu_memcpy)

    if sync_active_ns > 0 or _detect_sync_in_runtime(cuda_api):
        _finding_sync_wait(findings, sync_active_ns, total_ns, sync_evs, cuda_api)

    if cuda_api and gpu_compute:
        _finding_host_launch(findings, cuda_api, gpu_compute, total_ns)

    if cpu_samples:
        _finding_cpu_hotspot(findings, cpu_samples)

    if gpu_compute:
        _finding_tiny_kernels(findings, gpu_compute)

    # 深度 SQL 分析（移植自 nsys-ai skills，委托给 classify_sql.py）
    if include_deep_sql and trace.sqlite_path:
        try:
            from .classify_sql import run_deep_sql_analysis
            run_deep_sql_analysis(findings, trace)
        except Exception as e:
            logger.warning("深度 SQL 分析失败（非致命）：%s", e)

    findings.sort(key=lambda f: (
        {"critical": 0, "warning": 1, "info": 2}.get(f.severity, 3),
    ))

    # Assign stable IDs after sorting (content-based, not sort-order-based)
    for f in findings:
        f.stable_id = stable_finding_id(f.category, f.severity, f.time_range_ns, f.device_id)

    return summary, findings


# ═══════════════════════════════════════════════════════════════════════════════
# Finding 生成器
# ═══════════════════════════════════════════════════════════════════════════════

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
        f"空闲段 {g[0]/1e6:.1f}–{g[1]/1e6:.1f}ms（{(g[1]-g[0])/1e6:.1f}ms）"
        for g in top_gaps
    ]
    findings.append(NsysFinding(
        category="gpu_idle",
        severity=severity,
        title=f"GPU 空闲占 trace {idle_pct*100:.1f}%（{gpu_idle_ns/1e6:.1f}ms）",
        description=(
            f"GPU 在总时长 {total_ns/1e6:.1f}ms 中空闲了 {gpu_idle_ns/1e6:.1f}ms"
            f"（{idle_pct*100:.1f}%）。可能原因：CPU-GPU 同步停顿、数据流水线慢、内核启动间隙过大。"
        ),
        time_range_ns=(trace_start, trace_start + total_ns),
        evidence=[f"GPU 活跃 {(1-idle_pct)*100:.1f}%"] + gap_strs,
        next_step=(
            "在 GPU 空闲段期间分析 CPU 活动。检查 cudaDeviceSynchronize、"
            "数据加载瓶颈或 Python GIL。使用 NVTX ranges 关联空闲间隙与 Python 函数。"
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
        title=f"GPU 计算占 GPU 时间 {pct*100:.1f}%",
        description=(
            f"非 NCCL 的 GPU 计算内核占 GPU 活跃时间的 {pct*100:.1f}%"
            f"（{compute_active_ns/1e6:.1f}ms）。该 profile 以计算为主要瓶颈。"
        ),
        evidence=[f"热点内核：{k}" for k in top],
        related_hotspots=top,
        next_step=(
            "识别主导内核并优化。考虑使用 Nsight Compute 进行内核级分析"
            "（SM 利用率、内存带宽、寄存器压力）。"
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
        title=f"GPU 通信占 GPU 时间 {pct_of_active*100:.1f}%",
        description=(
            f"NCCL/MPI 通信内核占 GPU 活跃时间的 {pct_of_active*100:.1f}%"
            f"（{comm_active_ns/1e6:.1f}ms）。通信是该 profile 的显著开销。"
        ),
        evidence=[
            f"通信活跃 {comm_active_ns/1e6:.1f}ms",
            f"通信占 trace {pct_of_trace*100:.1f}%",
        ] + [f"热点通信内核：{k}" for k in top],
        related_hotspots=top,
        next_step=(
            "研究是否可实现计算通信重叠（张量并行、异步 all-reduce）。"
            "检查 collective 操作大小及 NCCL 是否受带宽限制。"
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
        findings.append(NsysFinding(
            category="comm_not_overlapped",
            severity="info",
            title=f"通信重叠良好（{overlap_pct*100:.1f}% NCCL 时间已隐藏）",
            description=(
                f"计算与 NCCL 通信重叠了 {overlap_pct*100:.1f}% 的 NCCL 时间。"
                f"仅 {exposed_ns/1e6:.1f}ms 通信暴露（串行化）。"
            ),
            evidence=[f"重叠 {overlap_ns/1e6:.1f}ms", f"暴露通信 {exposed_ns/1e6:.1f}ms"],
            next_step="通信重叠良好，无需操作。",
        ))
        return
    severity = "critical" if overlap_pct < 0.20 else "warning"
    findings.append(NsysFinding(
        category="comm_not_overlapped",
        severity=severity,
        title=f"通信重叠差：仅 {overlap_pct*100:.1f}% 被隐藏",
        description=(
            f"只有 {overlap_pct*100:.1f}% 的 NCCL 通信时间与计算重叠。"
            f"{exposed_ns/1e6:.1f}ms 通信暴露（串行化），直接增加关键路径。"
        ),
        evidence=[
            f"NCCL 活跃 {comm_active_ns/1e6:.1f}ms",
            f"重叠 {overlap_ns/1e6:.1f}ms（{overlap_pct*100:.1f}%）",
            f"暴露 {exposed_ns/1e6:.1f}ms",
        ],
        next_step=(
            "启用计算通信重叠：将异步 all-reduce 与反向传播重叠，"
            "启用 ZeRO-1/2 分桶 all-reduce，或切换到张量/序列并行。"
            "对于 PyTorch DDP，检查 gradient_as_bucket_view 和 overlap_grad_reduction。"
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
    htod = sum(1 for ev in gpu_memcpy if "htod" in ev.name.lower() or ev.extra.get("copy_kind") == 1)
    dtoh = sum(1 for ev in gpu_memcpy if "dtoh" in ev.name.lower() or ev.extra.get("copy_kind") == 2)
    total_bytes = sum(int(ev.extra.get("size_bytes", 0)) for ev in gpu_memcpy)
    evidence = [
        f"内存拷贝 {memcpy_active_ns/1e6:.1f}ms（trace 的 {pct_of_trace*100:.1f}%）",
        f"{len(gpu_memcpy)} 次操作：{htod} HtoD，{dtoh} DtoH",
    ]
    if total_bytes > 0:
        evidence.append(f"总计 {total_bytes/1e9:.2f}GB，约 {total_bytes/memcpy_active_ns:.1f}GB/s")
    findings.append(NsysFinding(
        category="gpu_memcpy_hotspot",
        severity=severity,
        title=f"内存拷贝消耗 trace 的 {pct_of_trace*100:.1f}%",
        description=(
            f"GPU 内存拷贝/memset 占 {memcpy_active_ns/1e6:.1f}ms"
            f"（trace 的 {pct_of_trace*100:.1f}%）。"
            "可能存在过多的 Host-Device 传输或非异步 memcpy 模式。"
        ),
        evidence=evidence,
        next_step=(
            "检查是否有同步 memcpy（无 Async 的 cudaMemcpy）。"
            "使用 pin memory（cudaHostAlloc）提升 HtoD 带宽。"
            "使用单独 stream 的 cudaMemcpyAsync 异步预取数据。"
        ),
    ))


def _detect_sync_in_runtime(cuda_api: list[TimelineEvent]) -> bool:
    return any(ev.name in _SYNC_API_NAMES for ev in cuda_api)


def _finding_sync_wait(
    findings: list[NsysFinding],
    sync_active_ns: int,
    total_ns: int,
    sync_evs: list[TimelineEvent],
    cuda_api: list[TimelineEvent],
) -> None:
    sync_api_calls = [ev for ev in cuda_api if ev.name in _SYNC_API_NAMES]
    sync_api_ns = sum(ev.dur_ns for ev in sync_api_calls)
    total_sync_ns = sync_active_ns + sync_api_ns
    pct = total_sync_ns / total_ns if total_ns > 0 else 0
    if pct < 0.03 and not sync_api_calls:
        return
    severity = "warning" if pct >= _SYNC_WARN_PCT else "info"
    api_counts: Counter[str] = Counter()
    api_times: dict[str, int] = {}
    for ev in sync_api_calls:
        api_counts[ev.name] += 1
        api_times[ev.name] = api_times.get(ev.name, 0) + ev.dur_ns
    top_apis = sorted(api_times.items(), key=lambda x: -x[1])[:5]
    evidence = [f"同步开销 {total_sync_ns/1e6:.1f}ms（trace 的 {pct*100:.1f}%）"]
    for name, ns in top_apis:
        evidence.append(f"{name}: {ns/1e6:.1f}ms × {api_counts[name]} 次")
    findings.append(NsysFinding(
        category="sync_wait",
        severity=severity,
        title=f"同步开销 {total_sync_ns/1e6:.1f}ms（trace 的 {pct*100:.1f}%）",
        description=(
            f"CPU 在 CUDA 同步调用中阻塞了 {total_sync_ns/1e6:.1f}ms。"
            "过多的同步会串行化 GPU 执行，产生空闲气泡。"
        ),
        evidence=evidence,
        related_events=[name for name, _ in top_apis],
        next_step=(
            "用 cudaStreamSynchronize 替换 cudaDeviceSynchronize（流级别同步）。"
            "用 cudaMemcpyAsync + events 替换同步 cudaMemcpy。"
            "通过批处理工作或使用 CUDA events 降低同步频率。"
        ),
    ))


def _finding_host_launch(
    findings: list[NsysFinding],
    cuda_api: list[TimelineEvent],
    gpu_compute: list[TimelineEvent],
    total_ns: int,
) -> None:
    kernel_count = len(gpu_compute)
    avg_kernel_ns = sum(ev.dur_ns for ev in gpu_compute) / kernel_count
    corr_ids = {ev.correlation_id for ev in gpu_compute if ev.correlation_id is not None}
    launch_calls = [ev for ev in cuda_api if ev.correlation_id in corr_ids]
    if not launch_calls:
        return
    avg_launch_ns = sum(ev.dur_ns for ev in launch_calls) / len(launch_calls)
    launch_overhead_ns = sum(ev.dur_ns for ev in launch_calls)
    overhead_pct = launch_overhead_ns / total_ns
    if avg_kernel_ns > 100_000 or overhead_pct < 0.05:
        return
    severity = "warning" if overhead_pct < 0.20 else "critical"
    findings.append(NsysFinding(
        category="host_launch_overhead",
        severity=severity,
        title=(
            f"Host 内核启动开销 {overhead_pct*100:.1f}%"
            f"（{kernel_count} 个内核，均 {avg_kernel_ns/1e3:.1f}µs）"
        ),
        description=(
            f"有 {kernel_count} 个 GPU 内核，平均时长 {avg_kernel_ns/1e3:.1f}µs，"
            f"但平均启动延迟为 {avg_launch_ns/1e3:.1f}µs。"
            "启动开销占内核执行时间的显著比例。"
        ),
        evidence=[
            f"{kernel_count} 个内核，平均 {avg_kernel_ns/1e3:.1f}µs",
            f"平均启动延迟 {avg_launch_ns/1e3:.1f}µs",
            f"总启动开销 {launch_overhead_ns/1e6:.1f}ms（{overhead_pct*100:.1f}%）",
        ],
        next_step=(
            "使用 CUDA kernel fusion 或 torch.compile 融合小内核。"
            "使用 CUDA graphs 重放内核序列，消除重复的启动延迟。"
        ),
    ))


def _finding_cpu_hotspot(
    findings: list[NsysFinding],
    cpu_samples: list[TimelineEvent],
) -> None:
    counter: Counter[str] = Counter(ev.name for ev in cpu_samples)
    total = len(cpu_samples)
    top = counter.most_common(10)
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
    findings.append(NsysFinding(
        category="cpu_hotspot",
        severity=severity,
        title=f"CPU 热点：{top_name!r}（{top_pct*100:.1f}% 采样）",
        description=(
            f"CPU 采样显示 {top_name!r} 占 {top_pct*100:.1f}% 的 CPU 采样。"
            "这可能正在导致 GPU 空闲时间。"
        ),
        evidence=[
            f"{name}: {cnt} 次采样（{cnt/total*100:.1f}%）"
            for name, cnt in top_filtered[:5]
        ],
        related_hotspots=[name for name, _ in top_filtered[:5]],
        next_step=(
            "调查 CPU 热点函数。对 Python 进程，可能是 GIL 争用、"
            "DataLoader 瓶颈或过多的 Python 开销。"
        ),
    ))


def _finding_tiny_kernels(
    findings: list[NsysFinding],
    gpu_compute: list[TimelineEvent],
) -> None:
    if len(gpu_compute) < _TINY_KERNEL_COUNT_THRESHOLD:
        return
    avg_ns = sum(ev.dur_ns for ev in gpu_compute) / len(gpu_compute)
    if avg_ns >= _TINY_KERNEL_THRESHOLD_US * 1000:
        return
    tiny = [ev for ev in gpu_compute if ev.dur_ns < _TINY_KERNEL_THRESHOLD_US * 1000]
    tiny_pct = len(tiny) / len(gpu_compute)
    counter: Counter[str] = Counter(ev.name for ev in tiny)
    top = counter.most_common(5)
    findings.append(NsysFinding(
        category="many_tiny_kernels",
        severity="warning",
        title=(
            f"{len(gpu_compute)} 个内核，{tiny_pct*100:.0f}% 短于 {_TINY_KERNEL_THRESHOLD_US}µs"
            f"（均 {avg_ns/1e3:.1f}µs）"
        ),
        description=(
            f"{len(tiny)}/{len(gpu_compute)} 个内核（{tiny_pct*100:.0f}%）"
            f"短于 {_TINY_KERNEL_THRESHOLD_US}µs。内核启动开销可能占主导。"
        ),
        evidence=[
            f"总内核数：{len(gpu_compute)}，微小内核：{len(tiny)}",
            f"平均内核时长：{avg_ns/1e3:.1f}µs",
        ] + [f"{name}: {cnt} 个" for name, cnt in top],
        related_hotspots=[name for name, _ in top],
        next_step=(
            "使用 CUDA graphs 或 torch.compile 消除逐内核启动开销。"
            "融合逐元素操作；尽可能增大 batch size 以摊薄启动成本。"
        ),
    ))


# ═══════════════════════════════════════════════════════════════════════════════
# 统计辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_top_events(
    events: list[TimelineEvent],
    total_ns: int,
    limit: int = 20,
) -> list[EventStat]:
    """聚合事件，按总耗时返回 Top-N。"""
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
    """返回按总耗时排列的 Top-N 事件名称。"""
    by_name: dict[str, int] = defaultdict(int)
    for ev in events:
        by_name[ev.name] += ev.dur_ns
    return [k for k, _ in sorted(by_name.items(), key=lambda x: -x[1])[:n]]


def _trace_start(trace: NsysTrace) -> int:
    """找到 trace 中最早的事件起始时间。优先使用 trace_start_ns。"""
    if trace.trace_start_ns > 0:
        return trace.trace_start_ns
    starts = [ev.start_ns for ev in trace.events if not ev.is_sample and ev.dur_ns > 0]
    return min(starts) if starts else 0


def _compute_per_device(
    gpu_events: list[TimelineEvent],
    total_ns: int,
) -> list[DeviceBreakdown]:
    """Compute per-device GPU active/idle breakdown.

    Returns an empty list if all events share the same device_id (or device_id is None),
    since the global summary is sufficient for single-device profiles.
    """
    # Group by device_id
    by_device: dict[int, list[TimelineEvent]] = defaultdict(list)
    for ev in gpu_events:
        did = ev.device_id if ev.device_id is not None else 0
        by_device[did].append(ev)

    if len(by_device) <= 1:
        # Single device — per-device breakdown adds no value
        return []

    result: list[DeviceBreakdown] = []
    for device_id in sorted(by_device.keys()):
        evs = by_device[device_id]
        intervals = [(ev.start_ns, ev.start_ns + ev.dur_ns) for ev in evs if ev.dur_ns > 0]
        active_ns = union_ns(intervals)
        idle_ns = max(0, total_ns - active_ns)
        pct_active = active_ns / total_ns if total_ns > 0 else 0.0
        top_names = _top_kernels_by_time(evs, n=3)
        result.append(DeviceBreakdown(
            device_id=device_id,
            active_ns=active_ns,
            idle_ns=idle_ns,
            pct_active=pct_active,
            top_event_names=top_names,
        ))
    return result
