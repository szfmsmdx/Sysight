"""deterministic evidence-window extraction for nsys."""

from __future__ import annotations

from .extract import find_gaps
from .models import EvidenceWindow, NsysFinding, NsysTrace, TimelineEvent
from .stacks import (
    collect_actionable_evidence_near_range,
    collect_callstack_focus_near_range,
    collect_callstacks_near_range,
    collect_window_callstacks,
    collect_window_coarse_location,
)


_WINDOW_IDENTITY_CATEGORY_LABEL = {
    "sync_wait": "同步",
    "cuda_api": "launch",
    "gpu_memcpy": "memcpy",
}

_FINDING_EVENT_CATEGORY = {
    "gpu_compute_hotspot": "gpu_compute",
    "many_tiny_kernels": "gpu_compute",
    "sql_top_kernels": "gpu_compute",
    "sql_nvtx_layer_breakdown": "gpu_compute",
    "gpu_comm_hotspot": "gpu_comm",
    "comm_not_overlapped": "gpu_comm",
    "sql_nccl_breakdown": "gpu_comm",
    "gpu_memcpy_hotspot": "gpu_memcpy",
    "gpu_memcpy_h2d": "gpu_memcpy",
    "gpu_memcpy_d2h": "gpu_memcpy",
    "gpu_memset": "gpu_memcpy",
    "low_memcpy_throughput": "gpu_memcpy",
    "sql_memory_bandwidth": "gpu_memcpy",
    "sync_wait": "sync_wait",
    "sql_sync_cost": "sync_wait",
    "host_launch_overhead": "cuda_api",
    "cpu_hotspot": "cpu_sample",
    "host_gil_contention": "nvtx",
    "sql_nvtx_hotspots": "nvtx",
}


def extract_evidence_windows(
    trace: NsysTrace,
    findings: list[NsysFinding],
    top_n: int = 3,
) -> list[EvidenceWindow]:
    """extract deterministic coarse evidence windows from findings."""
    if not findings:
        return []

    ordered_events = sorted(
        trace.events,
        key=lambda ev: (ev.start_ns, ev.dur_ns, ev.name),
    )
    windows: list[EvidenceWindow] = []
    seen: set[tuple[str, int, int, str]] = set()

    for finding in findings:
        for window in _extract_windows_for_finding(trace, ordered_events, finding, top_n=top_n):
            key = (window.problem_id, window.start_ns, window.end_ns, window.event_name)
            if key in seen:
                continue
            seen.add(key)
            windows.append(window)

    windows.sort(key=lambda w: (w.duration_ns, w.start_ns), reverse=True)
    _assign_window_rank_in_iter(windows)
    return windows



def _extract_windows_for_finding(
    trace: NsysTrace,
    ordered_events: list[TimelineEvent],
    finding: NsysFinding,
    *,
    top_n: int,
) -> list[EvidenceWindow]:
    problem_id = finding.stable_id or finding.category

    if finding.category in {"gpu_idle", "sql_gpu_idle_gaps"}:
        return _idle_gap_windows(trace, problem_id, top_n)

    event_category = _FINDING_EVENT_CATEGORY.get(finding.category, "gpu_compute")
    candidates = _candidate_events_for_finding(trace, finding, event_category)
    windows = [
        _window_from_event(problem_id, finding.category, event, ordered_events, trace)
        for event in candidates[:top_n]
    ]
    if windows:
        return windows

    if finding.time_range_ns is None:
        return []
    start_ns, end_ns = finding.time_range_ns
    return [
        _window_from_range(
            problem_id,
            finding.category,
            start_ns,
            end_ns,
            event_name=finding.title,
            event_category="finding",
            trace=trace,
        )
    ]



def _candidate_events_for_finding(
    trace: NsysTrace,
    finding: NsysFinding,
    event_category: str,
) -> list[TimelineEvent]:
    candidates = [ev for ev in trace.events if ev.category == event_category]
    if event_category == "cuda_api":
        launch_keywords = ("launch", "cudaLaunchKernel", "cuLaunchKernel")
        candidates = [
            ev for ev in candidates
            if any(keyword.lower() in ev.name.lower() for keyword in launch_keywords)
        ] or candidates

    hotspot_names = {name.lower() for name in finding.related_hotspots if name}
    event_names = {name.lower() for name in finding.related_events if name}
    if hotspot_names or event_names:
        filtered = [
            ev for ev in candidates
            if ev.name.lower() in hotspot_names
            or ev.name.lower() in event_names
            or any(name in ev.name.lower() for name in hotspot_names)
        ]
        if filtered:
            candidates = filtered

    if finding.time_range_ns is not None:
        start_ns, end_ns = finding.time_range_ns
        ranged = [
            ev for ev in candidates
            if _event_end_ns(ev) > start_ns and ev.start_ns < end_ns
        ]
        if ranged:
            candidates = ranged

    return sorted(candidates, key=lambda ev: (ev.dur_ns, ev.start_ns), reverse=True)



def _idle_gap_windows(trace: NsysTrace, problem_id: str, top_n: int) -> list[EvidenceWindow]:
    gpu_intervals = [
        (ev.start_ns, _event_end_ns(ev))
        for ev in trace.events
        if ev.category in {"gpu_compute", "gpu_comm", "gpu_memcpy"}
    ]
    if not gpu_intervals:
        return []

    trace_start = trace.trace_start_ns
    trace_end = trace.trace_end_ns or max(end for _, end in gpu_intervals)
    gaps = find_gaps(gpu_intervals, trace_start, trace_end)
    return [
        _window_from_range(
            problem_id,
            "gpu_idle",
            start_ns,
            end_ns,
            event_name="gpu_idle_gap",
            event_category="gap",
            trace=trace,
        )
        for start_ns, end_ns in sorted(
            gaps,
            key=lambda gap: gap[1] - gap[0],
            reverse=True,
        )[:top_n]
    ]



def _window_from_event(
    problem_id: str,
    category: str,
    event: TimelineEvent,
    ordered_events: list[TimelineEvent],
    trace: NsysTrace,
) -> EvidenceWindow:
    start_ns, end_ns = _clip_window_to_trace(trace, event.start_ns, _event_end_ns(event))
    before_event, after_event = _neighbor_event_names(ordered_events, event)
    nvtx_labels = _collect_nvtx_labels(trace, start_ns, end_ns)
    actionable = _collect_window_actionable_evidence(trace, event, start_ns, end_ns, nvtx_labels=nvtx_labels)
    return EvidenceWindow(
        problem_id=problem_id,
        category=category,
        start_ns=start_ns,
        end_ns=end_ns,
        duration_ns=max(1, end_ns - start_ns),
        device_id=event.device_id,
        stream_id=event.stream_id,
        correlation_id=event.correlation_id,
        event_name=event.name,
        event_category=event.category,
        before_event=before_event,
        after_event=after_event,
        runtime_api=_find_runtime_api(trace, event),
        nvtx_labels=nvtx_labels,
        coarse_location=collect_window_coarse_location(
            trace,
            event,
            start_ns,
            end_ns,
            nvtx_labels=nvtx_labels,
        ),
        callstack_summaries=collect_window_callstacks(trace, event, start_ns, end_ns),
        sample_callstack=list(actionable["sample_callstack"]),
        first_user_python_frame=actionable["first_user_python_frame"],
        actionable_chain=list(actionable["actionable_chain"]),
        actionable_leaf_reason=str(actionable["actionable_leaf_reason"]),
        why_not_actionable=str(actionable["why_not_actionable"]),
        window_rank_in_iter=None,
        kernel_constraints=_kernel_constraints_for_event(event),
    )



def _window_from_range(
    problem_id: str,
    category: str,
    start_ns: int,
    end_ns: int,
    *,
    event_name: str,
    event_category: str,
    trace: NsysTrace,
) -> EvidenceWindow:
    start_ns, end_ns = _clip_window_to_trace(trace, start_ns, end_ns)
    nvtx_labels = _collect_nvtx_labels(trace, start_ns, end_ns)
    actionable = collect_actionable_evidence_near_range(
        trace,
        start_ns,
        end_ns,
        nvtx_labels=nvtx_labels,
    )
    return EvidenceWindow(
        problem_id=problem_id,
        category=category,
        start_ns=start_ns,
        end_ns=end_ns,
        duration_ns=max(1, end_ns - start_ns),
        device_id=None,
        stream_id=None,
        correlation_id=None,
        event_name=event_name,
        event_category=event_category,
        before_event=None,
        after_event=None,
        runtime_api=None,
        nvtx_labels=nvtx_labels,
        coarse_location=collect_callstack_focus_near_range(
            trace,
            start_ns,
            end_ns,
            nvtx_labels=nvtx_labels,
        ),
        callstack_summaries=collect_callstacks_near_range(trace, start_ns, end_ns),
        sample_callstack=list(actionable["sample_callstack"]),
        first_user_python_frame=actionable["first_user_python_frame"],
        actionable_chain=list(actionable["actionable_chain"]),
        actionable_leaf_reason=str(actionable["actionable_leaf_reason"]),
        why_not_actionable=str(actionable["why_not_actionable"]),
        window_rank_in_iter=None,
        kernel_constraints={},
    )



def _collect_window_actionable_evidence(
    trace: NsysTrace,
    event: TimelineEvent,
    start_ns: int,
    end_ns: int,
    *,
    nvtx_labels: list[str] | None = None,
) -> dict[str, object]:
    thread_ids: set[int] = set()
    anchor_times = [start_ns, end_ns, event.start_ns]
    if event.global_tid is not None:
        thread_ids.add(event.global_tid)
    if event.correlation_id is not None:
        for candidate in trace.events:
            if candidate.correlation_id != event.correlation_id:
                continue
            anchor_times.append(candidate.start_ns)
            if candidate.global_tid is not None:
                thread_ids.add(candidate.global_tid)
    return collect_actionable_evidence_near_range(
        trace,
        min(anchor_times),
        max(anchor_times),
        thread_ids=thread_ids,
        nvtx_labels=nvtx_labels,
    )



def _neighbor_event_names(
    ordered_events: list[TimelineEvent],
    target: TimelineEvent,
) -> tuple[str | None, str | None]:
    try:
        idx = ordered_events.index(target)
    except ValueError:
        return None, None
    before_event = ordered_events[idx - 1].name if idx > 0 else None
    after_event = ordered_events[idx + 1].name if idx + 1 < len(ordered_events) else None
    return before_event, after_event



def _collect_nvtx_labels(
    trace: NsysTrace,
    start_ns: int,
    end_ns: int,
    limit: int = 3,
) -> list[str]:
    labels: list[str] = []
    for event in trace.events:
        if event.category != "nvtx":
            continue
        event_end = _event_end_ns(event)
        if event_end <= start_ns or event.start_ns >= end_ns:
            continue
        if event.name and event.name not in labels:
            labels.append(event.name)
        if len(labels) >= limit:
            break
    return labels



def _assign_window_rank_in_iter(windows: list[EvidenceWindow]) -> None:
    grouped: dict[tuple[str, str], list[EvidenceWindow]] = {}
    for window in windows:
        iter_label = _iter_label_for_window(window)
        rank_key = _window_rank_category(window)
        if not iter_label or not rank_key:
            continue
        grouped.setdefault((iter_label, rank_key), []).append(window)

    for group in grouped.values():
        for rank, window in enumerate(sorted(group, key=lambda item: (item.start_ns, item.end_ns, item.event_name)), start=1):
            window.window_rank_in_iter = rank



def _iter_label_for_window(window: EvidenceWindow) -> str | None:
    for label in window.nvtx_labels:
        text = (label or "").strip()
        if text.lower().startswith("iter_"):
            return text
    return None



def _window_rank_category(window: EvidenceWindow) -> str | None:
    category = _WINDOW_IDENTITY_CATEGORY_LABEL.get(window.event_category)
    if category:
        return category
    runtime_name = (window.runtime_api or window.event_name or "").lower()
    if "memcpy" in runtime_name:
        return "memcpy"
    if "launch" in runtime_name:
        return "launch"
    return None



def _find_runtime_api(trace: NsysTrace, event: TimelineEvent) -> str | None:
    if event.category == "cuda_api":
        return event.name
    if event.correlation_id is None:
        return None
    for candidate in trace.events:
        if candidate.category != "cuda_api":
            continue
        if candidate.correlation_id == event.correlation_id:
            return candidate.name
    return None



def _kernel_constraints_for_event(event: TimelineEvent) -> dict[str, str]:
    if event.category != "gpu_compute":
        return {}
    name = event.name.lower()
    tensor_core = "likely" if any(token in name for token in ("mma", "wmma", "hmma", "tensor")) else "unknown"
    duration_bucket = "tiny" if event.dur_ns < 50_000 else "regular"
    return {
        "kernel_name": event.name,
        "tensor_core": tensor_core,
        "arithmetic_intensity": "unknown",
        "duration_bucket": duration_bucket,
    }



def _event_end_ns(event: TimelineEvent) -> int:
    return event.start_ns + max(event.dur_ns, 1)



def _clip_window_to_trace(trace: NsysTrace, start_ns: int, end_ns: int) -> tuple[int, int]:
    trace_start = trace.trace_start_ns
    trace_end = trace.trace_end_ns or max(trace.trace_start_ns + trace.duration_ns, end_ns)
    clipped_start = min(max(start_ns, trace_start), trace_end)
    clipped_end = min(max(end_ns, clipped_start + 1), trace_end)
    return clipped_start, clipped_end
