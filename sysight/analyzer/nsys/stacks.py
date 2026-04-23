"""Callstack utilities and CPU hotspot aggregation for nsys."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .models import NsysTrace, SampleHotspot, SourceFrame, TimelineEvent

_BAD_STACK_FRAMES = frozenset({
    "",
    "0x0",
    "[Unknown]",
    "[Broken backtraces]",
    "start_thread",
    "__clone",
    "clone",
    "pythread_wrapper",
})

_WRAPPER_STACK_FRAMES = frozenset({
    "_PyEval_Vector",
    "_PyObject_Call",
    "_PyObject_MakeTpCall",
    "_PyObject_Call_Prepend",
    "PyObject_Call",
    "PyObject_Vectorcall",
    "method_vectorcall",
    "method_vectorcall_VARARGS_KEYWORDS",
    "type_call",
    "slot_tp_call",
    "slot_tp_iternext",
    "builtin_next",
    "thread_run",
    "PyEval_EvalCode",
    "run_eval_code_obj",
    "run_mod",
    "pyrun_file",
    "_PyRun_SimpleFileObject",
    "_PyRun_AnyFileObject",
    "Py_RunMain",
    "Py_BytesMain",
    "__libc_start_main",
})

_COLLAPSE_REPEATED_STACK_FRAMES = frozenset({
    "_PyEval_EvalFrameDefault",
    "_PyEval_Vector",
})

_LOW_SIGNAL_STACK_PREFIXES = (
    "__clock_gettime",
    "clock_gettime",
    "clock_nanosleep",
    "__nanosleep",
    "nanosleep",
    "pthread_cond_timedwait",
    "pthread_cond_wait",
    "__pthread_cond",
    "__sched_yield",
    "sched_yield",
    "__munmap",
    "munmap",
    "__pthread_mutex_unlock",
    "__lll_lock_wait",
    "futex",
)



def clean_callstack_frames(frames: list[str]) -> list[str]:
    """Remove unresolved frames and wrapper noise while keeping call order."""
    return [frame.symbol or "" for frame in clean_source_frames(_symbols_to_frames(frames))]



def normalize_callstack_frames(frames: list[str]) -> list[str]:
    """Drop low-signal syscall leafs when deeper actionable callers exist."""
    return [frame.symbol or "" for frame in normalize_source_frames(_symbols_to_frames(frames))]



def format_callstack_summary(frames: list[str], depth: int = 4) -> str:
    trimmed = normalize_callstack_frames(frames)[:depth]
    if not trimmed:
        return "—"
    return " <- ".join(trimmed)



def collect_window_callstacks(
    trace: NsysTrace,
    event: TimelineEvent,
    start_ns: int,
    end_ns: int,
    limit: int = 3,
) -> list[str]:
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

    return collect_callstacks_near_range(
        trace,
        min(anchor_times),
        max(anchor_times),
        thread_ids=thread_ids,
        limit=limit,
    )



def collect_window_coarse_location(
    trace: NsysTrace,
    event: TimelineEvent,
    start_ns: int,
    end_ns: int,
    *,
    nvtx_labels: list[str] | None = None,
) -> str | None:
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

    return collect_callstack_focus_near_range(
        trace,
        min(anchor_times),
        max(anchor_times),
        thread_ids=thread_ids,
        nvtx_labels=nvtx_labels,
    )



def collect_callstacks_near_range(
    trace: NsysTrace,
    start_ns: int,
    end_ns: int,
    *,
    thread_ids: set[int] | None = None,
    limit: int = 3,
) -> list[str]:
    candidates = _collect_callstack_candidates(trace, start_ns, end_ns, thread_ids=thread_ids)
    if not candidates:
        return []

    summaries: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        summary = candidate["summary"]
        if summary in seen:
            continue
        seen.add(summary)
        summaries.append(summary)
        if len(summaries) >= limit:
            break
    return summaries



def collect_callstack_focus_near_range(
    trace: NsysTrace,
    start_ns: int,
    end_ns: int,
    *,
    thread_ids: set[int] | None = None,
    nvtx_labels: list[str] | None = None,
) -> str | None:
    candidates = _collect_callstack_candidates(trace, start_ns, end_ns, thread_ids=thread_ids)
    for candidate in candidates:
        focus = summarize_callstack_focus(candidate["frames"], nvtx_labels=nvtx_labels)
        if focus:
            return focus
    if nvtx_labels:
        return summarize_callstack_focus([], nvtx_labels=nvtx_labels)
    return None



def collect_actionable_evidence_near_range(
    trace: NsysTrace,
    start_ns: int,
    end_ns: int,
    *,
    thread_ids: set[int] | None = None,
    nvtx_labels: list[str] | None = None,
) -> dict[str, object]:
    candidates = _collect_callstack_candidates(trace, start_ns, end_ns, thread_ids=thread_ids)
    if not candidates:
        focus = summarize_callstack_focus([], nvtx_labels=nvtx_labels) if nvtx_labels else None
        return {
            "sample_callstack": [],
            "first_user_python_frame": None,
            "actionable_chain": [],
            "actionable_leaf_reason": focus or "",
            "why_not_actionable": "当前窗口缺少稳定 CPU sample/backtrace，尚未形成上层调用链证据。",
        }

    frames = list(candidates[0]["frames"])
    focus = summarize_callstack_focus(frames, nvtx_labels=nvtx_labels) or ""
    user_python = _first_user_python_frame(frames)
    actionable_chain = _build_actionable_chain(frames, user_python)
    why_not_actionable = ""
    if not user_python:
        why_not_actionable = "当前窗口仍主要停留在运行时/库侧路径，缺少稳定 Python 调用点。"
    return {
        "sample_callstack": [frame.symbol or "" for frame in frames[:6]],
        "first_user_python_frame": user_python,
        "actionable_chain": actionable_chain,
        "actionable_leaf_reason": focus,
        "why_not_actionable": why_not_actionable,
    }



def build_cpu_hotspots(
    trace: NsysTrace,
    top_n: int = 20,
) -> list[SampleHotspot]:
    """Extract CPU sample hotspots and keep a readable representative stack."""
    cpu_samples = [ev for ev in trace.events if ev.is_sample]
    if not cpu_samples:
        return []

    groups: dict[tuple[str, ...], list[TimelineEvent]] = defaultdict(list)
    representatives: dict[tuple[str, ...], list[SourceFrame]] = {}
    fallback_groups: dict[tuple[str, ...], list[TimelineEvent]] = defaultdict(list)
    fallback_representatives: dict[tuple[str, ...], list[SourceFrame]] = {}
    for sample in cpu_samples:
        clean_frames = normalize_source_frames(extract_source_frames(sample))
        if not clean_frames:
            continue
        key = tuple(frame.symbol or "" for frame in clean_frames[:6])
        if len(clean_frames) >= 2:
            groups[key].append(sample)
            if len(clean_frames) > len(representatives.get(key, [])):
                representatives[key] = clean_frames
        else:
            fallback_groups[key].append(sample)
            if len(clean_frames) > len(fallback_representatives.get(key, [])):
                fallback_representatives[key] = clean_frames

    if not groups:
        groups = fallback_groups
        representatives = fallback_representatives

    total = len(cpu_samples)
    hotspots: list[SampleHotspot] = []
    ranked = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
    for frames, samples in ranked[:top_n]:
        count = len(samples)
        pct = count / total
        start_ns = min(sample.start_ns for sample in samples)
        end_ns = max(sample.start_ns for sample in samples)
        source_frames = representatives.get(frames, _symbols_to_frames(list(frames)))
        callstack = [frame.symbol or "" for frame in source_frames]
        symbol = callstack[0] if callstack else "cpu_sample"
        frame = SourceFrame(
            symbol=symbol,
            source_file=None,
            source_line=None,
            module=source_frames[0].module if source_frames else None,
            raw=format_callstack_summary(callstack, depth=4),
        )
        hotspots.append(SampleHotspot(
            frame=frame,
            count=count,
            pct=pct,
            event_window_ns=(start_ns, end_ns),
            callstack=callstack,
            coarse_location=summarize_callstack_focus(source_frames),
        ))

    return hotspots



def extract_source_frames(sample: TimelineEvent) -> list[SourceFrame]:
    extra = sample.extra if isinstance(sample.extra, dict) else {}
    structured = extra.get("callstack_frames")
    if isinstance(structured, list) and structured:
        frames: list[SourceFrame] = []
        for item in structured:
            if not isinstance(item, dict):
                continue
            symbol = _normalize_symbol(item.get("symbol"))
            module = str(item.get("module") or "").strip() or None
            if not symbol:
                continue
            frames.append(SourceFrame(symbol=symbol, source_file=None, source_line=None, module=module, raw=symbol))
        if frames:
            return frames

    raw_frames = extra.get("callstack")
    if isinstance(raw_frames, list) and raw_frames:
        return _symbols_to_frames([str(frame) for frame in raw_frames])

    return _symbols_to_frames([sample.name])



def summarize_callstack_focus(
    frames: list[SourceFrame],
    *,
    nvtx_labels: list[str] | None = None,
) -> str | None:
    clean_frames = normalize_source_frames(frames)
    symbols = [frame.symbol or "" for frame in clean_frames]
    joined = " <- ".join(symbols).lower()
    nvtx_text = " > ".join(nvtx_labels or []).lower()
    user_python = _first_user_python_frame(clean_frames)

    if "watchdog" in joined and "nccl" in joined:
        if "gil" in joined or "gil" in nvtx_text:
            return "NCCL watchdog 线程，疑似卡在 GIL 获取/恢复"
        return "NCCL watchdog 线程，疑似通信/监控等待"

    if "gil" in joined or "gil" in nvtx_text or "pygilstate_ensure" in joined or "pyeval_restorethread" in joined:
        if user_python:
            return f"Python GIL 等待/恢复，入口函数 `{user_python}`"
        return "Python GIL 等待/恢复路径"

    if any(token in joined for token in (
        "cudastreamsynchronize",
        "memcpy_and_sync",
        "copy_kernel_cuda",
        "copy_device_to_device",
        "cudamemcpyasync",
        "copy_impl(",
        "copy_(",
    )):
        if user_python:
            return f"Python 函数 `{user_python}` 触发 Torch 张量拷贝/同步"
        return "Torch 张量拷贝/同步路径"

    if any(token in joined for token in (
        "__cudapushcallconfiguration",
        "cudalaunchkernel",
        "launch <-",
        " launch",
    )):
        if user_python:
            return f"Python 函数 `{user_python}` 触发 CUDA kernel launch"
        return "CUDA kernel launch / runtime 路径"

    if "torch::autograd::engine::evaluate_function" in joined or "torch::autograd::engine::thread_main" in joined:
        return "PyTorch autograd worker 线程"

    if any(token in joined for token in (
        "mapallocator",
        "storageimpl::release_resources",
        "tensorimpl::~tensorimpl",
        "thpvariable_subclass_clear",
    )):
        return "Tensor/Storage 释放路径"

    if user_python:
        return f"Python 函数 `{user_python}` 触发的运行时路径"

    library_frame = _first_actionable_library_frame(clean_frames)
    if library_frame:
        return _humanize_library_frame(library_frame)

    if nvtx_labels:
        path = " > ".join(nvtx_labels[:2])
        return f"NVTX 区域 `{path}`"
    return None



def clean_source_frames(frames: list[SourceFrame]) -> list[SourceFrame]:
    cleaned: list[SourceFrame] = []
    seen_repeated: set[str] = set()
    previous: str | None = None
    for frame in frames:
        symbol = _normalize_symbol(frame.symbol)
        if not symbol:
            continue
        if _is_bad_stack_frame(symbol) or _is_wrapper_stack_frame(symbol):
            continue
        if symbol == previous:
            continue
        if symbol in _COLLAPSE_REPEATED_STACK_FRAMES:
            if symbol in seen_repeated:
                continue
            seen_repeated.add(symbol)
        cleaned.append(SourceFrame(
            symbol=symbol,
            source_file=frame.source_file,
            source_line=frame.source_line,
            module=frame.module,
            raw=frame.raw or symbol,
        ))
        previous = symbol
    return cleaned



def normalize_source_frames(frames: list[SourceFrame]) -> list[SourceFrame]:
    cleaned = clean_source_frames(frames)
    start = 0
    while start + 1 < len(cleaned) and _is_low_signal_stack_frame(cleaned[start].symbol or ""):
        start += 1
    return cleaned[start:]



def _collect_callstack_candidates(
    trace: NsysTrace,
    start_ns: int,
    end_ns: int,
    *,
    thread_ids: set[int] | None = None,
) -> list[dict[str, object]]:
    margin_ns = max(2_000_000, end_ns - start_ns)
    center = (start_ns + end_ns) // 2
    candidates: list[dict[str, object]] = []

    for sample in trace.events:
        if not sample.is_sample:
            continue
        if thread_ids and sample.global_tid not in thread_ids:
            continue
        if sample.start_ns < start_ns - margin_ns or sample.start_ns > end_ns + margin_ns:
            continue

        clean_frames = normalize_source_frames(extract_source_frames(sample))
        if not clean_frames:
            continue
        summary = format_callstack_summary([frame.symbol or "" for frame in clean_frames], depth=6)
        if summary == "—":
            continue
        focus = summarize_callstack_focus(clean_frames)
        candidates.append({
            "frames": clean_frames,
            "summary": summary,
            "focus": focus,
            "distance": abs(sample.start_ns - center),
            "quality": _callstack_quality(clean_frames, focus),
        })

    candidates.sort(key=lambda item: (int(item["quality"]), int(item["distance"])))
    return candidates



def _callstack_quality(frames: list[SourceFrame], focus: str | None) -> int:
    if focus and "Python 函数" in focus:
        return 0
    if focus and any(token in focus for token in (
        "NCCL watchdog",
        "GIL",
        "张量拷贝",
        "autograd worker",
        "Tensor/Storage",
    )):
        return 1
    if focus and "NVTX 区域" not in focus:
        return 2
    if _is_low_signal_only_stack(frames):
        return 4
    if len(frames) >= 2:
        return 3
    return 5



def _symbols_to_frames(frames: list[str]) -> list[SourceFrame]:
    result: list[SourceFrame] = []
    for frame in frames:
        symbol = _normalize_symbol(frame)
        if not symbol:
            continue
        result.append(SourceFrame(symbol=symbol, source_file=None, source_line=None, raw=symbol))
    return result



def _normalize_symbol(value: str | None) -> str:
    if value is None:
        return ""
    symbol = str(value).strip()
    if "@@" in symbol:
        symbol = symbol.split("@@", 1)[0]
    return symbol



def _first_user_python_frame(frames: list[SourceFrame]) -> str | None:
    for frame in frames:
        symbol = frame.symbol or ""
        module = (frame.module or "").lower()
        if "libpython" not in module:
            continue
        if symbol.startswith("Py") or symbol.startswith("_Py"):
            continue
        if symbol in _WRAPPER_STACK_FRAMES or symbol in _BAD_STACK_FRAMES:
            continue
        if _is_low_signal_stack_frame(symbol):
            continue
        if "::" in symbol or symbol.startswith("__"):
            continue
        return symbol
    return None



def _first_actionable_library_frame(frames: list[SourceFrame]) -> SourceFrame | None:
    for frame in frames:
        symbol = (frame.symbol or "").lower()
        if not symbol or _is_low_signal_stack_frame(symbol):
            continue
        if symbol.startswith("py") or symbol.startswith("_py"):
            continue
        if frame.module and "libpython" in frame.module.lower():
            continue
        return frame
    return None



def _build_actionable_chain(frames: list[SourceFrame], user_python: str | None) -> list[str]:
    chain: list[str] = []
    if user_python:
        chain.append(user_python)
    for frame in frames:
        symbol = frame.symbol or ""
        if not symbol or symbol == user_python:
            continue
        if _is_low_signal_stack_frame(symbol):
            continue
        if symbol in chain:
            continue
        chain.append(symbol)
        if len(chain) >= 4:
            break
    return chain



def _humanize_library_frame(frame: SourceFrame) -> str:
    symbol = frame.symbol or "—"
    low = symbol.lower()
    if symbol in {"PyEval_RestoreThread", "PyGILState_Ensure", "_PyEval_EvalFrameDefault", "cfunction_call", "launch"}:
        return "Python/CUDA 运行时包装路径"
    if "nccl" in low or "processgroupnccl" in low:
        return f"PyTorch distributed / NCCL 路径：{symbol}"
    if any(token in low for token in ("cuda", "cublas", "cudnn")):
        return f"CUDA 运行时路径：{symbol}"
    if any(token in low for token in ("at::", "c10::", "torch::", "thpvariable")):
        return f"PyTorch 运行时路径：{symbol}"
    if frame.module and frame.module not in {"[Unknown]", "[kernel.kallsyms]"}:
        return f"{Path(frame.module).name}: {symbol}"
    return symbol



def _is_bad_stack_frame(frame: str) -> bool:
    low = frame.lower()
    if frame in _BAD_STACK_FRAMES:
        return True
    if low.startswith("0x"):
        return True
    if low.startswith("nsys_"):
        return True
    return False



def _is_wrapper_stack_frame(frame: str) -> bool:
    return frame in _WRAPPER_STACK_FRAMES or frame.startswith("void c10::function_ref<")



def _is_low_signal_stack_frame(frame: str) -> bool:
    low = frame.lower()
    return any(low.startswith(prefix) for prefix in _LOW_SIGNAL_STACK_PREFIXES)



def _is_low_signal_only_stack(frames: list[SourceFrame]) -> bool:
    if not frames:
        return True
    noisy = {
        "PyEval_RestoreThread",
        "PyGILState_Ensure",
        "_PyEval_EvalFrameDefault",
        "_PyThreadState_PopFrame",
        "cfunction_call",
        "launch",
    }
    actionable = 0
    for frame in frames:
        symbol = frame.symbol or ""
        if not symbol:
            continue
        if symbol in noisy or _is_low_signal_stack_frame(symbol):
            continue
        actionable += 1
    return actionable == 0
