"""Code-location helpers: map suspicious windows to sampled stacks and likely call sites."""

from __future__ import annotations

from collections import Counter

from nsys_agent.analysis.queries import detect_idle_gaps
from nsys_agent.profile import Profile

_NOISY_MODULE_FRAGMENTS = (
    "[kernel.kallsyms]",
    "[unknown]",
    "[broken backtraces]",
    "libpython",
    "libc-",
    "libpthread",
    "libcuda.so",
    "libcudart",
    "libcupti",
    "libtoolsinjection",
    "ld-",
    "[vdso]",
)

_HIGH_VALUE_SYMBOL_HINTS = (
    "_to_copy",
    "copy_",
    "copy_impl",
    "copy_kernel_cuda",
    "variabletype",
    "thpfunction_apply",
    "dynamo__custom_eval_frame",
    "dynamo_eval_custom_code",
    "launch",
    "triton",
    "nccl",
    "streamsynchronize",
    "eventsynchronize",
)


def thread_name_for_tid(prof: Profile, global_tid: int) -> str:
    """Resolve a thread name from ThreadNames + StringIds."""
    if "ThreadNames" not in prof.schema.tables:
        return ""
    sql = """
        SELECT s.value
        FROM ThreadNames t
        JOIN StringIds s ON t.nameId = s.id
        WHERE t.globalTid = ?
        LIMIT 1
    """
    with prof._lock:
        row = prof.conn.execute(sql, (global_tid,)).fetchone()
    return (row[0] or "") if row else ""


def runtime_calls_in_window(
    prof: Profile,
    start_ns: int,
    end_ns: int,
    *,
    global_tid: int | None = None,
    limit: int = 20,
) -> list[dict]:
    """Return runtime API calls overlapping a time window."""
    if not prof.schema.runtime_table:
        return []
    sql = f"""
        SELECT
            r.globalTid,
            s.value AS api_name,
            r.start,
            r.[end],
            ROUND((r.[end] - r.start) / 1e6, 3) AS dur_ms
        FROM {prof.schema.runtime_table} r
        JOIN StringIds s ON r.nameId = s.id
        WHERE r.start < ? AND r.[end] > ?
    """
    params: list[object] = [end_ns, start_ns]
    if global_tid is not None:
        sql += " AND r.globalTid = ?"
        params.append(global_tid)
    sql += " ORDER BY (r.[end] - r.start) DESC LIMIT ?"
    params.append(limit)
    with prof._lock:
        return [dict(row) for row in prof.conn.execute(sql, params).fetchall()]


def nvtx_ranges_in_window(
    prof: Profile,
    start_ns: int,
    end_ns: int,
    *,
    global_tid: int | None = None,
    limit: int = 5,
) -> list[dict]:
    """Return overlapping NVTX ranges ordered from inner-most to outer-most."""
    table = prof.schema.nvtx_table
    if not table:
        return []

    if prof._nvtx_has_text_id:
        sql = f"""
            SELECT
                COALESCE(n.text, s.value) AS text,
                n.start,
                n.[end],
                ROUND((n.[end] - n.start) / 1e6, 3) AS dur_ms
            FROM {table} n
            LEFT JOIN StringIds s ON n.textId = s.id
            WHERE (n.text IS NOT NULL OR s.value IS NOT NULL)
              AND n.[end] > n.start
              AND n.start < ? AND n.[end] > ?
        """
    else:
        sql = f"""
            SELECT
                text,
                start,
                [end],
                ROUND(([end] - start) / 1e6, 3) AS dur_ms
            FROM {table}
            WHERE text IS NOT NULL AND [end] > start
              AND start < ? AND [end] > ?
        """
    params: list[object] = [end_ns, start_ns]
    if global_tid is not None:
        sql += " AND globalTid = ?"
        params.append(global_tid)
    sql += " ORDER BY ([end] - start) ASC LIMIT ?"
    params.append(limit)
    with prof._lock:
        return [dict(row) for row in prof.conn.execute(sql, params).fetchall()]


def candidate_threads_for_window(
    prof: Profile, start_ns: int, end_ns: int, *, top_n: int = 8
) -> list[dict]:
    """Return runtime-active threads in a window, ordered by total runtime duration."""
    if not prof.schema.runtime_table:
        return []
    sql = f"""
        SELECT
            r.globalTid,
            COUNT(*) AS api_count,
            SUM(r.[end] - r.start) AS total_ns
        FROM {prof.schema.runtime_table} r
        WHERE r.start < ? AND r.[end] > ?
        GROUP BY r.globalTid
        ORDER BY total_ns DESC
        LIMIT ?
    """
    with prof._lock:
        rows = [dict(row) for row in prof.conn.execute(sql, (end_ns, start_ns, top_n)).fetchall()]
    for row in rows:
        row["thread_name"] = thread_name_for_tid(prof, row["globalTid"])
    return rows


def sample_hotspots_for_thread(
    prof: Profile,
    global_tid: int,
    start_ns: int,
    end_ns: int,
    *,
    limit: int = 40,
) -> list[dict]:
    """Aggregate sampled stack frames for one thread in a window."""
    if "COMPOSITE_EVENTS" not in prof.schema.tables or "SAMPLING_CALLCHAINS" not in prof.schema.tables:
        return []
    sql = """
        WITH frames AS (
            SELECT
                ce.id,
                sc.stackDepth,
                ss.value AS symbol,
                ms.value AS module
            FROM COMPOSITE_EVENTS ce
            JOIN SAMPLING_CALLCHAINS sc ON ce.id = sc.id
            LEFT JOIN StringIds ss ON sc.symbol = ss.id
            LEFT JOIN StringIds ms ON sc.module = ms.id
            WHERE ce.globalTid = ?
              AND ce.start BETWEEN ? AND ?
        )
        SELECT
            symbol,
            module,
            COUNT(DISTINCT id) AS samples,
            MIN(stackDepth) AS min_depth,
            MAX(stackDepth) AS max_depth
        FROM frames
        GROUP BY symbol, module
        ORDER BY samples DESC
        LIMIT ?
    """
    with prof._lock:
        rows = [dict(row) for row in prof.conn.execute(sql, (global_tid, start_ns, end_ns, limit)).fetchall()]
    return rows


def classify_code_candidate(symbol: str, module: str, apis: list[dict]) -> str:
    """Give a short explanation for why a frame is interesting."""
    text = f"{symbol} {module}".lower()
    api_names = " ".join((api["api_name"] or "").lower() for api in apis)
    if "_to_copy" in text or "copy_" in text or "copy_impl" in text:
        if "synchronize" in api_names:
            return "Likely tensor copy path that later blocks on synchronization; check .to(...), .cpu(), or implicit host-device copies."
        return "Likely tensor copy path; check .to(...), .cpu(), non_blocking transfers, and pinned memory."
    if "thpfunction_apply" in text:
        return "PyTorch autograd apply path; inspect the corresponding backward/autograd step around this window."
    if "dynamo" in text or "triton" in text or "launch" in text:
        return "Compiler or Triton launch path; inspect compiled graph boundaries and launcher overhead."
    if "synchronize" in api_names:
        return "Window dominated by synchronization APIs; check explicit waits, barriers, and .item() calls in the hot path."
    return "Frequently sampled frame inside the suspicious window."


def _is_noise_frame(symbol: str, module: str) -> bool:
    symbol_text = (symbol or "").strip()
    module_text = (module or "").strip().lower()
    if not symbol_text or symbol_text == "0x0":
        return True
    if symbol_text == "[Broken backtraces]":
        return True
    return any(fragment in module_text for fragment in _NOISY_MODULE_FRAGMENTS)


def _is_interesting_frame(symbol: str, module: str) -> bool:
    text = f"{symbol} {module}".lower()
    if any(hint in text for hint in _HIGH_VALUE_SYMBOL_HINTS):
        return True
    return "libtorch" in text or "__triton_launcher" in text


def _score_frame(frame: dict) -> tuple[int, int, int]:
    bonus = 10 if _is_interesting_frame(frame["symbol"], frame["module"]) else 0
    prefer_shallower = -int(frame.get("min_depth") or 0)
    return (int(frame["samples"]) + bonus, bonus, prefer_shallower)


def top_code_candidates(
    prof: Profile,
    start_ns: int,
    end_ns: int,
    *,
    max_threads: int = 3,
    max_frames_per_thread: int = 3,
) -> list[dict]:
    """Return likely code-location candidates for a suspicious window."""
    thread_rows = candidate_threads_for_window(prof, start_ns, end_ns, top_n=8)
    preferred_threads = [
        row
        for row in thread_rows
        if (row.get("thread_name") or "").startswith("python") or "autograd" in (row.get("thread_name") or "")
    ]
    if not preferred_threads:
        preferred_threads = thread_rows[:max_threads]
    else:
        preferred_threads = preferred_threads[:max_threads]

    candidates: list[dict] = []
    for thread in preferred_threads:
        tid = int(thread["globalTid"])
        apis = runtime_calls_in_window(prof, start_ns, end_ns, global_tid=tid, limit=8)
        nvtx_ranges = nvtx_ranges_in_window(prof, start_ns, end_ns, global_tid=tid, limit=4)
        frames = sample_hotspots_for_thread(prof, tid, start_ns, end_ns, limit=60)
        filtered = [
            frame
            for frame in frames
            if not _is_noise_frame(frame["symbol"], frame["module"])
        ]
        filtered.sort(key=_score_frame, reverse=True)
        selected = []
        seen = set()
        for frame in filtered:
            key = (frame["symbol"], frame["module"])
            if key in seen:
                continue
            seen.add(key)
            frame = dict(frame)
            frame["reason"] = classify_code_candidate(frame["symbol"], frame["module"], apis)
            selected.append(frame)
            if len(selected) >= max_frames_per_thread:
                break

        if not selected:
            continue

        api_counter = Counter((api["api_name"] or "").split("_v")[0] for api in apis)
        candidates.append(
            {
                "global_tid": tid,
                "thread_name": thread.get("thread_name") or "",
                "api_count": int(thread.get("api_count") or 0),
                "total_runtime_ms": round((thread.get("total_ns") or 0) / 1e6, 3),
                "top_apis": [name for name, _ in api_counter.most_common(3)],
                "nvtx_ranges": nvtx_ranges,
                "frames": selected,
            }
        )
    return candidates


def locate_code_for_gaps(
    prof: Profile,
    device: int,
    trim: tuple[int, int] | None = None,
    *,
    max_windows: int = 3,
) -> list[dict]:
    """Locate likely code candidates for the most suspicious idle gaps."""
    gap_result = detect_idle_gaps(prof, device, trim, limit=12)
    gap_rows = gap_result["rows"]
    if not gap_rows:
        return []

    preferred = [
        row for row in gap_rows if (row.get("attribution") or {}).get("category") in {"synchronization", "memory_transfer"}
    ]
    windows = preferred[:max_windows] if preferred else gap_rows[:max_windows]

    results = []
    for row in windows:
        start_ns = int(row["start_ns"])
        end_ns = int(row["end_ns"])
        results.append(
            {
                "stream": int(row["streamId"]),
                "start_ns": start_ns,
                "end_ns": end_ns,
                "gap_ms": round((end_ns - start_ns) / 1e6, 3),
                "before_kernel": row.get("before_kernel") or "",
                "after_kernel": row.get("after_kernel") or "",
                "attribution": row.get("attribution") or {},
                "threads": top_code_candidates(prof, start_ns, end_ns),
            }
        )
    return results


def format_code_locations(windows: list[dict]) -> str:
    """Format code-location candidates as readable text."""
    if not windows:
        return "(No code-location candidates found)"

    lines = ["Code Location Candidates"]
    for index, window in enumerate(windows, start=1):
        attr = window.get("attribution") or {}
        lines.append(
            f"- Window {index}: stream {window['stream']}, gap={window['gap_ms']:.2f}ms, "
            f"after {window['before_kernel']} [{attr.get('category', 'unclassified')}]"
        )
        if attr.get("top_apis"):
            api_names = ", ".join(api["name"].split("_v")[0] for api in attr["top_apis"])
            lines.append(f"  Runtime overlap: {api_names}")
        for thread in window.get("threads", [])[:2]:
            thread_label = thread["thread_name"] or str(thread["global_tid"])
            lines.append(
                f"  Thread {thread_label} (tid={thread['global_tid']}, runtime={thread['total_runtime_ms']:.1f}ms, "
                f"top APIs: {', '.join(thread['top_apis']) or 'n/a'})"
            )
            if thread.get("nvtx_ranges"):
                labels = ", ".join((row["text"] or "(unnamed)") for row in thread["nvtx_ranges"][:2])
                lines.append(f"    NVTX: {labels}")
            for frame in thread["frames"][:2]:
                lines.append(
                    f"    - {frame['symbol']} :: {frame['module']} "
                    f"[samples={frame['samples']}]"
                )
                lines.append(f"      {frame['reason']}")
    return "\n".join(lines)
