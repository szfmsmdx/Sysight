"""Code-location helpers: map suspicious windows to callchains, samples, and likely call sites."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import PurePosixPath

from sysight.analysis.queries import detect_idle_gaps, runtime_activity_in_window
from sysight.profile import Profile

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

_GENERIC_NVTX_LABELS = {
    "",
    "(unnamed)",
    "Holding GIL",
    "Waiting for GIL",
    "Python Periodic Sampling",
    "GIL Trace",
}
_GENERIC_NVTX_PREFIXES = (
    "cublas",
    "cudnn",
    "nccl",
)
_PROJECT_PATH_HINTS = (
    "/trainer.runfiles/autocar/",
    "/common/python/",
    "/modules/",
)
_PROJECT_PATH_EXCLUDES = (
    "/site-packages/",
    "/dist-packages/",
    "/third_party/",
    "/external/",
    "/.local/lib/python",
    "/usr/lib/python",
    "/usr/local/lib/python",
)
_PROJECT_TOKEN_STOPWORDS = {
    "autocar",
    "base",
    "common",
    "module",
    "modules",
    "python",
    "pytorch",
    "pytorch2",
    "runfiles",
    "task",
    "tasks",
    "trainer",
    "training",
    "utils",
}
_LOW_SIGNAL_SYMBOL_FRAGMENTS = (
    "std::",
    "_rb_tree",
    "intrusive_ptr",
    "storageimpl::~",
    "tensorimpl::~",
    "cudacachingallocator",
    "_pyeval_",
    "_pyobject_",
    "pyobject_call",
    "method_vectorcall",
    "cfunction_call",
    "slot_tp_call",
)
_GENERIC_FUNCTION_LABELS = {
    "forward",
    "backward",
    "loss",
    "model.forward",
    "module.forward",
}
_GENERIC_FUNCTION_OWNERS = {
    "model",
    "module",
}


def _identifier_tokens(text: str) -> list[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    return [part.lower() for part in re.split(r"[^A-Za-z0-9]+", expanded) if part]


def _normalized_identifier(text: str) -> str:
    return "".join(_identifier_tokens(text))


def _is_generic_nvtx_label(text: str) -> bool:
    stripped = (text or "").strip()
    lowered = stripped.lower()
    if stripped in _GENERIC_NVTX_LABELS:
        return True
    return lowered.startswith(_GENERIC_NVTX_PREFIXES)


def _score_nvtx_label(text: str) -> tuple[int, int]:
    stripped = (text or "").strip()
    if not stripped:
        return (-100, 0)
    lowered = stripped.lower()
    if _is_generic_nvtx_label(stripped):
        return (-100, 0)

    score = 0
    if " > " in stripped:
        score += 100
    if lowered.endswith(".forward") or lowered.endswith(".backward"):
        score += 90
    elif lowered in {"forward", "backward", "loss"}:
        score += 20
    if lowered.startswith("iter_"):
        score += 10
    if any(
        token in lowered
        for token in (
            "model",
            "decoder",
            "encoder",
            "attention",
            "transformer",
            "layer",
            "query",
            "route",
            "trajectory",
            "collision",
            "head",
            "loss",
        )
    ):
        score += 35
    if any(char.isupper() for char in stripped):
        score += 10
    return (score, len(stripped))


def _extract_python_path(value: str) -> str | None:
    if not value:
        return None
    match = re.search(r"(/[^:\s]+\.py)(?::\d+)?$", value.strip())
    return match.group(1) if match else None


def _is_project_python_path(path: str) -> bool:
    lowered = path.lower()
    if not lowered.endswith(".py"):
        return False
    if any(fragment in lowered for fragment in _PROJECT_PATH_EXCLUDES):
        return False
    return any(fragment in path for fragment in _PROJECT_PATH_HINTS)


def _display_project_path(path: str) -> str:
    marker = "/trainer.runfiles/autocar/"
    if marker in path:
        return "autocar/" + path.split(marker, 1)[1]
    return path


def _project_python_files(prof: Profile) -> list[dict]:
    cached = getattr(prof, "_project_python_files_cache", None)
    if cached is not None:
        return cached

    with prof._lock:
        rows = prof.conn.execute(
            """
            SELECT DISTINCT value
            FROM StringIds
            WHERE length(value) < 400
              AND value LIKE '%.py%'
            """
        ).fetchall()

    project_entries: list[dict] = []
    fallback_entries: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        raw = row[0] if row and row[0] is not None else ""
        path = _extract_python_path(str(raw))
        if not path or path in seen:
            continue
        seen.add(path)
        path_obj = PurePosixPath(path)
        path_tokens = {
            token
            for token in _identifier_tokens("/".join(path_obj.parts[-8:]))
            if token not in _PROJECT_TOKEN_STOPWORDS
        }
        stem_tokens = {
            token for token in _identifier_tokens(path_obj.stem) if token not in _PROJECT_TOKEN_STOPWORDS
        }
        entry = {
            "path": path,
            "display_path": _display_project_path(path),
            "stem": path_obj.stem,
            "stem_norm": _normalized_identifier(path_obj.stem),
            "stem_tokens": stem_tokens,
            "path_tokens": path_tokens,
            "path_norm": _normalized_identifier(path),
        }
        if _is_project_python_path(path):
            project_entries.append(entry)
        else:
            fallback_entries.append(entry)

    selected = project_entries or fallback_entries
    setattr(prof, "_project_python_files_cache", selected)
    return selected


def _window_context_texts(window: dict) -> list[str]:
    texts = [
        window.get("label") or "",
        window.get("before_kernel") or "",
        window.get("after_kernel") or "",
        ((window.get("attribution") or {}).get("description") or ""),
    ]
    for row in window.get("nvtx_ranges", []):
        texts.append(row.get("text") or "")
    for thread in window.get("threads", []):
        texts.extend(thread.get("top_apis") or [])
        for row in thread.get("nvtx_ranges", []):
            texts.append(row.get("text") or "")
        for frame in thread.get("frames", []):
            texts.append(frame.get("symbol") or "")
    return [text for text in texts if text]


def _score_project_file_candidate(entry: dict, texts: list[str], *, category: str) -> int:
    score = 0
    for text in texts:
        tokens = {token for token in _identifier_tokens(text) if token not in _PROJECT_TOKEN_STOPWORDS}
        norm = _normalized_identifier(text)
        if not tokens and not norm:
            continue
        if entry["stem_norm"] and entry["stem_norm"] in norm:
            score += 40
        if entry["stem_tokens"] and entry["stem_tokens"].issubset(tokens):
            score += 24
        score += min(len(entry["stem_tokens"] & tokens), 3) * 8
        score += min(len(entry["path_tokens"] & tokens), 4) * 4

    stem_tokens = entry["stem_tokens"]
    if category == "memory_transfer" and stem_tokens & {"memory", "prefetch", "buffer"}:
        score += 8
    if category == "communication" and stem_tokens & {"trainer", "model", "decoder"}:
        score += 4
    if category == "iteration" and stem_tokens & {"trainer", "model"}:
        score += 4
    return score


def infer_project_files_for_window(prof: Profile, window: dict, *, limit: int = 3) -> list[dict]:
    texts = [text for text in _window_context_texts(window) if not _is_generic_nvtx_label(text)]
    if not texts:
        texts = _window_context_texts(window)
    category = ((window.get("attribution") or {}).get("category") or "").strip()

    ranked = []
    for entry in _project_python_files(prof):
        score = _score_project_file_candidate(entry, texts, category=category)
        if score <= 0:
            continue
        ranked.append(
            {
                "path": entry["path"],
                "display_path": entry["display_path"],
                "score": score,
            }
        )

    ranked.sort(key=lambda row: (-row["score"], len(row["display_path"])))
    deduped: list[dict] = []
    seen_paths: set[str] = set()
    for row in ranked:
        if row["path"] in seen_paths:
            continue
        seen_paths.add(row["path"])
        deduped.append(row)
        if len(deduped) >= limit:
            break
    return deduped


def _frame_text(frame: dict) -> str:
    return f"{frame.get('symbol') or ''} {frame.get('module') or ''}".strip().lower()


def _is_low_signal_frame(symbol: str, module: str) -> bool:
    text = f"{symbol} {module}".lower()
    return any(fragment in text for fragment in _LOW_SIGNAL_SYMBOL_FRAGMENTS)


def _semantic_stack_frames(frames: list[dict], *, limit: int = 5) -> list[dict]:
    semantic: list[dict] = []
    fallback: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for frame in frames:
        symbol = frame.get("symbol") or ""
        module = frame.get("module") or ""
        if _is_noise_frame(symbol, module):
            continue
        key = (symbol, module)
        if key in seen:
            continue
        seen.add(key)
        fallback.append(frame)
        if _is_low_signal_frame(symbol, module):
            continue
        semantic.append(frame)
        if len(semantic) >= limit:
            break
    return semantic or fallback[:limit]


def _short_symbol_name(symbol: str) -> str:
    value = (symbol or "").strip()
    if not value:
        return ""
    cleaned = value
    for marker in ("(_object*", "(", "<"):
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0]
    if "::" in cleaned:
        parts = [part for part in cleaned.split("::") if part]
        if parts:
            tail = parts[-1]
            if tail in {"call", "execute", "wait", "run", "pop"} and len(parts) >= 2:
                cleaned = "::".join(parts[-2:])
            else:
                cleaned = tail
    return cleaned or value


def _format_stack_preview(frames: list[dict], *, limit: int = 4) -> str:
    preview_frames = _semantic_stack_frames(frames, limit=limit)
    labels = [_short_symbol_name(frame.get("symbol") or "") for frame in preview_frames]
    labels = [label for label in labels if label]
    return " -> ".join(labels[:limit])


def _stack_signal_score(frames: list[dict], *, source: str, samples: int = 0, total_ms: float = 0.0) -> int:
    source_bonus = {
        "cuda_callchain": 60,
        "osrt_callchain": 40,
        "sampling": 20,
    }.get(source, 0)
    preview_text = " ".join(_frame_text(frame) for frame in _semantic_stack_frames(frames, limit=6))
    signal_bonus = 0
    if any(
        token in preview_text
        for token in (
            "synchronize",
            "backward",
            "autograd",
            "copy",
            "memcpy",
            "launch",
            "triton",
            "dynamo",
            "layer_norm",
            "nccl",
        )
    ):
        signal_bonus += 20
    if "torch" in preview_text or "pythonengine" in preview_text or "thp" in preview_text:
        signal_bonus += 8
    count_bonus = min(samples, 24)
    dur_bonus = min(int(total_ms), 24)
    return source_bonus + signal_bonus + count_bonus + dur_bonus


def _stack_signature(frames: list[dict], *, limit: int = 6) -> tuple[tuple[str, str], ...]:
    return tuple(
        (frame.get("symbol") or "", frame.get("module") or "")
        for frame in _semantic_stack_frames(frames, limit=limit)
    )


def _build_stack_candidate(
    frames: list[dict],
    *,
    source: str,
    samples: int = 0,
    total_ms: float = 0.0,
    call_count: int = 0,
    api_names: list[str] | None = None,
) -> dict | None:
    semantic_frames = _semantic_stack_frames(frames, limit=6)
    if not semantic_frames:
        return None
    return {
        "source": source,
        "samples": int(samples),
        "call_count": int(call_count),
        "total_ms": round(float(total_ms), 3),
        "api_names": api_names or [],
        "frames": semantic_frames,
        "preview": _format_stack_preview(semantic_frames),
        "score": _stack_signal_score(
            semantic_frames,
            source=source,
            samples=max(samples, call_count),
            total_ms=total_ms,
        ),
    }


def _parse_function_label(text: str) -> dict | None:
    label = (text or "").strip()
    if not label or _is_generic_nvtx_label(label):
        return None
    lowered = label.lower()
    if lowered in _GENERIC_FUNCTION_LABELS or lowered.startswith("iter_"):
        return None
    if "." not in label and "::" not in label and not any(char.isupper() for char in label):
        return None

    owner = ""
    method = ""
    if "." in label:
        owner, method = label.rsplit(".", 1)
    elif "::" in label:
        owner, method = label.rsplit("::", 1)
    else:
        owner = label

    owner = owner.strip()
    method = method.strip()
    owner_tokens = [token for token in _identifier_tokens(owner) if token not in _PROJECT_TOKEN_STOPWORDS]
    if owner.lower() in _GENERIC_FUNCTION_OWNERS:
        return None

    score = _score_nvtx_label(label)[0]
    if method in {"forward", "backward"}:
        score += 30
    if owner and owner[0].isupper():
        score += 8
    if len(owner_tokens) >= 3:
        score += 10
    if owner.lower().startswith("wrapped"):
        score -= 6

    return {
        "label": label,
        "owner": owner,
        "method": method,
        "owner_tokens": set(owner_tokens),
        "owner_norm": _normalized_identifier(owner),
        "score": score,
    }


def _window_function_candidates(window: dict) -> list[dict]:
    candidates: list[dict] = []
    seen: set[str] = set()
    texts: list[str] = []
    for row in window.get("nvtx_ranges", []):
        texts.append(row.get("text") or "")
    for thread in window.get("threads", []):
        for row in thread.get("nvtx_ranges", []):
            texts.append(row.get("text") or "")
    for text in texts:
        for part in str(text).split(" > "):
            parsed = _parse_function_label(part)
            if not parsed:
                continue
            key = parsed["label"]
            if key in seen:
                continue
            seen.add(key)
            candidates.append(parsed)
    candidates.sort(key=lambda row: (-row["score"], row["label"]))
    return candidates


def _score_function_file_candidate(entry: dict, candidate: dict, *, base_score: int = 0) -> int:
    score = base_score + int(candidate.get("score") or 0)
    owner_norm = candidate.get("owner_norm") or ""
    owner_tokens = candidate.get("owner_tokens") or set()
    if owner_norm:
        if entry["stem_norm"] and entry["stem_norm"] in owner_norm:
            score += 120
        if entry["stem_norm"] and owner_norm in entry["stem_norm"]:
            score += 60
        if entry["path_norm"] and owner_norm in entry["path_norm"]:
            score += 24
    if owner_tokens:
        score += min(len(entry["stem_tokens"] & owner_tokens), 4) * 16
        score += min(len(entry["path_tokens"] & owner_tokens), 6) * 8
        if entry["stem_tokens"] and entry["stem_tokens"].issubset(owner_tokens):
            score += 20
    if candidate.get("method") in {"forward", "backward"}:
        score += 10
    return score


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
            r.callchainId,
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


def osrt_calls_in_window(
    prof: Profile,
    start_ns: int,
    end_ns: int,
    *,
    global_tid: int | None = None,
    limit: int = 20,
) -> list[dict]:
    """Return OS runtime calls overlapping a time window."""
    table = prof.schema.osrt_api_table
    if not table:
        return []
    sql = f"""
        SELECT
            o.globalTid,
            s.value AS api_name,
            o.callchainId,
            o.start,
            o.[end],
            ROUND((o.[end] - o.start) / 1e6, 3) AS dur_ms
        FROM {table} o
        JOIN StringIds s ON o.nameId = s.id
        WHERE o.start < ? AND o.[end] > ?
    """
    params: list[object] = [end_ns, start_ns]
    if global_tid is not None:
        sql += " AND o.globalTid = ?"
        params.append(global_tid)
    sql += " ORDER BY (o.[end] - o.start) DESC LIMIT ?"
    params.append(limit)
    with prof._lock:
        return [dict(row) for row in prof.conn.execute(sql, params).fetchall()]


def callchain_frames(
    prof: Profile,
    table: str | None,
    callchain_id: int | None,
    *,
    source: str,
    limit: int = 24,
) -> list[dict]:
    """Resolve one callchain into stack frames."""
    if not table or not callchain_id or callchain_id <= 0:
        return []
    sql = f"""
        SELECT
            oc.stackDepth,
            ss.value AS symbol,
            ms.value AS module
        FROM {table} oc
        LEFT JOIN StringIds ss ON oc.symbol = ss.id
        LEFT JOIN StringIds ms ON oc.module = ms.id
        WHERE oc.id = ?
        ORDER BY oc.stackDepth
        LIMIT ?
    """
    with prof._lock:
        rows = [dict(row) for row in prof.conn.execute(sql, (callchain_id, limit)).fetchall()]
    frames = []
    for row in rows:
        frames.append(
            {
                "symbol": row.get("symbol"),
                "module": row.get("module"),
                "samples": 1,
                "min_depth": row.get("stackDepth") or 0,
                "max_depth": row.get("stackDepth") or 0,
                "source": source,
            }
        )
    return frames


def api_callstacks(
    prof: Profile,
    apis: list[dict],
    table: str | None,
    *,
    source: str,
    limit: int = 3,
) -> list[dict]:
    """Resolve representative API callchains for one thread/window."""
    if not table:
        return []

    grouped: dict[int, dict] = {}
    for api in apis:
        callchain_id = int(api.get("callchainId") or 0)
        if callchain_id <= 0:
            continue
        bucket = grouped.setdefault(
            callchain_id,
            {
                "call_count": 0,
                "total_ms": 0.0,
                "api_counter": Counter(),
            },
        )
        bucket["call_count"] += 1
        bucket["total_ms"] += float(api.get("dur_ms") or 0.0)
        api_name = (api.get("api_name") or "").split("_v")[0]
        if api_name:
            bucket["api_counter"][api_name] += 1

    stacks: list[dict] = []
    for callchain_id, bucket in grouped.items():
        frames = callchain_frames(prof, table, callchain_id, source=source, limit=24)
        stack = _build_stack_candidate(
            frames,
            source=source,
            call_count=bucket["call_count"],
            total_ms=bucket["total_ms"],
            api_names=list(bucket["api_counter"]),
        )
        if not stack:
            continue
        stacks.append(stack)

    stacks.sort(key=lambda row: (row["score"], row["total_ms"], row["call_count"]), reverse=True)
    return stacks[:limit]


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
    sql += " ORDER BY start ASC LIMIT ?"
    params.append(max(limit * 8, 32))
    with prof._lock:
        rows = [dict(row) for row in prof.conn.execute(sql, params).fetchall()]

    rows.sort(
        key=lambda row: (
            -_score_nvtx_label((row.get("text") or "").strip())[0],
            -(row.get("dur_ms") or 0),
            row.get("start") or 0,
        )
    )
    selected: list[dict] = []
    seen_texts: set[str] = set()
    for row in rows:
        text = (row.get("text") or "").strip() or "(unnamed)"
        if text in seen_texts:
            continue
        seen_texts.add(text)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def candidate_threads_for_window(
    prof: Profile, start_ns: int, end_ns: int, *, top_n: int = 8
) -> list[dict]:
    """Return runtime-active threads in a window, ordered by total runtime duration."""
    rows: list[dict] = []
    if prof.schema.runtime_table:
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
    if not rows and prof.schema.osrt_api_table:
        sql = f"""
            SELECT
                o.globalTid,
                COUNT(*) AS api_count,
                SUM(o.[end] - o.start) AS total_ns
            FROM {prof.schema.osrt_api_table} o
            WHERE o.start < ? AND o.[end] > ?
            GROUP BY o.globalTid
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


def sample_callstacks_for_thread(
    prof: Profile,
    global_tid: int,
    start_ns: int,
    end_ns: int,
    *,
    max_samples: int = 256,
    limit: int = 3,
) -> list[dict]:
    """Build representative sampled call stacks for one thread in a window."""
    if "COMPOSITE_EVENTS" not in prof.schema.tables or "SAMPLING_CALLCHAINS" not in prof.schema.tables:
        return []

    sql = """
        WITH sample_ids AS (
            SELECT id, start
            FROM COMPOSITE_EVENTS
            WHERE globalTid = ?
              AND start BETWEEN ? AND ?
            ORDER BY start
            LIMIT ?
        )
        SELECT
            s.id,
            s.start AS sample_start_ns,
            sc.stackDepth,
            ss.value AS symbol,
            ms.value AS module
        FROM sample_ids s
        JOIN SAMPLING_CALLCHAINS sc ON s.id = sc.id
        LEFT JOIN StringIds ss ON sc.symbol = ss.id
        LEFT JOIN StringIds ms ON sc.module = ms.id
        ORDER BY s.start, sc.stackDepth
    """
    with prof._lock:
        rows = [dict(row) for row in prof.conn.execute(sql, (global_tid, start_ns, end_ns, max_samples)).fetchall()]

    frames_by_sample: dict[int, list[dict]] = {}
    for row in rows:
        frames_by_sample.setdefault(int(row["id"]), []).append(
            {
                "symbol": row.get("symbol"),
                "module": row.get("module"),
                "samples": 1,
                "min_depth": row.get("stackDepth") or 0,
                "max_depth": row.get("stackDepth") or 0,
                "source": "sampling",
            }
        )

    grouped: dict[tuple[tuple[str, str], ...], dict] = {}
    for frames in frames_by_sample.values():
        stack = _build_stack_candidate(frames, source="sampling", samples=1)
        if not stack:
            continue
        signature = _stack_signature(stack["frames"])
        bucket = grouped.setdefault(
            signature,
            {
                "source": "sampling",
                "samples": 0,
                "call_count": 0,
                "total_ms": 0.0,
                "api_names": [],
                "frames": stack["frames"],
                "preview": stack["preview"],
                "score": stack["score"],
            },
        )
        bucket["samples"] += 1
        bucket["score"] = _stack_signal_score(
            bucket["frames"],
            source="sampling",
            samples=bucket["samples"],
        )

    stacks = list(grouped.values())
    stacks.sort(key=lambda row: (row["score"], row["samples"]), reverse=True)
    return stacks[:limit]


def infer_python_locations_for_window(prof: Profile, window: dict, *, limit: int = 3) -> list[dict]:
    """Infer project-level Python file/function candidates for a suspicious window."""
    base_project_rows = window.get("project_files") or infer_project_files_for_window(prof, window, limit=6)
    base_scores = {row["path"]: int(row.get("score") or 0) for row in base_project_rows}
    base_entries = {entry["path"]: entry for entry in _project_python_files(prof)}
    function_candidates = _window_function_candidates(window)

    ranked: list[dict] = []
    for candidate in function_candidates:
        for entry in base_entries.values():
            score = _score_function_file_candidate(
                entry,
                candidate,
                base_score=base_scores.get(entry["path"], 0),
            )
            if score <= 0:
                continue
            ranked.append(
                {
                    "path": entry["path"],
                    "display_path": entry["display_path"],
                    "function": candidate["label"],
                    "score": score,
                    "evidence": "nvtx",
                }
            )

    if not ranked:
        best_function = function_candidates[0]["label"] if function_candidates else ""
        for row in base_project_rows[:limit]:
            ranked.append(
                {
                    "path": row["path"],
                    "display_path": row["display_path"],
                    "function": best_function,
                    "score": int(row.get("score") or 0),
                    "evidence": "context",
                }
            )

    ranked.sort(
        key=lambda row: (
            -(row["score"]),
            0 if row.get("function") else 1,
            len(row["display_path"]),
        )
    )
    deduped: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in ranked:
        key = (row["path"], row.get("function") or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= limit:
            break
    return deduped


def classify_code_candidate(symbol: str, module: str, apis: list[dict]) -> str:
    """Give a short explanation for why a frame is interesting."""
    text = f"{symbol} {module}".lower()
    api_names = " ".join((api["api_name"] or "").lower() for api in apis)
    if "_to_copy" in text or "copy_" in text or "copy_impl" in text:
        if "synchronize" in api_names:
            return "很像张量拷贝路径，且后续被同步阻塞；优先检查 .to(...)、.cpu() 和隐式主机/设备拷贝。"
        return "很像张量拷贝路径；优先检查 .to(...)、.cpu()、non_blocking 传输和 pinned memory。"
    if "thpfunction_apply" in text:
        return "命中 PyTorch autograd apply 路径；优先检查这个时间窗附近对应的 backward/autograd 步骤。"
    if "dynamo" in text or "triton" in text or "launch" in text:
        return "命中编译器或 Triton 启动路径；优先检查编译图边界和 launcher 开销。"
    if "synchronize" in api_names:
        return "这个时间窗主要被同步 API 主导；优先检查显式 wait、barrier 和热点路径中的 .item() 调用。"
    return "这是可疑时间窗里被频繁采样到的代表性栈帧。"


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
    source_bonus = {
        "cuda_callchain": 30,
        "osrt_callchain": 18,
        "sampling": 0,
    }.get(frame.get("source"), 0)
    bonus = source_bonus + (10 if _is_interesting_frame(frame["symbol"], frame["module"]) else 0)
    prefer_shallower = -int(frame.get("min_depth") or 0)
    return (int(frame.get("samples") or 0) + bonus, bonus, prefer_shallower)


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
        osrt_apis = osrt_calls_in_window(prof, start_ns, end_ns, global_tid=tid, limit=8)
        nvtx_ranges = nvtx_ranges_in_window(prof, start_ns, end_ns, global_tid=tid, limit=4)
        stack_candidates: list[dict] = []
        stack_candidates.extend(
            api_callstacks(
                prof,
                apis,
                prof.schema.cuda_callchain_table,
                source="cuda_callchain",
                limit=2,
            )
        )
        stack_candidates.extend(
            api_callstacks(
                prof,
                osrt_apis,
                prof.schema.osrt_callchain_table,
                source="osrt_callchain",
                limit=2,
            )
        )
        stack_candidates.extend(sample_callstacks_for_thread(prof, tid, start_ns, end_ns, limit=2))
        deduped_stacks: list[dict] = []
        seen_stack_previews: set[str] = set()
        for stack in sorted(
            stack_candidates,
            key=lambda row: (row["score"], row.get("samples", 0), row.get("total_ms", 0.0)),
            reverse=True,
        ):
            preview = stack.get("preview") or ""
            if preview in seen_stack_previews:
                continue
            seen_stack_previews.add(preview)
            deduped_stacks.append(stack)
            if len(deduped_stacks) >= 3:
                break
        frames = []
        for api in apis:
            frames.extend(
                callchain_frames(
                    prof,
                    prof.schema.cuda_callchain_table,
                    api.get("callchainId"),
                    source="cuda_callchain",
                    limit=16,
                )
            )
            if frames:
                break
        if not frames:
            for api in osrt_apis:
                frames.extend(
                    callchain_frames(
                        prof,
                        prof.schema.osrt_callchain_table,
                        api.get("callchainId"),
                        source="osrt_callchain",
                        limit=16,
                    )
                )
                if frames:
                    break
        sample_frames = sample_hotspots_for_thread(prof, tid, start_ns, end_ns, limit=60)
        for frame in sample_frames:
            frame = dict(frame)
            frame["source"] = "sampling"
            frames.append(frame)
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
            frame["reason"] = classify_code_candidate(frame["symbol"], frame["module"], apis + osrt_apis)
            selected.append(frame)
            if len(selected) >= max_frames_per_thread:
                break

        if not selected:
            continue

        api_counter = Counter((api["api_name"] or "").split("_v")[0] for api in apis + osrt_apis)
        candidates.append(
            {
                "global_tid": tid,
                "thread_name": thread.get("thread_name") or "",
                "api_count": int(thread.get("api_count") or 0),
                "total_runtime_ms": round((thread.get("total_ns") or 0) / 1e6, 3),
                "top_apis": [name for name, _ in api_counter.most_common(3)],
                "nvtx_ranges": nvtx_ranges,
                "stacks": deduped_stacks,
                "frames": selected,
            }
        )
    return candidates


def locate_code_for_windows(
    prof: Profile,
    windows: list[dict],
    *,
    max_windows: int | None = None,
) -> list[dict]:
    """Attach code-location context to arbitrary suspicious windows."""
    selected_windows = windows[:max_windows] if max_windows is not None else windows
    results = []
    for window in selected_windows:
        start_ns = int(window["start_ns"])
        end_ns = int(window["end_ns"])
        enriched = dict(window)
        enriched.setdefault("duration_ms", round((end_ns - start_ns) / 1e6, 3))
        enriched["attribution"] = enriched.get("attribution") or runtime_activity_in_window(prof, start_ns, end_ns)
        enriched["nvtx_ranges"] = nvtx_ranges_in_window(prof, start_ns, end_ns, limit=6)
        enriched["threads"] = top_code_candidates(prof, start_ns, end_ns)
        enriched["project_files"] = infer_project_files_for_window(prof, enriched)
        enriched["python_locations"] = infer_python_locations_for_window(prof, enriched)
        results.append(enriched)
    return results


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
        results.append(
            {
                "kind": "idle_gap",
                "stream": int(row["streamId"]),
                "start_ns": int(row["start_ns"]),
                "end_ns": int(row["end_ns"]),
                "gap_ms": round((int(row["end_ns"]) - int(row["start_ns"])) / 1e6, 3),
                "before_kernel": row.get("before_kernel") or "",
                "after_kernel": row.get("after_kernel") or "",
                "attribution": row.get("attribution") or {},
            }
        )
    return locate_code_for_windows(prof, results)


def format_code_locations(windows: list[dict]) -> str:
    """Format code-location candidates as readable text."""
    if not windows:
        return "（未找到可用的代码定位候选）"

    lines = ["代码定位候选"]
    for index, window in enumerate(windows, start=1):
        attr = window.get("attribution") or {}
        category = attr.get("category", "unclassified")
        lines.append(
            f"- 窗口 {index}: stream {window['stream']}, gap={window['gap_ms']:.2f}ms, "
            f"位于 {window['before_kernel']} 之后 [{category}]"
        )
        if window.get("python_locations"):
            locations = []
            for row in window["python_locations"][:2]:
                label = row["display_path"]
                if row.get("function"):
                    label += f" :: {row['function']}"
                locations.append(label)
            lines.append(f"  Python 位置: {', '.join(locations)}")
        if window.get("project_files"):
            candidates = ", ".join(row["display_path"] for row in window["project_files"][:2])
            lines.append(f"  项目文件候选: {candidates}")
        if window.get("nvtx_ranges"):
            labels = ", ".join((row["text"] or "(unnamed)") for row in window["nvtx_ranges"][:3])
            lines.append(f"  窗口 NVTX: {labels}")
        if attr.get("top_apis"):
            api_names = ", ".join(api["name"].split("_v")[0] for api in attr["top_apis"])
            lines.append(f"  重叠 Runtime API: {api_names}")
        for thread in window.get("threads", [])[:2]:
            thread_label = thread["thread_name"] or str(thread["global_tid"])
            lines.append(
                f"  线程 {thread_label} (tid={thread['global_tid']}, runtime={thread['total_runtime_ms']:.1f}ms, "
                f"top APIs: {', '.join(thread['top_apis']) or 'n/a'})"
            )
            if thread.get("nvtx_ranges"):
                labels = ", ".join((row["text"] or "(unnamed)") for row in thread["nvtx_ranges"][:2])
                lines.append(f"    NVTX: {labels}")
            for stack in thread.get("stacks", [])[:2]:
                preview = stack.get("preview") or ""
                if not preview:
                    continue
                meta = []
                if stack.get("source"):
                    meta.append(stack["source"])
                if stack.get("samples"):
                    meta.append(f"samples={stack['samples']}")
                elif stack.get("call_count"):
                    meta.append(f"calls={stack['call_count']}")
                if stack.get("total_ms"):
                    meta.append(f"dur={stack['total_ms']:.1f}ms")
                suffix = f" [{', '.join(meta)}]" if meta else ""
                lines.append(f"    调用栈: {preview}{suffix}")
            for frame in thread["frames"][:2]:
                lines.append(
                    f"    - {frame['symbol']} :: {frame['module']} "
                    f"[samples={frame['samples']}]"
                )
                lines.append(f"      {frame['reason']}")
    return "\n".join(lines)
