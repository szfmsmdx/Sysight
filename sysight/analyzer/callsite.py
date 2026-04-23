"""Callsite scope, search, and source-context helpers.

This module isolates finding-aware callsite localization from the broader
repo-analysis pipeline so the static analyzer core can stay smaller and more
focused.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .scanners import CallSiteFacts, FileFacts

# File/function name keywords that indicate training-loop relevance.
# Used by derive_analysis_scope to select priority files.
_SCOPE_PATTERNS: dict[str, dict] = {
    "gpu_memcpy_hotspot": {
        "file_keywords": {"train", "trainer", "data", "loader", "dataset",
                          "collate", "forward", "step", "memory", "prefetch"},
        "func_keywords": {"train_step", "forward", "collate_fn", "step",
                          "__getitem__", "on_train_batch", "to_device",
                          "prefetch_pin_memory"},
        "call_names": ["to", "cuda", "copy_", "pin_memory"],
        "call_keywords": ["device", "cuda", "non_blocking"],
    },
    "gpu_idle": {
        "file_keywords": {"train", "trainer", "data", "loader", "schedule"},
        "func_keywords": {"train_step", "train_iter", "train_one_iter",
                          "forward", "step", "__iter__"},
        "call_names": ["synchronize", "barrier", "wait"],
        "call_keywords": [],
    },
    "gpu_comm_hotspot": {
        "file_keywords": {"train", "trainer", "ddp", "dist", "parallel"},
        "func_keywords": {"backward", "train_step", "train_iter", "iter_impl",
                          "reduce", "all_reduce"},
        "call_names": ["all_reduce", "barrier", "backward", "reduce_scatter",
                       "allgather", "no_sync"],
        "call_keywords": ["async_op"],
    },
    "sync_wait": {
        "file_keywords": {"train", "trainer", "model", "forward"},
        "func_keywords": {"train_step", "train_iter", "train_one_iter",
                          "iter_impl", "forward", "step"},
        "call_names": ["synchronize", "item", "wait", "cpu"],
        "call_keywords": [],
    },
    "gpu_memcpy_h2d": {
        "file_keywords": {"train", "trainer", "data", "loader", "dataset",
                          "collate", "forward", "step", "memory", "prefetch"},
        "func_keywords": {"train_step", "forward", "collate_fn", "step",
                          "__getitem__", "on_train_batch", "to_device",
                          "prefetch_pin_memory"},
        "call_names": ["to", "cuda", "copy_", "pin_memory"],
        "call_keywords": ["device", "cuda", "non_blocking"],
    },
    "gpu_memcpy_d2h": {
        "file_keywords": {"train", "trainer", "log", "metric", "eval"},
        "func_keywords": {"train_step", "forward", "log", "evaluate"},
        "call_names": ["cpu", "numpy", "item", "tolist"],
        "call_keywords": [],
    },
}
_DEFAULT_SCOPE_PATTERN = {
    "file_keywords": {"train", "trainer"},
    "func_keywords": set(),
    "call_names": [],
    "call_keywords": [],
}
_TRAIN_ITER_FRAGMENTS = frozenset({
    "train_one_iter", "train_iter_impl", "iter_impl", "train_iter",
    "train_step", "train_batch", "forward_backward", "step",
})


@dataclass
class AnalysisScope:
    """Finding-aware scan scope. Drives callsite search — not a full-repo scan."""

    finding_type: str
    priority_file_patterns: list[str]
    priority_func_patterns: list[str]
    call_names: list[str]
    call_keywords: list[str]
    seed_files: list[str]
    selected_files: list[str]
    excluded_patterns: list[str]
    reason: list[str]
    max_files: int = 50
    max_candidates: int = 100


@dataclass
class CallSiteCandidate:
    """A scored callsite returned by search_calls."""

    id: str
    path: str
    line: int
    call_name: str
    full_call_name: str | None
    receiver: str | None
    keywords: dict[str, str]
    loop_depth: int
    enclosing_function: str | None
    source_line: str
    score: float


@dataclass
class CallsiteContext:
    """Source context around a callsite, for LLM consumption."""

    callsite_id: str
    path: str
    line: int
    enclosing_function: str | None
    source_snippet: str
    callers: list[str]


def derive_analysis_scope(
    finding_type: str,
    files: dict[str, FileFacts],
    max_files: int = 50,
) -> AnalysisScope:
    """Build a finding-aware AnalysisScope from the scanned file set."""
    pattern = _SCOPE_PATTERNS.get(finding_type, _DEFAULT_SCOPE_PATTERN)
    file_kw: set[str] = pattern["file_keywords"]
    call_names: list[str] = pattern["call_names"]
    call_keywords: list[str] = pattern["call_keywords"]

    selected: list[str] = []
    reason: list[str] = []
    for rel, facts in files.items():
        if facts.language != "python":
            continue
        parts_for_match = [
            part for part in Path(rel.replace("\\", "/")).parts
            if not part.lower().endswith(".runfiles")
        ]
        low = "/".join(parts_for_match).lower()
        parts = set(low.replace("/", " ").replace("_", " ").replace(".", " ").split())
        if parts & file_kw:
            selected.append(rel)
            reason.append(f"keyword match: {rel}")

    if not selected:
        selected = [r for r, f in files.items() if f.language == "python"]
        reason.append("fallback: all python files")

    selected = selected[:max_files]
    return AnalysisScope(
        finding_type=finding_type,
        priority_file_patterns=sorted(file_kw),
        priority_func_patterns=sorted(pattern["func_keywords"]),
        call_names=call_names,
        call_keywords=call_keywords,
        seed_files=selected[:],
        selected_files=selected,
        excluded_patterns=[],
        reason=reason,
        max_files=max_files,
        max_candidates=100,
    )


def build_callsite_index(
    files: dict[str, FileFacts],
) -> dict[str, list[CallSiteFacts]]:
    """Build call_name -> [CallSiteFacts, ...] index from all scanned files."""
    index: dict[str, list[CallSiteFacts]] = defaultdict(list)
    for facts in files.values():
        for cs in facts.callsites:
            index[cs.call_name].append(cs)
    return dict(index)


def _score_callsite(
    cs: CallSiteFacts,
    *,
    in_scope: bool,
    func_patterns: list[str],
    search_kw: set[str],
) -> float:
    score = 0.0
    if in_scope:
        score += 3.0
    if cs.loop_depth >= 1:
        score += min(2.0, cs.loop_depth * 1.0)

    fn_name = (cs.enclosing_function or "").lower()
    if fn_name:
        if any(p in fn_name for p in func_patterns):
            score += 2.0
        if any(frag in fn_name for frag in _TRAIN_ITER_FRAGMENTS):
            score += 3.0

    call_text = " ".join(
        cs.args_repr + list(cs.keywords.values()) + [cs.source_line, cs.full_call_name or cs.call_name]
    ).lower()
    for kw in search_kw:
        if kw in cs.keywords:
            score += 1.0
        if kw.lower() in call_text:
            score += 1.0
    return score


def _collect_candidates(
    index: dict[str, list[CallSiteFacts]],
    *,
    search_names: list[str],
    scope_set: set[str],
    func_patterns: list[str],
    search_kw: set[str],
    scope_only: bool,
) -> list[CallSiteCandidate]:
    candidates: list[CallSiteCandidate] = []
    for name in search_names:
        for cs in index.get(name, []):
            in_scope = cs.path in scope_set
            if scope_only and scope_set and not in_scope:
                continue
            candidates.append(CallSiteCandidate(
                id=cs.id,
                path=cs.path,
                line=cs.line,
                call_name=cs.call_name,
                full_call_name=cs.full_call_name,
                receiver=cs.receiver,
                keywords=cs.keywords,
                loop_depth=cs.loop_depth,
                enclosing_function=cs.enclosing_function,
                source_line=cs.source_line,
                score=_score_callsite(
                    cs,
                    in_scope=in_scope,
                    func_patterns=func_patterns,
                    search_kw=search_kw,
                ),
            ))
    return candidates


def search_calls(
    index: dict[str, list[CallSiteFacts]],
    scope: AnalysisScope,
    names: list[str] | None = None,
    keywords: list[str] | None = None,
    limit: int = 50,
) -> list[CallSiteCandidate]:
    """Search the callsite index and return ranked candidates."""
    search_names = names if names is not None else scope.call_names
    search_kw = set(keywords) if keywords else set(scope.call_keywords)
    scope_set = set(scope.selected_files)
    func_patterns = [p.lower() for p in scope.priority_func_patterns]

    candidates = _collect_candidates(
        index,
        search_names=search_names,
        scope_set=scope_set,
        func_patterns=func_patterns,
        search_kw=search_kw,
        scope_only=True,
    )
    if not candidates:
        candidates = _collect_candidates(
            index,
            search_names=search_names,
            scope_set=scope_set,
            func_patterns=func_patterns,
            search_kw=search_kw,
            scope_only=False,
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:limit]


def get_callsite_context(
    callsite_id: str,
    files: dict[str, FileFacts],
    repo_root: str | Path | None = None,
) -> CallsiteContext | None:
    """Return source context for a callsite id."""
    parts = callsite_id.split(":")
    if len(parts) < 4:
        return None
    cs_path = parts[0]
    try:
        cs_line = int(parts[1])
    except ValueError:
        return None

    facts = files.get(cs_path)
    if facts is None:
        return None
    cs = next((c for c in facts.callsites if c.id == callsite_id), None)
    if cs is None:
        return None

    enclosing_fn_facts = None
    if cs.enclosing_function:
        short_key = cs.enclosing_function.split("::")[-1] if "::" in cs.enclosing_function else cs.enclosing_function
        enclosing_fn_facts = facts.functions.get(short_key)

    snippet_lines: list[str] = []
    if enclosing_fn_facts and enclosing_fn_facts.end_line and repo_root:
        try:
            full_path = Path(repo_root) / cs_path
            src_lines = full_path.read_text(encoding="utf-8").splitlines()
            start = max(enclosing_fn_facts.line - 1, 0)
            end = min(enclosing_fn_facts.end_line, len(src_lines))
            snippet_lines = src_lines[start:end]
        except (OSError, UnicodeDecodeError):
            pass
    if not snippet_lines:
        snippet_lines = [cs.source_line]

    callers: list[str] = []
    if enclosing_fn_facts:
        target_short = enclosing_fn_facts.name
        for file_facts in files.values():
            for fn in file_facts.functions.values():
                if target_short in fn.calls:
                    callers.append(fn.qualified_name)

    return CallsiteContext(
        callsite_id=callsite_id,
        path=cs_path,
        line=cs_line,
        enclosing_function=cs.enclosing_function,
        source_snippet="\n".join(snippet_lines),
        callers=callers[:20],
    )


__all__ = [
    "AnalysisScope",
    "CallSiteCandidate",
    "CallsiteContext",
    "build_callsite_index",
    "derive_analysis_scope",
    "get_callsite_context",
    "search_calls",
]
