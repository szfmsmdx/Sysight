"""scanner_cli — Structured CLI tools for Agent/Codex code localization.

This module exposes the static analysis capabilities of analyzer.py as
individual CLI commands that output bounded, structured JSON suitable for
LLM consumption. Each command is read-only and deterministic.

Design principles:
  - Single responsibility: each command does ONE thing well
  - Bounded output: all results are capped to prevent context explosion
  - Structured JSON: all output is machine-parseable
  - Deterministic: same input always produces same output

Commands:
  scanner manifest <repo>           — Stage 1: path discovery (no file reads)
  scanner index <repo>              — Stage 2: parse files, build index
  scanner search <repo> <query>     — Symbol/file search
  scanner lookup <repo> [--file F] [--symbol S] [--line N]  — Precise location lookup
  scanner callers <repo> <symbol>   — Find callers of a symbol
  scanner callees <repo> <symbol>   — Find callees of a symbol
  scanner impact <repo> <files...>  — Impact radius analysis
  scanner trace <repo> <target>     — Call chain trace
  scanner callsites <repo> [--call NAME] [--file F]  — Callsite search
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .analyzer import (
    scan_repo,
    build_dag,
    build_repo_index,
    RepoIndex,
    RepoScope,
    FileFacts,
    FunctionFacts,
    FileMatch,
    LocationMatch,
    fuzzy_file_match,
    lookup_by_file_line,
    lookup_by_symbol,
    callers_of,
    callees_of,
    search_symbols,
    impact_radius,
    trace_from,
    FileDAG,
)
from .callsite import (
    AnalysisScope,
    CallSiteCandidate,
    CallsiteContext,
    build_callsite_index,
    derive_analysis_scope,
    search_calls,
    get_callsite_context,
)

logger = logging.getLogger(__name__)


# ── Result dataclasses (all JSON-serializable) ─────────────────────────────────

@dataclass
class SymbolInfo:
    """Compact symbol representation for JSON output."""
    qualified_name: str
    short_name: str
    file: str
    line: int
    end_line: int | None
    is_gpu_kernel: bool
    calls_count: int


@dataclass
class SearchResult:
    """Result of scanner search command."""
    query: str
    matches: list[dict[str, Any]]  # list of {path, symbol, kind, line, score}
    total_count: int


@dataclass
class LookupResult:
    """Result of scanner lookup command."""
    query_type: str  # "file_line" | "symbol" | "file"
    file: str | None
    function: str | None
    line: int | None
    confidence: float
    reason: str
    alternatives: list[str]
    source_snippet: str | None
    callers: list[str]
    callees: list[str]


@dataclass
class CallersResult:
    """Result of scanner callers command."""
    symbol: str
    callers: list[str]
    count: int


@dataclass
class CalleesResult:
    """Result of scanner callees command."""
    symbol: str
    callees: list[str]
    count: int


@dataclass
class ImpactResult:
    """Result of scanner impact command."""
    seed_files: list[str]
    impacted_files: list[str]
    impacted_symbols: list[str]
    depth_map: dict[str, int]
    truncated: bool


@dataclass
class TraceStep:
    """Single step in a call chain."""
    from_symbol: str
    to_symbol: str
    to_file: str
    depth: int


@dataclass
class TraceChain:
    """A single traced call chain."""
    entry_path: str
    entry_symbol: str
    visited_files: list[str]
    visited_symbols: list[str]
    steps: list[TraceStep]
    truncated: bool


@dataclass
class TraceResult:
    """Result of scanner trace command."""
    target: str
    chains: list[TraceChain]


@dataclass
class CallsiteMatch:
    """A single callsite match."""
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
class CallsitesResult:
    """Result of scanner callsites command."""
    call_name: str | None
    file_filter: str | None
    matches: list[CallsiteMatch]
    total_count: int


# ── Core CLI functions ─────────────────────────────────────────────────────────

def run_search(
    repo_root: str,
    query: str,
    limit: int = 20,
    scope: RepoScope | None = None,
) -> SearchResult:
    """Search for symbols and files matching query.

    Returns ranked matches with scores. Supports partial matches.
    """
    root = Path(repo_root).resolve()
    files, _ = scan_repo(root, scope=scope)

    results = search_symbols(files, query, limit=limit)
    matches = [
        {
            "path": r.path,
            "symbol": r.symbol,
            "kind": r.kind,
            "line": r.line,
            "score": r.score,
        }
        for r in results
    ]

    return SearchResult(
        query=query,
        matches=matches,
        total_count=len(results),
    )


def run_lookup(
    repo_root: str,
    file: str | None = None,
    symbol: str | None = None,
    line: int | None = None,
    scope: RepoScope | None = None,
    include_context: bool = True,
) -> LookupResult:
    """Precise location lookup by file+line or symbol.

    Returns the best matching function/location with optional source context.
    """
    root = Path(repo_root).resolve()
    files, _ = scan_repo(root, scope=scope)
    index = build_repo_index(files)

    # Case 1: file + line lookup
    if file and line is not None:
        loc = lookup_by_file_line(file, line, index)
        source_snippet = None
        callers_list: list[str] = []
        callees_list: list[str] = []

        if loc.repo_file and loc.function and include_context:
            # Get source snippet
            try:
                src_path = root / loc.repo_file
                src_lines = src_path.read_text(encoding="utf-8").splitlines()
                start = max(0, loc.function.line - 1)
                end = loc.function.end_line or min(start + 30, len(src_lines))
                source_snippet = "\n".join(src_lines[start:end])
            except (OSError, UnicodeDecodeError):
                pass

            # Get callers/callees
            callers_list = callers_of(loc.function.qualified_name, files, limit=20)
            callees_list = callees_of(loc.function.qualified_name, files, limit=20)

        return LookupResult(
            query_type="file_line",
            file=loc.repo_file,
            function=loc.function.qualified_name if loc.function else None,
            line=line,
            confidence=loc.confidence,
            reason=loc.reason,
            alternatives=loc.alternatives[:5],
            source_snippet=source_snippet,
            callers=callers_list,
            callees=callees_list,
        )

    # Case 2: symbol lookup
    if symbol:
        locs = lookup_by_symbol(symbol, index, source_file=file)
        if not locs:
            return LookupResult(
                query_type="symbol",
                file=None,
                function=None,
                line=None,
                confidence=0.0,
                reason="not_found",
                alternatives=[],
                source_snippet=None,
                callers=[],
                callees=[],
            )

        loc = locs[0]  # best match
        source_snippet = None
        callers_list: list[str] = []
        callees_list: list[str] = []

        if loc.repo_file and loc.function and include_context:
            try:
                src_path = root / loc.repo_file
                src_lines = src_path.read_text(encoding="utf-8").splitlines()
                start = max(0, loc.function.line - 1)
                end = loc.function.end_line or min(start + 30, len(src_lines))
                source_snippet = "\n".join(src_lines[start:end])
            except (OSError, UnicodeDecodeError):
                pass

            callers_list = callers_of(loc.function.qualified_name, files, limit=20)
            callees_list = callees_of(loc.function.qualified_name, files, limit=20)

        return LookupResult(
            query_type="symbol",
            file=loc.repo_file,
            function=loc.function.qualified_name if loc.function else None,
            line=loc.line,
            confidence=loc.confidence,
            reason=loc.reason,
            alternatives=loc.alternatives[:5],
            source_snippet=source_snippet,
            callers=callers_list,
            callees=callees_list,
        )

    # Case 3: file-only lookup
    if file:
        fm = fuzzy_file_match(file, index)
        if fm is None:
            return LookupResult(
                query_type="file",
                file=None,
                function=None,
                line=None,
                confidence=0.0,
                reason="not_found",
                alternatives=[],
                source_snippet=None,
                callers=[],
                callees=[],
            )
        return LookupResult(
            query_type="file",
            file=fm.path,
            function=None,
            line=None,
            confidence=fm.confidence,
            reason=fm.reason,
            alternatives=fm.alternatives[:5],
            source_snippet=None,
            callers=[],
            callees=[],
        )

    return LookupResult(
        query_type="none",
        file=None,
        function=None,
        line=None,
        confidence=0.0,
        reason="no_query",
        alternatives=[],
        source_snippet=None,
        callers=[],
        callees=[],
    )


def run_callers(
    repo_root: str,
    symbol: str,
    limit: int = 20,
    scope: RepoScope | None = None,
) -> CallersResult:
    """Find all functions that call the given symbol."""
    root = Path(repo_root).resolve()
    files, _ = scan_repo(root, scope=scope)

    caller_list = callers_of(symbol, files, limit=limit)
    return CallersResult(
        symbol=symbol,
        callers=caller_list,
        count=len(caller_list),
    )


def run_callees(
    repo_root: str,
    symbol: str,
    limit: int = 20,
    scope: RepoScope | None = None,
) -> CalleesResult:
    """Find all functions called by the given symbol."""
    root = Path(repo_root).resolve()
    files, _ = scan_repo(root, scope=scope)

    callee_list = callees_of(symbol, files, limit=limit)
    return CalleesResult(
        symbol=symbol,
        callees=callee_list,
        count=len(callee_list),
    )


def run_impact(
    repo_root: str,
    changed_files: list[str],
    max_depth: int = 5,
    max_nodes: int = 200,
    scope: RepoScope | None = None,
) -> ImpactResult:
    """Calculate impact radius of file changes.

    Returns files and symbols potentially affected by the changes.
    """
    root = Path(repo_root).resolve()
    files, _ = scan_repo(root, scope=scope)
    dag = build_dag(files)

    ir = impact_radius(files, dag, changed_files, max_depth=max_depth, max_nodes=max_nodes)

    return ImpactResult(
        seed_files=ir.seed_files,
        impacted_files=ir.impacted_files,
        impacted_symbols=ir.impacted_symbols[:100],  # cap
        depth_map=ir.depth_map,
        truncated=ir.truncated,
    )


def run_trace(
    repo_root: str,
    target: str,
    symbol: str | None = None,
    max_depth: int = 8,
    max_steps: int = 300,
    scope: RepoScope | None = None,
) -> TraceResult:
    """Trace call chains from a file or symbol.

    Returns all call chains reachable from the target.
    """
    root = Path(repo_root).resolve()
    files, _ = scan_repo(root, scope=scope)
    dag = build_dag(files)

    chains = trace_from(files, dag, target, symbol=symbol, max_depth=max_depth, max_steps=max_steps)

    trace_chains = [
        TraceChain(
            entry_path=c.entry_path,
            entry_symbol=c.entry_symbol,
            visited_files=c.visited_files,
            visited_symbols=c.visited_symbols,
            steps=[TraceStep(**asdict(s)) for s in c.steps],
            truncated=c.truncated,
        )
        for c in chains
    ]

    return TraceResult(
        target=target,
        chains=trace_chains,
    )


def run_callsites(
    repo_root: str,
    call_name: str | None = None,
    file_filter: str | None = None,
    finding_type: str | None = None,
    limit: int = 50,
    scope: RepoScope | None = None,
) -> CallsitesResult:
    """Search for callsites by call name or finding type.

    Returns ranked callsite candidates with source context.
    """
    root = Path(repo_root).resolve()
    files, _ = scan_repo(root, scope=scope)

    # Build callsite index
    index = build_callsite_index(files)

    # Determine search parameters
    if finding_type:
        analysis_scope = derive_analysis_scope(finding_type, files)
        search_names = analysis_scope.call_names
        search_kw = set(analysis_scope.call_keywords)
    elif call_name:
        search_names = [call_name]
        search_kw = set()
    else:
        # Default: search for common GPU-related calls
        search_names = ["to", "cuda", "synchronize", "copy_", "pin_memory"]
        search_kw = set()

    # Build analysis scope for search
    if file_filter:
        # Filter to specific file
        analysis_scope = AnalysisScope(
            finding_type="custom",
            priority_file_patterns=[],
            priority_func_patterns=[],
            call_names=search_names,
            call_keywords=list(search_kw),
            seed_files=[file_filter],
            selected_files=[file_filter],
            excluded_patterns=[],
            reason=["file_filter"],
        )
    else:
        analysis_scope = derive_analysis_scope(finding_type or "default", files)
        analysis_scope.call_names = search_names
        analysis_scope.call_keywords = list(search_kw)

    candidates = search_calls(index, analysis_scope, limit=limit)

    matches = [
        CallsiteMatch(
            id=c.id,
            path=c.path,
            line=c.line,
            call_name=c.call_name,
            full_call_name=c.full_call_name,
            receiver=c.receiver,
            keywords=c.keywords,
            loop_depth=c.loop_depth,
            enclosing_function=c.enclosing_function,
            source_line=c.source_line,
            score=c.score,
        )
        for c in candidates
    ]

    return CallsitesResult(
        call_name=call_name,
        file_filter=file_filter,
        matches=matches,
        total_count=len(matches),
    )


def run_callsite_context(
    repo_root: str,
    callsite_id: str,
    scope: RepoScope | None = None,
) -> CallsiteContext | None:
    """Get full source context for a specific callsite."""
    root = Path(repo_root).resolve()
    files, _ = scan_repo(root, scope=scope)

    return get_callsite_context(callsite_id, files, repo_root=root)


# ── JSON serialization helpers ────────────────────────────────────────────────

def to_json(result: Any) -> str:
    """Serialize any result dataclass to JSON."""
    if hasattr(result, '__dataclass_fields__'):
        return json.dumps(asdict(result), indent=2, ensure_ascii=False)
    return json.dumps(result, indent=2, ensure_ascii=False)


__all__ = [
    # Result types
    "SymbolInfo",
    "SearchResult",
    "LookupResult",
    "CallersResult",
    "CalleesResult",
    "ImpactResult",
    "TraceStep",
    "TraceChain",
    "TraceResult",
    "CallsiteMatch",
    "CallsitesResult",
    # Core functions
    "run_manifest",
    "run_index",
    "run_search",
    "run_lookup",
    "run_callers",
    "run_callees",
    "run_impact",
    "run_trace",
    "run_callsites",
    "run_callsite_context",
    "to_json",
]
