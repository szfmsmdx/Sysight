"""sysight.analyzer — static code analysis + nsys profile diagnosis.

Files:
    repo.py      — repo scanning & analysis (pure, no CLI)
    nsys/        — nsys profile pipeline (T1–T5)
    cli.py       — CLI entry point + output rendering
    scanners/    — per-language AST parsers
"""

from .analyzer import (  # noqa: F401
    analyze_repo, render_summary, render_trace,
    discover_repo, scan_repo, build_repo_index, build_dag,
    fuzzy_file_match, lookup_by_file_line, lookup_by_symbol,
    callers_of, callees_of, get_repo_context,
    search_symbols, impact_radius, find_hubs, trace_from,
    RepoScope, RepoManifest, ContextBudget,
    RepoIndex, FileMatch, LocationMatch,
    RepoContextBundle, SourceSnippet,
    AnalysisResult, EntryPoint, CallChain, CallStep,
    FileDAG, HubNode, SearchResult, ImpactResult, SCANNERS,
)
from .callsite import (  # noqa: F401
    AnalysisScope,
    CallSiteCandidate,
    CallsiteContext,
    build_callsite_index,
    derive_analysis_scope,
    get_callsite_context,
    search_calls,
)

from .nsys import (  # noqa: F401
    analyze_nsys,
    extract_evidence_windows,
    prepare_analysis_request,
    register_cli_investigator,
)
from .nsys.models import (  # noqa: F401
    EvidenceWindow,
    InvestigationResult,
    NsysAnalysisRequest, NsysDiag,
    ProfileInput, SchemaInfo,
    NsysTrace, TimelineEvent,
    BottleneckSummary, BottleneckLabel, EventStat,
    SampleHotspot, SourceFrame,
    NsysFinding, stable_finding_id,
)



def __getattr__(name: str):
    if name == "main":
        from .cli import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
