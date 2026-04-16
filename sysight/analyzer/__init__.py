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

from .nsys import analyze_nsys, derive_repo_scope  # noqa: F401
from .nsys.models import (  # noqa: F401
    NsysAnalysisRequest, NsysDiag,
    ProfileInput, SchemaInfo,
    NsysTrace, TimelineEvent,
    BottleneckSummary, BottleneckLabel, EventStat,
    SampleHotspot, SourceFrame,
    MappedHotspot, EvidenceLink,
)

from .cli import main  # noqa: F401
