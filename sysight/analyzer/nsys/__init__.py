"""nsys — profile analyzer for NVIDIA Nsight Systems.

Entry point: analyze_nsys()

Returns: NsysDiag  (structured bottleneck + findings + repo hotspot mapping)

Pipeline stages T1–T8 as documented in analyzer.md v3:
  T1: resolve_profile_input  (.nsys-rep → sqlite; action_required if missing)
  T2: inspect_schema         (runtime schema probing → capabilities, table_roles)
  T3: extract_trace          (sqlite → NsysTrace with TimelineEvents)
  T4: classify_bottlenecks   (interval-union → BottleneckSummary + NsysFinding list)
  T5: Event stats built inside classify_bottlenecks
  T6: RepoMapper             (sample → repo file/function)  [via derive_repo_scope]
  T7: ContextEnricher        (callers/callees for high-confidence matches)
  T8: ReportSynthesizer      (assemble NsysDiag + EvidenceLink)

Scope-aware repo scanning:
  By default, analyze_nsys() does NOT do full repo scans for large repos.
  It derives a RepoScope from the trace evidence and scans only relevant files.
  Full scan is opt-in via NsysAnalysisRequest.repo_scope_mode="full".

Sub-module layout:
  input.py    — T1: ProfileInputResolver
  schema.py   — T2: SchemaInspector
  extract.py  — T3: TraceExtractor
  intervals.py — Interval math (union, intersect, gaps)
  classify.py — T4: BottleneckClassifier + NsysFinding generation
  models.py   — All dataclass definitions
"""

from __future__ import annotations

import logging
from pathlib import Path

from .classify import classify_bottlenecks
from .extract import (
    extract_trace,
    resolve_profile_input,
    inspect_schema,
    union_intervals, union_ns, intersect_total, find_gaps,
)
from .models import (
    BottleneckSummary,
    EvidenceLink,
    MappedHotspot,
    NsysDiag,
    NsysAnalysisRequest,
    NsysFinding,
    NsysTrace,
    ProfileInput,
    SchemaInfo,
    TimelineEvent,
)

from ..analyzer import (
    RepoScope,
    ContextBudget,
    scan_repo,
    build_repo_index,
    build_dag,
    lookup_by_file_line,
    lookup_by_symbol,
    callers_of,
    callees_of,
    fuzzy_file_match,
)

logger = logging.getLogger(__name__)

__all__ = [
    "analyze_nsys",
    "derive_repo_scope",
    "inspect_schema",
    "extract_trace",
    "classify_bottlenecks",
    # Models
    "NsysDiag", "NsysAnalysisRequest", "NsysFinding",
    "BottleneckSummary", "MappedHotspot", "EvidenceLink",
    "NsysTrace", "ProfileInput", "SchemaInfo", "TimelineEvent",
]


# ── Main entry point ─────────────────────────────────────────────────────────


def analyze_nsys(
    request: NsysAnalysisRequest,
) -> NsysDiag:
    """Main entry point for nsys profile analysis.

    Pipeline order:
      1. T1: resolve .nsys-rep → sqlite  (if missing, return action_required)
      2. T2: probe schema → capabilities & table_roles
      3. T3: extract timeline events → NsysTrace
      4.     derive_repo_scope(trace) → RepoScope (if include_repo_context)
      5. T4: classify bottlenecks → BottleneckSummary + list[NsysFinding]
      6.     map hotspots to repo (scope-aware scan)
      7.     enrich with callers/callees
      8.     synthesize NsysDiag + EvidenceLink
    """
    profile_path = Path(request.profile_path)

    # ── T1: Input resolution ─────────────────────────────────────────────────
    prof_input = resolve_profile_input(
        str(profile_path),
        sqlite_path=request.sqlite_path,
    )

    if prof_input.action_required or prof_input.sqlite_path is None:
        return NsysDiag(
            status="action_required",
            profile_path=str(profile_path),
            sqlite_path=prof_input.sqlite_path,
            required_action=prof_input.reason or (
                f"SQLite export required: run `nsys export -t sqlite "
                f"{profile_path}` to generate a .sqlite file, then retry."
            ),
            bottlenecks=None,
            findings=[],
            hotspots=[],
            evidence_links=[],
            repo_warnings=[],
            summary="Cannot proceed without sqlite export.",
        )

    sqlite_path = prof_input.sqlite_path

    # ── T2: Schema inspection ────────────────────────────────────────────────
    try:
        schema = inspect_schema(sqlite_path)
    except Exception as e:
        logger.error("Schema inspection failed: %s", e)
        return NsysDiag(
            status="error",
            profile_path=str(profile_path),
            sqlite_path=sqlite_path,
            required_action=None,
            bottlenecks=None,
            findings=[],
            hotspots=[],
            evidence_links=[],
            repo_warnings=[f"Schema inspection error: {e}"],
            summary=f"Failed to read SQLite schema: {e}",
        )

    if "cuda_kernel" not in schema.capabilities:
        return NsysDiag(
            status="error",
            profile_path=str(profile_path),
            sqlite_path=sqlite_path,
            required_action=(
                "Profile does not contain GPU kernel data. "
                "Re-capture with CUDA kernel tracing enabled: "
                "nsys profile --trace=cuda,nvtx,osrt ..."
            ),
            bottlenecks=None,
            findings=[],
            hotspots=[],
            evidence_links=[],
            repo_warnings=schema.warnings,
            summary="No GPU kernel data found in this profile.",
        )

    # ── T3: Trace extraction ─────────────────────────────────────────────────
    try:
        trace = extract_trace(sqlite_path, schema)
    except Exception as e:
        logger.error("Trace extraction failed: %s", e)
        return NsysDiag(
            status="error",
            profile_path=str(profile_path),
            sqlite_path=sqlite_path,
            required_action=None,
            bottlenecks=None,
            findings=[],
            hotspots=[],
            evidence_links=[],
            repo_warnings=schema.warnings + [f"Trace extraction error: {e}"],
            summary=f"Failed to extract trace: {e}",
        )

    # ── T4: Bottleneck classification + Findings ─────────────────────────────
    try:
        bottleneck_summary, findings = classify_bottlenecks(trace)
    except Exception as e:
        logger.error("Bottleneck classification failed: %s", e)
        bottleneck_summary = None
        findings = []
        trace.warnings.append(f"Classification error: {e}")

    # ── T6–T7: Repository mapping (scope-aware) ──────────────────────────────
    hotspots: list[MappedHotspot] = []
    evidence_links: list[EvidenceLink] = []
    repo_warnings: list[str] = list(trace.warnings)

    if request.include_repo_context and request.repo_root:
        try:
            hotspots, evidence_links, mapping_warnings = _map_hotspots_to_repo(
                trace, findings, request,
            )
            repo_warnings.extend(mapping_warnings)
        except Exception as e:
            logger.warning("Repo mapping failed (non-fatal): %s", e)
            repo_warnings.append(f"Repo mapping skipped: {e}")

    # ── T8: Synthesize NsysDiag ─────────────────────────────────────────────
    summary = _build_summary(bottleneck_summary, findings, trace)

    return NsysDiag(
        status="ok",
        profile_path=str(profile_path),
        sqlite_path=sqlite_path,
        required_action=None,
        bottlenecks=bottleneck_summary,
        findings=findings,
        hotspots=hotspots,
        evidence_links=evidence_links,
        repo_warnings=repo_warnings,
        summary=summary,
    )


# ── Repo mapping (T6–T7) ─────────────────────────────────────────────────────

def _map_hotspots_to_repo(
    trace: NsysTrace,
    findings: list[NsysFinding],
    request: NsysAnalysisRequest,
) -> tuple[list[MappedHotspot], list[EvidenceLink], list[str]]:
    """Map trace hotspots and findings to repository source locations."""
    from .models import SampleHotspot, SourceFrame
    from collections import Counter

    warnings: list[str] = []
    repo_root = Path(request.repo_root)

    if not repo_root.is_dir():
        warnings.append(f"repo_root not found: {repo_root}")
        return [], [], warnings

    # Derive repo scope from trace evidence
    scope = derive_repo_scope(trace, str(repo_root))
    scope.mode = request.repo_scope_mode
    scope.max_files = request.max_repo_files
    scope.max_file_bytes = request.max_file_bytes

    # Scan repo (targeted by default)
    files, scan_warnings = scan_repo(repo_root, scope=scope)
    warnings.extend(scan_warnings)

    if not files:
        warnings.append("No source files found matching trace scope.")
        return [], [], warnings

    index = build_repo_index(files)

    # Build hotspots from CPU samples
    hotspots: list[MappedHotspot] = []
    cpu_samples = [ev for ev in trace.events if ev.is_sample]

    if cpu_samples:
        counter: Counter[str] = Counter(ev.name for ev in cpu_samples)
        total = len(cpu_samples)
        for symbol, count in counter.most_common(request.top_hotspots):
            pct = count / total
            frame = SourceFrame(symbol=symbol, source_file=None, source_line=None)
            sample = SampleHotspot(frame=frame, count=count, pct=pct)

            # Try to map to repo
            matches = lookup_by_symbol(symbol, index)
            if matches and matches[0].confidence >= request.map_confidence_threshold:
                m = matches[0]
                fn_name = m.function.qualified_name if m.function else None
                callers = callers_of(fn_name, files) if fn_name else []
                callees_list = callees_of(fn_name, files) if fn_name else []
                hotspots.append(MappedHotspot(
                    sample=sample,
                    repo_file=m.repo_file,
                    function=fn_name,
                    match_confidence=m.confidence,
                    match_reason=m.reason,
                    callers=callers,
                    callees=callees_list,
                    alternatives=[str(a) for a in m.alternatives],
                ))
            else:
                hotspots.append(MappedHotspot(
                    sample=sample,
                    repo_file=None,
                    function=None,
                    match_confidence=0.0,
                    match_reason="none",
                    callers=[],
                    callees=[],
                    alternatives=[],
                ))

    # Build evidence links from kernel findings
    evidence_links: list[EvidenceLink] = []
    for finding in findings:
        for hotspot_name in finding.related_hotspots[:3]:
            matches = lookup_by_symbol(hotspot_name, index)
            if matches and matches[0].confidence >= request.map_confidence_threshold:
                m = matches[0]
                fn_name = m.function.qualified_name if m.function else None
                evidence_links.append(EvidenceLink(
                    bottleneck_category=finding.category,
                    event_name=hotspot_name,
                    event_category=finding.category,
                    hotspot_function=fn_name,
                    hotspot_file=m.repo_file,
                    link_type="symbol",
                    reason=f"Kernel name matched repo symbol with confidence {m.confidence:.2f}",
                    confidence=m.confidence * finding.confidence,
                ))

    return hotspots, evidence_links, warnings


# ── Summary builder ───────────────────────────────────────────────────────────

def _build_summary(
    bottlenecks: BottleneckSummary | None,
    findings: list[NsysFinding],
    trace: NsysTrace,
) -> str:
    """Build a short text summary for LLM quick-read."""
    lines: list[str] = []

    if bottlenecks:
        idle_pct = bottlenecks.gpu_idle_ns / bottlenecks.total_ns * 100 if bottlenecks.total_ns else 0
        lines.append(
            f"Trace: {bottlenecks.total_ns / 1e6:.1f}ms, "
            f"GPU active: {bottlenecks.gpu_active_ns / 1e6:.1f}ms ({100 - idle_pct:.1f}%), "
            f"GPU idle: {bottlenecks.gpu_idle_ns / 1e6:.1f}ms ({idle_pct:.1f}%)."
        )
        if bottlenecks.labels:
            top = bottlenecks.labels[0]
            lines.append(
                f"Primary bottleneck: {top.category} "
                f"({top.pct_of_trace * 100:.1f}% of trace)."
            )

    crit = [f for f in findings if f.severity == "critical"]
    warn = [f for f in findings if f.severity == "warning"]
    if crit:
        lines.append(f"{len(crit)} critical finding(s): " + "; ".join(f.title for f in crit[:3]))
    if warn:
        lines.append(f"{len(warn)} warning(s): " + "; ".join(f.title for f in warn[:3]))

    if not lines:
        lines.append(f"Trace {trace.profile_path}: {len(trace.events)} events extracted.")

    return " ".join(lines)


# ── RepoScope derivation from trace evidence ─────────────────────────────────

def derive_repo_scope(
    trace: NsysTrace,
    repo_root: str,
) -> RepoScope:
    """Generate a targeted RepoScope from profile evidence.

    Extracts candidate source files, symbols, and kernel names from:
      - TimelineEvent.extra["source_file"]
      - GPU kernel names
      - NVTX range names

    Priority-based scope: seed_files from profile evidence are highest priority;
    GPU-hinted file expansion is lower priority.

    Returns a RepoScope with mode="targeted" and reasonable defaults.
    Called AFTER T3 (trace extraction) but BEFORE T6 (repo scanning).
    """
    seed_files: list[str] = []
    seed_symbols: list[str] = []
    seed_kernels: list[str] = []

    for event in trace.events:
        if event.category in {"gpu_compute", "gpu_comm"}:
            seed_kernels.append(event.name)
        if event.category == "cuda_api" and "source_file" in event.extra:
            sf = event.extra["source_file"]
            if isinstance(sf, str) and sf:
                seed_files.append(sf)
        if event.category == "nvtx":
            name = event.name.strip()
            if name and not name.upper().startswith("NCCL"):
                seed_symbols.append(name)

    return RepoScope(
        mode="targeted",
        seed_files=sorted(set(seed_files)),
        seed_symbols=sorted(set(seed_symbols)),
        seed_kernels=sorted(set(seed_kernels)),
        include_extensions={".py", ".cpp", ".cc", ".cxx", ".c",
                            ".h", ".hpp", ".cu", ".cuh"},
        follow_imports_depth=1,
        max_files=500,
        max_file_bytes=512_000,
        include_gpu_related=True,
    )
