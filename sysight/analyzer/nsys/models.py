"""nsys/models.py — All dataclass definitions for the nsys analyzer pipeline.

This is the single source of truth for data structures.  All pipeline stages
(T1–T8) import from here; no other file defines these types.

Key design decisions:
- EventStat uses ``inclusive_pct`` (total_ns / trace.duration_ns) because
  events may overlap; the percentage can exceed 100% and that is expected.
- BottleneckLabel carries both ``pct_of_trace`` and ``pct_of_gpu_active`` so
  that gpu_idle / host_overhead (denominator = trace duration) and
  gpu_compute (denominator = gpu_active) are both unambiguous.
- TimelineEvent.dur_ns may be 0 for cpu_sample (point events); these must NOT
  participate in interval-union active-time calculations.
- EvidenceLink priority: correlation_id > time overlap > file:line > symbol.
- NsysFinding is the diagnostic layer: specific problems with severity,
  time range, and next-step guidance.  LLM/optimizer should read findings
  first; BottleneckSummary is the statistical layer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

# ── T1: Input ─────────────────────────────────────────────────────────────────

@dataclass
class ProfileInput:
    original_path: str
    sqlite_path: str | None     # None → needs export by executor
    action_required: bool
    reason: str = ""


# ── T2: Schema ────────────────────────────────────────────────────────────────

@dataclass
class SchemaInfo:
    tables: dict[str, list[str]]        # table_name → [col_name, ...]
    capabilities: list[str]             # e.g. ["gpu_kernel", "cpu_sample", "nvtx"]
    # Canonical table-name mapping: capability → actual table name in this db.
    # e.g. {"cuda_kernel": "CUPTI_ACTIVITY_KIND_KERNEL", "cuda_runtime": "CUPTI_ACTIVITY_KIND_RUNTIME"}
    table_roles: dict[str, str]
    version_hint: str
    warnings: list[str]


# ── T3: Timeline events ───────────────────────────────────────────────────────

@dataclass
class TimelineEvent:
    category: str
    # "gpu_compute" | "gpu_comm" | "gpu_memcpy"
    # "cuda_api"    | "nvtx"     | "cpu_sample"
    # "sync_wait"   | "osrt"
    name: str
    start_ns: int
    dur_ns: int
    # dur_ns == 0 is valid for cpu_sample (point events).
    # category == "cpu_sample" events must NOT enter interval-union calculations.
    is_sample: bool = False             # True for cpu_sample point events
    process_id: int | None = None       # required for multi-rank profiles
    thread_id: int | None = None
    global_tid: int | None = None       # raw nsys globalTid (serialized)
    device_id: int | None = None        # required for multi-GPU profiles
    stream_id: int | None = None
    correlation_id: int | None = None   # nsys CPU↔GPU correlation key
    rank: int | None = None             # MPI/NCCL rank (if known)
    extra: dict[str, str | int | float] = field(default_factory=dict)
    # extra examples: {"grid_x": 128, "block_x": 256, "size_bytes": 4096,
    #                  "copy_kind": 1, "src_kind": 1, "dst_kind": 2}


@dataclass
class NsysTrace:
    tool: str               # "nsys" — reserved for future multi-profiler support
    profile_path: str
    sqlite_path: str
    duration_ns: int
    trace_start_ns: int = 0     # absolute start of the profiled window (ns)
    trace_end_ns: int = 0       # absolute end of the profiled window (ns)
    events: list[TimelineEvent] = field(default_factory=list)
    schema_version: str = "unknown"
    warnings: list[str] = field(default_factory=list)


# ── T4: Bottleneck classification ─────────────────────────────────────────────

@dataclass
class BottleneckLabel:
    """Represents one bottleneck category.

    Two complementary percentage fields:
      - pct_of_trace:      active_ns / trace.duration_ns
                           Use for: gpu_idle, host_overhead, sync_wait
                           (denominator = wall time)
      - pct_of_gpu_active: active_ns / gpu_active_ns  (None if gpu_active_ns == 0)
                           Use for: gpu_compute, gpu_comm, gpu_memcpy
                           (denominator = GPU-busy time)

    The old single ``pct`` field is removed to avoid denominator ambiguity.
    """
    category: str           # see bottleneck category table in analyzer.md
    active_ns: int          # interval-union time for this category
    inclusive_ns: int       # raw duration sum (may exceed active_ns due to overlap)
    pct_of_trace: float     # active_ns / trace.duration_ns  (may be > 1.0 for comm overlap)
    pct_of_gpu_active: float | None   # active_ns / gpu_active_ns; None if no GPU activity
    confidence: float       # 0.0–1.0
    evidence: list[str]     # human-readable evidence strings for parent agent


@dataclass
class EventStat:
    """Statistics for a named event (kernel, memcpy, API call, etc.).

    ``inclusive_pct`` = total_ns / trace.duration_ns.
    This may exceed 100% if events overlap (e.g. concurrent GPU streams).
    Name chosen to be explicit about inclusive semantics.
    """
    name: str
    category: str
    count: int
    total_ns: int
    max_ns: int
    avg_ns: float
    inclusive_pct: float    # total_ns / trace.duration_ns  (inclusive, may exceed 1.0)


@dataclass
class DeviceBreakdown:
    """Per-device GPU activity breakdown.

    Used when a profile covers multiple GPUs.
    active_ns is the interval-union GPU time for this device only.
    """
    device_id: int
    active_ns: int
    idle_ns: int
    pct_active: float           # active_ns / total_trace_ns
    top_event_names: list[str]  # top kernels for this device


@dataclass
class BottleneckSummary:
    total_ns: int           # trace.duration_ns  (wall clock)
    gpu_active_ns: int      # interval-union of all GPU events (all devices)
    gpu_idle_ns: int        # total_ns - gpu_active_ns
    labels: list[BottleneckLabel]   # sorted by pct_of_trace descending
    top_events: list[EventStat]     # top 10 most time-consuming events (cross-category)
    per_device: list[DeviceBreakdown] = field(default_factory=list)
    # non-empty only when profile has events from multiple device_ids


# ── NsysFinding: diagnostic problem descriptions ──────────────────────────────

@dataclass
class NsysFinding:
    """A specific diagnosed performance problem.

    BottleneckSummary is statistical; NsysFinding is diagnostic.
    LLM/optimizer should read findings first for actionable insight.

    category values (non-exhaustive):
      "gpu_idle"             GPU was not executing any kernels
      "gpu_compute_hotspot"  Non-NCCL compute dominates GPU time
      "gpu_comm_hotspot"     NCCL/MPI communication dominates GPU time
      "comm_not_overlapped"  Communication is serialized (not overlapping compute)
      "gpu_memcpy_hotspot"   Memory transfers dominate
      "sync_wait"            CPU is blocked waiting for GPU synchronization
      "host_launch_overhead" Kernel launch latency is the bottleneck
      "cpu_hotspot"          CPU sampling shows a hot function
      "many_tiny_kernels"    High kernel count but very short average duration
      "low_memcpy_throughput" Memory copy bandwidth is abnormally low
    """
    category: str
    severity: str                           # "info" | "warning" | "critical"
    confidence: float                       # 0.0–1.0
    title: str
    description: str
    time_range_ns: tuple[int, int] | None = None  # optional time window
    process_id: int | None = None
    device_id: int | None = None
    rank: int | None = None
    evidence: list[str] = field(default_factory=list)       # human-readable evidence
    related_events: list[str] = field(default_factory=list) # event names involved
    related_hotspots: list[str] = field(default_factory=list)  # kernel/func names
    next_step: str = ""                     # recommended action for optimizer
    # Stable ID: survives re-ordering of findings list.
    # Generated from category + severity + time_range_ns by stable_finding_id().
    # Empty string until explicitly assigned (e.g. in classify_bottlenecks).
    stable_id: str = ""


# ── T5: Sample & event aggregation ───────────────────────────────────────────

@dataclass
class SourceFrame:
    symbol: str | None
    source_file: str | None
    source_line: int | None
    module: str | None = None   # shared library / .so name
    raw: str | None = None      # original string for debugging


@dataclass
class SampleHotspot:
    frame: SourceFrame
    count: int
    pct: float                              # count / total_samples
    event_window_ns: tuple[int, int] | None = None  # time window of this sample


# ── T6+T7: Repository mapping ────────────────────────────────────────────────

@dataclass
class MappedHotspot:
    sample: SampleHotspot
    repo_file: str | None
    function: str | None        # FunctionFacts.qualified_name
    match_confidence: float     # 0.0–1.0
    match_reason: str           # "file_line" | "symbol" | "kernel_name" | "none"
    callers: list[str]          # callers_of() result; populated when confidence >= threshold
    callees: list[str]          # callees_of() result; populated when confidence >= threshold
    alternatives: list[str]     # candidate files/functions when confidence is low


# ── T8: Task drafts & optimize tasks ─────────────────────────────────────────


def stable_finding_id(category: str, severity: str, time_range_ns: tuple[int, int] | None, device_id: int | None = None) -> str:
    """Generate a stable finding ID from content fields (not sort order).

    Based on: category + severity + time_range_ns + device_id.
    Returns an 8-char hex prefix, e.g. "gpu_idle:a3f2c1b0".
    Safe to use as a long-term reference in TaskDraft / OptimizeTask.
    """
    raw = f"{category}|{severity}|{time_range_ns}|{device_id}"
    h = hashlib.sha1(raw.encode(), usedforsecurity=False).hexdigest()[:8]
    return f"{category}:{h}"


@dataclass
class TargetLocation:
    """A confirmed code location that an OptimizeTask targets.

    Every target_location must be traceable to a deterministic source:
    callsite_id, file + line, NVTX region name, CPU sample, or correlation_id.
    LLM must NOT create a TargetLocation without at least one anchor.
    """
    file: str
    line: int
    call: str                           # source expression, e.g. "batch.to(device)"
    # At least one anchor must be non-None:
    callsite_id: str | None = None      # "<path>:<line>:<col>:<call_name>" from callsite index
    nvtx_region: str | None = None      # NVTX range label that led here
    correlation_id: int | None = None   # nsys correlation_id linking CPU API to GPU kernel
    cpu_sample_pct: float | None = None # CPU sample percentage if from cpu_sample hotspot
    anchor_type: str = "unknown"        # "callsite" | "nvtx" | "correlation_id" | "cpu_sample" | "file_line"
    note: str = ""                      # LLM's brief justification


@dataclass
class RejectedCandidate:
    """A callsite the LLM investigated but ruled out.

    Required field: LLM must not silently drop candidates.
    reason must explain *why* this callsite is not the bottleneck.
    """
    callsite_id: str                    # same id format as TargetLocation.callsite_id
    file: str
    line: int
    call: str
    reason: str                         # e.g. "only runs at init, not in training loop"


@dataclass
class ConfidenceBreakdown:
    """Per-source confidence components for an OptimizeTask.

    Scores are 0.0–1.0.  The final composite confidence is computed by the
    LLM investigator; it must not exceed the deterministic_finding score by
    more than 0.15 (LLM adds code-semantic confidence, not bottleneck math).
    """
    deterministic_finding: float        # from NsysFinding.confidence (immutable)
    callsite_score: float = 0.0         # evidence from static callsite index
    llm_verify: float = 0.0            # LLM code-semantic verification
    nvtx_match: float = 0.0            # NVTX region → source mapping match
    cpu_sample_match: float = 0.0      # CPU backtrace → source file match

    def composite(self) -> float:
        """Compute final confidence; deterministic_finding is the ceiling anchor."""
        sources = [self.callsite_score, self.llm_verify,
                   self.nvtx_match, self.cpu_sample_match]
        code_confidence = max((s for s in sources if s > 0), default=0.0)
        # code semantics can raise confidence, but not by more than +0.15
        return min(1.0, self.deterministic_finding + min(0.15, code_confidence * 0.15))


@dataclass
class TaskDraft:
    """Core-generated task skeleton for one finding.

    Produced deterministically by the analyzer core (no LLM).
    Contains the finding, verification metric, search seeds, and evidence
    windows.  target_locations is empty — the LLM investigator fills that in.

    evidence_windows: top time windows from the profile that caused this
      finding (device/stream/event/neighboring kernels/overlapping NVTX).
      Most useful when no repo is available — these windows ARE the
      closest-to-code information we have without source mapping.

    search_specs: rg-pattern seeds derived from the finding category.
      These are SEARCH ENTRY POINTS, not conclusions.  The LLM investigator
      must run them and validate matches against profile evidence before
      writing target_locations.
    """
    id: str                             # e.g. "gpu_memcpy_h2d_001_draft"
    finding_id: str                     # corresponds to NsysFinding
    hypothesis: str                     # "H2D memcpy 量异常，待确认代码位置"
    verification_metric: str            # "H2D memcpy total_ms 下降"
    candidate_callsites: list[str]      # callsite ids from search_calls (scored)
    target_locations: list[dict]        # empty until LLM investigator fills it
    inferred_by: str = "deterministic"
    # ── Profile-derived evidence (no LLM, no repo needed) ────────────────────
    evidence_windows: list[dict] = field(default_factory=list)
    # Each dict: {"start_ms": float, "end_ms": float, "device": int|None,
    #             "stream": int|None, "event": str,
    #             "before_kernel": str|None, "after_kernel": str|None,
    #             "overlap_nvtx": list[str]}
    search_specs: list[dict] = field(default_factory=list)
    # Each dict: {"pattern": str, "kind": "rg", "rationale": str}
    # kind="rg" means: rg -n "<pattern>" <repo_root>


@dataclass
class OptimizeTask:
    """LLM-confirmed task with target code locations.

    Upgraded from a TaskDraft after LLM investigator validates callsites.
    confidence may be > 0.5 because it combines deterministic finding
    confidence with callsite facts and LLM verification.

    rejected_candidates: callsites the LLM investigated but ruled out,
      with reasons.  Required — LLM must not silently ignore candidates.

    confidence_breakdown: per-source confidence scores that compose the
      final confidence value, for auditability.
    """
    id: str
    finding_id: str                     # must be NsysFinding.stable_id, not sort-index
    hypothesis: str
    evidence_links: list[str]           # EvidenceLink ids (audit trail)
    target_files: list[str]
    target_locations: list[TargetLocation]   # non-empty before optimizer may run
    proposed_change_kind: str
    verification_metric: str
    confidence: float                   # ConfidenceBreakdown.composite()
    risk: str                           # "low" | "medium" | "high"
    inferred_by: str = "llm_investigated"
    rejected_candidates: list[RejectedCandidate] = field(default_factory=list)
    # LLM must populate this; empty rejected_candidates is a validator error
    confidence_breakdown: ConfidenceBreakdown | None = None


# ── T8: Evidence links ────────────────────────────────────────────────────────

@dataclass
class EvidenceLink:
    """Connects a bottleneck category to a source code location.

    Link priority (from most to least reliable):
      1. correlation_id: CUDA API → GPU kernel (nsys-native, most reliable)
      2. time_overlap:   NVTX range / cpu_sample overlaps kernel window
      3. file_line:      SourceFrame.source_file + source_line → repo function
      4. symbol:         SourceFrame.symbol → repo function name match

    LLM-inferred links must set inferred_by="llm_investigated" and have
    confidence <= 0.50 unless backed by deterministic evidence.
    """
    bottleneck_category: str        # e.g. "gpu_comm"
    event_name: str                 # e.g. "ncclAllReduceKernel"
    event_category: str
    hotspot_function: str | None    # e.g. "trainer.backward_step"
    hotspot_file: str | None
    link_type: str                  # "correlation_id" | "time_overlap" | "file_line" | "symbol"
    reason: str                     # human-readable explanation
    confidence: float
    # Audit / extended fields
    id: str | None = None           # stable reference: "{category}:{event}:{link_type}:{idx}"
    correlation_id: int | None = None   # nsys CPU↔GPU correlation key (most reliable)
    device_id: int | None = None    # which GPU device this link was found on
    inferred_by: str = "deterministic"  # "deterministic" | "llm_investigated"


# ── T8: Final diagnosis ───────────────────────────────────────────────────────

@dataclass
class NsysDiag:
    status: str                     # "ok" | "action_required" | "error"
    profile_path: str
    sqlite_path: str | None
    required_action: str | None     # what the parent agent must do next
    bottlenecks: BottleneckSummary | None
    findings: list[NsysFinding]     # diagnostic problems (LLM reads this first)
    hotspots: list[MappedHotspot]
    evidence_links: list[EvidenceLink]
    task_drafts: list[TaskDraft]    # Core-generated task skeletons; LLM upgrades to OptimizeTask
    repo_warnings: list[str]
    summary: str                    # brief text for LLM quick-read; not primary data


# ── Request / scope types (for analyze_nsys() entry point) ───────────────────

@dataclass
class NsysAnalysisRequest:
    """Input to the analyze_nsys() entry point.

    Prefer passing sqlite_path explicitly; sibling-sqlite auto-detection is a
    convenience fallback only.  Executors may export to different directories.

    Repo-scan controls (no direct RepoScope import to keep model layer thin):
      repo_scope_mode:     "targeted" (default) | "full"
      max_repo_files:      maximum files to parse in targeted mode
      max_file_bytes:      skip files larger than this in targeted mode
      include_repo_context: if False, skip T6–T7 repo mapping entirely
    """
    repo_root: str
    profile_path: str
    sqlite_path: str | None = None          # explicit sqlite; overrides auto-detect
    top_hotspots: int = 20
    map_confidence_threshold: float = 0.4
    # Repo-scan controls
    repo_scope_mode: str = "targeted"
    max_repo_files: int = 500
    max_file_bytes: int = 512_000
    include_repo_context: bool = True
