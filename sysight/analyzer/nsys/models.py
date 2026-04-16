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
    events: list[TimelineEvent]
    schema_version: str
    warnings: list[str]


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
class BottleneckSummary:
    total_ns: int           # trace.duration_ns  (wall clock)
    gpu_active_ns: int      # interval-union of all GPU events
    gpu_idle_ns: int        # total_ns - gpu_active_ns
    labels: list[BottleneckLabel]   # sorted by pct_of_trace descending
    top_events: list[EventStat]     # top 10 most time-consuming events (cross-category)


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


# ── T8: Evidence links ────────────────────────────────────────────────────────

@dataclass
class EvidenceLink:
    """Connects a bottleneck category to a source code location.

    Link priority (from most to least reliable):
      1. correlation_id: CUDA API → GPU kernel (nsys-native, most reliable)
      2. time_overlap:   NVTX range / cpu_sample overlaps kernel window
      3. file_line:      SourceFrame.source_file + source_line → repo function
      4. symbol:         SourceFrame.symbol → repo function name match
    """
    bottleneck_category: str        # e.g. "gpu_comm"
    event_name: str                 # e.g. "ncclAllReduceKernel"
    event_category: str
    hotspot_function: str | None    # e.g. "trainer.backward_step"
    hotspot_file: str | None
    link_type: str                  # "correlation_id" | "time_overlap" | "file_line" | "symbol"
    reason: str                     # human-readable explanation
    confidence: float


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
