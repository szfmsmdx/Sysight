"""nsys/models.py — Dataclass definitions for the nsys analyzer pipeline.

Design principles (aligned with nsys-ai):
  - No numeric confidence scores.  Use severity (critical/warning/info) only.
  - No repo mapping.  The nsys analyzer is profile-only.
  - Return structured data, not formatted prose, from core logic.

Key design decisions:
  - EventStat uses ``inclusive_pct`` (total_ns / trace.duration_ns) because
    events may overlap; the percentage can exceed 100% and that is expected.
  - BottleneckLabel carries both ``pct_of_trace`` and ``pct_of_gpu_active`` so
    that gpu_idle / host_overhead (denominator = trace duration) and
    gpu_compute (denominator = gpu_active) are both unambiguous.
  - TimelineEvent.dur_ns may be 0 for cpu_sample (point events); these must NOT
    participate in interval-union active-time calculations.
  - NsysFinding is the diagnostic layer: specific problems with severity
    and next-step guidance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class GpuDeviceInfo:
    device_id: int
    name: str
    total_memory_bytes: int | None = None
    memory_bandwidth_bytes_per_s: int | None = None
    sm_count: int | None = None
    compute_capability: str | None = None
    bus_location: str | None = None
    chip_name: str | None = None


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
    table_roles: dict[str, str]
    version_hint: str
    warnings: list[str]
    gpu_devices: list[GpuDeviceInfo] = field(default_factory=list)


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
    """
    category: str
    active_ns: int          # interval-union time for this category
    inclusive_ns: int       # raw duration sum (may exceed active_ns due to overlap)
    pct_of_trace: float     # active_ns / trace.duration_ns  (may be > 1.0 for comm overlap)
    pct_of_gpu_active: float | None   # active_ns / gpu_active_ns; None if no GPU activity
    evidence: list[str]     # human-readable evidence strings


@dataclass
class EventStat:
    """Statistics for a named event (kernel, memcpy, API call, etc.).

    ``inclusive_pct`` = total_ns / trace.duration_ns.
    This may exceed 100% if events overlap (e.g. concurrent GPU streams).
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
      "sql_*"                Deep SQL analysis findings
    """
    category: str
    severity: str                           # "info" | "warning" | "critical"
    title: str
    description: str
    time_range_ns: tuple[int, int] | None = None  # optional time window
    process_id: int | None = None
    device_id: int | None = None
    rank: int | None = None
    evidence: list[str] = field(default_factory=list)       # human-readable evidence
    related_events: list[str] = field(default_factory=list) # event names involved
    related_hotspots: list[str] = field(default_factory=list)  # kernel/func names
    next_step: str = ""                     # recommended action
    # Stable ID: survives re-ordering of findings list.
    stable_id: str = ""


# ── T5: Sample aggregation ───────────────────────────────────────────────────

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
    callstack: list[str] = field(default_factory=list)
    coarse_location: str | None = None


# ── NVTX→Kernel attribution ──────────────────────────────────────────────────

@dataclass
class NvtxKernelAttribution:
    """Maps a GPU kernel to its enclosing NVTX range path.

    Produced by attribute_kernels_to_nvtx() (O(N+M) sort-merge sweep).
    Provides the precise chain: kernel → correlationId → Runtime API → NVTX stack.
    """
    kernel_name: str
    k_start_ns: int
    k_end_ns: int
    k_dur_ns: int
    nvtx_text: str          # innermost NVTX region label
    nvtx_path: str          # full path "outer > middle > inner"
    nvtx_depth: int         # nesting depth of the innermost region


# ── Stable finding ID ─────────────────────────────────────────────────────────

def stable_finding_id(category: str, severity: str, time_range_ns: tuple[int, int] | None, device_id: int | None = None) -> str:
    """Generate a stable finding ID from content fields (not sort order)."""
    raw = f"{category}|{severity}|{time_range_ns}|{device_id}"
    h = hashlib.sha1(raw.encode(), usedforsecurity=False).hexdigest()[:8]
    return f"{category}:{h}"


# ── Final diagnosis ───────────────────────────────────────────────────────────

@dataclass
class EvidenceWindow:
    problem_id: str
    category: str
    start_ns: int
    end_ns: int
    duration_ns: int
    device_id: int | None
    stream_id: int | None
    correlation_id: int | None
    event_name: str
    event_category: str
    before_event: str | None = None
    after_event: str | None = None
    runtime_api: str | None = None
    nvtx_labels: list[str] = field(default_factory=list)
    coarse_location: str | None = None
    callstack_summaries: list[str] = field(default_factory=list)
    sample_callstack: list[str] = field(default_factory=list)
    first_user_python_frame: str | None = None
    actionable_chain: list[str] = field(default_factory=list)
    actionable_leaf_reason: str = ""
    why_not_actionable: str = ""
    window_rank_in_iter: int | None = None
    kernel_constraints: dict[str, str] = field(default_factory=dict)


@dataclass
class InvestigationAnchor:
    window_id: str
    problem_id: str = ""
    category: str = ""
    event_name: str = ""
    file_path: str = ""
    line: int | None = None
    function: str = ""
    rationale: str = ""
    suggestion: str = ""
    status: str = "unknown"


@dataclass
class InvestigationQuestion:
    question_id: str
    problem_id: str = ""
    category: str = ""
    title: str = ""
    file_path: str = ""
    line: int | None = None
    function: str = ""
    rationale: str = ""
    suggestion: str = ""
    status: str = "unknown"
    window_ids: list[str] = field(default_factory=list)


@dataclass
class InvestigationResult:
    backend: str
    status: str                     # "ok" | "error" | "skipped" | "running"
    prompt: str
    output: str = ""
    error: str = ""
    command: list[str] = field(default_factory=list)
    output_path: str = ""
    pid: int | None = None
    summary: str = ""
    anchors: list[InvestigationAnchor] = field(default_factory=list)
    questions: list[InvestigationQuestion] = field(default_factory=list)
    artifact_dir: str = ""
    prompt_path: str = ""
    stdout_path: str = ""
    stderr_path: str = ""


@dataclass
class NsysDiag:
    status: str                     # "ok" | "action_required" | "error"
    profile_path: str
    sqlite_path: str | None
    required_action: str | None     # what the parent agent must do next
    bottlenecks: BottleneckSummary | None
    findings: list[NsysFinding]     # diagnostic problems
    hotspots: list[SampleHotspot]   # CPU sample hotspots (profile-only)
    warnings: list[str]
    summary: str                    # brief text for quick-read; not primary data
    gpu_devices: list[GpuDeviceInfo] = field(default_factory=list)
    windows: list[EvidenceWindow] = field(default_factory=list)
    investigation: InvestigationResult | None = None


# ── Request type (for analyze_nsys() entry point) ────────────────────────────

@dataclass
class NsysAnalysisRequest:
    """Input to the analyze_nsys() entry point.

    Prefer passing sqlite_path explicitly; sibling-sqlite auto-detection is a
    convenience fallback only.
    """
    profile_path: str = ""
    sqlite_path: str | None = None          # explicit sqlite; overrides auto-detect
    repo_root: str | None = None
    top_hotspots: int = 20
    top_windows_per_finding: int = 3
    run_investigation: bool = False
    investigation_backend: str | None = None
    investigation_model: str | None = None
    emit_stage_info: bool = False
    include_deep_sql: bool = True
    include_evidence_windows: bool = True
