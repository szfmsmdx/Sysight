"""Profile evidence dataclasses. Zero internal dependencies."""

from __future__ import annotations

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


@dataclass
class BottleneckReport:
    """Aggregated bottleneck classification for a profile."""
    category: str
    active_ns: int
    inclusive_ns: int
    pct_of_trace: float
    pct_of_gpu_active: float | None
    evidence: list[str] = field(default_factory=list)


@dataclass
class EvidenceWindow:
    """Time-bounded evidence window around a finding."""
    problem_id: str
    category: str
    start_ns: int
    end_ns: int
    duration_ns: int
    device_id: int | None = None
    stream_id: int | None = None
    correlation_id: int | None = None
    event_name: str = ""
    event_category: str = ""
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
class ProfileEvidence:
    """Compact profile evidence bundle for LLM consumption."""
    run_id: str = ""
    profile_path: str = ""
    sqlite_path: str = ""
    profile_hash: str = ""
    duration_ns: int = 0
    gpu_active_ns: int = 0
    gpu_idle_ns: int = 0
    gpu_devices: list[GpuDeviceInfo] = field(default_factory=list)
    bottlenecks: list[BottleneckReport] = field(default_factory=list)
    windows: list[EvidenceWindow] = field(default_factory=list)
    schema_warnings: list[str] = field(default_factory=list)
    extraction_warnings: list[str] = field(default_factory=list)
