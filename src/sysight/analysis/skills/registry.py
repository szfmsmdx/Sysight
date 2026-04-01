"""Registry for adapted built-in skills."""

from __future__ import annotations

from .analysis_summary import SKILL as ANALYSIS_SUMMARY
from .code_location import SKILL as CODE_LOCATION
from .cpu_gpu_pipeline import SKILL as CPU_GPU_PIPELINE
from .gpu_idle_gaps import SKILL as GPU_IDLE_GAPS
from .idle_gaps import SKILL as IDLE_GAPS
from .iteration_timing import SKILL as ITERATION_TIMING
from .kernel_launch_pattern import SKILL as KERNEL_LAUNCH_PATTERN
from .kernel_launch_overhead import SKILL as KERNEL_LAUNCH_OVERHEAD
from .memory_bandwidth import SKILL as MEMORY_BANDWIDTH
from .memory_transfers import SKILL as MEMORY_TRANSFERS
from .nccl_anomaly import SKILL as NCCL_ANOMALY
from .nccl_breakdown import SKILL as NCCL_BREAKDOWN
from .nvtx_kernel_map import SKILL as NVTX_KERNEL_MAP
from .nvtx_layer_breakdown import SKILL as NVTX_LAYER_BREAKDOWN
from .overlap_breakdown import SKILL as OVERLAP_BREAKDOWN
from .region_mfu import SKILL as REGION_MFU
from .root_cause_matcher import SKILL as ROOT_CAUSE_MATCHER
from .schema_inspect import SKILL as SCHEMA_INSPECT
from .speedup_estimator import SKILL as SPEEDUP_ESTIMATOR
from .stream_concurrency import SKILL as STREAM_CONCURRENCY
from .theoretical_flops import SKILL as THEORETICAL_FLOPS
from .thread_utilization import SKILL as THREAD_UTILIZATION
from .top_kernels import SKILL as TOP_KERNELS

_SKILLS = {
    skill.name: skill
    for skill in (
        ANALYSIS_SUMMARY,
        CODE_LOCATION,
        CPU_GPU_PIPELINE,
        GPU_IDLE_GAPS,
        IDLE_GAPS,
        ITERATION_TIMING,
        KERNEL_LAUNCH_PATTERN,
        MEMORY_TRANSFERS,
        MEMORY_BANDWIDTH,
        KERNEL_LAUNCH_OVERHEAD,
        NCCL_ANOMALY,
        NCCL_BREAKDOWN,
        NVTX_KERNEL_MAP,
        NVTX_LAYER_BREAKDOWN,
        OVERLAP_BREAKDOWN,
        REGION_MFU,
        ROOT_CAUSE_MATCHER,
        SCHEMA_INSPECT,
        SPEEDUP_ESTIMATOR,
        STREAM_CONCURRENCY,
        THEORETICAL_FLOPS,
        THREAD_UTILIZATION,
        TOP_KERNELS,
    )
}


def list_skills() -> list[str]:
    return sorted(_SKILLS)


def get_skill(name: str):
    return _SKILLS.get(name)


def all_skills():
    return [_SKILLS[name] for name in list_skills()]
