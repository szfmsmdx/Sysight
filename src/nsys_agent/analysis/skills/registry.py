"""Registry for adapted built-in skills."""

from __future__ import annotations

from .code_location import SKILL as CODE_LOCATION
from .idle_gaps import SKILL as IDLE_GAPS
from .iteration_timing import SKILL as ITERATION_TIMING
from .kernel_launch_overhead import SKILL as KERNEL_LAUNCH_OVERHEAD
from .memory_bandwidth import SKILL as MEMORY_BANDWIDTH
from .memory_transfers import SKILL as MEMORY_TRANSFERS
from .nccl_anomaly import SKILL as NCCL_ANOMALY
from .nvtx_kernel_map import SKILL as NVTX_KERNEL_MAP
from .nvtx_layer_breakdown import SKILL as NVTX_LAYER_BREAKDOWN
from .root_cause_matcher import SKILL as ROOT_CAUSE_MATCHER
from .top_kernels import SKILL as TOP_KERNELS

_SKILLS = {
    skill.name: skill
    for skill in (
        CODE_LOCATION,
        IDLE_GAPS,
        ITERATION_TIMING,
        MEMORY_TRANSFERS,
        MEMORY_BANDWIDTH,
        KERNEL_LAUNCH_OVERHEAD,
        NCCL_ANOMALY,
        NVTX_KERNEL_MAP,
        NVTX_LAYER_BREAKDOWN,
        ROOT_CAUSE_MATCHER,
        TOP_KERNELS,
    )
}


def list_skills() -> list[str]:
    return sorted(_SKILLS)


def get_skill(name: str):
    return _SKILLS.get(name)


def all_skills():
    return [_SKILLS[name] for name in list_skills()]
