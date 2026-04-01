"""Region-level MFU skill adapted from nsys-ai."""

from __future__ import annotations

from sysight.analysis.mfu import compute_region_mfu, format_region_mfu

from .base import Skill


def _run(
    prof,
    device=None,
    trim=None,
    name: str = "",
    theoretical_flops: float = 0.0,
    source: str = "nvtx",
    peak_tflops: float | None = None,
    num_gpus: int = 1,
    occurrence_index: int = 1,
    match_mode: str = "contains",
):
    return compute_region_mfu(
        prof,
        name=name,
        theoretical_flops=float(theoretical_flops),
        source=source,
        peak_tflops=float(peak_tflops) if peak_tflops is not None else None,
        num_gpus=int(num_gpus),
        occurrence_index=int(occurrence_index),
        device=device,
        match_mode=match_mode,
    )


SKILL = Skill(
    name="region_mfu",
    title="Region-Level MFU",
    description="Computes MFU for a named NVTX region or kernel pattern from theoretical FLOPs and profile timings.",
    runner=_run,
    formatter=format_region_mfu,
)
