"""Compute/communication overlap breakdown skill adapted from nsys-ai."""

from __future__ import annotations

from sysight.analysis.queries import overlap_analysis

from .base import Skill


def _run(prof, device=None, trim=None):
    target = device if device is not None else (prof.meta.devices[0] if prof.meta.devices else 0)
    return overlap_analysis(prof, target, trim)


def _format(result) -> str:
    if not result:
        return "(No overlap data available)"
    if "error" in result:
        return f"(Overlap analysis: {result['error']})"
    return "\n".join(
        [
            "── Compute/Communication Overlap ──",
            f"  Total span:    {result['total_ms']:.1f}ms",
            f"  Compute only:  {result['compute_only_ms']:.1f}ms",
            f"  NCCL only:     {result['nccl_only_ms']:.1f}ms",
            f"  Overlap:       {result['overlap_ms']:.1f}ms ({result['overlap_pct']}% of NCCL overlapped)",
            f"  Idle:          {result['idle_ms']:.1f}ms",
            f"  Kernels:       {result['compute_kernels']} compute + {result['nccl_kernels']} NCCL",
        ]
    )


SKILL = Skill(
    name="overlap_breakdown",
    title="Compute/Communication Overlap Breakdown",
    description="Quantifies compute-only, NCCL-only, overlap, and idle time on one GPU.",
    runner=_run,
    formatter=_format,
)
