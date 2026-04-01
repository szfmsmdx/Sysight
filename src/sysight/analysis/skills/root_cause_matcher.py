"""Adapted root-cause matcher built from the lightweight query set."""

from __future__ import annotations

from sysight.analysis.iterations import detect_iterations, iteration_summary
from sysight.analysis.nvtx import nvtx_layer_breakdown
from sysight.analysis.queries import (
    detect_idle_gaps,
    h2d_distribution,
    kernel_launch_overhead,
    match_root_causes,
    memory_bandwidth_summary,
    nccl_anomalies,
    nccl_breakdown,
    overlap_analysis,
    pageable_memcpy_summary,
    sync_api_summary,
)
from sysight.analysis.summary import gpu_summary

from .base import Skill


def _run(prof, device=None, trim=None):
    target = device if device is not None else (prof.meta.devices[0] if prof.meta.devices else 0)
    iterations = detect_iterations(prof, target, trim)
    data = {
        "target_device": target,
        "target_summary": gpu_summary(prof, target, trim),
        "idle_gaps": detect_idle_gaps(prof, target, trim),
        "memory_bandwidth": memory_bandwidth_summary(prof, target, trim),
        "h2d_distribution": h2d_distribution(prof, target, trim),
        "launch_overhead": kernel_launch_overhead(prof, target, trim),
        "nccl_breakdown": nccl_breakdown(prof, target, trim),
        "nccl_anomalies": nccl_anomalies(prof, target, trim),
        "overlap": overlap_analysis(prof, target, trim),
        "sync_summary": sync_api_summary(prof, target, trim),
        "pageable_memcpy": pageable_memcpy_summary(prof, target, trim),
        "iterations_summary": iteration_summary(iterations),
        "nvtx_regions": nvtx_layer_breakdown(prof, target, trim, limit=20),
    }
    return match_root_causes(data)


def _format(rows) -> str:
    if not rows:
        return "(No root-cause patterns matched)"
    lines = ["-- Root Cause Pattern Analysis --"]
    for row in rows:
        lines.append(f"- [{row['severity']}] {row['pattern']}")
        lines.append(f"  Evidence: {row['evidence']}")
        lines.append(f"  Fix: {row['recommendation']}")
    return "\n".join(lines)


SKILL = Skill(
    name="root_cause_matcher",
    title="Root Cause Matcher",
    description="Combines lightweight analysis signals into actionable anti-pattern diagnoses.",
    runner=_run,
    formatter=_format,
)
