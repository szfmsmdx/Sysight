"""Iteration timing skill adapted from nsys-ai."""

from __future__ import annotations

from nsys_agent.analysis.iterations import detect_iterations, iteration_summary

from .base import Skill


def _run(prof, device=None, trim=None):
    target = device if device is not None else (prof.meta.devices[0] if prof.meta.devices else 0)
    rows = detect_iterations(prof, target, trim)
    return {"rows": rows, "summary": iteration_summary(rows)}


def _format(result) -> str:
    rows = result["rows"]
    summary = result["summary"]
    if not rows:
        return "(No iterations detected)"

    is_heuristic = summary["heuristic"] if summary else False
    title = "── Iteration Timings (Heuristic) ──" if is_heuristic else "── Iteration Timings ──"
    lines = [title]
    for row in rows[:20]:
        lines.append(
            f"iter {row['iteration']:>2d}  {row['duration_ms']:>8.1f}ms  "
            f"({row['kernel_count']} kernels, {row['nccl_count']} NCCL)  compute={row['compute_ms']:.1f}ms"
        )
    if summary:
        lines.append("")
        lines.append(
            f"Average {summary['avg_ms']:.1f}ms | Median {summary['median_ms']:.1f}ms | "
            f"Min {summary['min_ms']:.1f}ms | Max {summary['max_ms']:.1f}ms"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="iteration_timing",
    title="Per-Iteration Timing",
    description="Detects repeated iterations from NVTX markers or large gaps and reports per-iteration timing.",
    runner=_run,
    formatter=_format,
)
