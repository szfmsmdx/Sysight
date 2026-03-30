"""Top kernel summary skill aligned with nsys-ai."""

from __future__ import annotations

from nsys_agent.analysis.queries import top_kernel_summary

from .base import Skill


def _run(prof, device=None, trim=None):
    target = device if device is not None else (prof.meta.devices[0] if prof.meta.devices else 0)
    return top_kernel_summary(prof, target, trim, limit=15)


def _format(rows) -> str:
    if not rows:
        return "(No kernels found)"
    lines = [
        "── Top GPU Kernels by Total Time ──",
        f"{'Kernel':<60s}  {'Count':>7s}  {'Total(ms)':>10s}  {'Avg(ms)':>9s}  {'Min(ms)':>9s}  {'Max(ms)':>9s}",
        "─" * 112,
    ]
    for row in rows:
        name = row["kernel_name"]
        if len(name) > 58:
            name = name[:55] + "..."
        lines.append(
            f"{name:<60s}  {row['invocations']:>7d}  {row['total_ms']:>10.2f}  "
            f"{row['avg_ms']:>9.2f}  {row['min_ms']:>9.2f}  {row['max_ms']:>9.2f}"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="top_kernels",
    title="Top GPU Kernels",
    description="Ranks GPU kernels by cumulative execution time, with avg/min/max duration.",
    runner=_run,
    formatter=_format,
)
