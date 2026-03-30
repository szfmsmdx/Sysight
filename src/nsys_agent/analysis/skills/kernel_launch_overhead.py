"""Adapted kernel-launch-overhead skill from nsys-ai."""

from __future__ import annotations

from nsys_agent.analysis.queries import kernel_launch_overhead

from .base import Skill


def _run(prof, device=None, trim=None):
    target = device if device is not None else (prof.meta.devices[0] if prof.meta.devices else 0)
    return kernel_launch_overhead(prof, target, trim)


def _format(rows) -> str:
    if not rows:
        return "(No kernel launch overhead data found)"
    lines = ["-- Kernel Launch Overhead --"]
    for row in rows[:10]:
        lines.append(
            f"- {row['kernel_name']}: api={row['api_ms']:.3f}ms, kernel={row['kernel_ms']:.3f}ms, "
            f"overhead={row['overhead_us']:.1f}us"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="kernel_launch_overhead",
    title="Kernel Launch Overhead",
    description="Measures the time from runtime launch call to actual kernel execution.",
    runner=_run,
    formatter=_format,
)
