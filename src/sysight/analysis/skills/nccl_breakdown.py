"""NCCL collective breakdown skill adapted from nsys-ai."""

from __future__ import annotations

from sysight.analysis.queries import nccl_breakdown

from .base import Skill


def _run(prof, device=None, trim=None):
    target = device if device is not None else (prof.meta.devices[0] if prof.meta.devices else 0)
    return nccl_breakdown(prof, target, trim)


def _format(rows) -> str:
    if not rows:
        return "(No NCCL operations found)"
    lines = [
        "── NCCL Collective Breakdown ──",
        f"{'Operation':<18s}  {'Count':>7s}  {'Total(ms)':>10s}  {'Avg(ms)':>9s}  {'Max(ms)':>9s}  {'Pct':>6s}",
        "─" * 78,
    ]
    for row in rows:
        lines.append(
            f"{row['type']:<18s}  {row['count']:>7d}  {row['total_ms']:>10.2f}  "
            f"{row['avg_ms']:>9.2f}  {row['max_ms']:>9.2f}  {row['pct']:>5.1f}%"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="nccl_breakdown",
    title="NCCL Collective Breakdown",
    description="Summarizes NCCL collective operations by type, count, time, and share of NCCL time.",
    runner=_run,
    formatter=_format,
)
