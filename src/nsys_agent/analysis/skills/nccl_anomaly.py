"""Adapted NCCL anomaly skill from nsys-ai."""

from __future__ import annotations

from nsys_agent.analysis.queries import nccl_anomalies

from .base import Skill


def _run(prof, device=None, trim=None):
    target = device if device is not None else (prof.meta.devices[0] if prof.meta.devices else 0)
    return nccl_anomalies(prof, target, trim)


def _format(rows) -> str:
    if not rows:
        return "(No NCCL anomalies detected)"
    lines = ["-- NCCL Anomalies --"]
    for row in rows[:10]:
        lines.append(
            f"- {row['op_type']}: {row['dur_ms']:.3f}ms on stream {row['streamId']} "
            f"({row['ratio_to_avg']}x avg)"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="nccl_anomaly",
    title="NCCL Anomaly Detection",
    description="Finds NCCL collectives whose duration is much larger than their peer average.",
    runner=_run,
    formatter=_format,
)
