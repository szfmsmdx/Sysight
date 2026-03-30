"""Adapted memory-bandwidth skill from nsys-ai."""

from __future__ import annotations

from nsys_agent.analysis.queries import memory_bandwidth_summary

from .base import Skill

_COPY_KIND_NAMES = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}


def _run(prof, device=None, trim=None):
    return memory_bandwidth_summary(prof, device, trim)


def _format(rows) -> str:
    if not rows:
        return "(No memory copy operations found)"
    lines = ["-- Memory Bandwidth --"]
    for row in rows:
        lines.append(
            f"- {_COPY_KIND_NAMES.get(row['copyKind'], row['copyKind'])}: count={row['op_count']}, "
            f"total={row['total_mb']:.2f}MB, avg_bw={row['avg_bandwidth_gbps']:.2f}GB/s, "
            f"peak_bw={row['peak_bandwidth_gbps']:.2f}GB/s"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="memory_bandwidth",
    title="Memory Bandwidth",
    description="Computes sustained and peak bandwidth for memcpy operations.",
    runner=_run,
    formatter=_format,
)
