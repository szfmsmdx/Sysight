"""Adapted memory-transfer skill from nsys-ai."""

from __future__ import annotations

from nsys_agent.analysis.queries import memory_transfer_summary

from .base import Skill

_COPY_KIND_NAMES = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}


def _run(prof, device=None, trim=None):
    return memory_transfer_summary(prof, device, trim)


def _format(rows) -> str:
    if not rows:
        return "(No memory transfers found)"
    lines = ["-- Memory Transfers --"]
    for row in rows:
        lines.append(
            f"- {_COPY_KIND_NAMES.get(row['copyKind'], row['copyKind'])}: {row['count']} ops, "
            f"{row['total_mb']:.2f}MB, {row['total_ms']:.2f}ms"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="memory_transfers",
    title="Memory Transfers",
    description="Breaks down H2D, D2H, D2D and P2P transfer time from the profile.",
    runner=_run,
    formatter=_format,
)
