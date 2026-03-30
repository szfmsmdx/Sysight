"""Code-location skill for suspicious gaps."""

from __future__ import annotations

from nsys_agent.analysis.code_location import format_code_locations, locate_code_for_gaps

from .base import Skill


def _run(prof, device=None, trim=None):
    target = device if device is not None else (prof.meta.devices[0] if prof.meta.devices else 0)
    return locate_code_for_gaps(prof, target, trim)


def _format(rows) -> str:
    return format_code_locations(rows)


SKILL = Skill(
    name="code_location",
    title="Code Location Candidates",
    description="Maps suspicious idle windows to runtime threads and sampled PyTorch/Triton call-site candidates.",
    runner=_run,
    formatter=_format,
)
