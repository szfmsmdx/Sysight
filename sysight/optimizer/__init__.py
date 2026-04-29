"""Optimizer component for Sysight.

Reads findings.json (produced by analyzer) and repo context,
and generates a set of unified diff patches inside patch_plan.json.
"""

from __future__ import annotations

from .cli import add_optimizer_subparser, dispatch_optimizer
from .plan import run_optimizer_agent
from .models import PatchPlan, Patch, MetricProbe

__all__ = [
    "add_optimizer_subparser",
    "dispatch_optimizer",
    "run_optimizer_agent",
    "PatchPlan",
    "Patch",
    "MetricProbe",
]
