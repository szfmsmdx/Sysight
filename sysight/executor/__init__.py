"""Executor component for Sysight.

Takes a patch_plan.json from the optimizer and physically runs
the loop of git apply, verification via metric probe, and conditionally commits or reverts.
"""

from __future__ import annotations

from .cli import add_executor_subparser, dispatch_executor
from .execute import execute_patch_plan
from .models import ExecutionReport, ExecutedPatch

__all__ = [
    "add_executor_subparser",
    "dispatch_executor",
    "execute_patch_plan",
    "ExecutionReport",
    "ExecutedPatch",
]
