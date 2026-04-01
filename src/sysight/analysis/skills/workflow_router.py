"""Workflow routing skill for profile-only vs workspace-aware agent paths."""

from __future__ import annotations

from sysight.workflow import format_workflow_route, resolve_workflow_route

from .base import Skill


def _run(prof, device=None, trim=None, workspace=None, program=None, profile_path=None):
    return resolve_workflow_route(
        profile_path=profile_path or getattr(prof, "path", ""),
        workspace_root=workspace,
        program_path=program,
    )


def _format(route) -> str:
    return format_workflow_route(route)


SKILL = Skill(
    name="workflow_router",
    title="Workflow Router",
    description="Resolves whether the current request should run as profile-only or workspace-aware analysis.",
    runner=_run,
    formatter=_format,
)
