"""Conversation-style analysis summary skill."""

from __future__ import annotations

from sysight.analysis.report import format_analysis_report, run_analysis
from sysight.workflow import resolve_workflow_route

from .base import Skill


def _run(prof, device=None, trim=None, workspace=None, program=None, profile_path=None):
    data = run_analysis(prof, device, trim)
    route = resolve_workflow_route(
        profile_path=profile_path or getattr(prof, "path", ""),
        workspace_root=workspace,
        program_path=program,
    )
    data["workflow_route"] = route.to_dict()
    return data


def _format(data) -> str:
    payload = dict(data)
    resolved_profile = getattr(payload.get("profile"), "path", "")
    payload.setdefault("requested_profile_path", resolved_profile)
    payload.setdefault("resolved_profile_path", resolved_profile)
    return format_analysis_report(payload)


SKILL = Skill(
    name="analysis_summary",
    title="Analysis Summary",
    description="Runs the default analysis pipeline and formats it as 结论 / 问题 / 下一步行动建议.",
    runner=_run,
    formatter=_format,
)
