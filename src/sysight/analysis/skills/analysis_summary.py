"""Conversation-style analysis summary skill."""

from __future__ import annotations

from sysight.analysis.report import format_analysis_report, run_analysis

from .base import Skill


def _run(prof, device=None, trim=None):
    return run_analysis(prof, device, trim)


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
