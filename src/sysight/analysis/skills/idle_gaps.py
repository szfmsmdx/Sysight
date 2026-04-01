"""Adapted idle-gap skill from nsys-ai."""

from __future__ import annotations

from sysight.analysis.queries import detect_idle_gaps

from .base import Skill


def _run(prof, device=None, trim=None):
    target = device if device is not None else (prof.meta.devices[0] if prof.meta.devices else 0)
    return detect_idle_gaps(prof, target, trim)


def _format(result) -> str:
    rows = result["rows"]
    summary = result["summary"]
    if not rows:
        return "(No significant GPU idle gaps found)"

    lines = ["-- GPU Idle Gaps --"]
    if summary:
        lines.append(
            f"Total: {summary['gap_count']} gaps, {summary['total_idle_ms']:.1f}ms idle "
            f"({summary['pct_of_profile']}% of profile)"
        )
        lines.append(
            f"Distribution: {summary['gaps_1_5ms']} x 1-5ms, {summary['gaps_5_50ms']} x 5-50ms, "
            f"{summary['gaps_gt50ms']} x >50ms"
        )
        lines.append("")
    for row in rows[:10]:
        attr = row.get("attribution") or {}
        lines.append(
            f"- stream {row['streamId']}: {row['gap_ns'] / 1e6:.2f}ms after {row['before_kernel']} "
            f"[{attr.get('category', 'unclassified')}]"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="idle_gaps",
    title="GPU Idle Gaps",
    description="Detects GPU idle gaps and labels likely causes such as sync stalls or blocked transfers.",
    runner=_run,
    formatter=_format,
)
