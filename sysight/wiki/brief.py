"""Memory brief builder — compact summary for LLM context injection (≤200 lines)."""

from __future__ import annotations

from pathlib import Path

from sysight.wiki.store import WikiRepository
from sysight.wiki.index import FTSIndex


def build_memory_brief(
    repo: WikiRepository | None = None,
    index: FTSIndex | None = None,
    namespace: str | None = None,
    max_experiences: int = 3,
) -> str:
    """Build a compact (≤200 line) memory summary for LLM system prompt.

    Includes:
      - Workspace overview (entry point, test commands, constraints)
      - Top-N relevant experiences
      - Recent session outcome
    """
    if repo is None:
        repo = WikiRepository()

    lines: list[str] = []
    lines.append("## Memory Brief")

    ns = namespace or "default"

    # Workspace overview
    overview = repo.read_page(f"workspaces/{ns}/overview.md")
    if overview:
        lines.append("### Workspace")
        overview_lines = overview.strip().split("\n")[:40]
        lines.extend(overview_lines)
        lines.append("")

    # Recent worklog
    worklog = repo.read_page(f"workspaces/{ns}/worklog.md")
    if worklog:
        lines.append("### Recent Activity")
        worklog_lines = worklog.strip().split("\n")[-30:]
        lines.extend(worklog_lines)
        lines.append("")

    # Top experiences
    experiences = repo.list_experiences()
    if experiences:
        lines.append("### Relevant Experiences")
        for exp in experiences[:max_experiences]:
            title = exp.get("title", exp.get("path", ""))
            cat = exp.get("category", "")
            cat_str = f" [{cat}]" if cat else ""
            lines.append(f"- {title}{cat_str}")

    result = "\n".join(lines)
    # Truncate to ≤200 lines
    result_lines = result.split("\n")
    if len(result_lines) > 200:
        result = "\n".join(result_lines[:200])
        result += "\n... (truncated)"

    return result
