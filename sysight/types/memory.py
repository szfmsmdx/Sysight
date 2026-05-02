"""Memory dataclasses. Zero internal dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryPage:
    """A single wiki page with YAML frontmatter and markdown body."""
    path: str                            # e.g. "workspaces/<ns>/overview.md"
    title: str = ""
    body: str = ""
    category: str | None = None          # C1-C7 for signal pages
    tags: list[str] = field(default_factory=list)
    scope: str = "workspace"             # "workspace" | "global" | "benchmark"
    source_run: str = ""                 # run_id that created this page
    created: str = ""                    # ISO timestamp
    content_hash: str = ""               # SHA256 of body


@dataclass
class MemoryBrief:
    """Compact memory summary for inclusion in LLM context (≤200 lines)."""
    namespace: str
    workspace_overview: str = ""
    top_experiences: list[str] = field(default_factory=list)      # ≤3
    recent_session_outcome: str = ""
    generated_at: str = ""
    total_lines: int = 0
