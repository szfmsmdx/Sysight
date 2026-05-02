"""memory_search — Search the knowledge wiki via FTS.

TODO: Stage 3 — wire up to knowledge/index.py FTS index.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sysight.tools.registry import ToolDef


@dataclass
class MemorySearchMatch:
    path: str
    title: str = ""
    snippet: str = ""
    score: float = 0.0


@dataclass
class MemorySearchResult:
    query: str
    total: int = 0
    matches: list[MemorySearchMatch] = field(default_factory=list)


def search(query: str, namespace: str | None = None, limit: int = 10) -> MemorySearchResult:
    """Search the knowledge wiki for relevant pages."""
    raise NotImplementedError("TODO: Stage 3 — wire to knowledge FTS")


SEARCH_TOOL = ToolDef(
    name="memory_search",
    description="Search the Sysight knowledge wiki (workspace wiki, experience pages, signal pages) for relevant information",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "namespace": {"type": "string", "description": "Optional namespace to restrict search to"},
            "limit": {"type": "integer", "default": 10, "description": "Maximum number of results"},
        },
        "required": ["query"],
    },
    fn=search,
    read_only=True,
)
