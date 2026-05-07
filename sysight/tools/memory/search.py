"""memory_search — Search the knowledge wiki via FTS."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sysight.tools.registry import ToolDef


@dataclass
class MemorySearchMatch:
    path: str
    title: str = ""
    snippet: str = ""
    section_title: str = ""
    score: float = 0.0


@dataclass
class MemorySearchResult:
    query: str
    total: int = 0
    matches: list[MemorySearchMatch] = field(default_factory=list)


_index: object | None = None


def _get_index() -> object:
    global _index
    if _index is None:
        from sysight.wiki.index import FTSIndex
        db_path = Path.cwd() / ".sysight" / "runs" / "runs.sqlite"
        wiki_dir = Path.cwd() / ".sysight" / "memory" / "wiki"
        _index = FTSIndex(db_path, wiki_dir)
    return _index


def search(query: str, namespace: str | None = None, limit: int = 10) -> MemorySearchResult:
    """Search the knowledge wiki for relevant pages."""
    from sysight.wiki.index import FTSIndex
    index: FTSIndex = _get_index()  # type: ignore[assignment]
    results = index.search(query, namespace=namespace, limit=limit)
    return MemorySearchResult(
        query=query,
        total=len(results),
        matches=[
            MemorySearchMatch(
                path=r.path,
                snippet=r.snippet,
                section_title=r.section_title,
                score=r.score,
            )
            for r in results
        ],
    )


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
