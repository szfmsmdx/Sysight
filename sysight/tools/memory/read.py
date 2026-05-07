"""memory_read — Read a wiki page from the knowledge store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sysight.tools.registry import ToolDef


@dataclass
class MemoryReadResult:
    path: str
    title: str = ""
    body: str = ""
    category: str | None = None
    tags: list[str] | None = None
    found: bool = False


def read(path: str, namespace: str | None = None) -> MemoryReadResult:
    """Read a wiki page from the knowledge store by path."""
    from sysight.wiki.store import WikiRepository
    repo = WikiRepository()
    content = repo.read_page(path)
    if content is None:
        return MemoryReadResult(path=path, found=False)
    title = ""
    body = content
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            for line in content[3:end].split("\n"):
                if line.startswith("title:"):
                    title = line.split(":", 1)[1].strip()
            body = content[end + 3:]
    return MemoryReadResult(path=path, title=title, body=body.strip(), found=True)


READ_TOOL = ToolDef(
    name="memory_read",
    description="Read a specific page from the Sysight knowledge wiki by path",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Wiki page path, e.g. 'workspaces/<ns>/overview.md'"},
            "namespace": {"type": "string", "description": "Optional namespace"},
        },
        "required": ["path"],
    },
    fn=read,
    read_only=True,
)
