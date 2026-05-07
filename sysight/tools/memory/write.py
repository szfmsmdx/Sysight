"""memory write tools — parent/analyzer-learn wiki mutations."""

from __future__ import annotations

from dataclasses import dataclass

from sysight.tools.registry import ToolDef


@dataclass
class MemoryWriteResult:
    path: str
    action: str
    ok: bool = True


def write(path: str, content: str, title: str = "", category: str | None = None,
          scope: str = "workspace") -> MemoryWriteResult:
    """Create or replace a wiki page."""
    from sysight.wiki.store import WikiRepository
    repo = WikiRepository()
    repo.write_page(path, content, title=title, category=category, scope=scope)
    return MemoryWriteResult(path=path, action="write")


def append(path: str, content: str) -> MemoryWriteResult:
    """Append markdown to a wiki page."""
    from sysight.wiki.store import WikiRepository
    repo = WikiRepository()
    repo.append_page(path, content)
    return MemoryWriteResult(path=path, action="append")


def replace(path: str, old: str, new: str, count: int = 1) -> MemoryWriteResult:
    """Replace text inside a wiki page."""
    from sysight.wiki.store import WikiRepository
    repo = WikiRepository()
    repo.replace_in_page(path, old, new, count=int(count))
    return MemoryWriteResult(path=path, action="replace")


WRITE_TOOL = ToolDef(
    name="memory_write",
    description="Create or replace a Sysight wiki page",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Wiki path, e.g. workspaces/<ns>/overview.md"},
            "content": {"type": "string", "description": "Complete markdown body"},
            "title": {"type": "string", "default": ""},
            "category": {"type": "string"},
            "scope": {"type": "string", "default": "workspace"},
        },
        "required": ["path", "content"],
    },
    fn=write,
    read_only=False,
)


APPEND_TOOL = ToolDef(
    name="memory_append",
    description="Append markdown to a Sysight wiki page",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Wiki path, e.g. workspaces/<ns>/overview.md"},
            "content": {"type": "string", "description": "Markdown to append"},
        },
        "required": ["path", "content"],
    },
    fn=append,
    read_only=False,
)


REPLACE_TOOL = ToolDef(
    name="memory_replace",
    description="Replace text inside a Sysight wiki page",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Wiki path, e.g. workspaces/<ns>/overview.md"},
            "old": {"type": "string", "description": "Existing text to replace"},
            "new": {"type": "string", "description": "Replacement text"},
            "count": {"type": "integer", "default": 1},
        },
        "required": ["path", "old", "new"],
    },
    fn=replace,
    read_only=False,
)
