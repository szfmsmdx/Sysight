"""memory_read — Read a wiki page from the knowledge store.

TODO: Stage 3 — wire up to knowledge/store.py WikiRepository.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    raise NotImplementedError("TODO: Stage 3 — wire to knowledge store")


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
