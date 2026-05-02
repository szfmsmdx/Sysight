"""Memory read/search tool registration (read-only for child agents)."""

from __future__ import annotations

__all__ = ["register_memory_tools"]


def register_memory_tools(registry) -> None:
    from sysight.tools.memory.search import SEARCH_TOOL
    from sysight.tools.memory.read import READ_TOOL

    registry.register(SEARCH_TOOL)
    registry.register(READ_TOOL)
