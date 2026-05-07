"""Memory read/search tool registration (read-only for child agents)."""

from __future__ import annotations

__all__ = ["register_memory_tools"]


def register_memory_tools(registry) -> None:
    from sysight.tools.memory.search import SEARCH_TOOL
    from sysight.tools.memory.read import READ_TOOL
    from sysight.tools.memory.write import WRITE_TOOL, APPEND_TOOL, REPLACE_TOOL

    registry.register(SEARCH_TOOL)
    registry.register(READ_TOOL)
    registry.register(WRITE_TOOL)
    registry.register(APPEND_TOOL)
    registry.register(REPLACE_TOOL)
