"""Tool registry and tool implementations for Sysight.

Import pattern:
  from sysight.tools.registry import ToolRegistry, ToolDef, ToolPolicy
  from sysight.tools import register_all_tools

  registry = ToolRegistry()
  register_all_tools(registry)
"""

from __future__ import annotations

__all__ = ["register_all_tools"]


def register_all_tools(registry) -> None:
    """Register all available tools into the given ToolRegistry."""
    from sysight.tools.scanner import register_scanner_tools
    from sysight.tools.nsys_sql import register_nsys_sql_tools
    from sysight.tools.sandbox import register_sandbox_tools
    from sysight.tools.memory import register_memory_tools

    register_scanner_tools(registry)
    register_nsys_sql_tools(registry)
    register_sandbox_tools(registry)
    register_memory_tools(registry)
