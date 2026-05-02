"""Scanner tool registration."""

from __future__ import annotations

__all__ = ["register_scanner_tools"]


def register_scanner_tools(registry) -> None:
    from sysight.tools.scanner.files import FILES_TOOL
    from sysight.tools.scanner.search import SEARCH_TOOL
    from sysight.tools.scanner.read import READ_TOOL
    from sysight.tools.scanner.callers import CALLERS_TOOL
    from sysight.tools.scanner.symbols import SYMBOLS_TOOL, CALLERS_SYMBOL_TOOL, CALLEES_TOOL, TRACE_TOOL
    from sysight.tools.scanner.variants import VARIANTS_TOOL

    registry.register(FILES_TOOL)
    registry.register(SEARCH_TOOL)
    registry.register(READ_TOOL)
    registry.register(CALLERS_TOOL)
    registry.register(SYMBOLS_TOOL)
    registry.register(CALLERS_SYMBOL_TOOL)
    registry.register(CALLEES_TOOL)
    registry.register(TRACE_TOOL)
    registry.register(VARIANTS_TOOL)
