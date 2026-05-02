"""Sandbox execution tool registration.

Sandbox tools are the ONLY write-capable tools. They operate on isolated git
worktrees, never on the user's working directory.
"""

from __future__ import annotations

__all__ = ["register_sandbox_tools"]


def register_sandbox_tools(registry) -> None:
    from sysight.tools.sandbox.create import CREATE_TOOL
    from sysight.tools.sandbox.destroy import DESTROY_TOOL
    from sysight.tools.sandbox.exec import EXEC_TOOL
    from sysight.tools.sandbox.apply import APPLY_TOOL
    from sysight.tools.sandbox.validate import VALIDATE_TOOL
    from sysight.tools.sandbox.measure import MEASURE_TOOL
    from sysight.tools.sandbox.commit import COMMIT_TOOL
    from sysight.tools.sandbox.revert import REVERT_TOOL

    registry.register(CREATE_TOOL)
    registry.register(DESTROY_TOOL)
    registry.register(EXEC_TOOL)
    registry.register(APPLY_TOOL)
    registry.register(VALIDATE_TOOL)
    registry.register(MEASURE_TOOL)
    registry.register(COMMIT_TOOL)
    registry.register(REVERT_TOOL)
