"""Tool registry: ToolDef, ToolPolicy, ToolResult, ToolRegistry.

Registry pattern — all tools are registered once and discovered by name.
ToolPolicy controls which tools are available to each pipeline phase.
"""

from __future__ import annotations

import fnmatch
import time
from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class ToolDef:
    """Definition of a single callable tool."""
    name: str                                     # "scanner_search"
    description: str
    parameters: dict                              # JSON Schema
    fn: Callable
    read_only: bool = True
    max_calls_per_task: int = 50


@dataclass
class ToolPolicy:
    """Access policy for a pipeline phase."""
    allowed_tools: set[str] = field(default_factory=set)   # "scanner_*" supported
    read_only: bool = True
    max_calls_per_task: int = 50
    max_wall_seconds: int = 600
    path_containment: dict[str, str] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result of executing a tool."""
    tool_name: str
    status: str                                   # "ok" | "error" | "policy_denied"
    data: Any = None
    error: str = ""
    elapsed_ms: float = 0


class ToolRegistry:
    """Central registry for all Sysight tools."""

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}
        self._call_counts: dict[str, int] = {}

    def register(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def execute(self, name: str, args: dict, policy: ToolPolicy) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(tool_name=name, status="error", error=f"Unknown tool: {name}")

        if not self._tool_allowed(name, policy):
            return ToolResult(tool_name=name, status="policy_denied", error=f"Tool {name} not allowed by policy")

        if policy.read_only and not tool.read_only:
            return ToolResult(tool_name=name, status="policy_denied", error=f"Tool {name} is not read-only")

        self._call_counts[name] = self._call_counts.get(name, 0) + 1
        if self._call_counts[name] > tool.max_calls_per_task:
            return ToolResult(tool_name=name, status="policy_denied", error="Max calls exceeded")

        t0 = time.monotonic()
        try:
            data = tool.fn(**args)
            elapsed_ms = (time.monotonic() - t0) * 1000
            return ToolResult(tool_name=name, status="ok", data=data, elapsed_ms=elapsed_ms)
        except Exception as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            return ToolResult(tool_name=name, status="error", error=str(e), elapsed_ms=elapsed_ms)

    def reset_call_counts(self) -> None:
        self._call_counts.clear()

    def list_read_only(self) -> list[ToolDef]:
        return [t for t in self._tools.values() if t.read_only]

    def list_for_policy(self, policy: ToolPolicy) -> list[ToolDef]:
        return [t for t in self._tools.values() if self._tool_allowed(t.name, policy)]

    def as_openai_tools(self, policy: ToolPolicy) -> list[dict]:
        return [
            {"type": "function", "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }}
            for t in self.list_for_policy(policy)
        ]

    def as_anthropic_tools(self, policy: ToolPolicy) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in self.list_for_policy(policy)
        ]

    def _tool_allowed(self, name: str, policy: ToolPolicy) -> bool:
        for allowed in policy.allowed_tools:
            if fnmatch.fnmatch(name, allowed):
                return True
        return False


# Pre-built policies for each pipeline phase
ANALYZE_POLICY = ToolPolicy(
    allowed_tools={"scanner_*", "nsys_sql_*", "memory_search", "memory_read", "classify"},
    read_only=True,
)

OPTIMIZE_POLICY = ToolPolicy(
    allowed_tools={"scanner_*", "sandbox_*"},
    read_only=False,
)

LEARN_POLICY = ToolPolicy(
    allowed_tools={"memory_search", "memory_read"},
    read_only=True,
)
