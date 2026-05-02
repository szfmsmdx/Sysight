"""MCP server adapter for Sysight.

Exposes read-only tools and memory read to external MCP clients.
Executable tools (sandbox) and memory writes are NOT exposed by default.
"""

from __future__ import annotations


__all__ = ["create_mcp_server"]


def create_mcp_server(registry, knowledge):
    """Create an MCP server exposing selected capabilities.

    TODO: Stage 9 — MCP adapter over CapabilityRegistry.
    """
    raise NotImplementedError("TODO: Stage 9")
