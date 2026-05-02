"""Nsight Systems SQLite analysis tool registration."""

from __future__ import annotations

__all__ = ["register_nsys_sql_tools"]


def register_nsys_sql_tools(registry) -> None:
    from sysight.tools.nsys_sql.kernels import KERNELS_TOOL
    from sysight.tools.nsys_sql.sync import SYNC_TOOL
    from sysight.tools.nsys_sql.memcpy import MEMCPY_TOOL
    from sysight.tools.nsys_sql.nccl import NCCL_TOOL
    from sysight.tools.nsys_sql.overlap import OVERLAP_TOOL
    from sysight.tools.nsys_sql.gaps import GAPS_TOOL
    from sysight.tools.nsys_sql.launch import LAUNCH_TOOL

    registry.register(KERNELS_TOOL)
    registry.register(SYNC_TOOL)
    registry.register(MEMCPY_TOOL)
    registry.register(NCCL_TOOL)
    registry.register(OVERLAP_TOOL)
    registry.register(GAPS_TOOL)
    registry.register(LAUNCH_TOOL)
