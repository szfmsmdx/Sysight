"""Facade and orchestrator for deep Nsight Systems SQL analysis."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing

from .models import NsysFinding, NsysTrace, NvtxKernelAttribution
from .sql_comm import _sql_nccl_breakdown
from .sql_compute import _sql_gpu_idle_gaps, _sql_top_kernels
from .sql_memory import _sql_memory_bandwidth
from .sql_nvtx import (
    _sql_nvtx_hotspots,
    _sql_nvtx_layer_breakdown,
    attribute_kernels_to_nvtx,
)
from .sql_profile import _sql_profile_health
from .sql_root_cause import _sql_root_cause_analysis
from .sql_shared import _find_table
from .sql_sync import _sql_sync_cost

logger = logging.getLogger(__name__)


def run_deep_sql_analysis(findings: list[NsysFinding], trace: NsysTrace) -> None:
    """深度 SQL 分析入口，由 classify_bottlenecks() 调用。"""
    if not trace.sqlite_path:
        return

    try:
        with closing(sqlite3.connect(trace.sqlite_path)) as conn:
            conn.row_factory = sqlite3.Row
            all_tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            has_strings = "StringIds" in all_tables

            kernel_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_KERNEL")
            runtime_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_RUNTIME")
            memcpy_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_MEMCPY")
            memset_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_MEMSET")
            sync_tbl = _find_table(all_tables, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION")
            nvtx_tbl = _find_table(all_tables, "NVTX_EVENTS")

            if kernel_tbl:
                _sql_top_kernels(findings, conn, kernel_tbl, has_strings, trace.duration_ns)
                _sql_gpu_idle_gaps(
                    findings,
                    conn,
                    kernel_tbl,
                    runtime_tbl,
                    has_strings,
                    nvtx_tbl=nvtx_tbl,
                )
                if has_strings:
                    _sql_nccl_breakdown(findings, conn, kernel_tbl)
                try:
                    _sql_root_cause_analysis(
                        findings,
                        conn,
                        kernel_tbl,
                        runtime_tbl,
                        memcpy_tbl,
                        memset_tbl,
                        sync_tbl,
                        nvtx_tbl,
                        has_strings,
                        trace.duration_ns,
                    )
                except Exception as exc:
                    logger.debug("sql_root_cause_analysis 失败（非致命）：%s", exc)
                try:
                    _sql_profile_health(
                        findings,
                        conn,
                        kernel_tbl,
                        runtime_tbl,
                        memcpy_tbl,
                        nvtx_tbl,
                        has_strings,
                        trace.duration_ns,
                    )
                except Exception as exc:
                    logger.debug("sql_profile_health 失败（非致命）：%s", exc)

            if memcpy_tbl:
                _sql_memory_bandwidth(findings, conn, memcpy_tbl)

            if sync_tbl:
                _sql_sync_cost(findings, conn, sync_tbl, trace.duration_ns)

            if nvtx_tbl:
                _sql_nvtx_hotspots(findings, conn, nvtx_tbl, has_strings, trace.duration_ns)

            if kernel_tbl and runtime_tbl and nvtx_tbl:
                try:
                    _sql_nvtx_layer_breakdown(
                        findings,
                        conn,
                        kernel_tbl,
                        runtime_tbl,
                        nvtx_tbl,
                        has_strings,
                        trace.duration_ns,
                    )
                except Exception as exc:
                    logger.debug("sql_nvtx_layer_breakdown 失败（非致命）：%s", exc)
    except sqlite3.Error as exc:
        logger.warning("深度 SQL 分析失败（非致命）：%s", exc)


__all__ = [
    "NvtxKernelAttribution",
    "attribute_kernels_to_nvtx",
    "run_deep_sql_analysis",
    "_sql_gpu_idle_gaps",
    "_sql_memory_bandwidth",
    "_sql_nccl_breakdown",
    "_sql_nvtx_hotspots",
    "_sql_nvtx_layer_breakdown",
    "_sql_profile_health",
    "_sql_root_cause_analysis",
    "_sql_sync_cost",
    "_sql_top_kernels",
]
