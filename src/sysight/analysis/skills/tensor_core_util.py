"""Tensor Core utilization estimation skill."""

from __future__ import annotations

from sysight.analysis.queries import tensor_core_summary

from .base import Skill


def _run(prof, device=None, trim=None):
    target = device if device is not None else (prof.meta.devices[0] if prof.meta.devices else 0)
    return tensor_core_summary(prof, target, trim)


def _format(result: dict) -> str:
    if not result or result.get("total_count", 0) == 0:
        return "(No kernel data found)"

    lines = [
        "── Tensor Core 利用率估算 ──",
        "",
        f"  总 Kernel 时间       : {result['total_ms']:.1f} ms  ({result['total_count']} 次调用)",
        f"  Tensor Core 时间     : {result['tc_ms']:.1f} ms  ({result['tc_time_pct']:.1f}%)",
        f"  非 Tensor Core 时间  : {result['non_tc_ms']:.1f} ms  ({100 - result['tc_time_pct']:.1f}%)",
        f"  TC Kernel 调用次数   : {result['tc_kernel_count']}  ({result['tc_count_pct']:.1f}%)",
        "",
        "  注：基于 kernel 名称模式匹配（cuBLAS HMMA/WMMA、FlashAttention、cuDNN NHWC、cutlass 等）",
        "",
    ]

    if result["top_tc_kernels"]:
        lines.append("  Top Tensor Core Kernels:")
        lines.append(f"  {'Kernel':<55s}  {'次数':>6s}  {'总时(ms)':>10s}  {'占比':>6s}")
        lines.append("  " + "─" * 85)
        for r in result["top_tc_kernels"]:
            name = r["name"] or ""
            if len(name) > 53:
                name = name[:50] + "..."
            lines.append(
                f"  {name:<55s}  {r['count']:>6d}  {r['total_ms']:>10.1f}  {r['pct']:>5.1f}%"
            )

    return "\n".join(lines)


SKILL = Skill(
    name="tensor_core_util",
    title="Tensor Core 利用率",
    description="通过 kernel 名称模式估算 Tensor Core 使用时间占比及主要 TC Kernel 列表。",
    runner=_run,
    formatter=_format,
)
