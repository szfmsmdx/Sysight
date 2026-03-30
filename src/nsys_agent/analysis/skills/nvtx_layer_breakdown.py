"""NVTX layer breakdown skill adapted from nsys-ai."""

from __future__ import annotations

from nsys_agent.analysis.nvtx import nvtx_layer_breakdown

from .base import Skill


def _run(prof, device=None, trim=None):
    target = device if device is not None else (prof.meta.devices[0] if prof.meta.devices else 0)
    return nvtx_layer_breakdown(prof, target, trim, limit=20)


def _format(rows) -> str:
    if not rows:
        return "(No NVTX regions with attributed kernels found)"
    if "error" in rows[0]:
        return f"Error: {rows[0]['error']}"

    lines = [
        "── NVTX Region GPU Time Breakdown ──",
        f"{'NVTX Region':<40s}  {'Depth':>5s}  {'Kernels':>7s}  {'Total(ms)':>10s}"
        f"  {'Compute':>9s}  {'NCCL':>9s}  {'NCCL%':>6s}",
        "─" * 96,
    ]
    for row in rows:
        name = row.get("nvtx_path") or row.get("nvtx_region") or "(unnamed)"
        if len(name) > 38:
            name = "..." + name[-35:]
        lines.append(
            f"{name:<40s}  {row['nvtx_depth']:>5d}  {row['kernel_count']:>7d}  {row['total_gpu_ms']:>10.2f}"
            f"  {row['compute_ms']:>9.2f}  {row['nccl_ms']:>9.2f}  {row['nccl_pct']:>5.1f}%"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="nvtx_layer_breakdown",
    title="NVTX Region GPU Time Breakdown",
    description="Attributes GPU time to NVTX regions and shows compute versus NCCL split per code region.",
    runner=_run,
    formatter=_format,
)
