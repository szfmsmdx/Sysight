"""NVTX to kernel mapping skill adapted from nsys-ai."""

from __future__ import annotations

from sysight.analysis.nvtx import nvtx_kernel_map

from .base import Skill


def _run(prof, device=None, trim=None):
    target = device if device is not None else (prof.meta.devices[0] if prof.meta.devices else 0)
    return nvtx_kernel_map(prof, target, trim, limit=30)


def _format(rows) -> str:
    if not rows:
        return "(No NVTX to kernel mappings found)"
    lines = [
        "── NVTX → Kernel Mapping ──",
        f"{'NVTX Range':<50s}  {'Kernel':<50s}  {'Start(ms)':>10s}  {'End(ms)':>10s}",
        "─" * 126,
    ]
    for row in rows:
        nvtx = row["nvtx_path"] or row["nvtx_text"] or "(unnamed)"
        kernel = row["kernel_name"] or "(unknown)"
        if len(nvtx) > 48:
            nvtx = nvtx[:45] + "..."
        if len(kernel) > 48:
            kernel = kernel[:45] + "..."
        lines.append(
            f"{nvtx:<50s}  {kernel:<50s}  {row['start_ms']:>10.3f}  {row['end_ms']:>10.3f}"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="nvtx_kernel_map",
    title="NVTX to Kernel Mapping",
    description="Maps NVTX code regions to the kernels that execute within them.",
    runner=_run,
    formatter=_format,
)
