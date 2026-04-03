"""Kernel Grid/Block dimension and resource usage summary skill."""

from __future__ import annotations

from sysight.analysis.queries import kernel_grid_block_summary

from .base import Skill


def _run(prof, device=None, trim=None):
    target = device if device is not None else (prof.meta.devices[0] if prof.meta.devices else 0)
    return kernel_grid_block_summary(prof, target, trim, limit=15)


def _short_name(name: str, max_len: int = 42) -> str:
    if not name:
        return "(unknown)"
    if len(name) > max_len:
        return name[:max_len - 3] + "..."
    return name


def _fmt_dim(x, y, z) -> str:
    """Format a 3D grid/block as compact string, dropping trailing 1s."""
    dims = [int(x or 1), int(y or 1), int(z or 1)]
    # Drop trailing 1s for readability
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()
    return "×".join(str(d) for d in dims)


def _format(rows: list[dict]) -> str:
    if not rows:
        return "(No kernel dimension data found)"

    lines = [
        "── Kernel Grid/Block 维度及资源统计（Top 15，按总时间排序）──",
        "",
        f"  {'Kernel':<44s}  {'次数':>5s}  {'总时(ms)':>9s}  "
        f"{'Grid(xyz)':>14s}  {'Block(xyz)':>14s}  {'Thr/Blk':>7s}  "
        f"{'Reg':>4s}  {'sShMem(KB)':>10s}  {'dShMem(KB)':>10s}",
        "  " + "─" * 130,
    ]

    for row in rows:
        name = _short_name(row.get("kernel_name") or "")
        grid = _fmt_dim(row.get("avg_grid_x"), row.get("avg_grid_y"), row.get("avg_grid_z"))
        block = _fmt_dim(row.get("avg_block_x"), row.get("avg_block_y"), row.get("avg_block_z"))
        tpb = int(row.get("threads_per_block") or 0)
        reg = int(row.get("registers_per_thread") or 0)
        static_kb = (row.get("static_smem_bytes") or 0) / 1024
        dynamic_kb = (row.get("dynamic_smem_bytes") or 0) / 1024
        warn = " ⚠" if row.get("low_occupancy_hint") else ""
        lines.append(
            f"  {name:<44s}  {row['invocations']:>5d}  {row['total_ms']:>9.2f}  "
            f"{grid:>14s}  {block:>14s}  {tpb:>7d}{warn:<2s}  "
            f"{reg:>4d}  {static_kb:>10.1f}  {dynamic_kb:>10.1f}"
        )

    lines += [
        "",
        "  说明：",
        "  · Grid/Block 维度为该 kernel 所有调用的均值，格式为 X×Y（Z=1时省略）",
        "  · Thr/Blk = blockX×blockY×blockZ；⚠ 表示 ≤64 线程/块，存在低 occupancy 风险",
        "  · sShMem = 静态共享内存；dShMem = 动态共享内存（均为均值）",
    ]

    return "\n".join(lines)


SKILL = Skill(
    name="kernel_grid_block",
    title="Kernel Grid/Block 维度统计",
    description="按 kernel 统计 Grid/Block 维度、线程数、寄存器用量和共享内存使用，识别低 occupancy 风险。",
    runner=_run,
    formatter=_format,
)
