"""Theoretical FLOPs skill adapted from nsys-ai."""

from __future__ import annotations

from sysight.analysis.mfu import compute_theoretical_flops, format_theoretical_flops

from .base import Skill


def _run(
    prof,
    device=None,
    trim=None,
    operation: str = "",
    hidden_dim: int = 0,
    seq_len: int = 0,
    num_layers: int = 1,
    ffn_dim: int | None = None,
    batch_size: int = 1,
    multiplier: int = 1,
    m_dim: int = 0,
    n_dim: int = 0,
    k_dim: int = 0,
):
    return compute_theoretical_flops(
        operation,
        hidden_dim=int(hidden_dim),
        seq_len=int(seq_len),
        num_layers=int(num_layers),
        ffn_dim=int(ffn_dim) if ffn_dim is not None else None,
        batch_size=int(batch_size),
        multiplier=int(multiplier),
        m_dim=int(m_dim),
        n_dim=int(n_dim),
        k_dim=int(k_dim),
    )


SKILL = Skill(
    name="theoretical_flops",
    title="Theoretical FLOPs Calculator",
    description="Computes exact theoretical FLOPs for transformer operations and generic GEMMs.",
    runner=_run,
    formatter=format_theoretical_flops,
)
