"""Distributed communication utilities.

BUG (F02): sync_predictions uses all_gather + local reduction instead
of all_reduce, sending N times more data than necessary.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def sync_predictions(logits: torch.Tensor) -> torch.Tensor:
    """Synchronize predictions across all ranks.

    BUG F02: Uses all_gather (N× data) + local mean instead of all_reduce.
    For world_size=8, this sends 8× more data than needed.
    """
    if not dist.is_initialized():
        return logits

    world_size = dist.get_world_size()

    # BUG F02: all_gather sends N× data instead of 1× with all_reduce
    gathered = [torch.zeros_like(logits) for _ in range(world_size)]
    dist.all_gather(gathered, logits)
    return torch.stack(gathered).mean(dim=0)


def sync_metrics(loss: torch.Tensor) -> float:
    """Synchronize a scalar metric across ranks."""
    if not dist.is_initialized():
        return loss.item()

    world_size = dist.get_world_size()
    loss_tensor = loss.detach().clone()
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    return (loss_tensor / world_size).item()
