"""Checkpoint manager with synchronous save.

BUG (F04): torch.save() called synchronously in the training loop,
blocking GPU computation while serializing to disk.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path


class CheckpointManager:
    """Manages model checkpoint saving.

    BUG F04: Synchronous torch.save() blocks training.
    """

    def __init__(self, save_dir: str | Path, keep_last: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    def save(self, model: nn.Module, optimizer, epoch: int, metrics: dict | None = None):
        """Save checkpoint synchronously.

        BUG F04: torch.save() is synchronous — blocks GPU while writing to disk.
        Should clone to CPU and save in background thread.
        """
        path = self.save_dir / f"checkpoint_epoch_{epoch:03d}.pt"

        # BUG F04: synchronous save blocks training
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics or {},
        }, path)

        self._cleanup_old()

    def _cleanup_old(self):
        """Remove old checkpoints beyond keep_last."""
        ckpts = sorted(self.save_dir.glob("checkpoint_epoch_*.pt"))
        if len(ckpts) > self.keep_last:
            for old in ckpts[:-self.keep_last]:
                old.unlink(missing_ok=True)
