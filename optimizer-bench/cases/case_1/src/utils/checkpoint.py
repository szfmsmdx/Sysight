"""Checkpoint save/load utilities.

BUG (F06 — FAKE finding injected in artifacts):
  The checkpoint save is actually already asynchronous (uses threading),
  but the fake finding claims it blocks training.
"""

from __future__ import annotations

import threading
import torch
import torch.nn as nn
from pathlib import Path


class CheckpointManager:
    """Manages model checkpoint saving.

    NOTE: save() already uses async pattern — the fake finding F06
    claiming it blocks training is intentionally wrong.
    """

    def __init__(self, save_dir: str | Path, keep_last: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self._save_threads: list[threading.Thread] = []

    def save(self, model: nn.Module, optimizer, step: int, metrics: dict | None = None):
        """Save checkpoint asynchronously.

        Clones state to CPU then saves in background thread so training
        is not blocked.
        """
        cpu_state = {
            k: v.cpu().clone() for k, v in model.state_dict().items()
        }
        cpu_opt = {
            k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
            for k, v in optimizer.state_dict().items()
        }

        def _save():
            path = self.save_dir / f"checkpoint_step_{step:06d}.pt"
            torch.save({
                "model": cpu_state,
                "optimizer": cpu_opt,
                "step": step,
                "metrics": metrics or {},
            }, path)
            self._cleanup_old()

        t = threading.Thread(target=_save, daemon=True)
        t.start()
        self._save_threads.append(t)

    def _cleanup_old(self):
        """Remove old checkpoints beyond keep_last."""
        ckpts = sorted(self.save_dir.glob("checkpoint_step_*.pt"))
        if len(ckpts) > self.keep_last:
            for old in ckpts[:-self.keep_last]:
                old.unlink(missing_ok=True)

    def wait_all(self):
        """Wait for all pending saves to complete."""
        for t in self._save_threads:
            t.join()
        self._save_threads.clear()
