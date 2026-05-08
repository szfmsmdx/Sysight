"""DDP training loop with gradient accumulation.

BUGS:
  F01: all-reduce happens on every micro-batch during gradient accumulation
       (missing no_sync() context manager)
  F03: unnecessary barrier() at start of each epoch
  F04: TF32 not enabled
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


class DDPTrainingLoop:
    """DDP training loop with communication inefficiencies."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        accum_steps: int = 4,
        log_interval: int = 20,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.accum_steps = accum_steps
        self.log_interval = log_interval

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch.

        BUG F01: no_sync() not used — all-reduce on every micro-batch
        BUG F03: barrier() at epoch start is unnecessary
        BUG F04: TF32 not enabled
        """
        self.model.train()

        # BUG F03: unnecessary barrier() at start of each epoch
        if dist.is_initialized():
            dist.barrier()

        total_loss = 0.0
        start_time = time.time()
        self.optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            images = batch["images"].to(self.device)
            labels = batch["labels"].to(self.device)

            # BUG F01: all-reduce on every micro-batch (no no_sync)
            loss = self.model(images)
            loss = F.cross_entropy(loss, labels)
            loss = loss / self.accum_steps
            loss.backward()

            if (step + 1) % self.accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accum_steps

            if step % self.log_interval == 0 and self._is_main():
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch} | Step {step:4d} | "
                    f"Loss: {loss.item() * self.accum_steps:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

        return total_loss / len(dataloader)

    @staticmethod
    def _is_main() -> bool:
        if dist.is_initialized():
            return dist.get_rank() == 0
        return True
