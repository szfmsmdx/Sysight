"""Training loop with data pipeline integration.

BUGS:
  F05: non_blocking=False on .to(device) — synchronous H2D transfer
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class TrainingLoop:
    """Training loop with data pipeline bottlenecks."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        log_interval: int = 50,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.log_interval = log_interval

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch.

        BUG F05: .to(device) without non_blocking=True — synchronous H2D.
        """
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(dataloader):
            # BUG F05: synchronous transfer — should use non_blocking=True
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.detach().item()

            if step % self.log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch} | Step {step:4d} | "
                    f"Loss: {loss.detach().item():.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

        return total_loss / len(dataloader)
