"""Training and validation loops.

BUGS:
  F02: loss.item() called every training step — forces CPU-GPU sync
  F03: validate() missing torch.no_grad() — builds computation graph
  F04: validate() uses torch.no_grad() but should use torch.inference_mode()
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class TrainingLoop:
    """Main training loop with embedded performance issues."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        log_interval: int = 10,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.log_interval = log_interval

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch.

        BUG F02: loss.item() called every step, forcing CPU-GPU sync.
        """
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # BUG F02: loss.item() called every step — forces CPU-GPU sync
            total_loss += loss.item()

            if step % self.log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch} | Step {step:4d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader) -> float:
        """Run validation.

        BUG F03: missing torch.no_grad() — builds computation graph
        BUG F04: should use torch.inference_mode() instead of no_grad()
        """
        self.model.eval()
        total_loss = 0.0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            total_loss += loss.item()

        return total_loss / len(dataloader)
