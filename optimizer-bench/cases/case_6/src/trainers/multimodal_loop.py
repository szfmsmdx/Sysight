"""Training loop for multi-modal model.

BUGS:
  F03: eval() called inside no_grad but model still tracks gradients for
       some operations due to missing inference_mode
  F04: Redundant forward pass in eval — computes all three logit types
       even when only fusion is needed
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class MultiModalTrainingLoop:
    """Training loop for multi-modal model."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        log_interval: int = 20,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.log_interval = log_interval

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            images = batch["images"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            outputs = self.model(input_ids, images)

            # Multi-task loss
            loss = (
                F.cross_entropy(outputs["text_logits"], labels) +
                F.cross_entropy(outputs["image_logits"], labels) +
                F.cross_entropy(outputs["fusion_logits"], labels)
            )

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

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict:
        """Run evaluation.

        BUG F03: Uses no_grad instead of inference_mode — version counter
        tracking still runs.
        BUG F04: Computes all three logit types even though only fusion
        accuracy is reported.
        """
        self.model.eval()
        correct = 0
        total = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            images = batch["images"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            # BUG F04: computes text, image, AND fusion logits — only fusion needed
            outputs = self.model(input_ids, images)

            preds = outputs["fusion_logits"].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return {"accuracy": correct / total if total > 0 else 0.0}
