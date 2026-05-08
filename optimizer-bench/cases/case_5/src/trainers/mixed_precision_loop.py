"""Mixed precision training loop with gradient accumulation.

BUGS:
  F01: AMP not used — all FP32, missing autocast + GradScaler
  F02: loss divided by accum_steps inside autocast — precision loss
  F03: optimizer.step() called every micro-batch instead of after accumulation
  F04: zero_grad() called at wrong time — resets accumulated gradients
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class MixedPrecisionLoop:
    """Training loop with mixed precision issues.

    BUG F01: No AMP — all computation in FP32, wasting memory and throughput.
    BUG F02: Loss scaling should happen outside autocast for numerical stability.
    BUG F03: optimizer.step() called every micro-batch — defeats accumulation.
    BUG F04: zero_grad() called at wrong time.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        accum_steps: int = 8,
        log_interval: int = 20,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.accum_steps = accum_steps
        self.log_interval = log_interval

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch.

        BUG F01: No autocast — all FP32.
        BUG F02: loss / accum_steps inside forward — precision issue.
        BUG F03: step() every micro-batch.
        BUG F04: zero_grad() timing wrong.
        """
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(dataloader):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            # BUG F01: No autocast context — all FP32
            logits = self.model(images)
            # BUG F02: division inside forward — should be outside
            loss = F.cross_entropy(logits, labels) / self.accum_steps
            loss.backward()

            # BUG F03: step() every micro-batch — defeats accumulation
            # BUG F04: zero_grad() after step resets accumulated grads
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.detach().item() * self.accum_steps

            if step % self.log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch} | Step {step:4d} | "
                    f"Loss: {loss.detach().item() * self.accum_steps:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

        return total_loss / len(dataloader)
