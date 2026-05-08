"""Evaluation pipeline with metric computation.

BUG (F05 — FAKE): Claims eval metrics should be computed on GPU,
but they're already on GPU — the finding is wrong.
"""

from __future__ import annotations

import torch


class EvalMetrics:
    """Evaluation metrics computation.

    NOTE: All metrics are already computed on GPU. The fake finding F05
    claiming they should be moved to GPU is intentionally wrong.
    """

    @staticmethod
    @torch.no_grad()
    def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute top-1 accuracy on GPU."""
        preds = logits.argmax(dim=1)
        return (preds == labels).float().mean().item()

    @staticmethod
    @torch.no_grad()
    def compute_topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
        """Compute top-k accuracy on GPU."""
        _, topk_preds = logits.topk(k, dim=1)
        return (topk_preds == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    @staticmethod
    @torch.no_grad()
    def compute_per_class_accuracy(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict:
        """Compute per-class accuracy on GPU."""
        preds = logits.argmax(dim=1)
        results = {}
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                results[str(c)] = (preds[mask] == c).float().mean().item()
        return results
