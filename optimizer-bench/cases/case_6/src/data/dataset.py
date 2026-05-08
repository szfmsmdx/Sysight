"""Synthetic multi-modal dataset."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticMultiModalDataset(Dataset):
    """Synthetic dataset with text tokens and images."""

    def __init__(
        self,
        num_samples: int,
        vocab_size: int,
        text_len: int = 64,
        image_size: int = 224,
        num_classes: int = 50,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples

        g = torch.Generator()
        g.manual_seed(seed)

        self.input_ids = torch.randint(0, vocab_size, (num_samples, text_len), generator=g)
        self.images = torch.rand(num_samples, 3, image_size, image_size, generator=g)
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=g)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.input_ids[idx],
            "images": self.images[idx],
            "labels": self.labels[idx],
        }
