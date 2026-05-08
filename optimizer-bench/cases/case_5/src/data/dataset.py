"""Synthetic image dataset."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticImageDataset(Dataset):
    """Synthetic image dataset for ViT training."""

    def __init__(self, num_samples: int, image_size: int = 224, num_classes: int = 100, seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        g = torch.Generator()
        g.manual_seed(seed)
        self.images = torch.rand(num_samples, 3, image_size, image_size, generator=g)
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=g)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return {"image": self.images[idx], "label": self.labels[idx]}
