"""DataLoader with DistributedSampler."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class SyntheticImageDataset(Dataset):
    """Synthetic image dataset for benchmarking."""

    def __init__(self, num_samples: int, num_classes: int = 10, seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        g = torch.Generator()
        g.manual_seed(seed)
        self.images = torch.randn(num_samples, 3, 32, 32, generator=g)
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=g)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return {
            "images": self.images[idx],
            "labels": self.labels[idx],
        }


def create_distributed_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader with DistributedSampler.

    NOTE: DistributedSampler.set_epoch() is called in the training loop
    to ensure different shuffling per epoch.
    """
    import torch.distributed as dist
    sampler = DistributedSampler(dataset, shuffle=shuffle) if dist.is_initialized() else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False,
    )
