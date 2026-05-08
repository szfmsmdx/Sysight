"""DataLoader factory.

BUG F03: pin_memory not enabled — CPU→GPU transfers are synchronous.
"""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 8,
) -> DataLoader:
    """Create a DataLoader.

    BUG F03: pin_memory=False (default) — GPU transfers are synchronous.
    Should set pin_memory=True for async H2D transfers.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,  # BUG F03
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False,
    )
