"""Synthetic text dataset for transformer training."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticTextDataset(Dataset):
    """Generates synthetic token sequences for benchmarking.

    Each sample is a random sequence of token IDs with a fixed vocabulary.
    """

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        vocab_size: int,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        g = torch.Generator()
        g.manual_seed(seed)
        self.data = torch.randint(
            0, vocab_size,
            (num_samples, seq_len + 1),
            generator=g,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        tokens = self.data[idx]
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }
