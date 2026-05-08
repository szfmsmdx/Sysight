"""Positional encoding with sinusoidal embeddings.

BUG (F05): position encoding tensor is recomputed from scratch on every
forward pass. Should be pre-computed once in __init__ and registered as
a non-persistent buffer.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    BUG F05: recomputes the full encoding matrix on every forward call.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)

        # BUG F05: recomputes the full encoding matrix on every forward call
        position = torch.arange(0, seq_len, device=x.device).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device).float()
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(seq_len, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return x + pe.unsqueeze(0)
