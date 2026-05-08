"""Multi-head self-attention with causal masking."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention.

    BUG (F01): causal_mask is recomputed from scratch on every forward pass
    using torch.triu + broadcasting. Should be pre-computed as a registered
    buffer and sliced by current sequence length.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        attn_weights = (q @ k.transpose(-2, -1)) / self.scale

        # BUG F01: causal mask recomputed from scratch every forward pass
        causal_mask = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_dropout(self.out_proj(out))
