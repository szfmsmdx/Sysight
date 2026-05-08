"""LLM model with KV cache.

BUGS:
  F01: KV cache uses list-of-tensors instead of contiguous buffer
  F02: Standard attention instead of flash attention
  F03: torch.compile not used
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCacheAttention(nn.Module):
    """Multi-head attention with KV cache for autoregressive generation.

    BUG F01: KV cache stored as list of tensors — each append triggers
    a reallocation. Should use a pre-allocated contiguous buffer.
    BUG F02: Uses standard attention instead of flash attention.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.max_seq_len = max_seq_len

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Forward pass with KV cache.

        Args:
            x: [batch, 1, d_model] for single-token generation
            kv_cache: {'k': list[Tensor], 'v': list[Tensor]} or None

        Returns:
            (output, updated_kv_cache)
        """
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            # BUG F01: list append causes reallocation on every token
            k_cache = kv_cache["k"] + [k_new]
            v_cache = kv_cache["v"] + [v_new]
            k = torch.cat(k_cache, dim=2)
            v = torch.cat(v_cache, dim=2)
        else:
            k_cache = [k_new]
            v_cache = [v_new]
            k = k_new
            v = v_new

        # BUG F02: standard attention — should use flash attention
        attn_weights = (q @ k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = attn_weights @ v

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out), {"k": k_cache, "v": v_cache}


class DecoderLayer(nn.Module):
    """Single decoder layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int = 2048):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = KVCacheAttention(d_model, n_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, kv_cache: dict | None = None):
        attn_out, kv_cache = self.attn(self.ln1(x), kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, kv_cache


class LLMModel(nn.Module):
    """Decoder-only LLM for inference.

    BUG F03: torch.compile not used — should compile the model for
    inference speedup.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, max_seq_len)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, kv_caches: list[dict] | None = None):
        x = self.token_embedding(input_ids)

        if kv_caches is None:
            kv_caches = [None] * len(self.layers)

        new_caches = []
        for layer, cache in zip(self.layers, kv_caches):
            x, new_cache = layer(x, cache)
            new_caches.append(new_cache)

        x = self.ln_final(x)
        return self.lm_head(x), new_caches
