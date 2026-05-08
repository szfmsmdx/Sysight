"""Multi-modal model with text encoder, image encoder, and cross-attention fusion.

BUGS:
  F01: Cross-attention recomputes K,V projections for image features on every call
  F02: Text encoder runs separately from image encoder — should be parallel
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """Transformer-based text encoder."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_len: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_final = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        x = self.embedding(input_ids) + self.pos_embed[:, :S, :]
        x = self.transformer(x)
        return self.ln_final(x)


class ImageEncoder(nn.Module):
    """CNN-based image encoder."""

    def __init__(self, d_model: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            self._make_block(64, 128, 2),
            self._make_block(128, 256, 2),
            self._make_block(256, 512, 2),
        )
        self.proj = nn.Conv2d(512, d_model, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    @staticmethod
    def _make_block(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, 224, 224]
        x = self.backbone(images)
        x = self.proj(x)
        x = self.avgpool(x)  # [B, D, 7, 7]
        B, D, H, W = x.shape
        return x.flatten(2).transpose(1, 2)  # [B, 49, D]


class CrossModalFusion(nn.Module):
    """Cross-attention fusion of text and image features.

    BUG F01: Image K,V projections recomputed on every forward call.
    For fixed image features (e.g., during multiple text queries), these
    should be cached.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.text_proj = nn.Linear(d_model, d_model)
        # BUG F01: image K,V projections — recomputed every call
        self.img_k_proj = nn.Linear(d_model, d_model)
        self.img_v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        """Cross-attention: text attends to image features.

        BUG F01: image K,V recomputed every call — should cache for fixed images.
        """
        B = text_feat.size(0)

        # Query from text
        q = self.text_proj(text_feat).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # BUG F01: K,V from image — recomputed every call
        k = self.img_k_proj(image_feat).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.img_v_proj(image_feat).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.out_proj(out)


class MultiModalModel(nn.Module):
    """Multi-modal model for text-image tasks.

    BUG F02: Text and image encoders run sequentially — should be parallel.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        text_layers: int,
        num_classes: int,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size, d_model, n_heads, text_layers)
        self.image_encoder = ImageEncoder(d_model)
        self.fusion = CrossModalFusion(d_model, n_heads)

        self.text_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, num_classes),
        )
        self.image_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, num_classes),
        )
        self.fusion_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, images: torch.Tensor) -> dict:
        """Forward pass.

        BUG F02: Sequential encoding — text and image encoders run one after
        another instead of in parallel (e.g., with CUDA streams).
        """
        # BUG F02: sequential — should use separate CUDA streams
        text_feat = self.text_encoder(input_ids)  # [B, S, D]
        image_feat = self.image_encoder(images)    # [B, 49, D]

        # Cross-modal fusion
        fused = self.fusion(text_feat, image_feat)  # [B, S, D]

        # Pooling
        text_logits = self.text_pool(text_feat.mean(dim=1))
        image_logits = self.image_pool(image_feat.mean(dim=1))
        fusion_logits = self.fusion_pool(fused.mean(dim=1))

        return {
            "text_logits": text_logits,
            "image_logits": image_logits,
            "fusion_logits": fusion_logits,
        }
