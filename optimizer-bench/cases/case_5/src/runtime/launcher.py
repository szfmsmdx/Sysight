"""Training launcher for mixed precision case."""

from __future__ import annotations

import torch

from src.models.vit import ViT
from src.data.dataset import SyntheticImageDataset
from src.data.loader import create_dataloader
from src.trainers.mixed_precision_loop import MixedPrecisionLoop


def launch_training(args):
    """Launch training with mixed precision issues."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=100,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    train_dataset = SyntheticImageDataset(num_samples=2000, seed=42)
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size)

    loop = MixedPrecisionLoop(model, optimizer, device, accum_steps=args.accum_steps)

    for epoch in range(args.epochs):
        train_loss = loop.train_epoch(train_loader, epoch)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

    print("Training complete.")
