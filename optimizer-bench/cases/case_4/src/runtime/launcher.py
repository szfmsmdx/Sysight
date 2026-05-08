"""Training launcher for data pipeline + checkpoint case."""

from __future__ import annotations

import torch

from src.models.resnet import ResNet50
from src.data.pipeline import SyntheticImageDataset
from src.data.loader import create_dataloader
from src.trainers.loop import TrainingLoop
from src.utils.checkpoint import CheckpointManager


def launch_training(args):
    """Launch training with data pipeline and checkpoint management."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ResNet50(num_classes=100).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
    )

    train_dataset = SyntheticImageDataset(num_samples=5000, seed=42)
    val_dataset = SyntheticImageDataset(num_samples=1000, seed=123)

    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)

    loop = TrainingLoop(model, optimizer, device)
    ckpt = CheckpointManager("./checkpoints")

    for epoch in range(args.epochs):
        train_loss = loop.train_epoch(train_loader, epoch)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

        # BUG F04: synchronous checkpoint save
        ckpt.save(model, optimizer, epoch, {"train_loss": train_loss})

    print("Training complete.")
