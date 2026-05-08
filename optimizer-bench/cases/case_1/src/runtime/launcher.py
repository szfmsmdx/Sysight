"""Training launcher — wires together model, data, and training loop."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.transformer import GPTModel
from src.data.dataset import SyntheticTextDataset
from src.data.loader import create_dataloader
from src.trainers.loop import TrainingLoop
from src.utils.checkpoint import CheckpointManager
from src.utils.metrics import MetricTracker


def launch_training(args):
    """Launch training with the given arguments."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model config
    vocab_size = 8192
    d_model = 512
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    max_seq_len = args.seq_len

    model = GPTModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Data
    train_dataset = SyntheticTextDataset(
        num_samples=2000,
        seq_len=args.seq_len,
        vocab_size=vocab_size,
        seed=42,
    )
    val_dataset = SyntheticTextDataset(
        num_samples=400,
        seq_len=args.seq_len,
        vocab_size=vocab_size,
        seed=123,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Training
    loop = TrainingLoop(model, optimizer, device)
    ckpt = CheckpointManager("./checkpoints")
    tracker = MetricTracker()

    for epoch in range(args.epochs):
        train_loss = loop.train_epoch(train_loader, epoch)
        val_loss = loop.validate(val_loader)

        tracker.update("train_loss", train_loss)
        tracker.update("val_loss", val_loss)

        print(f"Epoch {epoch} complete: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        ckpt.save(model, optimizer, epoch, tracker.summary())

    ckpt.wait_all()
    print("Training complete.")
    print(f"Final metrics: {tracker.summary()}")
