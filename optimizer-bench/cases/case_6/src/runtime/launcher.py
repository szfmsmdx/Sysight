"""Training launcher for multi-modal case."""

from __future__ import annotations

import torch

from src.models.multimodal import MultiModalModel
from src.data.dataset import SyntheticMultiModalDataset
from src.data.loader import create_dataloader
from src.trainers.multimodal_loop import MultiModalTrainingLoop


def launch_training(args):
    """Launch multi-modal training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MultiModalModel(
        vocab_size=8192,
        d_model=512,
        n_heads=8,
        text_layers=4,
        num_classes=50,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_dataset = SyntheticMultiModalDataset(num_samples=2000, vocab_size=8192, seed=42)
    val_dataset = SyntheticMultiModalDataset(num_samples=400, vocab_size=8192, seed=123)

    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)

    loop = MultiModalTrainingLoop(model, optimizer, device)

    for epoch in range(args.epochs):
        train_loss = loop.train_epoch(train_loader, epoch)
        eval_results = loop.evaluate(val_loader)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={eval_results['accuracy']:.4f}")

    print("Training complete.")
