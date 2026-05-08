"""DDP training launcher."""

from __future__ import annotations

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.models.resnet import ResNet
from src.data.loader import SyntheticImageDataset, create_distributed_dataloader
from src.trainers.ddp_loop import DDPTrainingLoop
from src.distributed.comm import sync_predictions


def launch_ddp_training(args):
    """Launch DDP training."""
    # Check if running under torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = local_rank == 0
    if is_main:
        print(f"World size: {world_size}, Device: {device}")

    # BUG F04: TF32 not enabled for Ampere+ GPUs (~1.5-2× matmul speedup)
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    model = ResNet(num_classes=10).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    train_dataset = SyntheticImageDataset(num_samples=2000, seed=42)
    val_dataset = SyntheticImageDataset(num_samples=400, seed=123)

    train_loader = create_distributed_dataloader(train_dataset, batch_size=args.batch_size)
    val_loader = create_distributed_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)

    loop = DDPTrainingLoop(model, optimizer, device, accum_steps=args.accum_steps)

    for epoch in range(args.epochs):
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        train_loss = loop.train_epoch(train_loader, epoch)

        if is_main:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

    if world_size > 1:
        dist.destroy_process_group()

    if is_main:
        print("Training complete.")
