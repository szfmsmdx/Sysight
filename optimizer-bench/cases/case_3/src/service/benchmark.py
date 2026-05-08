"""Inference benchmark runner."""

from __future__ import annotations

import torch

from src.models.llm import LLMModel
from src.service.inference_engine import InferenceEngine


def run_inference_benchmark(args):
    """Run LLM inference benchmark."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # BUG F03: torch.compile not used
    model = LLMModel(
        vocab_size=8192,
        d_model=512,
        n_heads=8,
        n_layers=4,
        d_ff=2048,
        max_seq_len=2048,
    ).to(device)
    model.eval()

    engine = InferenceEngine(model, device, max_new_tokens=args.max_new_tokens)
    result = engine.benchmark(args.num_requests)

    print(f"Benchmark complete: {result['tokens_per_sec']:.1f} tok/s")
