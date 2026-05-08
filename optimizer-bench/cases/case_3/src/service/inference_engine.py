"""Inference engine with batching and generation loop.

BUGS:
  F04: token-by-token generation without CUDA graph capture
  F05: CPU-side sampling with .item() sync
"""

from __future__ import annotations

import time
import torch
import torch.nn.functional as F

from src.models.llm import LLMModel


class InferenceEngine:
    """LLM inference engine for autoregressive generation.

    BUG F04: No CUDA graph capture — each decode step re-launches kernels.
    BUG F05: CPU-side argmax/top-k sampling with .item() sync.
    """

    def __init__(
        self,
        model: LLMModel,
        device: torch.device,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ):
        self.model = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate tokens autoregressively.

        BUG F04: No CUDA graph — each step is a fresh kernel launch.
        BUG F05: .item() sync on every token for sampling.
        """
        B = input_ids.size(0)
        generated = input_ids.clone()
        kv_caches = None

        for _ in range(self.max_new_tokens):
            # Only pass the last token for efficiency
            current_token = generated[:, -1:]

            logits, kv_caches = self.model(current_token, kv_caches)
            logits = logits[:, -1, :] / self.temperature

            # BUG F05: CPU-side sampling with .item() sync
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def benchmark(self, num_requests: int, prompt_len: int = 32) -> dict:
        """Run inference benchmark."""
        prompts = torch.randint(
            0, 8192,
            (num_requests, prompt_len),
            device=self.device,
        )

        start = time.time()
        total_tokens = 0

        for i in range(0, num_requests, 4):
            batch = prompts[i:i+4]
            output = self.generate(batch)
            total_tokens += output.numel()

        elapsed = time.time() - start
        tokens_per_sec = total_tokens / elapsed

        print(f"Generated {total_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")
        return {
            "total_tokens": total_tokens,
            "elapsed": elapsed,
            "tokens_per_sec": tokens_per_sec,
        }
