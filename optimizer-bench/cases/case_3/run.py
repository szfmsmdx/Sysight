"""Entry point for case_3 — LLM inference service."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.service.inference_engine import run_inference_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_v1.yaml")
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    run_inference_benchmark(args)


if __name__ == "__main__":
    main()
