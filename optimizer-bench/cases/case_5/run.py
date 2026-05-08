"""Entry point for case_5 — mixed precision + gradient accumulation."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.runtime.launcher import launch_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_v4.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--accum-steps", type=int, default=8)
    args = parser.parse_args()
    launch_training(args)


if __name__ == "__main__":
    main()
