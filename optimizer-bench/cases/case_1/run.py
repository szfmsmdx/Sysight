"""Entry point for case_1 — GPT-style transformer training."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.runtime.launcher import launch_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_v2.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    launch_training(args)


if __name__ == "__main__":
    main()
