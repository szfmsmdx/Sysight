"""Entry point for case_4 — data pipeline + checkpoint training."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.runtime.launcher import launch_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_v2.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    launch_training(args)


if __name__ == "__main__":
    main()
