"""Sysight reproducible run entry for nanoGPT worktrees.

This script ensures shakespeare_char data artifacts exist in the current
worktree, then runs the quick baseline training command.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _ensure_dataset(root: Path) -> None:
    data_dir = root / "data" / "shakespeare_char"
    required = [data_dir / "train.bin", data_dir / "val.bin", data_dir / "meta.pkl"]
    if all(p.exists() for p in required):
        return
    subprocess.check_call([sys.executable, str(data_dir / "prepare.py")], cwd=str(root))


def _detect_device() -> str:
    """Return the best available device: cuda > mps > cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    _ensure_dataset(root)

    # Auto-detect device when caller args don't already specify one.
    extra_args = list(sys.argv[1:])
    if not any(a.startswith("--device") for a in extra_args):
        device = _detect_device()
        extra_args = [f"--device={device}"] + extra_args

    cmd = [
        sys.executable,
        "train.py",
        "config/sysight_baseline.py",
        "--enable_nvtx=False",  # nvtx requires CUDA; harmless on other backends
        *extra_args,
    ]
    proc = subprocess.run(cmd, cwd=str(root))
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
