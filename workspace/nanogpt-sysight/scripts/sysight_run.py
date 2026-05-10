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


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    _ensure_dataset(root)
    cmd = [
        sys.executable,
        "train.py",
        "config/sysight_baseline.py",
        "--enable_nvtx=True",
        *sys.argv[1:],
    ]
    proc = subprocess.run(cmd, cwd=str(root))
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
