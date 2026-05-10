"""Shared paths for Sysight generated artifacts."""

from __future__ import annotations

import re
from pathlib import Path


def sysight_root() -> Path:
    root = Path.cwd() / ".sysight"
    root.mkdir(parents=True, exist_ok=True)
    return root


def cache_dir(*parts: str) -> Path:
    path = sysight_root() / "cache"
    for part in parts:
        if part:
            path = path / part
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(value: str, *, limit: int = 64) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value or "run").strip("-._")
    return (cleaned or "run")[:limit]
