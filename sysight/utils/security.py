"""Security utilities: path containment, command validation."""

from __future__ import annotations

from pathlib import Path


__all__ = [
    "validate_path_containment",
    "is_path_contained",
]


def is_path_contained(target: str | Path, allowed_root: str | Path) -> bool:
    """Check that target path is inside allowed_root (resolving symlinks)."""
    try:
        resolved_target = Path(target).resolve()
        resolved_root = Path(allowed_root).resolve()
        return str(resolved_target).startswith(str(resolved_root) + "/") or resolved_target == resolved_root
    except (OSError, ValueError):
        return False


def validate_path_containment(target: str | Path, allowed_root: str | Path) -> None:
    """Raise ValueError if target is outside allowed_root."""
    if not is_path_contained(target, allowed_root):
        raise ValueError(f"Path {target!r} is outside allowed root {allowed_root!r}")
