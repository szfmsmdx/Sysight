"""Git helpers: worktree management, commit hash retrieval."""

from __future__ import annotations

from pathlib import Path


__all__ = [
    "create_worktree",
    "remove_worktree",
    "get_commit_hash",
]


def create_worktree(repo: str | Path, name: str) -> Path:
    """Create a git worktree for sandboxed execution.

    TODO: Stage 2 — implement with subprocess git worktree add.
    """
    raise NotImplementedError("TODO: Stage 2")


def remove_worktree(worktree: str | Path, force: bool = False) -> None:
    """Remove a git worktree and prune the reference.

    TODO: Stage 2 — implement with subprocess git worktree remove.
    """
    raise NotImplementedError("TODO: Stage 2")


def get_commit_hash(repo: str | Path) -> str:
    """Get the current HEAD commit hash.

    TODO: Stage 2 — implement with subprocess git rev-parse HEAD.
    """
    raise NotImplementedError("TODO: Stage 2")
