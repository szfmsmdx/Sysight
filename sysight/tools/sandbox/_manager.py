"""Sandbox manager — tracks active git worktrees."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sandbox:
    sandbox_id: str
    repo: str                     # original repo path
    worktree_path: str            # git worktree path
    base_commit: str              # commit hash before any patches
    current_commit: str = ""
    patches_applied: int = 0


class SandboxManager:
    """Tracks active sandboxes in memory. In production this would be persistent."""

    _instance: SandboxManager | None = None
    _sandboxes: dict[str, Sandbox] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._sandboxes = {}
        return cls._instance

    def create(self, sandbox: Sandbox) -> None:
        self._sandboxes[sandbox.sandbox_id] = sandbox

    def get(self, sandbox_id: str) -> Sandbox | None:
        return self._sandboxes.get(sandbox_id)

    def remove(self, sandbox_id: str) -> None:
        self._sandboxes.pop(sandbox_id, None)

    @staticmethod
    def generate_id() -> str:
        return f"sandbox-{uuid.uuid4().hex[:12]}"
