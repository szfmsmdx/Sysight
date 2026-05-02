"""sandbox.create — Create an isolated git worktree for patch testing."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from sysight.tools.registry import ToolDef
from sysight.tools.sandbox._manager import Sandbox, SandboxManager


@dataclass
class SandboxRef:
    sandbox_id: str
    worktree_path: str
    base_commit: str


def create(repo: str) -> SandboxRef:
    """Create a git worktree for sandboxed execution."""
    root = Path(repo).resolve()
    manager = SandboxManager()

    # Get current commit
    base = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    if not base:
        raise RuntimeError(f"Not a git repository or no commits: {repo}")

    sandbox_id = SandboxManager.generate_id()
    worktree_dir = root.parent / f".sysight-sandbox-{sandbox_id}"

    subprocess.run(
        ["git", "-C", str(root), "worktree", "add", str(worktree_dir), base],
        capture_output=True, text=True, check=True,
    )

    sandbox = Sandbox(
        sandbox_id=sandbox_id, repo=str(root),
        worktree_path=str(worktree_dir), base_commit=base,
    )
    manager.create(sandbox)

    return SandboxRef(sandbox_id=sandbox_id, worktree_path=str(worktree_dir), base_commit=base)


CREATE_TOOL = ToolDef(
    name="sandbox_create",
    description="Create an isolated git worktree for patch testing. Returns a sandbox reference",
    parameters={"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
    fn=create, read_only=False,
)
