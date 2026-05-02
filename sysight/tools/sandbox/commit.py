"""sandbox.commit — Commit changes in the sandbox worktree."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from sysight.tools.registry import ToolDef
from sysight.tools.sandbox._manager import SandboxManager


@dataclass
class CommitResult:
    commit_hash: str = ""
    message: str = ""


def commit(sandbox_id: str, message: str) -> CommitResult:
    """Commit current changes in the sandbox worktree."""
    manager = SandboxManager()
    sb = manager.get(sandbox_id)
    if not sb:
        return CommitResult(message="Sandbox not found")

    subprocess.run(["git", "-C", sb.worktree_path, "add", "-A"],
                   capture_output=True, text=True)
    r = subprocess.run(["git", "-C", sb.worktree_path, "commit", "-m", message],
                       capture_output=True, text=True)

    if r.returncode == 0:
        h = subprocess.run(["git", "-C", sb.worktree_path, "rev-parse", "HEAD"],
                           capture_output=True, text=True).stdout.strip()
        return CommitResult(commit_hash=h, message=message)

    return CommitResult(message=f"Commit failed: {r.stderr[:200]}")


COMMIT_TOOL = ToolDef(
    name="sandbox_commit",
    description="Commit the current changes in the sandbox worktree",
    parameters={
        "type": "object",
        "properties": {"sandbox_id": {"type": "string"}, "message": {"type": "string"}},
        "required": ["sandbox_id", "message"],
    },
    fn=commit, read_only=False,
)
