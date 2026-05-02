"""sandbox.revert — Revert changes in the sandbox worktree."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from sysight.tools.registry import ToolDef
from sysight.tools.sandbox._manager import SandboxManager


@dataclass
class RevertResult:
    status: str = "reverted"
    error: str = ""


def revert(sandbox_id: str) -> RevertResult:
    """Revert all uncommitted changes in the sandbox worktree."""
    manager = SandboxManager()
    sb = manager.get(sandbox_id)
    if not sb:
        return RevertResult(status="error", error="Sandbox not found")

    # Reset to base commit
    r = subprocess.run(
        ["git", "-C", sb.worktree_path, "reset", "--hard", sb.base_commit],
        capture_output=True, text=True,
    )
    subprocess.run(["git", "-C", sb.worktree_path, "clean", "-fd"],
                   capture_output=True, text=True)

    if r.returncode != 0:
        return RevertResult(status="error", error=r.stderr[:200])

    return RevertResult(status="reverted")


REVERT_TOOL = ToolDef(
    name="sandbox_revert",
    description="Revert all uncommitted changes in the sandbox worktree, restoring clean state",
    parameters={"type": "object", "properties": {"sandbox_id": {"type": "string"}}, "required": ["sandbox_id"]},
    fn=revert, read_only=False,
)
