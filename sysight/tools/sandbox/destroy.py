"""sandbox.destroy — Remove a sandbox git worktree."""

from __future__ import annotations

import shutil
import subprocess

from sysight.tools.registry import ToolDef
from sysight.tools.sandbox._manager import SandboxManager


def destroy(sandbox_id: str) -> bool:
    """Remove a git worktree and clean up references."""
    manager = SandboxManager()
    sb = manager.get(sandbox_id)
    if not sb:
        return False

    # Remove worktree
    subprocess.run(
        ["git", "-C", sb.repo, "worktree", "remove", sb.worktree_path, "--force"],
        capture_output=True, text=True,
    )

    # Clean up any remaining files
    import os
    if os.path.exists(sb.worktree_path):
        shutil.rmtree(sb.worktree_path, ignore_errors=True)

    manager.remove(sandbox_id)
    return True


DESTROY_TOOL = ToolDef(
    name="sandbox_destroy",
    description="Remove a sandbox git worktree and clean up references",
    parameters={"type": "object", "properties": {"sandbox_id": {"type": "string"}}, "required": ["sandbox_id"]},
    fn=destroy, read_only=False,
)
