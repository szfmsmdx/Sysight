"""sandbox.exec — Execute a command in the sandbox."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from sysight.tools.registry import ToolDef
from sysight.tools.sandbox._manager import SandboxManager


@dataclass
class ExecResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    elapsed_ms: float = 0


def exec_cmd(sandbox_id: str, cmd: list[str], timeout: int = 60, cwd: str = "") -> ExecResult:
    """Execute a command inside the sandbox worktree."""
    manager = SandboxManager()
    sb = manager.get(sandbox_id)
    if not sb:
        return ExecResult(exit_code=-1, stderr=f"Sandbox {sandbox_id} not found")

    import time
    workdir = sb.worktree_path
    if cwd:
        workdir = f"{sb.worktree_path}/{cwd}"

    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout)
        elapsed = (time.monotonic() - t0) * 1000
        return ExecResult(
            exit_code=r.returncode, stdout=r.stdout, stderr=r.stderr,
            elapsed_ms=elapsed,
        )
    except subprocess.TimeoutExpired:
        return ExecResult(exit_code=-1, stderr=f"Timeout after {timeout}s",
                          elapsed_ms=timeout * 1000)


EXEC_TOOL = ToolDef(
    name="sandbox_exec",
    description="Execute an allowlisted command inside the sandbox worktree",
    parameters={
        "type": "object",
        "properties": {
            "sandbox_id": {"type": "string"},
            "cmd": {"type": "array", "items": {"type": "string"}},
            "timeout": {"type": "integer", "default": 60},
            "cwd": {"type": "string", "default": ""},
        },
        "required": ["sandbox_id", "cmd"],
    },
    fn=exec_cmd, read_only=False,
)
