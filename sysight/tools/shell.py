"""shell.exec — Execute a command in the repo directory.

Lightweight execution for warmup phase — no sandbox/git-worktree required.
Used for smoke tests, environment checks, and metric discovery.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from sysight.tools.registry import ToolDef


@dataclass
class ShellResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    elapsed_ms: float = 0


def shell_exec(repo: str, cmd: list[str], timeout: int = 30, cwd: str = "") -> ShellResult:
    """Execute a command in the repo directory.

    Args:
        repo: Absolute path to the repository root.
        cmd: Command and arguments as a list of strings.
        timeout: Maximum execution time in seconds.
        cwd: Relative subdirectory within repo (empty = repo root).
    """
    root = Path(repo).resolve()
    workdir = str(root)
    if cwd:
        workdir = str(root / cwd)

    t0 = time.monotonic()
    try:
        r = subprocess.run(
            cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout,
        )
        elapsed = (time.monotonic() - t0) * 1000
        return ShellResult(
            exit_code=r.returncode,
            stdout=r.stdout,
            stderr=r.stderr,
            elapsed_ms=elapsed,
        )
    except subprocess.TimeoutExpired:
        return ShellResult(
            exit_code=-1,
            stderr=f"Timeout after {timeout}s",
            elapsed_ms=timeout * 1000,
        )
    except FileNotFoundError:
        return ShellResult(
            exit_code=-1,
            stderr=f"Command not found: {cmd[0] if cmd else '?'}",
            elapsed_ms=(time.monotonic() - t0) * 1000,
        )


SHELL_TOOL = ToolDef(
    name="shell_exec",
    description=(
        "Execute a shell command in the repository directory. "
        "Use for smoke tests (verify the program can start), "
        "environment checks (python --version, nvidia-smi, nsys --version), "
        "and capturing program output to discover metrics. "
        "Returns exit_code, stdout, stderr, elapsed_ms."
    ),
    parameters={
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Absolute path to the repository root"},
            "cmd": {"type": "array", "items": {"type": "string"}, "description": "Command and arguments"},
            "timeout": {"type": "integer", "default": 30, "description": "Max execution time in seconds"},
            "cwd": {"type": "string", "default": "", "description": "Relative subdirectory within repo"},
        },
        "required": ["repo", "cmd"],
    },
    fn=shell_exec,
    read_only=False,
    max_calls_per_task=20,
)
