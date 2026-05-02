"""sandbox.validate — Run test/validation commands in the sandbox."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

from sysight.tools.registry import ToolDef
from sysight.tools.sandbox._manager import SandboxManager


@dataclass
class ValidateResult:
    all_passed: bool
    total: int = 0
    passed: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)


def validate(sandbox_id: str, commands: list[list[str]], timeout: int = 300) -> ValidateResult:
    """Run validation commands in the sandbox."""
    manager = SandboxManager()
    sb = manager.get(sandbox_id)
    if not sb:
        return ValidateResult(all_passed=False, failures=[f"Sandbox {sandbox_id} not found"])

    result = ValidateResult(all_passed=True)
    for cmd in commands:
        result.total += 1
        try:
            r = subprocess.run(cmd, cwd=sb.worktree_path, capture_output=True,
                               text=True, timeout=timeout)
            if r.returncode == 0:
                result.passed += 1
            else:
                result.failed += 1
                result.all_passed = False
                result.failures.append(f"{' '.join(cmd)}: exit={r.returncode}\n{r.stderr[:500]}")
        except subprocess.TimeoutExpired:
            result.failed += 1
            result.all_passed = False
            result.failures.append(f"{' '.join(cmd)}: timeout")

    return result


VALIDATE_TOOL = ToolDef(
    name="sandbox_validate",
    description="Run test/validation commands in the sandbox. Returns pass/fail counts and failure details",
    parameters={
        "type": "object",
        "properties": {
            "sandbox_id": {"type": "string"},
            "commands": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
            "timeout": {"type": "integer", "default": 300},
        },
        "required": ["sandbox_id", "commands"],
    },
    fn=validate, read_only=False,
)
