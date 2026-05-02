"""sandbox.apply — Apply a code change to a file in the sandbox.

Verifies old_span_hash before applying to ensure patch correctness.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from sysight.tools.registry import ToolDef
from sysight.tools.sandbox._manager import SandboxManager


@dataclass
class ApplyResult:
    status: str  # "applied" | "failed"
    file_path: str = ""
    old_span_hash: str = ""
    error: str = ""


def apply(sandbox_id: str, file_path: str, old_span_start: int,
          old_span_end: int, old_span_hash: str, replacement: str) -> ApplyResult:
    """Apply a code change to a file in the sandbox. Verifies old_span_hash."""
    manager = SandboxManager()
    sb = manager.get(sandbox_id)
    if not sb:
        return ApplyResult(status="failed", error=f"Sandbox {sandbox_id} not found")

    target = Path(sb.worktree_path) / file_path
    if not target.exists():
        return ApplyResult(status="failed", file_path=file_path, error="File not found")

    try:
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError as e:
        return ApplyResult(status="failed", file_path=file_path, error=str(e))

    if old_span_start < 1 or old_span_end > len(lines):
        return ApplyResult(status="failed", file_path=file_path,
                           error=f"Span {old_span_start}-{old_span_end} out of range (1-{len(lines)})")

    # Verify old_span_hash
    old_text = "".join(lines[old_span_start - 1:old_span_end])
    actual_hash = hashlib.sha1(old_text.encode(), usedforsecurity=False).hexdigest()[:12]
    if actual_hash != old_span_hash:
        return ApplyResult(status="failed", file_path=file_path,
                           old_span_hash=actual_hash,
                           error=f"Hash mismatch: expected {old_span_hash}, actual {actual_hash}")

    # Apply replacement
    if not replacement.endswith("\n"):
        replacement += "\n"
    new_lines = lines[:old_span_start - 1] + [replacement] + lines[old_span_end:]
    try:
        target.write_text("".join(new_lines), encoding="utf-8")
    except OSError as e:
        return ApplyResult(status="failed", file_path=file_path, error=str(e))

    return ApplyResult(status="applied", file_path=file_path, old_span_hash=old_span_hash)


APPLY_TOOL = ToolDef(
    name="sandbox_apply",
    description="Apply a code change (patch) to a file in the sandbox. Verifies old code span hash before applying",
    parameters={
        "type": "object",
        "properties": {
            "sandbox_id": {"type": "string"},
            "file_path": {"type": "string"},
            "old_span_start": {"type": "integer"},
            "old_span_end": {"type": "integer"},
            "old_span_hash": {"type": "string"},
            "replacement": {"type": "string"},
        },
        "required": ["sandbox_id", "file_path", "old_span_start", "old_span_end", "old_span_hash", "replacement"],
    },
    fn=apply, read_only=False,
)
