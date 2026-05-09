"""patcher — deterministic line-level patch apply / revert.

No LLM involved.  All operations are pure string manipulation on source
files, keyed by (file_path, old_span_start, old_span_end, old_span_hash)
to guarantee idempotency.

Usage from the Optimizer pipeline::

    from sysight.tools.patcher import PatchApplier

    applier = PatchApplier(repo_root="/path/to/repo")

    # Apply a patch
    ok, msg = applier.apply(
        file_path="src/ops/tensor_ops.py",
        old_span_start=42,       # 1-based
        old_span_end=55,          # 1-based, inclusive
        old_span_hash="abc123",   # SHA1[:12] of the old span
        replacement="new code\n",
    )

    # Revert the last applied patch on a file
    ok, msg = applier.revert("src/ops/tensor_ops.py")
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


def compute_span_hash(text: str) -> str:
    """SHA1[:12] of text, matching the PatchCandidate convention."""
    return hashlib.sha1(text.encode(), usedforsecurity=False).hexdigest()[:12]


@dataclass
class _PatchRecord:
    """Stored snapshot for revert."""
    file_path: str
    original_text: str
    old_span_start: int
    old_span_end: int
    old_span_hash: str
    replacement: str


class PatchApplier:
    """Deterministic patch apply / revert for a repo.

    Thread-unsafe by design (optimizer is single-threaded per repo).
    """

    def __init__(self, repo_root: str | Path):
        self._root = Path(repo_root).resolve()
        # file_path → list of _PatchRecord (most recent last)
        self._history: dict[str, list[_PatchRecord]] = {}

    # ── Apply ──

    def apply(
        self,
        file_path: str,
        old_span_start: int,
        old_span_end: int,
        old_span_hash: str,
        replacement: str,
    ) -> tuple[bool, str]:
        """Apply a patch and write the file.

        Returns (success, message).
        """
        full_path = self._root / file_path
        if not full_path.exists():
            return False, f"file not found: {file_path}"

        try:
            original = full_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return False, f"cannot read {file_path}: {e}"

        lines = original.splitlines()
        total = len(lines)

        if old_span_start < 1 or old_span_end > total or old_span_start > old_span_end:
            return False, (
                f"invalid span {old_span_start}-{old_span_end} "
                f"(file has {total} lines)"
            )

        # Extract old span (1-based → 0-based indexing)
        span_lines = lines[old_span_start - 1: old_span_end]
        span_text = "\n".join(span_lines)
        actual_hash = compute_span_hash(span_text)

        if actual_hash != old_span_hash:
            return False, (
                f"hash mismatch: expected {old_span_hash}, "
                f"got {actual_hash} — file may have changed since planning"
            )

        # Save snapshot for revert
        record = _PatchRecord(
            file_path=file_path,
            original_text=original,
            old_span_start=old_span_start,
            old_span_end=old_span_end,
            old_span_hash=old_span_hash,
            replacement=replacement,
        )
        self._history.setdefault(file_path, []).append(record)

        # Build new content
        new_lines = replacement.splitlines()
        lines[old_span_start - 1: old_span_end] = new_lines
        new_content = "\n".join(lines)

        # Preserve trailing newline if original had one
        if original.endswith("\n") and not new_content.endswith("\n"):
            new_content += "\n"

        try:
            full_path.write_text(new_content, encoding="utf-8")
        except OSError as e:
            return False, f"cannot write {file_path}: {e}"

        return True, f"applied patch to {file_path} lines {old_span_start}-{old_span_end}"

    # ── Revert ──

    def revert(self, file_path: str) -> tuple[bool, str]:
        """Revert the most recent patch on file_path.

        Restores the file to the state *before* that patch was applied.
        Returns (success, message).
        """
        records = self._history.get(file_path)
        if not records:
            return False, f"no patch history for {file_path}"

        record = records.pop()
        if not records:
            del self._history[file_path]

        full_path = self._root / file_path
        try:
            full_path.write_text(record.original_text, encoding="utf-8")
        except OSError as e:
            return False, f"cannot revert {file_path}: {e}"

        return True, f"reverted {file_path} to pre-patch state"

    def revert_all(self) -> list[tuple[str, bool, str]]:
        """Revert all applied patches, most-recent-first.

        Returns list of (file_path, success, message).
        """
        results = []
        # Process in reverse order of application
        all_paths = list(self._history.keys())
        for fp in reversed(all_paths):
            records = self._history.get(fp, [])
            while records:
                ok, msg = self.revert(fp)
                results.append((fp, ok, msg))
                records = self._history.get(fp, [])
        return results

    # ── Utility ──

    def read_file(self, file_path: str) -> str | None:
        """Read a file from the repo. Returns None if not found."""
        full_path = self._root / file_path
        if not full_path.exists():
            return None
        try:
            return full_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
