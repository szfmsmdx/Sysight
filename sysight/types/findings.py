"""Finding dataclasses. Zero internal dependencies."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal


def make_finding_id(category: str, file_path: str | None, line: int | None, function: str | None) -> str:
    """Generate a stable finding ID: '{category}:{file}:{line}:{function}'."""
    fp = file_path or ""
    ln = str(line) if line is not None else ""
    fn = function or ""
    raw = f"{category}:{fp}:{ln}:{fn}"
    digest = hashlib.sha1(raw.encode(), usedforsecurity=False).hexdigest()[:8]
    return f"{category}:{digest}"


@dataclass
class LocalizedFinding:
    """A single code-level performance finding with source location."""
    finding_id: str                      # "{category}:{file}:{line}:{function}"
    category: str                        # C1-C7
    title: str
    priority: Literal["high", "medium", "low"] = "medium"
    confidence: Literal["confirmed", "probable", "unresolved"] = "unresolved"
    evidence_refs: list[str] = field(default_factory=list)
    file_path: str | None = None
    function: str | None = None
    line: int | None = None
    description: str = ""
    suggestion: str = ""
    status: Literal["accepted", "rejected", "unresolved"] = "accepted"
    reject_reason: str = ""


@dataclass
class MemoryUpdate:
    """A suggested update to workspace or experience wiki."""
    path: str                            # e.g. "workspaces/<ns>/overview.md"
    content: str
    action: Literal["append", "replace", "upsert"] = "append"
    category: str | None = None          # for signal pages: C1-C7
    scope: Literal["workspace", "global", "benchmark"] = "workspace"
    reason: str = ""


@dataclass
class LocalizedFindingSet:
    """The output of the ANALYZE stage."""
    run_id: str
    summary: str = ""
    findings: list[LocalizedFinding] = field(default_factory=list)
    rejected: list[LocalizedFinding] = field(default_factory=list)
    memory_updates: list[MemoryUpdate] = field(default_factory=list)
    parse_error: str = ""
