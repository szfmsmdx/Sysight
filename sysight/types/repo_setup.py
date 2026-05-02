"""RepoSetup dataclass. Zero internal dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RepoSetup:
    """Per-repo configuration discovered during WARMUP."""
    entry_point: str = ""
    minimal_run: list[str] = field(default_factory=list)
    metric_grep: str | None = None
    metric_lower_is_better: bool = False
    needs_instrumentation: bool = False
    test_commands: list[list[str]] = field(default_factory=list)
    build_commands: list[list[str]] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    source: Literal["warmup_verified", "warmup_partial", "manual", "benchmark_case"] = "manual"
    verified_at: str = ""
