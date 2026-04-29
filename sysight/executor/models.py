"""Data models for sysight.executor.

Defines the output structure of the execution report (execution_report.json).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class ExecutedPatch:
    """The result of applying and measuring a single patch."""

    id: str
    status: str  # "committed" or "skipped"
    score_after: float | None
    delta: str

    @classmethod
    def from_dict(cls, data: dict) -> ExecutedPatch:
        return cls(
            id=data.get("id", ""),
            status=data.get("status", ""),
            score_after=data.get("score_after"),
            delta=data.get("delta", ""),
        )


@dataclass
class ExecutionReport:
    """The overall report after executing a patch plan."""

    baseline_score: float | None
    patches: list[ExecutedPatch]
    final_score: float | None

    @classmethod
    def from_dict(cls, data: dict) -> ExecutionReport:
        patches_data = data.get("patches", [])
        return cls(
            baseline_score=data.get("baseline_score"),
            patches=[ExecutedPatch.from_dict(p) for p in patches_data],
            final_score=data.get("final_score"),
        )

    def to_dict(self) -> dict:
        """Serialize to JSON-friendly dict."""
        return asdict(self)
