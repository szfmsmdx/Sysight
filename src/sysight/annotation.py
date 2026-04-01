"""Evidence annotation schema for lightweight analysis findings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Finding:
    """A single machine-authored finding tied to a time range."""

    type: str
    label: str
    start_ns: int
    end_ns: int | None = None
    stream: int | None = None
    gpu_id: int | None = None
    severity: str = "info"
    note: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class EvidenceReport:
    """A collection of findings for one analysis run."""

    title: str
    profile_path: str = ""
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "profile_path": self.profile_path,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def save_findings(report: EvidenceReport, path: str) -> None:
    """Persist findings as JSON."""
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report.to_dict(), handle, indent=2)
