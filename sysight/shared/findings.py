"""shared/findings.py — Standard Finding model and extraction utilities.

Bridges the gap between analyzer output and optimizer input.

Analyzer produces findings from TWO sources:
  1. SQL classifier → NsysFinding (high-level, category like gpu_idle)
  2. Codex localization → LocalizationQuestion/Anchor (code-level, with file/line/function)

Optimizer needs code-level findings (source #2) to generate patches.

This module:
  - Defines the canonical `Finding` dataclass used across the pipeline
  - Provides `extract_findings()` to parse analyzer outputs into standard findings
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Finding:
    """A single code-level performance finding, consumable by optimizer."""

    id: str
    category: str          # e.g. "C1", "C4", "C7"
    title: str
    file: str              # source file path (relative to repo root)
    function: str          # function / method name
    line: int | None       # source line number
    description: str       # what's wrong
    suggestion: str        # how to fix
    evidence: list[str]    # supporting evidence strings
    priority: str          # "high" | "medium" | "low"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Finding:
        return cls(
            id=data.get("id", ""),
            category=data.get("category", ""),
            title=data.get("title", ""),
            file=data.get("file", ""),
            function=data.get("function", ""),
            line=data.get("line"),
            description=data.get("description", ""),
            suggestion=data.get("suggestion", ""),
            evidence=data.get("evidence", []),
            priority=data.get("priority", "medium"),
        )


def _extract_json_from_text(text: str) -> dict | list | None:
    """Try to extract a JSON object or array from text (may be wrapped in ```json```)."""
    # Try fenced JSON block first
    fenced = re.search(r"```json\s*([\{\[][\s\S]*?[\}\]])\s*```", text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    # Try raw JSON — determine if it starts with [ or {
    # Must check which opener comes first to handle arrays like [{"id":...}]
    first_brace = text.find("{")
    first_bracket = text.find("[")
    if first_bracket >= 0 and (first_brace < 0 or first_bracket < first_brace):
        # Array format
        end = text.rfind("]")
        if end > first_bracket:
            try:
                return json.loads(text[first_bracket : end + 1])
            except json.JSONDecodeError:
                pass
    if first_brace >= 0:
        end = text.rfind("}")
        if end > first_brace:
            try:
                return json.loads(text[first_brace : end + 1])
            except json.JSONDecodeError:
                pass
    return None


def _findings_from_codex_output(data: dict) -> list[Finding]:
    """Parse findings from codex last_message.txt format.

    Codex output has a top-level "findings" array where each item looks like:
      {category, title, file, function, line, description, suggestion, evidence, priority}
    """
    raw_findings = data.get("findings", [])
    results = []
    for i, f in enumerate(raw_findings):
        fid = f.get("id", f"finding_{i:03d}")
        cat = f.get("category", "")
        title = f.get("title", "")
        file_path = f.get("file", "")
        function = f.get("function", "")
        line = f.get("line")
        if isinstance(line, str):
            try:
                line = int(line)
            except (ValueError, TypeError):
                line = None
        description = f.get("description", "")
        suggestion = f.get("suggestion", "")
        evidence = f.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        priority = f.get("priority", "medium")
        results.append(Finding(
            id=fid,
            category=cat,
            title=title,
            file=file_path,
            function=function,
            line=line,
            description=description,
            suggestion=suggestion,
            evidence=evidence,
            priority=priority,
        ))
    return results


def _findings_from_nsys_artifact(data: dict) -> list[Finding]:
    """Parse findings from .sysight/nsys/*.json artifact.

    This artifact has two potential sources of code-level findings:
      - localization.questions[]  (from Codex)
      - localization.anchors[]   (from Codex)

    The top-level findings[] are SQL-classifier findings (NsysFinding)
    which lack file/function info and are less useful for optimizer.
    """
    results = []
    localization = data.get("localization")
    if not localization or not isinstance(localization, dict):
        return results

    # Prefer questions (more detailed)
    for q in localization.get("questions", []):
        fid = q.get("question_id", "")
        cat = q.get("category", "")
        title = q.get("title", "")
        file_path = q.get("file_path", "")
        function = q.get("function", "")
        line = q.get("line")
        if isinstance(line, str):
            try:
                line = int(line)
            except (ValueError, TypeError):
                line = None
        rationale = q.get("rationale", "")
        suggestion = q.get("suggestion", "")
        results.append(Finding(
            id=fid,
            category=cat,
            title=title,
            file=file_path,
            function=function,
            line=line,
            description=rationale,
            suggestion=suggestion,
            evidence=[],
            priority="high" if cat in ("C1", "C4") else "medium",
        ))

    # Fall back to anchors if no questions
    if not results:
        for a in localization.get("anchors", []):
            fid = a.get("window_id", "")
            cat = a.get("category", "")
            title = a.get("event_name", "")
            file_path = a.get("file_path", "")
            function = a.get("function", "")
            line = a.get("line")
            if isinstance(line, str):
                try:
                    line = int(line)
                except (ValueError, TypeError):
                    line = None
            rationale = a.get("rationale", "")
            suggestion = a.get("suggestion", "")
            results.append(Finding(
                id=fid,
                category=cat,
                title=title,
                file=file_path,
                function=function,
                line=line,
                description=rationale,
                suggestion=suggestion,
                evidence=[],
                priority="medium",
            ))

    return results


def extract_findings(source: str | Path) -> list[Finding]:
    """Extract standard Findings from an analyzer output file.

    Supports:
      - Codex last_message.txt (raw JSON with top-level "findings" key)
      - .sysight/nsys/*.json artifact (with "localization" key)
      - A pre-made findings.json (list of Finding dicts)
    """
    path = Path(source)
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8").strip()

    # Try direct JSON parse first
    data = _extract_json_from_text(text)
    if data is None:
        return []

    # Case 1: pre-made findings.json — list of finding dicts
    if isinstance(data, list):
        return [Finding.from_dict(f) for f in data if isinstance(f, dict)]

    # Case 2: nsys artifact — has "localization" key (code-level findings)
    if "localization" in data:
        findings = _findings_from_nsys_artifact(data)
        if findings:
            return findings

    # Case 3: codex last_message.txt — has top-level "findings" with code-level info
    # Only extract if findings have file/function (code-level), not SQL classifier findings
    if "findings" in data and isinstance(data["findings"], list):
        items = data["findings"]
        if items and isinstance(items[0], dict) and ("file" in items[0] or "function" in items[0]):
            return _findings_from_codex_output(data)

    return []


def write_findings_json(findings: list[Finding], output_path: str | Path) -> Path:
    """Write a list of Findings to a standard JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [f.to_dict() for f in findings]
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def find_latest_analysis_output(repo_root: str | Path = ".") -> Path | None:
    """Find the most recent last_message.txt under .sysight/analysis-runs."""
    runs_dir = Path(repo_root) / ".sysight" / "analysis-runs"
    if not runs_dir.is_dir():
        return None

    candidates = sorted(runs_dir.glob("run-*/last_message.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def find_latest_nsys_artifact(repo_root: str | Path = ".") -> Path | None:
    """Find the most recent nsys JSON artifact under .sysight/nsys."""
    nsys_dir = Path(repo_root) / ".sysight" / "nsys"
    if not nsys_dir.is_dir():
        return None

    candidates = sorted(nsys_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None
