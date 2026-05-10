"""CSV worklog helpers for optimizer and outer agent loops."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sysight.utils.cache import sysight_root


@dataclass
class WorklogRow:
    timestamp: str = ""
    loop_id: str = ""
    iteration: int = 0
    stage: str = ""
    status: str = ""
    repo: str = ""
    profile: str = ""
    run_id: str = ""
    artifact_dir: str = ""
    commit_before: str = ""
    commit_after: str = ""
    metric_name: str = ""
    metric_before: float | None = None
    metric_after: float | None = None
    metric_delta_pct: float | None = None
    accepted_count: int = 0
    rejected_count: int = 0
    coding_summary: str = ""
    reason: str = ""
    errors: str = ""


def default_worklog_path() -> Path:
    path = sysight_root() / "worklog" / "agent_loop.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append_worklog_row(path: str | Path | None, row: WorklogRow | dict[str, Any]) -> Path:
    target = Path(path) if path else default_worklog_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = _row_dict(row)
    if not data.get("timestamp"):
        data["timestamp"] = _now_iso()
    headers = [f.name for f in fields(WorklogRow)]
    exists = target.exists() and target.stat().st_size > 0
    with target.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: _cell(data.get(key)) for key in headers})
    return target


def append_many(paths: list[str | Path | None], row: WorklogRow | dict[str, Any]) -> None:
    for path in paths:
        append_worklog_row(path, row)


def _row_dict(row: WorklogRow | dict[str, Any]) -> dict[str, Any]:
    if isinstance(row, WorklogRow):
        return {f.name: getattr(row, f.name) for f in fields(WorklogRow)}
    return dict(row)


def _cell(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    if value is None:
        return ""
    return value


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
