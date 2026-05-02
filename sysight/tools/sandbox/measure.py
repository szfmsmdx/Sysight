"""sandbox.measure — Repeated metric measurement in the sandbox."""

from __future__ import annotations

import re
import statistics
import subprocess
from dataclasses import dataclass, field

from sysight.tools.registry import ToolDef
from sysight.tools.sandbox._manager import SandboxManager


@dataclass
class MeasureResult:
    metric_name: str = ""
    values: list[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    unit: str = ""


def measure(sandbox_id: str, run_cmd: list[str], grep: str,
            runs: int = 5, warmup: int = 1, timeout: int = 600) -> MeasureResult:
    """Run the target program N times, extract a metric from stdout."""
    manager = SandboxManager()
    sb = manager.get(sandbox_id)
    if not sb:
        return MeasureResult(metric_name=grep)

    pattern = re.compile(grep)
    values: list[float] = []

    for i in range(warmup + runs):
        try:
            r = subprocess.run(run_cmd, cwd=sb.worktree_path, capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            continue

        if i < warmup:
            continue

        # Extract metric from stdout
        for line in r.stdout.splitlines():
            m = pattern.search(line)
            if m:
                try:
                    val = float(m.group(0).replace(",", ""))
                    values.append(val)
                except ValueError:
                    pass
                break

    if not values:
        return MeasureResult(metric_name=grep)

    return MeasureResult(
        metric_name=grep, values=values,
        mean=statistics.mean(values),
        std=statistics.stdev(values) if len(values) > 1 else 0.0,
    )


MEASURE_TOOL = ToolDef(
    name="sandbox_measure",
    description="Run the target program multiple times to extract a performance metric from stdout",
    parameters={
        "type": "object",
        "properties": {
            "sandbox_id": {"type": "string"},
            "run_cmd": {"type": "array", "items": {"type": "string"}},
            "grep": {"type": "string"},
            "runs": {"type": "integer", "default": 5},
            "warmup": {"type": "integer", "default": 1},
            "timeout": {"type": "integer", "default": 600},
        },
        "required": ["sandbox_id", "run_cmd", "grep"],
    },
    fn=measure, read_only=False,
)
