"""Metric tracking utilities."""

from __future__ import annotations

import torch


class MetricTracker:
    """Tracks running averages of scalar metrics."""

    def __init__(self):
        self._values: dict[str, list[float]] = {}
        self._counts: dict[str, int] = {}

    def update(self, name: str, value: float):
        """Record a metric value."""
        if name not in self._values:
            self._values[name] = []
            self._counts[name] = 0
        self._values[name].append(value)
        self._counts[name] += 1

    def average(self, name: str) -> float:
        """Get running average for a metric."""
        vals = self._values.get(name, [])
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def reset(self):
        """Reset all metrics."""
        self._values.clear()
        self._counts.clear()

    def summary(self) -> dict[str, float]:
        """Get summary of all metrics."""
        return {name: self.average(name) for name in self._values}
