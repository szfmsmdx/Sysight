"""Minimal skill abstraction for the analysis layer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from nsys_agent.profile import Profile


@dataclass(frozen=True)
class Skill:
    """Small wrapper around a reusable analysis unit."""

    name: str
    title: str
    description: str
    runner: Callable
    formatter: Callable

    def run(
        self, prof: Profile, device: int | None = None, trim: tuple[int, int] | None = None
    ):
        return self.runner(prof, device=device, trim=trim)

    def format(self, result) -> str:
        return self.formatter(result)
