"""Minimal skill abstraction for the analysis layer."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass

from sysight.profile import Profile


@dataclass(frozen=True)
class Skill:
    """Small wrapper around a reusable analysis unit."""

    name: str
    title: str
    description: str
    runner: Callable
    formatter: Callable

    def run(
        self,
        prof: Profile,
        device: int | None = None,
        trim: tuple[int, int] | None = None,
        **kwargs,
    ):
        signature = inspect.signature(self.runner)
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            accepted_kwargs = kwargs
        else:
            accepted_kwargs = {
                key: value for key, value in kwargs.items() if key in signature.parameters
            }
        return self.runner(prof, device=device, trim=trim, **accepted_kwargs)

    def format(self, result) -> str:
        return self.formatter(result)
