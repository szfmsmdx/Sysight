from __future__ import annotations

import abc

import typer


class Skill(abc.ABC):
    """Base class for CLI skills (plugins)."""

    @abc.abstractmethod
    def register(self, cli: typer.Typer) -> None:
        ...
