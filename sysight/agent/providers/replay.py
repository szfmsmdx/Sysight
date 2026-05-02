"""ReplayProvider — deterministic testing backend, no real API calls."""

from __future__ import annotations

import json
from pathlib import Path

from sysight.agent.provider import LLMConfig, LLMRequest, LLMResponse


class ReplayProvider:
    """Returns pre-recorded responses from fixture files. No real API calls."""

    def __init__(self, config: LLMConfig):
        self._config = config
        self._fixtures: list[LLMResponse] = []
        self._cursor: int = 0

    @property
    def name(self) -> str:
        return "replay"

    @property
    def model(self) -> str:
        return self._config.model

    def load_fixtures(self, responses: list[LLMResponse]) -> None:
        """Load pre-built LLMResponse objects."""
        self._fixtures = responses
        self._cursor = 0

    def load_fixture_file(self, path: str | Path) -> None:
        """Load responses from a JSON fixture file.

        TODO: Stage 4 — implement fixture file loading.
        """
        raise NotImplementedError("TODO: Stage 4")

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return the next pre-recorded response."""
        if self._cursor >= len(self._fixtures):
            return LLMResponse(
                content="",
                finish_reason="stop",
            )
        response = self._fixtures[self._cursor]
        self._cursor += 1
        return response
