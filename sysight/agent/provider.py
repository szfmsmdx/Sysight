"""LLM provider abstraction — config-driven, not class-per-provider.

Two API protocols exist:
  - OpenAI-compatible (/v1/chat/completions): OpenAI, DeepSeek, Groq, vLLM, etc.
  - Anthropic native (Messages API): Claude models

Users configure via YAML/dict, not by writing Python classes:

  config = {
      "provider": "openai",           # "openai" | "anthropic" | "replay"
      "model": "gpt-4o",
      "api_key": "$OPENAI_API_KEY",   # or env var name
      "base_url": None,               # None = default, or custom endpoint
      "temperature": 0,
      "max_tokens": 4096,
  }
  provider = create_provider(config)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class LLMConfig:
    """Configuration for an LLM provider. Loaded from YAML/dict by the user."""
    provider: str = ""                    # "openai" | "anthropic" | "replay"
    model: str = ""
    api_key: str = ""                     # literal key or "$ENV_VAR_NAME"
    base_url: str | None = None           # None = provider default
    temperature: float = 0
    max_tokens: int | None = None    # None = no limit / use API default

    @classmethod
    def from_dict(cls, d: dict) -> LLMConfig:
        return cls(
            provider=d.get("provider", ""),
            model=d.get("model", ""),
            api_key=d.get("api_key", ""),
            base_url=d.get("base_url"),
            temperature=d.get("temperature", 0),
            max_tokens=d.get("max_tokens", 4096),
        )

    def resolve_api_key(self) -> str:
        """Resolve api_key: if starts with '$', look up env var."""
        if self.api_key.startswith("$"):
            return os.environ.get(self.api_key[1:], "")
        return self.api_key


@dataclass
class LLMRequest:
    """A single completion request."""
    system_prompt: str = ""
    messages: list[dict] = field(default_factory=list)
    tools: list[dict] | None = None
    response_schema: dict | None = None


@dataclass
class ToolCallRequest:
    """A tool call from the LLM."""
    id: str = ""
    name: str = ""
    arguments: dict = field(default_factory=dict)


@dataclass
class UsageInfo:
    """Token usage information."""
    prompt_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResponse:
    """A single completion response."""
    content: str = ""
    structured_output: dict | None = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    usage: UsageInfo | None = None
    finish_reason: str = ""
    extra: dict = field(default_factory=dict)  # provider-specific fields (e.g. reasoning_content)


class LLMProvider(Protocol):
    """Protocol for LLM provider backends."""

    @property
    def name(self) -> str: ...

    def complete(self, request: LLMRequest) -> LLMResponse: ...


def create_provider(config: LLMConfig | dict) -> LLMProvider:
    """Factory: create the right provider from config.

    Supports:
      - "openai"   → OpenAI-compatible API (also works for DeepSeek, Groq, vLLM)
      - "anthropic" → Anthropic Messages API
      - "replay"   → replay provider for testing
    """
    if isinstance(config, dict):
        config = LLMConfig.from_dict(config)

    if config.provider == "anthropic":
        from sysight.agent.providers.anthropic import AnthropicProvider
        return AnthropicProvider(config)
    elif config.provider == "replay":
        from sysight.agent.providers.replay import ReplayProvider
        return ReplayProvider(config)
    else:
        # Default: OpenAI-compatible (covers openai, deepseek, groq, vllm, etc.)
        from sysight.agent.providers.openai_compatible import OpenAICompatibleProvider
        return OpenAICompatibleProvider(config)
