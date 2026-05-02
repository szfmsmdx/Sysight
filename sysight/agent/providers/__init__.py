"""LLM provider implementations.

Two real backends + one test backend:
  - OpenAICompatibleProvider: OpenAI, DeepSeek, Groq, vLLM, etc.
  - AnthropicProvider: Anthropic Messages API (Claude)
  - ReplayProvider: deterministic testing with pre-recorded responses

Users configure via LLMConfig dict, not by importing specific providers:
  from sysight.agent.provider import create_provider
  provider = create_provider({"provider": "openai", "model": "gpt-4o"})
"""

from __future__ import annotations
