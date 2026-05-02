"""OpenAICompatibleProvider — single class for all OpenAI-compatible APIs.

Works with: OpenAI, DeepSeek, Groq, vLLM, LM Studio, Ollama, and any other
endpoint that speaks the /v1/chat/completions protocol.

Configure via LLMConfig:
  {"provider": "openai", "model": "gpt-4o", "base_url": None}
  {"provider": "openai", "model": "deepseek-chat", "base_url": "https://api.deepseek.com/v1"}
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from sysight.agent.provider import (
    LLMConfig,
    LLMRequest,
    LLMResponse,
    ToolCallRequest,
    UsageInfo,
)


class OpenAICompatibleProvider:
    """Provider for OpenAI-compatible chat completions API.

    Handles: OpenAI, DeepSeek, Groq, vLLM, LM Studio, Ollama, etc.
    """

    _DEFAULT_BASE_URLS = {
        "openai": "https://api.openai.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "groq": "https://api.groq.com/openai/v1",
    }

    def __init__(self, config: LLMConfig):
        self._config = config
        self._api_key = config.resolve_api_key()
        self._base_url = config.base_url or self._DEFAULT_BASE_URLS.get(
            config.provider, "https://api.openai.com/v1"
        )

    @property
    def name(self) -> str:
        return self._config.provider

    @property
    def model(self) -> str:
        return self._config.model

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a chat completion request to the OpenAI-compatible endpoint."""
        url = self._base_url.rstrip("/") + "/chat/completions"

        body: dict = {
            "model": self._config.model,
            "messages": self._build_messages(request),
            "temperature": self._config.temperature,
        }

        if self._config.max_tokens and self._config.max_tokens > 0:
            body["max_tokens"] = self._config.max_tokens

        if request.tools:
            body["tools"] = request.tools

        if request.response_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": request.response_schema,
                },
            }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            return LLMResponse(
                content="",
                finish_reason="error",
            )
        except (urllib.error.URLError, OSError) as e:
            return LLMResponse(
                content="",
                finish_reason="error",
            )

        return self._parse_response(data)

    def _build_messages(self, request: LLMRequest) -> list[dict]:
        messages: list[dict] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        for m in request.messages:
            messages.append(m)
        return messages

    def _parse_response(self, data: dict) -> LLMResponse:
        choice = data.get("choices", [{}])[0] if data.get("choices") else {}
        message = choice.get("message", {})

        content = message.get("content") or ""
        finish_reason = choice.get("finish_reason", "stop")

        # Parse tool calls
        tool_calls = []
        raw_tool_calls = message.get("tool_calls") or []
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            args = {}
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                pass
            tool_calls.append(ToolCallRequest(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                arguments=args,
            ))

        # Parse usage
        usage = None
        if data.get("usage"):
            u = data["usage"]
            usage = UsageInfo(
                prompt_tokens=u.get("prompt_tokens", 0),
                output_tokens=u.get("completion_tokens", 0),
            )

        extra = {}
        if message.get("reasoning_content"):
            extra["reasoning_content"] = message["reasoning_content"]

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
            extra=extra,
        )
