"""OpenAICompatibleProvider — single class for all OpenAI-compatible APIs.

Works with: OpenAI, DeepSeek, Groq, vLLM, LM Studio, Ollama, and any other
endpoint that speaks the /v1/chat/completions protocol.

Configure via LLMConfig:
  {"provider": "openai", "model": "gpt-4o", "base_url": None}
  {"provider": "openai", "model": "deepseek-chat", "base_url": "https://api.deepseek.com/v1"}
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from sysight.agent.provider import (
    LLMConfig,
    LLMErrorInfo,
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
        }
        thinking = self._config.thinking
        if thinking is not None:
            body["thinking"] = thinking
        if not _thinking_enabled(thinking):
            body["temperature"] = self._config.temperature

        reasoning_effort = self._config.reasoning_effort
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort

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
            try:
                error_body = e.read().decode("utf-8", errors="replace")
            finally:
                e.close()
            error = _build_error_info(
                provider=self.name,
                status_code=e.code,
                body=error_body,
                request_id=e.headers.get("x-request-id", "") if e.headers else "",
            )
            return LLMResponse(
                content="",
                finish_reason="error",
                error=error,
                extra={"http_error": f"{e.code}: {_redact_error(error_body)[:1000]}"},
            )
        except (urllib.error.URLError, OSError) as e:
            error = LLMErrorInfo(
                provider=self.name,
                message=_redact_error(str(e))[:1000],
                type=e.__class__.__name__,
                retryable=True,
            )
            return LLMResponse(
                content="",
                finish_reason="error",
                error=error,
                extra={"http_error": _redact_error(str(e))[:1000]},
            )
        except Exception as e:
            error = LLMErrorInfo(
                provider=self.name,
                message=_redact_error(str(e))[:1000],
                type=e.__class__.__name__,
                retryable=True,
            )
            return LLMResponse(
                content="",
                finish_reason="error",
                error=error,
                extra={"http_error": f"{e.__class__.__name__}: {_redact_error(str(e))[:1000]}"},
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


_BEARER_RE = re.compile(r"(?i)(bearer\s+)([^\"'\s,}]+)")
_SECRET_ASSIGN_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|token|secret|password|passwd|credential)\b"
    r"\s*[:=]\s*(\"[^\"]*\"|'[^']*'|[^\s,}]+)"
)


def _redact_error(text: str) -> str:
    text = _BEARER_RE.sub(r"\1<REDACTED>", text)
    return _SECRET_ASSIGN_RE.sub(lambda m: f"{m.group(1)}=<REDACTED>", text)


def _thinking_enabled(thinking: object) -> bool:
    return isinstance(thinking, dict) and str(thinking.get("type", "")).lower() == "enabled"


def _build_error_info(
    *,
    provider: str,
    status_code: int,
    body: str,
    request_id: str = "",
) -> LLMErrorInfo:
    redacted = _redact_error(body)[:4000]
    parsed = _parse_error_payload(body)
    return LLMErrorInfo(
        provider=provider,
        status_code=status_code,
        code=str(parsed.get("code") or ""),
        message=_redact_error(str(parsed.get("message") or redacted[:1000]))[:1000],
        type=str(parsed.get("type") or "HTTPError"),
        param=parsed.get("param"),
        retryable=_is_retryable_status(status_code),
        request_id=request_id,
        raw_redacted=redacted,
    )


def _parse_error_payload(body: str) -> dict:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            return err
    return {}


def _is_retryable_status(status_code: int | None) -> bool:
    return status_code in {408, 409, 429, 500, 502, 503, 504}
