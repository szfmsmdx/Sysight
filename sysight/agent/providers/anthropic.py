"""AnthropicProvider — for Anthropic Messages API (Claude models).

Endpoint: POST https://api.anthropic.com/v1/messages
Docs: https://docs.anthropic.com/en/api/messages
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


class AnthropicProvider:
    """Provider for Anthropic's native Messages API."""

    _DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
    _ANTHROPIC_VERSION = "2023-06-01"

    def __init__(self, config: LLMConfig):
        self._config = config
        self._api_key = config.resolve_api_key()
        self._base_url = config.base_url or self._DEFAULT_BASE_URL
        self._cache_initialized = False

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._config.model

    def reset_cache(self) -> None:
        """Reset prompt cache state for a new independent session."""
        self._cache_initialized = False

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a message request to the Anthropic Messages API."""
        url = self._base_url.rstrip("/") + "/messages"

        # max_tokens is REQUIRED by Anthropic API
        # Per-request override takes priority, then config, then fallback 4096
        max_tokens = request.max_tokens or self._config.max_tokens or 4096

        is_first_request = not self._cache_initialized

        body: dict = {
            "model": self._config.model,
            "max_tokens": max_tokens,
            "messages": self._build_messages(request),
        }

        if request.system_prompt:
            if is_first_request:
                body["system"] = [{
                    "type": "text",
                    "text": request.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                body["system"] = request.system_prompt

        if request.tools:
            body["tools"] = self._convert_tools(request.tools)

        self._cache_initialized = True

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": self._ANTHROPIC_VERSION,
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
                request_id=e.headers.get("request-id", "") if e.headers else "",
            )
            return LLMResponse(content="", finish_reason="error",
                               error=error,
                               extra={"http_error": f"{e.code}: {_redact_error(error_body)[:1000]}"})
        except (urllib.error.URLError, OSError) as e:
            error = LLMErrorInfo(
                provider=self.name,
                message=_redact_error(str(e))[:1000],
                type=e.__class__.__name__,
                retryable=True,
            )
            return LLMResponse(content="", finish_reason="error",
                               error=error,
                               extra={"http_error": _redact_error(str(e))[:1000]})
        except Exception as e:
            error = LLMErrorInfo(
                provider=self.name,
                message=_redact_error(str(e))[:1000],
                type=e.__class__.__name__,
                retryable=True,
            )
            return LLMResponse(content="", finish_reason="error",
                               error=error,
                               extra={"http_error": f"{e.__class__.__name__}: {_redact_error(str(e))[:1000]}"})

        return self._parse_response(data)

    def _build_messages(self, request: LLMRequest) -> list[dict]:
        """Convert internal (OpenAI-style) messages to Anthropic format.

        Key difference: Anthropic requires ALL tool_results for a single
        assistant turn to be in ONE user message. OpenAI puts each in a
        separate message. This method merges consecutive tool messages.
        """
        messages: list[dict] = []
        pending_tool_results: list[dict] = []

        def _flush_tool_results():
            if pending_tool_results:
                messages.append({"role": "user", "content": pending_tool_results.copy()})
                pending_tool_results.clear()

        for m in request.messages:
            role = m.get("role", "user")
            if role == "system":
                continue

            if role == "tool":
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", ""),
                    "content": m.get("content", ""),
                })
            else:
                _flush_tool_results()
                if role == "assistant":
                    content = m.get("content", "")
                    if isinstance(content, list):
                        messages.append({"role": "assistant", "content": content})
                    elif "tool_calls" in m:
                        content_blocks: list[dict] = []
                        if m.get("content"):
                            content_blocks.append({"type": "text", "text": m["content"]})
                        for tc in m.get("tool_calls", []):
                            fn = tc.get("function", {})
                            args = fn.get("arguments", "{}")
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    pass
                            content_blocks.append({
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": fn.get("name", ""),
                                "input": args,
                            })
                        messages.append({"role": "assistant", "content": content_blocks})
                    else:
                        messages.append({"role": "assistant", "content": content})
                else:
                    messages.append({"role": role, "content": m.get("content", "")})

        _flush_tool_results()
        return messages

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI-format tools to Anthropic format."""
        anthropic_tools = []
        for t in tools:
            fn = t.get("function", t)
            anthropic_tools.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", fn.get("input_schema", {})),
            })
        return anthropic_tools

    def _parse_response(self, data: dict) -> LLMResponse:
        content_blocks = data.get("content", [])
        text_parts = []
        tool_calls = []

        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCallRequest(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=block.get("input", {}),
                ))

        stop_reason = data.get("stop_reason", "end_turn")
        finish_reason = "stop" if stop_reason in ("end_turn", "tool_use") else stop_reason

        usage = None
        if data.get("usage"):
            u = data["usage"]
            usage = UsageInfo(
                prompt_tokens=u.get("input_tokens", 0),
                output_tokens=u.get("output_tokens", 0),
            )

        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
            extra={"raw_content_blocks": content_blocks},
        )


_BEARER_RE = re.compile(r"(?i)(bearer\s+)([^\"'\s,}]+)")
_SECRET_ASSIGN_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|token|secret|password|passwd|credential)\b"
    r"\s*[:=]\s*(\"[^\"]*\"|'[^']*'|[^\s,}]+)"
)


def _redact_error(text: str) -> str:
    text = _BEARER_RE.sub(r"\1<REDACTED>", text)
    return _SECRET_ASSIGN_RE.sub(lambda m: f"{m.group(1)}=<REDACTED>", text)


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
