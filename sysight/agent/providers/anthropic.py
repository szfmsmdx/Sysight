"""AnthropicProvider — for Anthropic Messages API (Claude models).

Endpoint: POST https://api.anthropic.com/v1/messages
Docs: https://docs.anthropic.com/en/api/messages
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


class AnthropicProvider:
    """Provider for Anthropic's native Messages API."""

    _DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
    _ANTHROPIC_VERSION = "2023-06-01"

    def __init__(self, config: LLMConfig):
        self._config = config
        self._api_key = config.resolve_api_key()
        self._base_url = config.base_url or self._DEFAULT_BASE_URL

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._config.model

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a message request to the Anthropic Messages API."""
        url = self._base_url.rstrip("/") + "/messages"

        # max_tokens is REQUIRED by Anthropic API
        max_tokens = self._config.max_tokens if self._config.max_tokens else 4096

        body: dict = {
            "model": self._config.model,
            "max_tokens": max_tokens,
            "messages": self._build_messages(request),
        }

        if request.system_prompt:
            body["system"] = request.system_prompt

        if request.tools:
            body["tools"] = self._convert_tools(request.tools)

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
            error_body = e.read().decode("utf-8", errors="replace")
            return LLMResponse(content="", finish_reason="error",
                               extra={"http_error": f"{e.code}: {error_body[:300]}"})
        except (urllib.error.URLError, OSError) as e:
            return LLMResponse(content="", finish_reason="error",
                               extra={"http_error": str(e)[:300]})

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
