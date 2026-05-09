"""AgentLoop — explicit tool-calling loop with stop conditions.

Template Method pattern — the loop skeleton is fixed, tool execution is
delegated to ToolRegistry, and LLM calls are delegated to LLMProvider.

Each run() is an independent multi-turn tool-calling session.
No conversation history is inherited between run() calls.
Context is passed via structured artifacts.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field

from sysight.agent.context import AgentContext, ContextPolicy, to_jsonable
from sysight.agent.provider import LLMProvider, LLMRequest, LLMResponse
from sysight.agent.prompts.loader import PromptLoader


@dataclass
class AgentTask:
    """A single agent task specification."""
    run_id: str = ""
    task_id: str = ""
    task_type: str = ""                       # "analyze" | "optimize" | "learn"
    system_prompt: str = ""
    user_prompt: str = ""
    response_schema: dict | None = None
    max_turns: int = 30
    max_wall_seconds: int = 600
    max_tokens: int | None = None             # per-task output token limit; None = use provider default
    context_policy: ContextPolicy | None = None


@dataclass
class AgentResult:
    """Result of an AgentLoop run."""
    run_id: str = ""
    task_id: str = ""
    backend: str = ""
    model: str = ""
    status: str = ""                          # "ok" | "schema_error" | "tool_error" | "timeout" | "provider_error"
    output: dict = field(default_factory=dict)
    raw_content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    context_stats: dict = field(default_factory=dict)
    provider_error: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    turns: int = 0
    elapsed_ms: float = 0
    backoff_ms: float = 0


class AgentLoop:
    """One run() = one independent LLM multi-turn tool-calling session.

    No conversation history is inherited between run() calls.
    Context is passed via structured artifacts.
    """

    def __init__(self, provider: LLMProvider, registry, policy):
        self._provider = provider
        self._registry = registry
        self._policy = policy
        self._prompt_loader = PromptLoader()

    def run(self, task: AgentTask) -> AgentResult:
        """Execute the tool-calling loop.

        1. Build initial messages
        2. Call LLM
        3. If tool calls → execute via ToolRegistry → feed results
        4. If final output → validate → return
        5. Stop on: max_turns, max_wall_seconds, repeated_calls
        """
        t0 = time.monotonic()
        if hasattr(self._registry, "reset_call_counts"):
            self._registry.reset_call_counts()
        if hasattr(self._provider, "reset_cache"):
            self._provider.reset_cache()
        # Inject model_name into context policy for model-aware thresholds
        policy = task.context_policy or ContextPolicy()
        if not policy.model_name and hasattr(self._provider, 'model'):
            policy = ContextPolicy(
                model_name=getattr(self._provider, 'model', ''),
                compact_token_limit=policy.compact_token_limit,
                snip_token_limit=policy.snip_token_limit,
                hard_token_limit=policy.hard_token_limit,
                full_tool_result_once=policy.full_tool_result_once,
                compact_after_first_exposure=policy.compact_after_first_exposure,
                keep_recent_turns_full=policy.keep_recent_turns_full,
                large_result_threshold_tokens=policy.large_result_threshold_tokens,
                large_result_preview_tokens=policy.large_result_preview_tokens,
                restore_recent_files=policy.restore_recent_files,
                restore_file_count=policy.restore_file_count,
                restore_max_tokens_per_file=policy.restore_max_tokens_per_file,
                circuit_breaker_max=policy.circuit_breaker_max,
            )
        context = AgentContext(task.user_prompt, policy)
        system_prompt = task.system_prompt

        tool_calls_log: list[dict] = []
        context_stats_log: list[dict] = []
        errors: list[str] = []
        turns = 0
        backoff_total_s = 0.0

        while turns < task.max_turns:
            elapsed = (time.monotonic() - t0)
            if task.max_wall_seconds > 0 and elapsed > task.max_wall_seconds:
                return AgentResult(
                    run_id=task.run_id, task_id=task.task_id,
                    backend=self._provider.name, model=getattr(self._provider, 'model', ''),
                    status="timeout", errors=["max_wall_seconds exceeded"],
                    context_stats={"turns": context_stats_log, "last": context_stats_log[-1] if context_stats_log else {}},
                    turns=turns, elapsed_ms=elapsed * 1000,
                    backoff_ms=backoff_total_s * 1000,
                )

            turns += 1
            tools = self._registry.as_openai_tools(self._policy) if self._registry else []
            model_messages, context_stats = context.build_model_messages(turns)
            protocol_error = _validate_tool_protocol(model_messages)
            if protocol_error:
                errors.append(f"tool_protocol_error: {protocol_error}")
                return AgentResult(
                    run_id=task.run_id, task_id=task.task_id,
                    backend=self._provider.name, model=getattr(self._provider, 'model', ''),
                    status="tool_error", tool_calls=tool_calls_log,
                    context_stats={
                        "turns": context_stats_log,
                        "last": context_stats.to_dict(),
                    },
                    errors=errors, turns=turns,
                    backoff_ms=backoff_total_s * 1000,
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                )

            request = LLMRequest(
                system_prompt=system_prompt,
                messages=model_messages,
                tools=tools if tools else None,
                response_schema=None,  # Only enforce schema on final output
                debug_messages=context.full_log_messages(),
                context_stats=context_stats.to_dict(),
                max_tokens=task.max_tokens,
            )

            response = None
            retries = 0
            _BACKOFF_S = [5, 10, 20, 30, 40]
            while True:
                response = self._provider.complete(request)
                if response.finish_reason != "error":
                    if retries > 0:
                        print(f"  ↳ turn {turns} retry{retries} succeeded", file=sys.stderr)
                    break
                retryable = response.error and response.error.retryable
                if not retryable:
                    break
                if retries >= len(_BACKOFF_S):
                    break
                delay = _BACKOFF_S[retries]
                retries += 1
                msg = f"  ↳ turn {turns} [retry{retries}]: waiting {delay}s before retry"
                print(msg, file=sys.stderr)
                errors.append(msg)
                time.sleep(delay)
                backoff_total_s += delay

            if response.usage:
                context_stats.prompt_tokens = response.usage.prompt_tokens
                context_stats.output_tokens = response.usage.output_tokens
                # Feed token usage back for progressive compaction estimation
                context.update_token_usage(response.usage.prompt_tokens)
            context_stats_log.append(context_stats.to_dict())

            if response.finish_reason == "error":
                errors.append("provider_error")
                if response.extra.get("http_error"):
                    errors.append(f"provider_http_error: {response.extra['http_error']}")
                provider_error = response.error.to_dict() if response.error else {}
                if provider_error.get("code") or provider_error.get("type"):
                    errors.append(
                        "provider_error_info: "
                        f"type={provider_error.get('type') or 'unknown'} "
                        f"code={provider_error.get('code') or 'unknown'} "
                        f"retryable={provider_error.get('retryable')}"
                    )
                if retries > 0:
                    errors.append(f"retry_attempts: {retries}")
                return AgentResult(
                    run_id=task.run_id, task_id=task.task_id,
                    backend=self._provider.name, model=getattr(self._provider, 'model', ''),
                    status="provider_error", tool_calls=tool_calls_log,
                    context_stats={"turns": context_stats_log, "last": context_stats_log[-1] if context_stats_log else {}},
                    provider_error=provider_error,
                    errors=errors, turns=turns,
                    backoff_ms=backoff_total_s * 1000,
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                )

            # Add assistant message
            if response.tool_calls:
                # Tool calls — log and execute
                if response.extra.get("raw_content_blocks"):
                    # Anthropic format: use raw content blocks directly
                    assistant_msg = {
                        "role": "assistant",
                        "content": response.extra["raw_content_blocks"],
                    }
                else:
                    # OpenAI format
                    assistant_msg = {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                            }
                            for tc in response.tool_calls
                        ],
                    }
                    if response.extra.get("reasoning_content"):
                        assistant_msg["reasoning_content"] = response.extra["reasoning_content"]
                context.append_message(assistant_msg, turn=turns)

                # Execute each tool call
                for tc in response.tool_calls:
                    result = self._registry.execute(tc.name, tc.arguments, self._policy)
                    tool_calls_log.append({
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "status": result.status,
                        "error": result.error,
                        "elapsed_ms": result.elapsed_ms,
                    })

                    # Feed tool result back
                    result_text = (
                        json.dumps(to_jsonable(result.data), ensure_ascii=False, default=str)
                        if result.data is not None else result.error
                    )
                    context.append_tool_result({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    }, turn=turns, tool_name=tc.name, arguments=tc.arguments,
                        status=result.status, error=result.error, data=result.data)
            else:
                # Final output — no tool calls
                # Check if output was truncated (thinking exhausted token budget)
                if response.finish_reason in ("max_tokens", "length"):
                    errors.append(
                        f"LLM output truncated (finish={response.finish_reason}, "
                        f"output_tokens={response.usage.output_tokens if response.usage else '?'}). "
                        f"Consider increasing max_tokens."
                    )
                    # If there's no content at all, this is a hard failure
                    if not response.content.strip():
                        return AgentResult(
                            run_id=task.run_id, task_id=task.task_id,
                            backend=self._provider.name, model=getattr(self._provider, 'model', ''),
                            status="provider_error",
                            tool_calls=tool_calls_log,
                            usage={
                                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                                "output_tokens": response.usage.output_tokens if response.usage else 0,
                            },
                            context_stats={"turns": context_stats_log, "last": context_stats_log[-1] if context_stats_log else {}},
                            errors=errors, turns=turns,
                            backoff_ms=backoff_total_s * 1000,
                            elapsed_ms=(time.monotonic() - t0) * 1000,
                        )

                output = {}
                raw = response.content
                parse_error = None
                if raw:
                    try:
                        output = self._extract_json(raw)
                    except json.JSONDecodeError as e:
                        parse_error = str(e)
                        errors.append(f"schema_error: {parse_error}")
                        # Log to stderr for visibility
                        print(f"  ⚠ turn {turns} JSON parse failed: {parse_error[:200]}", file=sys.stderr)
                        # Try one more turn to let the model fix its JSON
                        if turns < task.max_turns:
                            context.append_message({
                                "role": "user",
                                "content": (
                                    f"你的上一个输出不是有效的 JSON。错误：{parse_error}\n\n"
                                    "请修复 JSON 格式问题后重新输出。注意：JSON 字符串内的双引号必须用 "
                                    "反斜杠转义（如 \\\"key\\\"）。只输出修复后的 JSON，不要加解释。"
                                ),
                            }, turn=turns)
                            continue  # Give the model another chance

                return AgentResult(
                    run_id=task.run_id, task_id=task.task_id,
                    backend=self._provider.name, model=getattr(self._provider, 'model', ''),
                    status="ok" if not errors else "schema_error",
                    output=output, raw_content=raw,
                    tool_calls=tool_calls_log,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "output_tokens": response.usage.output_tokens if response.usage else 0,
                    },
                    context_stats={"turns": context_stats_log, "last": context_stats_log[-1] if context_stats_log else {}},
                    errors=errors, turns=turns,
                    backoff_ms=backoff_total_s * 1000,
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                )

        # Max turns reached
        return AgentResult(
            run_id=task.run_id, task_id=task.task_id,
            backend=self._provider.name, model=getattr(self._provider, 'model', ''),
            status="tool_error", errors=["max_turns exceeded"],
            tool_calls=tool_calls_log,
            context_stats={"turns": context_stats_log, "last": context_stats_log[-1] if context_stats_log else {}},
            turns=turns,
            backoff_ms=backoff_total_s * 1000,
            elapsed_ms=(time.monotonic() - t0) * 1000,
        )

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON object from LLM output (may include markdown fences).

        Uses bracket counting to handle nested objects/arrays correctly,
        unlike non-greedy regex which fails on deeply nested JSON.

        If standard parsing fails, attempts to repair common LLM JSON errors
        (unescaped double quotes inside string values, trailing commas).
        """
        import re
        # Try markdown-fenced JSON first
        fence_m = re.search(r"```(?:json)?\s*", text)
        if fence_m:
            start = fence_m.end()
            # Find matching closing fence
            end_fence = text.find("```", start)
            if end_fence != -1:
                candidate = text[start:end_fence].strip()
                if candidate and candidate[0] in "{[":
                    extracted = _extract_bracket_balanced(candidate)
                    if extracted:
                        try:
                            return json.loads(extracted)
                        except json.JSONDecodeError as e:
                            repaired = _repair_json(extracted, e)
                            if repaired is not None:
                                return repaired

        # Try raw JSON — find first { or [
        text = text.strip()
        for i, ch in enumerate(text):
            if ch in "{[":
                extracted = _extract_bracket_balanced(text[i:])
                if extracted:
                    try:
                        return json.loads(extracted)
                    except json.JSONDecodeError as e:
                        repaired = _repair_json(extracted, e)
                        if repaired is not None:
                            return repaired
                break

        raise json.JSONDecodeError("No JSON found", text, 0)


def _extract_bracket_balanced(text: str) -> str | None:
    """Extract a bracket-balanced JSON string from text.

    Uses character-by-character bracket counting, respecting string literals.
    Returns the balanced substring or None if brackets are not balanced.
    """
    if not text or text[0] not in "{[":
        return None

    openers: dict[str, str] = {"{": "}", "[": "]"}
    closers: dict[str, str] = {"}": "{", "]": "["}
    opener = text[0]
    closer = openers[opener]
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[:i + 1]

    return None


def _repair_json(text: str, error: json.JSONDecodeError) -> dict | None:
    """Attempt to repair common LLM JSON errors and re-parse.

    Handles:
    - Unescaped double quotes inside string values
      (e.g., "desc": "json.dumps({"key": val})" → "desc": "json.dumps({\\"key\\": val})")
    - Trailing commas before ] or }

    Returns the parsed dict if repair succeeds, None otherwise.
    """
    MAX_REPAIR_ATTEMPTS = 30
    pos = error.pos
    repaired = text

    for _ in range(MAX_REPAIR_ATTEMPTS):
        if pos >= len(repaired):
            break

        ch = repaired[pos]

        if ch == '"':
            # Likely an unescaped quote inside a string value.
            repaired = repaired[:pos] + '\\"' + repaired[pos + 1:]
            try:
                return json.loads(repaired)
            except json.JSONDecodeError as e2:
                if e2.pos <= pos:
                    break
                pos = e2.pos
                continue

        elif ch == ',':
            # Check for trailing comma before ] or }
            after = repaired[pos + 1:].lstrip()
            if after and after[0] in ']}':
                repaired = repaired[:pos] + repaired[pos + 1:]
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError as e2:
                    if e2.pos <= pos:
                        break
                    pos = e2.pos
                    continue
            else:
                # Not a trailing comma — try looking backwards for unescaped quote
                prev_quote = _find_prev_unescaped_quote(repaired, pos)
                if prev_quote >= 0:
                    repaired = repaired[:prev_quote] + '\\"' + repaired[prev_quote + 1:]
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError as e2:
                        if e2.pos <= prev_quote:
                            break
                        pos = e2.pos
                        continue
                else:
                    break

        else:
            # Unknown error — try looking backwards for unescaped quote
            prev_quote = _find_prev_unescaped_quote(repaired, pos)
            if prev_quote >= 0:
                repaired = repaired[:prev_quote] + '\\"' + repaired[prev_quote + 1:]
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError as e2:
                    if e2.pos <= prev_quote:
                        break
                    pos = e2.pos
                    continue
            else:
                break

    return None


def _find_prev_unescaped_quote(text: str, before_pos: int) -> int:
    """Find the nearest unescaped double-quote before `before_pos`.

    Returns the index of the quote, or -1 if not found.
    """
    # Search backwards from before_pos-1
    for i in range(before_pos - 1, -1, -1):
        if text[i] == '"':
            # Check if escaped
            backslash_count = 0
            j = i - 1
            while j >= 0 and text[j] == '\\':
                backslash_count += 1
                j -= 1
            if backslash_count % 2 == 0:
                # Not escaped
                return i
    return -1


def _validate_tool_protocol(messages: list[dict]) -> str:
    """Validate OpenAI-style assistant tool_calls/tool result adjacency."""
    for idx, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            continue
        expected_ids = [call.get("id") for call in tool_calls if call.get("id")]
        following = messages[idx + 1: idx + 1 + len(expected_ids)]
        if len(following) != len(expected_ids):
            return f"assistant message {idx} has {len(expected_ids)} tool_calls but not enough following messages"
        actual_ids: list[str] = []
        for item in following:
            if item.get("role") != "tool":
                return f"assistant message {idx} tool_calls followed by non-tool role {item.get('role')!r}"
            actual_ids.append(item.get("tool_call_id"))
        if actual_ids != expected_ids:
            return f"assistant message {idx} tool_call_ids mismatch: expected={expected_ids} actual={actual_ids}"
    return ""
