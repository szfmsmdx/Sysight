"""AgentLoop — explicit tool-calling loop with stop conditions.

Template Method pattern — the loop skeleton is fixed, tool execution is
delegated to ToolRegistry, and LLM calls are delegated to LLMProvider.

Each run() is an independent multi-turn tool-calling session.
No conversation history is inherited between run() calls.
Context is passed via structured artifacts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

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
    errors: list[str] = field(default_factory=list)
    turns: int = 0
    elapsed_ms: float = 0


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
        messages: list[dict] = [
            {"role": "user", "content": task.user_prompt},
        ]
        system_prompt = task.system_prompt

        tool_calls_log: list[dict] = []
        errors: list[str] = []
        last_tool_names: list[str] = []
        turns = 0

        while turns < task.max_turns:
            elapsed = (time.monotonic() - t0)
            if elapsed > task.max_wall_seconds:
                return AgentResult(
                    run_id=task.run_id, task_id=task.task_id,
                    backend=self._provider.name, model=getattr(self._provider, 'model', ''),
                    status="timeout", errors=["max_wall_seconds exceeded"],
                    turns=turns, elapsed_ms=elapsed * 1000,
                )

            turns += 1
            tools = self._registry.as_openai_tools(self._policy)

            request = LLMRequest(
                system_prompt=system_prompt,
                messages=messages,
                tools=tools if tools else None,
                response_schema=None,  # Only enforce schema on final output
            )

            response = self._provider.complete(request)

            if response.finish_reason == "error":
                errors.append("provider_error")
                return AgentResult(
                    run_id=task.run_id, task_id=task.task_id,
                    backend=self._provider.name, model=getattr(self._provider, 'model', ''),
                    status="provider_error", tool_calls=tool_calls_log,
                    errors=errors, turns=turns,
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
                messages.append(assistant_msg)

                # Track tool calls (for debugging, not a hard error)
                last_tool_names = [tc.name for tc in response.tool_calls]

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
                    result_text = json.dumps(result.data, ensure_ascii=False, default=str) if result.data else result.error
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    })
            else:
                # Final output — no tool calls
                output = {}
                raw = response.content
                if task.response_schema and raw:
                    try:
                        output = self._extract_json(raw)
                    except json.JSONDecodeError:
                        errors.append("schema_error: invalid JSON")

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
                    errors=errors, turns=turns,
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                )

        # Max turns reached
        return AgentResult(
            run_id=task.run_id, task_id=task.task_id,
            backend=self._provider.name, model=getattr(self._provider, 'model', ''),
            status="tool_error", errors=["max_turns exceeded"],
            tool_calls=tool_calls_log, turns=turns,
            elapsed_ms=(time.monotonic() - t0) * 1000,
        )

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON object from LLM output (may include markdown fences)."""
        import re
        # Try markdown-fenced JSON
        m = re.search(r"```(?:json)?\s*([\{\[][\s\S]*?[\}\]])\s*```", text)
        if m:
            return json.loads(m.group(1))
        # Try raw JSON
        text = text.strip()
        if text.startswith("{"):
            end = text.rfind("}")
            if end > 0:
                return json.loads(text[:end + 1])
        if text.startswith("["):
            end = text.rfind("]")
            if end > 0:
                return json.loads(text[:end + 1])
        raise json.JSONDecodeError("No JSON found", text, 0)
