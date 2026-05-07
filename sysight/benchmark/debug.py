"""DebugProvider — wraps an LLMProvider to capture and print all request/response pairs."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sysight.agent.provider import LLMRequest, LLMResponse


class DebugProvider:
    """Wraps an LLMProvider to intercept, log, and print all complete() calls.

    Transparent to the AgentLoop — same interface, extra logging + real-time terminal output.
    """

    def __init__(self, real_provider, log: list[dict] | None = None, verbose: bool = True,
                 log_file: str | None = None):
        self._real = real_provider
        self._log = log if log is not None else []
        self._verbose = verbose
        self._log_file = Path(log_file) if log_file else None
        self._turn = 0
        self._file_entries = 0

    @property
    def name(self) -> str:
        return self._real.name

    @property
    def model(self) -> str:
        return getattr(self._real, 'model', '')

    def complete(self, request: LLMRequest) -> LLMResponse:
        self._turn += 1

        tools_list = []
        if request.tools:
            for t in request.tools:
                fn = t.get("function", {})
                tools_list.append(fn.get("name", "?"))

        msgs = request.messages
        debug_msgs = request.debug_messages or request.messages
        last_role = msgs[-1].get("role", "?") if msgs else "none"

        # --- Real-time terminal output: request ---
        if self._verbose:
            print(f"\n  {'─'*50}", file=sys.stderr)
            print(f"  Turn {self._turn}  |  {len(msgs)} msgs  |  last={last_role}", file=sys.stderr)
            if request.context_stats:
                print(f"  context: {json.dumps(request.context_stats, ensure_ascii=False)}", file=sys.stderr)
            print(f"  {'─'*50}", file=sys.stderr)

            # Turn 1 only: show system prompt + user prompt in terminal.
            # Later turns omit request body — tool results are logged to debug.log.
            if self._turn == 1:
                sp = request.system_prompt
                if sp:
                    print(f"  [system] {len(sp)} chars:", file=sys.stderr)
                    for line in sp.split("\n"):
                        print(f"    {line}", file=sys.stderr)

                for m in debug_msgs:
                    role = m.get("role", "?")
                    content = str(m.get("content", ""))
                    print(f"  [{role}] {len(content)} chars:", file=sys.stderr)
                    for line in content.split("\n"):
                        print(f"    {line}", file=sys.stderr)

            print(f"  [req] waiting for LLM...", file=sys.stderr, end="", flush=True)

        # --- Actual LLM call ---
        response = self._real.complete(request)

        # --- Real-time terminal output: response ---
        if self._verbose:
            usage_str = ""
            if response.usage:
                usage_str = f" | prompt={response.usage.prompt_tokens} out={response.usage.output_tokens}"
            print(f"\r  [res] finish={response.finish_reason}{usage_str}          ", file=sys.stderr)

            tool_calls = response.tool_calls
            if tool_calls:
                print(f"  [res] tool_calls ({len(tool_calls)}):", file=sys.stderr)
                for tc in tool_calls:
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                    print(f"    → {tc.name}({args_str})", file=sys.stderr)

            if response.content and not tool_calls:
                print(f"  [res] content ({len(response.content)} chars):", file=sys.stderr)
                for line in (response.content or "").split("\n"):
                    print(f"    {line}", file=sys.stderr)

            if response.extra:
                for k, v in response.extra.items():
                    val_str = str(v)
                    print(f"  [res] extra.{k} ({len(val_str)} chars):", file=sys.stderr)
                    for line in val_str.split("\n"):
                        print(f"    {line}", file=sys.stderr)
            if response.error:
                err = json.dumps(response.error.to_dict(), ensure_ascii=False, indent=2)
                print(f"  [res] provider_error ({len(err)} chars):", file=sys.stderr)
                for line in err.split("\n"):
                    print(f"    {line}", file=sys.stderr)

        # --- Build log entry (full detail for debug.log) ---
        entry = {
            "turn": self._turn,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request": {
                "system_prompt": request.system_prompt,
                "messages_summary": [
                    {"role": m.get("role"), "content_len": len(str(m.get("content", "")))}
                    for m in debug_msgs
                ],
                "messages": debug_msgs,
                "model_messages_summary": [
                    {"role": m.get("role"), "content_len": len(str(m.get("content", "")))}
                    for m in request.messages
                ],
                "context_stats": request.context_stats,
                "tools": tools_list,
            },
            "response": {
                "finish_reason": response.finish_reason,
                "content": response.content or "",
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.output_tokens if response.usage else 0,
                } if response.usage else None,
                "error": response.error.to_dict() if response.error else None,
            },
        }
        if response.extra:
            entry["response"]["extra"] = {
                k: str(v)
                for k, v in response.extra.items()
            }

        self._log.append(entry)

        # Flush entry to disk immediately so the log survives crashes
        if self._log_file:
            self._file_entries += 1
            self._flush_entry(entry)

        return response

    def _flush_entry(self, entry: dict) -> None:
        assert self._log_file is not None
        lines: list[str] = []
        prev_entries = self._file_entries - 1
        if prev_entries == 0:
            lines.append("=" * 72)
            lines.append(f"  CASE: {self._log_file.parent.name}")
            lines.append("=" * 72)
            lines.append("")
        lines.append(f"  ── Turn {entry['turn']} ──")
        req = entry.get("request", {})
        resp = entry.get("response", {})
        lines.append(f"  tools ({len(req.get('tools', []))}): {', '.join(req.get('tools', []))}")
        if req.get("context_stats"):
            lines.append(f"  context_stats: {json.dumps(req['context_stats'], ensure_ascii=False)}")
        if req.get("system_prompt"):
            lines.append(f"  system_prompt ({len(req['system_prompt'])} chars)")
        for m in req.get("messages_summary", []):
            lines.append(f"    [{m.get('role')}] ({m.get('content_len')} chars)")
        if req.get("messages"):
            for m in req["messages"]:
                content = str(m.get("content", ""))
                role = m.get("role", "?")
                lines.append(f"  [{role}] {len(content)} chars")
                for line in content.split("\n"):
                    lines.append(f"    {line}")
        lines.append(f"\n[RESPONSE]")
        lines.append(f"  finish_reason: {resp.get('finish_reason')}")
        if resp.get("usage"):
            u = resp["usage"]
            lines.append(f"  usage: prompt_tokens={u.get('prompt_tokens')}, output_tokens={u.get('output_tokens')}")
        if resp.get("tool_calls"):
            for tc in resp["tool_calls"]:
                args = json.dumps(tc.get("arguments", {}), ensure_ascii=False)
                lines.append(f"    → {tc.get('name')}({args})")
        if resp.get("content"):
            lines.append(f"  content ({len(resp['content'])} chars):\n{resp['content']}")
        if resp.get("extra"):
            for k, v in resp["extra"].items():
                lines.append(f"  extra.{k} ({len(str(v))} chars):\n{str(v)}")
        if resp.get("error"):
            lines.append(f"  error: {json.dumps(resp['error'], ensure_ascii=False, indent=2)}")
        lines.append("")
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
