"""Context management for multi-turn tool-calling agents.

Progressive compaction pipeline (inspired by Claude Code / Codex / DeepSeek TUI):

  Level 1 — Large-Result Persistence (zero model cost)
    Tool results >50K chars → persist full output to disk, keep 2KB preview.
    Model can re-read via scanner_read if needed.

  Level 2 — Time-Based Compaction (Codex "late compaction" strategy)
    Two-tier: when estimated tokens exceed *compact_token_limit* (90% of
    window), results older than *keep_recent_turns_full* → compact.
    Below that threshold, results older than *keep_recent_turns_full* × 3
    are still compacted to prevent unbounded growth in medium sessions.

  Level 3 — Token-Pressure Compaction
    When estimated prompt tokens exceed *hard_token_limit* (95% of window)
    → aggressively compact ALL old results and inject a recovery message
    containing recently-read file contents + session progress summary.

  Circuit Breaker (Claude Code pattern)
    After N consecutive Level-3 compaction events, further compaction is
    blocked to prevent infinite compaction loops.

Token estimation uses the anchor+delta method: server-reported prompt_tokens
from the last API response serve as the anchor; only the delta (new messages
added since then) is estimated client-side.  Typical error <5 %.

Model-aware thresholds (DeepSeek-TUI pattern)
    soft/hard token limits are auto-scaled based on the model's context
    window.  Models with 1M+ context (DeepSeek V4, Gemini 2.5) get much
    higher thresholds than 128K models.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 3.5  # conservative estimate for code-heavy content

# Model → context window mapping (DeepSeek-TUI "model-aware thresholds" pattern)
# Data sourced from official docs as of May 2026.
_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # OpenAI (GPT-5.5 released 2026-04-23, 1M API / 400K Codex)
    "gpt-5.5": 1_000_000,
    "gpt-5.4": 1_000_000,
    "gpt-5.2": 400_000,
    "gpt-5.1": 400_000,
    "gpt-5": 400_000,
    "o4": 200_000,
    "o4-mini": 200_000,
    # Anthropic (Opus 4.7: 200K std / 1M beta; 4.6: 1M GA)
    "claude-opus-4-7": 200_000,
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    # DeepSeek (V4 Pro/Flash: 1M, released 2026-04-24)
    "deepseek-v4-pro": 1_000_000,
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4": 1_000_000,
    "deepseek-v3.2": 128_000,
    "deepseek-r1": 128_000,
    # Google (Gemini 3.1 Pro / 3 Flash: 1M; 3 Pro: 1M API / 10M Vertex)
    "gemini-3.1-pro": 1_048_576,
    "gemini-3-flash": 1_048_576,
    "gemini-3-pro": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
}

# Default: DeepSeek V4 Pro (1M context, cost-effective for agentic workloads)
_DEFAULT_CONTEXT_WINDOW = 1_000_000

# Percentage of context window used as token limits (Codex "late compaction")
_COMPACT_THRESHOLD_RATIO = 0.90  # start time-based compaction at 90% of window
_HARD_LIMIT_RATIO = 0.95         # aggressive pressure compaction at 95% of window
_CIRCUIT_BREAKER_MAX = 3         # consecutive Level-3 events before blocking


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass
class ContextPolicy:
    """Controls how the model-facing context is built.

    If *model_name* is provided, soft/hard token limits are auto-scaled
    based on the model's known context window (DeepSeek-TUI pattern).
    You can still override by setting soft/hard_token_limit explicitly.
    """

    # ---- Model-aware thresholds (DeepSeek-TUI pattern) ----
    model_name: str = ""

    # ---- Token budget (auto-scaled if model_name is set) ----
    soft_token_limit: int = 0       # 0 → auto-compute from model_name
    compact_token_limit: int = 0    # 0 → auto-compute (90% of window); threshold for Level 2
    hard_token_limit: int = 0       # 0 → auto-compute (95% of window); threshold for Level 3

    # ---- Time-based compaction (Level 2) ----
    full_tool_result_once: bool = True
    compact_after_first_exposure: bool = True
    keep_recent_turns_full: int = 3   # increased from 1

    # ---- Large-file persistence (Level 1) ----
    # Thresholds in tokens (not chars).  Internally converted via _CHARS_PER_TOKEN.
    # Claude Code uses 50K chars (~14K tokens); V4 Pro aligned at 20K.
    large_result_threshold_tokens: int = 20_000
    large_result_preview_tokens: int = 600

    # ---- Recovery after aggressive compaction (Level 3) ----
    restore_recent_files: bool = True
    restore_file_count: int = 5
    # Claude Code: ≤5K tokens per recovered file
    restore_max_tokens_per_file: int = 5_000

    # ---- Circuit breaker (Claude Code pattern) ----
    circuit_breaker_max: int = _CIRCUIT_BREAKER_MAX

    def effective_soft_limit(self) -> int:
        """Return the effective soft token limit (auto-scaled if model_name set)."""
        if self.soft_token_limit > 0:
            return self.soft_token_limit
        window = self._context_window()
        return int(window * _COMPACT_THRESHOLD_RATIO)

    def effective_compact_limit(self) -> int:
        """Return the threshold for time-based compaction (Level 2).

        Below this limit, no compaction at all — keep full history.
        """
        if self.compact_token_limit > 0:
            return self.compact_token_limit
        window = self._context_window()
        return int(window * _COMPACT_THRESHOLD_RATIO)

    def effective_hard_limit(self) -> int:
        """Return the effective hard token limit (auto-scaled if model_name set)."""
        if self.hard_token_limit > 0:
            return self.hard_token_limit
        window = self._context_window()
        return int(window * _HARD_LIMIT_RATIO)

    def _context_window(self) -> int:
        """Look up context window for model_name, with fuzzy matching."""
        if self.model_name in _MODEL_CONTEXT_WINDOWS:
            return _MODEL_CONTEXT_WINDOWS[self.model_name]
        # Fuzzy: try case-insensitive substring match
        name_lower = self.model_name.lower()
        for key, window in _MODEL_CONTEXT_WINDOWS.items():
            if key in name_lower or name_lower in key:
                return window
        return _DEFAULT_CONTEXT_WINDOW


# ---------------------------------------------------------------------------
# Internal storage
# ---------------------------------------------------------------------------

@dataclass
class _StoredMessage:
    message: dict
    turn: int
    kind: str = "message"               # "message" | "tool"
    compact_message: dict | None = None
    full_chars: int = 0
    compact_chars: int = 0
    full_tokens: int = 0
    compact_tokens: int = 0
    # Large-file persistence
    is_persisted: bool = False
    persist_path: str = ""


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class ContextStats:
    request_turn: int
    model_messages: int = 0
    full_chars: int = 0
    model_chars: int = 0
    compacted_chars: int = 0
    full_tokens: int = 0
    model_tokens: int = 0
    compacted_tokens: int = 0
    compacted_tool_results: int = 0
    persisted_results: int = 0
    estimated_tokens: int = 0
    compaction_level: int = 0          # 0=none 1=large-persist 2=time 3=pressure
    recovery_injected: bool = False
    circuit_breaker_active: bool = False
    prompt_tokens: int = 0
    output_tokens: int = 0
    session_progress: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# AgentContext
# ---------------------------------------------------------------------------

class AgentContext:
    """Stores full messages and builds progressively compacted model views.

    Usage in the agent loop::

        ctx = AgentContext(user_prompt, policy, persist_dir=run_dir)
        for turn in range(max_turns):
            msgs, stats = ctx.build_model_messages(turn)
            response = provider.complete(...)
            ctx.update_token_usage(response.usage.prompt_tokens)
            # ... append tool results ...
    """

    def __init__(
        self,
        user_prompt: str,
        policy: ContextPolicy | None = None,
        *,
        persist_dir: str | Path | None = None,
    ):
        self.policy = policy or ContextPolicy()
        self._messages: list[_StoredMessage] = []
        self._anchor_tokens: int = 0
        self._anchor_msg_count: int = 0
        self._recently_read: list[dict] = []   # {path, repo, lines, turn}
        if persist_dir:
            self._persist_dir = Path(persist_dir)
        else:
            # Default: .sysight/tool-results/ in cwd (same namespace as memory/analysis-runs)
            self._persist_dir = Path.cwd() / ".sysight" / "tool-results"
        self._recovery_done: bool = False
        self._circuit_breaker_warning_done: bool = False  # separate from recovery

        # Circuit breaker state (Claude Code pattern)
        self._consecutive_pressure_events: int = 0
        self._circuit_breaker_tripped: bool = False

        # Session progress tracking (Codex "ghost history" prevention)
        self._session_files_read: list[str] = []      # ordered list of unique file paths
        self._session_tools_called: dict[str, int] = {}  # tool_name → count
        self._session_findings_count: int = 0

        self.append_message({"role": "user", "content": user_prompt}, turn=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append_message(self, message: dict, turn: int) -> None:
        self._messages.append(_StoredMessage(
            message=copy.deepcopy(message),
            turn=turn,
            kind="message",
            full_chars=_message_chars(message),
            full_tokens=_message_tokens(message),
        ))

    def append_tool_result(
        self,
        message: dict,
        *,
        turn: int,
        tool_name: str,
        arguments: dict,
        status: str,
        error: str,
        data: Any,
    ) -> None:
        full = copy.deepcopy(message)
        full_content = str(message.get("content", ""))

        # Track session progress
        self._session_tools_called[tool_name] = (
            self._session_tools_called.get(tool_name, 0) + 1
        )

        # Track recently-read files for recovery
        if tool_name == "scanner_read" and status == "ok":
            self._track_read_file(data, turn)

        # Build compact version (may persist to disk)
        compact_content, is_persisted, persist_path = build_tool_result_summary(
            tool_name=tool_name,
            arguments=arguments,
            status=status,
            error=error,
            data=data,
            full_content=full_content,
            persist_dir=self._persist_dir,
            large_threshold_tokens=self.policy.large_result_threshold_tokens,
            preview_tokens=self.policy.large_result_preview_tokens,
        )
        compact = copy.deepcopy(message)
        compact["content"] = compact_content

        full_chars_val = _message_chars(full)
        compact_chars_val = _message_chars(compact)
        # Don't store a compact version that isn't actually smaller.
        # The JSON wrapper overhead can exceed savings for small results.
        if compact_chars_val >= full_chars_val:
            compact = None
            compact_chars_val = 0

        self._messages.append(_StoredMessage(
            message=full,
            turn=turn,
            kind="tool",
            compact_message=compact,
            full_chars=full_chars_val,
            compact_chars=compact_chars_val,
            full_tokens=_message_tokens(full),
            compact_tokens=_message_tokens(compact) if compact else 0,
            is_persisted=is_persisted,
            persist_path=persist_path,
        ))

    def update_token_usage(self, prompt_tokens: int) -> None:
        """Feed back server-reported prompt_tokens as the estimation anchor."""
        self._anchor_tokens = prompt_tokens
        self._anchor_msg_count = len(self._messages)

    def build_model_messages(self, request_turn: int) -> tuple[list[dict], ContextStats]:
        """Build the model-facing message list with progressive compaction."""
        messages: list[dict] = []
        stats = ContextStats(request_turn=request_turn)
        stats.estimated_tokens = self._estimate_tokens()

        hard_limit = self.policy.effective_hard_limit()
        compact_limit = self.policy.effective_compact_limit()

        # Circuit breaker: if tripped, skip compaction entirely
        if self._circuit_breaker_tripped:
            use_pressure = False
            stats.circuit_breaker_active = True
        else:
            use_pressure = (
                hard_limit > 0
                and stats.estimated_tokens > hard_limit
            )

        # Time-based compaction (Level 2): only when context is filling up.
        # Codex "late compaction" strategy — no premature information loss.
        use_time_compact = (
            not use_pressure
            and compact_limit > 0
            and stats.estimated_tokens > compact_limit
        )

        # Update circuit breaker counter
        if use_pressure:
            self._consecutive_pressure_events += 1
            if self._consecutive_pressure_events >= self.policy.circuit_breaker_max:
                self._circuit_breaker_tripped = True
                stats.circuit_breaker_active = True
                use_pressure = False  # stop compacting
        else:
            # Reset on non-pressure turn
            self._consecutive_pressure_events = 0

        for stored in self._messages:
            full_chars = stored.full_chars
            full_tokens = stored.full_tokens
            stats.full_chars += full_chars
            stats.full_tokens += full_tokens

            use_compact = self._should_compact(
                stored, request_turn, use_pressure, use_time_compact,
            )
            selected = (
                stored.compact_message
                if use_compact and stored.compact_message
                else stored.message
            )
            selected = copy.deepcopy(selected)
            messages.append(selected)
            selected_chars = _message_chars(selected)
            selected_tokens = _message_tokens(selected)
            stats.model_chars += selected_chars
            stats.model_tokens += selected_tokens

            if use_compact:
                stats.compacted_tool_results += 1
                stats.compacted_chars += max(0, full_chars - selected_chars)
                stats.compacted_tokens += max(0, full_tokens - selected_tokens)
            if stored.is_persisted:
                stats.persisted_results += 1

        # Compaction level for observability
        if use_pressure:
            stats.compaction_level = 3
        elif stats.compacted_tool_results > 0:
            stats.compaction_level = 2
        elif stats.persisted_results > 0:
            stats.compaction_level = 1

        # Session progress (Codex "ghost history" prevention)
        stats.session_progress = self._build_session_progress()

        # Level 3 recovery: inject recently-read file contents + session progress
        if (
            use_pressure
            and self.policy.restore_recent_files
            and not self._recovery_done
        ):
            recovery = self._build_recovery_content()
            if recovery:
                messages.append({"role": "user", "content": recovery})
                stats.recovery_injected = True
                self._recovery_done = True

        # Circuit breaker warning: if tripped, add a notice to the model
        # (separate from recovery — both can coexist in the same turn)
        if self._circuit_breaker_tripped and not self._circuit_breaker_warning_done:
            messages.append({"role": "user", "content": _CIRCUIT_BREAKER_WARNING})
            self._circuit_breaker_warning_done = True  # only inject once

        stats.model_messages = len(messages)
        return messages, stats

    def full_log_messages(self) -> list[dict]:
        return [copy.deepcopy(item.message) for item in self._messages]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _track_read_file(self, data: Any, turn: int) -> None:
        # Tool results may be dataclass instances (e.g. ReadResult), not plain dicts.
        if dataclasses.is_dataclass(data):
            data = dataclasses.asdict(data)
        if not isinstance(data, dict):
            return
        path = data.get("path", "")
        if not path:
            return
        repo = data.get("repo", "")
        lines = data.get("lines") or []
        # Keep only the most recent read per path
        self._recently_read = [f for f in self._recently_read if f["path"] != path]
        self._recently_read.append({
            "path": path, "repo": repo, "lines": lines, "turn": turn,
        })
        # Trim
        excess = len(self._recently_read) - self.policy.restore_file_count
        if excess > 0:
            self._recently_read = self._recently_read[excess:]

        # Session progress: track unique file paths
        if path not in self._session_files_read:
            self._session_files_read.append(path)

    def _estimate_tokens(self) -> int:
        """Anchor + delta estimation."""
        if self._anchor_tokens == 0:
            total_chars = sum(s.full_chars for s in self._messages)
            return int(total_chars / _CHARS_PER_TOKEN)
        new_chars = sum(
            s.full_chars for s in self._messages[self._anchor_msg_count:]
        )
        return self._anchor_tokens + int(new_chars / _CHARS_PER_TOKEN)

    def _should_compact(
        self, stored: _StoredMessage, request_turn: int,
        pressure: bool, time_compact: bool,
    ) -> bool:
        if stored.kind != "tool" or stored.compact_message is None:
            return False
        if not self.policy.compact_after_first_exposure:
            return False
        # Don't compact if the compact version isn't actually smaller.
        # The JSON wrapper overhead can exceed savings for small results.
        if stored.compact_chars >= stored.full_chars:
            return False
        # Level 3 — pressure: compact everything except current turn
        if pressure:
            return (request_turn - stored.turn) > 0
        # Level 2 — time-based: when context is filling up (Codex strategy)
        if time_compact:
            if not self.policy.full_tool_result_once:
                return True
            return (request_turn - stored.turn) > self.policy.keep_recent_turns_full
        # Below compact_limit: still compact very old results to prevent
        # unbounded context growth in medium-length sessions.  Use a 3× larger
        # window so recent results stay full longer than under time_compact.
        if not self.policy.full_tool_result_once:
            return True
        return (request_turn - stored.turn) > self.policy.keep_recent_turns_full * 3

    def _build_session_progress(self) -> dict:
        """Build session progress summary for observability and recovery.

        This follows the Codex "ghost history" prevention pattern:
        even when individual tool results are compacted away, the model
        retains a high-level view of what has been explored.
        """
        return {
            "files_read": list(self._session_files_read),
            "tools_called": dict(self._session_tools_called),
            "total_tool_calls": sum(self._session_tools_called.values()),
        }

    def _build_recovery_content(self) -> str:
        """Build a recovery message with recently-read file contents + session progress.

        Combines:
        1. Session progress overview (files explored, tools used)
        2. Recently-read source code contents

        This follows Claude Code's post-compaction recovery pattern and
        Codex's "ghost history" prevention (verbatim context preservation).
        """
        parts: list[str] = []

        # Session progress overview (Codex "ghost history" prevention)
        progress = self._build_session_progress()
        parts.append("## 上下文恢复 — 调查进度概要")
        parts.append("")
        parts.append(f"已读取文件 ({len(progress['files_read'])}): "
                     ", ".join(progress['files_read'][:20]))
        tools_str = ", ".join(
            f"{k}({v})" for k, v in sorted(progress['tools_called'].items())
        )
        parts.append(f"工具调用: {tools_str}")
        parts.append(f"总调用次数: {progress['total_tool_calls']}")
        parts.append("")

        if self._recently_read:
            parts.append("### 关键文件内容")
            parts.append("")
            parts.append(
                "以下为之前读取的源码内容。请基于这些信息继续分析，"
                "无需重新读取相同文件。"
            )
            parts.append("")
            for entry in reversed(self._recently_read):
                path = entry["path"]
                repo = entry.get("repo", "")
                lines = entry.get("lines") or []
                header = f"#### {repo}/{path}" if repo else f"#### {path}"
                parts.append(header)
                parts.append("")
                # Token budget: convert to char budget internally
                char_budget = int(self.policy.restore_max_tokens_per_file * _CHARS_PER_TOKEN)
                used = 0
                shown = 0
                for line in lines:
                    if isinstance(line, dict):
                        text = f"{line.get('line', '')}: {line.get('text', '')}"
                    else:
                        text = str(line)
                    if used + len(text) > char_budget:
                        parts.append(
                            f"... (省略 {len(lines) - shown} 行，共 {len(lines)} 行)"
                        )
                        break
                    parts.append(text)
                    used += len(text) + 1
                    shown += 1
                parts.append("")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Circuit breaker warning (Claude Code pattern)
# ---------------------------------------------------------------------------

_CIRCUIT_BREAKER_WARNING = (
    "⚠️ 上下文压缩熔断器已触发：连续多次触发高压压缩，"
    "系统已停止进一步压缩以防止无限循环。"
    "请基于当前可用信息直接输出分析结果，不再进行额外的工具调用。"
)


# ---------------------------------------------------------------------------
# Tool-result summary builder
# ---------------------------------------------------------------------------

def build_tool_result_summary(
    *,
    tool_name: str,
    arguments: dict,
    status: str,
    error: str,
    data: Any,
    full_content: str,
    persist_dir: Path | None = None,
    large_threshold_tokens: int = 20_000,
    preview_tokens: int = 600,
) -> tuple[str, bool, str]:
    """Build a compact summary for a tool result.

    Returns (compact_content_json, is_persisted, persist_path).

    Thresholds are in tokens; internally converted to chars via _CHARS_PER_TOKEN.
    """
    # Level 1: persist oversized results to disk
    large_threshold_chars = int(large_threshold_tokens * _CHARS_PER_TOKEN)
    preview_chars = int(preview_tokens * _CHARS_PER_TOKEN)
    if persist_dir and len(full_content) > large_threshold_chars:
        persist_path = _persist_to_disk(full_content, persist_dir, tool_name)
        preview = full_content[:preview_chars]
        payload = {
            "sysight_persisted_tool_result": True,
            "tool": tool_name,
            "status": status,
            "persist_path": persist_path,
            "full_chars": len(full_content),
            "full_sha256": _sha256_text(full_content),
            "preview": preview,
            "note": (
                f"完整输出 ({len(full_content)} 字符) 已保存到 {persist_path}。"
                f"上方 preview 为前 {preview_chars} 字符。"
                "如需查看完整内容，使用 scanner_read 工具读取上述路径。"
            ),
        }
        return (
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            True,
            persist_path,
        )

    # Normal compact summary
    jsonable = to_jsonable(data)
    summary = _summarize_jsonable(tool_name, jsonable, full_content)
    payload = {
        "sysight_compacted_tool_result": True,
        "tool": tool_name,
        "status": status,
        "arguments_sha256": _sha256_json(arguments),
        "full_chars": len(full_content),
        "full_sha256": _sha256_text(full_content),
        "summary": summary,
        "error": error or None,
        "note": (
            "完整 tool result 已在历史记录中提供。上方 summary 包含关键数据，"
            "请直接基于已有信息生成 findings，无需重新调用工具。"
            "若确实需要不同的行范围，再使用 scanner_read 指定精确行号。"
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str), False, ""


def _persist_to_disk(content: str, base_dir: Path, tool_name: str) -> str:
    """Write large tool output to disk, return the file path string."""
    base_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".json" if content.strip().startswith("{") else ".txt"
    fd, path = tempfile.mkstemp(
        prefix=f"{tool_name}_", suffix=suffix, dir=str(base_dir),
    )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# Per-tool summarizers
# ---------------------------------------------------------------------------

def to_jsonable(value: Any) -> Any:
    """Convert tool results to JSON-compatible data without dataclass repr noise."""
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _summarize_jsonable(tool_name: str, data: Any, full_content: str) -> dict:
    if not isinstance(data, dict):
        return _generic_summary(full_content)

    if tool_name == "scanner_read":
        lines = data.get("lines") or []
        line_texts = _line_texts(lines)
        return {
            "repo": data.get("repo"),
            "path": data.get("path"),
            "total_lines": data.get("total_lines"),
            "shown_start": data.get("shown_start"),
            "shown_end": data.get("shown_end"),
            "line_count": len(lines),
            "content": line_texts,
            "note": "以上为已读取的源码内容。请直接使用，无需为相同行范围重新调用 scanner_read。",
        }

    if tool_name == "scanner_files":
        files = data.get("files") or []
        paths = [f.get("path") if isinstance(f, dict) else str(f) for f in files]
        return {
            "repo": data.get("repo"),
            "total": data.get("total"),
            "shown_paths": paths[:30],
            "omitted": max(0, len(paths) - 30),
        }

    if tool_name == "memory_search":
        matches = data.get("matches") or []
        return {
            "query": data.get("query"),
            "total": data.get("total"),
            "matches": [_shorten_mapping(m, max_string=300) for m in matches[:8]],
            "omitted": max(0, len(matches) - 8),
        }

    if tool_name == "memory_read":
        body = str(data.get("body") or "")
        return {
            "path": data.get("path"),
            "title": data.get("title"),
            "found": data.get("found"),
            "category": data.get("category"),
            "body_chars": len(body),
            "body_sha256": _sha256_text(body),
            "preview": _preview_lines(body.splitlines(), limit=10),
        }

    if tool_name.startswith("nsys_sql_"):
        return _summarize_nsys(data)

    return _generic_summary(full_content)


def _summarize_nsys(data: dict) -> dict:
    summary: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            summary[key] = _shorten_string(value)
        elif isinstance(value, list):
            summary[key] = [_shorten_mapping(v) for v in value[:8]]
            summary[f"{key}_omitted"] = max(0, len(value) - 8)
        elif isinstance(value, dict):
            summary[key] = _shorten_mapping(value)
    return summary


def _generic_summary(content: str) -> dict:
    return {
        "preview": _preview_lines(content.splitlines(), limit=12),
        "chars": len(content),
    }


def _line_texts(lines: list[Any]) -> list[str]:
    texts: list[str] = []
    for line in lines:
        if isinstance(line, dict):
            texts.append(f"{line.get('line')}: {line.get('text', '')}")
        else:
            texts.append(str(line))
    return texts


def _preview_lines(lines: list[str], limit: int) -> list[str]:
    return [str(line)[:500] for line in lines[:limit]]


def _shorten_mapping(value: Any, max_string: int = 180) -> Any:
    if isinstance(value, dict):
        return {k: _shorten_mapping(v, max_string=max_string) for k, v in value.items()}
    if isinstance(value, list):
        return [_shorten_mapping(v, max_string=max_string) for v in value[:8]]
    return _shorten_string(value, max_len=max_string)


def _shorten_string(value: Any, max_len: int = 180) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) <= max_len:
        return value
    digest = _sha256_text(value)[:12]
    return f"{value[:max_len]}... [sha256:{digest}, {len(value)} chars]"


def _message_chars(message: dict) -> int:
    return len(json.dumps(message, ensure_ascii=False, default=str))


def _message_tokens(message: dict) -> int:
    """Estimate token count for a message dict (chars / _CHARS_PER_TOKEN)."""
    return int(_message_chars(message) / _CHARS_PER_TOKEN)


def _sha256_json(value: Any) -> str:
    return _sha256_text(json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True, default=str))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
