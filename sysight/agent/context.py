"""Context management for multi-turn tool-calling agents.

Progressive compaction pipeline (inspired by MiniCode / Claude Code / Codex):

  Level 0 — Microcompact  (MiniCode microcompact pattern)
    At ≥50% utilization: clear old COMPACTABLE tool results (file lists, SQL
    summaries, etc.) to a CLEAR_MARKER placeholder.  Only the most recent
    KEEP_RECENT_TOOL_RESULTS results are retained.  Zero information loss for
    non-compactable tools.

  Level 1 — Large-Result Persistence  (zero model cost)
    Tool results >20K tokens → persist full output to disk, keep short preview.
    Model can re-read via scanner_read if needed.

  Level 2 — Time-Based Compaction  (Codex "late compaction" strategy)
    At ≥70% utilization: replace old tool results with template-generated
    summaries.  Results younger than keep_recent_turns_full are kept full.

  Level 2.5 — Snip  (MiniCode snipCompact pattern — deterministic, no LLM)
    At ≥80% utilization: physically remove a contiguous "safe" middle interval
    of messages.  Protected: system messages, recent N messages, and tool
    calls that involve write/edit operations.  A snip_boundary marker is
    inserted in place of the removed block.

  Level 3 — Token-Pressure Compaction
    At ≥95% utilization: aggressively compact ALL old tool results + inject a
    recovery message (recently-read file contents + session progress).

  Circuit Breaker  (MiniCode / Claude Code pattern)
    If Level 3 triggers consecutively ≥ N times *without the token count
    dropping below the compact threshold* (i.e. compaction is not helping),
    further compaction is blocked and a warning is injected.  The counter
    resets only when estimated_tokens < compact_limit * RESET_RATIO.

Token estimation
    Anchor+delta method: server-reported prompt_tokens from the last API
    response serves as the anchor; delta is computed from the COMPACTED chars
    (i.e. chars of the messages actually sent to the model, not full_chars) of
    messages added since the anchor.  Typical error <5%.

Model-aware thresholds  (DeepSeek-TUI pattern)
    All token limits auto-scale to the model's context window.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import tempfile
import uuid
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
    # OpenAI
    "gpt-5.5": 1_000_000,
    "gpt-5.4": 1_000_000,
    "gpt-5.2": 400_000,
    "gpt-5.1": 400_000,
    "gpt-5": 400_000,
    "o4": 200_000,
    "o4-mini": 200_000,
    # Anthropic
    "claude-opus-4-7": 200_000,
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    # DeepSeek
    "deepseek-v4-pro": 1_000_000,
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4": 1_000_000,
    "deepseek-v3.2": 128_000,
    "deepseek-r1": 128_000,
    # Google
    "gemini-3.1-pro": 1_048_576,
    "gemini-3-flash": 1_048_576,
    "gemini-3-pro": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
}

_DEFAULT_CONTEXT_WINDOW = 1_000_000

# Utilization thresholds (fraction of context window)
_MICROCOMPACT_UTILIZATION = 0.50   # Level 0: clear compactable tool results
_COMPACT_THRESHOLD_RATIO  = 0.70   # Level 2: time-based compaction
_SNIP_THRESHOLD_RATIO     = 0.80   # Level 2.5: deterministic snip
_HARD_LIMIT_RATIO         = 0.95   # Level 3: aggressive pressure

# Snip parameters (MiniCode snipCompact)
_SNIP_TARGET_USAGE           = 0.60   # target utilization after snip
_SNIP_KEEP_RECENT_MESSAGES   = 12     # never snip the most recent N messages
_SNIP_MIN_MESSAGES_TO_REMOVE = 4      # don't bother if we'd remove < 4 messages
_SNIP_MIN_TOKENS_TO_FREE     = 2_000  # don't bother if savings < 2K tokens

# Retention for microcompact
_KEEP_RECENT_TOOL_RESULTS = 3   # always keep the last N compactable tool results full

# Circuit breaker — reset only when well below compact threshold
_CIRCUIT_BREAKER_MAX  = 3
_CIRCUIT_BREAKER_RESET_RATIO = 0.80  # reset counter only when util < 80% of compact limit

# Tools whose results can be cleared (content-insensitive for the model)
_COMPACTABLE_TOOLS: frozenset[str] = frozenset({
    "scanner_files",
    "nsys_sql_list_tables",
    "nsys_sql_schema",
})

# Special marker replacing cleared tool results (MiniCode CLEAR_MARKER pattern)
_CLEAR_MARKER = "[tool result cleared — microcompact]"

# Tool names that involve writes/edits — their rounds are protected from Snip
_PROTECTED_TOOL_NAMES: frozenset[str] = frozenset({
    "scanner_write",
    "scanner_patch",
    "scanner_edit",
})


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass
class ContextPolicy:
    """Controls how the model-facing context is built.

    All token limits auto-scale from model_name when set to 0.
    """

    model_name: str = ""

    # Token budget (0 → auto-compute)
    compact_token_limit: int = 0    # Level 2 threshold
    snip_token_limit: int = 0       # Level 2.5 threshold
    hard_token_limit: int = 0       # Level 3 threshold

    # Time-based compaction (Level 2)
    full_tool_result_once: bool = True
    compact_after_first_exposure: bool = True
    keep_recent_turns_full: int = 3

    # Large-file persistence (Level 1)
    large_result_threshold_tokens: int = 20_000
    large_result_preview_tokens: int = 600

    # Recovery after Level 3
    restore_recent_files: bool = True
    restore_file_count: int = 5
    restore_max_tokens_per_file: int = 5_000

    # Circuit breaker
    circuit_breaker_max: int = _CIRCUIT_BREAKER_MAX

    def effective_compact_limit(self) -> int:
        if self.compact_token_limit > 0:
            return self.compact_token_limit
        return int(self._context_window() * _COMPACT_THRESHOLD_RATIO)

    def effective_snip_limit(self) -> int:
        if self.snip_token_limit > 0:
            return self.snip_token_limit
        return int(self._context_window() * _SNIP_THRESHOLD_RATIO)

    def effective_hard_limit(self) -> int:
        if self.hard_token_limit > 0:
            return self.hard_token_limit
        return int(self._context_window() * _HARD_LIMIT_RATIO)

    def effective_microcompact_limit(self) -> int:
        return int(self._context_window() * _MICROCOMPACT_UTILIZATION)

    def _context_window(self) -> int:
        if self.model_name in _MODEL_CONTEXT_WINDOWS:
            return _MODEL_CONTEXT_WINDOWS[self.model_name]
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
    kind: str = "message"               # "message" | "tool" | "snip_boundary"
    tool_name: str = ""                  # populated for kind=="tool"
    compact_message: dict | None = None
    full_chars: int = 0
    compact_chars: int = 0
    full_tokens: int = 0
    compact_tokens: int = 0
    # Large-file persistence
    is_persisted: bool = False
    persist_path: str = ""
    # Microcompact: content already cleared
    is_cleared: bool = False
    # Snip: this slot is a boundary marker (replaces deleted messages)
    is_snip_boundary: bool = False


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
    cleared_tool_results: int = 0       # microcompact
    persisted_results: int = 0
    snipped_messages: int = 0           # Level 2.5
    estimated_tokens: int = 0
    compaction_level: int = 0           # 0=none 1=persist 2=time 2.5→reported as 2 3=pressure
    snip_applied: bool = False
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

    Usage::

        ctx = AgentContext(user_prompt, policy, persist_dir=run_dir)
        for turn in range(max_turns):
            msgs, stats = ctx.build_model_messages(turn)
            response = provider.complete(...)
            ctx.update_token_usage(response.usage.prompt_tokens)
            ctx.append_message({"role": "assistant", ...}, turn=turn)
            ctx.append_tool_result(result_msg, turn=turn, tool_name=..., ...)
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

        # Token anchor (server-reported, used for delta estimation)
        self._anchor_tokens: int = 0
        # Chars of the message list *as sent to the model* at anchor time
        self._anchor_sent_chars: int = 0

        self._recently_read: list[dict] = []
        if persist_dir:
            self._persist_dir = Path(persist_dir)
        else:
            self._persist_dir = Path.cwd() / ".sysight" / "tool-results"

        self._recovery_done: bool = False
        self._circuit_breaker_warning_done: bool = False

        # Circuit breaker (MiniCode / Claude Code pattern)
        # Counter increments only when pressure fires AND tokens did NOT drop
        # below the compact_limit * RESET_RATIO between turns.
        self._consecutive_pressure_events: int = 0
        self._circuit_breaker_tripped: bool = False

        # Session progress tracking (Codex "ghost history" prevention)
        self._session_files_read: list[str] = []
        self._session_tools_called: dict[str, int] = {}
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
        if compact_chars_val >= full_chars_val:
            compact = None
            compact_chars_val = 0

        self._messages.append(_StoredMessage(
            message=full,
            turn=turn,
            kind="tool",
            tool_name=tool_name,
            compact_message=compact,
            full_chars=full_chars_val,
            compact_chars=compact_chars_val,
            full_tokens=_message_tokens(full),
            compact_tokens=_message_tokens(compact) if compact else 0,
            is_persisted=is_persisted,
            persist_path=persist_path,
        ))

    def update_token_usage(self, prompt_tokens: int) -> None:
        """Feed back server-reported prompt_tokens as the estimation anchor.

        We capture the compacted-chars of the current message list so that
        future delta estimation uses the *sent* size, not the full size.
        """
        self._anchor_tokens = prompt_tokens
        # Compute the chars that were actually sent at this anchor point
        # (same logic as _sent_chars_for_messages but against all stored messages)
        self._anchor_sent_chars = sum(
            self._effective_chars(s) for s in self._messages
        )

    def build_model_messages(self, request_turn: int) -> tuple[list[dict], ContextStats]:
        """Build the model-facing message list with progressive compaction."""
        stats = ContextStats(request_turn=request_turn)
        stats.estimated_tokens = self._estimate_tokens()

        hard_limit    = self.policy.effective_hard_limit()
        snip_limit    = self.policy.effective_snip_limit()
        compact_limit = self.policy.effective_compact_limit()
        micro_limit   = self.policy.effective_microcompact_limit()

        # ---- Level 0: Microcompact ----------------------------------------
        # Apply in-place on _messages before building the view (MiniCode pattern).
        # Only runs when approaching the compact zone.
        if stats.estimated_tokens > micro_limit:
            self._apply_microcompact()

        # ---- Determine compaction level ------------------------------------
        if self._circuit_breaker_tripped:
            use_pressure = False
            stats.circuit_breaker_active = True
        else:
            use_pressure = hard_limit > 0 and stats.estimated_tokens > hard_limit

        use_snip = (
            not use_pressure
            and snip_limit > 0
            and stats.estimated_tokens > snip_limit
        )

        use_time_compact = (
            not use_pressure
            and compact_limit > 0
            and stats.estimated_tokens > compact_limit
        )

        # ---- Circuit breaker counter (MiniCode pattern) -------------------
        # Increment only when genuinely in pressure; reset only when well clear.
        if use_pressure:
            self._consecutive_pressure_events += 1
            if self._consecutive_pressure_events >= self.policy.circuit_breaker_max:
                self._circuit_breaker_tripped = True
                stats.circuit_breaker_active = True
                use_pressure = False
        elif stats.estimated_tokens < compact_limit * _CIRCUIT_BREAKER_RESET_RATIO:
            # Only reset when clearly out of pressure zone
            self._consecutive_pressure_events = 0

        # ---- Level 2.5: Snip (deterministic, no LLM) ----------------------
        if use_snip:
            snipped = self._apply_snip(request_turn)
            if snipped:
                stats.snip_applied = True
                stats.snipped_messages = snipped

        # ---- Build model message list -------------------------------------
        messages: list[dict] = []
        for stored in self._messages:
            stats.full_chars  += stored.full_chars
            stats.full_tokens += stored.full_tokens

            if stored.is_snip_boundary:
                selected = copy.deepcopy(stored.message)
                messages.append(selected)
                stats.model_chars  += _message_chars(selected)
                stats.model_tokens += _message_tokens(selected)
                continue

            if stored.is_cleared:
                selected = copy.deepcopy(stored.message)
                messages.append(selected)
                stats.model_chars  += _message_chars(selected)
                stats.model_tokens += _message_tokens(selected)
                stats.cleared_tool_results += 1
                continue

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
            selected_chars  = _message_chars(selected)
            selected_tokens = _message_tokens(selected)
            stats.model_chars  += selected_chars
            stats.model_tokens += selected_tokens

            if use_compact:
                stats.compacted_tool_results += 1
                stats.compacted_chars  += max(0, stored.full_chars  - selected_chars)
                stats.compacted_tokens += max(0, stored.full_tokens - selected_tokens)
            if stored.is_persisted:
                stats.persisted_results += 1

        # Compaction level for observability
        if use_pressure:
            stats.compaction_level = 3
        elif stats.snip_applied:
            stats.compaction_level = 3   # snip is treated as aggressive
        elif stats.compacted_tool_results > 0:
            stats.compaction_level = 2
        elif stats.cleared_tool_results > 0:
            stats.compaction_level = 1
        elif stats.persisted_results > 0:
            stats.compaction_level = 1

        # Session progress
        stats.session_progress = self._build_session_progress()

        # Level 3 recovery injection
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

        # Circuit breaker warning
        if self._circuit_breaker_tripped and not self._circuit_breaker_warning_done:
            messages.append({"role": "user", "content": _CIRCUIT_BREAKER_WARNING})
            self._circuit_breaker_warning_done = True

        stats.model_messages = len(messages)
        return messages, stats

    def full_log_messages(self) -> list[dict]:
        return [copy.deepcopy(item.message) for item in self._messages]

    # ------------------------------------------------------------------
    # Microcompact (Level 0)  — MiniCode microcompact pattern
    # ------------------------------------------------------------------

    def _apply_microcompact(self) -> None:
        """Clear old compactable tool results in-place (no LLM needed).

        Mirrors MiniCode's microcompact.ts:
        - Collect indices of all COMPACTABLE tool results.
        - Keep the most recent KEEP_RECENT_TOOL_RESULTS; clear the rest.
        - Clearing = replace message content with _CLEAR_MARKER and set is_cleared.
        """
        compactable_indices = [
            i for i, s in enumerate(self._messages)
            if s.kind == "tool"
            and s.tool_name in _COMPACTABLE_TOOLS
            and not s.is_cleared
        ]
        if len(compactable_indices) <= _KEEP_RECENT_TOOL_RESULTS:
            return

        keep_from = len(compactable_indices) - _KEEP_RECENT_TOOL_RESULTS
        to_clear = set(compactable_indices[:keep_from])

        for i in to_clear:
            stored = self._messages[i]
            cleared_msg = copy.deepcopy(stored.message)
            cleared_msg["content"] = _CLEAR_MARKER
            stored.message = cleared_msg
            # Also clear compact version so we don't accidentally send it
            stored.compact_message = None
            stored.compact_chars = 0
            stored.compact_tokens = 0
            stored.is_cleared = True

    # ------------------------------------------------------------------
    # Snip (Level 2.5)  — MiniCode snipCompact pattern
    # ------------------------------------------------------------------

    def _apply_snip(self, request_turn: int) -> int:
        """Physically remove a contiguous safe middle interval.

        Returns the number of messages removed (0 if nothing was snipped).

        Protection rules (mirrors MiniCode's markProtectedGroups):
        - system messages and snip_boundary markers are protected
        - The first user message (index 0) is always protected
        - The most recent _SNIP_KEEP_RECENT_MESSAGES messages are protected
        - Any tool call group that involves a write/edit tool is protected

        The "safe interval" is the longest contiguous unprotected span.
        We remove messages from this interval until we've freed enough tokens.
        """
        msgs = self._messages
        n = len(msgs)
        if n < _SNIP_MIN_MESSAGES_TO_REMOVE + _SNIP_KEEP_RECENT_MESSAGES:
            return 0

        hard_limit  = self.policy.effective_hard_limit()
        snip_limit  = self.policy.effective_snip_limit()
        window      = self.policy._context_window()
        target_tokens = int(window * _SNIP_TARGET_USAGE)

        current_tokens = self._estimate_tokens()
        desired_to_free = max(_SNIP_MIN_TOKENS_TO_FREE, current_tokens - target_tokens)

        # Build protected mask
        protected = [False] * n
        # Always protect the first message
        protected[0] = True
        # Protect recent tail
        recent_start = max(1, n - _SNIP_KEEP_RECENT_MESSAGES)
        for i in range(recent_start, n):
            protected[i] = True
        # Protect system / boundary messages
        for i, s in enumerate(msgs):
            if s.kind in ("message",) and msgs[i].message.get("role") == "system":
                protected[i] = True
            if s.is_snip_boundary:
                protected[i] = True
        # Protect write/edit tool groups: find assistant messages containing
        # tool calls to protected tools, and protect the entire surrounding round.
        for i, s in enumerate(msgs):
            if s.kind == "tool" and s.tool_name in _PROTECTED_TOOL_NAMES:
                # Protect this result and scan back to find the assistant call
                protected[i] = True
                for j in range(i - 1, max(0, i - 5), -1):
                    protected[j] = True
                    if msgs[j].kind == "message" and msgs[j].message.get("role") == "assistant":
                        break

        # Find the best contiguous unprotected interval (longest by token count)
        best_start, best_end, best_tokens = -1, -1, 0
        cur_start = -1
        cur_tokens = 0
        for i in range(1, n):  # skip index 0 (always protected)
            if not protected[i]:
                if cur_start == -1:
                    cur_start = i
                cur_tokens += msgs[i].full_tokens
            else:
                if cur_start != -1 and cur_tokens > best_tokens:
                    best_start, best_end, best_tokens = cur_start, i, cur_tokens
                cur_start = -1
                cur_tokens = 0
        if cur_start != -1 and cur_tokens > best_tokens:
            best_start, best_end, best_tokens = cur_start, n, cur_tokens

        if best_start == -1:
            return 0

        # Trim the interval to free the desired token count
        # Remove from the beginning of the interval (oldest messages first)
        freed = 0
        end_of_deletion = best_start
        for i in range(best_start, best_end):
            freed += msgs[i].full_tokens
            end_of_deletion = i + 1
            if freed >= desired_to_free:
                break

        removal_count = end_of_deletion - best_start
        if removal_count < _SNIP_MIN_MESSAGES_TO_REMOVE:
            return 0
        if freed < _SNIP_MIN_TOKENS_TO_FREE:
            return 0

        # Build boundary marker message
        boundary_content = (
            f"[snip_boundary: {removal_count} messages removed, "
            f"~{freed} tokens freed. "
            f"Context before this point has been truncated to reduce context size.]"
        )
        boundary_msg = {"role": "user", "content": boundary_content}
        boundary_stored = _StoredMessage(
            message=boundary_msg,
            turn=request_turn,
            kind="snip_boundary",
            is_snip_boundary=True,
            full_chars=_message_chars(boundary_msg),
            full_tokens=_message_tokens(boundary_msg),
        )

        # Replace the deleted slice with the boundary marker
        self._messages = (
            msgs[:best_start]
            + [boundary_stored]
            + msgs[end_of_deletion:]
        )

        # Invalidate anchor so next estimation uses full recalc
        self._anchor_tokens = 0
        self._anchor_sent_chars = 0

        return removal_count

    # ------------------------------------------------------------------
    # Token estimation helpers
    # ------------------------------------------------------------------

    def _effective_chars(self, stored: _StoredMessage) -> int:
        """Chars of this message as it would be sent (compact if available)."""
        if stored.is_cleared:
            return len(json.dumps(stored.message, ensure_ascii=False, default=str))
        if stored.compact_message and stored.compact_chars > 0:
            return stored.compact_chars
        return stored.full_chars

    def _estimate_tokens(self) -> int:
        """Anchor + delta estimation using *sent* chars (not full chars).

        Fixes the original bug where delta was computed from full_chars even
        when compact versions were being sent.
        """
        if self._anchor_tokens == 0:
            total_chars = sum(self._effective_chars(s) for s in self._messages)
            return int(total_chars / _CHARS_PER_TOKEN)
        # Delta: chars added since the anchor snapshot
        current_sent_chars = sum(self._effective_chars(s) for s in self._messages)
        delta_chars = max(0, current_sent_chars - self._anchor_sent_chars)
        return self._anchor_tokens + int(delta_chars / _CHARS_PER_TOKEN)

    # ------------------------------------------------------------------
    # Compaction decision
    # ------------------------------------------------------------------

    def _should_compact(
        self, stored: _StoredMessage, request_turn: int,
        pressure: bool, time_compact: bool,
    ) -> bool:
        if stored.kind != "tool" or stored.compact_message is None:
            return False
        if not self.policy.compact_after_first_exposure:
            return False
        if stored.compact_chars >= stored.full_chars:
            return False
        if pressure:
            return (request_turn - stored.turn) > 0
        if time_compact:
            if not self.policy.full_tool_result_once:
                return True
            return (request_turn - stored.turn) > self.policy.keep_recent_turns_full
        return False  # below compact_limit → no time-based compaction

    # ------------------------------------------------------------------
    # Session tracking helpers
    # ------------------------------------------------------------------

    def _track_read_file(self, data: Any, turn: int) -> None:
        if dataclasses.is_dataclass(data):
            data = dataclasses.asdict(data)
        if not isinstance(data, dict):
            return
        path = data.get("path", "")
        if not path:
            return
        repo  = data.get("repo", "")
        lines = data.get("lines") or []
        self._recently_read = [f for f in self._recently_read if f["path"] != path]
        self._recently_read.append({"path": path, "repo": repo, "lines": lines, "turn": turn})
        excess = len(self._recently_read) - self.policy.restore_file_count
        if excess > 0:
            self._recently_read = self._recently_read[excess:]
        if path not in self._session_files_read:
            self._session_files_read.append(path)

    def _build_session_progress(self) -> dict:
        return {
            "files_read": list(self._session_files_read),
            "tools_called": dict(self._session_tools_called),
            "total_tool_calls": sum(self._session_tools_called.values()),
        }

    def _build_recovery_content(self) -> str:
        parts: list[str] = []
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
                path  = entry["path"]
                repo  = entry.get("repo", "")
                lines = entry.get("lines") or []
                header = f"#### {repo}/{path}" if repo else f"#### {path}"
                parts.append(header)
                parts.append("")
                char_budget = int(self.policy.restore_max_tokens_per_file * _CHARS_PER_TOKEN)
                used = 0
                shown = 0
                for line in lines:
                    text = (
                        f"{line.get('line', '')}: {line.get('text', '')}"
                        if isinstance(line, dict) else str(line)
                    )
                    if used + len(text) > char_budget:
                        parts.append(f"... (省略 {len(lines) - shown} 行，共 {len(lines)} 行)")
                        break
                    parts.append(text)
                    used += len(text) + 1
                    shown += 1
                parts.append("")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Circuit breaker warning
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
    """
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
    return int(_message_chars(message) / _CHARS_PER_TOKEN)


def _sha256_json(value: Any) -> str:
    return _sha256_text(
        json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True, default=str)
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
