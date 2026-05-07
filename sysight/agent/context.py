"""Context management for multi-turn tool-calling agents.

The agent keeps a full debug log, but sends a compact model view after a
tool result has already been shown once. This avoids replaying large tool
outputs on every turn while preserving valid tool-call/tool-result pairs.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ContextPolicy:
    """Controls how the model-facing context is built."""

    full_tool_result_once: bool = True
    compact_after_first_exposure: bool = True
    keep_recent_turns_full: int = 1


@dataclass
class _StoredMessage:
    message: dict
    turn: int
    kind: str = "message"
    compact_message: dict | None = None
    full_chars: int = 0
    compact_chars: int = 0


@dataclass
class ContextStats:
    request_turn: int
    model_messages: int = 0
    full_chars: int = 0
    model_chars: int = 0
    compacted_chars: int = 0
    compacted_tool_results: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class AgentContext:
    """Stores full messages and builds compact messages for model requests."""

    def __init__(self, user_prompt: str, policy: ContextPolicy | None = None):
        self.policy = policy or ContextPolicy()
        self._messages: list[_StoredMessage] = []
        self.append_message({"role": "user", "content": user_prompt}, turn=0)

    def append_message(self, message: dict, turn: int) -> None:
        self._messages.append(_StoredMessage(
            message=copy.deepcopy(message),
            turn=turn,
            kind="message",
            full_chars=_message_chars(message),
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
        compact_content = build_tool_result_summary(
            tool_name=tool_name,
            arguments=arguments,
            status=status,
            error=error,
            data=data,
            full_content=str(message.get("content", "")),
        )
        compact = copy.deepcopy(message)
        compact["content"] = compact_content
        self._messages.append(_StoredMessage(
            message=full,
            turn=turn,
            kind="tool",
            compact_message=compact,
            full_chars=_message_chars(full),
            compact_chars=_message_chars(compact),
        ))

    def build_model_messages(self, request_turn: int) -> tuple[list[dict], ContextStats]:
        messages: list[dict] = []
        stats = ContextStats(request_turn=request_turn)
        for stored in self._messages:
            full_chars = stored.full_chars
            stats.full_chars += full_chars
            use_compact = self._should_compact(stored, request_turn)
            selected = stored.compact_message if use_compact and stored.compact_message else stored.message
            selected = copy.deepcopy(selected)
            messages.append(selected)
            selected_chars = _message_chars(selected)
            stats.model_chars += selected_chars
            if use_compact:
                stats.compacted_tool_results += 1
                stats.compacted_chars += max(0, full_chars - selected_chars)
        stats.model_messages = len(messages)
        return messages, stats

    def full_log_messages(self) -> list[dict]:
        return [copy.deepcopy(item.message) for item in self._messages]

    def _should_compact(self, stored: _StoredMessage, request_turn: int) -> bool:
        if stored.kind != "tool" or stored.compact_message is None:
            return False
        if not self.policy.compact_after_first_exposure:
            return False
        if not self.policy.full_tool_result_once:
            return True
        return (request_turn - stored.turn) > self.policy.keep_recent_turns_full


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


def build_tool_result_summary(
    *,
    tool_name: str,
    arguments: dict,
    status: str,
    error: str,
    data: Any,
    full_content: str,
) -> str:
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
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


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


def _sha256_json(value: Any) -> str:
    return _sha256_text(json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True, default=str))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
