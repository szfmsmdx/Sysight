from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List, Optional

from server.config import settings


class LogStore:
    def __init__(self) -> None:
        self._subscribers = {}  # type: Dict[str, List[asyncio.Queue]]

    def _log_path(self, task_id: str) -> Path:
        return settings.log_root / f"{task_id}.log"

    async def append(self, task_id: str, line: str) -> None:
        path = self._log_path(task_id)
        with open(path, "a") as f:
            f.write(line)
        for q in self._subscribers.get(task_id, []):
            q.put_nowait(line)

    async def read(self, task_id: str, offset: int = 0) -> tuple:
        """Read log from byte offset. Returns (data, new_offset)."""
        path = self._log_path(task_id)
        if not path.exists():
            return "", 0
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
        return data.decode(errors="replace"), offset + len(data)

    def subscribe(self, task_id: str) -> asyncio.Queue:
        q = asyncio.Queue()  # type: asyncio.Queue
        self._subscribers.setdefault(task_id, []).append(q)
        return q

    def unsubscribe(self, task_id: str, q: asyncio.Queue) -> None:
        subs = self._subscribers.get(task_id, [])
        if q in subs:
            subs.remove(q)
        if not subs:
            self._subscribers.pop(task_id, None)

    async def close_task(self, task_id: str) -> None:
        for q in self._subscribers.get(task_id, []):
            q.put_nowait(None)


log_store = LogStore()
