from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

from server.config import settings
from server.db import get_db
from server.models import TaskStatus
from server.services.log_store import log_store


class TaskManager:
    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_tasks)
        self._processes = {}  # type: Dict[str, asyncio.subprocess.Process]

    async def submit(
        self,
        command: str,
        conda_env: Optional[str] = None,
        working_dir: Optional[str] = None,
    ) -> str:
        task_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        db = await get_db()
        await db.execute(
            "INSERT INTO tasks (id, command, conda_env, status, created_at, working_dir) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (task_id, command, conda_env, TaskStatus.pending.value, now, working_dir),
        )
        await db.commit()
        asyncio.create_task(self._run(task_id, command, conda_env, working_dir))
        return task_id

    async def _run(
        self,
        task_id: str,
        command: str,
        conda_env: Optional[str] = None,
        working_dir: Optional[str] = None,
    ) -> None:
        await self._semaphore.acquire()
        try:
            db = await get_db()
            now = datetime.now(timezone.utc).isoformat()
            await db.execute(
                "UPDATE tasks SET status=?, started_at=? WHERE id=?",
                (TaskStatus.running.value, now, task_id),
            )
            await db.commit()

            if conda_env:
                cmd = f"conda run -n {conda_env} --no-capture-output {command}"
            else:
                cmd = command

            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=working_dir,
            )
            self._processes[task_id] = proc

            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                await log_store.append(task_id, line.decode(errors="replace"))

            exit_code = await proc.wait()
            status = TaskStatus.success if exit_code == 0 else TaskStatus.failed
            finished = datetime.now(timezone.utc).isoformat()
            await db.execute(
                "UPDATE tasks SET status=?, exit_code=?, finished_at=? WHERE id=?",
                (status.value, exit_code, finished, task_id),
            )
            await db.commit()
        except asyncio.CancelledError:
            db = await get_db()
            await db.execute(
                "UPDATE tasks SET status=?, finished_at=? WHERE id=?",
                (TaskStatus.cancelled.value, datetime.now(timezone.utc).isoformat(), task_id),
            )
            await db.commit()
        finally:
            self._processes.pop(task_id, None)
            await log_store.close_task(task_id)
            self._semaphore.release()

    async def cancel(self, task_id: str) -> bool:
        proc = self._processes.get(task_id)
        if proc is None:
            return False
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
        db = await get_db()
        await db.execute(
            "UPDATE tasks SET status=?, finished_at=? WHERE id=?",
            (TaskStatus.cancelled.value, datetime.now(timezone.utc).isoformat(), task_id),
        )
        await db.commit()
        await log_store.close_task(task_id)
        return True


task_manager = TaskManager()
