from __future__ import annotations

import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from server.auth import verify_api_key
from server.config import settings
from server.db import get_db
from server.models import (
    LogChunk,
    TaskCreate,
    TaskInfo,
    TaskListResponse,
    TaskStatus,
)
from server.services.log_store import log_store
from server.services.task_manager import task_manager

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("", response_model=TaskInfo, dependencies=[Depends(verify_api_key)])
async def create_task(task: TaskCreate) -> TaskInfo:
    working_dir = task.working_dir
    if task.upload_id:
        upload_path = settings.workspace_root / f"{task.upload_id}.tar.gz"
        if upload_path.exists():
            extract_dir = settings.workspace_root / task.upload_id
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(upload_path, "r:gz") as tar:
                tar.extractall(extract_dir)
            working_dir = str(extract_dir)
            upload_path.unlink()

    task_id = await task_manager.submit(task.command, task.conda_env, working_dir)
    db = await get_db()
    row = await db.execute("SELECT * FROM tasks WHERE id=?", (task_id,))
    data = await row.fetchone()
    return _row_to_task(data)


@router.get("", response_model=TaskListResponse, dependencies=[Depends(verify_api_key)])
async def list_tasks(limit: int = 50, offset: int = 0) -> TaskListResponse:
    db = await get_db()
    rows = await db.execute(
        "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    )
    tasks = [_row_to_task(row) async for row in rows]
    count_row = await db.execute("SELECT COUNT(*) FROM tasks")
    total = (await count_row.fetchone())[0]
    return TaskListResponse(tasks=tasks, total=total)


@router.get("/{task_id}", response_model=TaskInfo, dependencies=[Depends(verify_api_key)])
async def get_task(task_id: str) -> TaskInfo:
    db = await get_db()
    row = await db.execute("SELECT * FROM tasks WHERE id=?", (task_id,))
    data = await row.fetchone()
    if not data:
        raise HTTPException(404, "Task not found")
    return _row_to_task(data)


@router.delete("/{task_id}", dependencies=[Depends(verify_api_key)])
async def cancel_task(task_id: str) -> Dict[str, str]:
    success = await task_manager.cancel(task_id)
    return {"status": "cancelled" if success else "not_running"}


@router.get("/{task_id}/logs", response_model=LogChunk, dependencies=[Depends(verify_api_key)])
async def get_logs(task_id: str, offset: int = 0) -> LogChunk:
    data, new_offset = await log_store.read(task_id, offset)
    return LogChunk(data=data, offset=new_offset)


@router.websocket("/{task_id}/logs/ws")
async def logs_websocket(websocket: WebSocket, task_id: str):
    await websocket.accept()
    try:
        api_key = websocket.headers.get("x-api-key")
        if api_key != settings.api_key:
            await websocket.close(code=1008, reason="Invalid API key")
            return

        existing, _ = await log_store.read(task_id, 0)
        if existing:
            await websocket.send_text(existing)

        q = log_store.subscribe(task_id)
        try:
            while True:
                line = await q.get()
                if line is None:
                    break
                await websocket.send_text(line)
        finally:
            log_store.unsubscribe(task_id, q)
    except WebSocketDisconnect:
        pass


def _row_to_task(row) -> TaskInfo:
    return TaskInfo(
        id=row["id"],
        command=row["command"],
        conda_env=row["conda_env"],
        status=TaskStatus(row["status"]),
        exit_code=row["exit_code"],
        created_at=datetime.fromisoformat(row["created_at"]),
        started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
        working_dir=row["working_dir"],
    )
