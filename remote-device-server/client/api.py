from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from client.config import API_KEY, SERVER_URL


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY}


def _url(path: str) -> str:
    return f"{SERVER_URL}{path}"


# ── Sync helpers (for simple CLI commands) ──


def health() -> dict[str, Any]:
    r = httpx.get(_url("/health"), headers=_headers(), timeout=5)
    r.raise_for_status()
    return r.json()


def create_task(
    command: str,
    conda_env: str | None = None,
    working_dir: str | None = None,
    upload_id: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"command": command}
    if conda_env:
        body["conda_env"] = conda_env
    if working_dir:
        body["working_dir"] = working_dir
    if upload_id:
        body["upload_id"] = upload_id
    r = httpx.post(_url("/tasks"), json=body, headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def list_tasks(limit: int = 50, offset: int = 0) -> dict[str, Any]:
    r = httpx.get(_url("/tasks"), params={"limit": limit, "offset": offset}, headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def get_task(task_id: str) -> dict[str, Any]:
    r = httpx.get(_url(f"/tasks/{task_id}"), headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def cancel_task(task_id: str) -> dict[str, Any]:
    r = httpx.delete(_url(f"/tasks/{task_id}"), headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def get_logs(task_id: str, offset: int = 0) -> dict[str, Any]:
    r = httpx.get(_url(f"/tasks/{task_id}/logs"), params={"offset": offset}, headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def upload_file(file_path: Path) -> dict[str, Any]:
    with open(file_path, "rb") as f:
        r = httpx.post(
            _url("/files/upload"),
            files={"file": (file_path.name, f, "application/gzip")},
            headers=_headers(),
            timeout=300,
        )
    r.raise_for_status()
    return r.json()


def download_file(remote_path: str, local_path: Path) -> None:
    with httpx.stream("GET", _url("/files/download"), params={"path": remote_path}, headers=_headers(), timeout=300) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_bytes(1024 * 64):
                f.write(chunk)


def get_monitor() -> dict[str, Any]:
    r = httpx.get(_url("/monitor"), headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def list_envs() -> list[dict[str, str]]:
    r = httpx.get(_url("/envs"), headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()
