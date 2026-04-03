from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from server.auth import verify_api_key
from server.config import settings

router = APIRouter(prefix="/files", tags=["files"])


@router.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_file(file: UploadFile = File(...)) -> Dict[str, str]:
    """Upload a tarball for task execution. Returns upload_id."""
    upload_id = uuid.uuid4().hex[:12]
    dest = settings.workspace_root / f"{upload_id}.tar.gz"
    with open(dest, "wb") as f:
        chunk = await file.read(1024 * 1024)
        while chunk:
            f.write(chunk)
            chunk = await file.read(1024 * 1024)
    return {"upload_id": upload_id, "filename": file.filename or "unknown"}


@router.get("/download", dependencies=[Depends(verify_api_key)])
async def download_file(path: str) -> FileResponse:
    """Download a file from the server."""
    file_path = Path(path).resolve()
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(file_path, filename=file_path.name)
