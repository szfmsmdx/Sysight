from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from server.auth import verify_api_key
from server.models import MonitorSnapshot
from server.services.monitor import collect_snapshot

router = APIRouter(prefix="/monitor", tags=["monitor"])


@router.get("", response_model=MonitorSnapshot, dependencies=[Depends(verify_api_key)])
async def get_snapshot() -> MonitorSnapshot:
    return collect_snapshot()


@router.get("/stream", dependencies=[Depends(verify_api_key)])
async def stream_monitor():
    async def generate():
        while True:
            snapshot = collect_snapshot()
            yield f"data: {snapshot.model_dump_json()}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(generate(), media_type="text/event-stream")
