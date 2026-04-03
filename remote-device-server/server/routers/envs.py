from __future__ import annotations

import json
import subprocess
from typing import List

from fastapi import APIRouter, Depends

from server.auth import verify_api_key
from server.models import CondaEnvInfo

router = APIRouter(prefix="/envs", tags=["envs"])


@router.get("", response_model=List[CondaEnvInfo], dependencies=[Depends(verify_api_key)])
async def list_conda_envs() -> List[CondaEnvInfo]:
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        envs = []
        for path in data.get("envs", []):
            name = path.split("/")[-1]
            envs.append(CondaEnvInfo(name=name, path=path))
        return envs
    except Exception:
        return []
