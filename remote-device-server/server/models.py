from __future__ import annotations

import enum
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel


class TaskStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    success = "success"
    failed = "failed"
    cancelled = "cancelled"


class TaskCreate(BaseModel):
    command: str
    conda_env: Optional[str] = None
    working_dir: Optional[str] = None
    upload_id: Optional[str] = None


class TaskInfo(BaseModel):
    id: str
    command: str
    conda_env: Optional[str] = None
    status: TaskStatus = TaskStatus.pending
    exit_code: Optional[int] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    working_dir: Optional[str] = None


class TaskListResponse(BaseModel):
    tasks: List[TaskInfo]
    total: int


class LogChunk(BaseModel):
    data: str
    offset: int


class MemoryInfo(BaseModel):
    total_gb: float
    used_gb: float
    available_gb: float
    percent: float


class GpuInfo(BaseModel):
    index: int
    name: str
    utilization_percent: int
    memory_used_mb: int
    memory_total_mb: int
    temperature_c: int


class DiskInfo(BaseModel):
    total_gb: float
    used_gb: float
    free_gb: float
    percent: float


class MonitorSnapshot(BaseModel):
    cpu_percent: List[float]
    memory: MemoryInfo
    gpus: List[GpuInfo]
    disk: DiskInfo
    timestamp: datetime


class CondaEnvInfo(BaseModel):
    name: str
    path: str
