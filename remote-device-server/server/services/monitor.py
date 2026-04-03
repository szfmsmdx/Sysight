from __future__ import annotations

from datetime import datetime, timezone

import psutil

from server.models import (
    DiskInfo,
    GpuInfo,
    MemoryInfo,
    MonitorSnapshot,
)


def _get_gpu_info() -> list[GpuInfo]:
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus: list[GpuInfo] = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            gpus.append(
                GpuInfo(
                    index=i,
                    name=name,
                    utilization_percent=util.gpu,
                    memory_used_mb=mem.used // (1024 * 1024),
                    memory_total_mb=mem.total // (1024 * 1024),
                    temperature_c=temp,
                )
            )
        return gpus
    except Exception:
        return []


def collect_snapshot() -> MonitorSnapshot:
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    return MonitorSnapshot(
        cpu_percent=psutil.cpu_percent(percpu=True),
        memory=MemoryInfo(
            total_gb=round(vm.total / 1e9, 2),
            used_gb=round(vm.used / 1e9, 2),
            available_gb=round(vm.available / 1e9, 2),
            percent=vm.percent,
        ),
        gpus=_get_gpu_info(),
        disk=DiskInfo(
            total_gb=round(disk.total / 1e9, 2),
            used_gb=round(disk.used / 1e9, 2),
            free_gb=round(disk.free / 1e9, 2),
            percent=disk.percent,
        ),
        timestamp=datetime.now(timezone.utc),
    )
