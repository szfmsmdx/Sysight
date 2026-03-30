"""Thin wrapper around Nsight Systems SQLite exports."""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import subprocess  # nosec B404
import threading
from dataclasses import dataclass, field

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_table_name(name: str | None) -> str | None:
    if name is None:
        return None
    if not _SAFE_IDENTIFIER_RE.match(name):
        raise ValueError(f"Unsafe table name from schema: {name!r}")
    return name


class NsightSchema:
    """Schema discovery for Nsight Systems SQLite exports."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self.tables = [
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
        self.version = self._detect_version()
        self.kernel_table = _validate_table_name(self._detect_table("CUPTI_ACTIVITY_KIND_KERNEL"))
        self.runtime_table = _validate_table_name(self._detect_table("CUPTI_ACTIVITY_KIND_RUNTIME"))
        self.memcpy_table = _validate_table_name(self._detect_table("CUPTI_ACTIVITY_KIND_MEMCPY"))
        self.memset_table = _validate_table_name(self._detect_table("CUPTI_ACTIVITY_KIND_MEMSET"))
        self.nvtx_table = _validate_table_name(self._detect_table("NVTX_EVENTS"))

    def _detect_table(self, prefix: str) -> str | None:
        if prefix in self.tables:
            return prefix
        matches = sorted(table for table in self.tables if table.startswith(prefix))
        return matches[0] if matches else None

    def _read_kv_table(self, table: str) -> dict[str, str]:
        if table not in self.tables:
            return {}
        cols = [row[1] for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()]
        key_col = next((cand for cand in ("key", "Key", "NAME", "Name") if cand in cols), None)
        val_col = next((cand for cand in ("value", "Value", "VAL", "Val") if cand in cols), None)
        if not key_col or not val_col:
            return {}
        out: dict[str, str] = {}
        for key, value in self._conn.execute(f"SELECT {key_col}, {val_col} FROM {table}").fetchall():
            if key is not None and value is not None:
                out[str(key)] = str(value)
        return out

    def _detect_version(self) -> str | None:
        metadata: dict[str, str] = {}
        for table in ("META_DATA_EXPORT", "META_DATA_CAPTURE"):
            metadata.update(self._read_kv_table(table))
        for key, value in metadata.items():
            if "version" in key.lower():
                return value
            if "Nsight Systems" in value:
                return value
        return None


@dataclass
class GpuInfo:
    """Hardware metadata for one GPU."""

    device_id: int
    name: str = ""
    pci_bus: str = ""
    sm_count: int = 0
    memory_bytes: int = 0
    kernel_count: int = 0
    streams: list[int] = field(default_factory=list)


@dataclass
class ProfileMeta:
    """Discovered metadata from an Nsight profile."""

    devices: list[int]
    streams: dict[int, list[int]]
    time_range: tuple[int, int]
    kernel_count: int
    nvtx_count: int
    tables: list[str]
    gpu_info: dict[int, GpuInfo] = field(default_factory=dict)


class Profile:
    """Handle to an opened Nsight Systems SQLite database."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._owns_conn = True
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.schema = NsightSchema(self.conn)
        self._nvtx_has_text_id = self._detect_nvtx_text_id()
        self.meta = self._discover()

    def _detect_nvtx_text_id(self) -> bool:
        table = self.schema.nvtx_table
        if not table:
            return False
        cols = [row[1] for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()]
        return "textId" in cols

    def _discover(self) -> ProfileMeta:
        if not self.schema.kernel_table:
            raise RuntimeError("This profile does not contain GPU kernel activity.")

        kernel_table = self.schema.kernel_table
        devices = [
            row[0]
            for row in self.conn.execute(
                f"SELECT DISTINCT deviceId FROM {kernel_table} ORDER BY deviceId"
            )
        ]
        streams: dict[int, list[int]] = {}
        for row in self.conn.execute(
            f"SELECT DISTINCT deviceId, streamId FROM {kernel_table} ORDER BY deviceId, streamId"
        ):
            streams.setdefault(row[0], []).append(row[1])
        tr = self.conn.execute(f"SELECT MIN(start), MAX([end]) FROM {kernel_table}").fetchone()
        kernel_count = self.conn.execute(f"SELECT COUNT(*) FROM {kernel_table}").fetchone()[0]
        nvtx_count = 0
        if self.schema.nvtx_table:
            nvtx_count = self.conn.execute(f"SELECT COUNT(*) FROM {self.schema.nvtx_table}").fetchone()[
                0
            ]
        return ProfileMeta(
            devices=devices,
            streams=streams,
            time_range=(tr[0] or 0, tr[1] or 0),
            kernel_count=kernel_count,
            nvtx_count=nvtx_count,
            tables=self.schema.tables,
            gpu_info=self._gpu_info(devices, streams),
        )

    def _gpu_info(self, devices: list[int], streams: dict[int, list[int]]) -> dict[int, GpuInfo]:
        info: dict[int, GpuInfo] = {}
        kernel_counts = {
            row[0]: row[1]
            for row in self.conn.execute(
                f"SELECT deviceId, COUNT(*) FROM {self.schema.kernel_table} GROUP BY deviceId"
            )
        }
        hardware = {}
        if "TARGET_INFO_GPU" in self.schema.tables and "TARGET_INFO_CUDA_DEVICE" in self.schema.tables:
            for row in self.conn.execute(
                """
                SELECT c.cudaId AS dev, g.name, g.busLocation, g.smCount AS sms, g.totalMemory AS mem
                FROM TARGET_INFO_GPU g
                JOIN TARGET_INFO_CUDA_DEVICE c ON g.id = c.gpuId
                GROUP BY c.cudaId
                """
            ):
                hardware[row["dev"]] = {
                    "name": row["name"] or "",
                    "pci_bus": row["busLocation"] or "",
                    "sm_count": row["sms"] or 0,
                    "memory_bytes": row["mem"] or 0,
                }
        for dev in devices:
            hw = hardware.get(dev, {})
            info[dev] = GpuInfo(
                device_id=dev,
                name=hw.get("name", ""),
                pci_bus=hw.get("pci_bus", ""),
                sm_count=hw.get("sm_count", 0),
                memory_bytes=hw.get("memory_bytes", 0),
                kernel_count=kernel_counts.get(dev, 0),
                streams=streams.get(dev, []),
            )
        return info

    def kernels(self, device: int | None, trim: tuple[int, int] | None = None) -> list[dict]:
        """Return kernels for one device or all devices."""
        sql = f"""
            SELECT
                k.deviceId,
                k.start,
                k.[end],
                k.streamId,
                k.correlationId,
                s.value AS name,
                COALESCE(d.value, s.value) AS demangled
            FROM {self.schema.kernel_table} k
            JOIN StringIds s ON k.shortName = s.id
            LEFT JOIN StringIds d ON k.demangledName = d.id
            WHERE 1=1
        """
        params: list[object] = []
        if device is not None:
            sql += " AND k.deviceId = ?"
            params.append(device)
        if trim:
            sql += " AND k.start >= ? AND k.[end] <= ?"
            params.extend(trim)
        sql += " ORDER BY k.start"
        with self._lock:
            return [dict(row) for row in self.conn.execute(sql, params)]

    def kernel_map(self, device: int) -> dict[int, dict]:
        """Return kernels for one device indexed by correlationId."""
        sql = f"""
            SELECT
                k.start,
                k.[end],
                k.streamId,
                k.correlationId,
                s.value AS name,
                COALESCE(d.value, s.value) AS demangled
            FROM {self.schema.kernel_table} k
            JOIN StringIds s ON k.shortName = s.id
            LEFT JOIN StringIds d ON k.demangledName = d.id
            WHERE k.deviceId = ?
            ORDER BY k.start
        """
        with self._lock:
            return {
                row["correlationId"]: {
                    "start": row["start"],
                    "end": row["end"],
                    "stream": row["streamId"],
                    "correlationId": row["correlationId"],
                    "name": row["name"],
                    "demangled": row["demangled"],
                }
                for row in self.conn.execute(sql, (device,))
            }

    def gpu_threads(self, device: int) -> set[int]:
        """Return CPU thread IDs that launch kernels for one GPU."""
        if not self.schema.runtime_table:
            return set()
        sql = f"""
            SELECT DISTINCT r.globalTid
            FROM {self.schema.runtime_table} r
            JOIN {self.schema.kernel_table} k ON r.correlationId = k.correlationId
            WHERE k.deviceId = ?
        """
        with self._lock:
            return {row[0] for row in self.conn.execute(sql, (device,))}

    def runtime_calls(
        self,
        threads: set[int],
        window: tuple[int, int],
    ) -> dict[int, list[sqlite3.Row]]:
        """Return runtime calls indexed by thread ID."""
        if not self.schema.runtime_table or not threads:
            return {}
        sql = f"""
            SELECT start, [end], correlationId
            FROM {self.schema.runtime_table}
            WHERE globalTid = ? AND start >= ? AND [end] <= ?
            ORDER BY start
        """
        out: dict[int, list[sqlite3.Row]] = {}
        with self._lock:
            for tid in threads:
                out[tid] = self.conn.execute(sql, (tid, window[0], window[1])).fetchall()
        return out

    def nvtx_events(
        self,
        threads: set[int],
        window: tuple[int, int],
    ) -> list[sqlite3.Row]:
        """Return NVTX events for a thread set in a time window."""
        table = self.schema.nvtx_table
        if not table or not threads:
            return []

        tids = ",".join(str(tid) for tid in sorted(threads))
        if self._nvtx_has_text_id:
            sql = f"""
                SELECT COALESCE(n.text, s.value) AS text, n.globalTid, n.start, n.[end]
                FROM {table} n
                LEFT JOIN StringIds s ON n.textId = s.id
                WHERE (n.text IS NOT NULL OR s.value IS NOT NULL)
                  AND n.[end] > n.start
                  AND n.start >= ? AND n.start <= ?
                  AND n.globalTid IN ({tids})
                ORDER BY n.start
            """
        else:
            sql = f"""
                SELECT text, globalTid, start, [end]
                FROM {table}
                WHERE text IS NOT NULL AND [end] > start
                  AND start >= ? AND start <= ?
                  AND globalTid IN ({tids})
                ORDER BY start
            """
        with self._lock:
            return self.conn.execute(sql, window).fetchall()

    def close(self) -> None:
        if self._owns_conn:
            self.conn.close()

    def __enter__(self) -> "Profile":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def resolve_profile_path(path: str) -> str:
    """Resolve a `.nsys-rep` into a `.sqlite` file when needed."""
    if not path.lower().endswith(".nsys-rep"):
        return path
    output = path[:-9] + ".sqlite"
    if (
        os.path.exists(path)
        and os.path.exists(output)
        and os.path.getsize(output) > 0
        and os.path.getmtime(output) >= os.path.getmtime(path)
    ):
        return output
    nsys_exe = shutil.which("nsys")
    if not nsys_exe:
        raise RuntimeError(
            "Profile is .nsys-rep; conversion requires `nsys` on PATH. "
            "Export manually with: nsys export --type sqlite -o <out.sqlite> <file.nsys-rep>"
        )
    subprocess.run(  # nosec B603
        [nsys_exe, "export", "--type=sqlite", "-o", output, "--force-overwrite=true", path],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if not (os.path.exists(output) and os.path.getsize(output) > 0):
        raise RuntimeError(f"nsys export did not produce a usable sqlite file at {output!r}")
    return output


def open(path: str) -> Profile:
    """Open a profile path."""
    return Profile(resolve_profile_path(path))
