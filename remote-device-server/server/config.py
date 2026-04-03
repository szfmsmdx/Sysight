from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    api_key: str = "change-me-to-a-strong-random-key"
    host: str = "0.0.0.0"
    port: int = 44401
    workspace_root: str = "/tmp/rds_workspace"
    log_root: str = "/tmp/rds_logs"
    max_concurrent_tasks: int = 4
    db_path: str = "/tmp/rds.db"

    def __post_init__(self):
        _INT_FIELDS = {"port", "max_concurrent_tasks"}
        _PATH_FIELDS = {"workspace_root", "log_root", "db_path"}

        for name in ("api_key", "host", "port", "workspace_root",
                      "log_root", "max_concurrent_tasks", "db_path"):
            env_key = "RDS_" + name.upper()
            val = os.environ.get(env_key)
            if val is not None:
                if name in _INT_FIELDS:
                    setattr(self, name, int(val))
                else:
                    setattr(self, name, val)

        self.port = int(self.port)
        self.max_concurrent_tasks = int(self.max_concurrent_tasks)
        self.workspace_root = Path(self.workspace_root)
        self.log_root = Path(self.log_root)
        self.db_path = Path(self.db_path)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)


settings = Settings()
