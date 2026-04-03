from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_config():
    # type: () -> dict
    config = {}
    config_file = Path.home() / ".rds" / "config"
    if config_file.exists():
        for line in config_file.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                config[k.strip()] = v.strip()
    return config


_cfg = _load_config()

SERVER_URL = os.environ.get("RDS_SERVER_URL", _cfg.get("server_url", ""))
API_KEY = os.environ.get("RDS_API_KEY", _cfg.get("api_key", ""))

if not SERVER_URL or not API_KEY:
    print(
        "Error: RDS_SERVER_URL and RDS_API_KEY must be configured.\n"
        "\n"
        "Option 1 - config file (~/.rds/config):\n"
        "  mkdir -p ~/.rds\n"
        "  echo 'server_url=http://YOUR_SERVER_IP:PORT' >> ~/.rds/config\n"
        "  echo 'api_key=YOUR_API_KEY' >> ~/.rds/config\n"
        "\n"
        "Option 2 - environment variables:\n"
        "  export RDS_SERVER_URL=http://YOUR_SERVER_IP:PORT\n"
        "  export RDS_API_KEY=YOUR_API_KEY",
        file=sys.stderr,
    )
    sys.exit(1)
