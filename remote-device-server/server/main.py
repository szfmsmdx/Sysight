from contextlib import asynccontextmanager

from fastapi import FastAPI

from server.config import settings
from server.db import close_db, get_db
from server.routers import envs, files, monitor, tasks


@asynccontextmanager
async def lifespan(app: FastAPI):
    await get_db()
    yield
    await close_db()


app = FastAPI(title="Remote Device Server", version="0.1.0", lifespan=lifespan)

app.include_router(tasks.router)
app.include_router(files.router)
app.include_router(monitor.router)
app.include_router(envs.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


def _print_banner():
    import socket

    # 获取本机 IP
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips and ip != "127.0.0.1":
                ips.append(ip)
    except Exception:
        pass
    if not ips:
        ips = ["127.0.0.1"]

    print("=" * 50)
    print("  Remote Device Server")
    print("=" * 50)
    print(f"  Host:      {settings.host}")
    print(f"  Port:      {settings.port}")
    print(f"  API Key:   {settings.api_key}")
    print(f"  Workspace: {settings.workspace_root}")
    print(f"  Log dir:   {settings.log_root}")
    print(f"  DB:        {settings.db_path}")
    print()
    print("  Client config (~/.rds/config):")
    for ip in ips:
        print(f"    server_url=http://{ip}:{settings.port}")
    print(f"    api_key={settings.api_key}")
    print("=" * 50)
    print()


def run():
    import uvicorn

    _print_banner()
    uvicorn.run(
        "server.main:app",
        host=settings.host,
        port=settings.port,
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )


if __name__ == "__main__":
    run()
