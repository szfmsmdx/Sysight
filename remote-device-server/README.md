# Remote Device Server

通过 HTTP API 在远端 GPU 机器上执行任务、查看日志、监控资源。

## 快速开始

### 1. 配置连接信息

创建配置文件 `~/.rds/config`：

```bash
mkdir -p ~/.rds
cat > ~/.rds/config << 'EOF'
server_url=http://YOUR_SERVER_IP:PORT
api_key=YOUR_API_KEY
EOF
```

或者使用环境变量：

```bash
export RDS_SERVER_URL=http://YOUR_SERVER_IP:PORT
export RDS_API_KEY=YOUR_API_KEY
```

### 2. 安装 Client（可选）

如果使用本仓库的 CLI 工具：

```bash
git clone https://github.com/xgbj/remote-device-server.git
cd remote-device-server
pip install -e .

# 测试连接
rds health
```

## Agent 集成指南

任何 agent 都可以通过 HTTP 调用以下 API 将本地项目推送到远端 GPU 执行。不需要安装本仓库的 client，直接用 `curl` / `httpx` / `requests` 即可。

### 典型工作流：推送代码并执行

```bash
# 从配置文件读取（或使用环境变量）
SERVER_URL="http://YOUR_SERVER_IP:PORT"
API_KEY="YOUR_API_KEY"

# 1. 打包本地项目
cd /path/to/your/project
tar -czf /tmp/project.tar.gz -C /path/to/your/project .

# 2. 上传到远端
curl -X POST ${SERVER_URL}/files/upload \
  -H "X-API-Key: ${API_KEY}" \
  -F "file=@/tmp/project.tar.gz"
# 返回: {"upload_id": "abc123", "filename": "project.tar.gz"}

# 3. 提交任务（server 自动解压到工作目录）
curl -X POST ${SERVER_URL}/tasks \
  -H "X-API-Key: ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"command": "pip install -r requirements.txt && python run.py", "upload_id": "abc123"}'
# 返回: {"id": "task_id", "status": "pending", ...}

# 4. 轮询日志直到完成
curl ${SERVER_URL}/tasks/{task_id}/logs?offset=0 \
  -H "X-API-Key: ${API_KEY}"
# 返回: {"data": "...", "offset": 1234}
# 用返回的 offset 继续请求获取增量日志

# 5. 检查任务状态
curl ${SERVER_URL}/tasks/{task_id} \
  -H "X-API-Key: ${API_KEY}"
# status: pending | running | success | failed | cancelled
```

### API 参考

所有请求需要 header: `X-API-Key: YOUR_API_KEY`

#### 健康检查

```
GET /health
→ {"status": "ok"}
```

#### 任务管理

```
POST /tasks
Body: {"command": "...", "conda_env": null, "working_dir": null, "upload_id": null}
→ {"id": "...", "status": "pending", "command": "...", ...}

GET /tasks?limit=50&offset=0
→ {"tasks": [...], "total": N}

GET /tasks/{task_id}
→ {"id": "...", "status": "success", "exit_code": 0, ...}

DELETE /tasks/{task_id}
→ {"status": "cancelled"}
```

#### 日志

```
GET /tasks/{task_id}/logs?offset=0
→ {"data": "stdout+stderr内容", "offset": 字节偏移}

WebSocket /tasks/{task_id}/logs/ws
→ 实时推送日志行，任务结束后连接关闭
```

#### 文件操作

```
POST /files/upload
Form: file=@local.tar.gz
→ {"upload_id": "...", "filename": "..."}

GET /files/download?path=/remote/path/to/file
→ 文件内容（二进制流）
```

#### 系统监控

```
GET /monitor
→ {"cpu_percent": [...], "memory": {...}, "gpus": [{...}], "disk": {...}}
```

### Agent 实现示例（Python）

```python
import httpx
import tarfile
import time
from pathlib import Path

# 从配置文件或环境变量读取
def load_config():
    import os
    config_file = Path.home() / ".rds" / "config"
    if config_file.exists():
        config = {}
        for line in config_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                config[k.strip()] = v.strip()
        return config
    return {
        "server_url": os.environ.get("RDS_SERVER_URL"),
        "api_key": os.environ.get("RDS_API_KEY"),
    }

config = load_config()
SERVER = config["server_url"]
HEADERS = {"X-API-Key": config["api_key"]}

def run_on_remote(project_dir: str, command: str) -> str:
    """打包本地项目，推送到远端 GPU 执行，返回完整日志。"""

    # 1. 打包
    tar_path = "/tmp/_rds_upload.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(project_dir, arcname=".")

    # 2. 上传
    with open(tar_path, "rb") as f:
        r = httpx.post(f"{SERVER}/files/upload", headers=HEADERS,
                       files={"file": ("project.tar.gz", f)}, timeout=300)
    upload_id = r.json()["upload_id"]

    # 3. 提交任务
    r = httpx.post(f"{SERVER}/tasks", headers=HEADERS, json={
        "command": command,
        "upload_id": upload_id,
    }, timeout=30)
    task_id = r.json()["id"]

    # 4. 轮询日志直到完成
    logs = ""
    offset = 0
    while True:
        r = httpx.get(f"{SERVER}/tasks/{task_id}/logs", headers=HEADERS,
                      params={"offset": offset}, timeout=10)
        chunk = r.json()
        if chunk["data"]:
            logs += chunk["data"]
            offset = chunk["offset"]

        r = httpx.get(f"{SERVER}/tasks/{task_id}", headers=HEADERS, timeout=10)
        status = r.json()["status"]
        if status in ("success", "failed", "cancelled"):
            # 取最后一批日志
            r = httpx.get(f"{SERVER}/tasks/{task_id}/logs", headers=HEADERS,
                          params={"offset": offset}, timeout=10)
            logs += r.json()["data"]
            break
        time.sleep(2)

    return logs

# 用法
output = run_on_remote(
    project_dir="/home/user/my-cuda-kernel",
    command="pip install -r requirements.txt -q && python run.py --mode perf",
)
print(output)
```

### 注意事项

- 上传的 tar.gz 会被解压到临时工作目录，command 在该目录下执行
- 最大并发任务数为 4，超出会排队等待
- 日志通过文件持久化，任务结束后仍可查询
- 如需指定 conda 环境，在 body 中传 `"conda_env": "env_name"`

## Server 部署

在远端 GPU 机器上：

```bash
# 1. 安装依赖
pip install fastapi uvicorn[standard] pydantic aiosqlite pynvml psutil python-multipart httpx typer rich websockets

# 2. 配置环境变量
export RDS_API_KEY=your-secret-key
export RDS_HOST=0.0.0.0
export RDS_PORT=44401

# 3. 启动 server
python -m server.main
```

或使用快速启动脚本：

```bash
bash scripts/quick_start.sh
```
