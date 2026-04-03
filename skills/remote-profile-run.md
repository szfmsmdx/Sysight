---
name: remote-profile-run
description: 当用户想在远端 GPU 机器上运行 nsys profile、提交训练/推理任务、或把本地脚本推送到远端执行时使用。通过 remote-device-server 客户端与远端建立连接，完成"提交任务 → 实时拉取日志 → 取回结果文件"的完整流程。也可以单独用于查询远端 GPU 状态、任务历史、拉取 nsys-rep/sqlite 文件等操作。
---

# Remote Profile Run Skill

这个 skill 把"在远端 A100 机器上执行命令并取回结果"变成标准化流程，
核心用途：远端跑 nsys profile → 下载 sqlite → 本地 Sysight 分析。

---

## 环境前提

### 客户端（本机 Mac）

RDS CLI 已安装在 venv：
```
/Users/szf/Desktop/Sysight/remote-device-server/.venv/bin/rds
```

配置文件 `~/.rds/config`（已写入）：
```
server_url=http://10.164.56.75:44402
api_key=rds_szf
```

简写别名（在 shell 中使用）：
```bash
RDS=/Users/szf/Desktop/Sysight/remote-device-server/.venv/bin/rds
```

### 服务端（远端机器）

- IP: `10.164.56.75`，Port: `44402`
- 8 × NVIDIA A100-SXM4-80GB
- 服务已在 tmux 中运行，Ctrl+C 停止
- 项目代码路径：`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-perception/shenzhaofeng02/`
- workspace 目录（上传文件解压到此）：`/mnt/dolphinfs/ssd_pool/docker/user/hadoop-perception/shenzhaofeng02/workspace`

---

## 标准工作流

### 场景 A：一次性远端命令（nvidia-smi、查文件、简单测试）

```bash
RDS=/Users/szf/Desktop/Sysight/remote-device-server/.venv/bin/rds

# 提交并等待完成
TASK=$($RDS run "nvidia-smi" | grep "Task submitted:" | awk '{print $3}')
sleep 3
$RDS logs $TASK
```

或者用 Python API 直接同步等待（推荐用于 agent 自动化）：

```python
import httpx, time

SERVER = "http://10.164.56.75:44402"
HEADERS = {"X-API-Key": "rds_szf"}

def run_remote(command: str, timeout: int = 300) -> str:
    """提交命令，等待完成，返回完整日志字符串。"""
    r = httpx.post(f"{SERVER}/tasks", headers=HEADERS,
                   json={"command": command}, timeout=30)
    task_id = r.json()["id"]

    deadline = time.time() + timeout
    offset = 0
    logs = ""
    while time.time() < deadline:
        # 拉增量日志
        r = httpx.get(f"{SERVER}/tasks/{task_id}/logs",
                      headers=HEADERS, params={"offset": offset}, timeout=10)
        chunk = r.json()
        if chunk["data"]:
            logs += chunk["data"]
            offset = chunk["offset"]
        # 检查状态
        r = httpx.get(f"{SERVER}/tasks/{task_id}", headers=HEADERS, timeout=10)
        status = r.json()["status"]
        if status in ("success", "failed", "cancelled"):
            # 拉最后一批日志
            r = httpx.get(f"{SERVER}/tasks/{task_id}/logs",
                          headers=HEADERS, params={"offset": offset}, timeout=10)
            logs += r.json()["data"]
            break
        time.sleep(2)
    return logs
```

### 场景 B：推送本地脚本到远端执行

```python
import httpx, tarfile, time, tempfile
from pathlib import Path

SERVER = "http://10.164.56.75:44402"
HEADERS = {"X-API-Key": "rds_szf"}

def push_and_run(local_dir: str, command: str, timeout: int = 600) -> str:
    """打包本地目录，上传到远端，执行命令，返回日志。"""
    # 1. 打包
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
        tar_path = f.name
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(local_dir, arcname=".")

    # 2. 上传
    with open(tar_path, "rb") as f:
        r = httpx.post(f"{SERVER}/files/upload", headers=HEADERS,
                       files={"file": ("upload.tar.gz", f)}, timeout=300)
    upload_id = r.json()["upload_id"]

    # 3. 提交任务
    r = httpx.post(f"{SERVER}/tasks", headers=HEADERS,
                   json={"command": command, "upload_id": upload_id}, timeout=30)
    task_id = r.json()["id"]
    print(f"Task submitted: {task_id}")

    # 4. 等待并流式打印日志
    offset, logs = 0, ""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = httpx.get(f"{SERVER}/tasks/{task_id}/logs",
                      headers=HEADERS, params={"offset": offset}, timeout=10)
        chunk = r.json()
        if chunk["data"]:
            print(chunk["data"], end="", flush=True)
            logs += chunk["data"]
            offset = chunk["offset"]
        r = httpx.get(f"{SERVER}/tasks/{task_id}", headers=HEADERS, timeout=10)
        if r.json()["status"] in ("success", "failed", "cancelled"):
            break
        time.sleep(2)
    return logs
```

### 场景 C：远端跑 nsys profile → 下载 sqlite → 本地分析（完整流程）

```python
import httpx, time
from pathlib import Path

SERVER = "http://10.164.56.75:44402"
HEADERS = {"X-API-Key": "rds_szf"}
REMOTE_PROFILE_DIR = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-perception/shenzhaofeng02"

def profile_and_analyze(
    run_command: str,         # 例如 "python train.py --config config.yaml"
    profile_name: str,        # 例如 "my_exp_001"
    local_output_dir: str = "/Users/szf/Desktop/Sysight/profiles",
    conda_env: str | None = None,
) -> str:
    """远端 nsys 采集 → 导出 sqlite → 下载到本地 → 返回本地 sqlite 路径。"""

    remote_sqlite = f"{REMOTE_PROFILE_DIR}/{profile_name}.sqlite"

    # 1. 用 nsys 采集
    nsys_cmd = (
        f"nsys profile --output {REMOTE_PROFILE_DIR}/{profile_name} "
        f"--export sqlite --force-overwrite true "
        f"{run_command}"
    )
    r = httpx.post(f"{SERVER}/tasks", headers=HEADERS,
                   json={"command": nsys_cmd, "conda_env": conda_env}, timeout=30)
    task_id = r.json()["id"]
    print(f"nsys profile task: {task_id}")

    # 等待完成
    _wait_task(task_id)

    # 2. 如果 nsys 没有直接输出 sqlite，手动导出
    export_cmd = (
        f"[ -f {remote_sqlite} ] || "
        f"nsys export --type sqlite -o {remote_sqlite} "
        f"{REMOTE_PROFILE_DIR}/{profile_name}.nsys-rep"
    )
    r = httpx.post(f"{SERVER}/tasks", headers=HEADERS,
                   json={"command": export_cmd}, timeout=30)
    _wait_task(r.json()["id"])

    # 3. 下载 sqlite 到本地
    local_sqlite = Path(local_output_dir) / f"{profile_name}.sqlite"
    with httpx.stream("GET", f"{SERVER}/files/download",
                      headers=HEADERS, params={"path": remote_sqlite},
                      timeout=300) as resp:
        resp.raise_for_status()
        with open(local_sqlite, "wb") as f:
            for chunk in resp.iter_bytes(65536):
                f.write(chunk)
    print(f"Downloaded: {local_sqlite}")
    return str(local_sqlite)


def _wait_task(task_id: str, timeout: int = 1800, poll: int = 5) -> dict:
    deadline = time.time() + timeout
    offset = 0
    while time.time() < deadline:
        r = httpx.get(f"{SERVER}/tasks/{task_id}/logs",
                      headers=HEADERS, params={"offset": offset}, timeout=10)
        chunk = r.json()
        if chunk["data"]:
            print(chunk["data"], end="", flush=True)
            offset = chunk["offset"]
        r = httpx.get(f"{SERVER}/tasks/{task_id}", headers=HEADERS, timeout=10)
        info = r.json()
        if info["status"] in ("success", "failed", "cancelled"):
            if info["status"] != "success":
                raise RuntimeError(f"Task {task_id} ended with status {info['status']}")
            return info
        time.sleep(poll)
    raise TimeoutError(f"Task {task_id} timed out after {timeout}s")
```

---

## 常用 CLI 速查

```bash
RDS=/Users/szf/Desktop/Sysight/remote-device-server/.venv/bin/rds

$RDS health                          # 连通性检查
$RDS run "nvidia-smi"                # 提交一次性命令
$RDS run "command" --conda myenv     # 指定 conda 环境
$RDS ps                              # 列出最近任务
$RDS logs <task_id>                  # 查日志（全量）
$RDS logs <task_id> --follow         # 实时流式日志（WebSocket）
$RDS info <task_id>                  # 查任务详情（status/exit_code）
$RDS cancel <task_id>                # 取消任务
```

---

## 远端关键路径

| 用途 | 路径 |
|------|------|
| 代码主目录 | `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-perception/shenzhaofeng02/` |
| RDS workspace（上传解压位置）| `/mnt/.../shenzhaofeng02/workspace/` |
| RDS 日志 | `/mnt/.../shenzhaofeng02/rds_logs/` |
| nsys 输出建议存放 | `/mnt/.../shenzhaofeng02/` （便于下载） |

---

## 注意事项

- `httpx` 已在 `remote-device-server/.venv` 中安装，也可以在 Sysight 主环境中使用
- `run_remote()` / `push_and_run()` 函数可直接复制到临时脚本里运行
- 任务最大并发 4，超出排队等待
- ⚠️ **远端机器当前没有安装 `nsys`**（已验证）。场景 C 的 nsys 采集流程需要先在远端安装 nsys，或者在已有 nsys 的机器上执行 profile，再把 `.nsys-rep` / `.sqlite` 传到这台 A100 机器来运行模型
- 现阶段最实用的用法是：**用这台 A100 跑训练/推理脚本**，如果对方已经安装了 nsys 则可以直接带 `nsys profile` 前缀；如果没有则先安装（`conda install -c nvidia nsys-cli` 或联系运维）
- nsys profile 耗时通常较长（数分钟），`_wait_task` 的 `timeout` 默认 1800s
- 下载大 sqlite 文件时 `timeout=300` 可能不够，视文件大小调整
- 服务需要在远端 tmux 中手动维护，掉线后重新 `source .env && python -m server.main`
