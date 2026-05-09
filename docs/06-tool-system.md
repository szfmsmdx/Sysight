# 工具系统

## 概述

Sysight 的工具系统是 LLM 与外部世界交互的唯一通道。LLM 不直接读文件、不直接查 SQL——所有操作都通过工具完成。

设计原则：
1. **Registry 模式** — 所有工具统一注册，按名称发现
2. **Policy 控制** — 每个 Pipeline 阶段有不同的工具访问权限
3. **Read-only 优先** — 默认只允许读操作，写操作需要显式授权

---

## 架构

```
┌─────────────────────────────────────────────┐
│               ToolRegistry                   │
│                                             │
│  register(tool: ToolDef)                    │
│  execute(name, args, policy) → ToolResult   │
│  as_openai_tools(policy) → list[dict]       │
│                                             │
│  _tools: dict[str, ToolDef]                 │
│  _call_counts: dict[str, int]               │
└─────────────────────────────────────────────┘
         │
         │ 注册
         ▼
┌─────────────────────────────────────────────┐
│               ToolDef                        │
│                                             │
│  name: str           # "scanner_read"       │
│  description: str    # LLM 看到的功能描述    │
│  parameters: dict    # JSON Schema          │
│  fn: Callable        # 实际执行函数          │
│  read_only: bool     # 是否只读              │
│  max_calls_per_task  # 单任务最大调用次数     │
└─────────────────────────────────────────────┘
```

---

## 工具分类

### Scanner 工具 — 源码阅读

| 工具 | 说明 |
|------|------|
| `scanner_read` | 读取文件内容，支持 `start`/`end` 行号范围 |
| `scanner_search` | 在仓库中搜索代码（regex） |
| `scanner_files` | 列出目录结构 |
| `scanner_symbols` | 查找函数/类定义 |
| `scanner_callers` | 查找调用方 |
| `scanner_variants` | 查找代码变体（不同配置分支） |

### nsys_sql 工具 — Profile 查询

| 工具 | 说明 |
|------|------|
| `nsys_sql_gaps` | GPU 空闲间隙分析 |
| `nsys_sql_kernels` | Top kernel 耗时统计 |
| `nsys_sql_memcpy` | H2D/D2H 内存搬运统计 |
| `nsys_sql_sync` | 同步点分析 |
| `nsys_sql_nvtx` | NVTX 区间耗时 |
| `nsys_sql_nccl` | NCCL 通信统计 |
| `nsys_sql_launch` | Kernel launch 开销 |
| `nsys_sql_overlap` | 计算/通信 overlap 分析 |

### Memory 工具 — 知识库读写

| 工具 | 说明 |
|------|------|
| `memory_read` | 读取 wiki 页面 |
| `memory_search` | 搜索 wiki 内容 |
| `memory_write` | 写入 wiki 页面（仅 LEARN 阶段可用） |

### Sandbox 工具 — 代码执行

| 工具 | 说明 |
|------|------|
| `sandbox_create` | 创建隔离执行环境 |
| `sandbox_exec` | 在沙箱中执行命令 |
| `sandbox_measure` | 测量执行时间 |
| `sandbox_destroy` | 销毁沙箱 |

### Patcher — 代码修改

| 组件 | 说明 |
|------|------|
| `PatchApplier` | 确定性 patch apply/revert，非 LLM 工具 |

> Patcher 不是 LLM 工具——它是 EXECUTE 阶段的代码组件。LLM 只生成 PatchCandidate，不直接修改文件。

---

## ToolPolicy — 访问控制

每个 Pipeline 阶段有独立的 ToolPolicy：

```python
@dataclass
class ToolPolicy:
    allowed_tools: set[str]       # 允许的工具名（支持 "scanner_*" 通配符）
    read_only: bool = True        # 是否只允许读操作
    max_calls_per_task: int = 50  # 单任务最大工具调用次数
    max_wall_seconds: int = 600   # 工具调用超时
    path_containment: dict        # 路径限制（如限制 scanner_read 只能读 repo 内文件）
    max_reads_per_file: int = 0   # 单文件最大读取次数（0 = 不限制）
```

### 各阶段的 ToolPolicy

| 阶段 | 允许的工具 | read_only | max_calls |
|------|-----------|-----------|-----------|
| ANALYZE | `nsys_sql_*`, `scanner_*`, `memory_read` | ✅ | 100 |
| OPTIMIZE | `scanner_read`, `scanner_search`, `scanner_files` | ✅ | 30 |
| LEARN | `memory_read`, `memory_search`, `memory_write` | ❌ | 20 |

---

## 工具执行流程

```
ToolRegistry.execute(name, args, policy)
  │
  ├── 1. 查找 ToolDef
  │     └── 未找到 → ToolResult(status="error")
  │
  ├── 2. 检查 Policy
  │     ├── 工具不在 allowed_tools 中 → "policy_denied"
  │     ├── 工具是写操作但 policy.read_only → "policy_denied"
  │     └── 调用次数超限 → "policy_denied"
  │
  ├── 3. 执行
  │     ├── 计时
  │     ├── 调用 tool.fn(**args)
  │     └── 捕获异常
  │
  └── 4. 返回 ToolResult
        ├── status: "ok" | "error" | "policy_denied"
        ├── data: 返回值
        └── elapsed_ms: 执行耗时
```

---

## 工具结果在上下文中的处理

工具结果返回给 LLM 之前，会经过 `to_jsonable()` 转换：

```python
def to_jsonable(data):
    """将 dataclass / Path / set 等类型转为 JSON 兼容格式"""
```

对于超大结果（>20K tokens），AgentContext 会自动持久化到磁盘，只保留简短预览。LLM 可以通过 `scanner_read` 重新读取完整内容。

---

## 工具注册

所有工具在 `sysight/tools/__init__.py` 的 `register_all_tools()` 中统一注册：

```python
def register_all_tools(registry: ToolRegistry):
    # Scanner 工具
    from sysight.tools.scanner import read, search, files, symbols, callers, variants
    registry.register(ToolDef(
        name="scanner_read",
        description="Read a file with optional line range...",
        parameters={...},
        fn=read,
    ))
    # ... 更多工具

    # nsys_sql 工具
    from sysight.tools.nsys_sql import gaps, kernels, memcpy, ...
    # ...

    # Memory 工具
    from sysight.tools.memory import read, search, write
    # ...
```

---

## 设计决策

### 为什么不让 LLM 直接读文件

1. **安全**：通过 ToolPolicy 限制 LLM 只能读 repo 内的文件
2. **可控**：可以限制调用次数、单文件读取次数
3. **可审计**：所有工具调用都记录在 debug.log 中
4. **可压缩**：超大结果可以持久化到磁盘，不占 context window

### 为什么 Patcher 不是 LLM 工具

LLM 生成 patch 后，由代码侧（非 LLM）执行 apply。原因：
- LLM 输出的行号可能不准 → 代码侧用 hash 验证
- LLM 不应该有修改文件的能力 → 安全边界
- apply/revert 是确定性操作 → 不需要推理能力