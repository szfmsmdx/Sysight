# AgentLoop — 多轮工具调用引擎

## 概述

AgentLoop 是 Sysight 的 LLM 交互引擎。每个 Pipeline 阶段（ANALYZE、OPTIMIZE、LEARN）都通过一次独立的 `AgentLoop.run()` 调用来完成。

核心设计原则：

1. **每次 run() 是独立的** — 不跨 session 继承对话历史，上下文通过结构化 artifact 传递
2. **Template Method 模式** — 循环骨架固定，工具执行和 LLM 调用分别委托给 ToolRegistry 和 LLMProvider
3. **渐进式上下文压缩** — 当 token 用量接近上限时，自动压缩历史消息

---

## 架构

```
┌──────────────────────────────────────────────────┐
│                   AgentLoop                       │
│                                                  │
│  run(task: AgentTask) → AgentResult              │
│                                                  │
│  while turns < max_turns:                        │
│    1. context.build_model_messages()  ← 压缩     │
│    2. provider.complete(request)       ← LLM     │
│    3. if tool_calls:                             │
│         registry.execute(tool, args)   ← 工具    │
│         context.append_tool_result()   ← 记录    │
│    4. if final_output:                           │
│         validate → return              ← 结束    │
│                                                  │
│  Stop conditions:                                │
│    - max_turns exceeded                          │
│    - max_wall_seconds exceeded                   │
│    - provider error (non-retryable)              │
│    - repeated tool calls (circuit breaker)        │
└──────────────────────────────────────────────────┘
```

---

## AgentTask — 任务定义

```python
@dataclass
class AgentTask:
    run_id: str                    # 运行标识
    task_id: str                   # 任务标识
    task_type: str                 # "analyze" | "optimize" | "learn"
    system_prompt: str             # 系统提示词
    user_prompt: str               # 用户提示词（含预注入数据）
    response_schema: dict | None   # 最终输出的 JSON Schema
    max_turns: int = 30            # 最大轮次
    max_wall_seconds: int = 600    # 最大墙钟时间
    max_tokens: int | None         # 输出 token 限制
    context_policy: ContextPolicy  # 上下文压缩策略
```

### 各阶段的 AgentTask 配置

| 阶段 | max_turns | max_wall_seconds | 可用工具 |
|------|-----------|-----------------|---------|
| ANALYZE | 30 | 600 | nsys_sql_*, scanner_*, memory_read |
| OPTIMIZE | 20 | 600 | scanner_read, scanner_search, scanner_files |
| LEARN | 10 | 120 | memory_read, memory_search, memory_write |

---

## 停止条件

AgentLoop 有 5 种停止方式：

| 停止条件 | 触发时机 | result.status |
|---------|---------|---------------|
| **正常完成** | LLM 返回无 tool_calls 的最终输出 | `"ok"` |
| **Schema 错误** | 最终输出不符合 response_schema | `"schema_error"` |
| **工具错误** | 工具协议违规（如 tool call 和 text 混用） | `"tool_error"` |
| **超时** | turns > max_turns 或 wall > max_wall_seconds | `"timeout"` |
| **Provider 错误** | LLM API 返回不可重试的错误 | `"provider_error"` |

### 重试机制

对于可重试的 provider 错误（rate limit、server error），AgentLoop 内置指数退避重试：

```python
_BACKOFF_S = [5, 10, 20, 30, 40]  # 最多 5 次重试
```

---

## 上下文管理

### 问题

ANALYZE 阶段可能产生大量工具调用结果（SQL 查询结果、源码文件内容），如果不加控制，很快就会超出模型的 context window。

Sysight 的 ANALYZE 一次典型运行：
- 28 turns
- ~900K prompt tokens
- ~50K output tokens

如果不压缩，实际 token 消耗会更高。

### 渐进式压缩策略

Sysight 实现了四级渐进式压缩（参考 MiniCode / Claude Code / Codex 的设计）：

```
Level 0 — Microcompact（≥50% 利用率）
  ├── 清除旧的 COMPACTABLE 工具结果
  └── 保留最近 KEEP_RECENT 条结果

Level 1 — Large-Result Persistence（零模型成本）
  ├── 工具结果 >20K tokens → 持久化到磁盘
  └── 只保留简短预览

Level 2 — Time-Based Compaction（≥70% 利用率）
  ├── 用模板生成的摘要替换旧工具结果
  └── 最近 keep_recent_turns_full 轮的结果保持完整

Level 2.5 — Snip（≥80% 利用率，确定性，不调用 LLM）
  ├── 物理删除中间"安全"区间的消息
  ├── 保护：system 消息、最近 N 条消息、写操作相关的 tool call
  └── 插入 snip_boundary 标记

Level 3 — Token-Pressure Compaction（≥95% 利用率）
  ├── 激进压缩所有旧工具结果
  └── 注入恢复消息（最近读取的文件内容 + session 进度）
```

### Circuit Breaker

如果 Level 3 连续触发 N 次但 token 数没有降到阈值以下（说明压缩无效），则阻止进一步压缩并注入警告。

### Token 估算

采用 **anchor+delta** 方法：
- **anchor**：上一次 API 返回的 `prompt_tokens`
- **delta**：新增消息的字符数 / 3.5（chars_per_token 估计值）
- 典型误差 <5%

### Model-Aware 阈值

所有 token 限制根据模型的实际 context window 自动缩放：

```python
_MODEL_CONTEXT_WINDOWS = {
    "gpt-5.5": 1_000_000,
    "gpt-5.4": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-4-5": 200_000,
    # ...
}
```

---

## 工具调用协议

### 工具调用流程

```
1. LLM 返回 tool_calls
   └── assistant 消息加入 context

2. 逐个执行 tool call
   ├── registry.execute(name, args, policy)
   └── 结果加入 context（role: "tool"）

3. 下一轮 LLM 调用
   └── context.build_model_messages() 包含工具结果
```

### 协议验证

每次构建消息前，AgentLoop 会验证工具调用协议：
- 不允许 tool call 和 text content 同时存在（Anthropic 除外）
- 不允许连续的 tool role 消息
- 不允许 tool call 没有对应的 tool result

### 多 Provider 兼容

AgentLoop 通过 `LLMProvider` 抽象支持多种 LLM backend：

```python
class LLMProvider(ABC):
    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse: ...

@dataclass
class LLMRequest:
    system_prompt: str
    messages: list[dict]
    tools: list[dict] | None
    response_schema: dict | None
    max_tokens: int | None
```

支持的 provider：
- **OpenAI Compatible** — GPT 系列及兼容 API
- **Anthropic** — Claude 系列
- **Replay** — 用于 debug，回放之前的 LLM 响应

---

## 与 Pipeline 的集成

每个 Pipeline 阶段创建自己的 AgentLoop 实例：

```python
# ANALYZE 阶段
loop = AgentLoop(provider, registry, ANALYZE_POLICY)
task = AgentTask(
    run_id=run_id,
    task_id=f"analyze-{run_id}",
    task_type="analyze",
    system_prompt=analyze_system_prompt,
    user_prompt=analyze_user_prompt,  # 含预注入的 profile 摘要
    max_turns=30,
    max_wall_seconds=600,
)
result = loop.run(task)
```

不同阶段使用不同的 `ToolPolicy`，控制 LLM 能访问哪些工具：

```python
# ANALYZE: 可以查询 SQL、阅读源码、读取 memory
ANALYZE_POLICY = ToolPolicy(
    allowed_tools={"nsys_sql_*", "scanner_*", "memory_read"},
    read_only=True,
    max_calls_per_task=100,
)

# OPTIMIZE: 只能阅读源码
OPTIMIZE_POLICY = ToolPolicy(
    allowed_tools={"scanner_read", "scanner_search", "scanner_files"},
    read_only=True,
    max_calls_per_task=30,
)

# LEARN: 可以读写 memory
LEARN_POLICY = ToolPolicy(
    allowed_tools={"memory_read", "memory_search", "memory_write"},
    read_only=False,
    max_calls_per_task=20,
)
```