# 知识库与记忆系统

## 概述

Sysight 的知识库（Wiki）是一个基于文件的持久化存储系统，用于积累和复用性能优化经验。

设计原则：
1. **Parent-only writes** — 只有 Pipeline 的 LEARN 阶段可以写入，子 Agent 只能读取
2. **Workspace 隔离** — 每个代码仓库有独立的 namespace，互不干扰
3. **Global experience** — 跨仓库的通用优化经验存储在全局 experience 中

---

## 存储结构

```
.sysight/memory/wiki/
├── workspaces/
│   └── <namespace>/           # 每个仓库一个 namespace
│       ├── overview.md        # 仓库结构、入口、配置、调用关系
│       └── experience.md      # 该仓库特有的优化经验
└── experiences/
    └── experience.md          # 全局优化经验（跨仓库共享）
```

### Namespace 生成

```python
# namespace = workspace_namespace(repo_root)
# 基于 repo 路径的 hash，保证同一仓库总是映射到同一 namespace
```

---

## WikiRepository — CRUD 操作

```python
class WikiRepository:
    def read_page(path: str) -> str | None
    def write_page(path: str, content: str, ...) -> Path
    def append_page(path: str, content: str) -> None
    def replace_in_page(path: str, old: str, new: str) -> None
    def search(query: str) -> list[dict]
```

### 页面格式

每个 wiki 页面使用 YAML frontmatter + Markdown body：

```markdown
---
title: "Workspace Overview"
category: "overview"
tags: ["training", "single-gpu"]
scope: "workspace"
source_run: "run-f15b4613"
updated_at: "2026-05-07T18:10:26Z"
---

# Workspace Overview

## Entry Point
python run.py --config configs/train.yaml

## Hot Paths
- src/train.py
- src/model.py
...
```

---

## LEARN 阶段 — 知识提取

LEARN 阶段在 Pipeline 中被调用两次：

### 第一次：ANALYZE 之后

输入：ANALYZE 的 findings

LLM 分析 finding 模式，提取可复用的知识：
- 这个仓库常见的性能问题类型
- 特定文件/函数的已知问题
- 配置文件的常见陷阱

### 第二次：EXECUTE 之后

输入：findings + patches + timer 结果

LLM 分析优化效果，提取经验：
- 哪些优化确实有效（timer 证实）
- 哪些 finding 是假阳性（被 OPTIMIZE 拒绝）
- 哪些优化技巧可以推广到其他仓库

### LEARN AgentLoop

```
┌─────────────────────────────────────────────────────┐
│                  LEARN AgentLoop                     │
│                                                     │
│  1. memory_read 当前 wiki 状态                       │
│     ├── workspace overview                          │
│     └── global experience                           │
│                                                     │
│  2. 对比新结果                                       │
│     ├── findings（ANALYZE 输出）                     │
│     └── patches + timer（EXECUTE 输出）              │
│                                                     │
│  3. 决定更新                                         │
│     ├── 新反模式 → memory_write experience           │
│     ├── 新优化技巧 → memory_write experience         │
│     ├── 过时知识 → memory_write 替换                 │
│     └── 无新信息 → 跳过                              │
│                                                     │
│  约束：max_turns=10, max_wall_seconds=120           │
└─────────────────────────────────────────────────────┘
```

### 输出格式

```json
{
  "summary": "本次学习摘要",
  "memory_updates": [
    {
      "path": "workspaces/my-repo/experience.md",
      "action": "append",
      "content": "## DataLoader 配置\n- num_workers=0 导致 GPU 空闲 93%..."
    },
    {
      "path": "experiences/experience.md",
      "action": "replace",
      "old": "旧内容",
      "new": "新内容"
    }
  ]
}
```

### 安全边界

LEARN 只能写入 `workspaces/` 和 `experiences/` 路径，防止越权操作：

```python
def _apply_memory_update(update, knowledge):
    path = update.get("path", "")
    if not (path.startswith("workspaces/") or path.startswith("experiences/")):
        return  # 拒绝
```

---

## ANALYZE 如何使用 Memory

ANALYZE 阶段的 system prompt 中包含 memory 使用指引：

1. **优先读取 experience**：在形成假设之前，先 `memory_read(experience)` 查看历史反模式
2. **使用 overview 缩小范围**：warmup 生成的 overview 包含入口文件、调用关系，直接使用
3. **不要重复确认**：只有当 memory 中缺少信息时，才补做 scanner 扫描

### 预注入到 Prompt

WARMUP 生成的 overview 和全局 experience 会在 ANALYZE 启动前预注入到 user prompt 中，减少 LLM 的工具调用次数：

```python
# analyze.py 中的 _build_global_brief()
brief_parts = []

# 注入 workspace overview
overview = knowledge.read_page(f"workspaces/{ns}/overview.md")
if overview:
    brief_parts.append(f"## Workspace Overview\n{overview}")

# 注入 global experience
exp = knowledge.read_page("experiences/experience.md")
if exp:
    brief_parts.append(f"## Global Experience\n{exp[:2000]}")
```

---

## RunLedger — 运行记录

RunLedger 记录每次 Pipeline 运行的元信息，用于追踪和回溯：

```python
class RunLedger:
    def init() -> None
    def record_session(run_id, memory_namespace, ...) -> None
    def recent_session(run_id) -> dict | None
```

记录内容：
- run_id
- memory_namespace（对应的 wiki namespace）
- 时间戳
- 阶段完成状态

---

## 设计决策

### 为什么用文件而非数据库

1. **可读性**：Markdown 文件可以直接查看、编辑、diff
2. **可版本控制**：wiki 内容可以纳入 git
3. **简单**：不需要额外的数据库依赖
4. **LLM 友好**：Markdown 是 LLM 最擅长处理的格式

### 为什么分离 workspace 和 global experience

- **Workspace experience**：特定仓库的优化经验（如"这个项目的 DataLoader 配置在 configs/train.yaml 第 42 行"）
- **Global experience**：跨仓库通用的优化模式（如"Python 循环内逐 token 调用 embedding 会触发大量微 kernel"）

分离后，新仓库可以从 global experience 受益，而不会被其他仓库的特定信息干扰。

### 为什么 Parent-only writes

子 Agent（ANALYZE、OPTIMIZE）只读 memory，只有 LEARN 阶段可以写入。这保证了：
- 知识质量：只有经过 LEARN 阶段归纳的知识才会被写入
- 安全性：子 Agent 不会意外覆盖或污染知识库
- 可审计：所有写入都有明确的来源（LEARN 阶段的输出）