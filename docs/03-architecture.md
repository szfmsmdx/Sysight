# 整体架构

## 设计哲学

Sysight 的设计遵循三个核心原则：

1. **硬编码 Pipeline，而非 LLM 自主决策** — 阶段顺序是固定的，LLM 只在每个阶段内部做它擅长的事（分析、评判、生成），不负责编排
2. **确定性验证优先** — 所有 LLM 输出都经过代码侧校验（hash 验证、smoke test、timer 对比），不信任 LLM 的"自我评估"
3. **领域知识内化** — C1-C7 分类体系、nsys SQL 查询、cuda_timer 注入都是硬编码的领域知识，不依赖 LLM 的 GPU 知识

---

## 系统分层

```
┌─────────────────────────────────────────────────┐
│                    CLI 层                        │
│  sysight warmup / analyze / optimize / full     │
├─────────────────────────────────────────────────┤
│                 Pipeline 层                      │
│  PipelineRunner → 7 阶段编排                     │
├─────────────────────────────────────────────────┤
│                 Agent 层                         │
│  AgentLoop → 多轮工具调用                        │
│  AgentContext → 渐进式上下文压缩                  │
├─────────────────────────────────────────────────┤
│                 工具层                           │
│  scanner / nsys_sql / memory / sandbox / patcher │
├─────────────────────────────────────────────────┤
│                 存储层                           │
│  WikiRepository / RunLedger / 文件系统            │
└─────────────────────────────────────────────────┘
```

---

## Pipeline 总览

Sysight 的核心是一个 **7 阶段 Pipeline**：

```
WARMUP → ANALYZE → INSTRUMENT → LEARN → OPTIMIZE → EXECUTE → LEARN
  │         │          │          │         │          │         │
  │         │          │          │         │          │         │
  ▼         ▼          ▼          ▼         ▼          ▼         ▼
仓库扫描  性能分析   计时器注入  知识积累  代码生成   应用验证   知识积累
(确定性)  (LLM)     (LLM+代码) (LLM)    (LLM)     (确定性)   (LLM)
```

### 阶段职责

| 阶段 | 类型 | 输入 | 输出 | LLM |
|------|------|------|------|-----|
| **WARMUP** | 确定性 | 代码仓库 | RepoSetup（入口、配置、调用关系） | ❌ |
| **ANALYZE** | LLM | nsys profile + 仓库 | LocalizedFindingSet（文件:函数:行号） | ✅ |
| **INSTRUMENT** | LLM+代码 | FindingSet + 仓库 | 注入 cuda_timer 的源码 | ✅ |
| **LEARN** | LLM | FindingSet + PatchSet | wiki 更新 | ✅ |
| **OPTIMIZE** | LLM | FindingSet + 仓库 | PatchCandidate[]（计划，不修改文件） | ✅ |
| **EXECUTE** | 确定性 | PatchCandidate[] + 仓库 | PatchResult[] + timer 对比 | ❌ |
| **LEARN** | LLM | FindingSet + PatchResult[] | wiki 更新 | ✅ |

### 为什么这样设计

- **WARMUP 不用 LLM**：仓库扫描是纯工程问题（找入口文件、解析配置、构建调用图），LLM 做这个又慢又贵又不准
- **ANALYZE 用 LLM**：从 profile 数据推断性能问题需要推理能力，这是 LLM 的强项
- **OPTIMIZE 用 LLM**：评判 finding 真伪、生成精确 patch 需要理解代码语义
- **EXECUTE 不用 LLM**：apply patch、smoke test、timer 对比都是确定性操作，不需要推理
- **LEARN 用 LLM**：从分析结果中提取可复用的知识需要归纳能力

---

## 阶段间通信

### 通信方式：内存对象 + JSON 持久化

```
┌──────────┐  RepoSetup   ┌──────────┐  FindingSet  ┌──────────┐
│  WARMUP  │─────────────▶│ ANALYZE  │─────────────▶│INSTRUMENT│
└──────────┘              └──────────┘              └──────────┘
                                │                        │
                                │ analyze_raw.json       │ instrument_result.json
                                ▼                        ▼
                           .sysight/                .sysight/
                           analysis-runs/           analysis-runs/
                                │                        │
                                ▼                        ▼
┌──────────┐  FindingSet   ┌──────────┐  PatchCandidate[]  ┌──────────┐
│  LEARN   │◀─────────────│ OPTIMIZE │───────────────────▶│ EXECUTE  │
└──────────┘              └──────────┘                    └──────────┘
      │                        │                               │
      │ wiki 更新              │ optimize_result.json          │ execute_result.json
      ▼                        ▼                               ▼
  .sysight/                .sysight/                       .sysight/
  memory/wiki/             optimizer-runs/                 execute-runs/
```

### 设计决策

**为什么用内存对象传递而非纯 JSON？**

Pipeline 内部阶段之间通过 Python 对象（`LocalizedFindingSet`、`PatchCandidate[]`）传递数据。这避免了序列化/反序列化开销，且类型安全。

**为什么还要写 JSON 文件？**

1. **持久化**：分析结果可以跨 session 复用（`sysight optimize run-f15b4613`）
2. **可审计**：每个阶段的输出都有文件记录，方便 debug
3. **解耦**：阶段可以独立运行，不依赖前序阶段的内存状态

**为什么不在 Prompt 中传完整 JSON？**

ANALYZE 输出的 findings JSON 可能很大（24 个 finding，每个含描述、建议、证据），直接塞进 OPTIMIZE 的 prompt 会消耗大量 token。实际做法是只传 optimizer 需要的字段：

```python
# 只传 optimizer 需要的字段
findings_data = [{
    "finding_id": f.finding_id,
    "title": f.title,
    "file_path": f.file_path,
    "function": f.function,
    "line": f.line,
    "description": f.description,
    "suggestion": f.suggestion,
} for f in findings.findings if f.status == "accepted"]
```

### 与主流 Agent 框架的对比

| 框架 | 通信方式 | Sysight 的做法 |
|------|---------|---------------|
| LangGraph | Shared State（TypedDict） | 内存 dataclass 对象 |
| AutoGen | Message Passing（对话） | 结构化 artifact 传递 |
| CrewAI | Task Output → Next Task Input | Pipeline 硬编码顺序 |
| **Sysight** | **内存对象 + JSON 快照** | **两者结合** |

Sysight 不采用 Message Passing 的原因：阶段之间传递的是结构化数据（findings、patches），不是自然语言对话。用 dataclass 比用消息更类型安全、更省 token。

---

## 目录结构

```
sysight/
├── agent/              # Agent 层
│   ├── loop.py         # AgentLoop — 多轮工具调用循环
│   ├── context.py      # AgentContext — 渐进式上下文压缩
│   ├── provider.py     # LLMProvider 抽象
│   ├── providers/      # 具体 provider 实现
│   └── prompts/        # 系统提示词
├── pipeline/           # Pipeline 层
│   ├── runner.py       # PipelineRunner — 7 阶段编排
│   ├── warmup.py       # WARMUP 阶段
│   ├── analyze.py      # ANALYZE 阶段
│   ├── instrument.py   # INSTRUMENT 阶段
│   ├── optimize.py     # OPTIMIZE 阶段
│   ├── execute.py      # EXECUTE 阶段
│   └── learn.py        # LEARN 阶段
├── tools/              # 工具层
│   ├── registry.py     # ToolRegistry — 工具注册与执行
│   ├── scanner/        # 源码扫描工具
│   ├── nsys_sql/       # nsys profile SQL 查询工具
│   ├── memory/         # wiki 读写工具
│   ├── sandbox/        # 沙箱执行工具
│   └── patcher.py      # 确定性 patch 应用
├── types/              # 数据类型
│   ├── findings.py     # LocalizedFinding, LocalizedFindingSet
│   ├── optimization.py # PatchCandidate, PatchResult
│   └── repo_setup.py   # RepoSetup, ExecutionConfig
├── wiki/               # 知识库
│   ├── store.py        # WikiRepository — CRUD
│   ├── ledger.py       # RunLedger — 运行记录
│   └── ...
├── benchmark/          # 基准测试
│   ├── runner.py       # BenchmarkRunner（分析能力）
│   └── optimizer_runner.py  # OptimizerBenchmarkRunner（优化能力）
├── cli/                # CLI 入口
│   └── cli.py
└── utils/              # 工具函数
    ├── cuda_timer.py   # GPU/CPU 计时器
    └── ...
```