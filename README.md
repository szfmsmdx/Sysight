# Sysight

给一份 nsys profile 和一个代码仓库，自动完成"找问题 → 定位源码 → 生成 patch → 沙箱验证 → 积累经验"的完整闭环，无需人工介入。

在 nanoGPT Shakespeare 字符级训练任务上实测：定位到三处 CPU 端瓶颈，生成 4 个 patch，两轮迭代将单步迭代时间从 **14.12ms 压缩到 12.96ms（提升 8.2%）**。完整记录见 [docs/summary.md](docs/summary.md)。

---

## 工作流

```
nsys profile (.sqlite)  +  代码仓库
            │
            ▼
   ┌─────────────────┐
   │    WARMUP       │  扫描仓库：入口命令、热路径、配置
   └────────┬────────┘
            │ RepoSetup
            ▼
   ┌─────────────────┐
   │    ANALYZE      │  LLM + nsys_sql 工具 → LocalizedFindingSet
   └────────┬────────┘    （文件 : 函数 : 行号 + 证据链）
            │
            ▼
   ┌─────────────────┐
   │    OPTIMIZE     │  LLM 读源码、判断真伪、生成 patch 计划
   └────────┬────────┘
            │ PatchCandidate[]
            ▼
   ┌─────────────────┐
   │    EXECUTE      │  git worktree 沙箱 → apply → smoke test → 测量
   └────────┬────────┘    delta > 0 则 accept，否则 revert
            │
            ▼
   ┌─────────────────┐
   │     LEARN       │  LLM 将 finding + patch 结果写入 wiki
   └─────────────────┘    （下一轮自动参考）
```

Pipeline 硬编码：LLM 只在推理环节参与（analyze、plan、learn），文件扫描、patch apply、性能测量全由代码执行。

---

## 能力

| 能力 | 说明 |
|---|---|
| **Profile 分析** | 自动查询 nsys SQLite，C1–C7 七类瓶颈分类，输出带文件行号的 finding |
| **源码定位** | 每个 finding 精确到 `文件:函数:行号`，附 profile 证据链 |
| **Patch 生成** | LLM 判断真伪，对确认问题生成最小化 patch |
| **沙箱验证** | 独立 git worktree apply → smoke test → 计时对比 → 接受或回滚 |
| **知识积累** | 每轮结果写入 wiki，下一轮自动注入，不重复踩坑 |
| **基准测试** | 6 个分析 case + 6 个优化 case，量化能力评估 |

---

## 七类性能问题（C1–C7）

| 类别 | 名称 | 典型场景 |
|---|---|---|
| **C1** | Host Scheduling | `DataLoader num_workers=0`、`pin_memory=False` |
| **C2** | Kernel Launch Overhead | Python 循环触发大量小 CUDA kernel |
| **C3** | Synchronization | `.item()`、`.cpu()`、`cudaDeviceSynchronize()` 阻塞 |
| **C4** | Memory Copy | 热路径中的 `.to(device)`、多余的 H2D/D2H |
| **C5** | Compute Inefficiency | 重复计算、可消除的 `clone`/`cat`/`contiguous` |
| **C6** | Communication | all_reduce/all_gather 配置不当，DDP/FSDP overlap 不足 |
| **C7** | Python Pipeline | DataLoader 内逐 sample 循环、热路径 JSON 序列化 |

---

## 安装

```bash
git clone <repo-url> && cd Sysight
pip install -e .
```

需要 Python ≥ 3.10。

---

## 配置

创建 `.sysight/config.yaml`：

```yaml
analyze:
  provider: openai_compatible
  model: "deepseek-v3"
  api_key: "$DEEPSEEK_API_KEY"        # 支持 $ENV_VAR 引用
  base_url: "https://api.deepseek.com/v1"
  temperature: 0

optimize:
  provider: anthropic
  model: "claude-sonnet-4-5"
  api_key: "$ANTHROPIC_API_KEY"
```

支持的 provider：`openai_compatible`（OpenAI、DeepSeek、Groq、vLLM 等）、`anthropic`。

---

## 快速上手

```bash
# 完整 pipeline（多轮迭代）
sysight agent-loop --repo ./my-training-repo --profile profiles/trace.sqlite

# 分步运行
sysight warmup ./my-training-repo
sysight analyze profiles/trace.sqlite --repo ./my-training-repo
sysight optimize <run_id> --repo ./my-training-repo

# 单工具调用
sysight tool scanner_files ./my-training-repo
sysight tool nsys_sql_kernels profiles/trace.sqlite --limit 20
```

---

## 项目结构

```
sysight/
├── types/          共享 dataclass（零依赖，所有模块共用）
├── tools/          工具注册中心 + 所有 LLM 可调用工具
│   ├── scanner/    源码工具：files、search、read、symbols、callers
│   ├── nsys_sql/   nsys SQLite 工具：kernels、sync、memcpy、nccl、overlap、gaps
│   ├── sandbox/    git worktree 隔离：create、apply、validate、measure、commit
│   └── memory/     wiki 只读工具（子 agent 只读）
├── wiki/           知识管理：store、FTS 索引、ledger、promotion
├── agent/          LLM 抽象层：providers、AgentLoop、上下文压缩
├── pipeline/       阶段编排：warmup、analyze、optimize、execute、learn
├── benchmark/      Analyze Benchmark + Optimize Benchmark runner
├── cli/            统一 CLI 入口 + MCP server 适配器
└── utils/          安全、repo 工具、渲染、文本

test/               单元测试 + 集成测试
workspace/          目标仓库挂载目录
docs/               设计文档
```

---

## 依赖层级

```
types/        ← 零依赖
  ↑
tools/        ← 依赖 types/
  ↑
wiki/         ← 依赖 types/ + tools/memory
  ↑
agent/        ← 依赖 types/ + tools/
  ↑
pipeline/     ← 依赖以上全部
  ↑
cli/          ← 依赖 pipeline/
```

内层不 import 外层，跨层依赖通过 Protocol 注入。

---

## 测试

```bash
# 单元测试（无网络，< 1s）
pytest test/test_types test/test_tools test/test_wiki -q

# Pipeline + Agent 测试
pytest test/test_pipeline test/test_agent -q

# 全量
pytest -q

# HTTP 集成测试（需要在 .sysight/config.yaml 配置 API key）
pytest test/test_agent/test_agent_loop.py::TestAgentLoopIntegration -v
```

---

## 文档

| 文件 | 内容 |
|---|---|
| [docs/01-overview.md](docs/01-overview.md) | 项目总览与文档导航 |
| [docs/02-pipeline.md](docs/02-pipeline.md) | Pipeline 设计：5 阶段编排与数据流 |
| [docs/03-analyzer.md](docs/03-analyzer.md) | Analyzer：profile 引擎 + scanner |
| [docs/04-optimizer.md](docs/04-optimizer.md) | Optimizer：代码生成、沙箱验证与 patch 应用 |
| [docs/05-agent-context.md](docs/05-agent-context.md) | Agent 层：AgentLoop 与渐进式上下文压缩 |
| [docs/06-wiki-memory.md](docs/06-wiki-memory.md) | Wiki Memory：存储设计与 LEARN 阶段 |
| [docs/07-benchmark.md](docs/07-benchmark.md) | 基准测试：分析 bench + 优化 bench |
| [docs/summary.md](docs/summary.md) | Demo 实录：nanoGPT 两轮 pipeline 完整记录 |

---

## 致谢

- [nsys-ai](https://github.com/GindaChen/nsys-ai) — nsys SQLite 分析思路，本项目的重要灵感来源
- [MiniCode](https://github.com/LiuMengxuan04/MiniCode) — agentic coding 框架参考
- [nsys-bench](https://github.com/szfmsmdx/nsys-bench) — benchmark 用例与 ground truth

---

## License

MIT
