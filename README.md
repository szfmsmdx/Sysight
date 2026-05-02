# Sysight

AI 驱动的 GPU 性能分析与优化工具。输入 Nsight Systems profile + 目标仓库源码，自动定位性能瓶颈、生成优化 patch、积累可复用的性能经验。

## 背景

GPU 性能优化有两个难点：profile 数据量巨大（一个 8-GPU training step 可能产生数十 GB 的 trace），而且从"某个 kernel 耗时长"到"改哪行代码"之间隔着很长的推理链。

Sysight 用一个确定性的分析管道预处理 profile 数据，然后把结构化的 evidence 交给 LLM 做源码级定位。分析结果不是一次性的——通过的 wiki 系统积累下来，下次分析同一个 repo 时自动注入上下文，越用越强。

核心思路来自 [self-improving-nsys-expert-blueprint.md](./self-improving-nsys-expert-blueprint.md) 和 [architecture-review-20260429.md](./architecture-review-20260429.md)。

## 工作流

```
WARMUP（每个 repo 一次）
  scanner_files → 找入口点 → 试跑 → RepoSetup → wiki

ANALYZE（每个 profile 一次）
  nsys_sql + classify → C1-C7 分类 → AgentLoop(LLM + scanner)
  → LocalizedFindingSet → 验证 → wiki + ledger

OPTIMIZE（每个 session N 次）
  逐 finding: LLM → PatchCandidate → sandbox(apply+validate+measure)
  → keep/revert → ledger

LEARN（每个 session 后）
  ledger 记账 → benchmark 评分 → worklog → 经验提取
```

四个阶段可以独立运行，也通过 `pipeline/runner.py` 串联。

## 项目结构

```
sysight/
  types/          共享数据类（零依赖，所有模块共用）
  tools/          工具注册中心 + 26 个工具实现
    registry.py   ToolRegistry / ToolDef / ToolPolicy
    scanner/      源码静态分析（files, search, read, symbols, callers, variants）
    nsys_sql/     Nsight Systems SQLite 查询（kernels, sync, memcpy, nccl, overlap, gaps, launch）
    sandbox/      Git worktree 隔离执行（create, exec, apply, validate, measure, commit, revert）
    memory/       知识库搜索（search, read——子 agent 只读）
    classify.py   C1-C7 瓶颈分类
  wiki/           知识库（workspace wiki + 经验 wiki + FTS + ledger + promotion + skills）
  agent/          LLM 抽象层
    provider.py   LLMProvider 协议 + LLMConfig
    loop.py       AgentLoop（tool-calling 循环 + stop conditions）
    prompts/      PromptLoader + 7 个 prompt fragment
    providers/    OpenAI-compatible / Anthropic / Replay 三种后端
    config_loader.py  YAML config 加载
  pipeline/       四阶段编排（warmup, analyze, optimize, learn）+ PipelineRunner
  cli/            统一 CLI 入口 + MCP server 适配器
  utils/          内部工具（security, repo, render, text）
```

## 各模块职责

### `types/`

所有模块共享的 dataclass 定义。不 import 任何 `sysight.*` 模块。

- `evidence.py` — GpuDeviceInfo, BottleneckReport, EvidenceWindow, ProfileEvidence
- `findings.py` — LocalizedFinding, LocalizedFindingSet, MemoryUpdate + `make_finding_id()`
- `optimization.py` — PatchCandidate, PatchResult, ExecutionConfig + `compute_span_hash()`
- `repo_setup.py` — RepoSetup
- `memory.py` — MemoryPage, MemoryBrief

### `tools/`

ToolRegistry 注册中心 + 所有可被 LLM 调用的工具。每个工具是纯函数 + ToolDef wrapper。

| 工具组 | 工具数 | 只读 | 功能 |
|--------|--------|------|------|
| scanner | 7 | ✓ | 源码文件列举、全文搜索、按行读取、符号解析、调用者/被调用者分析、变体查找 |
| nsys_sql | 7 | ✓ | SQLite profile 查询：kernels、sync、memcpy、nccl、overlap、gaps、launch |
| sandbox | 8 | ✗ | Git worktree 隔离：create、exec、apply（含 hash 验证）、validate、measure、commit、revert、destroy |
| memory | 2 | ✓ | 知识库搜索和读取（子 agent 只读，写入由 parent 控制） |
| classify | 1 | ✓ | 确定性 C1-C7 瓶颈分类 |

ToolPolicy 按阶段控制：ANALYZE 只能调用只读工具，OPTIMIZE 可以使用 sandbox 工具（在隔离 worktree 中操作）。

### `wiki/`

知识管理系统。Markdown 文件是 source of truth，SQLite 是 rebuildable index。

- `store.py` — WikiRepository：CRUD、YAML frontmatter、命名空间隔离、signal page
- `index.py` — FTSIndex：全文搜索（当前 grep fallback，计划升级 SQLite FTS5）
- `brief.py` — `build_memory_brief()`：生成 ≤200 行的压缩上下文注入 LLM prompt
- `ledger.py` — RunLedger：SQLite 记录 runs、findings、patches、benchmark、candidates
- `promotion.py` — CandidateValidator：5 条门控规则，benchmark scope 不能 promote 到 global
- `skills.py` — SkillRegistry：发现和加载 `.sysight/skills/<name>/SKILL.md`

写入权限由 parent 进程控制。子 agent 只能通过 `memory_search` / `memory_read` 读取。

### `agent/`

LLM 抽象层。通过 YAML config 驱动，不需要改代码切换 provider。

- `provider.py` — LLMProvider 协议 + LLMConfig/LLMRequest/LLMResponse + `create_provider()` factory
- `providers/openai_compatible.py` — 一个类处理 OpenAI / DeepSeek / Groq / vLLM 等所有 OpenAI 兼容 API
- `providers/anthropic.py` — Anthropic Messages API
- `providers/replay.py` — 测试用 replay backend
- `loop.py` — AgentLoop：多轮 tool-calling 循环，stop conditions（max_turns, max_time, repeated_calls）
- `prompts/loader.py` — PromptLoader：按 task_type 组装 fragment，benchmark hints 默认不加载
- `prompts/fragments/` — 7 个 .md fragment（common_role, evidence_sop, optimizer_sop, output_schema, safety, benchmark_hints）
- `config_loader.py` — 零依赖 YAML 解析 + `$ENV_VAR` 引用

### `pipeline/`

阶段编排。每个阶段可独立运行，通过 PipelineRunner 串联。

- `warmup.py` — `run_warmup()`：scanner 探索 repo，产出 RepoSetup，写入 workspace wiki
- `analyze.py` — `run_analyze()`：classify → AgentLoop → 验证 → wiki + ledger
- `optimize.py` — `run_optimize()`：逐 finding → LLM → sandbox → keep/revert → ledger
- `learn.py` — `run_learn()`：ledger 记账 + benchmark 评分 + 可选 LLM worklog/经验提取
- `runner.py` — `PipelineRunner`：WARMUP → ANALYZE → OPTIMIZE → LEARN

### `cli/`

- `cli.py` — 统一 CLI：`sysight warmup/analyze/optimize/learn/full/tool`
- `mcp_server.py` — MCP 适配器（只读工具 + memory read）

### `utils/`

- `security.py` — 路径包含检查、安全校验
- `repo.py` — git helpers（worktree 创建/清理、commit hash）
- `render.py` — 终端报告渲染
- `text.py` — CJK 文本格式化

## 快速开始

### 安装

```bash
pip install -e .
```

### 配置

编辑 `.sysight/config.yaml`：

```yaml
analyze:
  provider: "openai"
  model: "deepseek-v4-pro"
  api_key: "sk-..."
  base_url: "https://api.deepseek.com/v1"
  temperature: 0

optimize:
  provider: "anthropic"
  model: "claude-sonnet-4-6"
  api_key: "sk-ant-..."
```

`api_key` 支持 `$ENV_VAR` 引用。不填 `max_tokens` 则不设限。

### 使用

```bash
# 探索 repo
sysight warmup ./my-model

# 分析 profile
sysight analyze trace.sqlite --repo ./my-model

# 单工具调用
sysight tool scanner_files .
sysight tool nsys_sql_kernels trace.sqlite --limit 10
```

## 测试

```bash
# 非 HTTP 测试（<0.2s）
python3 -m unittest discover -s test/test_types -v
python3 -m unittest discover -s test/test_tools -v
python3 -m unittest discover -s test/test_wiki -v

# HTTP 集成测试（需要 .sysight/config.yaml 配好 API key）
python3 -m unittest test.test_agent.test_agent_loop.TestAgentLoopIntegration -v
```

不包含 smoke test。测试要么验证真实计算逻辑，要么通过真实 HTTP 调用验证端到端行为。

## 依赖方向

```
types/          ← 零依赖
  ↑
tools/          ← 只依赖 types/
  ↑
wiki/           ← 依赖 types/ + tools/memory
  ↑
agent/          ← 依赖 types/ + tools/
  ↑
pipeline/       ← 依赖以上全部
  ↑
cli/            ← 依赖 pipeline/
```

内层不 import 外层。跨层依赖通过 Protocol 注入。

## 相关项目

- [llmwiki](https://github.com/kennyatgithub/llmwiki) — LLM-maintained knowledge workspace，FTS5 + 引用图设计的参考来源
- [repositories-wiki](https://github.com/szfmsmdx/repositories-wiki) — 从源码自动生成 wiki，update-wiki skill 的参考来源
- [nsys-bench](https://github.com/szfmsmdx/nsys-bench) — benchmark 用例和 ground truth
- [self-improving-nsys-expert-blueprint.md](./self-improving-nsys-expert-blueprint.md) — 系统设计蓝图
- [architecture-review-20260429.md](./architecture-review-20260429.md) — 架构审查文档
