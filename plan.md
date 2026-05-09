# Sysight Pipeline 架构与优化方案

## 一、总体架构

Sysight 是一个**专业 GPU 性能优化 agent**，采用硬编码的 staged pipeline 架构。各阶段通过 Python dataclass 在内存中传递数据，JSON 文件作为持久化缓存和调试快照。

```
WARMUP ──→ ANALYZE ──→ INSTRUMENT ──→ LEARN(1) ──→ OPTIMIZE ──→ EXECUTE ──→ LEARN(2)
   │           │            │              │             │            │
   ▼           ▼            ▼              ▼             ▼            ▼
RepoSetup   FindingSet  TimerSpec[]    findings_json  Patch[]    ExecuteResult
  JSON       对象+JSON    +JSON           (prompt内嵌)  对象+JSON    +JSON
```

### 各阶段职责

| 阶段 | 职责 | 输入 | 输出 | LLM 参与 |
|------|------|------|------|---------|
| **WARMUP** | 环境探查：入口命令、测试命令、代码地图 | `repo: str` | `RepoSetup` (JSON 缓存) | ❌ 无 |
| **ANALYZE** | Profile 分析：AgentLoop → 性能发现 | `profile.sqlite`, `RepoSetup` | `LocalizedFindingSet` + `analyze_raw.json` | ✅ AgentLoop (≤50 turns) |
| **INSTRUMENT** | 计时器插桩：LLM 决定埋点位置 → 程序化注入 | `LocalizedFindingSet` | `TimerSpec[]` + `instrument_result.json` | ✅ AgentLoop (≤10 turns) |
| **LEARN(1)** | 知识学习：从 findings 中提取经验写入 wiki | `findings_json` | `LearnResult` | ✅ AgentLoop (≤10 turns) |
| **OPTIMIZE** | 补丁生成：AgentLoop 评估 findings → 生成 PatchCandidate[] | `LocalizedFindingSet` | `PatchCandidate[]` + `optimize_result.json` | ✅ AgentLoop (≤20 turns) |
| **EXECUTE** | 补丁应用：apply → smoke test → timer 对比 | `PatchCandidate[]`, `TimerSpec[]` | `ExecuteResult` + `execute_result.json` | ❌ 无（确定性） |
| **LEARN(2)** | 知识学习：从 findings + patches 中提取经验 | `findings_json`, `patches_json` | `LearnResult` | ✅ AgentLoop (≤10 turns) |

### 关键设计原则

1. **Pipeline 硬编码**：专业 agent 的 workflow 是确定性的，LLM 只在关键决策点介入（analyze 的 SQL 查询、instrument 的埋点决策、optimize 的补丁生成）。
2. **领域压缩表示**：不传递原始数据，而是传递领域特定的摘要（profile brief、finding 摘要）。
3. **结构化验证门**：每个阶段输出经过领域验证器（文件存在性、smoke test、timer 对比）后才进入下一阶段。
4. **角色分解**：不同阶段使用不同的 system prompt 和工具集（analyze 用 `nsys_sql_*`，optimize 用 `scanner_*`，learn 用 `memory_*`）。

---

## 二、阶段间通信机制

### 2.1 当前设计

阶段间通信采用**双重模式**：

1. **内存传递**（主要）：Python dataclass 对象在 `PipelineRunner.run_full()` 中直接传递
   - `analyze_result.finding_set` → instrument / optimize / learn
   - `patches` → execute / learn

2. **JSON 持久化**（辅助）：各阶段将结果写入 JSON 文件，用于：
   - 调试快照（`analyze_raw.json`, `optimize_result.json`）
   - 跨进程读取（`instrument_result.json` → execute）
   - 断点恢复（warmup cache）

### 2.2 调研结论

通过对 LangGraph、AutoGen、CrewAI、Smolagents、SWE-agent、OpenAI Agents SDK、DSPy 等主流框架的调研，以及对 Devin、Claude Code、Aider、Cursor、CudaForge 等专业 agent 的分析，得出以下结论：

**Sysight 的设计整体合理**：
- 作为专业 agent，硬编码 staged pipeline + dataclass 传递是正确的选择，与 SWE-agent、Aider、CudaForge 等专业 agent 一致。
- 不需要引入 LangGraph、AutoGen 等重量级框架——`PipelineRunner` + dataclass 已经足够清晰。
- JSON 文件作为持久化存储没有问题，与内存 dataclass 传递是两条互补的路径，不冲突。

**核心优化方向**：

| 优先级 | 优化项 | 收益 | 说明 |
|--------|--------|------|------|
| **P0** | LLM 上下文格式优化 | 高 | 将 prompt 中的 JSON 替换为 Markdown 表格/摘要 |
| **P0** | 文件耦合解耦 | 中 | instrument → execute 改为内存传递 TimerSpec[] |
| **P1** | 统一状态对象 | 高 | 定义 `PipelineState` 消除 `__dict__` 类型丢失 |
| **P2** | 领域压缩表示 | 中 | 为 optimize 生成 finding 摘要卡 |
| **P3** | 验证门模式 | 中 | 统一各阶段的输出验证抽象 |

### 2.3 方案一：LLM 上下文格式优化（P0）

**问题**：`_serialize_findings()` 和 instrument/optimize 阶段将 findings 以原始 JSON 嵌入 LLM prompt，存在以下问题：
- Token 消耗大（JSON 的 `"key": "value"` 格式冗余度高）
- LLM 解析困难（嵌套 JSON 在文本中容易出错）
- 缺乏对 LLM 友好的格式化

**方案**：为 LLM 生成 Markdown 表格/摘要，JSON 文件仅用于程序化读取。

```python
def _format_findings_for_llm(findings: list[LocalizedFinding]) -> str:
    """生成 LLM 友好的 findings 表格。"""
    lines = [
        "## 性能分析发现",
        "",
        "| ID | 类别 | 优先级 | 文件:行 | 函数 | 指标 | 建议 |",
        "|----|------|--------|---------|------|------|------|",
    ]
    for f in findings:
        lines.append(
            f"| {f.finding_id} | {f.category} | {f.priority} | "
            f"`{f.file_path}:{f.line}` | {f.function or '-'} | "
            f"{f.metric} | {f.suggestion[:60]} |"
        )
    return "\n".join(lines)
```

**影响范围**：
- `sysight/pipeline/runner.py`：`_serialize_findings()` 保留（用于 JSON 持久化），新增 `_format_findings_for_llm()` 用于 prompt
- `sysight/pipeline/instrument.py`：`_llm_infer_timer_specs()` 中的 `findings_json` 替换为表格格式
- `sysight/pipeline/optimize.py`：`_run_optimize_loop()` 中的 `findings_json` 替换为表格格式
- `sysight/pipeline/learn.py`：`run_learn()` 中的 `findings_json` 替换为表格格式

**预期收益**：
- Token 消耗减少 50-60%（5 findings: JSON ~800 tokens → 表格 ~300 tokens）
- LLM 输出质量提升（表格比 JSON 更易理解）

### 2.4 方案二：文件耦合解耦（P0）

**问题**：`instrument_result.json` 作为 instrument → execute 的唯一桥梁，execute 需要通过 `analyze_run_dir` 定位文件。

**方案**：在 `PipelineRunner` 中直接传递 `TimerSpec[]` 对象，JSON 文件降级为可选的调试快照。

```python
# 当前（文件耦合）
def run_execute(self, patches, repo, run_id, analyze_run_dir):
    ...

# 优化后（内存传递）
def run_execute(self, patches, repo, run_id, timer_specs=None, analyze_run_dir=None):
    ...
```

**影响范围**：
- `sysight/pipeline/runner.py`：`run_full()` 中传递 `instrument_result.timers`
- `sysight/pipeline/execute.py`：`run_execute()` 新增 `timer_specs` 参数

### 2.5 方案三：统一状态对象（P1）

**问题**：`PipelineResult` 使用 `f.__dict__` 丢失类型信息，各阶段返回值类型不统一。

**方案**：定义 `PipelineState` dataclass 作为各阶段的统一读写接口。

```python
@dataclass
class PipelineState:
    run_id: str
    repo_root: Path
    profile_path: Path
    
    # 各阶段输出
    repo_setup: RepoSetup | None = None
    finding_set: LocalizedFindingSet | None = None
    timer_specs: list[TimerSpec] = field(default_factory=list)
    patches: list[PatchCandidate] = field(default_factory=list)
    execute_result: ExecuteResult | None = None
    
    # 元数据
    errors: list[str] = field(default_factory=list)
    stages_completed: list[str] = field(default_factory=list)
```

**影响范围**：
- 新增 `sysight/types/pipeline_state.py`
- `sysight/pipeline/runner.py`：`run_full()` 使用 `PipelineState` 替代分散的局部变量
- 各阶段函数签名可能需要调整

---

## 三、各阶段详细设计

### 3.1 WARMUP

**当前状态**：已完成，无需修改。

- 确定性扫描仓库：入口命令、测试命令、代码地图、import 链
- 输出 `RepoSetup` 对象，持久化到 `.sysight/warmup-caches/<case>-<hash8>.json`
- 无 LLM 参与

### 3.2 ANALYZE

**当前状态**：已完成重构。

- 输入：`profile.sqlite` + `RepoSetup`（来自 warmup 缓存）
- 流程：
  1. `_build_global_brief()` 生成 profile 统计报告（Markdown）
  2. `_build_memory_refs()` 构建 wiki 引用
  3. `AgentLoop` 运行（≤50 turns），LLM 通过 `nsys_sql_*` 工具查询 profile
  4. `_parse_finding_set()` 解析 LLM 输出的 JSON findings
  5. `_validate_findings()` 验证文件存在性和路径安全性
- 输出：`LocalizedFindingSet` + `analyze_raw.json` + `debug.log`
- 工具集：`nsys_sql_*`, `scanner_*`, `memory_search`, `memory_read`

**数据结构**：

```python
@dataclass
class LocalizedFinding:
    finding_id: str          # "{category}:{hash8}"
    category: str            # C1-C7
    title: str
    priority: Literal["high", "medium", "low"]
    confidence: Literal["confirmed", "probable", "unresolved"]
    evidence_refs: list[str]
    metric: str              # 量化指标（如 "12.5ms", "843 calls"）
    file_path: str | None
    function: str | None
    line: int | None
    description: str
    suggestion: str          # 方向性建议
    status: Literal["accepted", "rejected", "unresolved"]
    reject_reason: str
```

### 3.3 INSTRUMENT

**当前状态**：已完成 LLM 驱动模式。

- 输入：`LocalizedFindingSet`（来自 analyze）
- 流程：
  1. `_llm_infer_timer_specs()`：AgentLoop (≤10 turns)，LLM 读取源码确定埋点位置
  2. `_insert_timers()`：程序化注入 `cuda_timer` 到源文件
  3. `_write_result_json()`：输出 `instrument_result.json`
- 输出：`TimerSpec[]` + `instrument_result.json`
- 工具集：`scanner_read`, `scanner_search`, `scanner_files`

**TimerSpec 结构**：

```python
@dataclass
class TimerSpec:
    finding_id: str
    timer_label: str       # "F01_data_loader_iter"
    file: str
    wrap_start: int        # 1-based
    wrap_end: int          # 1-based, inclusive
    reason: str
```

**计时工具**：`cuda_timer`（`sysight/utils/cuda_timer.py`）
- 基于 `torch.cuda.Event`，CPU fallback 使用 `time.perf_counter()`
- Context-manager API：`with cuda_timer("label")(): ...`
- 输出格式：`[SYSIGHT_TIMER] label: X.XXX ms`

### 3.4 LEARN

**当前状态**：已完成。

- 输入：`findings_json` + `patches_json`（JSON 字符串嵌入 prompt）
- 流程：AgentLoop (≤10 turns)，LLM 读取 wiki → 对比 findings/patches → 写入新知识
- 输出：`LearnResult`
- 工具集：`memory_read`, `memory_write`, `memory_search`

**调用时机**：
- LEARN(1)：analyze 完成后，从 findings 中学习
- LEARN(2)：execute 完成后，从 findings + patches 中学习

### 3.5 OPTIMIZE

**当前状态**：已完成。

- 输入：`LocalizedFindingSet`（来自 analyze）
- 流程：
  1. **Phase 1: Plan** — AgentLoop (≤20 turns)，LLM 评估 findings、读取源码、生成 PatchCandidate[]
  2. **Phase 2: Fill hashes** — 代码端计算 `old_span_hash`（确定性，不依赖 LLM）
- 输出：`PatchCandidate[]` + `optimize_result.json`
- 工具集：`scanner_read`, `scanner_search`, `scanner_files`
- **注意**：OPTIMIZE 只生成补丁计划，不修改源文件。修改由 EXECUTE 阶段执行。

**PatchCandidate 结构**：

```python
@dataclass
class PatchCandidate:
    patch_id: str
    finding_ids: list[str]
    file_path: str
    old_span_start: int
    old_span_end: int
    old_span_hash: str      # Phase 2 由代码端计算
    replacement: str
    rationale: str
    validation_commands: list[list[str]]
```

### 3.6 EXECUTE

**当前状态**：已完成。

- 输入：`PatchCandidate[]` + `TimerSpec[]`（来自 instrument_result.json）
- 流程：
  1. **Phase 0: Baseline** — 优先读取 `timer_before.json`，fallback 到 live measurement
  2. **Phase 1: Apply** — 通过 `PatchApplier` 程序化 apply 所有补丁（bottom-up 排序）
  3. **Phase 2: Verify** — smoke test + timer 对比，失败自动 revert
- 输出：`ExecuteResult` + `execute_result.json`
- 无 LLM 参与（完全确定性）

**PatchApplier**（`sysight/tools/patcher.py`）：
- `apply(file_path, old_span_start, old_span_end, old_span_hash, replacement)` — 行级 patch
- 通过 SHA1 hash 校验旧代码，防止误改
- 保存原始内容快照，支持 `revert_all()` 回退

**验证层级**：
- Level 1（始终执行）：smoke test（validation_commands + import 检查 + test_commands）
- Level 2（有 timer 数据时）：运行程序 → 解析 `[SYSIGHT_TIMER]` → 计算 delta_pct

---

## 四、文件结构

```
sysight/
├── pipeline/
│   ├── runner.py          # PipelineRunner — 编排所有阶段
│   ├── warmup.py          # WARMUP — 环境探查
│   ├── analyze.py         # ANALYZE — profile 分析
│   ├── instrument.py      # INSTRUMENT — 计时器插桩
│   ├── optimize.py        # OPTIMIZE — 补丁生成
│   ├── execute.py         # EXECUTE — 补丁应用与验证
│   └── learn.py           # LEARN — 知识学习
├── agent/
│   ├── loop.py            # AgentLoop — 多轮工具调用循环
│   ├── context.py         # AgentContext — 渐进式上下文压缩
│   ├── provider.py        # LLMProvider — LLM 调用抽象
│   └── prompts/
│       ├── loader.py      # PromptLoader — prompt 加载与构建
│       └── fragments/
│           ├── analyze_system.md
│           ├── instrument_system.md
│           ├── optimize_system.md
│           ├── learn_system.md
│           └── benchmark_hints.md
├── tools/
│   ├── registry.py        # ToolRegistry — 工具注册与执行
│   ├── patcher.py         # PatchApplier — 确定性行级 patch
│   ├── shell.py           # Shell 工具
│   └── classify.py        # 分类工具
├── types/
│   ├── findings.py        # LocalizedFinding, LocalizedFindingSet
│   ├── optimization.py    # PatchCandidate, PatchResult
│   ├── repo_setup.py      # RepoSetup
│   ├── evidence.py        # 证据类型
│   └── memory.py          # 记忆类型
└── utils/
    ├── cuda_timer.py      # cuda_timer 计时工具
    ├── repo.py            # 仓库工具
    ├── text.py            # 文本工具
    ├── security.py        # 安全工具
    └── render.py          # 渲染工具
```

### 持久化文件

```
.sysight/
├── warmup-caches/<case>-<hash8>.json     # RepoSetup 缓存
├── analysis-runs/<run_id>/
│   ├── analyze_raw.json                  # 分析结果
│   ├── debug.log                         # analyze LLM I/O 日志
│   ├── instrument_result.json            # 插桩结果
│   └── timer_before.json                 # 基准计时（可选）
├── optimizer-runs/<run_id>/
│   ├── optimize_result.json              # 优化补丁
│   └── optimize_debug.log                # optimize LLM I/O 日志
├── execute-runs/<run_id>/
│   └── execute_result.json               # 执行结果
└── memory/wiki/                          # 知识库
```

---

## 五、已完成事项

### 5.1 Pipeline 重构

- ✅ WARMUP 阶段精简：移除未使用参数、优化循环、重构 `_build_overview`
- ✅ ANALYZE 阶段重构：拆分 `_build_global_brief`、提取 `_validate_findings` 辅助函数
- ✅ INSTRUMENT 阶段：LLM 驱动模式 + 程序化注入
- ✅ OPTIMIZE 阶段：Plan-Execute-Verify 架构
- ✅ EXECUTE 阶段：PatchApplier + smoke test + timer 对比
- ✅ LEARN 阶段：AgentLoop wiki 学习

### 5.2 工具与基础设施

- ✅ `cuda_timer`：GPU/CPU 双模式计时工具
- ✅ `PatchApplier`：确定性行级 patch apply/revert
- ✅ `AgentContext`：渐进式上下文压缩（Microcompact → Snip → Pressure）
- ✅ `DebugProvider`：LLM I/O 日志记录
- ✅ `RunLedger`：运行记录与 findings 追踪

### 5.3 Benchmark 支持

- ✅ CPU fallback timer（Mac 环境可运行）
- ✅ optimizer-bench 自动合成 exec_config
- ✅ optimizer-bench 源码自动快照/恢复
- ✅ timer_before.json 预构建支持

---

## 六、待实施优化

### 6.1 P0：LLM 上下文格式优化

将 prompt 中的 findings JSON 替换为 Markdown 表格格式。

**改动文件**：
- `sysight/pipeline/runner.py`：新增 `_format_findings_for_llm()`
- `sysight/pipeline/instrument.py`：`_llm_infer_timer_specs()` 使用表格格式
- `sysight/pipeline/optimize.py`：`_run_optimize_loop()` 使用表格格式
- `sysight/pipeline/learn.py`：`run_learn()` 使用表格格式

### 6.2 P0：文件耦合解耦

instrument → execute 改为内存传递 `TimerSpec[]`。

**改动文件**：
- `sysight/pipeline/runner.py`：`run_full()` 传递 `instrument_result.timers`
- `sysight/pipeline/execute.py`：`run_execute()` 新增 `timer_specs` 参数

### 6.3 P1：统一状态对象

定义 `PipelineState` dataclass 消除 `__dict__` 类型丢失。

**改动文件**：
- 新增 `sysight/types/pipeline_state.py`
- `sysight/pipeline/runner.py`：使用 `PipelineState` 替代分散变量

### 6.4 P2：领域压缩表示

为 optimize 阶段生成 finding 摘要卡。

**改动文件**：
- `sysight/pipeline/optimize.py`：新增 `_build_finding_brief()`

### 6.5 P3：验证门模式

统一各阶段的输出验证抽象。

**改动文件**：
- 新增 `sysight/pipeline/gates.py`
- 各阶段集成验证门调用

---

## 七、设计决策记录

1. **不引入 LangGraph/AutoGen 等框架**：Sysight 的 `PipelineRunner` + dataclass 已足够清晰，引入重量级框架增加复杂度但收益有限。

2. **JSON 文件保留**：作为持久化缓存和调试快照仍有价值，与内存 dataclass 传递互补。

3. **Pipeline 硬编码**：专业 agent 的 workflow 应确定性执行，LLM 只在关键决策点介入。

4. **角色分解**：不同阶段使用不同的 system prompt 和工具集，而非单一通用 agent。

5. **验证确定性**：EXECUTE 阶段的验证完全确定性（smoke test + timer 对比），不依赖 LLM 判断。