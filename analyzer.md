# Nsys Analyzer Agent — 架构设计文档 v3.1

## 0. 范围与边界

Sysight pipeline：`analyzer` / `optimizer` / `executor` 三模块对等，互不依赖。

**analyzer 的硬边界**（来自 `.claude/rules/security.md`）：

- 不执行目标代码，不导入目标 repo 模块
- 不调用 shell（addr2line、nsys export 等），不写文件
- `.nsys-rep → sqlite` 的 export 由 executor / parent agent 完成；analyzer 可接受 `.nsys-rep` 路径作为 artifact reference，但不会解析它；如果没有显式 `sqlite_path` 或 sibling sqlite，则返回 `status="action_required"`
- 只做诊断，不做优化

**核心原则：profile-first, repo-on-demand**

> nsys analyzer 先用 sqlite 找证据，再按证据定向读 repo，而不是先全量理解 repo。
> 默认不做 full repo scan。full scan 仅用于 CLI 调试、小仓库分析，或 parent agent 显式请求。

**v0 目标**：`nsys sqlite → bottleneck evidence → repo source location` 闭环，简单可落地。

```
                  ┌──────────────── analyzer ─────────────────┐
                  │                                           │
 .sqlite ─────────┤  nsys/          repo.py + RepoIndex      ├──► NsysDiag
                  │  (profile层)    (静态仓库层)               │
                  └───────────────────────────────────────────┘
                            ▲
              executor / parent agent 负责 .nsys-rep → .sqlite
```

---

## 1. 设计原则

| 原则 | 说明 |
|------|------|
| **profile-first, repo-on-demand** | 先用 nsys sqlite 找证据，再按证据定向读 repo；不提前全量扫描 |
| **v0 先做闭环** | 不提前实现多 profiler adapter；等第二个 profiler 出现时再抽 Port |
| **数据结构扩展优先于接口抽象** | `NsysTrace` 里加 `tool: str` 字段预留扩展，不提前写 ABC |
| **显式数据流** | 所有中间结果为 dataclass，无隐藏状态，子 Agent 可截取任意层 |
| **Schema 自适应** | 运行时探测 sqlite 表/列，不硬编码单一版本 |
| **stdlib 优先** | `sqlite3`、`difflib`、`pathlib`；不引入大框架 |
| **duration 用 interval union** | 避免重叠 GPU 事件重复计算，`active_ns ≤ total_ns` 始终成立 |
| **source mapping 准确优先** | 定位失败时给低 confidence + alternatives，不强行输出结论 |

---

## 2. 子任务拆解

| # | 子任务 | 职责 | 输入 | 输出 |
|---|--------|------|------|------|
| T1 | **输入解析** | 检查 sqlite 是否可读；`.nsys-rep` 无 sqlite 时返回 `action_required`；优先使用显式 `sqlite_path`，再尝试 sibling sqlite | `profile_path`, `sqlite_path?` | `ProfileInput` |
| T2 | **Schema 探测** | 读取 sqlite 的表/列结构，产出 capabilities（不只是 version_hint） | sqlite path | `SchemaInfo` |
| T3 | **Timeline 抽取** | GPU kernel、CUDA API、memcpy、NVTX、CPU sample → `TimelineEvent[]` | sqlite + schema | `NsysTrace` |
| T4 | **瓶颈分类** | interval union 计算各类别占比；cpu_sample 点事件不参与 union；输出 confidence + evidence | `NsysTrace` | `BottleneckSummary` |
| T5 | **事件与 Sample 聚合** | top kernel/memcpy/api 统计；CPU sample / call stack / launch site 聚合 | `NsysTrace` | `list[SampleHotspot]` |
| T6 | **仓库定位** | `derive_repo_scope(trace)` → scope-aware scan → `build_repo_index()` → `fuzzy_file_match()` / `lookup_by_file_line()` | `SampleHotspot[]` + sqlite evidence | `list[MappedHotspot]` |
| T7 | **上下文补全** | 给 confidence 足够的热点补 callers/callees；按 `ContextBudget` 限制 | `MappedHotspot[]` + `RepoIndex` | `MappedHotspot[]`（enriched） |
| T8 | **报告合成** | 连接 bottleneck ↔ hotspot，生成 `EvidenceLink`；EvidenceLink 优先级：`correlation_id > time_overlap > file_line > symbol` | T4 + T7 | `NsysDiag` |

---

## 3. 整体分析流程

```
INPUT
  NsysAnalysisRequest
  (profile_path, repo_root, sqlite_path?, scope?)
        │
        ▼
┌──────────────────┐
│ T1               │  ProfileInputResolver
│ ProfileInput     │  sqlite 不存在 → action_required=True
└────────┬─────────┘         │
         │ sqlite ok         └─► 返回 NsysDiag(status="action_required")
         ▼
┌──────────────────┐
│ T2               │  SchemaInspector
│ SchemaInfo       │  探测 tables/columns/capabilities
└────────┬─────────┘
         ▼
┌──────────────────┐
│ T3               │  TraceExtractor
│ NsysTrace        │  统一为 TimelineEvent[]
└────────┬─────────┘
         │
    ┌────┴────────────────────┐
    ▼                         ▼
┌──────────┐         ┌──────────────────────────────────────────┐
│ T4       │         │ REPO LAYER (scope-aware, 三阶段)          │
│ Bottleneck│        │                                          │
│ Classifier│        │  Stage 1: discover_repo()                │
│ (interval │        │    → RepoManifest (路径级, 无文件读取)    │
│  union)  │         │                                          │
│          │         │  derive_repo_scope(trace, repo_root)     │
│ *cpu_sample        │    → RepoScope(mode="targeted")          │
│  不参与  │         │                                          │
│  union*  │         │  Stage 2: scan_repo(root, scope)        │
└──────┬───┘         │    → FileFacts[]                         │
       │             │  build_repo_index() → RepoIndex          │
       │             │                                          │
       │             │  T5: SampleAggregator                    │
       │             │    → SampleHotspot[]                     │
       │             │                                          │
       │             │  T6: RepoMapper                          │
       │             │    1. fuzzy_file_match()                 │
       │             │    2. lookup_by_file_line()  ← 优先      │
       │             │    3. lookup_by_symbol()     ← fallback  │
       │             │    4. kernel name match      ← fallback  │
       │             │    → MappedHotspot[]                     │
       │             │                                          │
       │             │  Stage 3 (T7): ContextEnricher          │
       │             │    callers_of() / callees_of()           │
       │             │    仅对 confidence ≥ threshold 的热点    │
       │             │    按 ContextBudget 限制                 │
       │             │    → MappedHotspot[] (enriched)          │
       │             └───────┬──────────────────────────────────┘
       │                     │
       └──────────┬───────────┘
                  ▼
        ┌──────────────────┐
        │ T8               │  ReportSynthesizer
        │ NsysDiag         │  连接 bottleneck ↔ hotspot
        │ (JSON-able)      │  生成 EvidenceLink[]
        └──────────────────┘
```

---

## 4. 核心数据结构（nsys/models.py）

```python
from __future__ import annotations
from dataclasses import dataclass, field

# ── T1: 输入 ──────────────────────────────────────────────────────────
@dataclass
class ProfileInput:
    original_path: str
    sqlite_path: str | None     # None → 需要 export
    action_required: bool
    reason: str = ""

# ── T2: Schema ────────────────────────────────────────────────────────
@dataclass
class SchemaInfo:
    tables: dict[str, list[str]]
    capabilities: list[str]
    version_hint: str
    warnings: list[str]

# ── T3: Timeline 事件 ─────────────────────────────────────────────────
@dataclass
class TimelineEvent:
    category: str
    name: str
    start_ns: int
    dur_ns: int
    is_sample: bool = False         # cpu_sample 点事件，不参与 interval union
    process_id: int | None = None
    thread_id: int | None = None
    device_id: int | None = None
    stream_id: int | None = None
    correlation_id: int | None = None   # T8 EvidenceLink 最高优先级来源
    extra: dict[str, str | int | float] = field(default_factory=dict)

@dataclass
class NsysTrace:
    tool: str               # "nsys"
    profile_path: str
    sqlite_path: str
    duration_ns: int
    events: list[TimelineEvent]
    schema_version: str
    warnings: list[str]

# ── T4: 瓶颈分类 ─────────────────────────────────────────────────────
# 注意：两个互补的 pct 字段，避免分母混用：
#   pct_of_trace:     active_ns / trace.duration_ns  → 适合 gpu_idle / host_overhead
#   pct_of_gpu_active: active_ns / gpu_active_ns     → 适合 gpu_compute / gpu_comm

@dataclass
class BottleneckLabel:
    category: str
    active_ns: int          # interval union 后的时间
    inclusive_ns: int       # 原始 duration sum（可能 > active_ns，有重叠）
    pct_of_trace: float     # active_ns / trace.duration_ns
    pct_of_gpu_active: float | None   # active_ns / gpu_active_ns；gpu_idle 时为 None
    confidence: float
    evidence: list[str]

@dataclass
class EventStat:
    name: str
    category: str
    count: int
    total_ns: int
    max_ns: int
    inclusive_pct: float    # total_ns / trace.duration_ns（inclusive，可能超 100%）
    # 注意：不叫 pct 是为了明确表达 inclusive 语义（事件可能重叠）

@dataclass
class BottleneckSummary:
    total_ns: int           # trace.duration_ns
    gpu_active_ns: int      # 所有 GPU events union 后的总时间
    gpu_idle_ns: int        # total_ns - gpu_active_ns
    labels: list[BottleneckLabel]   # 按 pct_of_trace 降序
    top_events: list[EventStat]     # top 10 最耗时事件（跨类别）

# ── T5: Sample 聚合 ───────────────────────────────────────────────────
@dataclass
class SourceFrame:
    symbol: str | None
    source_file: str | None
    source_line: int | None
    module: str | None = None
    raw: str | None = None

@dataclass
class SampleHotspot:
    frame: SourceFrame
    count: int
    pct: float
    event_window_ns: tuple[int, int] | None = None

# ── T6+T7: 仓库定位 ───────────────────────────────────────────────────
@dataclass
class MappedHotspot:
    sample: SampleHotspot
    repo_file: str | None
    function: str | None
    match_confidence: float
    match_reason: str       # "file_line" | "symbol" | "kernel_name" | "none"
    callers: list[str]
    callees: list[str]
    alternatives: list[str]

# ── T8: 证据连接 ──────────────────────────────────────────────────────
# EvidenceLink 优先级：
#   1. correlation_id：CUDA API → GPU kernel（nsys 原生，最可靠）
#   2. time_overlap：NVTX range / cpu_sample 与 kernel 时间窗口重叠
#   3. file_line：SourceFrame.source_file + source_line → repo function
#   4. symbol：SourceFrame.symbol → repo function 名称匹配

@dataclass
class EvidenceLink:
    bottleneck_category: str
    event_name: str
    event_category: str
    hotspot_function: str | None
    hotspot_file: str | None
    link_type: str          # "correlation_id" | "time_overlap" | "file_line" | "symbol"
    reason: str
    confidence: float

# ── T8: 最终诊断 ──────────────────────────────────────────────────────
@dataclass
class NsysDiag:
    status: str             # "ok" | "action_required" | "error"
    profile_path: str
    sqlite_path: str | None
    required_action: str | None
    bottlenecks: BottleneckSummary | None
    hotspots: list[MappedHotspot]
    evidence_links: list[EvidenceLink]
    repo_warnings: list[str]
    summary: str

# ── 入口请求 ──────────────────────────────────────────────────────────
@dataclass
class NsysAnalysisRequest:
    repo_root: str
    profile_path: str
    sqlite_path: str | None = None  # 显式指定，优先于 sibling 自动检测
    top_hotspots: int = 20
    map_confidence_threshold: float = 0.4
```

---

## 5. 瓶颈类别定义

| 类别 | 典型证据 | 后续建议 |
|------|---------|---------|
| `gpu_compute` | 非 NCCL kernel active 时间占比高 | 用 Nsight Compute 做 kernel 微架构分析 |
| `gpu_comm` | NCCL / P2P / NVLink kernel 或 NVTX comm range 占比高 | 检查通信拓扑与 overlap 策略 |
| `gpu_memcpy` | HtoD / DtoH / DtoD memcpy 占比高 | 检查 zero-copy / pinned memory / prefetch |
| `host_overhead` | CUDA API launch 间隔大，GPU 空洞明显，CPU sample 热点突出 | 检查 DataLoader / Python GIL / 序列化开销 |
| `sync_wait` | `cudaStreamSync` / `cudaDeviceSync` / blocking event wait 时间高 | 改用异步流水线或 CUDA Graph |
| `gpu_idle` | trace 总时长内 GPU event 覆盖率低 | 检查 CPU-GPU 串行化点 |
| `unknown` | 证据不足，不强行判断 | 返回 `confidence=0`，给 parent agent 决策 |

> **重要**：nsys 描述 timeline 与调度问题，**不等价于** kernel 微架构低效。
> `gpu_compute` 高 = "GPU 时间主要消耗在 compute kernel，下一步建议 ncu 分析"，不直接断言算子本身低效。

> **cpu_sample 特殊说明**：`category == "cpu_sample"` 时 `dur_ns` 通常为 0（点采样），
> `is_sample=True` 字段标记此类事件。分类器必须跳过这类事件的 interval union 计算。

---

## 6. 代码目录结构

```
sysight/analyzer/
├── __init__.py       # 导出全部公共 API
├── repo.py           # 静态仓库分析（scope-aware，三阶段）
├── scanners/         # 各语言 Scanner
│   ├── __init__.py
│   ├── base.py       # FunctionFacts(end_line, is_gpu_kernel, extra)
│   │                   FileFacts(extra)
│   ├── python.py     # end_line via ast.end_lineno; triton/distributed GPU tags
│   ├── cpp.py        # end_line via brace match; __global__ / cuda_launch detection
│   ├── rust.py       # end_line via brace match
│   ├── go.py         # end_line via brace match
│   └── java.py       # end_line via brace match
│
└── nsys/
    ├── __init__.py   # analyze_nsys() + derive_repo_scope() 对外唯一入口
    └── models.py     # 所有 dataclass（唯一数据定义源）
```

---

## 7. repo.py 关键接口

### 7.1 Scope 与 Manifest 数据结构

```python
@dataclass
class RepoScope:
    mode: str = "targeted"          # "targeted" | "entry" | "full"
    seed_files: list[str]           # 精确 seed 路径
    seed_symbols: list[str]         # 函数/kernel 名（用于文件名匹配）
    seed_kernels: list[str]         # GPU kernel 名
    include_extensions: set[str]
    follow_imports_depth: int = 1
    max_files: int = 500
    max_file_bytes: int = 512_000
    include_gpu_related: bool = True

@dataclass
class ContextBudget:
    max_files: int = 30
    max_functions: int = 80
    max_lines_per_function: int = 80
    max_total_chars: int = 80_000

@dataclass
class RepoManifest:
    repo_root: str
    files: list[str]
    languages: dict[str, int]
    candidate_gpu_files: list[str]
    candidate_entry_files: list[str]
    ignored_dir_count: int
    warnings: list[str]
```

### 7.2 RepoIndex（去掉 by_suffix，避免大仓库内存放大）

```python
@dataclass
class RepoIndex:
    files: dict[str, FileFacts]
    by_basename: dict[str, list[str]]           # basename → [full_path]
    by_symbol: dict[str, list[FunctionFacts]]   # short_name → [FunctionFacts]
    by_qualified: dict[str, FunctionFacts]      # qualified_name → FunctionFacts
    symbol_to_file: dict[str, str]              # qualified_name → full_path
# 注意：by_suffix 已删除，suffix match 在 by_basename 候选里 on-the-fly 做
```

### 7.3 三阶段 API

```python
# Stage 1：路径级，不读文件内容
def discover_repo(repo_root: Path) -> RepoManifest: ...

# Stage 2：scope-aware 解析；scope=None 时退化为 full scan（向后兼容）
def scan_repo(
    repo_root: Path,
    scope: RepoScope | None = None,
    manifest: RepoManifest | None = None,
) -> tuple[dict[str, FileFacts], list[str]]: ...

def build_repo_index(files: dict[str, FileFacts]) -> RepoIndex: ...

# Stage 3：按需上下文展开，受 ContextBudget 限制
def get_repo_context(
    repo_root: str,
    targets: list[str],
    budget: ContextBudget | None = None,
    files: dict[str, FileFacts] | None = None,
    index: RepoIndex | None = None,
) -> RepoContextBundle: ...
```

### 7.4 Lookup 函数

```python
# nsys 路径匹配优先级：
# 1. exact  2. normalized suffix  3. basename unique
# 4. basename + dir similarity   5. difflib fuzzy   6. None
def fuzzy_file_match(path_or_basename: str, index: RepoIndex) -> FileMatch | None: ...

# 依赖 FunctionFacts.end_line；先精确区间，再最近函数起始
def lookup_by_file_line(source_file: str, line: int, index: RepoIndex) -> LocationMatch: ...

# 1. exact qualified_name  2. short name exact  3. substring fuzzy
def lookup_by_symbol(
    symbol: str, index: RepoIndex, source_file: str | None = None
) -> list[LocationMatch]: ...

def callers_of(qualified_symbol: str, files: dict[str, FileFacts], limit: int = 20) -> list[str]: ...
def callees_of(qualified_symbol: str, files: dict[str, FileFacts], limit: int = 20) -> list[str]: ...
```

### 7.5 Mapper 定位顺序（T6）

```
sample.frame
    │
    ├─ source_file + source_line 都有？
    │   → fuzzy_file_match() → lookup_by_file_line()   [最准，优先]
    │
    ├─ 只有 symbol？
    │   → lookup_by_symbol()                           [次选]
    │
    ├─ 只有 kernel name？（GPU kernel）
    │   → 匹配 .cu / .cuh 中 __global__ 函数名          [fallback]
    │
    └─ 全部失败
        → MappedHotspot(match_confidence=0, alternatives=[...])

补上下文（T7）：
    仅当 match_confidence >= threshold 时执行
    callers_of() → MappedHotspot.callers
    callees_of() → MappedHotspot.callees
    受 ContextBudget.max_functions 限制
```

---

## 8. nsys/__init__.py — 对外唯一入口

```python
def analyze_nsys(request: NsysAnalysisRequest) -> NsysDiag:
    # T1: resolve sqlite
    profile = resolve_profile_input(request.profile_path, request.sqlite_path)
    if profile.action_required:
        return NsysDiag(status="action_required", ...)

    # T2 T3
    schema = inspect_schema(profile.sqlite_path)
    trace  = extract_trace(profile.sqlite_path, schema)

    # derive scope from profile evidence (profile-first)
    scope = derive_repo_scope(trace, request.repo_root)

    # T4 T5
    bottlenecks = classify_bottlenecks(trace)
    samples     = aggregate_samples(trace, top_n=request.top_hotspots)

    # T6 T7（scope-aware, NOT full scan）
    manifest = discover_repo(Path(request.repo_root))     # Stage 1
    files, warnings = scan_repo(Path(request.repo_root), scope=scope, manifest=manifest)  # Stage 2
    index = build_repo_index(files)
    mapped = map_to_repo(samples, index,
                         confidence_threshold=request.map_confidence_threshold)

    # T8
    return synthesize_report(profile, bottlenecks, mapped, warnings)

def derive_repo_scope(trace: NsysTrace, repo_root: str) -> RepoScope:
    # 从 trace events 中提取 source_file / kernel_name / NVTX range
    # 返回 mode="targeted" 的 RepoScope
    ...
```

---

## 9. scanners 增强（已实现）

### 9.1 FunctionFacts 新增字段

```python
@dataclass
class FunctionFacts:
    name: str
    qualified_name: str
    line: int
    calls: list[str]
    end_line: int | None = None       # ✅ 已实现：所有 scanner 补全
    is_gpu_kernel: bool = False       # ✅ 已实现：__global__ / @triton.jit
    extra: dict[...] = ...            # ✅ 已实现

@dataclass
class FileFacts:
    ...
    extra: dict[...] = ...            # ✅ 已实现：gpu_tags / cuda_kernel_count
```

### 9.2 C++/CUDA scanner 识别

- ✅ `__global__ void kernel_name(...)` — CUDA kernel 定义，`is_gpu_kernel=True`
- ✅ `kernel_name<<<grid, block>>>(...)` — CUDA kernel launch，记录到 `fn.extra["cuda_launches"]`
- ✅ `.cu`/`.cuh` 文件级 `gpu_tags=["cuda_file"]`

### 9.3 Python scanner 识别

- ✅ `@triton.jit` — Triton kernel，`is_gpu_kernel=True`, `gpu_tags=["triton_jit"]`
- ✅ `torch.distributed.*` / `dist.all_reduce` — `gpu_tags=["nccl_comm"]`
- ✅ `torch.utils.data.DataLoader` — `gpu_tags=["dataloader"]`
- ✅ `torch.autograd.profiler.record_function` — `gpu_tags=["autograd_rec"]`

---

## 10. 优先级路线图

### P0（已完成）✅

| 任务 | 状态 |
|------|------|
| `FunctionFacts.end_line` | ✅ 所有 scanner |
| `FileFacts.extra` / `FunctionFacts.extra` | ✅ |
| `RepoScope` / `RepoManifest` / `ContextBudget` | ✅ |
| `discover_repo()` | ✅ |
| `scan_repo(scope=)` | ✅ |
| `build_repo_index()` | ✅ 去掉 by_suffix |
| `fuzzy_file_match()` | ✅ |
| `lookup_by_file_line()` | ✅ |
| `nsys/models.py` | ✅ `EventStat.inclusive_pct`, `BottleneckLabel` 双 pct |
| `analyze_nsys()` 骨架 + `NsysAnalysisRequest` | ✅ |
| `derive_repo_scope()` 骨架 | ✅ |

### P1（下一步）

| 任务 | 说明 |
|------|------|
| `lookup_by_symbol()` 增强 | 已有基础实现，需要与 kernel name demangling 配合 |
| `callers_of()` / `callees_of()` | ✅ 已实现基础版 |
| `get_repo_context()` | ✅ 已实现基础版 |
| nsys T1–T5 实现 | sqlite parser / schema probing / trace extractor / bottleneck classifier |
| EvidenceLink 使用 correlation_id | T8 核心，依赖 T3 correlation_id 字段 |

### P2

| 任务 | 说明 |
|------|------|
| `compare_nsys(before, after)` | Research loop 迟早需要 |
| Triton / PyTorch distributed 精确识别 | ML repo 精准度提升 |
| multi-rank / multi-GPU 支持 | `process_id` / `device_id` 字段已预留 |

### P3

| 任务 | 说明 |
|------|------|
| persistent cache | 由 executor / orchestrator 管理写入 |
| MCP adapter | 等核心闭环稳定后薄封装同一批 Python API |
| 多 profiler Port 抽象 | 等第二个 profiler 出现时再做 |

---

## 11. 不做的事（v0 范围外）

| 能力 | 原因 |
|------|------|
| SQLite GraphStore / NetworkX | 太重，内存 dataclass 足够 |
| embeddings / vector search | nsys 映射是精确定位问题，不是语义搜索 |
| community detection / bridge nodes | 适合架构审查，不适合 profile 诊断 |
| addr2line / shell symbolization | 违反 security.md；外部符号化交 executor |
| refactor suggestions / wiki | analyzer 不优化，违反边界 |
| MCP tools / CLI registry | v0 不需要产品化包装 |
| ProfilerPort ABC | v0 只有 nsys；`NsysTrace.tool` 字段预留 |
| full repo scan as default | profile-first 原则；full 只用于 CLI 调试 |
