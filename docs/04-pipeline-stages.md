# Pipeline 各阶段详细设计

## WARMUP — 仓库预热

### 职责

扫描代码仓库，自动发现：
- 入口命令（`python run.py --config ...`）
- 活跃配置文件
- 热路径源文件
- 调用关系图
- 依赖噪声（需要跳过的第三方库路径）

### 设计

**确定性，不调用 LLM。**

WARMUP 是纯工程问题：找入口文件、解析 shell 脚本、提取配置、构建调用图。LLM 做这些又慢又贵又不准。

### 流程

```
1. 发现入口命令
   ├── 扫描 run.sh / Makefile / README 中的 python/torchrun 命令
   ├── 解析命令行参数，提取 --config 指向的配置文件
   └── 验证命令可执行（dry-run）

2. 构建文件索引
   ├── 扫描所有 .py 文件
   ├── 排除依赖噪声（.venv, site-packages, third_party）
   └── 按目录结构组织

3. 分析调用关系
   ├── AST 解析 import 关系
   ├── 从入口文件出发，追踪 import 链
   └── 标记热路径文件（被入口直接或间接引用的文件）

4. 生成 overview
   └── 写入 .sysight/memory/wiki/workspaces/<ns>/overview.md
```

### 输出

```python
@dataclass
class RepoSetup:
    entry_point: str          # "python run.py --config configs/train.yaml"
    active_config: str        # "configs/train.yaml"
    hot_paths: list[str]      # ["src/train.py", "src/model.py", ...]
    test_commands: list       # [["python", "-m", "pytest", "tests/"]]
    build_commands: list      # []
    env_vars: dict            # {}
    source: str               # "warmup_verified" | "warmup_partial"
```

### 缓存

WARMUP 结果缓存到 `.sysight/warmup-caches/`，以 repo 路径的 hash 为 key。后续运行默认读缓存，`--force` 强制重新扫描。

---

## ANALYZE — 性能分析

### 职责

从 nsys profile 中挖掘所有性能问题，定位到源码的**文件、函数、行号**。

### 设计

**单次 AgentLoop，LLM 通过工具自主调查。**

ANALYZE 是 Sysight 最核心的阶段。它需要：
1. 读懂 nsys profile 的 SQL 数据
2. 形成 C1-C7 假设
3. 阅读源码验证假设
4. 精确定位行号
5. 输出结构化 findings

这些步骤需要推理能力，是 LLM 的强项。但 LLM 不直接读 SQL——它通过 `nsys_sql_*` 工具查询预计算的统计数据。

### 流程

```
┌─────────────────────────────────────────────────────┐
│                  ANALYZE AgentLoop                   │
│                                                     │
│  1. 阅读 profile 摘要（预注入到 system prompt）       │
│  2. 形成 C1-C7 假设                                 │
│  3. 调用 nsys_sql_* 工具查询详细数据                 │
│  4. 调用 scanner_read 阅读源码验证                   │
│  5. 调用 scanner_search 追踪调用链                   │
│  6. 确认行号后输出 findings JSON                     │
│                                                     │
│  约束：max_turns=30, max_wall_seconds=600           │
└─────────────────────────────────────────────────────┘
```

### Profile 数据预注入

在启动 AgentLoop 之前，ANALYZE 会预计算一份 profile 摘要，直接注入到 user prompt 中。这避免了 LLM 在第一轮就要调用大量 SQL 工具。

摘要包含：

| 数据 | 来源 |
|------|------|
| GPU 利用率 | `nsys_sql_gaps` — 计算 GPU 空闲占比 |
| Top kernel 耗时 | `nsys_sql_kernels` — 按耗时排序的 kernel 列表 |
| 内存搬运统计 | `nsys_sql_memcpy` — H2D/D2H 带宽和次数 |
| 同步点 | `nsys_sql_sync` — cudaDeviceSynchronize 调用 |
| NVTX 区间 | `nsys_sql_nvtx` — 用户标记的区间耗时 |
| NCCL 通信 | `nsys_sql_nccl` — all_reduce/all_gather 耗时 |

### 输出

```python
@dataclass
class LocalizedFinding:
    finding_id: str          # "C1:d19b28bf"
    category: str            # "C1" - "C7"
    title: str               # "DataLoader num_workers=0 导致主线程阻塞喂数"
    priority: str            # "high" | "medium" | "low"
    confidence: str          # "confirmed" | "probable" | "unresolved"
    evidence_refs: list[str] # ["GPU空闲93.1%", "H2D带宽仅8.48 GB/s"]
    metric: str              # "93.1%"
    file_path: str           # "src/data/module.py"
    function: str            # "build_loader"
    line: int                # 30
    description: str         # 详细描述
    suggestion: str          # 修复建议
    status: str              # "accepted" | "rejected" | "unresolved"
```

### Finding 规则

- **Atomic**：一个 finding 对应一行源码的一个具体操作
- **Loop**：循环本身（C2）和循环体内操作（C3/C4/C7）各自独立输出
- **定义 vs 使用**：`line` 指向问题值的赋值/定义行，而非传参调用行
- **Confirmed only**：只有确认在 active execution path 上的才输出

---

## INSTRUMENT — 计时器注入

### 职责

基于 ANALYZE 的 findings，在源码中精确插入 `cuda_timer`，用于后续 EXECUTE 阶段的性能对比。

### 设计

**LLM 决策 + 代码执行。**

LLM 负责决定每个 finding 的计时器应该包裹哪些行（`wrap_start`/`wrap_end`），代码负责实际的注入操作。

### 流程

```
1. LLM 决策
   ├── 阅读每个 finding 指向的源码
   ├── 确定 wrap_start / wrap_end（精确行号）
   ├── 处理重叠范围（合并或调整）
   └── 输出 TimerSpec[]

2. 代码注入
   ├── 写入 _sysight_timer.py 到 repo root
   ├── 在每个目标文件顶部添加 import
   ├── 按 bottom-up 顺序插入 with cuda_timer("label")():
   └── 自动缩进包裹的代码块
```

### cuda_timer 设计

```python
# 自动检测 CUDA 可用性
class _CudaTimer:
    def __init__(self, label: str):
        if HAS_CUDA:
            self._start_evt = torch.cuda.Event(enable_timing=True)
            self._end_evt = torch.cuda.Event(enable_timing=True)
        else:
            # CPU fallback: time.perf_counter()
            ...

    def __call__(self):
        # context manager
        # 输出: [SYSIGHT_TIMER] label: 12.345 ms
```

关键设计：
- **CUDA Event** 用于 GPU 精确计时（而非 wall-clock）
- **CPU fallback** 用于无 GPU 环境（macOS、CI）
- **统一输出格式** `[SYSIGHT_TIMER]` 方便 EXECUTE 阶段解析

### 输出

```python
@dataclass
class TimerSpec:
    finding_id: str
    timer_label: str
    file: str
    wrap_start: int    # 1-based
    wrap_end: int      # 1-based, inclusive
    reason: str
```

---

## OPTIMIZE — 代码优化

### 职责

评判 ANALYZE 的 findings，对确认的真问题生成精确的代码修复 patch。

### 设计

**单次 AgentLoop，LLM 自主评判 + 生成，代码侧填充 hash。**

OPTIMIZE 分两阶段：

| 阶段 | 执行者 | 说明 |
|------|--------|------|
| Phase 1: Plan | LLM | 评判 finding → 阅读源码 → 生成 PatchCandidate |
| Phase 2: Fill hashes | 代码 | 计算 old_span_hash（确定性，不信任 LLM） |

### 为什么 hash 由代码计算

LLM 输出的 `old_span` 行号可能因为各种原因不准确（文件被修改过、LLM 数错行）。代码侧重新读取文件，计算实际内容的 hash，确保 patch apply 时能检测到不匹配。

### LLM 的自主权

OPTIMIZE 的 LLM 有完全自主权：
- **可以拒绝 finding**：如果源码中已有对应优化（假阳性），直接跳过
- **可以合并 finding**：同文件连续区域的多个 finding 合并为一个 patch
- **不需要解释跳过原因**：不确定的直接跳过，节省 token

### 流程

```
┌─────────────────────────────────────────────────────┐
│                OPTIMIZE AgentLoop                    │
│                                                     │
│  1. 阅读所有 finding（只传必要字段）                  │
│  2. 对每个 finding：                                 │
│     ├── scanner_read 阅读目标行及上下文               │
│     ├── 评判：真问题 / 假阳性 / 不确定                │
│     └── 真问题 → 生成 patch                          │
│  3. 输出 patches JSON                                │
│                                                     │
│  约束：max_turns=20, max_wall_seconds=600           │
│  工具：scanner_read, scanner_search, scanner_files   │
└─────────────────────────────────────────────────────┘
```

### 输出

```python
@dataclass
class PatchCandidate:
    patch_id: str
    finding_ids: list[str]     # 可关联多个 finding
    file_path: str
    old_span_start: int        # 替换起始行（1-based）
    old_span_end: int          # 替换结束行（1-based, inclusive）
    old_span_hash: str         # 代码侧填充，LLM 不计算
    replacement: str           # 完整替换代码
    rationale: str             # 修改原因
    validation_commands: list  # 语法检查命令
```

---

## EXECUTE — 应用与验证

### 职责

应用 OPTIMIZE 生成的 patch，运行 smoke test，对比 timer 数据，决定保留或回滚。

### 设计

**完全确定性，不调用 LLM。**

EXECUTE 是 Sysight 的安全网。所有 LLM 输出在这里经过严格验证：

```
Phase 0: Baseline
  └── 读取 timer_before.json（或运行程序采集）

Phase 1: Apply
  ├── 按 bottom-up 排序 patch（同文件内从下往上 apply）
  ├── 逐 patch apply（hash 验证 → 替换）
  └── 任一失败 → 全部 revert

Phase 2: Verify
  ├── Smoke test（import check + test_commands）
  ├── 失败 → 全部 revert
  ├── 注入 cuda_timer（在 patch 之后）
  ├── 运行程序采集 timer_after
  └── 计算 delta_pct

Phase 3: Decision
  └── 所有 patch 通过 → committed
```

### 关键设计决策

**为什么 bottom-up apply？**

同文件多个 patch 时，如果从上往下 apply，第一个 patch 会改变后续 patch 的行号。从下往上 apply 避免了这个问题。

**为什么 timer 注入在 patch 之后？**

`_fill_span_hashes` 计算的 hash 是基于原始源码的。如果先注入 timer 再 apply patch，hash 就对不上了。

### Smoke Test 策略

```
1. 优先使用 patch 自带的 validation_commands
   └── python -c "compile(open('src/xxx.py').read(), 'xxx.py', 'exec')"

2. Fallback: import check
   └── python -c "import src.xxx"

3. Fallback: warmup 中的 test_commands
   └── python -m pytest tests/
```

### 输出

```python
@dataclass
class ExecuteResult:
    run_id: str
    patches: list[PatchResult]    # 每个 patch 的结果
    verify: VerifyResult          # smoke test + timer 对比
    errors: list[str]

@dataclass
class VerifyResult:
    smoke_passed: bool
    timer_before: dict[str, float]   # label → ms
    timer_after: dict[str, float]    # label → ms
    delta_pct: dict[str, float]      # label → %
    status: str                      # "committed" | "reverted"
```

---

## LEARN — 知识积累

### 职责

从 ANALYZE 和 EXECUTE 的结果中提取可复用的知识，写入 wiki。

### 设计

**单次 AgentLoop，LLM 通过 memory_read/memory_write 工具操作 wiki。**

LEARN 在 Pipeline 中被调用两次：
1. ANALYZE 之后：学习 finding 模式
2. EXECUTE 之后：学习哪些优化有效

### 流程

```
1. 读取当前 wiki 状态
   ├── workspace overview
   └── global experience

2. 对比新结果
   ├── findings（ANALYZE 输出）
   └── patches + timer 结果（EXECUTE 输出）

3. 决定更新
   ├── 新的反模式 → 写入 experience
   ├── 新的优化技巧 → 写入 experience
   ├── 过时的知识 → 替换
   └── 无新信息 → 跳过

4. 输出 memory_updates
```

### Wiki 结构

```
.sysight/memory/wiki/
├── workspaces/
│   └── <namespace>/
│       ├── overview.md       # 仓库结构、入口、配置
│       └── experience.md     # 该仓库特有的优化经验
└── experiences/
    └── experience.md         # 全局优化经验（跨仓库）
```