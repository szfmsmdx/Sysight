# Sysight Pipeline 重构方案

## 一、总体架构

```
warmup（环境探查）
  ↓ RepoSetup（entry_point, test_commands, code_map）
  ↓ .sysight/warmup-caches/<hash>.json

analyzer（profile 分析）
  ↓ LocalizedFindingSet（含量化 metric + 方向性 suggestion）
  ↓ .sysight/analysis-runs/<run_id>/analyze_raw.json

instrument（cuda_timer 插桩）
  ├ 输入：analyze_raw.json
  ├ 确定性源码读取 → 单次 LLM 调用 → 程序化注入
  ↓ .sysight/analysis-runs/<run_id>/instrument_result.json

optimizer（代码修改 + 验证）
  ├ 输入：analyze_raw.json + instrument_result.json + warmup_cache
  ├ Phase 1: Plan — findings 按文件分组为 change_sets
  ├ Phase 2: Execute — 每个 change_set 单次 LLM 调用 → PatchCandidate[]
  ├ Phase 3: Verify — 确定性：apply → smoke test → grep [SYSIGHT_TIMER]
  ↓ .sysight/analysis-runs/<run_id>/optimize_result.json

learn（memory 更新）
  ├ 输入：optimize_result.json
  ↓ .sysight/memory/wiki/experience.md
```

---

## 二、各阶段设计

### 2.1 Analyzer

**改动**：小

- `LocalizedFinding` 增加 `metric` 字段：量化指标（如 `"12.5ms"`, `"843 calls"`），从 profile 证据中提取
- `suggestion` 简化为方向性建议（如 "考虑使用 torch.compile 优化 forward pass"），不再包含 patch_hint
- 移除 `patch_hint`，因为 patch 生成已移到 Optimizer 阶段

### 2.2 Instrument

**改动**：重写

**新流程**：
1. 读取 findings 指向的源文件（带行号）
2. 单次 LLM 调用：输入 findings + 源码上下文 → 输出 `TimerSpec[]`（JSON 格式）
3. 程序化注入 `cuda_timer` 模板 + `with cuda_timer(...)():` 块到源文件
4. 输出 `instrument_result.json`（含 timer labels 供 Optimizer Verify 使用）

**计时工具**：`cuda_timer`（基于 `torch.cuda.Event`）
- 内联注入到目标文件头部，自包含
- Context-manager API：`with cuda_timer("label")(): ...`
- 每次计时结束打印 `[SYSIGHT_TIMER] label: X.XXX ms`
- 支持多次计时（存储 elapsed_ms 列表）
- 提供 `summary()` 类方法打印聚合统计

**关键设计**：
- LLM 只负责决定"在哪里埋点"（TimerSpec），不负责写代码
- 代码注入是确定性的（程序化），不依赖 LLM
- Bottom-up 插入保证行号偏移正确

### 2.3 Optimizer

**改动**：重写

**新流程**：Plan-Execute-Verify

```
Phase 0: Baseline
  运行程序 → 捕获 [SYSIGHT_TIMER] 日志 → 记录 baseline 计时

Phase 1: Plan（确定性）
  将 accepted findings 按 file_path 分组为 ChangeSet[]
  同文件的多个 findings 合并，减少 LLM 调用和避免 patch 冲突

Phase 2: Execute（每个 ChangeSet 一次 LLM 调用）
  对每个 ChangeSet:
    1. 读取目标文件源码（带行号）
    2. 单次 LLM 调用 → PatchCandidate[]
    3. 通过 PatchApplier 程序化 apply patch

Phase 3: Verify（确定性，无 LLM）
  Level 1（默认）：smoke test
    - patch 自带的 validation_commands
    - 简单 import 检查
    - warmup 的 test_commands
  Level 2（有 cuda_timer 数据时）：计时对比
    - 运行程序 → 捕获 [SYSIGHT_TIMER] 日志
    - 解析每个 timer_label 的耗时
    - 计算 delta_pct，判断优化效果
  失败时自动 revert
```

**PatchApplier**：确定性的行级 patch apply/revert 工具
- `apply(file_path, old_span_start, old_span_end, old_span_hash, replacement)`
- 通过 SHA1 hash 校验旧代码，防止误改
- 保存原始内容快照，支持 `revert()` 回退

**关键设计**：
- 每个文件一个 ChangeSet，避免跨文件 patch 冲突
- Verify 完全确定性，不依赖 LLM 判断
- 基于 `[SYSIGHT_TIMER]` 日志归因验证，无需重新跑 `nsys profile`
- 失败自动 revert，保证代码安全

### 2.4 Warmup

**改动**：无需修改

当前 warmup 已是纯 Phase 1（环境验证 + 代码地图），无需修改。

---

## 三、数据流

所有阶段的输入输出都是磁盘文件，阶段间无内存耦合，可以单独重跑任意阶段。

```
.sysight/
├── warmup-caches/<hash>.json          # RepoSetup
├── analysis-runs/<run_id>/
│   ├── analyze_raw.json               # LocalizedFindingSet
│   ├── debug.log                      # analyze LLM debug log
│   ├── instrument_result.json         # TimerSpec[] + verify_hint
│   ├── instrument_debug.log           # instrument LLM debug log
│   ├── optimize_result.json           # ChangeSet[] + VerifyResult[] + PatchResult[]
│   └── optimize_debug.log             # optimize LLM debug log
└── memory/wiki/                       # knowledge base
```

---

## 四、新增/修改文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `sysight/pipeline/cuda_timer.py` | 新增 | cuda_timer 模板，内联注入用 |
| `sysight/tools/patcher.py` | 新增 | 确定性行级 patch apply/revert |
| `sysight/pipeline/instrument.py` | 重写 | 单次 LLM → 程序化注入 |
| `sysight/pipeline/optimize.py` | 重写 | Plan-Execute-Verify 流程 |
| `sysight/types/findings.py` | 修改 | LocalizedFinding 增加 metric 字段 |
| `sysight/agent/prompts/fragments/instrument_system.md` | 新增 | instrument LLM system prompt |
| `sysight/agent/prompts/fragments/optimize_system.md` | 新增 | optimizer LLM system prompt |
| `sysight/agent/prompts/fragments/optimizer_sop.md` | 修改 | 更新 SOP |
| `sysight/agent/prompts/loader.py` | 修改 | 加载新 prompt 片段 |
| `sysight/pipeline/runner.py` | 修改 | 传递 run_dir，移除 repo_setup 依赖 |
| `sysight/cli/cli.py` | 修改 | optimize 命令增加 --debug, 传 run_dir |

---

## 五、Analyzer 优化计划（已记录，待后续迭代）

### D：引导 LLM 先 SQL 后按需读代码

在 `analyze_system.md` 的"## 1. 形成假设"里补充：形成初始假设后，优先使用 SQL 工具精确量化各假设的 profile 证据，再根据 SQL 结果判断哪些假设值得进入代码验证。

### F：积极驱逐已完成的代码 compaction

在 `AgentContext` 中增加阶段感知驱逐机制：当检测到 LLM 开始发起 SQL 查询时，主动将所有 `scanner_read` tool results 压缩为文件摘要块。
