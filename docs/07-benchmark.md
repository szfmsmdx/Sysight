# 基准测试系统

## 概述

Sysight 有两套独立的基准测试，分别评估**分析能力**和**优化能力**。

| 测试 | 评估目标 | 数据集 | 运行方式 |
|------|---------|--------|---------|
| **Analyze Benchmark** | 从 nsys profile 中发现性能问题的能力 | `nsys-bench/` (6 cases) | `python -m sysight.benchmark` |
| **Optimize Benchmark** | 评判 finding 真伪 + 生成正确 patch 的能力 | `optimizer-bench/` (6 cases) | `sysight bench-optimize` |

---

## Analyze Benchmark

### 数据集：nsys-bench

6 个精心构造的 benchmark case，每个包含一个有性能问题的 Python 训练/推理程序、一个 nsys profile（`.sqlite`）和一份 ground truth（预埋的 finding 列表）。

| Case | 场景 | 预埋 finding 数 | 特点 |
|------|------|----------------|------|
| case_1 | 单卡训练 | 16 | DataLoader + 同步 + 计算浪费 |
| case_2 | 多卡 DDP | 17 | 通信 + 同步 + 配置 |
| case_3 | 推理服务 | 17 | KV cache + batching + 推理循环 |
| case_4 | 混合精度训练 | 16 | AMP + checkpoint + pipeline |
| case_5 | Pipeline 并行 | 17 | micro-batch + 调度 + 通信 |
| case_6 | 多模态训练 | 17 | vision + text + fusion |

### Case 结构

```
nsys-bench/cases/case_1/
├── case.yaml              # case 元信息
├── run.py                 # 入口
├── configs/               # 配置文件
├── profiles/              # nsys profile (.sqlite)
├── src/                   # 源码（含预埋问题）
│   ├── trainers/
│   ├── models/
│   ├── data/
│   └── utils/
└── tests/
    └── findings/
        └── case_1_findings.json  # ground truth
```

### Ground Truth 格式

```json
{
  "case_id": "case_1",
  "total_points": 16,
  "findings": [
    {
      "id": "case_1_f001",
      "category": "C4",
      "file": "src/trainers/loop.py",
      "function": "training_step",
      "line": 31,
      "score": 1,
      "needle": "images = batch[\"images\"].to(self.device)",
      "description": "Image batch is transferred to the target device inside every training step."
    }
  ]
}
```

### 评分方式

Sysight 的 ANALYZE 输出 findings 后，与 ground truth 进行匹配：

```
匹配规则：
  finding.category == truth.category
  AND finding.file_path == truth.file
  AND finding.function == truth.function
  AND finding.line == truth.line
```

得分 = 匹配到的 finding 数。满分 = ground truth 中的 finding 总数。

### 运行

```bash
# 单个 case
python -m sysight.benchmark --cases case_1

# 所有 case
python -m sysight.benchmark --all

# debug 模式（打印 LLM I/O）
python -m sysight.benchmark --cases case_1 --debug
```

### 输出

```
========================================================================
  BENCHMARK RESULTS  --  20260507-181026  --  mode: llm
========================================================================
  Case                  Score  Turns        Tokens     Time      %
  ------------------------------------------------------------------
  case_1             15/16        28     897,772   17m13s    94%
  case_2             15/17        32   1,021,668   22m53s    88%
  ------------------------------------------------------------------
  TOTAL              30/33        60   1,919,440    40m6s    91%

  Prompt tokens: 1,868,713  |  Output tokens: 50,727  |  Total: 1,919,440
  Avg prompt/turn: 31,145  |  Avg output/turn: 845
```

每个 case 的输出目录：

```
.sysight/bench-runs/20260507-181026/case_1/
├── warmup_raw.json     # WARMUP 结果
├── analyze_raw.json    # ANALYZE 结果（完整 findings）
├── answer.json         # 用于评分的精简 answer
├── score.json          # 评分结果
└── debug.log           # 逐轮 LLM 交互日志
```

---

## Optimize Benchmark

### 数据集：optimizer-bench

6 个 case，每个包含：
- 有性能问题的 Python 程序
- 预构建的 `analyze_raw.json`（模拟 ANALYZE 阶段输出）
- 预构建的 `instrument_result.json`（计时器规格）
- 预构建的 `timer_before.json`（baseline 计时数据）
- Ground truth：哪些 finding 是真问题（real）、哪些是假问题（fake）、期望的 patch 行数

### Case 结构

```
optimizer-bench/cases/case_1/
├── case.yaml
├── run.py
├── configs/
├── src/                   # 源码（含预埋问题）
├── artifacts/             # 预构建的中间产物
│   ├── analyze_raw.json
│   ├── instrument_result.json
│   ├── timer_before.json
│   └── warmup_result.json
└── tests/
    └── findings/
        └── case_1_ground_truth.json
```

### Ground Truth 格式

```json
{
  "case_id": "case_1",
  "max_score": 100,
  "real_finding_ids": [
    "C5:3f8a1b2c",
    "C3:a1b2c3d4",
    "C5:e4f5a6b7",
    "C2:c8d9e0f1",
    "C5:2a3b4c5d"
  ],
  "fake_finding_ids": [
    "C4:f6a7b8c9"
  ],
  "expected_patch_lines": {
    "C5:3f8a1b2c": 15,
    "C3:a1b2c3d4": 8,
    "C5:e4f5a6b7": 5,
    "C2:c8d9e0f1": 3,
    "C5:2a3b4c5d": 20
  }
}
```

### 评分维度

Optimize Benchmark 从四个维度评分：

| 维度 | 权重 | 评分规则 |
|------|------|---------|
| **Correctness** | 40 | 所有 patch apply 成功 + smoke test 通过 → 40；apply 成功但 smoke 失败 → 20；apply 失败 → 0 |
| **Performance** | 30 | 对每个 real finding，timer delta < -5% → 1.0；delta < 0 → 0.5；否则 → 0。取平均 × 30 |
| **Judgment** | 20 | 正确接受 real finding（TP）、正确拒绝 fake finding（TN）的 F1 分数 × 20 |
| **Minimality** | 10 | patch 行数 ≤ 期望行数 × 1.2 → 1.0；≤ 期望 × 2.0 → 0.5；否则 → 0。取平均 × 10 |

### 运行

```bash
# 单个 case
sysight bench-optimize case_1

# 所有 case
sysight bench-optimize --all

# debug 模式
sysight bench-optimize case_1 --debug
```

### 输出

```
========================================================================
  OPTIMIZER BENCHMARK SUMMARY  20260507-201037
========================================================================
  case_1: 85/100 (5 patches, 120.5s)
    Correctness:  40/40
    Performance:  25/30
    Judgment:     20/20  (TP=5 FP=0 FN=0 TN=1)
    Minimality:    0/10
  ────────────────────────────────────────
  GRAND TOTAL: 85/100
========================================================================
```

### 自动清理

Optimize Benchmark 在每次 case 运行前后自动 snapshot/restore 源文件，确保测试不会污染代码仓库。

---

## SOTA 追踪

`.sysight/bench-runs/sota.md` 记录每个 case 的历史最佳成绩：

| Case | SOTA | 说明 |
|------|------|------|
| case_1 | 15/16 (94%) | 与历史最佳并列 |
| case_2 | 17/17 (100%) | 当前满分 |
| case_3 | 12/17 (71%) | 当前最高 |
| case_4 | 9/16 (56%) | 当前最高 |
| case_5 | 17/17 (100%) | 当前满分 |
| case_6 | 15/17 (88%) | 当前最高 |

统计口径：只认每次 bench run 目录中的 `summary.txt` 作为分数来源。case 级 SOTA 以单 case 在某次 run 中的最高 Score/Total 为准。