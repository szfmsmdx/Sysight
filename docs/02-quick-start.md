# 快速开始

## 安装

```bash
# 克隆仓库
git clone <repo-url> && cd Sysight

# 安装依赖
pip install -e .
```

依赖：Python 3.10+、CUDA 工具链（仅 GPU 端需要）、Nsight Systems CLI。

---

## 配置 LLM Provider

在项目根目录创建 `.sysight/config.yaml`：

```yaml
analyze:
  provider: openai_compatible
  api_key: "your-api-key"
  base_url: "https://api.openai.com/v1"
  model: "gpt-5"

optimize:
  provider: openai_compatible
  api_key: "your-api-key"
  base_url: "https://api.openai.com/v1"
  model: "gpt-5"

instrument:
  provider: openai_compatible
  api_key: "your-api-key"
  base_url: "https://api.openai.com/v1"
  model: "gpt-5"

learn:
  provider: openai_compatible
  api_key: "your-api-key"
  base_url: "https://api.openai.com/v1"
  model: "gpt-5"
```

每个阶段可以使用不同的 provider 和 model。

---

## 三步走：分析 → 优化 → 验证

### Step 1: 预热（WARMUP）

让 Sysight 了解你的代码仓库结构：

```bash
sysight warmup ./my-training-repo
```

这一步是确定性的（不调用 LLM），会扫描仓库的入口文件、配置文件、调用关系，生成 workspace overview。

输出示例：

```
============================================================
  Warmup 完成 — warmup_verified
============================================================
  入口:     python run.py --config configs/train.yaml
  profile:  profiles/trace.sqlite
  热路径:   12 文件

  Cache: .sysight/warmup-caches/abc123.json
  Overview: .sysight/memory/wiki/workspaces/my-training-repo/overview.md
```

### Step 2: 分析（ANALYZE）

用 nsys profile 分析性能问题：

```bash
sysight analyze profiles/trace.sqlite --repo ./my-training-repo
```

Sysight 会调用 LLM，通过工具读取 profile 数据和源码，输出结构化的 findings。

输出示例：

```
  Analyze output → .sysight/analysis-runs/run-f15b4613/

  ANALYZE COMPLETE  run_id=run-f15b4613
  Findings:    24
  Rejected:    3
  Elapsed:     45230 ms
  Tokens:      prompt=89772 output=4823
  Turns:       28
```

产物：

| 文件 | 说明 |
|------|------|
| `analyze_raw.json` | 完整的 findings 列表，含文件、函数、行号、描述、建议 |
| `debug.log` | 逐轮 LLM 交互日志 |

### Step 3: 优化 + 验证（OPTIMIZE + EXECUTE）

基于分析结果生成修复并验证：

```bash
sysight optimize run-f15b4613 --repo ./my-training-repo
```

这一步会自动完成：
1. **OPTIMIZE**：LLM 评判每个 finding，对真问题生成 patch
2. **EXECUTE**：apply patch → smoke test → timer 对比

输出示例：

```
  OPTIMIZE COMPLETE  run_id=run-f15b4613
  Patches:      8

  EXECUTE COMPLETE  run_id=run-f15b4613
  Patches:      8
  Kept:         7
  Reverted:     1

  Timer comparison:
    data_loader: -12.3% ↓ (2.4ms → 2.1ms)
    forward_pass: -8.7% ↓ (15.2ms → 13.9ms)
    loss_logging: -95.1% ↓ (1.2ms → 0.06ms)
```

---

## 一键全流程

```bash
sysight full profiles/trace.sqlite --repo ./my-training-repo
```

等价于依次执行 WARMUP → ANALYZE → INSTRUMENT → LEARN → OPTIMIZE → EXECUTE → LEARN。

---

## 运行基准测试

### 分析能力测试

```bash
# 运行单个 case
python -m sysight.benchmark --cases case_1

# 运行所有 case
python -m sysight.benchmark --all
```

### 优化能力测试

```bash
sysight bench-optimize --all
```

---

## CLI 命令总览

| 命令 | 说明 |
|------|------|
| `sysight warmup <repo>` | 扫描仓库，生成 overview |
| `sysight analyze <profile> --repo <repo>` | 分析 nsys profile |
| `sysight instrument <run_id> --repo <repo>` | 基于 findings 插入计时器 |
| `sysight optimize <run_id> --repo <repo>` | 优化 + 验证 |
| `sysight learn <run_id>` | 从分析结果中学习 |
| `sysight full <profile> --repo <repo>` | 一键全流程 |
| `sysight bench-optimize` | 优化能力基准测试 |
| `sysight tool <category> <name> [args]` | 直接调用单个工具 |

---

## 输出目录结构

```
.sysight/
├── analysis-runs/          # 分析结果
│   └── run-f15b4613/
│       ├── analyze_raw.json
│       ├── debug.log
│       └── instrument_result.json
├── optimizer-runs/         # 优化结果
│   └── run-a1b2c3d4/
│       ├── optimize_debug.log
│       └── optimize_result.json
├── execute-runs/           # 执行结果
│   └── run-e5f6g7h8/
│       └── execute_result.json
├── bench-runs/             # 基准测试结果
│   └── 20260507-181026/
│       ├── case_1/
│       └── summary.txt
├── warmup-caches/          # 预热缓存
└── memory/wiki/            # 知识库
    ├── workspaces/
    └── experiences/
```