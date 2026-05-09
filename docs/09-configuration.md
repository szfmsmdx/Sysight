# 配置与部署

## 配置文件

Sysight 的配置文件位于 `.sysight/config.yaml`：

```yaml
# ── LLM Provider 配置 ──

analyze:
  provider: openai_compatible     # openai_compatible | anthropic | replay
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

每个阶段可以使用不同的 provider 和 model。例如，ANALYZE 用更强的模型，LEARN 用更便宜的模型。

---

## Provider 类型

### OpenAI Compatible

适用于 OpenAI API 及兼容服务（Azure、本地部署等）：

```yaml
analyze:
  provider: openai_compatible
  api_key: "sk-..."
  base_url: "https://api.openai.com/v1"
  model: "gpt-5"
```

### Anthropic

适用于 Claude 系列模型：

```yaml
analyze:
  provider: anthropic
  api_key: "sk-ant-..."
  model: "claude-sonnet-4-5"
```

### Replay

用于 debug，回放之前的 LLM 响应（不调用真实 API）：

```yaml
analyze:
  provider: replay
  replay_file: ".sysight/analysis-runs/run-xxx/debug.log"
```

---

## 环境变量

所有配置项也可以通过环境变量设置（优先级高于配置文件）：

| 环境变量 | 对应配置 |
|---------|---------|
| `SYSIGHT_ANALYZE_API_KEY` | `analyze.api_key` |
| `SYSIGHT_ANALYZE_BASE_URL` | `analyze.base_url` |
| `SYSIGHT_ANALYZE_MODEL` | `analyze.model` |
| `SYSIGHT_OPTIMIZE_API_KEY` | `optimize.api_key` |
| ... | ... |

---

## 依赖

### Python 依赖

```
# 核心依赖
openai>=1.0.0          # OpenAI API 客户端
anthropic>=0.30.0      # Anthropic API 客户端
pyyaml>=6.0            # YAML 配置解析

# 工具依赖（可选，按需安装）
torch>=2.0.0           # GPU timer（仅 GPU 端需要）
```

### 系统依赖

- **Nsight Systems CLI**：用于采集 nsys profile（`nsys profile` 命令）
- **CUDA Toolkit**：GPU 端运行需要
- **Python 3.10+**

---

## 安装

```bash
# 从源码安装
git clone <repo-url> && cd Sysight
pip install -e .

# 或通过 uv
uv pip install -e .
```

---

## 输出目录

所有输出默认写入 `.sysight/`：

```
.sysight/
├── config.yaml             # 配置文件
├── analysis-runs/          # 分析结果
│   └── run-<hash>/
│       ├── analyze_raw.json
│       ├── debug.log
│       └── instrument_result.json
├── optimizer-runs/         # 优化结果
│   └── run-<hash>/
│       ├── optimize_debug.log
│       └── optimize_result.json
├── execute-runs/           # 执行结果
│   └── run-<hash>/
│       └── execute_result.json
├── bench-runs/             # 基准测试结果
│   └── <timestamp>/
│       ├── case_1/
│       └── summary.txt
├── optimizer-bench-runs/   # 优化基准测试结果
│   └── <timestamp>/
│       ├── case_1/
│       └── summary.json
├── warmup-caches/          # 预热缓存
│   └── <hash>.json
└── memory/wiki/            # 知识库
    ├── workspaces/
    │   └── <namespace>/
    │       ├── overview.md
    │       └── experience.md
    └── experiences/
        └── experience.md
```

---

## 模型推荐

根据阶段特点推荐不同的模型配置：

| 阶段 | 推荐模型 | 原因 |
|------|---------|------|
| ANALYZE | GPT-5 / Claude Opus 4 | 需要强推理能力，turns 多，context 大 |
| OPTIMIZE | GPT-5 / Claude Sonnet 4 | 需要精确的代码理解，turns 较少 |
| INSTRUMENT | GPT-5 / Claude Sonnet 4 | 需要理解代码结构，确定计时器范围 |
| LEARN | GPT-5 / Claude Haiku 4 | 归纳总结，不需要强推理 |

### Context Window 要求

ANALYZE 阶段可能消耗 ~900K prompt tokens（28 turns），建议使用 context window ≥ 200K 的模型。Sysight 的上下文压缩系统会根据模型的实际 context window 自动调整压缩阈值。

---

## 安全考虑

### API Key 管理

- 配置文件中的 `api_key` 可以替换为环境变量引用
- 建议使用 `.env` 文件或系统环境变量管理密钥
- `.sysight/` 目录应加入 `.gitignore`

### 文件系统安全

- Scanner 工具通过 `path_containment` 限制只能读取 repo 内的文件
- Patcher 只修改 repo 内的 `.py` 文件
- LEARN 阶段只能写入 `workspaces/` 和 `experiences/` 路径

### 代码执行安全

- EXECUTE 阶段的 smoke test 在 repo 目录内运行
- 只执行 warmup 中验证过的命令
- 有 timeout 保护（smoke test 60s，measurement 300s）