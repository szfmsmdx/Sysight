# worklog

## 2026-03-30 项目初始化与上游调研

### 工作内容
- 阅读了 `nsys-ai` 的 CLI、profile、analysis、web、agent 与 skill 相关模块，梳理出上游项目的核心工作流、可复用能力和当前仓库的拆分边界。
- 基于调研结果，起草了 `Sysight` 的项目定位，明确三阶段主线：profile 采集、轻量 analysis、定向 investigation。
- 初始化工作日志机制，把 `worklog.md` 作为后续持续记录入口，避免后面的设计、实现和验证过程散落在对话里。

### 修改文件
| 文件名 | 修改内容 |
|--------|----------|
| `README.md` | 初始化项目说明，定义目标、三阶段流程、规划目录、致谢对象和暂定协议 |
| `worklog.md` | 初始化工作日志文件，并写入首轮上游调研与项目规划记录 |

### 成果结论
- 明确了 `Sysight` 不做上游的重型 web/TUI/chat 复刻，而是优先保留高价值 analysis 能力。
- 项目已经有了清晰的三阶段拆分边界，后续工作可以围绕 `profiling / analysis / investigation` 逐段推进。

## 2026-03-30 Analysis MVP

### 工作内容
- 创建了 `test/` 目录，并将 `basemodel_8gpu.sqlite` 接入当前仓库，作为 analysis 阶段的本地验证输入。
- 新建 `Sysight` 的最小 Python 包和 CLI，先落地 `info`、`summary`、`analyze` 三个面向 sqlite 的分析入口。
- 从 `nsys-ai` 中抽取并重写了 analysis 所需的最小核心，包括 profile 读取、GPU 摘要、idle gap、NCCL、H2D、同步、launch overhead、root cause 和 findings 导出。

### 修改文件
| 文件名 | 修改内容 |
|--------|----------|
| `README.md` | 增加用户项目放置规则，并补充 `workspace/`、`test/`、`src/sysight/` 等结构说明 |
| 本地 CLI 配置 | 初期曾加过最小打包配置，后续已移除，当前仅保留本地运行方式 |
| `src/sysight/__init__.py` | 新增包版本信息 |
| `src/sysight/__main__.py` | 新增模块入口，支持 `python -m sysight` |
| `src/sysight/cli.py` | 新增最小 CLI，提供 `info`、`summary`、`analyze` 三个命令 |
| `src/sysight/profile.py` | 新增轻量 profile 打开与 schema 发现逻辑 |
| `src/sysight/annotation.py` | 新增 findings/evidence 数据结构与 JSON 导出 |
| `src/sysight/analysis/summary.py` | 新增 per-GPU 摘要和自动 commentary 逻辑 |
| `src/sysight/analysis/queries.py` | 新增核心 SQL 分析查询与 root cause / evidence 构建逻辑 |
| `src/sysight/analysis/report.py` | 新增 analysis 编排和终端报告格式化逻辑 |
| `test/basemodel_8gpu.sqlite` | 接入测试 sqlite，供本地直接验证 |

### 成果结论
- `Sysight` 已具备最小 analysis 主链路：输入 1 个 sqlite，可输出 profile 信息、per-GPU 摘要、根因推断与 findings。
- 新项目已经从运行时层面摆脱了上游的 web、TUI、chat 依赖，先把 analysis 收敛为本地 CLI。

## 2026-03-30 代码定位能力建设

### 工作内容
- 将“异常时间窗 -> runtime 线程 -> sampled stack -> 可疑代码位置候选”整理成 `code_location` skill，并接入现有 analysis 层。
- 新增基于 `COMPOSITE_EVENTS`、`SAMPLING_CALLCHAINS`、`ThreadNames` 和 runtime 窗口关联的定位逻辑，用于从 idle gap 中反推出线程和高频栈帧。
- 把 `code_location` 结果接入 `analyze` 默认输出，使轻量分析报告除了现象和根因外，还能直接给出 PyTorch / Triton / copy path 的定位候选。

### 修改文件
| 文件名 | 修改内容 |
|--------|----------|
| `src/sysight/analysis/code_location.py` | 新增线程筛选、sample 栈聚合、候选帧过滤和结果格式化逻辑 |
| `src/sysight/analysis/report.py` | 在默认 analysis 输出中加入 `Code Location Candidates` 段落 |
| `src/sysight/analysis/skills/code_location.py` | 新增 `code_location` skill 入口 |
| `src/sysight/analysis/skills/registry.py` | 注册 `code_location` skill |
| `README.md` | 将 `code_location` 纳入已提炼 skill 列表 |

### 成果结论
- `Sysight` 已经能从“发现问题”进一步推进到“给出问题对应的线程 / frame 候选”，不再停留在纯现象级诊断。
- 代码定位链路已被封装为独立 skill，后续可以持续优化筛选规则而不需要每次回头重新拼 SQL。

## 2026-03-30 Analysis 对齐与 Markdown 报告

### 工作内容
- 对比当前项目和上游 `nsys-ai` 的 analysis 能力差异，优先补齐了与“问题 -> 代码 / NVTX 定位”最相关的部分，而没有引入 web 端。
- 新增 `top_kernels`、`iteration_timing`、`nvtx_kernel_map`、`nvtx_layer_breakdown` 等轻量 analysis 能力，并接入统一 skill 注册表。
- 为 `Profile` 层补充 `kernel_map`、`gpu_threads`、`runtime_calls`、`nvtx_events` 等 helper，支撑 iteration 检测和 NVTX 归因。
- 重写 `analyze` 输出流程，使其除了终端报告外，还默认生成一份 Markdown 报告，并可选导出 findings JSON。
- 将 Markdown 报告固定为“结果分析 / 存在问题 / 下一步行动指南 / 附录”结构，并在“存在问题”部分尽量绑定 NVTX 区域、runtime 线程和 sampled stack frame。
- 新建 `profile-analysis-report` skill，并配套 `scripts/run-profile-analysis.sh`，把“用户甩一个 sqlite 进来后直接分析并给出报告地址”的流程整理为稳定入口。

### 修改文件
| 文件名 | 修改内容 |
|--------|----------|
| `src/sysight/profile.py` | 新增 kernel/runtime/NVTX 相关 helper，用于更丰富的 analysis 归因 |
| `src/sysight/analysis/nvtx.py` | 新增轻量 NVTX 归因、layer breakdown、kernel map 逻辑 |
| `src/sysight/analysis/iterations.py` | 新增 iteration 检测和 iteration 汇总逻辑 |
| `src/sysight/analysis/queries.py` | 新增 `top_kernel_summary` 与更接近上游的 root cause 匹配逻辑 |
| `src/sysight/analysis/report.py` | 接入 NVTX / iteration / top kernel 数据，并新增 Markdown 报告输出 |
| `src/sysight/analysis/__init__.py` | 导出 Markdown formatter |
| `src/sysight/cli.py` | `analyze` 命令支持写 Markdown 报告，默认落到 `outputs/` |
| `src/sysight/analysis/skills/registry.py` | 注册 `iteration_timing`、`nvtx_kernel_map`、`nvtx_layer_breakdown`、`top_kernels` |
| `src/sysight/analysis/skills/iteration_timing.py` | 新增 iteration timing skill |
| `src/sysight/analysis/skills/nvtx_kernel_map.py` | 新增 NVTX -> kernel 映射 skill |
| `src/sysight/analysis/skills/nvtx_layer_breakdown.py` | 新增 NVTX region GPU time breakdown skill |
| `src/sysight/analysis/skills/top_kernels.py` | 新增 top kernels skill |
| `scripts/run-profile-analysis.sh` | 新增统一分析脚本，生成 Markdown 报告与 findings JSON |
| `skills/profile-analysis-report/SKILL.md` | 新增“路径输入 -> 分析 -> 结论 -> 报告地址” workflow skill |
| `README.md` | 更新当前 analysis 输出形态和已支持 skill 清单 |

### 成果结论
- 当前 `analysis` 已经不是“只会在终端说几句结论”的原型，而是能同时输出终端摘要、Markdown 报告和 findings JSON。
- `basemodel_8gpu.sqlite` 上已经验证了 NVTX region 热点、iteration 节奏和更具体的 code/NVTX 定位信息。
- 项目已经具备“用户给一个 sqlite 路径 -> 直接分析 -> 告知完整报告地址”的标准化入口。

## 2026-03-30 Worklog 整理与主从编排 Skill

### 工作内容
- 重写 `worklog.md`，把此前分散的记录按主题重组为连续、可追溯的工作日志，并补上最新 analysis 对齐与 skill 整理工作。
- 重构 `skills/agent.md` 的目标定义，明确它不是抽象概念，而是“主进程理解任务后，启动多个 `codex exec` 子进程执行具体分析任务”的 workflow skill。
- 新增一个最小可运行的 orchestrator demo 脚本，让主进程可以围绕一个 profile 路径并行启动多个子 Codex 进程，各自跑一个 analysis skill，不生成最终报告，只产出 worker 级输出。
- 验证 `codex --help` 和 `codex exec --help` 接口，确认当前环境下主从并行思路在命令行层面是可落地的。

### 修改文件
| 文件名 | 修改内容 |
|--------|----------|
| `worklog.md` | 重写全文件，整理历史工作并追加本轮日志 |
| `skills/agent.md` | 重写 skill 说明，明确主从编排流程、脚本入口、输入输出和限制 |
| `scripts/parallel-orchestrator-demo.sh` | 新增最小主从编排 demo，主进程并行启动多个 `codex exec` worker |

### 成果结论
- `worklog.md` 已经从零散记录整理成按主题分段的连续日志，后续追加和回顾都更直接。
- `agent` skill 已经有了“可执行脚本 + 明确流程”的最小闭环，不再只是概念描述。
- 主从并行思路现在可以先围绕 analysis skill 跑起来，后续再逐步升级为真正的任务分发与结果聚合体系。

## 2026-03-30 README 报告整理与仓库忽略规则

### 工作内容
- 按“动机 / 设计思路 / 项目简介 / 项目结构 / Done / ToDo / 致谢”的结构重写 `README.md`，把项目说明改成更适合直接对外展示的 Markdown 报告。
- 在 README 里明确回答了“如何发现 Torch 项目的问题与优化点”以及“是否可以做自动化 profile 和 analyse agent”这两个核心问题。
- 更新 `.gitignore`，把本地参考项目 `nsys-ai/`、分析输出目录、工作区目录和常见 Python 本地产物统一排除，避免上传自己仓库时把不该带上的内容一起传出去。
- 将对上游项目的感谢从本地相对路径引用改为公开仓库链接，避免 README 依赖本地目录存在。

### 修改文件
| 文件名 | 修改内容 |
|--------|----------|
| `README.md` | 重写为对外可读的项目说明，补充动机、设计思路、Done、ToDo 和公开致谢链接 |
| `.gitignore` | 补全仓库忽略规则，明确排除 `nsys-ai/`、`outputs/`、`workspace/` 和 Python 本地产物 |
| `worklog.md` | 追加本轮文档与仓库整理记录 |

### 成果结论
- 当前仓库已经具备一份更清晰的对外说明，能够直接回答“这个项目要解决什么问题、现在做到哪里、下一步做什么”。
- 后续上传或初始化 Git 仓库时，不会再把本地参考副本 `nsys-ai/` 误传进去。

## 2026-03-30 Analysis Backend 对齐补齐

### 工作内容
- 继续对照本地 `nsys-ai` 上游仓库，按“不要重型 web/TUI/chat，只补 analysis backend”这个边界梳理剩余缺口。
- 新增并接入 `schema_inspect`、`nccl_breakdown`、`overlap_breakdown`、`speedup_estimator`、`stream_concurrency`、`kernel_launch_pattern`、`cpu_gpu_pipeline`、`thread_utilization` 等缺失 analysis skills。
- 新增 `theoretical_flops` 和 `region_mfu` 两个上游关键分析能力，并在当前仓库里补了一套轻量版 FLOPs / MFU 计算逻辑，包括 GPU peak TFLOPS 查表、NVTX region 匹配、kernel 归因和 MFU 计算。
- 为 `gpu_idle_gaps` 增加兼容 alias，避免因为 skill 命名差异导致 agent 无法命中已存在的 idle gap 分析能力。
- 扩展 `skill run`，支持通过 `--arg key=value` 传递额外参数，使参数化 skill 不需要额外 agent patch 就能直接调用。
- 在 CLI 中补充 `report`、`overlap`、`nccl`、`iters`、`schema`、`theoretical-flops`、`region-mfu` 等分析入口，并在 sample sqlite 上逐项验证。
- 更新 `README.md`，明确当前项目的定位是“尽量对齐上游分析功能，但不默认继承重型 web UI”。

### 修改文件
| 文件名 | 修改内容 |
|--------|----------|
| `src/sysight/analysis/mfu.py` | 新增轻量 FLOPs / region MFU 计算逻辑和格式化输出 |
| `src/sysight/analysis/skills/base.py` | `Skill.run` 支持额外参数透传 |
| `src/sysight/analysis/skills/registry.py` | 注册新增的 analysis skills，包括 `region_mfu` 和 `theoretical_flops` |
| `src/sysight/analysis/skills/schema_inspect.py` | 新增 schema inspection skill |
| `src/sysight/analysis/skills/nccl_breakdown.py` | 新增 NCCL collective breakdown skill |
| `src/sysight/analysis/skills/overlap_breakdown.py` | 新增 compute / communication overlap skill |
| `src/sysight/analysis/skills/speedup_estimator.py` | 新增 speedup estimate skill |
| `src/sysight/analysis/skills/stream_concurrency.py` | 新增 stream concurrency skill |
| `src/sysight/analysis/skills/kernel_launch_pattern.py` | 新增 kernel launch pattern skill |
| `src/sysight/analysis/skills/cpu_gpu_pipeline.py` | 新增 CPU-GPU pipeline skill |
| `src/sysight/analysis/skills/thread_utilization.py` | 新增 CPU thread utilization skill |
| `src/sysight/analysis/skills/theoretical_flops.py` | 新增 theoretical FLOPs skill |
| `src/sysight/analysis/skills/region_mfu.py` | 新增 region MFU skill |
| `src/sysight/analysis/skills/gpu_idle_gaps.py` | 新增上游 `gpu_idle_gaps` 的兼容 alias |
| `src/sysight/cli.py` | 新增 `--arg` 支持和多个 analysis CLI 子命令 |
| `README.md` | 更新功能、状态、Quick Start 和 roadmap 描述 |
| `worklog.md` | 追加本轮 analysis backend 对齐记录 |

### 成果结论
- 当前仓库在“单 profile analysis backend”这一层已经覆盖本地 `nsys-ai` 上游的内置 analysis skills，剩余显著差异主要收敛到尚未实现的 diff，以及明确未纳入当前目标范围的 web / TUI / chat 表层。
- `theoretical_flops` 和 `region_mfu` 已经不再是口头规划，而是当前仓库中可直接运行的分析能力。
