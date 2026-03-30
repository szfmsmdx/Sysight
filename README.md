# nsys-agent

## 动机

对于一个 Torch 模型或者一整个训练 / 推理项目，性能问题通常不是直接写在代码里的。

同样是“跑得慢”，根因可能完全不同：

- 可能是 GPU 大量空转，CPU 没把活及时喂进去
- 可能是 kernel launch 太碎，调度开销吃掉了吞吐
- 可能是 H2D / D2H 拷贝过多，数据路径设计有问题
- 可能是某个同步点把流水线打断了
- 可能是 NCCL 通信异常，导致多卡训练效率很差
- 也可能只是某几个热点 kernel 占了绝大部分时间

只看 Python / Torch 代码，很难直接回答“到底哪里出了问题、哪里最值得优化”。而原始 profile 虽然有信息密度，但时间线很大、阅读门槛高、也不容易快速定位回业务代码。

这个项目要解决的问题就是：

1. 先把 Torch 项目的真实性能行为 profile 出来
2. 再把 profile 结构化分析成问题、证据和代码定位线索
3. 最后给出有针对性的下一步优化方向，而不是泛泛地说“去看看瓶颈”

## 设计思路

### 1. 先拿真实运行证据，而不是先猜

性能优化不应该从“凭经验拍脑袋”开始，而应该从一次可复现、可裁剪、可回看的 profile 开始。

因此 `nsys-agent` 的第一原则是：

- 优先拿到 Nsight Systems 的时间线证据
- 尽量保留 NVTX、runtime call、sampled stack 等可归因信息
- 所有分析都尽量建立在 sqlite / 结构化数据之上

### 2. 分析阶段要低 token、可组合、可复用

大模型不适合直接吞一整份长时间线再自由发挥。更合理的做法是先把分析拆成多个窄技能：

- `top_kernels`
- `iteration_timing`
- `idle_gaps`
- `memory_transfers`
- `memory_bandwidth`
- `kernel_launch_overhead`
- `nccl_anomaly`
- `nvtx_kernel_map`
- `nvtx_layer_breakdown`
- `code_location`
- `root_cause_matcher`

每个 skill 只回答一个明确问题，最后再由主流程聚合成报告。这样可以减少上下文噪声，也更适合后续做多 agent 并行分析。

### 3. 结论必须尽量落到代码

只说“GPU 利用率低”没有太大价值。更重要的是回答：

- 是哪一段时间窗有问题
- 对应的是哪个 runtime 线程
- 当时高频 sampled stack 是什么
- 是否能关联到 NVTX region
- 是否能给出 PyTorch / Triton / 通信 / 数据搬运等方向的定位候选

所以当前项目会尽量把“现象 -> 证据 -> 代码位置候选”这条链路做完整。

### 4. agent 的目标不是一上来就自动改所有代码

这个问题的答案是：可以做 agent，而且当前项目就是朝这个方向设计的。

但第一阶段更务实的目标不是“全自动重写优化代码”，而是先实现：

1. 自动化 profile
2. 自动化 analyse
3. 自动化生成 Markdown 报告和 findings
4. 自动给出下一步 investigation / optimization 建议

等这个闭环稳定以后，再继续往“自动定向优化、自动 benchmark 验证、自动回归比较”推进。

## 项目简介

`nsys-agent` 是一个面向 PyTorch 模型与项目代码的性能分析 agent，目标是把人工 profile + 人工读时间线 + 人工总结建议的流程，收敛成一条更轻量、更结构化、更容易自动化的工作流。

当前项目采用三阶段设计：

1. `profiling`
   负责 Torch 项目埋点、运行封装、`nsys` 调用和 profile 产物管理。
2. `analysis`
   负责基于 sqlite 做轻量分析，识别热点、异常时间窗、根因模式和代码定位线索。
3. `investigation`
   负责围绕问题继续用 `ncu`、benchmark、最小复现和定向实验去深挖。

项目当前优先级很明确：

- Torch-first，而不是通用 profile 浏览器
- 轻量 CLI 和 Markdown 报告优先，而不是先做重型 Web / TUI
- 结构化 evidence 优先，而不是一次性把所有上下文丢给模型
- 先把“发现问题和定位问题”做扎实，再逐步进入“自动优化”

## 项目结构

当前仓库已经存在的主要结构如下：

```text
nsys-agent/
├── README.md
├── .gitignore
├── worklog.md
├── pyproject.toml
├── src/nsys_agent/                 # 核心实现
│   ├── cli.py
│   ├── profile.py
│   ├── annotation.py
│   └── analysis/                   # sqlite 轻量分析主链路
├── scripts/                        # 统一入口脚本与 demo
├── skills/                         # 本仓库内的工作技能说明
├── test/                           # 本地验证输入
├── outputs/                        # 分析产物输出目录
├── workspace/                      # 用户待分析项目工作区
└── nsys-ai/                        # 本地参考副本，仅调研使用，不随仓库发布
```

其中几个关键目录的职责是：

- `src/nsys_agent/`：agent 自己的实现代码
- `scripts/`：一键分析脚本和并行 orchestrator demo
- `test/`：用于验证 analysis 的本地样例
- `workspace/`：后续放用户 Torch 项目或最小复现脚本
- `outputs/`：Markdown 报告、findings JSON、orchestrator 日志等生成物

如果用户给一个待分析项目，推荐放到：

```text
workspace/<project_name>/
```

这样可以把 agent 自己的代码、用户项目代码和分析结果分开管理。

## Done

- 已初始化 `nsys-agent` 的最小 Python 包和 CLI。
- 已支持 `info`、`summary`、`analyze`、`skill list`、`skill run` 等基础命令。
- 已接入 sqlite profile 读取和基础 schema 发现逻辑。
- 已完成轻量 analysis 主链路，可输出终端摘要、Markdown 报告和 findings JSON。
- 已从上游思路中提炼并落地多个 analysis skills，包括热点 kernel、iteration、NVTX、idle gaps、memory、NCCL 和 code location。
- 已支持把异常时间窗进一步绑定到 runtime 线程、sampled stack 和候选代码位置。
- 已提供 `scripts/run-profile-analysis.sh`，可把“给一个 profile -> 输出报告”整理成稳定入口。
- 已提供 `scripts/parallel-orchestrator-demo.sh`，验证主进程调度多个子任务并行分析的 workflow。

## ToDo

- 补全 `profiling` 阶段，让 agent 能直接包裹 Torch 项目运行并生成标准 profile。
- 增强对用户项目结构的理解能力，包括训练入口、推理入口、数据路径和关键模块识别。
- 做更稳定的 NVTX 注入 / 约定，提升“问题 -> 代码区域”定位质量。
- 补全 `investigation` 阶段，把 `ncu`、benchmark 和定向复现实验串起来。
- 增加 before / after diff 分析能力，用于验证优化是否真的生效。
- 继续完善多 agent 主从编排，让不同 analysis skill 能自动并行、自动聚合。
- 在明确安全边界后，探索“提出补丁 -> 跑 benchmark -> 回写结论”的定向优化闭环。
- 补测试、补打包、补正式许可证文件。

## 致谢

本项目当前的设计和实现明显受到了以下项目 / 工具的启发：

- [nsys-ai](https://github.com/GindaChen/nsys-ai)：提供了非常有价值的 profile 读取、analysis skill、agent 化分析和 evidence 组织思路。当前仓库中的 `nsys-ai/` 本地目录仅用于调研和参考，不作为本仓库发布内容的一部分。
- NVIDIA Nsight Systems：提供 CUDA / NCCL / NVTX / runtime 时间线证据，是整个分析链路的数据基础。
- NVIDIA Nsight Compute：提供 kernel 级深挖能力，是后续 investigation 阶段的重要工具。
- PyTorch：本项目主要服务的执行框架，也是定位与优化建议的核心上下文。

## 开源协议

项目当前仍在快速迭代阶段，暂定采用 **MIT License**，后续会补正式 `LICENSE` 文件。
