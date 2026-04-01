# Sysight

[English](./README.md)

面向 Nsight Systems profile 分析的轻量级、Torch-first、agent-first 后端。

`Sysight` 旨在把 Nsight Systems profile 转换成结构化结论、代码定位线索和下一步优化建议，重点服务于 PyTorch 训练与推理场景，是一个本地使用的轻量分析项目。

这个项目首先面向 agent 工作流：长期目标不是再做一个厚重的 profile 查看器，而是提供一个紧凑、可编排的后端，让 agent 能够调用它完成 profile、analysis 和后续迭代。

正常使用方式是对 agent 直接说一句话，把待分析的 profile 路径交给它。CLI 只是 agent 底层调用的后端入口，不是主要的用户交互界面。

## 功能

- 面向训练和推理场景的 Torch-first 分析
- 基于 SQLite 的分析流水线，强调低上下文和 skill 化拆分
- 面向 agent 的分析后端，重点输出简洁结论、定位线索和下一步行动建议
- 内置 kernel hotspot、iteration timing、idle gaps、launch overhead、memory transfers、NCCL anomalies、overlap、NVTX attribution、code-location hints、MFU、theoretical FLOPs 等分析能力
- 支持输出 Markdown 和 JSON，方便交给后续 agent 或更深一步的调查流程
- 提供早期的多 worker 并行 orchestration demo

## 当前状态

目前仓库主要聚焦在 analysis 后端：轻量级 profile 检查、skill 化分析、MFU/FLOPs 辅助能力，以及基于 Nsight Systems 数据生成报告。围绕真实用户项目的自动化 profiling、更自然的 agent 交互回路，以及更深的 investigation workflow 仍在 roadmap 中。

## 快速开始

1. 把 Nsight Systems 的 `.sqlite` / `.sqlite3` / `.nsys-rep` 文件放进项目里。

2. 直接对 agent 说：

```text
帮我分析一下 test/basemodel_8gpu.sqlite
```

或者：

```text
帮我分析一下 /abs/path/to/profile.sqlite
```

或者直接给一个 Nsight Systems 的采集结果：

```text
帮我分析一下 test/basemodel_8gpu.nsys-rep
```

3. agent 会自动调用分析后端，并在终端返回一份三段式简要报告：

- `结论`
- `问题`
- `下一步行动建议`

4. 当前支持的输入文件包括：

- `*.sqlite`
- `*.sqlite3`
- `*.nsys-rep`

5. 如果输入是 `*.nsys-rep`，Sysight 会先尝试自动导出同名的 `*.sqlite`，底层执行命令为：

```bash
nsys export --type=sqlite -o <same-stem>.sqlite --force-overwrite=true <file.nsys-rep>
```

这一步要求本机 `PATH` 上能找到 `nsys`。如果本机没有 `nsys`，就先手动导出，或者使用仓库里的辅助脚本：

```bash
scripts/export-nsys-sqlite.sh path/to/profile.nsys-rep
```

6. 同时，Sysight 会生成分析产物，例如：

```text
<profile>.sqlite                 # 只有输入是 .nsys-rep 且需要转换时才会生成
outputs/<profile>.report.md
outputs/<profile>.findings.json
```

## Agent 工作流

这个项目的目标方向是 agent-first。

当前 agent 工作流先分成两条主路径：

1. `profile-only`：用户只给 `.sqlite` / `.sqlite3` / `.nsys-rep` 路径，Sysight 直接分析
2. `workspace-aware`：用户额外提供 `workspace/` 和 `program.md` 合同文件，agent 在分析时结合项目入口、性能目标、关键路径和限制来解释 findings

因此当前仓库里的主路径是：

1. 用户用自然语言提供 profile 路径
2. agent 可选地根据 `profile + workspace + program.md` 先判断工作流模式
3. agent 自动调用 analysis backend
4. agent 返回 `结论 / 问题 / 下一步行动建议`
5. Sysight 落一份完整的 `report.md` 报告，供后续深入分析

所以 CLI 现在存在，是为了支撑 agent 的底层执行，不是主要的人机交互形式。

### 推荐使用顺序

- 如果用户手上只有 profile，先走 `profile-only`，快速拿到分析摘要和 `report.md`。
- 如果用户还能提供 `workspace/program.md`，再升级到 `workspace-aware`，获得更强的项目语义归因。
- 当前阶段优先把 `analyze` / `report` 跑顺，再考虑更深的 profiling 和优化闭环。

### Workspace 合同

如果用户希望走 `workspace-aware` 路径，建议在 `workspace/` 下放一份 `program.md`。当前仓库已经提供了一个起步模板：`workspace/program.md`。

模板当前要求用户补充这些内容：

- 任务和项目背景
- 框架 / 技术栈
- 启动命令
- 性能目标
- 关键路径
- 约束
- 成功标准
- 输出契约

## 后端调试命令

如果只是调试底层分析能力，可以在仓库根目录手动运行：

```bash
PYTHONPATH=src python3 -m sysight --help
PYTHONPATH=src python3 -m sysight analyze path/to/profile.sqlite
PYTHONPATH=src python3 -m sysight analyze path/to/profile.sqlite --workspace workspace --program program.md
PYTHONPATH=src python3 -m sysight route --profile path/to/profile.sqlite --workspace workspace --program program.md
PYTHONPATH=src python3 -m sysight skill list
PYTHONPATH=src python3 -m sysight skill run workflow_router path/to/profile.sqlite --workspace workspace --program program.md
scripts/export-nsys-sqlite.sh path/to/profile.nsys-rep
```

## 仓库结构

```text
sysight/
├── README.md
├── README.zh-CN.md
├── LICENSE
├── worklog.md
├── src/sysight/                 # 对外公开包入口
├── scripts/                     # 开发工具和 orchestration demo
├── skills/                      # 内部 skill 说明和流程文档
├── test/                        # 本地 profile 样例输入
├── outputs/                     # 生成的分析产物
└── workspace/                   # 用户工作区，包含 program.md 合同模板
```

## 路线图

- [ ] 增加 profiling 阶段，能够包装真实 Torch 入口并产出标准 Nsight Systems capture
- [ ] 强化围绕 `program.md` 的 workspace-aware analysis，包括入口、模块和数据路径理解
- [ ] 强化 NVTX 约定和 attribution 质量
- [ ] 增加围绕 `ncu`、targeted benchmark 和可控复现的 investigation workflow
- [ ] 支持优化前后 profile 对比，做效果验证
- [ ] 让多 agent orchestration 更自动化，而不是只停留在 demo
- [ ] 探索一个安全的优化闭环：提出修改、验证收益、再反馈给 agent
- [ ] 补齐测试和本地项目发布前的工程化工作

## 致谢

- [nsys-ai](https://github.com/GindaChen/nsys-ai) 提供了 profile 读取、skill 化分析和 agent-oriented evidence 组织方面的重要上游思路
- NVIDIA Nsight Systems 提供 timeline、CUDA、NCCL 和 NVTX profiling 数据
- NVIDIA Nsight Compute 用于更细粒度的 kernel 级分析
- PyTorch 是这个项目当前主要面向的执行环境

## 许可证

[MIT](./LICENSE)
