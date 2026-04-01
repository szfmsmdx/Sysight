# Sysight

[English](./README.md)

面向 Nsight Systems profile 分析的轻量级、Torch-first、agent-first 后端。

`Sysight` 旨在把 Nsight Systems profile 转换成结构化结论、代码定位线索和下一步优化建议，重点服务于 PyTorch 训练与推理场景，是一个本地使用的轻量分析项目。

这个项目首先面向 agent 工作流：长期目标不是再做一个厚重的 profile 查看器，而是提供一个紧凑、可编排的后端，让 agent 能够调用它完成 profile、analysis 和后续迭代。

正常使用方式是对 agent 直接说一句话，把待分析的 profile 路径交给它。CLI 只是 agent 底层调用的后端入口，不是主要的用户交互界面。

## 功能

- 面向训练和推理场景的 Torch-first 分析
- 基于 SQLite 的分析流水线，强调低上下文和 skill 化拆分
- 继承 `nsys-ai` 中有价值的分析能力，但不保留其更重的 Web/TUI/chat 外层
- 内置 kernel hotspot、iteration timing、idle gaps、launch overhead、memory transfers、NCCL anomalies、overlap、NVTX attribution、code-location hints、MFU、theoretical FLOPs 等分析能力
- 支持输出 Markdown 和 JSON，方便交给后续 agent 或更深一步的调查流程
- 提供早期的多 worker 并行 orchestration demo

## 当前状态

目前仓库主要聚焦在 analysis 后端：轻量级 profile 检查、skill 化分析、MFU/FLOPs 辅助能力，以及基于 Nsight Systems 数据生成报告。项目目标是尽量贴近 `nsys-ai` 中真正有价值的分析部分，但默认不继承其更重的 Web 优先交互层。围绕真实用户项目的自动化 profiling、更自然的 agent 交互回路，以及更深的 investigation workflow 仍在 roadmap 中。

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

3. agent 会自动调用分析后端，并在终端返回一份三段式简要报告：

- `结论`
- `问题`
- `下一步行动建议`

4. 同时在 `outputs/` 下生成完整 Markdown 报告，例如：

```text
outputs/<profile>.report.md
```

## Agent 工作流

这个项目的目标方向是 agent-first。

当前仓库里的主路径是：

1. 用户用自然语言提供 profile 路径
2. agent 自动调用 analysis backend
3. agent 返回 `结论 / 问题 / 下一步行动建议`
4. Sysight 落一份完整的 `report.md` 报告，供后续深入分析

所以 CLI 现在存在，是为了支撑 agent 的底层执行，不是主要的人机交互形式。

## 后端调试命令

如果只是调试底层分析能力，可以在仓库根目录手动运行：

```bash
PYTHONPATH=src python3 -m sysight --help
PYTHONPATH=src python3 -m sysight analyze path/to/profile.sqlite
PYTHONPATH=src python3 -m sysight skill list
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
└── workspace/                   # 用户项目工作区
```

## 路线图

- [ ] 增加 profiling 阶段，能够包装真实 Torch 入口并产出标准 Nsight Systems capture
- [ ] 提升对用户代码库的理解能力，包括入口、模块和数据路径
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
