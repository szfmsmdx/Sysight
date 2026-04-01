---
name: parallel-orchestrator
description: 当主进程需要先理解仓库上下文，再把若干独立 analysis 子任务分发给多个 `codex exec` 子进程并行执行时使用。适用于“主窗口做调度，子进程做只读分析”的场景，当前仓库已提供可运行的 demo 脚本。
---

# Skill: parallel-orchestrator

这个 skill 的目标不是直接替代主 agent，而是给主 agent 一个最小可运行的主从 workflow。

用户侧的主交互仍然应该是自然语言，例如“帮我分析一下某个 sqlite / nsys-rep 文件”；这个 skill 只负责在主 agent 已经接到任务之后，如何把底层只读 analysis 子任务拆开并行执行。

在当前仓库里，agent 工作流先分成两条主路径：

1. `profile-only`：用户只给 profile 路径，先快速拿到分析摘要和 `report.md`
2. `workspace-aware`：用户同时给 `workspace/` 和 `program.md`，再升级到更强的项目语义归因

当前阶段建议优先把 `analyze` / `report` 跑顺，再考虑更深的 profiling 和优化闭环。

主进程在进入并行 orchestration 之前，建议先调用下面任一入口做分流：

```bash
PYTHONPATH=src python3 -m sysight route --profile <profile> --workspace <workspace> --program program.md
PYTHONPATH=src python3 -m sysight skill run workflow_router <profile> --workspace <workspace> --program program.md
```

如果用户需要一个 `program.md` 模板，当前仓库已在 `workspace/program.md` 放了一份可填写骨架，内容参考了 autoresearch 的 contract 思路，但调整为更适合 Sysight 的 analysis 工作协议。

主从 workflow 如下：

1. 主进程先读项目和任务背景
2. 主进程先判断当前是 `profile-only` 还是 `workspace-aware`
3. 主进程把独立子任务拆出来
4. 主进程启动多个 `codex exec` 子进程
5. 子进程各自执行一个明确的只读分析任务
6. 主进程收集输出并做后续聚合，最后仍然面向用户输出 `结论 / 问题 / 下一步行动建议`

当前版本先保证“跑起来”，不追求复杂调度、长会话交互或自动合并 patch。

## 适用场景

- 对同一个 profile 并行跑多个 analysis skill
- 代码库探索后，把多个独立只读问题分发给子进程
- 主窗口保留为调度视角，子进程只做窄任务分析

不适合：

- 多个子进程并行改同一批文件
- 需要复杂人工反馈回路的长任务
- 需要强一致上下文共享的深度协作编辑

## 当前仓库中的最小可运行入口

```bash
scripts/parallel-orchestrator-demo.sh <profile.sqlite|profile.sqlite3|profile.nsys-rep> [gpu_id]
```

这个 demo 会：

- 以当前窗口为“主进程”
- 围绕同一个 profile 启动 4 个子 `codex exec`
- 让子进程分别执行：
  - `top_kernels`
  - `nvtx_layer_breakdown`
  - `root_cause_matcher`
  - `code_location`
- 子进程只跑 skill，不生成最终 Markdown 报告
- 所有输出落到 `outputs/orchestrator/<profile>-<timestamp>/`

## 主进程职责

主进程在调用脚本前，应先做两件事：

1. 读懂当前目标
2. 用 `route` / `workflow_router` 判断输入模式
3. 选择可以并行的子任务

在当前 demo 中，主进程不做复杂智能拆分，而是固定分发 4 个 analysis task。

后续如果要扩展，应优先保持这个原则：

- 每个子进程只负责一个边界明确的任务
- 子任务之间尽量无共享写入
- 主进程负责最终取舍和结论整合
- 最终对用户的回答格式仍然应收敛成 `结论 / 问题 / 下一步行动建议`

## 子进程约束

当前 demo 中，每个子进程都遵循以下约束：

- 不修改文件
- 不生成 Markdown 报告
- 只运行一个指定的 `sysight skill run ...` 命令
- 最后返回简短文本总结

这保证了主从模式的第一版足够稳定，不会因为子进程自由度太大而失控。

## 运行步骤

### 1. 主进程准备

- 确认 `codex` CLI 可执行
- 确认 profile 文件存在
- 确认在仓库根目录或给出 profile 绝对路径

### 2. 启动 demo

```bash
scripts/parallel-orchestrator-demo.sh test/basemodel_8gpu.sqlite
```

或指定 GPU：

```bash
scripts/parallel-orchestrator-demo.sh test/basemodel_8gpu.sqlite 0
```

### 3. 查看结果

脚本会输出：

- 每个 worker 的状态
- 每个 worker 的 message 文件路径
- 每个 worker 的 log 文件路径
- 一个统一的 orchestrator 输出目录

## 输出目录约定

输出目录格式：

```text
outputs/orchestrator/<profile-stem>-<timestamp>/
```

目录下每个 worker 至少有两个文件：

- `<task>.message.txt`
- `<task>.log`

## 推荐主从流程

主进程建议按下面顺序工作：

1. 先本地阅读任务和仓库背景
2. 判断哪些问题适合并行
3. 调用 orchestrator 脚本启动子进程
4. 读取各 worker 输出
5. 在主窗口里统一汇总结论

## 限制

- 当前 demo 只是“并行只读分析”的最小形态
- 主进程和子进程之间还没有长会话消息回传机制
- 子任务列表是固定的，尚未根据任务动态裁剪
- 还没有 patch 级写入协同机制

## 后续可扩展方向

- 把固定 task 列表改成主进程动态生成
- 为 worker 输出增加结构化 schema
- 增加主进程自动聚合报告
- 在安全边界明确后，再引入受控的子进程写文件能力
