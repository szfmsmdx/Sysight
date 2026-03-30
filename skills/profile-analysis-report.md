---
name: profile-analysis-report
description: 当用户直接提供 Nsight Systems 的 `.sqlite`、`.sqlite3` 或 `.nsys-rep` 路径，并希望立刻得到分析结论、问题定位和完整 Markdown 报告地址时使用。运行仓库脚本生成终端摘要、Markdown 报告和 findings JSON，并在回复里明确给出关键结论与报告路径。
---

# Profile Analysis Report

这个 skill 负责把“用户丢一个 profile 文件路径进来”这件事变成一个标准化流程：

1. 运行分析
2. 在终端里快速给出高价值结论
3. 告知完整 Markdown 报告和 findings JSON 的地址

不要把结果只停留在“我已经分析了”，必须给出结论和文件路径。

## 何时触发

当用户：

- 直接给出一个 `.sqlite` / `.sqlite3` / `.nsys-rep` 路径
- 说“帮我分析这个 profile”
- 希望“给我一个结论 + 完整报告地址”

## 标准执行步骤

### 1. 校验输入

- 确认路径存在
- 若用户没有指定 GPU，默认按全局概览 + target GPU 0 处理

### 2. 执行分析

优先使用仓库脚本：

```bash
scripts/run-profile-analysis.sh <profile-path> [gpu_id]
```

这个脚本会：

- 调用 `PYTHONPATH=src python3 -m nsys_agent analyze`
- 在 `outputs/` 下生成 Markdown 报告
- 在 `outputs/` 下生成 findings JSON
- 在终端打印最终文件地址

如果必须手动执行，则使用：

```bash
PYTHONPATH=src python3 -m nsys_agent analyze <profile-path> --markdown <markdown-path> --findings <findings-path>
```

### 3. 读取结果

至少读取三类信息：

- 终端分析输出中的核心结论
- Markdown 报告中的“存在问题 / 下一步行动指南”
- 生成文件的绝对路径

### 4. 回复用户

回复里必须同时包含：

- 简短结论：2～5 条最重要的发现
- 问题定位：至少给出 1～2 个具体的代码 / NVTX / frame 位置
- 报告地址：Markdown 报告绝对路径
- 如有 findings JSON，也给出它的路径

不要只复述“报告已生成”，必须先给结论，再给地址。

## 输出约束

默认按下面的顺序组织回复：

### 终端结论

- 概括 profile 的整体状态
- 点出最严重的 3～5 个问题
- 对每个问题尽量附带数值证据

### 定位信息

- 若有 NVTX region，优先给 NVTX region
- 若有 sampled stack / runtime 线程，补充 frame 名称
- 若两者都有，优先写成“问题 -> NVTX -> frame”

### 报告地址

- Markdown 报告绝对路径
- findings JSON 绝对路径

## 偏好规则

- 优先复用现有 `analyze` 流程，不要临时绕过它自己拼 SQL
- 优先告诉用户“最值得先看的问题”，不要一次塞太多噪音
- 如果 iteration 检测是 heuristic fallback，要明确告诉用户
- 如果没有 NVTX 归因，要明确说明当前定位主要来自 runtime + sampled stack

## 示例命令

```bash
scripts/run-profile-analysis.sh test/basemodel_8gpu.sqlite
scripts/run-profile-analysis.sh test/basemodel_8gpu.sqlite 0
```

## 示例回复骨架

```markdown
分析已经完成。这个 profile 的主要问题集中在同步开销、持续 H2D 传输和 NCCL 热点。

- GPU 0 的主要热点是 `...`，占 `...%`
- 发现 `...` 个 idle gap，总计 `...ms`
- H2D 传输 `...MB / ...ms`
- 同步 API `...` 次，总计 `...ms`

问题定位上，优先看：
- NVTX `...`
- frame `...`

完整报告：
- Markdown: `/abs/path/to/report.md`
- Findings: `/abs/path/to/findings.json`
```
