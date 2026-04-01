---
name: profile-analysis-report
description: 当用户直接提供 Nsight Systems 的 `.sqlite`、`.sqlite3` 或 `.nsys-rep` 路径，并希望 agent 直接给出“结论 / 问题 / 下一步行动建议”以及完整 `report.md` 路径时使用。运行仓库脚本生成终端摘要和 Markdown 报告，并在回复里按三段式格式输出关键结论与报告路径。
---

# Profile Analysis Report

这个 skill 负责把“用户丢一个 profile 文件路径进来，然后说帮我分析一下”这件事变成一个标准化流程：

1. 运行分析
2. 在终端里输出 `结论 / 问题 / 下一步行动建议`
3. 告知完整 Markdown 报告地址

不要把结果只停留在“我已经分析了”，必须给出结论和文件路径。

## 何时触发

当用户：

- 直接给出一个 `.sqlite` / `.sqlite3` / `.nsys-rep` 路径
- 说“帮我分析这个 profile”
- 希望“给我一个终端简要结论 + 完整报告地址”

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

- 调用 `PYTHONPATH=src python3 -m sysight analyze`
- 若输入是 `.nsys-rep`，会先触发底层自动导出同名 `.sqlite`
- 在 `outputs/` 下生成 Markdown 报告
- 可选生成 findings JSON
- 在终端打印最终文件地址

如果必须手动执行，则使用：

```bash
PYTHONPATH=src python3 -m sysight analyze <profile-path> --markdown <markdown-path> --findings <findings-path>
```

如果只想先把 `.nsys-rep` 显式导出成 `.sqlite`，使用：

```bash
scripts/export-nsys-sqlite.sh <profile.nsys-rep> [output.sqlite]
```

### 3. 读取结果

至少读取三类信息：

- 终端分析输出中的核心结论
- Markdown 报告中的“存在问题 / 下一步行动指南”
- 生成的 `report.md` 绝对路径

如果输入是 `.nsys-rep` 且发生了自动转换，还应留意对应的 `.sqlite` 路径，便于后续复用。

### 4. 回复用户

回复里必须同时包含：

- `结论`：2～5 条最重要的发现
- `问题`：列出最值得优先处理的问题，尽量附数值证据
- `下一步行动建议`：给出 2～4 条可执行建议
- 报告地址：Markdown 报告绝对路径

不要只复述“报告已生成”，必须先给结论，再给地址。

## 输出约束

默认按下面的顺序组织回复：

### 结论

- 概括 profile 的整体状态
- 点出最严重的 3～5 个问题
- 对每个问题尽量附带数值证据

### 问题

- 若有 NVTX region，优先给 NVTX region
- 若有 sampled stack / runtime 线程，补充 frame 名称
- 若两者都有，优先写成“问题 -> NVTX -> frame”

### 下一步行动建议

- 优先告诉用户最值得先处理的 2～4 个动作
- 建议要能直接执行，不要只写泛泛方向
- 如果 iteration 检测是 heuristic fallback，要明确说明

### 报告地址

- Markdown 报告绝对路径

## 偏好规则

- 优先复用现有 `analyze` 流程，不要临时绕过它自己拼 SQL
- 优先告诉用户“最值得先看的问题”，不要一次塞太多噪音
- 如果 iteration 检测是 heuristic fallback，要明确告诉用户
- 如果没有 NVTX 归因，要明确说明当前定位主要来自 runtime + sampled stack

## 示例命令

```bash
scripts/run-profile-analysis.sh test/basemodel_8gpu.sqlite
scripts/run-profile-analysis.sh test/basemodel_8gpu.sqlite 0
scripts/run-profile-analysis.sh test/basemodel_8gpu.nsys-rep
scripts/export-nsys-sqlite.sh test/basemodel_8gpu.nsys-rep
```

## 示例回复骨架

```markdown
结论

- GPU 0 当前最主要的时间消耗集中在 `...`
- profile 中发现 `...` 个明显 idle gap，总计 `...ms`
- 同步开销达到 `...ms`，已经影响主执行节奏

问题

- `Excessive Synchronization`：主要集中在 `NVTX ...`，相关 frame 为 `...`
- `Continuous H2D Transfers`：在 `...` 时间窗内持续出现，累计 `...MB`

下一步行动建议

- 先检查 `...` 附近是否存在不必要的同步点
- 评估 `...` 路径上的 H2D 是否可以提前或合并
- 给训练主循环补稳定的 NVTX iteration marker，方便后续精确定位

完整报告

- Markdown: `/abs/path/to/report.md`
```
