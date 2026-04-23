---
name: nsys-investigation
description: Sysight analyzer nsys 调查 harness 入口。供人类开发者阅读，非 Codex 入口。
---

# Nsight 调查 Harness 说明

本目录是 Sysight analyzer 调用 Codex 进行 nsys 性能调查的 harness。

## 文件说明

| 文件 | 用途 |
|------|------|
| `TASK.txt` | Codex 唯一入口 prompt，包含任务说明、CLI 工具列表、调查原则、输出 schema |
| `memory/workspace.md` | workspace 专属记忆：关键文件功能说明、nsys 总结；workspace 更换时全量刷新 |
| `memory/experience.md` | 通用 nsys 分析经验：错题本、wrapper 规则、查询技巧；跨 workspace 复用 |

## Codex 调用方式

由 `sysight/analyzer/nsys/investigation.py` 自动构建 prompt 并调用：
- 只读取 `TASK.txt` 作为 prompt 模板
- TASK.txt 内嵌了所有必要规则，Codex 按需读取 memory 文件

## 维护说明

- 修改调查规则：编辑 `TASK.txt` 中的"调查原则"部分
- 修改输出格式：编辑 `TASK.txt` 中的"输出格式"部分
- 添加通用经验：直接编辑 `memory/experience.md`
- workspace 切换：清空并重写 `memory/workspace.md`
- 调查结束请不要忘记维护 `memory/*`