# Sysight — AI 驱动的 GPU 性能分析与自动优化

## Sysight 是什么

Sysight 是一个 AI Agent，它能**自动分析 GPU 程序的性能瓶颈，并直接生成可落地的代码修复**。

你给它一个 nsys profile（`.sqlite`）和一个代码仓库，它就能：

1. **读懂 profile** — 从 nsys 报告中提取 GPU 空闲、kernel 碎片、同步等待、内存搬运等信号
2. **定位源码** — 沿着调用链追踪，把每个性能问题精确到**文件、函数、行号**
3. **生成修复** — 对确认的问题直接输出 patch（代码替换），不是建议，是可执行的 diff
4. **验证效果** — 自动 apply patch → smoke test → 跑 timer 对比前后耗时

整个过程**全自动**，不需要人工介入。

---

## 能做什么

| 能力 | 说明 |
|------|------|
| **性能分析** | 从 nsys profile 自动发现 C1-C7 七大类性能问题 |
| **源码定位** | 每个 finding 精确到文件:函数:行号，附带 profile 证据 |
| **自动修复** | LLM 评判 finding 真伪，对真问题生成最小化 patch |
| **效果验证** | apply → smoke test → timer 对比，回归自动 revert |
| **知识积累** | 每次分析结果写入 wiki，后续分析自动参考历史经验 |
| **基准测试** | 内置 6 个 benchmark case，量化评估分析和优化能力 |

---

## 基准测试结果

### 分析能力（Analyze Benchmark）

在 6 个精心构造的 benchmark case 上，Sysight 需要从 nsys profile 中找出所有预埋的性能问题。

| Case | 描述 | 满分 | SOTA 得分 | 准确率 |
|------|------|------|-----------|--------|
| case_1 | 单卡训练：DataLoader + 同步 + 计算浪费 | 16 | 15/16 | 94% |
| case_2 | 多卡 DDP：通信 + 同步 + 配置 | 17 | 17/17 | 100% |
| case_3 | 推理服务：KV cache + batching | 17 | 12/17 | 71% |
| case_4 | 混合精度训练：AMP + checkpoint | 16 | 9/16 | 56% |
| case_5 | Pipeline 并行：micro-batch + 调度 | 17 | 17/17 | 100% |
| case_6 | 多模态训练：vision + text + fusion | 17 | 15/17 | 88% |

> SOTA 数据来源：`.sysight/bench-runs/sota.md`，统计口径为单 case 在某次 run 中的最高 Score/Total。

### 优化能力（Optimize Benchmark）

在 optimizer-bench 的 6 个 case 上，Sysight 需要评判 finding 真伪并生成正确 patch。

评分维度：

| 维度 | 权重 | 含义 |
|------|------|------|
| Correctness | 40 | patch 能否成功 apply + smoke test 通过 |
| Performance | 30 | 修复后 timer 是否有实际性能提升（delta < -5% 满分） |
| Judgment | 20 | 正确接受真 finding、拒绝假 finding（F1 分数） |
| Minimality | 10 | patch 改动行数是否在合理范围内 |

---

## 七大类性能问题（C1-C7）

Sysight 的分析覆盖 GPU 程序的全部性能维度：

| 类别 | 含义 | 典型问题 |
|------|------|---------|
| **C1** | Host Scheduling | DataLoader worker=0、pin_memory=False、线程配置 |
| **C2** | Kernel Launch Overhead | Python 循环触发大量小 CUDA kernel |
| **C3** | Synchronization | `.item()`、`.cpu()`、`cudaDeviceSynchronize()` |
| **C4** | Memory Copy | 热路径中的 `.to(device)`、H2D/D2H 搬运 |
| **C5** | Compute Inefficiency | 重复计算、可消除的 clone/cat/contiguous |
| **C6** | Communication | all_reduce/all_gather/barrier、DDP/FSDP 配置 |
| **C7** | Python Pipeline | DataLoader 内逐 sample 循环、json 序列化 |

---

## 与同类工具的区别

| | Sysight | nsys CLI | 人工分析 | 通用 Coding Agent |
|---|---|---|---|---|
| 读懂 nsys profile | ✅ 自动 | ✅ 原始数据 | ✅ 需要经验 | ❌ 不理解 GPU |
| 定位源码行号 | ✅ 精确 | ❌ | ✅ 手动 | ⚠️ 可能不准 |
| 生成可执行 patch | ✅ | ❌ | ✅ 手动 | ⚠️ 不保证正确 |
| 验证修复效果 | ✅ timer 对比 | ❌ | ✅ 手动 | ❌ |
| 知识积累 | ✅ wiki | ❌ | ⚠️ 靠记忆 | ❌ |
| 领域知识 | ✅ C1-C7 体系 | ❌ | ✅ 需要经验 | ❌ |

Sysight 不是通用 coding agent——它**只做 GPU 性能优化这一件事**，但把这件事做到端到端自动化。