# [LaunchReview] 20260514 AI 驱动的 GPU 训练性能自动优化

---

**「项目成果」**

构建了 Sysight，一个面向 GPU 训练性能优化的端到端 AI Pipeline。给定一份 nsys profile 和一个代码仓库，Sysight 自动完成从 profile 深度分析、问题定位到源码级 patch 生成、效果验证、经验沉淀的完整闭环，**全程无需人工干预**。

在 nanoGPT Shakespeare 字符级训练任务上的端到端 Demo 结果：

- **性能提升**：**三轮迭代**将单步迭代时间从 **8.05ms → 2.05ms**，提升 **74.5%**
- **自主决策**：12 个 trial，成功接受 10/12（83.3%）

---

# 一、背景

basemodel 的训练效率直接决定了模型的迭代速度。一个 training job 跑得越快，同样的时间窗口内我们能做的实验就越多，模型迭代周期就越短。

但现实情况是：**很多 training job 跑起来了，GPU 却没有被打满**。大量时间花在了 CPU 端的数据准备、同步等待、通信调度上，GPU 在旁边闲着。这些浪费是真实存在的，而且通常不小——在 nanoGPT 这个干净的小模型上，我们找到的几处 CPU 端问题就直接带来了 8% 的提升；在真实训练仓库中，类似的隐藏损耗只会更多。

**信息**

我们希望用 AI 把这件事自动化：**给一份 nsys profile，给一个代码仓库，AI 自己把优化做完**，不需要人一步步去操作。

## 1.1 分析和优化的完整流程是什么样的

要优化一个 training job 的 GPU 性能，完整的流程是这样的：

<!-- 流程图 -->

## 1.2 技术方案：基于 Harness Engine 的 Agent 架构

当前主流的 AI coding agent——Codex、Claude Code、Devin——底层都是同一套范式：一个固定的 **harness engine** 负责管理 LLM 调用循环、工具注册与执行、上下文压缩，具体的任务逻辑通过 prompt 和工具集来定义。引擎是确定的，能力边界由工具和 prompt 决定。

Sysight 也基于这个范式搭建。我们的 harness engine 流程很简单：

1. **定义 Task**：每个阶段（WARMUP、ANALYZE、OPTIMIZE、EXECUTE、LEARN）是一个独立的 `AgentTask`，包含 system prompt、user prompt、工具白名单和输出 schema
2. **AgentLoop 执行**：统一的 tool-calling 循环——调 LLM → 解析 tool call → 执行工具 → 喂回结果 → 继续，直到 LLM 输出结构化结果或触发停止条件
3. **PipelineRunner 编排**：按顺序串联各阶段，前一阶段的输出作为后一阶段的输入，状态通过结构化 artifact 传递

我们的搭建思路是**把"做什么"和"怎么做"分开**：

- **做什么**由 prompt 定义——告诉 LLM 当前阶段的目标、可用工具、输出格式
- **怎么做**由引擎保证——工具调用、上下文压缩、超时控制、错误恢复，全部在 AgentLoop 和 PipelineRunner 里硬编码

这样，新增一个阶段只需要写一份 prompt + 注册工具，不需要改引擎代码。反过来，优化上下文压缩策略、调整 token 预算，也只需要改引擎配置，不影响各阶段的 prompt 逻辑。

---

# 二、目标和达成情况

- **✅ 能找到问题。** 6 个测试 case 覆盖单卡、多卡 DDP、Pipeline 并行等场景，平均准确率 85%，其中 2 个 case 达到 100%。
- **✅ 能判断真假。** 对真实问题生成 patch，对假阳性正确拒绝——Demo 中 3 个 trial 的接受/拒绝决策全部正确。
- **✅ 能改对代码。** 生成的 patch 能成功 apply，通过 smoke test，避免部分运行崩溃问题。
- **✅ 能验证效果。** 每个 patch 实测采样取均值，有效的接受，退步的自动回滚。
- **✅ 能积累经验。** 每轮结果自动写入知识库，下一轮分析时自动引用——Demo 写入 5 条经验，iter-2 正确参考。
- **✅ 端到端跑通。** nanoGPT 训练任务，全程无人干预：14.12ms → 12.96ms，仅一步迭代时间降低 8.2%。
- **✅** 八卡 basemodel 训练 pipeline 也已跑通

---

# 三、解决方案

> 项目地址：[szfmsmdx/Sysight: 一个给 AI 训练任务自动跑 profiling、定位瓶颈、生成 patch、验证效果的端到端优化 agent。](https://github.com/szfmsmdx/Sysight)
> pipeline 设计文档见内部文档

Sysight 是一个 5 阶段的硬编码 Pipeline。每个阶段有明确的职责边界：该 LLM 推理的地方用 LLM，该确定性执行的地方用代码，两者不混淆。

```
┌─────────────────────────────────────────────────────────────────┐
│                        Sysight Pipeline                         │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  WARMUP  │───▶│ ANALYZE  │───▶│ OPTIMIZE │───▶│ EXECUTE  │  │
│  │  扫代码  │    │ 读profile│    │ 生成patch│    │ 实测验证 │  │
│  │  建索引  │    │ 找问题   │    │ 计划     │    │ 接受/回滚│  │
│  └──────────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘  │
│                       │               │               │        │
│                       ▼               ▼               ▼        │
│                  ┌──────────────────────────────────────┐      │
│                  │              LEARN                    │      │
│                  │   分析结论 + 优化经验 → 写入 wiki      │      │
│                  └──────────────────────────────────────┘      │
│                                                                 │
│   LLM 参与：ANALYZE / OPTIMIZE / LEARN                          │
│   纯代码：  WARMUP / EXECUTE                                    │
└─────────────────────────────────────────────────────────────────┘
```

下面按阶段说明每个部分的设计思路——重点不是"做了什么"，而是"遇到了什么问题所以设计成这样"。

## 3.1 WARMUP：先让系统"认识"这个仓库

在让 LLM 分析之前，代码侧先扫一遍仓库：建文件索引、找到训练入口命令、识别热路径上的关键函数。这一步完全不用 LLM——扫文件是确定性操作，不需要推理。

> 从入口文件出发 AST 解析 import 链，递归追踪出完整的**依赖图**，结果缓存下来后续阶段直接复用。

## 3.2 ANALYZE：让 LLM 读懂 profile，但不能让它淹死在数据里

> 详细设计见内部文档

这是整个 Pipeline 里最重的阶段。nsys profile 是一个几十张表的 SQLite 数据库，原始 kernel 记录上百万行。直接丢给 LLM 有两个问题：塞不进 context window，而且 LLM 会在海量细节里迷失方向。

我们的做法是两层处理：

1. **第一层，预注入摘要。** 在 LLM 启动前，代码侧先算出一份高信噪比的摘要——**top kernel 耗时排行、GPU 空闲间隙分布、内存搬运统计 ... ...**——直接注入到 prompt 里。这份摘要通常就能让 LLM 直接锁定方向，**省掉大量摸索性的工具调用**。
2. **第二层，结构化工具。** LLM 不直接写 SQL，而是通过 8 个 nsys_sql_* 工具按需查询细节（kernel 明细、同步点、NCCL 通信等）。配合 Scanner 源码引擎（scanner_read / scanner_search / scanner_callers），LLM 能从 profile 里的 kernel 名称一路追溯到触发它的 Python 源码行。

这套"预计算摘要 + 按需查询"的组合，把一次 ANALYZE 的工具调用控制在 **15–20 轮**，覆盖 C1–C7 七类 GPU 性能问题[^1]（数据加载、kernel launch、同步阻塞、内存搬运、计算低效、通信、Python 管线）。

<!-- 摘要 prompt 示意图 -->
摘要prompt示意图

## 3.3 OPTIMIZE：LLM 只出计划，不碰文件

> 详细设计见内部文档

optimizer 角色简单来说主要担任两个任务：

- 否定 analyzer 的 finding
  - ANALYZE 输出的 finding 不一定是真的——profile 里的信号可能是**假阳性**，也可能代码已经处理过了。optimizer 阶段的 LLM 需要逐个判断：这个问题真实存在吗？值得改吗？
- 写代码
  - **LLM 只能读代码，不能写代码。** 它输出的是一份 patch 计划（"把哪个文件的第几行到第几行替换成什么"），实际的文件写入由 EXECUTE 阶段完成。
  - 为什么不让 LLM 直接改文件？因为一旦 LLM 可以直接写，就没有了 hash 验证（写错了不知道）、没有 revert 机制（写坏了不好恢复）、没有测量（不知道是否真的有效）。把"生成计划"和"执行计划"分离，每一步都有明确的责任归属。
  - 另外，patch 计划里包含目标代码行的 hash——apply 时如果 hash 不匹配（文件被改过、LLM 行号数错），直接拒绝，不静默覆错。

<!-- patch 示意图 -->
patch 示意图，以 patch list 为单独进行逐步的测试以及 git commit 合并代码

## 3.4 EXECUTE：只信实测数字

OPTIMIZE 阶段 LLM 生成了 patch 计划，但计划对不对、有没有效果，LLM 自己说了不算。如果让 LLM 来判断"这个 patch 有没有用"，它大概率会根据"理论上应该有效"来下结论——trial-003 就是典型，LLM 认为 deferred sync 能 overlap D2H，但实测退步 5.37%。

所以 EXECUTE 被拆成一个独立阶段，核心逻辑就一条：**LLM 出计划，代码做验证，两者不混淆。** EXECUTE 完全不调 LLM，全部是确定性代码——跑脚本、读日志、比数字、做决策，每一步都是 `if` 语句和 shell 脚本，没有推理，没有"可能"。

每个 trial 的流程：

1. 创建 git worktree，和主仓库完全隔离
2. 跑一次训练脚本，脚本内部迭代 ~20 步，每步记录 iteration_ms【**warm 阶段会定向读取到相关内容，以及有 instrument 阶段会确保这个字段的有效，不会导致一些推理情况下没有 iteration 字段的问题**】，取均值作为 **baseline**
3. apply patch + hash 验证（hash 不匹配直接拒绝）
4. smoke test（语法检查 + import）
5. 再跑一次训练脚本，同样 ~20 步取均值，和 baseline 比较
6. 判断metric：
   1. 变快了 → cherry-pick 到主仓库；
   2. 没变或退步 → git reset 回滚

## 3.5 LEARN：让经验留下来

> 具体设计文件见内部文档

LEARN 在 Pipeline 中被调用两次——ANALYZE 之后记录分析发现，EXECUTE 之后记录优化结论。经验写入两级 wiki：

- **workspace 级**：该仓库特有的信息，比如入口命令、热路径文件、之前踩过的坑
- **llmwiki 级**：跨仓库可复用的通用经验，比如"numpy 数据加载要向量化"、"MPS 上 D2H defer 没有收益"

一个重要的设计原则：**只有 LEARN 阶段可以写 wiki，ANALYZE 和 OPTIMIZE 只能读。** 这保证了 wiki 的质量——只有经过实测验证的结论才能写入，而不是 LLM 的"我觉得可能是这样"。

**拒绝的经验同样有价值。** trial-003（deferred loss.item()）退步 5.37% 被拒绝后，这条经验被写入 llmwiki/loss-item-defer-risk.md，下一个仓库遇到同样的 C3 finding 会直接跳过，不会重复踩坑。

<!-- 优化经验总结示意图 -->
优化部分经验总结示意图

<!-- 通用分析经验总结示意图 -->
通用分析经验总结示意图

## 3.6 上下文管理：长对话怎么不崩

> 详细设计见内部文档
> 参考 claude code 和 codex 的设计逻辑

ANALYZE 一次 15–20 轮对话，累计 ~900K tokens。直接截断不行——LLM 需要记得前面查了什么；不压缩也不行——context window 很快就满了。

我们用了**四级渐进式压缩**，根据 token 压力从轻到重依次触发：

1. **token 利用率到 50%**：清除旧的工具调用结果（`scanner_read`、`nsys_sql_*` 的返回值），这些信息 LLM 已经消费过了
2. **到 70%**：用模板摘要替换旧工具结果，比如 "Turn 5: 查了 top kernels，最慢是 ncclAllReduce"
3. **到 80%**：确定性删除中间区间的消息，不调 LLM，保护 system prompt 和最近的上下文
4. **到 95%**：激进压缩，注入 "SESSION RECOVERY" 恢复消息让 LLM 继续工作

中间踩了不少坑。比如压缩后 token 数不降反升——因为当前轮的工具结果本身就很大（`scanner_read` 返回了一个 500 行的文件），压缩旧内容腾出来的空间立刻被新结果填满。不加 **circuit breaker** 会陷入死循环，我们在连续触发 N 次后强制停止压缩，让 AgentLoop 尽快输出结果。

---

# 四、端到端 Demo：nanoGPT 迭代记录

> 完整实录见内部文档

## 4.1 测试环境

| 项目 | 配置 |
| --- | --- |
| 硬件 | NVIDIA A100-SXM4-80GB，单卡 |
| 模型 | nanoGPT，4 层 Transformer，n_embd=128，n_head=4，block_size=128 |
| 参数量 | 0.80M，约 600 行 Python |
| 任务 | Shakespeare 字符级语言模型训练，max_iters=24 |
| 精度 | bfloat16 |
| LLM | deepseek-v4 pro（analyze / optimize）、deepseek-v4 flash（learn） |
| 评测指标 | 训练循环单步迭代时间 iteration_time_ms（越低越好） |
| Profiling | Nsight Systems 2025.4.1，trace=cuda,nvtx |

## 4.2 总体结果

三轮迭代，12 个 patch（9 accepted + 3 rejected），单步迭代时间从 **8.05ms 降至 2.05ms，累计降低 74.5%**。

```
8.05ms ──Iter-1──▶ 5.39ms ──Iter-2──▶ 2.09ms ──Iter-3──▶ 2.05ms
        -33.0%            -61.2%            -1.9%
                                        累计 -74.5%
```

## 4.3 各轮详细记录

### Iter-1

基线 8.05ms，手动分析代码热点后合并提交 6 项优化，降至 5.39ms（-33.0%）。6 项 patch 合并为一个 commit 提交，无逐项独立测量数据。

| # | 类别 | 问题 | 修改 | 文件 |
| --- | --- | --- | --- | --- |
| 1 | C7 | torch.compile 未启用，细碎 kernel 6556 个 | compile=True | config/sysight_baseline.py:27 |
| 2 | C7 | warmup 仅 forward，首次 backward 仍触发重编译 | warmup 补齐 forward+backward | train.py:284-291 |
| 3 | C2 | get_batch 逐样本 list comprehension，64 次 Python 循环 | numpy 高级索引向量化批量处理 | train.py:149-166 |
| 4 | C5 | 每次 forward 重创 torch.arange 生成 position tensor | register_buffer 预分配 pos buffer | model.py:139+175 |
| 5 | C4 | 每 batch 重新 np.memmap 打开文件 | reload_memmap_each_batch=False | config |
| 6 | C3 | loss.item() 每步触发 CPU-GPU 同步 | log_interval=10，减少同步频率 | config/sysight_baseline.py:24 |

**Iter-1 结果**：8.05ms → 5.39ms（-33.0%）。

### Iter-2

**Analyzer 定位**：读取 Iter-1 的 nsys profile（GPU idle 99.7%，14s 内仅 6.6ms 实际计算），定位 18 个 finding，覆盖 C1-C7 全类别。关键发现：

| # | 类别 | 问题 | 位置 | 优先级 |
| --- | --- | --- | --- | --- |
| 1 | C7 | torch.compile 默认 mode='default'，编译开销占 trace >80%（4 个大间隙共 ~12.8s） | train.py:243 | high |
| 2 | C1 | 自定义数据加载无流水线并行，GPU 等待数据期间完全空闲 | train.py:149 | medium |
| 3 | C7 | bfloat16 下 GradScaler 为空操作但仍被逐迭代调用 4 次 | train.py:355-364 | low |
| 4 | C5 | get_batch 中 np.arange(block_size) 每次调用重新创建 | train.py:156 | medium |
| 5 | C4 | pin_memory() 产生额外 CPU 拷贝，H2D 实际带宽仅 9.93 GB/s | train.py:163 | medium |
| 6 | C3 | loss.item() 每 log_interval 触发 CPU-GPU 同步 | train.py:377 | medium |

**Optimizer 代码修改**：LLM 逐个验证 finding 真伪，生成 3 个 trial：

| Trial | 对应 Finding | Patch | 改动说明 |
| --- | --- | --- | --- |
| trial-001 | C7 编译开销 | compile mode='reduce-overhead' | 启用 CUDA graph，将逐 kernel launch 打包为单次 graph replay |
| trial-002 | C5 + C7 | GradScaler bypass + np.arange 预计算 | bf16 下跳过 4 次 no-op 调用 + off = np.arange(block_size) 提升到模块级 |
| trial-003 | C4 | 消除 .astype(np.int64) 中间数组 | uint16 直接上 GPU 再转 int64，省去 CPU 侧 numpy 分配 |

**Execute 执行效果**：

| Trial | 策略 | 优化前 | 优化后 | 变化 | 结果 |
| --- | --- | --- | --- | --- | --- |
| trial-001 | CUDA graph（reduce-overhead） | 5.39ms | 2.16ms | **+59.9%** | ✅ accepted |
| trial-002 | GradScaler bypass + arange 缓存 | 2.16ms | 2.09ms | **+3.2%** | ✅ accepted |
| trial-003 | .astype 消除 | 2.09ms | 2.25ms | -7.7% | ❌ rejected |

trial-003 被正确拒绝：uint16→GPU→int64 路径在小 batch（32×128=4K tokens）下，DMA 启动延迟抵消了类型转换省下的时间，且 .pin_memory() 后的异步传输路径被破坏。

**LEARN 阶段**：更新 workspace overview，创建 4 个跨 workspace 经验页面：torch-compile-short-run-overhead、gradscaler-noop-bfloat16、loss-item-implicit-sync、custom-data-loading-without-prefetch。

**Iter-2 结果**：5.39ms → 2.09ms（-61.2%）。

### Iter-3

**Analyzer 分析**：重新采集 nsys profile（Iter-2 优化后代码），GPU 利用率仍仅 0.045%（14.60s 内仅 6.57ms 实际计算）。compile reduce-overhead 的 CUDA graph 捕获开销依然主导。精确定位 11 个 finding：

| # | 类别 | 问题 | 优先级 |
| --- | --- | --- | --- |
| 1 | C7 | reduce-overhead 的 CUDA graph 捕获在 24 iter 短运行中仍占主导（4.92s max gap） | high |
| 2 | C7 | get_batch() 同步 CPU 管线无后台预取，H2D 仅 9.38 GB/s | high |
| 3 | C4 | H2D 路径含冗余 pin_memory 中间拷贝 | medium |
| 4 | C7 | torch.randint 每 batch 在 CPU 生成随机索引，再 .numpy() 回 CPU | medium |
| 5 | C5 | attention 输出 .transpose().contiguous().view() 触发 D2D 拷贝（134 次，14.32MB） | low |
| 6 | C3 | loss.item() 同步（与上轮相同） | medium |

**Optimizer 代码修改**：

| Trial | 对应 Finding | Patch | 改动说明 |
| --- | --- | --- | --- |
| trial-001 | C4 + C7 | GPU 数据预加载 | tiny Shakespeare 数据集（~1MB）启动时一次性上传 GPU，get_batch 直接在 GPU tensor 上索引，消除每 batch 的 CPU numpy / astype / pin_memory / H2D 全链路 |
| trial-002 | C7 | compile=False | 关闭 torch.compile，测试短迭代下 eager mode 是否由于省去 graph capture 而更优 |
| trial-003 | C5 + C7 | GPU arange 缓存 + .reshape() | 预计算 GPU offset tensor，model.py 用 .reshape() 替代 .contiguous().view() |

**Execute 阶段**：

| Trial | 策略 | 优化前 | 优化后 | 变化 | 结果 |
| --- | --- | --- | --- | --- | --- |
| trial-001 | GPU 数据预加载 | 2.26ms | 2.05ms | **+9.3%** | ✅ accepted |
| trial-002 | compile=False | 2.05ms | 6.23ms | -203.9% | ❌ rejected |
| trial-003 | arange 缓存 + reshape | 2.05ms | 2.05ms | 0% | ✅ accepted |

trial-002 被正确拒绝：compile=False 后 eager mode 每 iter 需逐 kernel launch，虽然省去了 ~14s 的 graph capture，但 per-iteration 延迟从 2ms 暴涨到 6ms（+203%）。trial-003 零变化无退化——torch.compile 已对 .contiguous().view() 做融合优化，.reshape() 在编译路径下等效。

**Iter-3 结果**：2.26ms → 2.05ms（-9.3%，GPU 数据预加载贡献全部收益）。

## 4.4 关键发现

1. **CUDA graph**：torch.compile(mode='reduce-overhead') 将 5.39ms 砍到 2.16ms（+59.9%）。对于 tiny model（4 层 / 128 embd），per-iteration kernel launch overhead 是主要瓶颈——CUDA graph 将数百次 kernel launch 合并为一次 graph replay。
2. **GPU 驻留数据收益显著**：Iter-3 将 tiny 数据集一次性上传 GPU，get_batch 变为纯 GPU tensor 索引（D2D 替代 H2D），又省下 9.3%。对于 block_size=128、batch_size=32 的 workload，每 batch 仅 32KB 数据，DMA 启动延迟远大于传输本身。
3. **compile=False 在短迭代中是反模式**：Iter-3 trial-002 证明，即使 warmup + compile 花了十几秒，per-iteration 的 kernel fusion 和 CUDA graph replay 收益在 24 iter 内就远超编译成本。eager mode 的逐 kernel launch 在 tiny model 上反而慢 3 倍。
4. **Pipeline 的 accept/reject 决策全部正确**：3 个 rejected trial（.astype 消除、compile=False ×2）的退步原因都被正确识别和自动回退，没有误接受任何导致 regression 的 patch。git worktree 安全网机制确保每次回退干净彻底。

## 4.5 小结

三轮 Iteration、12 个 patch、9 个 accept、3 个 reject，nanoGPT 单步训练时间从 8.05ms 降至 2.05ms，累计优化 74.5%。Pipeline 在每个阶段均产出可验证的收益，且自动拒绝了所有会导致退化的 patch——**三个 trial 的 accept/reject 决策全部正确，无一误判**。证明了 analyze → optimize → execute → learn 闭环在真实 A100 GPU 训练场景下的有效性。

---

# 五、基准测试体系

> 详细设计见内部文档

Sysight 的两个核心能力——**"找问题"和"修问题"**——分开评估，互不干扰。Analyze Benchmark 测分析准确率，Optimize Benchmark 测判断和修复能力，后者用预构建的 finding 数据，不依赖 ANALYZE 的实际输出。

## 5.1 Analyze Benchmark

6 个场景，每个包含预埋问题的源码 + nsys profile + ground truth（精确到文件/函数/行号，四要素全匹配才得分）：

| Case | 场景 | 准确率 |
| --- | --- | --- |
| case_1 | 单卡训练（DataLoader + 同步 + 计算浪费） | **94%** |
| case_2 | 多卡 DDP（通信 + 同步 + 配置） | **100%** |
| case_3 | 推理服务（KV cache + batching） | 71% |
| case_4 | 混合精度训练（AMP + checkpoint） | 56% |
| case_5 | Pipeline 并行（micro-batch + 调度） | **100%** |
| case_6 | 多模态训练（vision + text + fusion） | **88%** |

平均 **85%**，case_2 和 case_5 满分。case_4（AMP）偏低，profile 信号不如 C1/C3/C7 直观，后续需要补充 AMP 相关的分析规则。

## 5.2 Optimize Benchmark

**评估目标**：给定 findings（含真假混合），能否正确判断真伪并生成有效 patch。与 Analyze Benchmark 解耦——使用预构建的 `analyze_raw.json`，不依赖 ANALYZE 的实际输出。

6 个 case，每个包含预埋问题的源码 + 预构建 findings（真问题 + 假阳性混合）：

| Case | 场景 | 真 finding | 假 finding | 核心考点 |
| --- | --- | :---: | :---: | --- |
| case_1 | GPT 训练管线 | 5 | 1 | 注意力掩码缓存、loss.item() 同步、重复计算 |
| case_2 | 多卡 DDP 训练 | 4 | 1 | NCCL 通信瓶颈、梯度同步策略 |
| case_3 | LLM 推理服务 | 5 | 1 | KV cache 优化、attention 计算效率 |
| case_4 | 数据管线 + checkpoint | 5 | 1 | DataLoader 瓶颈、checkpoint I/O 开销 |
| case_5 | 混合精度训练 | 4 | 1 | AMP 配置、梯度累积策略 |
| case_6 | 多模态训练 | 4 | 1 | 跨模态 attention、评估管线瓶颈 |

四维评分，满分 100：**Correctness**（patch 能 apply + smoke test 通过，40 分）、**Performance**（实测有提升，30 分）、**Judgment**（正确接受真 finding、拒绝假 finding，20 分）、**Minimality**（patch 不冗余，10 分）。

> 详细评分规则见 [Benchmark 体系设计](docs/07-benchmark.md)。

---

# 六、总结与反思

## 6.1 搭建 Agent 的问题

把 Sysight 从零搭起来，核心挑战不是"能不能调 LLM"，而是**怎么让 LLM 在长链路里稳定地产出可消费的结构化结果**。

在具体搭建 agent 的时候，主要遇到这两个问题：

**1. 如何设计 Pipeline 流程？** 目前有很多设计范式——ReAct、Plan-Execute、Multi-Agent——选择什么范式去搭建 Pipeline？

GPU 性能优化的链路太长（读 profile → 定位瓶颈 → 生成 patch → 实测验证 → 沉淀经验），纯 ReAct 扛不住——context 线性膨胀，LLM 跑到后半段已经记不清前面查了什么，而且容易在"分析"和"改代码"之间反复横跳。

我们的选择是**混合架构**：Stage 级别用 Plan-Execute，单个 Agent 内部用 ReAct。Plan-Execute 把 5 个阶段拆开，每个阶段有明确的输入/输出契约，LLM 不需要跨阶段记忆上下文。ReAct 用在 ANALYZE 和 OPTIMIZE 内部，让 LLM 根据中间结果动态调整调查方向——这个探索过程无法预先写成固定脚本。

**2. 为什么要设计这个 Pipeline？** 能否极简化做成一段 skill，让 Codex、Claude Code 这类终端 agent 工具直接运行？

GPU 性能优化不是纯代码任务——读 nsys profile、跑脚本取 baseline、hash 校验、git worktree 隔离，这些是确定性操作，不需要推理。如果写成 skill，每一步都变成 LLM 的 tool call，不仅慢，而且不可靠——LLM 可能跳过 hash 校验，可能错误解读测量结果（trial-003 就是典型：LLM 认为有效，实测退步 5.37%）。

Pipeline 的做法是把"确定性执行"和"LLM 推理"严格分开：WARMUP 和 EXECUTE 不走 LLM，用代码保证正确性；ANALYZE、OPTIMIZE、LEARN 只在需要理解判断的地方介入。这样关键路径上不会有"LLM 忘了做某一步"的风险，而且每个阶段可以独立迭代。

**上下文膨胀是第三个硬问题。** 一次 ANALYZE 需要 25–30 轮工具调用，nsys SQLite 查询结果动辄几千行，如果不做压缩，context window 很快就满了。我们参考了 MiniCode 的压缩策略，实现了四级渐进式压缩：Microcompact（清除旧工具结果）→ Persistence（落盘持久化）→ Compaction（LLM 摘要压缩）→ Snip（物理删除中间消息）。每一级有独立的触发阈值，逐级递进，避免过早丢失信息。

**工具调用稳定性是第四个问题。** LLM 有时会传错参数格式、调用不存在的工具、或者在单轮里重复调用同一个工具几十次。我们在 ToolRegistry 层加了调用次数上限和参数校验，AgentLoop 层加了重复调用检测——连续 3 次相同工具调用直接终止，避免死循环烧 token。

**输出格式一致性是第五个问题。** ANALYZE 需要输出结构化的 finding 列表，OPTIMIZE 需要输出可执行的 patch 计划。如果 LLM 输出的 JSON schema 不对，下游阶段直接崩溃。我们给每个 Task 定义了 `response_schema`，AgentLoop 在收到 LLM 输出后做 schema 校验，不通过则把错误信息喂回 LLM 让它修正，最多重试 3 次。

## 6.2 运行时问题

这里问题比较多，总结几个占用时间比较多的地方。

**Memory 管理。** 这里主要指 LLM 在长链路中三个典型的 memory 问题：上下文撑爆、跨阶段遗忘、幻觉。

上下文撑爆的解法是四级渐进式压缩（3.6 节），这里不再重复。

跨阶段遗忘更隐蔽。Pipeline 的每个阶段是独立 AgentLoop，上下文不共享——WARMUP 扫出了无用文件列表和热路径，但 ANALYZE 启动后 LLM 完全不记得，反复 `scanner_read` 无关文件，白白烧 token。根因不是 LLM "忘了"，而是这些信息根本没进它的上下文。

解法是**预注入，不依赖记忆**。每个阶段启动时，`build_memory_brief` 把 wiki 中的 workspace overview 和全局 experience 压缩成摘要，直接写入 user prompt。LLM 不需要主动去读 wiki——相关信息一开始就在眼前。对于"排除无用文件"这类硬规则，直接写在 `scanner_read` 工具描述里，从工具层阻断。

幻觉问题则是 LLM 会编造不存在的文件路径或行号。防线有两层：patch 计划中每行代码带 hash，apply 时 hash 不匹配直接拒绝；finding 输出后经 `response_schema` 校验，不通过则喂回 LLM 修正，最多 3 轮。

**稳定输出。** 当 tool-calling 轮数多起来后，LLM 容易忘记输出格式要求，导致 JSON 解析失败。我们通过三层手段兜底：宽解析（解析代码中定义更宽松的字段匹配）→ API 层 `response_format` 约束 → 最多 3 轮重新生成，确保最终一定能拿到合法 JSON。

**Agent 效率。** 环境依赖是第一个门槛——nsys CLI 只支持 Linux + CUDA，Mac 本地开发时 profile refresh 会静默降级。git worktree 生命周期管理也容易出错，崩溃后残留的 worktree 需要手动清理。MPS vs CUDA 的结论不可互推（trial-003 在 MPS 上退步 5.37%，CUDA 多流环境下可能相反），小模型上的优化结论也不代表大模型。

## 6.3 学习总结

在搭建过程中，也看了不少社区博客和论文（Anthropic 的 *Building Effective Agents*、Lilian Weng 的 *LLM Powered Autonomous Agents* 等），发现很多痛点不是 Sysight 独有，而是整个 agent 领域的共性问题。

**从简开始，不要过度设计。** Anthropic 的核心观点：最成功的 agent 用的都是简单模式，不是复杂框架。agent 本质是用延迟和成本换任务质量，这个 tradeoff 要谨慎。我们在 Sysight 上也验证了——最初想把 ANALYZE 和 OPTIMIZE 合并成一个 ReAct loop，结果 context 膨胀失控，LLM 在分析和改代码之间反复横跳。拆成 5 个独立 stage 后反而更好。

**评估是最大的难点。** agent 输出是非确定性的——同一个 profile 两次 ANALYZE 可能报不同 finding。传统单元测试完全不够用。我们的应对是分层：Analyze Benchmark 测"找问题"准确率，Optimize Benchmark 测"修问题"能力，端到端 Demo 用实测 iter_time 做最终裁决。

**LLM 倾向于"过度优化"。** 经常报出 technically correct 但实际影响微乎其微的 finding。后来在 prompt 里加了影响面评估，低于阈值的自动过滤。

**假阳性是常态。** profile 里有信号 ≠ 代码有问题。LLM 只看数据容易误判，必须结合源码上下文。这就是 WARMUP 先建源码索引的原因。

**工具设计决定能力上限。** 最初让 LLM 用 `grep`/`find` 查代码，经常写出错误命令或返回结果撑爆 context。封装 `scanner_read`/`scanner_search`/`scanner_callers` 后，调用准确率大幅提升。工具不是在给 LLM 增加能力，而是在减少犯错机会。

**经验要抽象，不能 case-specific。** "在 nanoGPT 的 model.py 第 42 行不要用 deferred sync"对下一个仓库毫无帮助。好的经验是模式级的——"MPS 后端上 deferred sync 通常没收益，因为 MPS 没有真正的异步流"。

**确定性执行比智能决策可靠。** EXECUTE 用 `if new_time < old_time` 决定接受/回滚，零失误。如果让 LLM 来判断，trial-003 大概率被错误接受。能硬编码的地方就硬编码，不要为了"智能"引入不确定性。

---

# 七、附录

- nanoGPT Demo 完整实录（内部文档）
- Pipeline 设计详解（内部文档）
- Analyzer 设计详解：[docs/03-analyzer.md](./03-analyzer.md)
- Optimizer 设计详解：[docs/04-optimizer.md](./04-optimizer.md)
- AgentLoop 与上下文管理：[docs/05-agent-context.md](./05-agent-context.md)
- Wiki Memory 设计：[docs/06-wiki-memory.md](./06-wiki-memory.md)
- Benchmark 体系设计：[docs/07-benchmark.md](./07-benchmark.md)

[^1]: C1–C7 七类 GPU 性能问题：
  - C1：CPU 调度不均衡，GPU 吃不饱
  - C2：过多小 kernel 调用，缺少 batching/fusion
  - C3：host-device 同步等待（`.item()`、`cudaDeviceSynchronize()`）
  - C4：不必要或阻塞的 H2D/D2H 数据搬运
  - C5：低效计算（重复计算、差的 tensor layout、未融合算子）
  - C6：分布式通信开销或通信-计算 overlap 不足
  - C7：Python 层热路径开销（DataLoader、tokenizer、logging 等）
