# Task
- 说明这次希望 Sysight 帮你完成什么，例如：分析现有 profile、定位 bottleneck、生成优化计划。
- 如果已经有 profile，写明 profile 路径；如果还没有，也说明后续希望 profile 哪个入口。

# Project
- 用 3~8 句话介绍项目背景、主要业务流程、当前阶段（训练/推理/离线批处理）。
- 如果有固定数据集、固定 batch、固定服务拓扑，也在这里说明。

# Framework
- 写清楚主要框架和技术栈，例如：PyTorch 2.x、CUDA 12.x、Triton、FlashInfer、TensorRT、NCCL。
- 如果项目里有自定义 CUDA/Triton kernel，或大量依赖第三方推理框架，也明确写出来。

# Entry
- 写清楚项目如何启动，最好给出一条可执行命令。
- 如果启动依赖环境变量、配置文件、模型权重、数据目录，也一起写明。

```bash
python main.py --config configs/infer.yaml --model-path /path/to/model
```

# Performance Goal
- 写清楚本次优化目标和优先级，例如：
  - 优先降低 P99 latency
  - 保持精度不变前提下提升吞吐
  - 提升 GPU MFU
  - 降低 step time / allreduce 暴露时间 / H2D 开销
- 尽量给出当前基线和目标值。

# Important Paths
- 列出你认为最值得关注的代码、配置和脚本路径。
- 如果你已经怀疑某些模块是热点，也请直接写出来。

- `src/model/decoder.py`
- `src/runtime/launcher.py`
- `configs/infer.yaml`
- `scripts/run_infer.sh`

# Constraints
- 这里写 agent 当前必须遵守的限制。
- 建议至少覆盖下面这些点：
  - 当前阶段是否允许自动改代码
  - 是否允许重新跑 profile
  - 是否允许安装依赖
  - 是否允许修改启动参数
  - 是否只允许生成报告和建议

推荐写法：
- 当前阶段先只做 analysis 和 report，不自动改代码。
- 未经明确说明，不重新采集 profile。
- 不安装新依赖，不修改部署环境。
- 所有结论必须区分 evidence 和 hypothesis。

# Success Criteria
- 写清楚什么结果算这次工作完成。
- 推荐至少包含：
  - 至少识别出几个最重要的问题
  - 每个问题是否要附代码/函数定位
  - 是否必须给出可执行下一步建议
  - 是否需要形成优化计划但暂不落代码

推荐写法：
- 至少给出 3 个带证据的问题。
- 每个问题都要给出最可疑的代码区域或函数。
- 给出 2~4 条下一步行动建议，并说明优先级。

# Output Contract
- 说明你希望 agent 最终产出什么。
- 当前建议固定成：
  - 终端摘要：`结论 / 问题 / 下一步行动建议`
  - 一个完整的 `report.md`
  - 如证据不足，要明确说明还缺什么上下文

推荐写法：
- 先输出终端简化结论。
- 再给出完整 report.md 路径。
- 报告中优先绑定 NVTX、runtime 线程、sampled stack 和代码/函数候选。

# Notes
- 这一份 `program.md` 参考了 autoresearch 的“先定义 contract，再让 agent 执行”的思路，但针对 Sysight 做了调整。
- 当前阶段请把它理解为“analysis 工作协议”，不是自动优化 loop 的执行脚本。
- 如果当前只有 profile 没有 workspace，也可以先不填这份文件；但一旦希望 agent 结合项目语义做更强归因，建议补齐。 
