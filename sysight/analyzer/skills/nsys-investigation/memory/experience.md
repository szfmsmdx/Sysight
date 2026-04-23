# 通用分析经验

本文件记录与 workspace 无关的通用 nsys 分析经验，跨 workspace 复用。

每条经验格式：
```
## [短标题]
- 场景：什么情况下会遇到
- 陷阱 / 规则：正确做法或错误原因
- 示例（可选）
```

---


---

## 不把 `reduce_kernel` 直接当作 NCCL 通信
- 场景：Nsight/Sysight 报告把 `reduce_kernel` 归到 `gpu_comm`，但 kernel 名称是 `at::native::reduce_kernel` 而不是 NCCL kernel。
- 规则：先查源码是否真的初始化了 process group，再查 profile StringIds/kernel name 是否存在 `nccl`/`NCCL`；若两者都没有，应优先按本地 PyTorch reduction 或 metric 统计处理，不要贸然输出 C6 NCCL 结论。

---

## 短 profile 中首步冷启动需要单独剥离
- 场景：NVTX step_000 远大于后续 steady steps，GPU kernel 总时间很小但 stream gap 极大。
- 规则：不要直接把整个 trace 的 GPU idle 归因到模型计算；先看 step_000 内首个真实 GEMM/主计算 kernel 的相对时间，并检查 OSRT/CUDA runtime 是否显示 lazy init/driver 等待。优化建议通常是加 warmup、profile steady-state、或从瓶颈排序中排除 step_000。

---

## 配置里的 profile window 不等于实际 Nsight capture window
- 场景：配置文件存在 `profile_window_start`/`profile_window_end`，但 profile 脚本仍用 full-run `nsys profile python ...`。
- 规则：先确认代码是否调用 `cudaProfilerStart/Stop`，或 nsys 是否使用 `--capture-range`/NVTX capture；否则配置字段不会影响 Nsight Systems 捕获范围，首轮 CUDA lazy init 仍会进入报告。

---

## 配置 batch_size 可能被服务层 micro-batching 覆盖
- 场景：配置显示 batch_size>1，但 NVTX 中 serve_batch 数量接近 request_count，GPU 侧大量微小 kernel 且利用率极低。
- 规则：不要只看配置文件判断实际 batch；必须检查 scheduler/queue 是否把 window 拆成单请求 micro-batch。若实际 batch 被拆散，优先定位到拆分代码行，而不是只在模型算子层面解释低利用率。

## LLM decode 中“cache 参数存在”不代表真的用了 KV cache
- 场景：forward 签名包含 cache，profile 仍显示每个 decode step 大量重复 attention/GEMM/softmax kernel。
- 规则：检查 forward 是否只处理最新 token 并复用历史 K/V；如果仍把完整 `state.current` 传入 embedding 和 blocks，cache 只是状态占位，不能视为增量解码。

---

## 将 sampler sleep 按 batch 量纲还原到稳态 step
- 场景：短 profile 被首步冷启动主导，但后续 steady step 只有几毫秒，同时 DataLoader/Sampler 中存在 `sleep` 或慢速 CPU 逻辑。
- 规则：不要只看全 trace idle；应按 `batch_size`、采样条件和 `num_workers` 估算每个 batch 的 host 等待。如果 `num_workers=0`，Sampler.__iter__ 中的等待会直接落在训练主线程，常常解释稳态 GPU 空闲。

---

## NVTX step 范围放在 DataLoader 取 batch 之后时要区分输入等待和 step 内冷启动
- 场景：训练循环形如 `for step, batch in loader:` 后才进入 `with nvtx.range(step)`，profile 同时显示首个 step 极长、GPU idle 极高。
- 规则：该 NVTX step 不包含取 batch 之前的 DataLoader 等待；若 step_000 远大于后续 step，应优先按 CUDA/PyTorch lazy init 或 full-run capture 污染处理。DataLoader worker/pin_memory 仍可作为稳态喂数问题分析，但不要把 step_000 的全部耗时直接归因到 DataLoader。

---

## MoE 路由热路径不要把 CUDA tensor 转 Python list
- 场景：MoE/router 先在 GPU 上得到 routes，然后在 Python 中逐样本判断 expert 选择。
- 规则：`routes[item].tolist()` 会触发 D2H 和同步，Nsight 中常表现为 D2H copy、sync_wait 和 GPU stream gap；应使用 GPU tensor mask/gather/scatter 或按 expert 分组的 batched dispatch。

## 用配置乘法反推 MoE 小 GEMM 数量
- 场景：profile 中出现大量 2-5us 的 GEMV/GEMM kernel，且代码有 batch item 循环、expert 循环、top_k 路由或 replay 分支。
- 规则：用 steps × batch_size × expert_count × dispatch次数 × expert内Linear数 估算 kernel 数量；若与 Nsight top kernel count 对齐，优先优化 per-sample/per-expert Python dispatch、未选中 expert 计算和零贡献 replay，而不是把问题归因到 GPU 算力不足。