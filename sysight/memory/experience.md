# 通用分析经验

本文件记录与 workspace 无关的通用 nsys 分析经验，跨 workspace 复用。

## 写作规范

每条经验 **必须** 包含以下三个字段：

```
## [短标题]
- 场景：什么情况下会遇到
- 规则：正确做法或错误原因
- 示例（可选）：典型数字或代码片段
```

追加新经验时：
- 标题用动宾短语或关键信号描述，**不加序号**
- 规则字段写明**必须做什么 / 禁止做什么**
- 示例字段只在有具体数字或代码时添加，否则省略
- 每条经验后加一行 `---` 分隔

---

## 不把 `reduce_kernel` 直接当作 NCCL 通信

- 场景：Nsight/Sysight 报告把 `reduce_kernel` 归到 `gpu_comm`，但 kernel 名称是 `at::native::reduce_kernel` 而不是 NCCL kernel。
- 规则：先查源码是否真的初始化了 process group，再查 profile StringIds/kernel name 是否存在 `nccl`/`NCCL`；若两者都没有，应优先按本地 PyTorch reduction 或 metric 统计处理，不要贸然输出 C6 NCCL 结论。当 profile 中 reduce_kernel 被归类为 gpu_comm 但没有 NCCL 字符串/进程组初始化时，检查 metric 路径的 per-parameter grad norm；steps × 参数 tensor 数常可精确解释 reduce_kernel count，这类问题应归为本地 reduction 的 kernel launch/metric overhead，而不是通信瓶颈。

---

## 短 profile 中首步冷启动需要单独剥离

- 场景：NVTX step_000 远大于后续 steady steps，GPU kernel 总时间很小但 stream gap 极大。
- 规则：不要直接把整个 trace 的 GPU idle 归因到模型计算；先看 step_000 内首个真实 GEMM/主计算 kernel 的相对时间，并检查 OSRT/CUDA runtime 是否显示 lazy init/driver 等待。优化建议通常是加 warmup、profile steady-state、或从瓶颈排序中排除 step_000。当首个 NVTX step 远大于稳态 step 且 CUDA Runtime API 总时长解释不了差值时，应查询 OSRT poll/ioctl；若大量时间落在 OSRT driver 等待，优先按 CUDA/PyTorch 首次调度或 lazy init 污染处理，优化 profile 方法应加 warmup/capture range，而不是把全 trace idle 直接归因到模型计算或数据加载。

---

## 配置里的 profile window 不等于实际 Nsight capture window

- 场景：配置文件存在 `profile_window_start`/`profile_window_end`，但 profile 脚本仍用 full-run `nsys profile python ...`。
- 规则：先确认代码是否调用 `cudaProfilerStart/Stop`，或 nsys 是否使用 `--capture-range`/NVTX capture；否则配置字段不会影响 Nsight Systems 捕获范围，首轮 CUDA lazy init 仍会进入报告。

---

## 配置 batch_size 可能被服务层 micro-batching 覆盖

- 场景：配置显示 batch_size>1，但 NVTX 中 serve_batch 数量接近 request_count，GPU 侧大量微小 kernel 且利用率极低。
- 规则：不要只看配置文件判断实际 batch；必须检查 scheduler/queue 是否把 window 拆成单请求 micro-batch。若实际 batch 被拆散，优先定位到拆分代码行，而不是只在模型算子层面解释低利用率。

---

## LLM decode 中"cache 参数存在"不代表真的用了 KV cache

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

---

## 用配置乘法反推 MoE 小 GEMM 数量

- 场景：profile 中出现大量 2-5us 的 GEMV/GEMM kernel，且代码有 batch item 循环、expert 循环、top_k 路由或 replay 分支。
- 规则：用 steps × batch_size × expert_count × dispatch次数 × expert内Linear数 估算 kernel 数量；若与 Nsight top kernel count 对齐，优先优化 per-sample/per-expert Python dispatch、未选中 expert 计算和零贡献 replay，而不是把问题归因到 GPU 算力不足。

---

## 用 D2H 次数反推训练监控里的隐式同步

- 场景：Nsight 显示 D2H 总字节几乎为 0，但 D2H 次数与 step 数呈稳定倍数关系，同时伴随少量 sync_wait / stream wait。
- 规则：优先在日志、监控、callback 路径搜索 `.item()`、`.cpu()`、`.numpy()`、`.tolist()`、`torch.cuda.synchronize()`；这类 tiny transfer 往往不是带宽瓶颈，而是把训练主线程和 GPU 强制对齐的同步点。

---

## 用 `triu_tril_kernel` 次数反推 decode 热路径里的 mask 重建

- 场景：Nsight top kernel 出现 `triu_tril_kernel`，同时 softmax/GEMM 次数与 `decode_steps × block_calls × num_heads` 对齐。
- 规则：优先检查 attention block 是否在每次 forward 中执行 `torch.tril(torch.ones(...))` 现造 causal mask；若 `triu_tril_kernel` 次数与 block forward 数一致，说明 mask 构造已进入热路径，应缓存 mask 或改用带 causal fast path 的 attention 实现。

---

## 用 softmax/GEMM 次数识别 Python 按 head 拆分 attention

- 场景：`softmax_warp_forward` 和小型 GEMM 的 count 精确等于 `decode_steps × block_calls × num_heads`，且单核时长普遍只有几微秒。
- 规则：这通常不是 GPU 算力不足，而是多头 attention 被 Python for-loop 拆成 per-head 小 kernel；优先把 head 维并入 batched 张量，或直接改用 SDPA / fused attention。

---

## 不要把脚本生成的 rank 文件名当成真实多卡运行

- 场景：profile 脚本循环设置类似 `NSYS_BENCH_PROFILE_RANK=0/1` 的环境变量，并输出 `rank0/rank1` 文件；配置里同时还有 `world_size > 1`。
- 规则：必须同时验证代码是否读取该环境变量、是否调用 `init_process_group`、以及源码/profile 是否出现 `nccl`。如果三者都没有，`rank0/rank1` 只是重复单进程运行，不能据此输出 C6 通信结论。

---

## 用 H2D/D2H 次数识别热路径里的 device-host-device 回环

- 场景：staging / preprocess 代码先把 tensor `.to(device)`，随后又为了 shadow、debug 或占位逻辑写出 `tensor.cpu().to(device)`。
- 规则：如果 profile 里的 H2D/D2H 次数按 step 呈固定倍数、单次字节又很小，应优先搜索 `cpu().to(device)` 这类往返代码；它常不是带宽瓶颈，但会稳定制造额外 copy，并放大同步点的影响。

---

## 用 H2D 次数按 batch tensor 字段数反推 staged prefetch

- 场景：profile 里 H2D 次数接近 `steps × 输入 tensor 字段数`，代码中又存在 host batch dict -> device batch dict 的 `prepare/transfer` 阶段。
- 规则：优先检查 `images`、`tokens`、`labels` 等字段是否分别 `.to(device)`；如果 `pin_memory=false` 或未使用 `non_blocking=True`，这些 staged copy 会稳定制造每步固定模式的小 H2D，并常与 `cudaMemcpyAsync` 次数一一对应。

---

## 用 `gatherTopK` / `bitonicSortKVInPlace` 次数反推按样本 `torch.topk`

- 场景：Nsight top kernel 出现 `gatherTopK`、`bitonicSortKVInPlace` 这类 top-k/sort kernel，count 精确接近 `steps × batch_size`。
- 规则：优先检查 router/planner 是否在 Python 循环里按样本执行 `torch.topk(probabilities[item], k)`；若 kernel 次数与样本数对齐，通常应改成 batched `torch.topk(..., dim=-1)`，而不是先拆样本再做 top-k。

---

## 用 step 周期对齐 callback / validation / checkpoint 热路径

- 场景：NVTX top ranges 里只有固定编号的 step 明显更长，而模型 kernel 本身都很小。
- 规则：先把长 step 的编号与 `eval_interval`、`checkpoint_interval`、日志频率对齐；若 `3/6/9`、`0/2/4/...` 这类周期吻合，应优先检查 step 内 callback、validation、checkpoint 的同步、D2H 和 I/O，而不是先把长尾归因到模型算子。

---

## 用 embedding/indexSelect 次数反推逐 token Python 循环

- 场景：Nsight top kernel 出现 `indexSelectSmallIndex` 或 `embedding_backward_feature_kernel`，count 接近 `steps × seq_len`。
- 规则：优先检查 embedding/projection 是否在 Python 中按 token 位置循环调用；应改为对完整 `[batch, seq]` tokens 做 batched embedding，再对 `[batch, seq, hidden]` 做批量 Linear/Norm。

---

## 用 CatArrayBatchedCopy 次数反推循环内 torch.cat

- 场景：profile 出现 `CatArrayBatchedCopy_*` 且 count 接近 `steps × token_count`，同时 D2D copy 次数较多。
- 规则：优先检查是否在 token/sample 循环内反复 `torch.cat` 递增 tensor；通常应 list 收集后一次 cat，或直接改为批量张量计算避免 cat。

---

## 用 SyntheticDataset 即时造样本识别主线程喂数

- 场景：benchmark 或 toy case 使用 synthetic dataset，profile 显示 step 间 GPU 大量 idle，而 DataLoader 又是单线程。
- 规则：不要只看 collate/transform；还要检查 `Dataset.__getitem__` 是否每样本新建 `torch.Generator`、`manual_seed`、`torch.randn` 或拼接文本。这类即时造样本会直接占用 host gap。

---

## 用 decode 调度器里的 `torch.cat` 识别伪增量解码

- 场景：生成代码有 `cache` 参数，但 Nsight 仍显示 attention/softmax/GEMM 次数按 `decode_steps × block_calls × num_heads` 成倍增长。
- 规则：先检查 scheduler 是否在每个 token 后 `torch.cat` 扩展 `state.current`，再看模型是否把整个增长序列重新送入 forward。若 cache 只保存 step/summary 而非 per-layer K/V，应判定为完整前缀重算，而不是已启用 KV cache。

---

## 用 D2H 总次数精确拆分多个周期性同步点

- 场景：profile 中 D2H 字节几乎为 0，但 D2H count 与训练 step、validation interval、checkpoint interval 的组合完全对齐。
- 规则：逐行统计 `.item()`、`.cpu().tolist()`、checkpoint checksum、metric logging 的触发次数，用 `num_steps`、`eval_interval`、`checkpoint_interval` 还原总 D2H；若能精确相加，应分别输出每个同步行，不要把所有 D2H 合并成笼统的日志同步问题。

---

## 用 D2D copy 字节分布识别 MoE dispatch 中的 per-sample clone

- 场景：D2D copy 的 bytes 分布呈现 hidden_size×4 的小块，且 count 可分解为 `steps × dispatch_calls × batch_size × 2`。
- 规则：优先检查 MoE/dispatch 中是否存在 per-sample clone 与 torch.cat；若同时有 `batch_size×dense_dim×4` 的每步一次 D2D，通常对应整批输入在 tower 前的 clone/contiguous。

---

## backward 没有 optimizer 消费时按零贡献计算排查

- 场景：profile 出现 embedding_backward、reduce backward 或大量 autograd kernel，但训练循环只调用 loss.backward。
- 规则：必须检查源码是否存在 optimizer.zero_grad、optimizer.step 或梯度消费路径；若没有，backward 应归为 C5 redundant/zero-contribution code path，推理 benchmark 应移除，训练 benchmark 应补齐完整优化器步骤后再 profile。




---

## 单卡 reduce kernel 不等于 NCCL
- 场景：Nsight 报告里出现 reduce_kernel 或工具误把单流 kernel 归到 gpu_comm，但源码没有分布式初始化。
- 规则：C6 必须同时看到 profile StringIds 含 nccl/NCCL 且源码有 torch.distributed/init_process_group/all_reduce；否则将 reduce_kernel 归为碎粒度 CUDA op 或模型 reduce，不输出通信 finding。
- 示例：case_5 的 StringIds 无 nccl，源码无 torch.distributed，reduce_kernel 来自 at::native::reduce_kernel。
---