# Workspace 记忆

本文件记录当前 repo_root 专属的结构与分析信息，跨同一 workspace 的多次调查复用。
格式由 analyzer 首次调查时按 SOP 填写，后续追加新发现。

---

<!-- 首次调查时由 analyzer 填写，格式如下：

## 基本配置
active_config=<config 文件路径>
batch_size=?、num_steps/batches=?、model=?、loader_workers=?、pin_memory=?
profile 脚本=<名称>；capture-range=<有/无>；warmup=<有/无>
首步 NVTX=?ms，稳态 step=?ms；NCCL/nccl StringIds=<有/无>

## 文件链路
入口：<run.py / main 文件>
  -> <配置/launcher>
  -> <trainer / 核心循环文件>
  -> <data pipeline 文件>
  -> <model 文件>
  -> <callback / monitor 文件>

-->


---

## 基本配置
active_config=configs/config_v6.yaml
batch_size=6、num_steps=12、eval_interval=2、checkpoint_interval=3、feature_dim=32、hidden_size=64、num_heads=4、model=eval_v5、dataset=alternating_c、loader_workers=0、pin_memory=false、callback=sync_v4、checkpoint=metadata_v3、metric=dist_v2
profile 脚本=scripts/profile_local.sh；capture-range=无；warmup=无
首步 NVTX=505.993ms，稳态 step=2.500-9.012ms；NCCL/nccl StringIds=无，repo 内 init_process_group=无；profile 中 reduce_kernel/gpu_comm 应优先按本地 reduction/metric overhead 排查，不直接按 C6 NCCL 归因

## 文件链路
入口：run.py
  -> src/runtime/launcher.py + configs/config_v6.yaml
  -> src/workflow/runner.py（stager / forward / loss / backward / optimizer_step / callback / checkpoint / validation）
  -> src/data/factory.py -> src/data/catalog.py -> src/data/collate.py
  -> src/pipeline/staging.py
  -> src/models/registry.py -> src/models/core.py -> src/models/ops.py
  -> src/callbacks/sync.py、src/checkpoint/manager.py、src/eval/validator.py、src/communication/metrics.py

## 本次增量发现
- 主要瓶颈模式：短 profile + 冷启动采集 + tiny batch/model 导致 GPU active 仅 3.5ms、kernel 980 个且 avg 3.3us，cudaLaunchKernel API inclusive 103.842ms。
- 高风险源码点：scripts/profile_local.sh:11 无 warmup/capture-range；configs/config_v6.yaml:16 loader_workers=0；src/runtime/device.py:10 set_num_threads(1)；src/workflow/runner.py:34 micro-step 训练循环；src/models/ops.py:8/10 Python head loop + cat；src/models/core.py:24/25 redundant shadow encoder；src/callbacks/sync.py:14/15 每步同步和 item；src/eval/validator.py:20/21 D2H/tolist/item；src/communication/metrics.py:13/17 重复 metric reduction/item。
- C6 结论：本轮查询 StringIds 未发现 nccl，scanner search 未发现 init_process_group；虽然 analyzer 报 gpu_comm/reduce_kernel，但不输出确定 NCCL 通信瓶颈。若远程 launcher 外部初始化 process group，再单独复核 src/communication/metrics.py:15-16 的 all_reduce/broadcast。