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
active_config=configs/config_v2.yaml
batch_size=8、steps=24、model_variant=v5、trainer_variant=standard、image_size=32、seq_len=32、hidden_size=128、loader_workers=0、pin_memory=False、device=cuda
profile 脚本=run.py；capture-range=无（profile_window_start=10/profile_window_end=20 未在训练循环中使用）；warmup=无
首步 NVTX=1203.567ms，稳态 step=19.799ms（iteration_001-023 平均）；NCCL/nccl StringIds=无

## 文件链路
入口：run.py
  -> src/runtime/launcher.py / src/runtime/settings.py（解析 configs/config_v2.yaml，构建 CaseConfig）
  -> src/trainers/factory.py -> src/trainers/loop.py（fit/training_step，batch_to_device / model_forward / loss_backward / optimizer_step / sync_metrics）
  -> src/data/module.py -> src/data/source.py / src/data/transforms.py（DataLoader num_workers=0 pin_memory=False；SyntheticPairDataset；collate_records/image_transform/encode_text）
  -> src/models/registry.py -> src/models/variants/v5.py -> src/models/assembly.py -> src/modules/fusion.py -> src/models/blocks.py -> src/ops/tensor_ops.py（VisionTower/TextTower/FusionHead；stack_text_states 与 append_visual_tokens 为细粒度 dispatch 来源）
  -> src/callbacks/training.py -> src/utils/monitor.py（MetricRecorder/RuntimeObserver 在 sync_metrics 中读取 GPU 标量并同步）

---

## 本次调查增量（2026-04-28）
- 首步冷启动/采集窗口问题确认：iteration_000=1203.567ms，稳态 iteration_001-023 平均约 19.8ms；configs/config_v2.yaml 的 profile_window_start/profile_window_end 未在 src/trainers/loop.py 中使用。
- 数据链路确认：src/data/module.py 固定 worker_count=0、page_locked=False；SyntheticPairDataset.__getitem__ 逐样本创建 Generator/randn；collate_records 串行执行 image_transform、encode_text、pack_metadata。
- 模型热路径确认：TextTower 通过 stack_text_states 按 seq_len=32 逐 token 调 embedding+linear；FusionHead 通过 append_visual_tokens 按 16 个 visual token 循环 mix+cat+clone；VisionTower.forward 存在重复 stem 和 shadow.mul(0.0) 零贡献路径。
- 同步链路确认：MetricRecorder.capture 每步 loss.item、logits.cpu().numpy、json.dumps；RuntimeObserver.after_step 每步 torch.cuda.synchronize 和 logits mean item。
- 分布式检查：源码搜索未发现 init_process_group/all_reduce/nccl，已有 profile 记忆中 NCCL/nccl StringIds=无；本次不输出 C6 finding。