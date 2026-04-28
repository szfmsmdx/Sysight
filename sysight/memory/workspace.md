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
