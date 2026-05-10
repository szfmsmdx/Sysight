# sysight quick baseline config for iterative optimization/profiling

out_dir = "out-sysight-baseline"
wandb_log = False

dataset = "shakespeare_char"
gradient_accumulation_steps = 1
batch_size = 32
block_size = 128

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

learning_rate = 1e-3
max_iters = 24
lr_decay_iters = 24
min_lr = 1e-4
warmup_iters = 4

eval_interval = 10_000
eval_iters = 1
log_interval = 1
always_save_checkpoint = False

compile = False
device = "cuda"
