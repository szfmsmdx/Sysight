# sysight · analyzer

Static source-code analyzer and nsys profile mapper.  
No external dependencies — stdlib only (`sqlite3`, `difflib`, `pathlib`, `ast`).

```
profile evidence  ──►  repo scan (targeted)  ──►  NsysDiag
                        fuzzy file match
                        file-line lookup
                        callers / callees
```

---

## Architecture

```
analyzer/
├── analyzer.py      # three-stage analysis core (manifest → scan → context)
├── cli.py           # CLI entry point + output rendering
├── scanners/
│   ├── base.py      # FunctionFacts, FileFacts, BaseScanner
│   ├── python.py    # AST-based; triton/distributed GPU tags
│   └── cpp.py       # __global__ / cuda_launch detection
└── nsys/
    ├── models.py    # all dataclasses (single source of truth)
    ├── extract.py   # T1 input resolution, T2 schema probe, T3 trace extraction, interval math
    ├── classify.py  # T4 bottleneck classification + findings
    └── __init__.py  # analyze_nsys() + derive_repo_scope() entry points
```

**Supported languages:** Python, C/C++/CUDA. Other file types are silently skipped.

**Three-stage repo pipeline** — designed for large repos:

| Stage | Function | Cost |
|-------|----------|------|
| 1 | `discover_repo()` | path walk only, no file reads |
| 2 | `scan_repo(scope=)` | targeted parse of relevant files |
| 3 | `get_repo_context()` | on-demand snippets, bounded by `ContextBudget` |

Default scan mode is `targeted` (profile-driven). `full` is opt-in.

---

## CLI

```bash
# ── Repo analysis ───────────────────────────────────────────────────────

# Full scan: entry points, call chains, hub nodes
sysight <repo>
sysight <repo> --json
sysight <repo> --top 5 --depth 6

# Stage-1 manifest (path walk only, instant on large repos)
sysight <repo> manifest
sysight <repo> manifest --json

# ── Search ──────────────────────────────────────────────────────────────

sysight <repo> search <query>
sysight <repo> search train --limit 10
sysight <repo> search backward --json

# ── Call chain trace ────────────────────────────────────────────────────

sysight <repo> trace <file-or-symbol>
sysight <repo> trace trainer.py
sysight <repo> trace trainer.py --symbol train_step
sysight <repo> trace train_step --json

# ── Impact radius ────────────────────────────────────────────────────────

sysight <repo> impact <file> [<file> ...]
sysight <repo> impact src/model.py src/loss.py
sysight <repo> impact src/model.py --depth 3 --json

# ── Nsys profile analysis ────────────────────────────────────────────────

# Basic: bottleneck breakdown + findings (no repo mapping)
sysight <repo> nsys <profile.nsys-rep>
sysight <repo> nsys <profile.sqlite>
sysight <repo> nsys <profile.nsys-rep> --no-repo

# With repo hotspot mapping
sysight <repo> nsys <profile.nsys-rep> --top-hotspots 20
sysight <repo> nsys <profile.nsys-rep> --scope full
sysight <repo> nsys <profile.nsys-rep> --json

# Explicit sqlite path (skip auto-detect)
sysight <repo> nsys <profile.nsys-rep> --sqlite <profile.sqlite>
```

Common flags:

| Flag | Default | Effect |
|------|---------|--------|
| `--json` | off | Machine-readable JSON output |
| `--verbose` / `-v` | off | Debug logging |
| `--top N` | 10 | Max entry points (repo scan) |
| `--depth N` | 8 | Max call-chain / impact depth |
| `--top-hotspots N` | 20 | Top N hotspots mapped to repo (nsys) |
| `--scope targeted\|full` | targeted | Repo scan mode for nsys hotspot mapping |
| `--no-repo` | off | Skip repo mapping (bottleneck analysis only) |

---

## nsys output format

```
Profile: run.nsys-rep
SQLite:  run.sqlite
Status:  ok

Trace:  4821.3ms  |  GPU active: 3102.1ms (64.3%)  |  GPU idle: 1719.2ms (35.7%)

Bottleneck breakdown:
  gpu_compute            82.3% of trace     3971.4ms  (xx.x% of GPU active)
  gpu_comm                9.1% of trace      439.2ms
  sync_wait               5.4% of trace      260.5ms
  gpu_memcpy              3.2% of trace      150.2ms

Top 5 events:
  [gpu_compute] volta_sgemm_128x64_nt                       1284.1ms  ×2048
  [gpu_comm]    ncclAllReduceRingLLKernel_sum_f16             439.2ms  ×512
  ...

Findings (2):
  ⚠ [WARNING] High GPU idle ratio (35.7%)
      → Check for CPU-GPU sync points (cudaDeviceSynchronize)
  ℹ [INFO] NCCL AllReduce dominates comm time

Top hotspots mapped to repo (5):
   8.3%  volta_sgemm_128x64_nt                    → src/model/linear.cu [Linear.forward]  conf=0.91
   ...

Summary: Trace 4821.3ms, GPU active: 3102.1ms (64.3%), ...
```

---

## Status

| Component | Status |
|-----------|--------|
| Repo scan (Python + C/C++/CUDA) | ✅ |
| `fuzzy_file_match` / `lookup_by_file_line` | ✅ |
| `callers_of` / `callees_of` | ✅ |
| nsys sqlite parser (T1–T4) | ✅ |
| Bottleneck classification + findings | ✅ |
| Repo hotspot mapping | ✅ |
