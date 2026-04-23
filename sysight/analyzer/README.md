# sysight · analyzer

Profile-side analysis pipeline and agent tooling for GPU performance diagnosis.  
No external dependencies — stdlib only (`sqlite3`, `pathlib`, `ast`).

---

## Architecture

```
nsys profile (.sqlite)
  → extract trace           # T1-T3: schema probe, event extraction, interval math
  → classify bottlenecks    # T4-T5: findings, SQL deep analysis
  → evidence windows        # callstack summaries, coarse location
  → Codex investigation     # Stage 6: agent-driven code localization
```

```
analyzer/
├── cli.py                  # CLI entry (sysight nsys / nsys-sql / scanner)
├── analyzer.py             # repo three-stage analysis core
├── callsite.py             # callsite index for scanner
├── scanner_cli.py          # scanner subcommands
├── scanners/
│   ├── base.py             # FunctionFacts, FileFacts, BaseScanner
│   ├── python.py           # AST-based Python scanner
│   └── cpp.py              # CUDA/C++ scanner
└── nsys/
    ├── models.py           # all dataclasses (single source of truth)
    ├── extract.py          # trace extraction + interval math
    ├── classify.py         # bottleneck classification + findings
    ├── classify_sql.py     # SQL deep analysis (kernels/sync/nccl/nvtx/health)
    ├── classify_sql_nvtx.py# NVTX layer breakdown
    ├── stacks.py           # callstack summarization + coarse location
    ├── windows.py          # evidence window extraction
    ├── render.py           # terminal rendering
    ├── investigation.py    # Stage 6: Codex prompt build + execution + memory flush
    ├── sql_cli.py          # nsys-sql subcommand implementations
    └── skills/nsys-investigation/
        ├── TASK.txt        # Codex prompt template
        ├── SKILL.md        # harness reference
        └── memory/         # workspace.md + experience.md
```

---

## CLI

### Profile analysis

```bash
# Profile statistics only
sysight nsys /path/to/trace.sqlite

# With Codex investigation (waits synchronously)
sysight nsys /path/to/trace.sqlite --repo-root /path/to/repo --report full

# JSON output
sysight nsys /path/to/trace.sqlite --json
```

### Direct SQLite inspection

```bash
sysight nsys-sql nvtx    /path/to/trace.sqlite [--limit 20]
sysight nsys-sql kernels /path/to/trace.sqlite [--limit 20]
sysight nsys-sql gaps    /path/to/trace.sqlite [--limit 10]
sysight nsys-sql sync    /path/to/trace.sqlite
sysight nsys-sql memcpy  /path/to/trace.sqlite
sysight nsys-sql nccl    /path/to/trace.sqlite
sysight nsys-sql overlap /path/to/trace.sqlite
sysight nsys-sql stream-concurrency /path/to/trace.sqlite
sysight nsys-sql schema  /path/to/trace.sqlite
```

### Static repo tools

```bash
sysight scanner callsites /path/to/repo --call to        # search call sites (most common)
sysight scanner search    /path/to/repo <query>           # search symbols / filenames
sysight scanner lookup    /path/to/repo --file f --line n # locate context at line
sysight scanner callers   /path/to/repo <symbol>          # who calls this
sysight scanner callees   /path/to/repo <symbol>          # what this calls
sysight scanner trace     /path/to/repo <symbol>          # call chain trace
sysight scanner manifest  /path/to/repo                   # repo manifest
sysight scanner index     /path/to/repo                   # build index
```

---

## Stage 6 Investigation

When `--report full` is set, the analyzer launches a Codex subprocess to perform code-level localization:

- Pre-injects profile data (nvtx / sync / memcpy / kernels / gaps / kernel-launch) into the prompt
- Codex uses `sysight scanner` tools to locate exact file/function/line
- Results are parsed from JSON output and written back to the analysis result
- Memory is persisted to `skills/nsys-investigation/memory/` for cross-run accumulation

Artifact directory: `.sysight/codex_runs/run-<id>/` (prompt, stdout, stderr, last_message).

---

## Tests

```bash
cd /path/to/Sysight
PYTHONPATH=src python3 -m unittest discover -s test -v
```

---

## Component Status

| Component | Status |
|-----------|--------|
| nsys SQLite extraction (T1–T3) | ✅ |
| Bottleneck classification + findings (T4–T5) | ✅ |
| SQL deep analysis (kernels / sync / nccl / nvtx / health) | ✅ |
| Evidence windows + callstack summarization | ✅ |
| nsys-sql CLI | ✅ |
| scanner CLI (callsites / search / lookup / callers / callees / trace) | ✅ |
| Stage 6 Codex investigation | ✅ |
| Memory accumulation (workspace + experience) | ✅ |
