# sysight · analyzer

Profile-side analysis pipeline and localization tooling for GPU performance diagnosis.  
Current scope stays within `analyzer v0.3`; downstream optimization/execution loops are documented at the repo root, not implemented here.  
No external dependencies — stdlib only (`sqlite3`, `pathlib`, `ast`).

---

## Architecture

```
nsys profile (.sqlite)
  → extract trace           # schema probe, event extraction, interval math
  → classify bottlenecks    # findings, SQL deep analysis
  → evidence windows        # callstack summaries, coarse location
  → code localization       # CLI-backed source localization + memory update
```

```
analyzer/
├── cli.py                  # CLI entry (sysight nsys / nsys-sql / scanner)
├── SKILL.txt               # Codex agent investigation prompt template
├── scanner/
│   ├── __init__.py         # Scanner facade
│   ├── callsites.py        # call sites indexing
│   ├── fs.py               # file enumeration
│   ├── reader.py           # file reading with line numbers
│   ├── search.py           # AST text/regex search
│   ├── symbols.py          # callers / callees / trace 
│   └── variants.py         # mapping classes/methods
└── nsys/
    ├── __init__.py         # core analysis logic
    ├── models.py           # all dataclasses (single source of truth)
    ├── extract.py          # trace extraction + interval math
    ├── classify.py         # bottleneck classification + findings
    ├── classify_sql.py     # deep SQL facade + root-cause/profile health
    ├── sql_compute.py      # kernel / idle-gap SQL analyzers
    ├── sql_memory.py       # memcpy bandwidth SQL analyzers
    ├── sql_comm.py         # NCCL SQL analyzers
    ├── sql_sync.py         # synchronization SQL analyzers
    ├── sql_shared.py       # shared SQL helpers
    ├── stacks.py           # callstack summarization + coarse location
    ├── text.py             # text formatting utilities
    ├── windows.py          # evidence window extraction
    ├── render.py           # terminal rendering
    ├── localization.py     # CLI-backed code localization + memory flush
    └── sql_cli.py          # nsys-sql subcommand implementations
```

---

## CLI

### Profile analysis

```bash
# Profile statistics only
sysight nsys /path/to/trace.sqlite --no-codex

# With Codex investigation (waits synchronously)
sysight nsys /path/to/trace.sqlite --repo-root /path/to/repo

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
sysight nsys-sql kernel-launch    /path/to/trace.sqlite
sysight nsys-sql schema  /path/to/trace.sqlite
```

### Static repo tools

```bash
sysight scanner files     <repo>                          # list files
sysight scanner search    <repo> <query>                  # search symbols / filenames
sysight scanner read      <repo> <file> [--start n]       # read file with line numbers
sysight scanner callsites <repo> --call <sym>             # search call sites
sysight scanner symbols   <repo> --file <f>               # list symbols in a file
sysight scanner callers   <repo> <sym>                    # who calls this
sysight scanner callees   <repo> --file <f> --symbol <s>  # what this calls
sysight scanner trace     <repo> <sym>                    # call chain trace
sysight scanner variants  <repo>                          # variant mapping
```

---

## Code Localization And Memory

By default, the analyzer launches its configured CLI localization backend (current default: `codex`) unless `--no-codex` is passed:

- Pre-injects profile data (nvtx / sync / memcpy / kernels / gaps / kernel-launch) into the prompt
- Uses `sysight scanner` tools to resolve exact file/function/line from profile evidence
- Parses JSON output back into the analyzer result
- Persists workspace and experience memory under `.sysight/memory/` via `sysight/analyzer/memory/`

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
| nsys SQLite extraction | ✅ |
| Bottleneck classification + findings | ✅ |
| SQL deep analysis (kernels / sync / nccl / nvtx / health) | ✅ |
| Evidence windows + callstack summarization | ✅ |
| nsys-sql CLI | ✅ |
| scanner CLI (files / search / read / callsites / symbols / callers / callees / trace / variants) | ✅ |
| CLI-backed code localization | ✅ |
| Memory accumulation (workspace + experience) | ✅ |
