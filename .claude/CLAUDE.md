# Project Overview

Sysight is a repo-loop optimization pipeline with three peer modules:
- `analyzer`
- `optimizer`
- `executor`

Current scope is `analyzer v0.1`.
The analyzer must quickly:
- understand repo structure
- find likely training / inference entry scripts
- trace the main intra-repo file / function call chain
- return structured output for downstream modules

# Common Commands

```bash
cd /Users/szf/Desktop/Sysight

# run tests
PYTHONPATH=src python3 -m unittest discover -s test -v

# analyze a repo (shim entry point, works from project root)
PYTHONPATH=src python3 analyzer.py /path/to/repo
PYTHONPATH=src python3 analyzer.py /path/to/repo --max-entries 1 --max-depth 3 --max-steps 30
PYTHONPATH=src python3 analyzer.py /path/to/repo --verbose
PYTHONPATH=src python3 analyzer.py /path/to/repo --json

# or invoke the package directly
PYTHONPATH=src python3 -m sysight.analyzer.core /path/to/repo
```

# Important Constraints

- Keep code simple.
- Keep the repo simple.
- Prefer Python stdlib first.
- Prefer explicit data flow over hidden state.
- Avoid large frameworks unless they are clearly necessary.
- Keep `analyzer`, `optimizer`, and `executor` fully decoupled.
- `analyzer` must not optimize or execute.
- Every new feature should be easy to plug in or remove.

# Detail Rules

Load these rule files on demand:
- `@.claude/rules/code-style.md`
- `@.claude/rules/testing.md`
- `@.claude/rules/security.md`
- `@.claude/rules/worklog.md`

Rule loading guidance:
- editing Python implementation: load `code-style.md`
- editing tests: load `testing.md`
- adding file IO, shell execution, or path handling: load `security.md`
- updating milestones, version notes, or delivery records: load `worklog.md`
