## v0.1 — 2026-04-16

- Summary: `sysight analyzer` 首个可用版本，包含仓库三阶段分析和 nsys T1-T5 诊断流水线。
- 核心架构：
  - `analyzer.py`：仓库分析核心（Discovery → Targeted Scan → Context Expansion），移除 Go/Rust/Java scanner
  - `cli.py`：CLI 入口，全中文输出，dispatch 到 repo 分析和 nsys 分析
  - `nsys/extract.py`：合并 input/schema/intervals，实现 T1-T3（Profile 解析、Schema 探测、事件提取）
  - `nsys/classify.py`：T4-T5 瓶颈分类与代码热点映射
  - `nsys/models.py`：数据模型定义
  - `scanners/`：Python 静态分析 scanner（base + python）
- Bug 修复：
  - `globalTid` 列兼容性：新增 `_sel_col()` 动态 schema 感知查询，旧版 nsys schema 缺列时降级为 `NULL AS globalTid`，解决所有 GPU 事件提取失败导致的误判 "GPU idle 100%"
  - CLI 缩进语法错误修复
- 中文化：所有面向用户的终端输出改为中文（Warning/Error/Summary/Findings 等）
- 其他：
  - `.gitignore` 补充完整（Python 构建产物、IDE、nsys profile 大文件、日志等）
  - Entry point：`pyproject.toml` 中 `sysight = "sysight.analyzer.cli:main"`

## 历史记录

- Date: 2026-04-15
- Summary: built and hardened `analyzer v0.1` before moving on to optimizer / executor.
- Changes:
  - added `analyzer.py` with lightweight Python and Rust entry detection plus static call-chain tracing
  - added `test_analyzer.py` with Python training / inference and Rust main-entry coverage
  - hardened file scanning to ignore vendor / generated trees like `external/`, `site-packages/`, and `*.runfiles/`
  - changed default CLI output from raw JSON to a human-readable summary, with `--json` for raw output
  - reorganized agent guidance into `.claude/CLAUDE.md` and `.claude/rules/`
  - **DAG-based entry detection**: added `FileDAG` (file-level directed import graph); zero-indegree files get +2 score bonus, low-indegree (≤1) get +1; `_build_file_dag` builds DAG from parsed import bindings in O(files × imports)
  - **`--verbose` / `-v` flag**: activates `logging.DEBUG` output showing DAG stats, per-file indegree bonuses, and scoring decisions; default is silent
  - `EntryPointDetector.detect` and `CallChainTracer.__init__` both accept an optional `dag` parameter; `RepositoryAnalyzer.analyze` constructs the DAG and passes it through
- Verification:
  - ran `python3 -m unittest -v test_analyzer.py` — 4 tests pass
  - ran analyzer against `/Users/szf/Desktop/rtk` to confirm real-repo output shape
  - ran analyzer against `/Users/szf/Desktop/basemodel` to verify it no longer crashes on large vendor trees
  - ran against `/Users/szf/Desktop/walle2` (6556 files): scores improved from 12 → 14 for top entries, new `takeover_scene_classifier_train.py` correctly surfaces as training entry
