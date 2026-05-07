"""BenchmarkRunner — run WARMUP + ANALYZE against nsys-bench cases, score results."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sysight.agent.provider import create_provider


@dataclass
class CaseResult:
    case_id: str = ""
    status: str = ""                          # "ok" | "error"
    score: int = 0
    total: int = 0
    matched_ids: list[str] = field(default_factory=list)
    findings_count: int = 0
    warmup: dict = field(default_factory=dict)
    analyze: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0
    debug_log: list[dict] = field(default_factory=list)


class BenchmarkRunner:
    """Run benchmark cases through WARMUP → ANALYZE, score against ground truth."""

    def __init__(
        self,
        nsys_bench_dir: str | Path = "nsys-bench",
        output_dir: str | Path = ".sysight/bench-runs",
        debug: bool = False,
        force_warmup: bool = False,
        no_warmup: bool = False,
    ):
        self._nsys_bench_dir = Path(nsys_bench_dir).resolve()
        self._output_dir = Path(output_dir).resolve()
        self._debug = debug
        self._force_warmup = force_warmup
        self._no_warmup = no_warmup
        self._registry = None
        self._knowledge = None
        self._configs = {}

    def run(self, case_ids: list[str]) -> dict:
        """Run all specified cases. Returns summary dict."""
        self._setup()

        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        run_dir = self._output_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"Output dir: {run_dir}")

        results: list[CaseResult] = []
        for case_id in case_ids:
            case_out = run_dir / case_id
            case_out.mkdir(parents=True, exist_ok=True)
            print(f"\n{'='*60}")
            print(f"  CASE: {case_id}")
            print(f"  LOG:  {case_out}")
            print(f"{'='*60}")
            result = self._run_case(case_id, case_out)
            results.append(result)
            self._print_case_result(result)

        self._write_summary(results, run_dir, timestamp)
        self._print_summary(results, timestamp)

        return {
            "timestamp": timestamp,
            "run_dir": str(run_dir),
            "cases": {r.case_id: {"score": r.score, "total": r.total, "status": r.status} for r in results},
        }

    def _setup(self):
        """Initialize registry, configs, knowledge (same as CLI _setup)."""
        from sysight.tools.registry import ToolRegistry
        from sysight.tools import register_all_tools
        from sysight.agent.config_loader import load_config
        from sysight.wiki.store import WikiRepository

        registry = ToolRegistry()
        register_all_tools(registry)
        self._registry = registry

        self._configs = load_config()
        self._knowledge = WikiRepository()

    def _run_case(self, case_id: str, case_out: Path) -> CaseResult:
        case_dir = self._nsys_bench_dir / "cases" / case_id
        truth_path = self._nsys_bench_dir / "tests" / "findings" / f"{case_id}_findings.json"

        if not case_dir.is_dir():
            return CaseResult(case_id=case_id, status="error", errors=[f"case dir not found: {case_dir}"])
        if not truth_path.exists():
            return CaseResult(case_id=case_id, status="error", errors=[f"truth file not found: {truth_path}"])

        repo_root = str(case_dir)
        truth = json.loads(truth_path.read_text(encoding="utf-8"))

        # Find sqlite profile
        case_yaml = case_dir / "case.yaml"
        sqlite_path = None
        if case_yaml.exists():
            import yaml
            try:
                with open(case_yaml) as f:
                    cfg = yaml.safe_load(f)
                profile_cfg = cfg.get("profile", {})
                rel = profile_cfg.get("sqlite", "")
                if rel:
                    candidate = case_dir / rel
                    if candidate.exists():
                        sqlite_path = str(candidate)
            except Exception:
                pass

        if not sqlite_path:
            # Fallback: find any .sqlite in profiles/
            profiles_dir = case_dir / "profiles"
            if profiles_dir.is_dir():
                sqlites = sorted(profiles_dir.glob("*.sqlite"))
                if sqlites:
                    sqlite_path = str(sqlites[0])

        if not sqlite_path:
            return CaseResult(case_id=case_id, status="error", errors=["no sqlite profile found"])

        # Create provider for analyze
        analyze_cfg = self._configs.get("analyze")
        if not analyze_cfg or not analyze_cfg.api_key:
            return CaseResult(case_id=case_id, status="error", errors=["no analyze provider configured"])

        provider = create_provider(analyze_cfg)
        debug_log: list[dict] = []

        t0 = time.monotonic()
        result = CaseResult(case_id=case_id, status="ok", total=truth.get("total_points", 0))

        # -- WARMUP (cached by default; --force-warmup re-runs; --no-warmup skips entirely) --
        if self._no_warmup:
            print(f"  [warmup] skipped (--no-warmup)")
        else:
            print(f"  [warmup] scanning repo...")
            try:
                from sysight.pipeline.warmup import run_warmup
                wr = run_warmup(repo_root, self._registry, self._knowledge, force=self._force_warmup)
                result.warmup = {
                    "entry_point": wr.repo_setup.entry_point,
                    "test_commands": wr.repo_setup.test_commands,
                    "build_commands": wr.repo_setup.build_commands,
                    "source": wr.repo_setup.source,
                    "errors": wr.errors,
                }
                print(f"  [warmup] entry={wr.repo_setup.entry_point}, source={wr.repo_setup.source}")
            except Exception as e:
                result.errors.append(f"warmup: {e}")
                print(f"  [warmup] FAILED: {e}")

        # -- ANALYZE (always logged to debug.log; --debug controls terminal output) --
        print(f"  [analyze] running LLM investigation...")
        from sysight.benchmark.debug import DebugProvider
        actual_provider = DebugProvider(provider, debug_log, verbose=self._debug, log_file=str(case_out / "debug.log"))
        if self._debug:
            print(f"  [analyze] debug mode ON — printing LLM I/O to terminal")

        try:
            from sysight.pipeline.analyze import run_analyze
            ar = run_analyze(
                sqlite_path, repo_root,
                self._registry, actual_provider, self._knowledge,
            )
            result.analyze = {
                "run_id": ar.run_id,
                "summary": ar.finding_set.summary,
                "findings": [
                    {
                        "finding_id": f.finding_id,
                        "category": f.category,
                        "title": f.title,
                        "priority": f.priority,
                        "file_path": f.file_path,
                        "function": f.function,
                        "line": f.line,
                        "confidence": f.confidence,
                        "evidence_refs": f.evidence_refs,
                        "description": f.description,
                        "suggestion": f.suggestion,
                        "status": f.status,
                    }
                    for f in ar.finding_set.findings
                ],
                "rejected": len(ar.finding_set.rejected),
                "errors": ar.errors,
                "tool_calls": ar.tool_calls,
                "context_stats": ar.context_stats,
                "provider_error": ar.provider_error,
                "evidence_pack": ar.evidence_pack,
                "elapsed_ms": ar.elapsed_ms,
                "backoff_ms": ar.backoff_ms,
            }
            result.findings_count = len(ar.finding_set.findings)
            result.errors.extend(ar.errors)
            print(f"  [analyze] found {result.findings_count} findings, {len(ar.finding_set.rejected)} rejected, {ar.elapsed_ms/1000:.1f}s")
        except Exception as e:
            result.errors.append(f"analyze: {e}")
            result.status = "error"
            print(f"  [analyze] FAILED: {e}")
            result.elapsed_s = time.monotonic() - t0
            result.debug_log = debug_log
            return result

        # -- SCORE --
        accepted_findings = [
            f for f in result.analyze.get("findings", [])
            if f.get("status") == "accepted"
        ]
        answer = {
            "findings": [
                {
                    "category": f["category"],
                    "file": f["file_path"],
                    "function": f["function"],
                    "line": f["line"],
                }
                for f in accepted_findings
            ]
        }

        score_data = None
        try:
            sys.path.insert(0, str(self._nsys_bench_dir / "tools"))
            from score_answer import score_answer
            score_data = score_answer(answer, truth)
            sys.path.pop(0)
            result.score = score_data["score"]
            result.matched_ids = score_data["matched_ids"]
            print(f"  [score] {result.score}/{result.total} ({100*result.score/max(1,result.total):.0f}%) matched={result.matched_ids}")
        except Exception as e:
            result.errors.append(f"scoring: {e}")
            print(f"  [score] FAILED: {e}")

        result.elapsed_s = time.monotonic() - t0
        result.debug_log = debug_log

        # Write per-case output
        self._write_case_outputs(result, case_out, answer, score_data)

        return result

    def _write_case_outputs(self, result: CaseResult, out_dir: Path, answer: dict, score_result: dict | None):
        """Write per-case output files."""
        (out_dir / "warmup_raw.json").write_text(
            json.dumps(result.warmup, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "analyze_raw.json").write_text(
            json.dumps(result.analyze, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "answer.json").write_text(
            json.dumps(answer, indent=2, ensure_ascii=False), encoding="utf-8")
        if score_result:
            (out_dir / "score.json").write_text(
                json.dumps(score_result, indent=2, ensure_ascii=False), encoding="utf-8")

        # debug.log is written turn-by-turn by DebugProvider; no batch write needed

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        if m > 0:
            return f"{m}m{s}s"
        return f"{s}s"

    @staticmethod
    def _case_stats(result: CaseResult) -> dict:
        """Extract token / turn / time metrics from a case result."""
        stats = {"turns": 0, "prompt_tokens": 0, "output_tokens": 0, "elapsed_s": result.elapsed_s, "backoff_s": 0}
        analyze = result.analyze or {}
        stats["backoff_s"] = (analyze.get("backoff_ms", 0) or 0) / 1000
        ctx = analyze.get("context_stats") or {}
        for turn_info in ctx.get("turns", []):
            stats["turns"] += 1
            stats["prompt_tokens"] += int(turn_info.get("prompt_tokens", 0) or 0)
            stats["output_tokens"] += int(turn_info.get("output_tokens", 0) or 0)
        return stats

    def _build_summary_lines(
        self, results: list[CaseResult], timestamp: str
    ) -> list[str]:
        lines: list[str] = []
        sep = "=" * 72
        lines.append(sep)
        lines.append(f"  BENCHMARK RESULTS  --  {timestamp}  --  mode: llm")
        lines.append(sep)

        # Score table
        lines.append(f"  {'Case':<16} {'Score':>10}  {'Turns':>5}  {'Tokens':>12}  {'Time':>7}  {'%':>5}")
        lines.append(f"  {'-'*66}")
        total_score = 0
        total_points = 0
        total_turns = 0
        total_prompt = 0
        total_output = 0
        total_elapsed = 0.0
        total_backoff = 0.0

        for r in results:
            pct = 100 * r.score / max(1, r.total)
            s = self._case_stats(r)
            tokens = s["prompt_tokens"] + s["output_tokens"]
            lines.append(
                f"  {r.case_id:<16} {r.score:>4}/{r.total:<5}"
                f"  {s['turns']:>5}  {tokens:>10,}  {self._fmt_time(s['elapsed_s']):>7}  {pct:>4.0f}%"
            )
            total_score += r.score
            total_points += r.total
            total_turns += s["turns"]
            total_prompt += s["prompt_tokens"]
            total_output += s["output_tokens"]
            total_elapsed += s["elapsed_s"]
            total_backoff += s["backoff_s"]

        lines.append(f"  {'-'*66}")
        total_pct = 100 * total_score / max(1, total_points)
        total_tokens = total_prompt + total_output
        lines.append(
            f"  {'TOTAL':<16} {total_score:>4}/{total_points:<5}"
            f"  {total_turns:>5}  {total_tokens:>10,}  {self._fmt_time(total_elapsed):>7}  {total_pct:>4.0f}%"
        )

        # Token / turn breakdown
        lines.append("")
        lines.append(f"  Prompt tokens: {total_prompt:,}  |  Output tokens: {total_output:,}  |  Total: {total_tokens:,}")
        if total_turns:
            lines.append(f"  Avg prompt/turn: {total_prompt // total_turns:,}  |  Avg output/turn: {total_output // total_turns:,}")
        if total_backoff > 0:
            pct = 100 * total_backoff / max(1, total_elapsed)
            lines.append(f"  Backoff wait: {self._fmt_time(total_backoff)} ({pct:.1f}% of total wall time)")
        lines.append(sep)
        return lines

    def _write_summary(self, results: list[CaseResult], run_dir: Path, timestamp: str):
        """Write summary.txt."""
        lines = self._build_summary_lines(results, timestamp)
        (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nOutput written to: {run_dir}")

    def _print_case_result(self, result: CaseResult):
        """Print single case result to terminal."""
        pct = 100 * result.score / max(1, result.total)
        status_icon = "OK" if result.status == "ok" else "FAIL"
        print(f"  {status_icon}: {result.case_id}  {result.score}/{result.total} ({pct:.0f}%)  [{self._fmt_time(result.elapsed_s)}]")
        if result.errors:
            for e in result.errors:
                print(f"    error: {e}")

    def _print_summary(self, results: list[CaseResult], timestamp: str):
        """Print summary to terminal."""
        for line in self._build_summary_lines(results, timestamp):
            print(line)
