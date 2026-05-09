"""OptimizerBenchmarkRunner — run OPTIMIZE against optimizer-bench cases, score results.

Mirrors BenchmarkRunner (analyze) pattern:
  - Creates .sysight/optimizer-bench-runs/<timestamp>/case_X/
  - Each case gets its own optimize_debug.log + optimize_result.json
  - Scores against ground truth using score_optimizer.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class CaseResult:
    case_id: str = ""
    status: str = ""                          # "ok" | "error"
    score: dict = field(default_factory=dict)  # {correctness, performance, judgment, minimality, total}
    patches_count: int = 0
    kept_count: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0


class OptimizerBenchmarkRunner:
    """Run optimizer-bench cases through OPTIMIZE, score against ground truth."""

    def __init__(
        self,
        bench_dir: str | Path = "optimizer-bench",
        output_dir: str | Path = ".sysight/optimizer-bench-runs",
        debug: bool = False,
    ):
        self._bench_dir = Path(bench_dir).resolve()
        self._output_dir = Path(output_dir).resolve()
        self._debug = debug
        self._registry = None
        self._knowledge = None
        self._configs = {}

    def run(self, case_ids: list[str]) -> dict:
        """Run all specified cases. Returns summary dict."""
        self._setup()

        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        run_dir = self._output_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"Optimizer bench output dir: {run_dir}")

        results: list[CaseResult] = []
        for case_id in case_ids:
            case_out = run_dir / case_id
            case_out.mkdir(parents=True, exist_ok=True)
            print(f"\n{'='*60}")
            print(f"  OPTIMIZE CASE: {case_id}")
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
            "cases": {
                r.case_id: {
                    "score": r.score.get("total", 0),
                    "max": r.score.get("max_total", 100),
                    "status": r.status,
                }
                for r in results
            },
        }

    def _setup(self):
        """Initialize registry, configs, knowledge."""
        from sysight.tools.registry import ToolRegistry
        from sysight.tools import register_all_tools
        from sysight.agent.config_loader import load_config
        from sysight.wiki.store import WikiRepository

        registry = ToolRegistry()
        register_all_tools(registry)
        self._registry = registry

        try:
            self._configs = load_config()
        except FileNotFoundError:
            self._configs = {}

        self._knowledge = WikiRepository()

    def _run_case(self, case_id: str, case_out: Path) -> CaseResult:
        """Run optimizer on a single case."""
        case_dir = self._bench_dir / "cases" / case_id
        truth_path = self._bench_dir / "tests" / "findings" / f"{case_id}_ground_truth.json"

        if not case_dir.is_dir():
            return CaseResult(case_id=case_id, status="error",
                              errors=[f"case dir not found: {case_dir}"])
        if not truth_path.exists():
            return CaseResult(case_id=case_id, status="error",
                              errors=[f"truth file not found: {truth_path}"])

        t0 = time.monotonic()

        try:
            truth = json.loads(truth_path.read_text(encoding="utf-8"))

            # Load pre-built artifacts
            artifacts_dir = case_dir / "artifacts"
            analyze_raw = artifacts_dir / "analyze_raw.json"
            if not analyze_raw.exists():
                return CaseResult(case_id=case_id, status="error",
                                  errors=[f"analyze_raw.json not found in artifacts"])

            # Load findings from pre-built analyze_raw.json
            findings = self._load_findings(analyze_raw)

            # Get provider
            provider = self._get_provider("optimize")
            if not provider:
                return CaseResult(case_id=case_id, status="error",
                                  errors=["no optimize provider configured"])

            # Run optimizer (plan only — no file changes)
            from sysight.pipeline.optimize import run_optimize
            patches = run_optimize(
                findings,
                str(case_dir),
                self._registry,
                provider,
                verbose=self._debug,
                run_dir=case_out,
            )

            # Run execute (apply patches, smoke test, timer comparison)
            from sysight.pipeline.execute import run_execute
            result = run_execute(
                patches,
                str(case_dir),
                run_id=findings.run_id,
                analyze_run_dir=artifacts_dir,
                run_dir=case_out,
            )

            elapsed_s = time.monotonic() - t0

            # Score against ground truth
            score = self._score_case(truth, result, case_out)

            return CaseResult(
                case_id=case_id,
                status="ok",
                score=score,
                patches_count=len(result.patches),
                kept_count=sum(1 for p in result.patches if p.status == "kept"),
                errors=result.errors,
                elapsed_s=elapsed_s,
            )

        except Exception as e:
            elapsed_s = time.monotonic() - t0
            return CaseResult(
                case_id=case_id,
                status="error",
                errors=[f"{type(e).__name__}: {e}"],
                elapsed_s=elapsed_s,
            )

    def _load_findings(self, analyze_raw_path: Path):
        """Load findings from pre-built analyze_raw.json."""
        data = json.loads(analyze_raw_path.read_text(encoding="utf-8"))
        from sysight.types.findings import LocalizedFindingSet, LocalizedFinding
        return LocalizedFindingSet(
            run_id=data.get("run_id", ""),
            summary=data.get("summary", ""),
            findings=[
                LocalizedFinding(
                    finding_id=f.get("finding_id", ""),
                    category=f.get("category", ""),
                    title=f.get("title", ""),
                    priority=f.get("priority", "medium"),
                    confidence=f.get("confidence", "unresolved"),
                    metric=f.get("metric", ""),
                    file_path=f.get("file_path"),
                    function=f.get("function"),
                    line=f.get("line"),
                    description=f.get("description", ""),
                    suggestion=f.get("suggestion", ""),
                    status=f.get("status", "accepted"),
                )
                for f in data.get("findings", [])
                if f.get("status") == "accepted"
            ],
        )

    def _get_provider(self, stage: str):
        """Get LLM provider for a stage."""
        from sysight.agent.provider import create_provider
        cfg = self._configs.get(stage)
        if not cfg or not cfg.api_key:
            return None
        return create_provider(cfg)

    def _score_case(self, truth: dict, result, case_out: Path) -> dict:
        """Score optimizer output against ground truth.

        Uses the same scoring logic as score_optimizer.py but inline
        to avoid subprocess dependency.
        """
        real_ids = set(truth.get("real_finding_ids", []))
        fake_ids = set(truth.get("fake_finding_ids", []))
        all_gt_ids = real_ids | fake_ids
        expected_lines = truth.get("expected_patch_lines", {})

        patches = result.patches
        verify = result.verify

        # ── 1. Correctness (40) ──
        correctness = 0
        if patches:
            all_kept = all(p.status == "kept" for p in patches)
            smoke_passed = getattr(verify, 'smoke_passed', False)
            if all_kept and smoke_passed:
                correctness = 40
            elif all_kept:
                correctness = 20
            else:
                correctness = 0

        # ── 2. Performance (30) ──
        performance = 0
        if real_ids:
            delta_pct = getattr(verify, 'delta_pct', {})
            perf_scores = []
            for fid in real_ids:
                delta = None
                for label, pct in delta_pct.items():
                    if fid in label or label in fid:
                        delta = pct
                        break
                if delta is not None and delta < -5.0:
                    perf_scores.append(1.0)
                elif delta is not None and delta < 0:
                    perf_scores.append(0.5)
                else:
                    perf_scores.append(0.0)
            if perf_scores:
                performance = int(30 * sum(perf_scores) / len(perf_scores))

        # ── 3. Judgment (20) — F1 on accept/reject ──
        judgment = 0
        accepted_ids = set()
        rejected_ids = set()
        if all_gt_ids:
            for p in patches:
                fids = set(p.finding_ids)
                if p.status == "kept":
                    accepted_ids.update(fids)
                else:
                    rejected_ids.update(fids)

            tp = len(accepted_ids & real_ids)
            fp = len(accepted_ids & fake_ids)
            fn = len(rejected_ids & real_ids)
            tn = len(rejected_ids & fake_ids)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            judgment = int(20 * f1)

        # ── 4. Minimality (10) ──
        minimality = 0
        if patches and expected_lines:
            min_scores = []
            for p in patches:
                fids = p.finding_ids
                if not fids:
                    continue
                expected = expected_lines.get(fids[0], 999)
                actual = getattr(p, 'lines_changed', 999)
                if actual <= expected * 1.2:
                    min_scores.append(1.0)
                elif actual <= expected * 2.0:
                    min_scores.append(0.5)
                else:
                    min_scores.append(0.0)
            if min_scores:
                minimality = int(10 * sum(min_scores) / len(min_scores))

        total = correctness + performance + judgment + minimality

        return {
            "correctness": correctness,
            "performance": performance,
            "judgment": judgment,
            "minimality": minimality,
            "total": total,
            "max_total": truth.get("max_score", 100),
            "details": {
                "tp": len(accepted_ids & real_ids) if all_gt_ids else 0,
                "fp": len(accepted_ids & fake_ids) if all_gt_ids else 0,
                "fn": len(rejected_ids & real_ids) if all_gt_ids else 0,
                "tn": len(rejected_ids & fake_ids) if all_gt_ids else 0,
            },
        }

    def _print_case_result(self, result: CaseResult) -> None:
        """Print single case result."""
        if result.status == "error":
            print(f"  ✗ ERROR: {result.errors}")
            return

        s = result.score
        print(f"\n  Score: {s.get('total', 0)}/{s.get('max_total', 100)}")
        print(f"    Correctness:  {s.get('correctness', 0):2d}/40")
        print(f"    Performance:  {s.get('performance', 0):2d}/30")
        print(f"    Judgment:     {s.get('judgment', 0):2d}/20  "
              f"(TP={s['details']['tp']} FP={s['details']['fp']} "
              f"FN={s['details']['fn']} TN={s['details']['tn']})")
        print(f"    Minimality:   {s.get('minimality', 0):2d}/10")
        print(f"    Patches: {result.patches_count} ({result.kept_count} kept)")
        print(f"    Elapsed: {result.elapsed_s:.1f}s")

    def _print_summary(self, results: list[CaseResult], timestamp: str) -> None:
        """Print overall summary."""
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  OPTIMIZER BENCHMARK SUMMARY  {timestamp}")
        print(f"{sep}")

        grand_total = 0
        grand_max = 0
        for r in results:
            if r.status == "error":
                print(f"  {r.case_id}: ERROR — {r.errors[0] if r.errors else 'unknown'}")
                continue
            s = r.score
            print(f"  {r.case_id}: {s.get('total', 0)}/{s.get('max_total', 100)} "
                  f"({r.patches_count} patches, {r.elapsed_s:.1f}s)")
            grand_total += s.get("total", 0)
            grand_max += s.get("max_total", 100)

        print(f"  {'─'*40}")
        print(f"  GRAND TOTAL: {grand_total}/{grand_max}")
        print(f"{sep}\n")

    def _write_summary(self, results: list[CaseResult], run_dir: Path, timestamp: str) -> None:
        """Write summary JSON to run directory."""
        summary = {
            "timestamp": timestamp,
            "cases": {},
            "grand_total": 0,
            "grand_max": 0,
        }
        for r in results:
            summary["cases"][r.case_id] = {
                "status": r.status,
                "score": r.score,
                "patches_count": r.patches_count,
                "kept_count": r.kept_count,
                "errors": r.errors,
                "elapsed_s": r.elapsed_s,
            }
            if r.status == "ok":
                summary["grand_total"] += r.score.get("total", 0)
                summary["grand_max"] += r.score.get("max_total", 100)

        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
