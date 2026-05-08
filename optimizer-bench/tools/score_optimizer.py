"""Optimizer benchmark scoring tool.

Scores optimizer output against ground truth across four dimensions:
  correctness (40) — patch apply + smoke test success
  performance (30) — timer delta_pct for each real finding
  judgment   (20) — correctly accept/reject findings (F1 score)
  minimality (10) — patch diff size within expected bounds
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_ground_truth(case_dir: Path) -> dict:
    """Load ground truth for a case."""
    gt_path = case_dir.parent.parent / "tests" / "findings" / f"{case_dir.name}_ground_truth.json"
    return json.loads(gt_path.read_text(encoding="utf-8"))


def load_optimizer_output(run_dir: Path) -> dict:
    """Load optimizer output from a run directory."""
    result_path = run_dir / "optimize_result.json"
    if not result_path.exists():
        return {}
    return json.loads(result_path.read_text(encoding="utf-8"))


def score_case(ground_truth: dict, optimizer_output: dict) -> dict:
    """Score a single case.

    Returns dict with per-dimension scores and total.
    """
    real_ids = set(ground_truth["real_finding_ids"])
    fake_ids = set(ground_truth["fake_finding_ids"])
    all_gt_ids = real_ids | fake_ids
    expected_lines = ground_truth.get("expected_patch_lines", {})

    patches = optimizer_output.get("patches", [])
    verify = optimizer_output.get("verify", {})

    # ── 1. Correctness (40) ──
    correctness = 0
    if patches:
        all_kept = all(p.get("status") == "kept" for p in patches)
        smoke_passed = verify.get("smoke_passed", False)
        if all_kept and smoke_passed:
            correctness = 40
        elif all_kept:
            correctness = 20  # applied but smoke failed
        else:
            correctness = 0   # apply failed

    # ── 2. Performance (30) ──
    performance = 0
    if real_ids:
        delta_pct = verify.get("delta_pct", {})
        perf_scores = []
        for fid in real_ids:
            # Match by finding_id → timer_label mapping
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
    if all_gt_ids:
        accepted_ids = set()
        rejected_ids = set()
        for p in patches:
            fids = p.get("finding_ids", [])
            if p.get("status") == "kept":
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
            fids = p.get("finding_ids", [])
            if not fids:
                continue
            expected = expected_lines.get(fids[0], 999)
            actual = p.get("lines_changed", 999)
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
        "case_id": ground_truth.get("case_id", ""),
        "correctness": correctness,
        "performance": performance,
        "judgment": judgment,
        "minimality": minimality,
        "total": total,
        "max_total": ground_truth.get("max_score", 100),
        "details": {
            "tp": len(accepted_ids & real_ids) if all_gt_ids else 0,
            "fp": len(accepted_ids & fake_ids) if all_gt_ids else 0,
            "fn": len(rejected_ids & real_ids) if all_gt_ids else 0,
            "tn": len(rejected_ids & fake_ids) if all_gt_ids else 0,
        },
    }


def score_all(cases_dir: Path, runs_dir: Path) -> dict:
    """Score all cases and return summary."""
    results = {}
    for case_dir in sorted(cases_dir.iterdir()):
        if not case_dir.is_dir() or not case_dir.name.startswith("case_"):
            continue
        gt = load_ground_truth(case_dir)
        run_dir = runs_dir / case_dir.name
        output = load_optimizer_output(run_dir)
        if not output:
            results[case_dir.name] = {"error": "no optimizer output found"}
            continue
        results[case_dir.name] = score_case(gt, output)
    return results


def print_summary(results: dict) -> None:
    """Print scoring summary to stdout."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  OPTIMIZER BENCHMARK RESULTS")
    print(f"{sep}")

    grand_total = 0
    grand_max = 0

    for case_name in sorted(results.keys()):
        r = results[case_name]
        if "error" in r:
            print(f"  {case_name}: ERROR — {r['error']}")
            continue
        print(f"\n  {case_name}")
        print(f"    Correctness:  {r['correctness']:2d}/40")
        print(f"    Performance:  {r['performance']:2d}/30")
        print(f"    Judgment:     {r['judgment']:2d}/20  "
              f"(TP={r['details']['tp']} FP={r['details']['fp']} "
              f"FN={r['details']['fn']} TN={r['details']['tn']})")
        print(f"    Minimality:   {r['minimality']:2d}/10")
        print(f"    ─────────────────────")
        print(f"    Total:        {r['total']:2d}/{r['max_total']}")
        grand_total += r["total"]
        grand_max += r["max_total"]

    print(f"\n{sep}")
    print(f"  GRAND TOTAL: {grand_total}/{grand_max}")
    print(f"{sep}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Score optimizer benchmark results")
    parser.add_argument("--cases-dir", default="cases", help="Path to cases directory")
    parser.add_argument("--runs-dir", default=".sysight/optimizer-runs", help="Path to run outputs")
    parser.add_argument("--case", help="Score a single case (e.g. case_1)")
    args = parser.parse_args()

    cases_dir = Path(args.cases_dir).resolve()
    runs_dir = Path(args.runs_dir).resolve()

    if args.case:
        case_dir = cases_dir / args.case
        if not case_dir.is_dir():
            print(f"ERROR: case not found: {case_dir}", file=sys.stderr)
            sys.exit(1)
        gt = load_ground_truth(case_dir)
        run_dir = runs_dir / args.case
        output = load_optimizer_output(run_dir)
        if not output:
            print(f"ERROR: no optimizer output for {args.case}", file=sys.stderr)
            sys.exit(1)
        result = score_case(gt, output)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        results = score_all(cases_dir, runs_dir)
        print_summary(results)
