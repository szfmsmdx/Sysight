"""Validate optimizer-bench structure and artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


BENCH_DIR = Path(__file__).parent.parent


def validate_case(case_dir: Path) -> list[str]:
    """Validate a single case directory. Returns list of errors."""
    errors = []
    name = case_dir.name

    # Required files
    required = [
        "case.yaml",
        "requirements.txt",
        "run.py",
        "configs",
        "src/__init__.py",
        "artifacts/warmup_result.json",
        "artifacts/analyze_raw.json",
        "artifacts/instrument_result.json",
        "artifacts/timer_before.json",
    ]
    for r in required:
        if not (case_dir / r).exists():
            errors.append(f"{name}: missing {r}")

    # Validate analyze_raw.json
    analyze_path = case_dir / "artifacts" / "analyze_raw.json"
    if analyze_path.exists():
        try:
            data = json.loads(analyze_path.read_text())
            findings = data.get("findings", [])
            if not findings:
                errors.append(f"{name}: analyze_raw.json has no findings")
            for f in findings:
                if "finding_id" not in f:
                    errors.append(f"{name}: finding missing finding_id")
        except json.JSONDecodeError as e:
            errors.append(f"{name}: invalid analyze_raw.json: {e}")

    # Validate instrument_result.json
    inst_path = case_dir / "artifacts" / "instrument_result.json"
    if inst_path.exists():
        try:
            data = json.loads(inst_path.read_text())
            timers = data.get("timers", [])
            if not timers:
                errors.append(f"{name}: instrument_result.json has no timers")
        except json.JSONDecodeError as e:
            errors.append(f"{name}: invalid instrument_result.json: {e}")

    # Validate ground truth
    gt_path = BENCH_DIR / "tests" / "findings" / f"{name}_ground_truth.json"
    if gt_path.exists():
        try:
            data = json.loads(gt_path.read_text())
            real = set(data.get("real_finding_ids", []))
            fake = set(data.get("fake_finding_ids", []))
            if not real:
                errors.append(f"{name}: ground truth has no real_finding_ids")
            if not fake:
                errors.append(f"{name}: ground truth has no fake_finding_ids")
            overlap = real & fake
            if overlap:
                errors.append(f"{name}: finding_ids overlap between real and fake: {overlap}")
        except json.JSONDecodeError as e:
            errors.append(f"{name}: invalid ground truth: {e}")
    else:
        errors.append(f"{name}: missing ground truth file")

    return errors


def main():
    cases_dir = BENCH_DIR / "cases"
    all_errors = []

    for case_dir in sorted(cases_dir.iterdir()):
        if not case_dir.is_dir() or not case_dir.name.startswith("case_"):
            continue
        errors = validate_case(case_dir)
        all_errors.extend(errors)

    if all_errors:
        print("VALIDATION ERRORS:")
        for e in all_errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("✓ All cases validated successfully!")
        # Count findings
        total_real = 0
        total_fake = 0
        for case_dir in sorted(cases_dir.iterdir()):
            if not case_dir.is_dir() or not case_dir.name.startswith("case_"):
                continue
            gt_path = BENCH_DIR / "tests" / "findings" / f"{case_dir.name}_ground_truth.json"
            gt = json.loads(gt_path.read_text())
            real = len(gt["real_finding_ids"])
            fake = len(gt["fake_finding_ids"])
            total_real += real
            total_fake += fake
            print(f"  {case_dir.name}: {real} real + {fake} fake findings")
        print(f"  Total: {total_real} real + {total_fake} fake = {total_real + total_fake} findings")


if __name__ == "__main__":
    main()
