"""Entry point: python -m sysight.benchmark --case case_1,case_2 [--debug]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="sysight.benchmark",
        description="Benchmark Sysight WARMUP + ANALYZE against nsys-bench cases",
    )
    parser.add_argument(
        "--case", required=True,
        help="Comma-separated case IDs, e.g. 'case_1,case_3,case_5'",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Log all LLM requests/responses to debug.log",
    )
    parser.add_argument(
        "--force-warmup", action="store_true",
        help="Force re-run warmup even if cache exists",
    )
    parser.add_argument(
        "--no-warmup", action="store_true",
        help="Skip warmup entirely (use existing cache; error if no cache found)",
    )
    parser.add_argument(
        "--output-dir", default=".sysight/bench-runs",
        help="Output directory for benchmark results (default: .sysight/bench-runs)",
    )
    parser.add_argument(
        "--nsys-bench-dir", default="nsys-bench",
        help="Path to nsys-bench project (default: ./nsys-bench)",
    )
    args = parser.parse_args(argv)

    case_ids = [c.strip() for c in args.case.split(",") if c.strip()]
    if not case_ids:
        print("Error: --case requires at least one case ID", file=sys.stderr)
        sys.exit(1)

    nsys_bench_dir = Path(args.nsys_bench_dir)
    if not nsys_bench_dir.is_dir():
        print(f"Error: nsys-bench dir not found: {nsys_bench_dir}", file=sys.stderr)
        sys.exit(1)

    from sysight.benchmark.runner import BenchmarkRunner
    runner = BenchmarkRunner(
        nsys_bench_dir=nsys_bench_dir,
        output_dir=args.output_dir,
        debug=args.debug,
        force_warmup=args.force_warmup,
        no_warmup=args.no_warmup,
    )
    runner.run(case_ids)


if __name__ == "__main__":
    main()
