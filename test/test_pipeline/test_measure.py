from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sysight.pipeline.measure import compare_measurements, run_measurement
from sysight.types.optimization import MeasurementPlan, MetricSpec


class TestMeasurement(unittest.TestCase):
    def test_extracts_and_aggregates_stdout_metric(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "run.py").write_text(
                "print('iteration 10 ms')\n"
                "print('iteration 20 ms')\n"
                "print('iteration 30 ms')\n",
                encoding="utf-8",
            )
            plan = MeasurementPlan(
                run_command=["python", "run.py"],
                metrics=[
                    MetricSpec(
                        name="iteration_ms",
                        regex=r"iteration (\d+) ms",
                        aggregation="mean",
                        drop_first_n=1,
                        lower_is_better=True,
                        primary=True,
                    )
                ],
            )

            result = run_measurement(root, plan, root / "out", "baseline")

            self.assertEqual(result.status, "ok")
            self.assertEqual(result.samples["iteration_ms"], [20.0, 30.0])
            self.assertEqual(result.primary_value, 25.0)

    def test_compare_lower_is_better(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan = MeasurementPlan(
                run_command=["python", "run.py"],
                metrics=[MetricSpec(name="iteration_ms", regex=r"iteration (\d+) ms", primary=True)],
            )
            (root / "run.py").write_text("print('iteration 10 ms')\n", encoding="utf-8")
            before = run_measurement(root, plan, root / "out", "before")
            (root / "run.py").write_text("print('iteration 5 ms')\n", encoding="utf-8")
            after = run_measurement(root, plan, root / "out", "after")

            accepted, delta_pct, _ = compare_measurements(before, after, plan)

            self.assertTrue(accepted)
            self.assertEqual(delta_pct, 50.0)


if __name__ == "__main__":
    unittest.main()
