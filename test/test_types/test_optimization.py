"""Tests for compute_span_hash — the only behavior in contracts/optimization."""

import unittest

from sysight.types.optimization import (
    MeasurementPlan,
    MetricSpec,
    compute_span_hash,
)


class TestComputeSpanHash(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(compute_span_hash("x = 1"), compute_span_hash("x = 1"))

    def test_different_inputs_produce_different_hash(self):
        self.assertNotEqual(compute_span_hash("x = 1"), compute_span_hash("x = 2"))

    def test_output_is_12_hex_chars(self):
        h = compute_span_hash("hello world")
        self.assertEqual(len(h), 12)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))

    def test_empty_string_hashes(self):
        self.assertEqual(len(compute_span_hash("")), 12)


class TestMeasurementPlan(unittest.TestCase):
    def test_first_metric_becomes_primary(self):
        plan = MeasurementPlan.from_dict({
            "run_command": ["python", "run.py"],
            "metrics": [{"name": "iter_ms", "regex": r"iter (\d+) ms"}],
        })

        self.assertTrue(plan.is_valid())
        self.assertEqual(plan.primary_metric.name, "iter_ms")
        self.assertTrue(plan.metrics[0].primary)

    def test_metric_spec_defaults(self):
        metric = MetricSpec.from_dict({"name": "throughput", "regex": r"tok/s=(\d+)"})

        self.assertEqual(metric.group, 1)
        self.assertEqual(metric.aggregation, "mean")
        self.assertTrue(metric.lower_is_better)


if __name__ == "__main__":
    unittest.main()
