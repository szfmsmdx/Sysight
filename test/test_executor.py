"""Tests for sysight.executor."""

import unittest
from sysight.executor.models import ExecutionReport, ExecutedPatch
from sysight.executor.execute import determine_lower_is_better

class TestExecutorModels(unittest.TestCase):
    def test_execution_report_roundtrip(self):
        report_dict = {
            "baseline_score": 10.5,
            "patches": [
                {
                    "id": "patch_001",
                    "status": "committed",
                    "score_after": 15.0,
                    "delta": "+4.5"
                },
                {
                    "id": "patch_002",
                    "status": "skipped",
                    "score_after": 14.5,
                    "delta": "-0.5, reverted"
                }
            ],
            "final_score": 15.0
        }
        
        report = ExecutionReport.from_dict(report_dict)
        self.assertEqual(report.baseline_score, 10.5)
        self.assertEqual(len(report.patches), 2)
        self.assertEqual(report.patches[0].id, "patch_001")
        self.assertEqual(report.final_score, 15.0)

        out_dict = report.to_dict()
        self.assertEqual(out_dict, report_dict)

class TestExecutorHeuristics(unittest.TestCase):
    def test_determine_lower_is_better(self):
        self.assertTrue(determine_lower_is_better("iter_time"))
        self.assertTrue(determine_lower_is_better("LATENCY"))
        self.assertTrue(determine_lower_is_better("step_time_ms"))
        self.assertFalse(determine_lower_is_better("MFU"))
        self.assertFalse(determine_lower_is_better("throughput"))
        self.assertFalse(determine_lower_is_better("score"))

if __name__ == "__main__":
    unittest.main()
