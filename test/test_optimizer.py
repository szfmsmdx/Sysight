"""Tests for sysight.optimizer."""

import unittest
from sysight.optimizer.models import MetricProbe, Patch, PatchPlan
from sysight.optimizer.plan import parse_patch_plan_output

class TestOptimizerModels(unittest.TestCase):
    def test_patch_plan_roundtrip(self):
        plan_dict = {
            "summary": "Fix experts.py overhead",
            "metric_probe": {
                "run_cmd": "python run.py",
                "grep_pattern": "iter_time",
                "baseline_hint": "run baseline first"
            },
            "patches": [
                {
                    "id": "patch_001",
                    "finding_id": "case_5_f010",
                    "file": "src/models/experts.py",
                    "diff": "--- a/src/models/experts.py\\n+++ b/src/models/experts.py\\n-    pass\\n+    pass",
                    "rationale": "Avoid batch loops",
                    "expected_metric": "iter_time -30%"
                }
            ]
        }
        
        plan = PatchPlan.from_dict(plan_dict)
        self.assertEqual(plan.summary, "Fix experts.py overhead")
        self.assertEqual(plan.metric_probe.grep_pattern, "iter_time")
        self.assertEqual(len(plan.patches), 1)
        self.assertEqual(plan.patches[0].finding_id, "case_5_f010")

        out_dict = plan.to_dict()
        self.assertEqual(out_dict, plan_dict)

class TestOptimizerPlan(unittest.TestCase):
    def test_parse_patch_plan_output(self):
        # Agent output containing thought process and then JSON block
        output = '''I will create a patch plan.
        
```json
{
  "summary": "Optimize experts",
  "metric_probe": {
    "run_cmd": "echo 1",
    "grep_pattern": "mfu",
    "baseline_hint": ""
  },
  "patches": []
}
```
Done.
'''
        plan = parse_patch_plan_output(output)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.summary, "Optimize experts")
        self.assertEqual(plan.metric_probe.grep_pattern, "mfu")
        self.assertEqual(len(plan.patches), 0)

    def test_parse_patch_plan_output_no_json(self):
        output = "I failed to generate."
        plan = parse_patch_plan_output(output)
        self.assertIsNone(plan)

if __name__ == "__main__":
    unittest.main()
