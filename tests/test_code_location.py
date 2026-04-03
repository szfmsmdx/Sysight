from __future__ import annotations

import unittest
from pathlib import Path

from sysight.analysis.code_location import (
    _parse_function_label,
    format_code_locations,
    locate_code_for_gaps,
)
from sysight.profile import Profile


_PROFILE_PATH = (
    Path(__file__).resolve().parents[1] / "profiles" / "pnc_prof_0330_lane_selection_fakedata.sqlite"
)


class CodeLocationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = Profile(str(_PROFILE_PATH))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.profile.close()

    def test_parse_function_label_filters_iteration_markers(self) -> None:
        self.assertIsNone(_parse_function_label("iter_21"))
        parsed = _parse_function_label("PlanningSharedEncoderV5.forward")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["label"], "PlanningSharedEncoderV5.forward")

    def test_gap_location_contains_python_location_and_stack(self) -> None:
        windows = locate_code_for_gaps(self.profile, 0, max_windows=1)
        self.assertTrue(windows)
        window = windows[0]
        self.assertTrue(window["python_locations"])
        self.assertTrue(any(row.get("function") for row in window["python_locations"]))
        self.assertTrue(any(thread.get("stacks") for thread in window["threads"]))
        self.assertTrue(
            any(
                stack.get("preview")
                for thread in window["threads"]
                for stack in thread.get("stacks", [])
            )
        )

        formatted = format_code_locations(windows)
        self.assertIn("Python 位置:", formatted)
        self.assertIn("调用栈:", formatted)


if __name__ == "__main__":
    unittest.main()
