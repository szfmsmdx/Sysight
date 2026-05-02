"""Tests for make_finding_id — the only behavior in contracts/findings."""

import unittest

from sysight.types.findings import make_finding_id


class TestMakeFindingId(unittest.TestCase):
    def test_deterministic(self):
        a = make_finding_id("C2", "src/model.py", 42, "forward")
        b = make_finding_id("C2", "src/model.py", 42, "forward")
        self.assertEqual(a, b)

    def test_differs_by_category(self):
        a = make_finding_id("C2", "src/x.py", 10, "f")
        b = make_finding_id("C4", "src/x.py", 10, "f")
        self.assertNotEqual(a, b)

    def test_differs_by_file(self):
        a = make_finding_id("C2", "src/a.py", 10, "f")
        b = make_finding_id("C2", "src/b.py", 10, "f")
        self.assertNotEqual(a, b)

    def test_differs_by_line(self):
        a = make_finding_id("C2", "src/x.py", 10, "f")
        b = make_finding_id("C2", "src/x.py", 20, "f")
        self.assertNotEqual(a, b)

    def test_differs_by_function(self):
        a = make_finding_id("C2", "src/x.py", 10, "f")
        b = make_finding_id("C2", "src/x.py", 10, "g")
        self.assertNotEqual(a, b)

    def test_format_is_category_colon_8hex(self):
        fid = make_finding_id("C7", "a.py", 1, "main")
        self.assertRegex(fid, r"^C7:[0-9a-f]{8}$")

    def test_none_fields_handled(self):
        fid = make_finding_id("C1", None, None, None)
        self.assertTrue(fid.startswith("C1:"))
        self.assertEqual(len(fid), 11)


if __name__ == "__main__":
    unittest.main()
