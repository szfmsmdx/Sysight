import sqlite3
import tempfile
import unittest
from pathlib import Path


class TestNsysSqlNvtx(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sqlite = Path(self.tmp.name) / "profile.sqlite"
        conn = sqlite3.connect(str(self.sqlite))
        conn.execute("CREATE TABLE NVTX_EVENTS (start INTEGER NOT NULL, [end] INTEGER, text TEXT, textId INTEGER)")
        conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER NOT NULL, [end] INTEGER NOT NULL)")
        conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER NOT NULL, [end] INTEGER NOT NULL)")
        conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER NOT NULL, [end] INTEGER NOT NULL)")
        conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION (start INTEGER NOT NULL, [end] INTEGER NOT NULL)")
        conn.execute("INSERT INTO NVTX_EVENTS VALUES (0, 100, 'forward', NULL)")
        conn.execute("INSERT INTO NVTX_EVENTS VALUES (100, 160, 'forward', NULL)")
        conn.execute("INSERT INTO NVTX_EVENTS VALUES (200, 260, 'metrics', NULL)")
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10, 20)")
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (110, 120)")
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (11, 19)")
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (210, 220)")
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES (230, 240)")
        conn.commit()
        conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_aggregates_nvtx_ranges_and_contained_events(self):
        from sysight.tools.nsys_sql.nvtx import nvtx

        result = nvtx(str(self.sqlite), limit=10)
        by_name = {item.name: item for item in result.ranges}

        self.assertIn("forward", by_name)
        self.assertEqual(by_name["forward"].count, 2)
        self.assertEqual(by_name["forward"].total_ns, 160)
        self.assertEqual(by_name["forward"].kernel_count, 2)
        self.assertEqual(by_name["forward"].runtime_count, 1)
        self.assertEqual(by_name["metrics"].memcpy_count, 1)
        self.assertEqual(by_name["metrics"].sync_count, 1)


if __name__ == "__main__":
    unittest.main()
