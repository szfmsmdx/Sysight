"""Tests for sysight.tools.registry."""

import unittest

from sysight.tools.registry import (
    ToolDef,
    ToolPolicy,
    ToolResult,
    ToolRegistry,
    ANALYZE_POLICY,
    OPTIMIZE_POLICY,
)


def _echo(**kwargs):
    return kwargs


ECHO_TOOL_RO = ToolDef(
    name="test.echo_ro",
    description="Echo read-only",
    parameters={"type": "object", "properties": {}, "required": []},
    fn=_echo,
    read_only=True,
)

ECHO_TOOL_RW = ToolDef(
    name="test.echo_rw",
    description="Echo write-capable",
    parameters={"type": "object", "properties": {}, "required": []},
    fn=_echo,
    read_only=False,
)

NO_ARG_TOOL = ToolDef(
    name="test.noarg",
    description="Takes no args",
    parameters={"type": "object", "properties": {}, "required": []},
    fn=lambda: 42,
    read_only=True,
)


class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        self.reg = ToolRegistry()
        self.reg.register(ECHO_TOOL_RO)
        self.reg.register(ECHO_TOOL_RW)
        self.reg.register(NO_ARG_TOOL)

    def test_get_existing(self):
        t = self.reg.get("test.echo_ro")
        self.assertIsNotNone(t)
        self.assertEqual(t.name, "test.echo_ro")

    def test_get_nonexistent(self):
        self.assertIsNone(self.reg.get("nonexistent"))

    def test_execute_ok(self):
        policy = ToolPolicy(allowed_tools={"test.*"}, read_only=True)
        result = self.reg.execute("test.echo_ro", {"x": 1}, policy)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data, {"x": 1})

    def test_execute_unknown_tool(self):
        policy = ToolPolicy(allowed_tools={"test.*"}, read_only=True)
        result = self.reg.execute("nonexistent", {}, policy)
        self.assertEqual(result.status, "error")
        self.assertIn("Unknown tool", result.error)

    def test_execute_not_allowed(self):
        policy = ToolPolicy(allowed_tools={"scanner.*"}, read_only=True)
        result = self.reg.execute("test.echo_ro", {}, policy)
        self.assertEqual(result.status, "policy_denied")

    def test_execute_read_only_policy_blocks_write_tool(self):
        policy = ToolPolicy(allowed_tools={"test.*"}, read_only=True)
        result = self.reg.execute("test.echo_rw", {}, policy)
        self.assertEqual(result.status, "policy_denied")
        self.assertIn("not read-only", result.error)

    def test_execute_write_tool_with_write_policy(self):
        policy = ToolPolicy(allowed_tools={"test.*"}, read_only=False)
        result = self.reg.execute("test.echo_rw", {"key": "val"}, policy)
        self.assertEqual(result.status, "ok")

    def test_execute_no_args_tool(self):
        policy = ToolPolicy(allowed_tools={"test.*"}, read_only=True)
        result = self.reg.execute("test.noarg", {}, policy)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data, 42)

    def test_execute_max_calls(self):
        ECHO_TOOL_RO.max_calls_per_task = 2
        policy = ToolPolicy(allowed_tools={"test.*"}, read_only=True)
        self.reg.execute("test.echo_ro", {}, policy)
        self.reg.execute("test.echo_ro", {}, policy)
        result = self.reg.execute("test.echo_ro", {}, policy)
        self.assertEqual(result.status, "policy_denied")
        self.assertIn("Max calls", result.error)

    def test_reset_call_counts(self):
        ECHO_TOOL_RO.max_calls_per_task = 1
        policy = ToolPolicy(allowed_tools={"test.*"}, read_only=True)
        self.reg.execute("test.echo_ro", {}, policy)
        self.reg.reset_call_counts()
        result = self.reg.execute("test.echo_ro", {}, policy)
        self.assertEqual(result.status, "ok")

    def test_list_read_only(self):
        tools = self.reg.list_read_only()
        names = {t.name for t in tools}
        self.assertIn("test.echo_ro", names)
        self.assertNotIn("test.echo_rw", names)

    def test_list_for_policy_wildcard(self):
        policy = ToolPolicy(allowed_tools={"test.echo_*"}, read_only=True)
        tools = self.reg.list_for_policy(policy)
        names = {t.name for t in tools}
        self.assertIn("test.echo_ro", names)
        self.assertIn("test.echo_rw", names)

    def test_list_for_policy_exact_match(self):
        policy = ToolPolicy(allowed_tools={"test.echo_ro"}, read_only=True)
        tools = self.reg.list_for_policy(policy)
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "test.echo_ro")

    def test_as_openai_tools(self):
        policy = ToolPolicy(allowed_tools={"test.noarg"}, read_only=True)
        tools = self.reg.as_openai_tools(policy)
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "test.noarg")

    def test_as_anthropic_tools(self):
        policy = ToolPolicy(allowed_tools={"test.echo_ro"}, read_only=True)
        tools = self.reg.as_anthropic_tools(policy)
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "test.echo_ro")
        self.assertIn("input_schema", tools[0])


class TestPreBuiltPolicies(unittest.TestCase):
    def test_analyze_policy_read_only(self):
        self.assertTrue(ANALYZE_POLICY.read_only)

    def test_analyze_policy_includes_scanner(self):
        reg = ToolRegistry()
        from sysight.tools.scanner.files import FILES_TOOL
        reg.register(FILES_TOOL)
        tools = reg.list_for_policy(ANALYZE_POLICY)
        self.assertGreaterEqual(len(tools), 1)

    def test_optimize_policy_not_read_only(self):
        self.assertFalse(OPTIMIZE_POLICY.read_only)


if __name__ == "__main__":
    unittest.main()
