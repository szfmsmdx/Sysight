import json
import unittest

from sysight.agent.loop import AgentLoop, AgentTask
from sysight.agent.provider import LLMResponse, ToolCallRequest, UsageInfo
from sysight.benchmark.debug import DebugProvider
from sysight.tools.registry import ToolDef, ToolPolicy, ToolRegistry


LONG_RESULT = "LONG_RESULT_MARKER " + ("x" * 1200)


def _long_tool(label: str = "first"):
    return {"label": label, "payload": LONG_RESULT}


class RecordingProvider:
    def __init__(self):
        self.requests = []

    @property
    def name(self):
        return "recording"

    @property
    def model(self):
        return "fake"

    def complete(self, request):
        self.requests.append(request)
        turn = len(self.requests)
        if turn == 1:
            return LLMResponse(
                finish_reason="tool_calls",
                tool_calls=[ToolCallRequest(id="call-1", name="test_long", arguments={"label": "first"})],
                usage=UsageInfo(prompt_tokens=10, output_tokens=1),
            )
        if turn == 2:
            return LLMResponse(
                finish_reason="tool_calls",
                tool_calls=[ToolCallRequest(id="call-2", name="test_long", arguments={"label": "second"})],
                usage=UsageInfo(prompt_tokens=20, output_tokens=1),
            )
        return LLMResponse(
            content=json.dumps({"summary": "ok", "findings": [], "memory_updates": []}),
            finish_reason="stop",
            usage=UsageInfo(prompt_tokens=30, output_tokens=3),
        )


class RepeatingToolProvider:
    def __init__(self, repeats=4):
        self.requests = []
        self.repeats = repeats

    @property
    def name(self):
        return "repeating"

    @property
    def model(self):
        return "fake"

    def complete(self, request):
        self.requests.append(request)
        turn = len(self.requests)
        if turn <= self.repeats:
            return LLMResponse(
                finish_reason="tool_calls",
                tool_calls=[ToolCallRequest(
                    id=f"call-{turn}",
                    name="test_long",
                    arguments={"label": f"repeat-{turn}"},
                )],
                usage=UsageInfo(prompt_tokens=10, output_tokens=1),
            )
        return LLMResponse(
            content=json.dumps({"summary": "ok", "findings": [], "memory_updates": []}),
            finish_reason="stop",
            usage=UsageInfo(prompt_tokens=30, output_tokens=3),
        )


class TestAgentContext(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(ToolDef(
            name="test_long",
            description="Return a long payload",
            parameters={
                "type": "object",
                "properties": {"label": {"type": "string"}},
                "required": [],
            },
            fn=_long_tool,
            read_only=True,
        ))
        self.policy = ToolPolicy(allowed_tools={"test_long"}, read_only=True)

    def test_tool_result_full_once_then_compacted(self):
        provider = RecordingProvider()
        loop = AgentLoop(provider, self.registry, self.policy)

        result = loop.run(AgentTask(
            run_id="r1",
            task_id="t1",
            task_type="analyze",
            user_prompt="investigate",
            max_turns=3,
        ))

        self.assertEqual(result.status, "ok")
        second_request_tool = _tool_message(provider.requests[1].messages, "call-1")
        self.assertIn("LONG_RESULT_MARKER", second_request_tool["content"])
        self.assertNotIn("sysight_compacted_tool_result", second_request_tool["content"])

        third_first_tool = _tool_message(provider.requests[2].messages, "call-1")
        third_second_tool = _tool_message(provider.requests[2].messages, "call-2")
        self.assertIn("sysight_compacted_tool_result", third_first_tool["content"])
        self.assertLess(len(third_first_tool["content"]), len(second_request_tool["content"]))
        self.assertIn("LONG_RESULT_MARKER", third_second_tool["content"])
        self.assertGreater(result.context_stats["last"]["compacted_tool_results"], 0)

    def test_debug_log_keeps_full_messages(self):
        real_provider = RecordingProvider()
        debug_log = []
        provider = DebugProvider(real_provider, log=debug_log, verbose=False)
        loop = AgentLoop(provider, self.registry, self.policy)

        loop.run(AgentTask(
            run_id="r1",
            task_id="t1",
            task_type="analyze",
            user_prompt="investigate",
            max_turns=3,
        ))

        final_debug_messages = debug_log[2]["request"]["messages"]
        final_model_summary = debug_log[2]["request"]["model_messages_summary"]
        first_full_tool = _tool_message(final_debug_messages, "call-1")

        self.assertIn("LONG_RESULT_MARKER", first_full_tool["content"])
        self.assertGreater(
            len(str(final_debug_messages)),
            sum(item["content_len"] for item in final_model_summary),
        )

    def test_repeated_tool_calls_do_not_inject_user_notes(self):
        provider = RepeatingToolProvider(repeats=4)
        loop = AgentLoop(provider, self.registry, self.policy)

        result = loop.run(AgentTask(
            run_id="r1",
            task_id="t1",
            task_type="analyze",
            user_prompt="investigate",
            max_turns=5,
        ))

        self.assertEqual(result.status, "ok")
        for request in provider.requests:
            _assert_openai_tool_protocol(request.messages)
            self.assertNotIn(
                "You have been calling the same tools",
                json.dumps(request.messages, ensure_ascii=False),
            )


def _tool_message(messages, tool_call_id):
    for message in messages:
        if message.get("role") == "tool" and message.get("tool_call_id") == tool_call_id:
            return message
    raise AssertionError(f"tool message not found: {tool_call_id}")


def _assert_openai_tool_protocol(messages):
    for idx, message in enumerate(messages):
        tool_calls = message.get("tool_calls") or []
        if message.get("role") != "assistant" or not tool_calls:
            continue
        expected_ids = [
            call.get("id")
            for call in tool_calls
            if call.get("id")
        ]
        following = messages[idx + 1: idx + 1 + len(expected_ids)]
        actual_ids = [
            item.get("tool_call_id")
            for item in following
            if item.get("role") == "tool"
        ]
        if actual_ids != expected_ids:
            raise AssertionError(
                f"assistant tool_calls at message {idx} not immediately followed "
                f"by matching tool results: expected={expected_ids} actual={actual_ids}"
            )


if __name__ == "__main__":
    unittest.main()
