import unittest

from sysight.agent.provider import LLMErrorInfo, LLMRequest, LLMResponse
from sysight.benchmark.debug import DebugProvider


class FakeProvider:
    @property
    def name(self):
        return "fake"

    @property
    def model(self):
        return "fake-model"

    def complete(self, request):
        return LLMResponse(
            content="line-a\nline-b\nline-c",
            finish_reason="stop",
            extra={"reasoning_content": "reason-1\nreason-2\nreason-3"},
        )


class ErrorProvider:
    @property
    def name(self):
        return "fake"

    @property
    def model(self):
        return "fake-model"

    def complete(self, request):
        return LLMResponse(
            finish_reason="error",
            error=LLMErrorInfo(
                provider="fake",
                status_code=400,
                code="invalid_request",
                message="bad request",
                type="invalid_request_error",
                retryable=False,
            ),
        )


class TestBenchmarkDebug(unittest.TestCase):
    def test_debug_provider_keeps_full_log_entry(self):
        log = []
        provider = DebugProvider(FakeProvider(), log=log, verbose=False)
        request = LLMRequest(
            system_prompt="\n".join(f"system-{i}" for i in range(20)),
            messages=[{"role": "user", "content": "\n".join(f"msg-{i}" for i in range(20))}],
        )

        provider.complete(request)

        self.assertEqual(log[0]["request"]["system_prompt"], request.system_prompt)
        self.assertEqual(log[0]["request"]["messages"][0]["content"], request.messages[0]["content"])
        self.assertIn("line-c", log[0]["response"]["content"])
        self.assertIn("reason-3", log[0]["response"]["extra"]["reasoning_content"])
        self.assertNotIn("more lines", str(log[0]))
        self.assertNotIn("more chars", str(log[0]))

    def test_debug_provider_records_structured_error(self):
        log = []
        provider = DebugProvider(ErrorProvider(), log=log, verbose=False)

        provider.complete(LLMRequest(messages=[{"role": "user", "content": "hi"}]))

        self.assertEqual(log[0]["response"]["finish_reason"], "error")
        self.assertEqual(log[0]["response"]["error"]["code"], "invalid_request")


if __name__ == "__main__":
    unittest.main()
