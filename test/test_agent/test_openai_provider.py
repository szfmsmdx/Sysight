import unittest
import urllib.error
import urllib.request
from io import BytesIO

from sysight.agent.provider import LLMConfig, LLMRequest
from sysight.agent.providers.openai_compatible import OpenAICompatibleProvider
from sysight.agent.providers.openai_compatible import _redact_error


class TestOpenAICompatibleProvider(unittest.TestCase):
    def test_redacts_secret_like_error_values(self):
        text = 'authorization: Bearer sk-live token="abc123" api_key=xyz message=bad'

        redacted = _redact_error(text)

        self.assertNotIn("abc123", redacted)
        self.assertNotIn("xyz", redacted)
        self.assertIn("<REDACTED>", redacted)
        self.assertIn("message=bad", redacted)

    def test_deepseek_v4_enables_thinking_and_max_effort(self):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"choices":[{"message":{"content":"{}"},'
                    b'"finish_reason":"stop"}],"usage":{"prompt_tokens":1,'
                    b'"completion_tokens":2}}'
                )

        def fake_urlopen(req, timeout):
            captured.update(__import__("json").loads(req.data.decode("utf-8")))
            return FakeResponse()

        original = urllib.request.urlopen
        urllib.request.urlopen = fake_urlopen
        try:
            provider = OpenAICompatibleProvider(LLMConfig(
                provider="openai",
                model="deepseek-v4-pro",
                api_key="test",
                base_url="https://api.deepseek.com/v1",
            ))
            provider.complete(LLMRequest(messages=[{"role": "user", "content": "hi"}]))
        finally:
            urllib.request.urlopen = original

        self.assertEqual(captured["thinking"], {"type": "enabled"})
        self.assertEqual(captured["reasoning_effort"], "max")
        self.assertNotIn("temperature", captured)

    def test_http_error_has_structured_error_info(self):
        def fake_urlopen(req, timeout):
            body = b'{"error":{"message":"bad api_key=abc","type":"invalid_request_error","code":"bad_request","param":"messages"}}'
            raise urllib.error.HTTPError(
                req.full_url, 400, "Bad Request", {}, BytesIO(body)
            )

        original = urllib.request.urlopen
        urllib.request.urlopen = fake_urlopen
        try:
            provider = OpenAICompatibleProvider(LLMConfig(
                provider="openai",
                model="deepseek-v4-pro",
                api_key="test",
                base_url="https://api.deepseek.com/v1",
            ))
            response = provider.complete(LLMRequest(messages=[{"role": "user", "content": "hi"}]))
        finally:
            urllib.request.urlopen = original

        self.assertEqual(response.finish_reason, "error")
        self.assertIsNotNone(response.error)
        self.assertEqual(response.error.status_code, 400)
        self.assertEqual(response.error.code, "bad_request")
        self.assertIn("<REDACTED>", response.error.message)
        self.assertFalse(response.error.retryable)

    def test_deepseek_v4_no_reasoning_effort_when_thinking_disabled(self):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"choices":[{"message":{"content":"{}"},'
                    b'"finish_reason":"stop"}],"usage":{"prompt_tokens":1,'
                    b'"completion_tokens":2}}'
                )

        def fake_urlopen(req, timeout):
            captured.update(__import__("json").loads(req.data.decode("utf-8")))
            return FakeResponse()

        original = urllib.request.urlopen
        urllib.request.urlopen = fake_urlopen
        try:
            provider = OpenAICompatibleProvider(LLMConfig(
                provider="openai",
                model="deepseek-v4-pro",
                api_key="test",
                base_url="https://api.deepseek.com/v1",
                thinking={"type": "disabled"},
            ))
            provider.complete(LLMRequest(messages=[{"role": "user", "content": "hi"}]))
        finally:
            urllib.request.urlopen = original

        self.assertEqual(captured["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", captured)


if __name__ == "__main__":
    unittest.main()
