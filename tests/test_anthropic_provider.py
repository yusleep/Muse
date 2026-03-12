"""Tests for Anthropic api_style support in providers."""
import unittest
from muse.providers import HttpClient, LLMClient, ProviderError


def _make_anthropic_client(fake_http, model="claude-opus-4-6"):
    """Helper: LLMClient wired to an Anthropic provider config."""
    router_config = {
        "auth": {
            "profiles": {
                "anthro_key": {"apiKey": "sk-ant-test"}
            }
        },
        "providers": {
            "anthropic": {
                "baseUrl": "https://api.anthropic.com",
                "apiStyle": "anthropic",
                "auth": "anthro_key",
                "models": {
                    f"anthropic/{model}": {"model": model}
                },
            }
        },
        "models": {
            "default": {"primary": f"anthropic/{model}", "fallbacks": []},
            "outline": {"primary": f"anthropic/{model}", "fallbacks": []},
            "writing": {"primary": f"anthropic/{model}", "fallbacks": []},
            "review": {"primary": f"anthropic/{model}", "fallbacks": []},
            "polish": {"primary": f"anthropic/{model}", "fallbacks": []},
            "reasoning": {"primary": f"anthropic/{model}", "fallbacks": []},
        },
        "modelAliases": {},
    }
    client = LLMClient(
        api_key="unused",
        base_url="https://unused.local",
        model="unused",
        http=fake_http,
        model_router_config=router_config,
    )
    return client


class _FakeHttp(HttpClient):
    def __init__(self, response):
        super().__init__(timeout_seconds=1)
        self.calls = []
        self._response = response

    def post_json(self, url, payload, headers=None):
        self.calls.append({"url": url, "payload": payload, "headers": headers or {}})
        return self._response

    def post_json_sse(self, url, payload, headers=None):
        return self.post_json(url, payload, headers=headers)


class TestAnthropicApiStyle(unittest.TestCase):
    def _anthropic_response(self, text="Test response."):
        return {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": "claude-opus-4-6",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

    def test_endpoint_is_v1_messages(self):
        """Requests must go to /v1/messages, not /v1/chat/completions."""
        fake = _FakeHttp(self._anthropic_response())
        client = _make_anthropic_client(fake)
        client.text(system="sys", user="hello", route="default", max_tokens=100)
        self.assertTrue(fake.calls[0]["url"].endswith("/v1/messages"))

    def test_auth_uses_x_api_key_not_bearer(self):
        """x-api-key header must be present; Authorization header must NOT be."""
        fake = _FakeHttp(self._anthropic_response())
        client = _make_anthropic_client(fake)
        client.text(system="sys", user="hello", route="default", max_tokens=100)
        hdrs = fake.calls[0]["headers"]
        self.assertIn("x-api-key", hdrs)
        self.assertEqual(hdrs["x-api-key"], "sk-ant-test")
        self.assertNotIn("Authorization", hdrs)

    def test_anthropic_version_header_present(self):
        """anthropic-version header must be set to 2023-06-01."""
        fake = _FakeHttp(self._anthropic_response())
        client = _make_anthropic_client(fake)
        client.text(system="sys", user="hello", route="default", max_tokens=100)
        hdrs = fake.calls[0]["headers"]
        self.assertIn("anthropic-version", hdrs)
        self.assertEqual(hdrs["anthropic-version"], "2023-06-01")

    def test_payload_has_system_as_top_level_key(self):
        """system prompt must be top-level 'system' field, not inside messages."""
        fake = _FakeHttp(self._anthropic_response())
        client = _make_anthropic_client(fake)
        client.text(system="Be a thesis writer.", user="hello", route="default", max_tokens=100)
        payload = fake.calls[0]["payload"]
        self.assertIn("system", payload)
        self.assertEqual(payload["system"], "Be a thesis writer.")
        # messages should only have user role
        for msg in payload.get("messages", []):
            self.assertNotEqual(msg.get("role"), "system")

    def test_payload_has_user_message(self):
        """messages array must have exactly one user message."""
        fake = _FakeHttp(self._anthropic_response())
        client = _make_anthropic_client(fake)
        client.text(system="sys", user="hello world", route="default", max_tokens=100)
        payload = fake.calls[0]["payload"]
        msgs = payload.get("messages", [])
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[0]["content"], "hello world")

    def test_max_tokens_in_payload(self):
        """max_tokens must be passed, not max_output_tokens."""
        fake = _FakeHttp(self._anthropic_response())
        client = _make_anthropic_client(fake)
        client.text(system="sys", user="hello", route="default", max_tokens=1234)
        payload = fake.calls[0]["payload"]
        self.assertEqual(payload.get("max_tokens"), 1234)
        self.assertNotIn("max_output_tokens", payload)

    def test_response_text_extracted_from_content_array(self):
        """_extract_llm_message must handle Anthropic content array format."""
        fake = _FakeHttp(self._anthropic_response("Thesis output text."))
        client = _make_anthropic_client(fake)
        result = client.text(system="sys", user="hello", route="default", max_tokens=100)
        self.assertEqual(result, "Thesis output text.")

    def test_no_response_format_in_payload(self):
        """Anthropic does not accept response_format — must not be in payload."""
        fake = _FakeHttp(self._anthropic_response('{"chapters": []}'))
        client = _make_anthropic_client(fake)
        client.structured(system="Return JSON.", user="topic", route="outline", max_tokens=500)
        payload = fake.calls[0]["payload"]
        self.assertNotIn("response_format", payload)

    def test_response_with_type_message_required(self):
        """Anthropic extraction requires type=message in response."""
        fake = _FakeHttp({
            "content": [{"type": "text", "text": "should not match"}],
            "usage": {"total_tokens": 1},
        })
        client = _make_anthropic_client(fake)
        # Without type=message, should fall through to other extractors (and fail)
        with self.assertRaises(ProviderError):
            client.text(system="sys", user="hello", route="default", max_tokens=100)

    def test_empty_content_array_raises_error(self):
        """Empty content array in Anthropic response should raise ProviderError."""
        fake = _FakeHttp({
            "type": "message",
            "content": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
        })
        client = _make_anthropic_client(fake)
        with self.assertRaises(ProviderError):
            client.text(system="sys", user="hello", route="default", max_tokens=100)

    def test_temperature_in_payload(self):
        """Temperature must be included in the Anthropic payload."""
        fake = _FakeHttp(self._anthropic_response())
        client = _make_anthropic_client(fake)
        client.text(system="sys", user="hello", route="default", max_tokens=100)
        payload = fake.calls[0]["payload"]
        self.assertIn("temperature", payload)

    def test_anthropic_does_not_force_sse_transport(self):
        """Anthropic providers must use JSON transport (not SSE streaming)."""

        class _NoSseHttp(_FakeHttp):
            def post_json_sse(self, url, payload, headers=None):
                raise ProviderError("SSE transport is not supported for Anthropic")

        fake = _NoSseHttp(self._anthropic_response())
        client = _make_anthropic_client(fake)
        client.text(system="sys", user="hello", route="default", max_tokens=100)
        self.assertEqual(len(fake.calls), 1)
