"""Tests for _build_request_payload with tools parameter."""

from __future__ import annotations

import unittest

from muse.services.providers import _ModelAttempt, _build_request_payload


def _make_attempt(api_style: str, endpoint: str = "http://localhost/v1/chat/completions") -> _ModelAttempt:
    return _ModelAttempt(
        route_name="default",
        model_id="test/model",
        provider_name="test",
        endpoint_url=endpoint,
        api_style=api_style,
        model_name="model",
        header_candidates=[{}],
        params={},
        requires_streaming=False,
    )


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "academic_search",
            "description": "Search for academic papers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    }
]


class PayloadToolsTests(unittest.TestCase):
    def test_openai_chat_completions_includes_tools(self):
        attempt = _make_attempt("chat_completions")
        payload = _build_request_payload(
            attempt=attempt,
            system="sys",
            user="usr",
            temperature=0.2,
            response_format=None,
            max_tokens=100,
            tools=SAMPLE_TOOLS,
        )
        self.assertIn("tools", payload)
        self.assertEqual(payload["tools"], SAMPLE_TOOLS)
        self.assertEqual(payload["tool_choice"], "auto")

    def test_openai_chat_completions_no_tools_key_when_none(self):
        attempt = _make_attempt("chat_completions")
        payload = _build_request_payload(
            attempt=attempt,
            system="sys",
            user="usr",
            temperature=0.2,
            response_format=None,
            max_tokens=100,
        )
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)

    def test_anthropic_includes_tools_in_anthropic_format(self):
        attempt = _make_attempt("anthropic", "http://localhost/v1/messages")
        payload = _build_request_payload(
            attempt=attempt,
            system="sys",
            user="usr",
            temperature=0.2,
            response_format=None,
            max_tokens=100,
            tools=SAMPLE_TOOLS,
        )
        self.assertIn("tools", payload)
        tool_entry = payload["tools"][0]
        self.assertIn("name", tool_entry)
        self.assertIn("input_schema", tool_entry)
        self.assertEqual(tool_entry["name"], "academic_search")

    def test_responses_includes_tools_as_function_tools(self):
        attempt = _make_attempt("responses", "http://localhost/v1/responses")
        payload = _build_request_payload(
            attempt=attempt,
            system="sys",
            user="usr",
            temperature=0.2,
            response_format=None,
            max_tokens=100,
            tools=SAMPLE_TOOLS,
        )
        self.assertIn("tools", payload)
        tool_entry = payload["tools"][0]
        self.assertEqual(tool_entry["type"], "function")
        self.assertIn("name", tool_entry)

    def test_codex_streaming_omits_tools(self):
        attempt = _ModelAttempt(
            route_name="default",
            model_id="codex/o4-mini",
            provider_name="codex",
            endpoint_url="https://chatgpt.com/backend-api/codex/responses",
            api_style="responses",
            model_name="o4-mini",
            header_candidates=[{}],
            params={},
            requires_streaming=True,
        )
        payload = _build_request_payload(
            attempt=attempt,
            system="sys",
            user="usr",
            temperature=0.2,
            response_format=None,
            max_tokens=100,
            streaming=True,
            tools=SAMPLE_TOOLS,
        )
        self.assertNotIn("tools", payload)

    def test_tool_choice_can_be_overridden(self):
        attempt = _make_attempt("chat_completions")
        payload = _build_request_payload(
            attempt=attempt,
            system="sys",
            user="usr",
            temperature=0.2,
            response_format=None,
            max_tokens=100,
            tools=SAMPLE_TOOLS,
            tool_choice="required",
        )
        self.assertEqual(payload["tool_choice"], "required")


if __name__ == "__main__":
    unittest.main()
