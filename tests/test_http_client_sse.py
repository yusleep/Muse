from __future__ import annotations

import json
import unittest
from unittest.mock import patch


class _FakeSseResponse:
    def __init__(self, lines: list[bytes]):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self._lines)


class HttpClientSseTests(unittest.TestCase):
    def test_post_json_sse_parses_chat_completion_chunks(self):
        from muse.services.http import HttpClient

        events = [
            {"object": "chat.completion.chunk", "choices": [{"delta": {"content": "Hello"}}]},
            {
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": " world"}}],
                "usage": {"total_tokens": 1},
            },
        ]
        lines = [f"data: {json.dumps(evt)}\n".encode("utf-8") for evt in events]
        lines.append(b"data: [DONE]\n")

        def fake_urlopen(_req, timeout=0):  # noqa: ANN001
            del timeout
            return _FakeSseResponse(lines)

        with patch("urllib.request.urlopen", fake_urlopen):
            client = HttpClient(timeout_seconds=1)
            try:
                result = client.post_json_sse("http://example.com/v1/chat/completions", payload={"stream": True})
            except Exception as exc:  # pragma: no cover
                self.fail(f"expected SSE chunk parsing, but got error: {exc}")

        self.assertEqual(result["output_text"], "Hello world")
        self.assertEqual(result["usage"]["total_tokens"], 1)

    def test_post_json_sse_parses_tool_calls_without_text(self):
        from muse.services.http import HttpClient

        events = [
            {
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "academic_search", "arguments": "{\"query\":\"graph\""},
                                }
                            ],
                        }
                    }
                ],
            },
            {
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": " neural networks\"}"},
                                }
                            ]
                        }
                    }
                ],
                "usage": {"total_tokens": 1},
            },
            {
                "object": "chat.completion.chunk",
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
            },
        ]
        lines = [f"data: {json.dumps(evt)}\n".encode("utf-8") for evt in events]
        lines.append(b"data: [DONE]\n")

        def fake_urlopen(_req, timeout=0):  # noqa: ANN001
            del timeout
            return _FakeSseResponse(lines)

        with patch("urllib.request.urlopen", fake_urlopen):
            client = HttpClient(timeout_seconds=1)
            result = client.post_json_sse("http://example.com/v1/chat/completions", payload={"stream": True})

        tool_calls = result.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["function"]["name"], "academic_search")
        self.assertEqual(tool_calls[0]["function"]["arguments"], "{\"query\":\"graph\" neural networks\"}")
        self.assertEqual(result["usage"]["total_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
