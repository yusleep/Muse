from __future__ import annotations

import asyncio
import unittest


class DanglingToolCallMiddlewareTests(unittest.TestCase):
    def test_import(self):
        from muse.middlewares.dangling_tool_call import DanglingToolCallMiddleware

        self.assertTrue(callable(DanglingToolCallMiddleware))

    def test_conforms_to_protocol(self):
        from muse.middlewares.base import Middleware
        from muse.middlewares.dangling_tool_call import DanglingToolCallMiddleware

        middleware = DanglingToolCallMiddleware()
        self.assertIsInstance(middleware, Middleware)

    def test_before_invoke_passthrough(self):
        from muse.middlewares.dangling_tool_call import DanglingToolCallMiddleware

        middleware = DanglingToolCallMiddleware()
        state = {"x": 1}
        out = asyncio.run(middleware.before_invoke(state, {}))
        self.assertEqual(out, state)

    def test_no_tool_calls_passthrough(self):
        from muse.middlewares.dangling_tool_call import DanglingToolCallMiddleware

        middleware = DanglingToolCallMiddleware()
        result = {"text": "hello"}
        out = asyncio.run(middleware.after_invoke({}, result, {}))
        self.assertEqual(out, result)

    def test_complete_tool_calls_passthrough(self):
        from muse.middlewares.dangling_tool_call import DanglingToolCallMiddleware

        middleware = DanglingToolCallMiddleware()
        result = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "tc_1", "function": {"name": "search", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "tc_1", "content": "results"},
            ]
        }
        out = asyncio.run(middleware.after_invoke({}, result, {}))
        self.assertEqual(out, result)

    def test_dangling_tool_call_gets_error_response(self):
        from muse.middlewares.dangling_tool_call import DanglingToolCallMiddleware

        middleware = DanglingToolCallMiddleware()
        result = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "tc_1", "function": {"name": "search", "arguments": "{}"}},
                        {"id": "tc_2", "function": {"name": "write", "arguments": "{}"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            ]
        }
        out = asyncio.run(middleware.after_invoke({}, result, {}))
        messages = out["messages"]
        tool_responses = [message for message in messages if message.get("role") == "tool"]
        tc2_responses = [
            message for message in tool_responses if message.get("tool_call_id") == "tc_2"
        ]
        self.assertEqual(len(tc2_responses), 1)
        self.assertIn("error", tc2_responses[0]["content"].lower())

    def test_multiple_dangling_all_patched(self):
        from muse.middlewares.dangling_tool_call import DanglingToolCallMiddleware

        middleware = DanglingToolCallMiddleware()
        result = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "tc_a", "function": {"name": "f1", "arguments": "{}"}},
                        {"id": "tc_b", "function": {"name": "f2", "arguments": "{}"}},
                        {"id": "tc_c", "function": {"name": "f3", "arguments": "{}"}},
                    ],
                }
            ]
        }
        out = asyncio.run(middleware.after_invoke({}, result, {}))
        tool_responses = [
            message for message in out["messages"] if message.get("role") == "tool"
        ]
        self.assertEqual(len(tool_responses), 3)
        response_ids = {message["tool_call_id"] for message in tool_responses}
        self.assertEqual(response_ids, {"tc_a", "tc_b", "tc_c"})

    def test_no_messages_key_passthrough(self):
        from muse.middlewares.dangling_tool_call import DanglingToolCallMiddleware

        middleware = DanglingToolCallMiddleware()
        result = {"references": [{"title": "A"}]}
        out = asyncio.run(middleware.after_invoke({}, result, {}))
        self.assertEqual(out, result)

    def test_repair_message_content(self):
        from muse.middlewares.dangling_tool_call import DanglingToolCallMiddleware

        middleware = DanglingToolCallMiddleware()
        result = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "tc_x",
                            "function": {"name": "verify_doi", "arguments": "{}"},
                        }
                    ],
                }
            ]
        }
        out = asyncio.run(middleware.after_invoke({}, result, {}))
        tool_msg = [message for message in out["messages"] if message.get("role") == "tool"][0]
        self.assertIn("verify_doi", tool_msg["content"])


if __name__ == "__main__":
    unittest.main()
