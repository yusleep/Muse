"""Tests for ClarificationMiddleware."""

from __future__ import annotations

import unittest
from unittest.mock import patch


class ClarificationMiddlewareTests(unittest.TestCase):
    def _make_middleware(self):
        from muse.middlewares.clarification_middleware import ClarificationMiddleware

        return ClarificationMiddleware()

    def test_intercepts_ask_clarification_tool_call(self):
        middleware = self._make_middleware()
        tool_calls = [
            {
                "name": "ask_clarification",
                "id": "tc_1",
                "args": {
                    "question": "How many?",
                    "clarification_type": "missing_info",
                },
            }
        ]

        matched = middleware.should_intercept(tool_calls)
        self.assertIsNotNone(matched)
        self.assertEqual(matched["name"], "ask_clarification")

    def test_passes_through_non_clarification_tool_calls(self):
        middleware = self._make_middleware()
        tool_calls = [
            {"name": "write_section", "id": "tc_2", "args": {"text": "hello"}},
            {"name": "self_review", "id": "tc_3", "args": {}},
        ]

        matched = middleware.should_intercept(tool_calls)
        self.assertIsNone(matched)

    def test_interrupt_payload_structure(self):
        middleware = self._make_middleware()
        tool_call = {
            "name": "ask_clarification",
            "id": "tc_99",
            "args": {
                "question": "Which format?",
                "clarification_type": "approach_choice",
                "context": "Two citation formats are common",
                "options": [
                    {
                        "label": "APA",
                        "description": "American Psychological Association",
                    },
                    {"label": "GB/T", "description": "Chinese national standard"},
                ],
            },
        }

        payload = middleware.build_interrupt_payload(tool_call)
        self.assertEqual(payload["question"], "Which format?")
        self.assertEqual(payload["clarification_type"], "approach_choice")
        self.assertEqual(payload["context"], "Two citation formats are common")
        self.assertEqual(len(payload["options"]), 2)
        self.assertEqual(payload["tool_call_id"], "tc_99")
        self.assertEqual(payload["source"], "ask_clarification")

    def test_human_response_converted_to_tool_message(self):
        middleware = self._make_middleware()
        message = middleware.build_tool_message(
            tool_call_id="tc_99",
            human_response="Use GB/T 7714",
        )
        self.assertEqual(message["role"], "tool")
        self.assertEqual(message["tool_call_id"], "tc_99")
        self.assertEqual(message["content"], "Use GB/T 7714")

    def test_wrap_node_installs_runtime_handler_for_tool(self):
        from muse.tools.orchestration import ask_clarification

        middleware = self._make_middleware()

        def node_fn(state):
            del state
            return {
                "answer": ask_clarification.invoke(
                    {
                        "question": "How many chapters?",
                        "clarification_type": "missing_info",
                    }
                )
            }

        wrapped = middleware.wrap_node(node_fn)
        with patch.object(
            middleware,
            "fire_interrupt",
            return_value={"approved": True, "comment": "五章"},
        ):
            result = wrapped({})

        self.assertEqual(result["answer"], '{"approved": true, "comment": "五章"}')


if __name__ == "__main__":
    unittest.main()
