"""Tests for MuseChatModel adapter wrapping LLMClient."""

from __future__ import annotations

import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class _StubHttp:
    """Minimal HttpClient stub for LLMClient."""

    def post_json(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        return {
            "choices": [
                {"message": {"content": "stub response"}},
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    def post_json_sse(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        return self.post_json(url, payload, headers)

    def get_json(self, url: str, headers: dict | None = None) -> dict:
        return {}


class MuseChatModelTests(unittest.TestCase):
    def _make_model(self, route: str = "default"):
        from muse.models.adapter import MuseChatModel
        from muse.services.providers import LLMClient

        llm_client = LLMClient(
            api_key="test-key",
            base_url="http://localhost:11434/v1",
            model="test-model",
            http=_StubHttp(),
        )
        return MuseChatModel(llm_client=llm_client, route=route)

    def test_is_base_chat_model(self):
        from langchain_core.language_models.chat_models import BaseChatModel

        model = self._make_model()
        self.assertIsInstance(model, BaseChatModel)

    def test_invoke_returns_ai_message(self):
        model = self._make_model()
        result = model.invoke([HumanMessage(content="Hello")])
        self.assertIsInstance(result, AIMessage)
        self.assertEqual(result.content, "stub response")

    def test_system_message_extracted(self):
        model = self._make_model()
        result = model.invoke(
            [
                SystemMessage(content="You are helpful."),
                HumanMessage(content="Hi"),
            ]
        )
        self.assertIsInstance(result, AIMessage)
        self.assertEqual(result.content, "stub response")

    def test_llm_type_property(self):
        model = self._make_model()
        self.assertEqual(model._llm_type, "muse-chat-model")

    def test_multiple_human_messages_concatenated(self):
        model = self._make_model()
        result = model.invoke(
            [
                HumanMessage(content="First part."),
                HumanMessage(content="Second part."),
            ]
        )
        self.assertIsInstance(result, AIMessage)

    def test_bind_tools_returns_runnable(self):
        from langchain_core.tools import tool

        @tool
        def dummy_tool(query: str) -> str:
            """Search for papers."""

            return "result"

        model = self._make_model()
        bound = model.bind_tools([dummy_tool])
        self.assertIsNotNone(bound)

    def test_tool_call_response_parsed(self):
        from muse.models.adapter import MuseChatModel
        from muse.services.providers import LLMClient

        class _ToolCallHttp:
            def post_json(self, url, payload, headers=None):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "academic_search",
                                            "arguments": json.dumps({"query": "graph neural networks"}),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {},
                }

            def post_json_sse(self, url, payload, headers=None):
                return self.post_json(url, payload, headers)

            def get_json(self, url, headers=None):
                return {}

        llm_client = LLMClient(
            api_key="k",
            base_url="http://localhost/v1",
            model="m",
            http=_ToolCallHttp(),
        )
        model = MuseChatModel(llm_client=llm_client, route="default")
        result = model.invoke([HumanMessage(content="search for papers")])
        self.assertIsInstance(result, AIMessage)
        self.assertTrue(len(result.tool_calls) > 0)
        self.assertEqual(result.tool_calls[0]["name"], "academic_search")


if __name__ == "__main__":
    unittest.main()
