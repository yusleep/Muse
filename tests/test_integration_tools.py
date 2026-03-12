"""Integration tests: MuseChatModel + ToolRegistry + tools end-to-end."""

from __future__ import annotations

import json
import unittest
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


class _ToolCallingHttp:
    """HTTP stub that captures payloads and returns tool_calls or text responses."""

    def __init__(self):
        self.call_count = 0
        self.captured_payloads: list[dict[str, Any]] = []

    def post_json(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        self.call_count += 1
        self.captured_payloads.append(payload)

        if self.call_count == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc123",
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
                "usage": {"prompt_tokens": 50, "completion_tokens": 20},
            }
        return {
            "choices": [
                {
                    "message": {"content": "Based on the search results, GNNs are..."}
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

    def post_json_sse(self, url, payload, headers=None):
        return self.post_json(url, payload, headers)

    def get_json(self, url, headers=None):
        return {}


class _FakeSearchClient:
    def search_multi_source(self, topic, discipline, extra_queries=None):
        return (
            [
                {
                    "ref_id": "@smith2024gnn",
                    "title": "Graph Neural Networks",
                    "authors": ["Alice Smith"],
                    "year": 2024,
                    "doi": "10.1000/gnn",
                    "venue": "NeurIPS",
                    "abstract": "A survey.",
                    "source": "semantic_scholar",
                    "verified_metadata": True,
                }
            ],
            [topic],
        )


class _FakeMetadataClient:
    def verify_doi(self, doi):
        return True

    def crosscheck_metadata(self, ref):
        return True


class IntegrationToolsTests(unittest.TestCase):
    def test_registry_assembles_profile_and_binds_to_model(self):
        from muse.models.adapter import MuseChatModel
        from muse.services.providers import LLMClient
        from muse.tools.academic_search import make_academic_search_tool
        from muse.tools.citation import make_crosscheck_metadata_tool, make_verify_doi_tool
        from muse.tools.registry import ToolRegistry

        http = _ToolCallingHttp()
        llm_client = LLMClient(
            api_key="key",
            base_url="http://localhost/v1",
            model="test",
            http=http,
        )
        model = MuseChatModel(llm_client=llm_client, route="default")

        registry = ToolRegistry()
        registry.register(
            make_academic_search_tool(_FakeSearchClient(), default_discipline="CS"),
            group="research",
        )
        registry.register(make_verify_doi_tool(_FakeMetadataClient()), group="review")
        registry.register(make_crosscheck_metadata_tool(_FakeMetadataClient()), group="review")
        registry.define_profile("chapter", groups=["research"])
        registry.define_profile("citation", groups=["review"])

        chapter_tools = registry.get_tools_for_profile("chapter")
        self.assertEqual(len(chapter_tools), 1)
        self.assertEqual(chapter_tools[0].name, "academic_search")

        bound_model = model.bind_tools(chapter_tools)
        self.assertIsNotNone(bound_model)

        result = bound_model.invoke(
            [
                SystemMessage(content="You are a research assistant."),
                HumanMessage(content="Find papers about graph neural networks"),
            ]
        )
        self.assertIsInstance(result, AIMessage)
        self.assertTrue(len(result.tool_calls) > 0)
        self.assertEqual(result.tool_calls[0]["name"], "academic_search")

    def test_tool_execution_and_follow_up(self):
        from muse.models.adapter import MuseChatModel
        from muse.services.providers import LLMClient
        from muse.tools.academic_search import make_academic_search_tool

        http = _ToolCallingHttp()
        llm_client = LLMClient(
            api_key="key",
            base_url="http://localhost/v1",
            model="test",
            http=http,
        )
        model = MuseChatModel(llm_client=llm_client, route="default")
        search_tool = make_academic_search_tool(_FakeSearchClient())

        result1 = model.invoke([HumanMessage(content="Search for GNN papers")])
        self.assertTrue(len(result1.tool_calls) > 0)
        tool_call = result1.tool_calls[0]

        tool_output = search_tool.invoke(tool_call["args"])
        self.assertIn("Graph Neural Networks", tool_output)

        result2 = model.invoke(
            [
                HumanMessage(content="Search for GNN papers"),
                result1,
                ToolMessage(content=tool_output, tool_call_id=tool_call["id"]),
            ]
        )
        self.assertIsInstance(result2, AIMessage)
        self.assertIn("GNNs", result2.content)

    def test_payload_includes_tools_when_bound(self):
        from muse.models.adapter import MuseChatModel
        from muse.services.providers import LLMClient
        from muse.tools.academic_search import make_academic_search_tool

        http = _ToolCallingHttp()
        llm_client = LLMClient(
            api_key="key",
            base_url="http://localhost/v1",
            model="test",
            http=http,
        )
        model = MuseChatModel(llm_client=llm_client, route="default")
        search_tool = make_academic_search_tool(_FakeSearchClient())

        bound = model.bind_tools([search_tool])
        bound.invoke([HumanMessage(content="Find papers")])

        self.assertTrue(len(http.captured_payloads) > 0)
        payload = http.captured_payloads[0]
        self.assertIn("tools", payload)
        self.assertEqual(payload["tools"][0]["function"]["name"], "academic_search")

    def test_citation_profile_tools(self):
        from muse.tools.citation import make_crosscheck_metadata_tool, make_verify_doi_tool
        from muse.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(make_verify_doi_tool(_FakeMetadataClient()), group="review")
        registry.register(make_crosscheck_metadata_tool(_FakeMetadataClient()), group="review")
        registry.define_profile("citation", groups=["review"])

        tools = registry.get_tools_for_profile("citation")
        names = {tool_item.name for tool_item in tools}
        self.assertEqual(names, {"verify_doi", "crosscheck_metadata"})

        if tools[0].name == "verify_doi":
            doi_result = tools[0].invoke({"doi": "10.1000/test"})
        else:
            doi_result = tools[1].invoke({"doi": "10.1000/test"})
        self.assertIsInstance(doi_result, str)


if __name__ == "__main__":
    unittest.main()
