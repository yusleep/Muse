"""Tests for the academic_search LangChain tool."""

from __future__ import annotations

import unittest
from typing import Any


class _FakeSearchClient:
    """Stub matching AcademicSearchClient interface."""

    def __init__(self):
        self.last_query = None
        self.last_discipline = None

    def search_multi_source(
        self, topic: str, discipline: str, extra_queries: list[str] | None = None
    ) -> tuple[list[dict[str, Any]], list[str]]:
        self.last_query = topic
        self.last_discipline = discipline
        return (
            [
                {
                    "ref_id": "@smith2024graph",
                    "title": "Graph Neural Networks Survey",
                    "authors": ["Alice Smith"],
                    "year": 2024,
                    "doi": "10.1000/gnn",
                    "venue": "NeurIPS",
                    "abstract": "A survey of GNN methods.",
                    "source": "semantic_scholar",
                    "verified_metadata": True,
                }
            ],
            extra_queries or [topic],
        )


class AcademicSearchToolTests(unittest.TestCase):
    def test_tool_returns_string(self):
        from muse.tools.academic_search import make_academic_search_tool

        client = _FakeSearchClient()
        tool = make_academic_search_tool(client)
        result = tool.invoke({"query": "graph neural networks"})
        self.assertIsInstance(result, str)
        self.assertIn("Graph Neural Networks Survey", result)

    def test_tool_has_correct_name(self):
        from muse.tools.academic_search import make_academic_search_tool

        client = _FakeSearchClient()
        tool = make_academic_search_tool(client)
        self.assertEqual(tool.name, "academic_search")

    def test_tool_has_description(self):
        from muse.tools.academic_search import make_academic_search_tool

        client = _FakeSearchClient()
        tool = make_academic_search_tool(client)
        self.assertTrue(len(tool.description) > 10)

    def test_tool_passes_query_to_client(self):
        from muse.tools.academic_search import make_academic_search_tool

        client = _FakeSearchClient()
        tool = make_academic_search_tool(client)
        tool.invoke({"query": "transformer architectures"})
        self.assertEqual(client.last_query, "transformer architectures")

    def test_tool_with_discipline(self):
        from muse.tools.academic_search import make_academic_search_tool

        client = _FakeSearchClient()
        tool = make_academic_search_tool(client, default_discipline="Computer Science")
        tool.invoke({"query": "attention mechanisms"})
        self.assertEqual(client.last_discipline, "Computer Science")

    def test_tool_handles_empty_results(self):
        from muse.tools.academic_search import make_academic_search_tool

        class _EmptySearch:
            def search_multi_source(self, topic, discipline, extra_queries=None):
                return ([], [topic])

        tool = make_academic_search_tool(_EmptySearch())
        result = tool.invoke({"query": "obscure topic"})
        self.assertIsInstance(result, str)
        self.assertIn("No papers found", result)

    def test_tool_is_langchain_base_tool(self):
        from langchain_core.tools import BaseTool
        from muse.tools.academic_search import make_academic_search_tool

        tool = make_academic_search_tool(_FakeSearchClient())
        self.assertIsInstance(tool, BaseTool)


if __name__ == "__main__":
    unittest.main()
