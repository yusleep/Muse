"""Tests for muse/tools/research.py"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace


class ResearchToolTests(unittest.TestCase):
    def test_academic_search_returns_json_list(self):
        from muse.tools.research import academic_search

        result_str = academic_search.func(
            query="graph neural networks",
            max_results=3,
            runtime=None,
        )
        result = json.loads(result_str)
        self.assertIsInstance(result, list)

    def test_retrieve_local_refs_without_index(self):
        from muse.tools.research import retrieve_local_refs

        result_str = retrieve_local_refs.func(
            query="transformer architecture",
            top_k=5,
            runtime=None,
        )
        result = json.loads(result_str)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_retrieve_local_refs_falls_back_to_state_references(self):
        from muse.tools._context import clear_state, set_state
        from muse.tools.research import retrieve_local_refs

        set_state(
            {
                "references": [
                    {
                        "ref_id": "@lamport1982",
                        "title": "The Byzantine Generals Problem",
                        "authors": ["Leslie Lamport"],
                        "year": 1982,
                        "abstract": "Consensus in the presence of Byzantine faults.",
                        "source": "state",
                    }
                ]
            }
        )
        try:
            result_str = retrieve_local_refs.func(
                query="Byzantine fault tolerance consensus distributed systems",
                top_k=5,
                runtime=None,
            )
        finally:
            clear_state()

        result = json.loads(result_str)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["ref_id"], "@lamport1982")

    def test_retrieve_local_refs_prefers_paper_index_when_ready(self):
        from muse.tools.research import retrieve_local_refs

        class _FailingRagIndex:
            def retrieve(self, query, top_k=5):
                raise AssertionError("rag_index should not be used when paper index is ready")

        class _PaperIndex:
            def query(self, text, top_k=5):
                return [
                    {
                        "ref_id": "@indexed",
                        "section_title": "Results",
                        "text": "Indexed full-text result.",
                    }
                ]

        runtime = SimpleNamespace(
            context=SimpleNamespace(
                services=SimpleNamespace(
                    paper_index=_PaperIndex(),
                    rag_index=_FailingRagIndex(),
                )
            ),
            state={"paper_index_ready": True},
        )

        result = json.loads(
            retrieve_local_refs.func(
                query="durability results",
                top_k=5,
                runtime=runtime,
            )
        )

        self.assertEqual(result[0]["ref_id"], "@indexed")

    def test_get_paper_section_returns_json_when_section_exists(self):
        from muse.tools.research import get_paper_section

        class _PaperIndex:
            def get_section(self, paper_id, section_title, query, top_k=10):
                return [
                    {
                        "paper_id": paper_id,
                        "section_title": section_title,
                        "text": f"{query} details",
                    }
                ]

        runtime = SimpleNamespace(
            context=SimpleNamespace(
                services=SimpleNamespace(paper_index=_PaperIndex())
            )
        )

        result = json.loads(
            get_paper_section.func(
                paper_id="paper-1",
                section_title="Results",
                query="durability",
                runtime=runtime,
            )
        )

        self.assertEqual(result[0]["section_title"], "Results")

    def test_web_search_returns_stub_without_provider(self):
        from muse.tools.research import web_search

        result = web_search.func(query="LangGraph documentation", runtime=None)
        self.assertIsInstance(result, str)
        self.assertIn("No web search provider configured", result)

    def test_web_search_uses_runtime_provider_when_available(self):
        from muse.tools.research import web_search

        calls = []

        class _WebSearchClient:
            def search(self, query):
                calls.append(query)
                return [{"title": "LangGraph Docs", "url": "https://example.com/langgraph"}]

        runtime = SimpleNamespace(
            context=SimpleNamespace(
                services=SimpleNamespace(web_search_client=_WebSearchClient())
            )
        )

        result = web_search.func(query="LangGraph documentation", runtime=runtime)

        self.assertEqual(calls, ["LangGraph documentation"])
        self.assertEqual(json.loads(result)[0]["title"], "LangGraph Docs")

    def test_web_fetch_returns_string(self):
        from muse.tools.research import web_fetch

        result = web_fetch.invoke({"url": "https://example.com", "prompt": "summarize"})
        self.assertIsInstance(result, str)

    def test_read_pdf_returns_string(self):
        from muse.tools.research import read_pdf

        result = read_pdf.invoke({"file_path": "/nonexistent/file.pdf", "pages": "1-3"})
        self.assertIn("error", result.lower())

    def test_image_search_returns_string(self):
        from muse.tools.research import image_search

        result = image_search.invoke({"query": "neural network architecture diagram"})
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
