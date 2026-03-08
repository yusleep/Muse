import tempfile
import unittest
from pathlib import Path


class LlamaIndexAdapterTests(unittest.TestCase):
    def test_retrieval_adapter_matches_protocol(self):
        from adapters.llamaindex.retriever import LlamaIndexRetrievalAdapter
        from muse.adapters import RetrievalService

        adapter = LlamaIndexRetrievalAdapter(
            documents=[
                {"ref_id": "@a", "title": "Graph Workflow", "abstract": "LangGraph graph workflow."},
                {"ref_id": "@b", "title": "Other", "abstract": "Unrelated."},
            ]
        )

        self.assertIsInstance(adapter, RetrievalService)
        results = adapter.retrieve("graph workflow", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["ref_id"], "@a")

    def test_ingestion_adapter_uses_existing_refs_loader(self):
        from adapters.llamaindex.ingestion import LlamaIndexIngestionAdapter

        with tempfile.TemporaryDirectory() as tmp:
            refs_dir = Path(tmp) / "refs"
            refs_dir.mkdir(parents=True, exist_ok=True)
            (refs_dir / "paper.txt").write_text("Graph workflow and thesis orchestration", encoding="utf-8")

            adapter = LlamaIndexIngestionAdapter()
            docs = adapter.load_directory(str(refs_dir))

            self.assertEqual(len(docs), 1)
            self.assertIn("Graph workflow", docs[0]["abstract"])

    def test_external_search_adapters_delegate_single_source_fetch(self):
        from adapters.external_search.arxiv import ArxivSearchAdapter
        from adapters.external_search.openalex import OpenAlexSearchAdapter
        from adapters.external_search.semantic_scholar import SemanticScholarSearchAdapter

        class _Client:
            def search_semantic_scholar(self, query, limit=20):
                return [{"source": "semantic_scholar", "query": query, "limit": limit}]

            def search_openalex(self, query, limit=20):
                return [{"source": "openalex", "query": query, "limit": limit}]

            def search_arxiv(self, query, limit=20):
                return [{"source": "arxiv", "query": query, "limit": limit}]

        client = _Client()
        self.assertEqual(SemanticScholarSearchAdapter(client).search("graph", limit=3)[0]["source"], "semantic_scholar")
        self.assertEqual(OpenAlexSearchAdapter(client).search("graph", limit=3)[0]["source"], "openalex")
        self.assertEqual(ArxivSearchAdapter(client).search("graph", limit=3)[0]["source"], "arxiv")


if __name__ == "__main__":
    unittest.main()
