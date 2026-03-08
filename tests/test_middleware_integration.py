from __future__ import annotations

import os
import tempfile
import unittest

from muse.config import Settings


class _FakeSearch:
    def search_multi_source(self, topic, discipline, extra_queries=None):
        return (
            [
                {
                    "ref_id": "@a2024x",
                    "title": "X",
                    "authors": ["A"],
                    "year": 2024,
                    "doi": None,
                    "venue": "V",
                    "abstract": "...",
                    "source": "test",
                    "verified_metadata": True,
                }
            ],
            extra_queries or [topic],
        )


class _FakeLLM:
    def __init__(self):
        self.calls = 0

    def structured(self, *, system, user, route="default", max_tokens=2500):
        self.calls += 1
        if self.calls == 1:
            return {"queries": ["q1"]}
        if self.calls == 2:
            return {
                "research_gaps": [],
                "core_concepts": [],
                "methodology_domain": "cs",
                "suggested_contributions": [],
            }
        return {
            "chapters": [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "Intro",
                    "target_words": 2000,
                    "complexity": "low",
                    "subsections": [{"title": "Background"}],
                }
            ]
        }


class _FakeServices:
    def __init__(self):
        self.llm = _FakeLLM()
        self.search = _FakeSearch()
        self.local_refs = []
        self.rag_index = None


def _make_settings(tmp_dir: str, log_path: str | None = None) -> Settings:
    del log_path
    return Settings(
        llm_api_key="x",
        llm_base_url="http://localhost",
        llm_model="stub",
        model_router_config={},
        runs_dir=tmp_dir,
        semantic_scholar_api_key=None,
        openalex_email=None,
        crossref_mailto=None,
        refs_dir=None,
        checkpoint_dir=None,
    )


class MiddlewareIntegrationTests(unittest.TestCase):
    def test_build_graph_with_middleware_runs(self):
        from muse.graph.launcher import build_graph, invoke

        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(tmp)
            services = _FakeServices()
            graph = build_graph(settings, services=services, thread_id="mw-test")
            result = invoke(
                graph,
                {
                    "project_id": "mw-test",
                    "topic": "Middleware testing",
                    "discipline": "CS",
                    "language": "en",
                    "format_standard": "APA",
                    "output_format": "markdown",
                },
                thread_id="mw-test",
            )
            self.assertIn("references", result)
            self.assertIn("outline", result)

    def test_build_default_middleware_chain(self):
        from muse.middlewares import MiddlewareChain, build_default_chain

        chain = build_default_chain()
        self.assertIsInstance(chain, MiddlewareChain)

    def test_build_default_chain_with_log_path(self):
        from muse.middlewares import build_default_chain

        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "test.jsonl")
            del log_path
            chain = build_default_chain(log_dir=tmp)
            self.assertIsNotNone(chain)


if __name__ == "__main__":
    unittest.main()
