"""Integration tests for structured HITL flow."""

from __future__ import annotations

import tempfile
import unittest

from muse.config import Settings


class _StubSearch:
    def search_multi_source(self, topic, discipline, extra_queries=None):
        return (
            [
                {
                    "ref_id": "@test2024",
                    "title": "Test",
                    "authors": ["A"],
                    "year": 2024,
                    "doi": "10.1/t",
                    "venue": "V",
                    "abstract": "A.",
                    "source": "stub",
                    "verified_metadata": True,
                }
            ],
            extra_queries or [topic],
        )


class _StubLLM:
    def structured(self, *, system, user, route="default", max_tokens=2500):
        if "search queries" in system.lower() or "generate 7" in system.lower():
            return {"queries": ["test query"]}
        if "analyze this research topic" in system.lower():
            return {
                "research_gaps": ["g"],
                "core_concepts": ["c"],
                "methodology_domain": "cs",
                "suggested_contributions": ["s"],
            }
        if "generate a thesis outline" in system.lower():
            return {
                "chapters": [
                    {
                        "chapter_id": "ch_01",
                        "chapter_title": "Intro",
                        "target_words": 500,
                        "complexity": "low",
                        "subsections": [{"title": "Background"}],
                    }
                ]
            }
        return {}


class _StubServices:
    def __init__(self):
        self.llm = _StubLLM()
        self.search = _StubSearch()
        self.local_refs = []
        self.rag_index = None


class StructuredHitlIntegrationTests(unittest.TestCase):
    def _make_graph(self, tmp_dir: str):
        from muse.graph.launcher import build_graph

        settings = Settings(
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
        return build_graph(
            settings,
            services=_StubServices(),
            thread_id="hitl-int",
            auto_approve=False,
        )

    def test_top_level_interrupt_has_structured_payload(self):
        from muse.graph.launcher import invoke

        with tempfile.TemporaryDirectory() as tmp_dir:
            graph = self._make_graph(tmp_dir)
            result = invoke(
                graph,
                {
                    "project_id": "hitl-int",
                    "topic": "Test",
                    "discipline": "cs",
                    "language": "zh",
                    "format_standard": "GB/T 7714-2015",
                    "output_format": "markdown",
                },
                thread_id="hitl-int",
            )

        payload = result["__interrupt__"][0].value
        self.assertIn("question", payload)
        self.assertIn("options", payload)
        self.assertIn("context", payload)

    def test_resume_with_option_label(self):
        from muse.graph.launcher import invoke

        with tempfile.TemporaryDirectory() as tmp_dir:
            graph = self._make_graph(tmp_dir)
            first = invoke(
                graph,
                {
                    "project_id": "hitl-int",
                    "topic": "Test",
                    "discipline": "cs",
                    "language": "zh",
                    "format_standard": "GB/T 7714-2015",
                    "output_format": "markdown",
                },
                thread_id="hitl-int",
            )
            self.assertIn("__interrupt__", first)

            resumed = invoke(
                graph,
                None,
                thread_id="hitl-int",
                resume={"stage": "research", "approved": True, "option": "continue"},
            )

        self.assertIn("__interrupt__", resumed)
        self.assertEqual(resumed["__interrupt__"][0].value["stage"], "outline")

    def test_resume_with_freetext_comment(self):
        from muse.graph.launcher import invoke

        with tempfile.TemporaryDirectory() as tmp_dir:
            graph = self._make_graph(tmp_dir)
            first = invoke(
                graph,
                {
                    "project_id": "hitl-int",
                    "topic": "Test",
                    "discipline": "cs",
                    "language": "zh",
                    "format_standard": "GB/T 7714-2015",
                    "output_format": "markdown",
                },
                thread_id="hitl-int",
            )
            self.assertIn("__interrupt__", first)

            resumed = invoke(
                graph,
                None,
                thread_id="hitl-int",
                resume={
                    "stage": "research",
                    "approved": True,
                    "comment": "Looks good, continue.",
                },
            )

        self.assertIn("__interrupt__", resumed)
        self.assertEqual(resumed["__interrupt__"][0].value["stage"], "outline")

    def test_backward_compat_bool_resume(self):
        from muse.graph.launcher import invoke

        with tempfile.TemporaryDirectory() as tmp_dir:
            graph = self._make_graph(tmp_dir)
            first = invoke(
                graph,
                {
                    "project_id": "hitl-int",
                    "topic": "Test",
                    "discipline": "cs",
                    "language": "zh",
                    "format_standard": "GB/T 7714-2015",
                    "output_format": "markdown",
                },
                thread_id="hitl-int",
            )
            self.assertIn("__interrupt__", first)

            resumed = invoke(
                graph,
                None,
                thread_id="hitl-int",
                resume=True,
            )

        self.assertTrue(isinstance(resumed, dict))


if __name__ == "__main__":
    unittest.main()
