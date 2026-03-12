import tempfile
import unittest
from unittest.mock import patch

from muse.config import Settings


class _InterruptSearch:
    def search_multi_source(self, topic, discipline, extra_queries=None):
        return (
            [
                {
                    "ref_id": "@smith2024graph",
                    "title": "Graph Systems",
                    "authors": ["Alice Smith"],
                    "year": 2024,
                    "doi": "10.1000/graph",
                    "venue": "GraphConf",
                    "abstract": "Graph-native thesis workflow.",
                    "source": "semantic_scholar",
                    "verified_metadata": True,
                }
            ],
            extra_queries or [topic],
        )


class _InterruptLLM:
    def structured(self, *, system, user, route="default", max_tokens=2500):
        if "Generate 7 diverse English academic search queries" in system:
            return {"queries": ["graph workflow"]}
        if "Analyze this research topic" in system:
            return {
                "research_gaps": ["durability"],
                "core_concepts": ["langgraph"],
                "methodology_domain": "systems",
                "suggested_contributions": ["checkpointed writing flow"],
            }
        if "Generate a thesis outline" in system:
            return {
                "chapters": [
                    {
                        "chapter_id": "ch_01",
                        "chapter_title": "绪论",
                        "target_words": 1200,
                        "complexity": "low",
                        "subsections": [{"title": "研究背景"}],
                    }
                ]
            }
        raise AssertionError(f"unexpected prompt: {system}")


class _Services:
    def __init__(self):
        self.llm = _InterruptLLM()
        self.search = _InterruptSearch()
        self.local_refs = []
        self.rag_index = None


class HitlInterruptTests(unittest.TestCase):
    def test_interrupt_payload_contains_structured_fields(self):
        from muse.graph.nodes.review import build_interrupt_node

        node = build_interrupt_node("research", auto_approve=False)
        with patch("muse.graph.nodes.review.interrupt", side_effect=lambda payload: payload):
            result = node(
                {
                    "project_id": "run-structured",
                    "references": [{"ref_id": "@a"}],
                    "search_queries": ["graph workflow"],
                }
            )

        payload = result["review_feedback"][0]
        self.assertIsInstance(payload["question"], str)
        self.assertIsInstance(payload["clarification_type"], str)
        self.assertIsInstance(payload["options"], list)

    def test_interrupt_backward_compat_bool_resume(self):
        from muse.graph.nodes.review import build_interrupt_node

        node = build_interrupt_node("outline", auto_approve=False)
        with patch("muse.graph.nodes.review.interrupt", return_value=True):
            result = node({"project_id": "run-bool", "chapter_plans": []})

        feedback = result["review_feedback"][0]
        self.assertEqual(feedback["stage"], "outline")
        self.assertTrue(feedback["approved"])

    def test_interrupt_payload_stage_specific_options(self):
        from muse.graph.nodes.review import build_interrupt_node

        node = build_interrupt_node("research", auto_approve=False)
        with patch("muse.graph.nodes.review.interrupt", side_effect=lambda payload: payload):
            result = node(
                {
                    "project_id": "run-options",
                    "references": [{"ref_id": "@a"}],
                    "search_queries": ["graph workflow"],
                }
            )

        labels = {option["label"] for option in result["review_feedback"][0]["options"]}
        self.assertEqual(labels, {"continue", "add_keywords", "add_manually"})

    def test_graph_interrupts_after_search_and_resume_reaches_outline_interrupt(self):
        from muse.graph.launcher import build_graph, invoke

        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                llm_api_key="x",
                llm_base_url="http://localhost",
                llm_model="stub",
                model_router_config={},
                runs_dir=tmp,
                semantic_scholar_api_key=None,
                openalex_email=None,
                crossref_mailto=None,
                refs_dir=None,
                checkpoint_dir=None,
            )
            graph = build_graph(settings, services=_Services(), thread_id="run-3", auto_approve=False)

            first = invoke(
                graph,
                {
                    "project_id": "run-3",
                    "topic": "LangGraph thesis automation",
                    "discipline": "Computer Science",
                    "language": "zh",
                    "format_standard": "GB/T 7714-2015",
                    "output_format": "markdown",
                },
                thread_id="run-3",
            )
            self.assertIn("__interrupt__", first)
            self.assertEqual(first["__interrupt__"][0].value["stage"], "research")

            resumed = invoke(
                graph,
                None,
                thread_id="run-3",
                resume={"stage": "research", "approved": True},
            )
            self.assertIn("__interrupt__", resumed)
            self.assertEqual(resumed["__interrupt__"][0].value["stage"], "outline")


if __name__ == "__main__":
    unittest.main()
