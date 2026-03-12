import os
import tempfile
import unittest
from unittest.mock import patch

from muse.config import Settings
from muse.graph.launcher import build_graph, invoke
from muse.runtime import Runtime


class _RuntimeSearch:
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


class _RuntimeMetadata:
    def verify_doi(self, doi):
        return True

    def crosscheck_metadata(self, ref):
        return True


class _RuntimeLLM:
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
        if "Write one thesis subsection with citations" in system:
            return {
                "text": "Drafted subsection content with citation.",
                "citations_used": ["@smith2024graph"],
                "key_claims": ["Graph orchestration improves durability."],
                "transition_out": "",
                "glossary_additions": {},
                "self_assessment": {"confidence": 0.9, "weak_spots": [], "needs_revision": False},
            }
        if "strict thesis reviewer" in system:
            return {
                "scores": {
                    "coherence": 4,
                    "logic": 4,
                    "citation": 4,
                    "term_consistency": 4,
                    "balance": 4,
                    "redundancy": 4,
                },
                "review_notes": [],
            }
        if "Polish the academic thesis chapter" in system:
            return {"final_text": "Polished chapter text.", "polish_notes": ["统一术语。"]}
        if "你是一位学术论文摘要撰写专家" in system:
            return {"abstract": "中文摘要", "keywords": ["图工作流"]}
        if "You are an academic abstract writer" in system:
            return {"abstract": "English abstract", "keywords": ["graph workflow"]}
        raise AssertionError(f"unexpected prompt: {system}")

    def entailment(self, *, premise, hypothesis, route="reasoning"):
        return "entailment"


class _Services:
    def __init__(self):
        self.llm = _RuntimeLLM()
        self.search = _RuntimeSearch()
        self.metadata = _RuntimeMetadata()
        self.local_refs = []
        self.rag_index = None


class _FinalizingCitationReactAgent:
    def invoke(self, agent_input, config):
        del config
        from muse.tools.citation import finalize_citation_review, record_citation_assessment

        for item in agent_input.get("citation_worklist", []):
            record_citation_assessment.invoke(
                {
                    "cite_key": item["cite_key"],
                    "claim_id": item["claim_id"],
                    "verdict": "verified",
                    "support_score": 0.95,
                    "confidence": "high",
                    "reason": "supported",
                    "detail": "Runtime flow citation verified.",
                    "evidence_excerpt": item.get("evidence", ""),
                }
            )
        finalize_citation_review.invoke({"summary": "runtime flow citation review complete"})
        return {"messages": []}


class RuntimeFlowTests(unittest.TestCase):
    def _make_settings(self, runs_dir: str) -> Settings:
        return Settings(
            llm_api_key="x",
            llm_base_url="http://localhost",
            llm_model="stub",
            model_router_config={},
            runs_dir=runs_dir,
            semantic_scholar_api_key=None,
            openalex_email=None,
            crossref_mailto=None,
            refs_dir=None,
            checkpoint_dir=None,
        )

    def test_graph_interrupt_waits_for_hitl_when_auto_approve_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = build_graph(self._make_settings(tmp), services=_Services(), thread_id="flow-hitl", auto_approve=False)

            result = invoke(
                graph,
                {
                    "project_id": "flow-hitl",
                    "topic": "LangGraph thesis automation",
                    "discipline": "Computer Science",
                    "language": "zh",
                    "format_standard": "GB/T 7714-2015",
                    "output_format": "markdown",
                },
                thread_id="flow-hitl",
            )

            self.assertIn("__interrupt__", result)
            self.assertEqual(result["__interrupt__"][0].value["stage"], "research")

    def test_graph_auto_approve_runs_to_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "muse.graph.subgraphs.citation._build_react_citation_agent",
                return_value=_FinalizingCitationReactAgent(),
            ):
                graph = build_graph(
                    self._make_settings(tmp),
                    services=_Services(),
                    thread_id="flow-done",
                    auto_approve=True,
                )

                result = invoke(
                    graph,
                    {
                        "project_id": "flow-done",
                        "topic": "LangGraph thesis automation",
                        "discipline": "Computer Science",
                        "language": "zh",
                        "format_standard": "GB/T 7714-2015",
                        "output_format": "markdown",
                    },
                    thread_id="flow-done",
                )

            self.assertNotIn("__interrupt__", result)
            self.assertEqual(result["verified_citations"], ["@smith2024graph"])
            self.assertEqual(result["flagged_citations"], [])
            self.assertTrue(os.path.isfile(result["output_filepath"]))

    def test_runtime_exposes_graph_builder(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Runtime(self._make_settings(tmp))
            graph = runtime.build_graph(thread_id="topic", auto_approve=True)
            self.assertTrue(hasattr(graph, "invoke"))


if __name__ == "__main__":
    unittest.main()
