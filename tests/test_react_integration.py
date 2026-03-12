"""Integration test: full pipeline with ReAct sub-graphs."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from muse.config import Settings


class _IntegrationLLM:
    """Minimal LLM stub for full-pipeline integration."""

    def __init__(self):
        self._review_calls = 0

    def structured(self, *, system, user, route="default", max_tokens=2500):
        if "search queries" in system.lower() or "topic analysis" in system.lower():
            return {"queries": ["test query"], "analysis": "test analysis"}
        if "outline" in system.lower():
            return {
                "chapter_plans": [
                    {
                        "chapter_id": "ch_01",
                        "chapter_title": "Introduction",
                        "subtask_plan": [
                            {
                                "subtask_id": "sub_01",
                                "title": "Background",
                                "target_words": 500,
                            }
                        ],
                    }
                ]
            }
        if "Write one thesis subsection" in system:
            return {
                "text": "Test subsection content.",
                "citations_used": [],
                "key_claims": ["Test claim."],
                "transition_out": "",
                "glossary_additions": {},
                "self_assessment": {
                    "confidence": 0.9,
                    "weak_spots": [],
                    "needs_revision": False,
                },
            }
        if "strict thesis reviewer" in system:
            self._review_calls += 1
            return {
                "scores": {
                    "coherence": 5,
                    "logic": 5,
                    "citation": 5,
                    "term_consistency": 5,
                    "balance": 5,
                    "redundancy": 5,
                },
                "review_notes": [],
            }
        if "polish" in system.lower():
            return {"text": "Polished text.", "notes": []}
        if "abstract" in system.lower():
            return {"abstract": "Test abstract."}
        return {}

    def text(self, *, system, user, route="default", max_tokens=2500):
        return "Generated text."

    def entailment(self, *, premise, hypothesis, route="reasoning"):
        return "entailment"


class _IntegrationMetadata:
    def verify_doi(self, doi):
        return True

    def crosscheck_metadata(self, ref):
        return True


class _IntegrationServices:
    def __init__(self):
        self.llm = _IntegrationLLM()
        self.metadata = _IntegrationMetadata()
        self.search = None
        self.rag_index = None
        self.local_refs = []


class _FinalizingCitationReactAgent:
    def invoke(self, agent_input, config):
        del config
        from muse.tools.citation import finalize_citation_review, record_citation_assessment

        worklist = agent_input.get("citation_worklist", [])
        for item in worklist:
            record_citation_assessment.invoke(
                {
                    "cite_key": item["cite_key"],
                    "claim_id": item["claim_id"],
                    "verdict": "verified",
                    "support_score": 0.95,
                    "confidence": "high",
                    "reason": "supported",
                    "detail": "Integration citation verified.",
                    "evidence_excerpt": item.get("evidence", ""),
                }
            )
        finalize_citation_review.invoke({"summary": "integration citation review complete"})
        return {"messages": []}


class FullPipelineIntegrationTest(unittest.TestCase):
    def test_main_graph_compiles(self):
        from muse.graph.main_graph import build_graph

        graph = build_graph(services=_IntegrationServices(), auto_approve=True)
        self.assertIsNotNone(graph)

    def test_chapter_subgraph_fallback_path(self):
        from muse.graph.subgraphs.chapter import build_chapter_subgraph_node

        node_fn = build_chapter_subgraph_node(services=_IntegrationServices())
        result = node_fn(
            {
                "chapter_plan": {
                    "chapter_id": "ch_01",
                    "chapter_title": "Introduction",
                    "subtask_plan": [
                        {
                            "subtask_id": "sub_01",
                            "title": "Background",
                            "target_words": 500,
                        }
                    ],
                },
                "references": [],
                "topic": "Test",
                "language": "zh",
                "subtask_results": [],
                "merged_text": "",
                "quality_scores": {},
                "review_notes": [],
                "revision_instructions": {},
                "iteration": 0,
                "max_iterations": 1,
                "citation_uses": [],
                "claim_text_by_id": {},
            }
        )
        self.assertIn("chapters", result)
        self.assertIn("ch_01", result["chapters"])
        self.assertIn("merged_text", result["chapters"]["ch_01"])

    def test_citation_subgraph_react_path(self):
        from muse.graph.subgraphs.citation import build_citation_subgraph_node

        with patch(
            "muse.graph.subgraphs.citation._build_react_citation_agent",
            return_value=_FinalizingCitationReactAgent(),
        ):
            node_fn = build_citation_subgraph_node(services=_IntegrationServices())
            result = node_fn(
                {
                    "references": [
                        {
                            "ref_id": "@a",
                            "title": "A",
                            "doi": "10.1/a",
                            "authors": ["X"],
                            "year": 2024,
                        }
                    ],
                    "citation_uses": [{"cite_key": "@a", "claim_id": "c1"}],
                    "claim_text_by_id": {"c1": "Test claim."},
                    "citation_ledger": {},
                    "verified_citations": [],
                    "flagged_citations": [],
                }
            )
        self.assertIn("citation_ledger", result)

    def test_composition_subgraph_fallback_path(self):
        from muse.graph.subgraphs.composition import build_composition_subgraph_node

        node_fn = build_composition_subgraph_node()
        result = node_fn(
            {
                "final_text": "Test text.",
                "abstract_zh": "摘要",
                "abstract_en": "Abstract",
                "paper_package": {},
                "language": "zh",
            }
        )
        self.assertIn("final_text", result)
        self.assertIn("paper_package", result)
        self.assertTrue(result["paper_package"].get("terminology_normalized"))

    def test_main_graph_passes_settings_to_react_subgraphs(self):
        from muse.graph.main_graph import build_graph

        settings = Settings(
            llm_api_key="x",
            llm_base_url="http://localhost",
            llm_model="stub",
            model_router_config={},
            runs_dir="runs",
            semantic_scholar_api_key=None,
            openalex_email=None,
            crossref_mailto=None,
            refs_dir=None,
            checkpoint_dir=None,
        )
        captured: dict[str, object] = {}

        def fake_chapter_node(*, services, settings=None):
            captured["chapter"] = settings
            return lambda state, config=None: state

        def fake_citation_node(*, services, settings=None):
            captured["citation"] = settings
            return lambda state, config=None: state

        def fake_composition_node(*, services=None, settings=None):
            captured["composition"] = settings
            return lambda state, config=None: state

        with patch("muse.graph.main_graph.build_chapter_subgraph_node", side_effect=fake_chapter_node), patch(
            "muse.graph.main_graph.build_citation_subgraph_node",
            side_effect=fake_citation_node,
        ), patch(
            "muse.graph.main_graph.build_composition_subgraph_node",
            side_effect=fake_composition_node,
        ):
            build_graph(settings=settings, services=_IntegrationServices(), auto_approve=True)

        self.assertIs(captured["chapter"], settings)
        self.assertIs(captured["citation"], settings)
        self.assertIs(captured["composition"], settings)


if __name__ == "__main__":
    unittest.main()
