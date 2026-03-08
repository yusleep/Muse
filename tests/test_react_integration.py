"""Integration test: full pipeline with ReAct sub-graphs in fallback mode."""

from __future__ import annotations

import unittest


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

    def test_citation_subgraph_fallback_path(self):
        from muse.graph.subgraphs.citation import build_citation_subgraph_node

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


if __name__ == "__main__":
    unittest.main()
