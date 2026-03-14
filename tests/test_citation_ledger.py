import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from muse.config import Settings


class _LedgerLLM:
    def entailment(self, *, premise, hypothesis, route="reasoning"):
        return "entailment"


class _LedgerMetadata:
    def verify_doi(self, doi):
        return True

    def crosscheck_metadata(self, ref):
        return True


class _LedgerServices:
    def __init__(self):
        self.llm = _LedgerLLM()
        self.metadata = _LedgerMetadata()


class _StrictMetadata:
    def __init__(self):
        self.doi_checks = []
        self.metadata_checks = []

    def verify_doi(self, doi):
        self.doi_checks.append(doi)
        return True

    def crosscheck_metadata(self, ref):
        self.metadata_checks.append(ref.get("ref_id"))
        return False


class CitationLedgerTests(unittest.TestCase):
    def test_citation_subgraph_populates_ledger(self):
        from muse.graph.subgraphs.citation import build_citation_graph

        graph = build_citation_graph(services=_LedgerServices())
        result = graph.invoke(
            {
                "references": [
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
                "citation_uses": [
                    {
                        "cite_key": "@smith2024graph",
                        "claim_id": "claim-1",
                        "chapter_id": "ch_01",
                        "subtask_id": "sub_01",
                    }
                ],
                "claim_text_by_id": {"claim-1": "Graph orchestration improves durability."},
                "citation_ledger": {},
                "verified_citations": [],
                "flagged_citations": [],
            }
        )

        self.assertIn("claim-1", result["citation_ledger"])
        self.assertEqual(result["citation_ledger"]["claim-1"]["repair_status"], "verified")
        self.assertEqual(result["verified_citations"], ["@smith2024graph"])
        self.assertEqual(result["flagged_citations"], [])

    def test_citation_subgraph_fuzzy_matches_natural_language_cite_key(self):
        from muse.graph.subgraphs.citation import build_citation_graph

        graph = build_citation_graph(services=_LedgerServices())
        result = graph.invoke(
            {
                "references": [
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
                "citation_uses": [
                    {
                        "cite_key": "Smith et al., 2024",
                        "claim_id": "claim-1",
                        "chapter_id": "ch_01",
                        "subtask_id": "sub_01",
                    }
                ],
                "claim_text_by_id": {"claim-1": "Graph orchestration improves durability."},
                "citation_ledger": {},
                "verified_citations": [],
                "flagged_citations": [],
            }
        )

        self.assertEqual(result["verified_citations"], ["Smith et al., 2024"])
        self.assertEqual(result["flagged_citations"], [])
        self.assertIn("Graph-native thesis workflow.", result["citation_ledger"]["claim-1"]["evidence_excerpt"])

    def test_citation_subgraph_requires_metadata_crosscheck_even_with_valid_doi(self):
        from muse.graph.subgraphs.citation import build_citation_graph

        metadata = _StrictMetadata()

        class _StrictServices:
            def __init__(self):
                self.llm = _LedgerLLM()
                self.metadata = metadata

        graph = build_citation_graph(services=_StrictServices())
        result = graph.invoke(
            {
                "references": [
                    {
                        "ref_id": "@smith2024graph",
                        "title": "Graph Systems",
                        "authors": ["Alice Smith"],
                        "year": 2024,
                        "doi": "10.1000/graph",
                        "venue": "GraphConf",
                        "abstract": "Graph-native thesis workflow.",
                        "source": "semantic_scholar",
                        "verified_metadata": False,
                    }
                ],
                "citation_uses": [
                    {
                        "cite_key": "@smith2024graph",
                        "claim_id": "claim-1",
                        "chapter_id": "ch_01",
                        "subtask_id": "sub_01",
                    }
                ],
                "claim_text_by_id": {"claim-1": "Graph orchestration improves durability."},
                "citation_ledger": {},
                "verified_citations": [],
                "flagged_citations": [],
            }
        )

        self.assertEqual(result["verified_citations"], [])
        self.assertEqual(result["flagged_citations"][0]["reason"], "metadata_mismatch")
        self.assertEqual(metadata.doi_checks, ["10.1000/graph"])
        self.assertEqual(metadata.metadata_checks, ["@smith2024graph"])


class _FinalizingCitationReactAgent:
    def invoke(self, agent_input, config, **kwargs):
        del config, kwargs
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
                    "detail": "Citation ledger integration verified.",
                    "evidence_excerpt": item.get("evidence", ""),
                }
            )
        finalize_citation_review.invoke({"summary": "citation ledger integration complete"})
        return {"messages": []}


class GraphPhase3FlowTests(unittest.TestCase):
    def test_graph_runs_to_markdown_export_with_citation_ledger(self):
        from muse.graph.launcher import build_graph, invoke

        class _Search:
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

        class _LLM:
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
                    return {"final_text": "Polished thesis chapter.", "polish_notes": ["ok"]}
                if "摘要撰写专家" in system:
                    return {"abstract": "中文摘要。", "keywords": ["图", "流程"]}
                if "academic abstract writer" in system:
                    return {"abstract": "English abstract.", "keywords": ["graph", "workflow"]}
                raise AssertionError(f"unexpected prompt: {system}")

            def entailment(self, *, premise, hypothesis, route="reasoning"):
                return "entailment"

        class _Metadata:
            def verify_doi(self, doi):
                return True

            def crosscheck_metadata(self, ref):
                return True

        class _Services:
            def __init__(self):
                self.llm = _LLM()
                self.search = _Search()
                self.metadata = _Metadata()
                self.local_refs = []
                self.rag_index = None

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
            with patch(
                "muse.graph.subgraphs.citation._build_react_citation_agent",
                return_value=_FinalizingCitationReactAgent(),
            ):
                graph = build_graph(settings, services=_Services(), thread_id="run-4", auto_approve=True)
                result = invoke(
                    graph,
                    {
                        "project_id": "run-4",
                        "topic": "LangGraph thesis automation",
                        "discipline": "Computer Science",
                        "language": "zh",
                        "format_standard": "GB/T 7714-2015",
                        "output_format": "markdown",
                    },
                    thread_id="run-4",
                )

            self.assertTrue(result["citation_ledger"])
            self.assertTrue(Path(result["output_filepath"]).exists())


if __name__ == "__main__":
    unittest.main()
