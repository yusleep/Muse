import tempfile
import unittest

from muse.config import Settings


class _ParallelSearch:
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


class _ParallelLLM:
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
                    },
                    {
                        "chapter_id": "ch_02",
                        "chapter_title": "系统设计",
                        "target_words": 1200,
                        "complexity": "low",
                        "subsections": [{"title": "架构设计"}],
                    },
                ]
            }
        if "Write one thesis subsection with citations" in system:
            return {
                "text": "Drafted subsection content.",
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
        raise AssertionError(f"unexpected prompt: {system}")


class _Services:
    def __init__(self):
        self.llm = _ParallelLLM()
        self.search = _ParallelSearch()
        self.local_refs = []
        self.rag_index = None


class ParallelChapterTests(unittest.TestCase):
    def test_main_graph_fans_out_and_merges_chapter_results(self):
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
            graph = build_graph(settings, services=_Services(), thread_id="run-2", auto_approve=True)
            result = invoke(
                graph,
                {
                    "project_id": "run-2",
                    "topic": "LangGraph thesis automation",
                    "discipline": "Computer Science",
                    "language": "zh",
                    "format_standard": "GB/T 7714-2015",
                    "output_format": "markdown",
                },
                thread_id="run-2",
            )

            self.assertEqual(set(result["chapters"].keys()), {"ch_01", "ch_02"})
            self.assertEqual(len(result["paper_package"]["chapter_results"]), 2)
            self.assertIn("[绪论]", result["paper_package"]["thesis_summary"])


if __name__ == "__main__":
    unittest.main()
