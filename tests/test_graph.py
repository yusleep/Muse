import tempfile
import unittest
from pathlib import Path

from muse.config import Settings


class _FakeSearch:
    def search_multi_source(self, topic, discipline, extra_queries=None):
        queries = extra_queries or [topic, f"{topic} {discipline}"]
        return (
            [
                {
                    "ref_id": "@smith2024graph",
                    "title": "Graph Systems",
                    "authors": ["Alice Smith"],
                    "year": 2024,
                    "doi": "10.1000/graph",
                    "venue": "GraphConf",
                    "abstract": "A graph-native academic writing workflow.",
                    "source": "semantic_scholar",
                    "verified_metadata": True,
                }
            ],
            queries,
        )


class _FakeLLM:
    def __init__(self):
        self.calls = 0

    def structured(self, *, system, user, route="default", max_tokens=2500):
        self.calls += 1
        if self.calls == 1:
            return {"queries": ["graph workflow", "langgraph thesis writing"]}
        if self.calls == 2:
            return {
                "research_gaps": ["checkpoint orchestration"],
                "core_concepts": ["langgraph"],
                "methodology_domain": "systems",
                "suggested_contributions": ["durable thesis pipeline"],
            }
        return {
            "chapters": [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "绪论",
                    "target_words": 3000,
                    "complexity": "medium",
                    "subsections": [{"title": "研究背景"}],
                }
            ]
        }


class _FakeServices:
    def __init__(self):
        self.llm = _FakeLLM()
        self.search = _FakeSearch()
        self.local_refs = []
        self.rag_index = None


class GraphShellTests(unittest.TestCase):
    def test_muse_state_merge_dict_reducer_merges_updates(self):
        from muse.graph.state import _merge_dict

        self.assertEqual(_merge_dict({"a": 1}, {"b": 2}), {"a": 1, "b": 2})

    def test_muse_state_paper_package_uses_merge_dict_reducer(self):
        from typing import Annotated, get_args, get_origin, get_type_hints

        from muse.graph.state import MuseState, _merge_dict

        hints = get_type_hints(MuseState, include_extras=True)
        paper_package_hint = hints["paper_package"]

        self.assertIs(get_origin(paper_package_hint), Annotated)
        self.assertIn(_merge_dict, get_args(paper_package_hint)[1:])

    def test_build_graph_runs_initialize_search_outline(self):
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
            services = _FakeServices()
            graph = build_graph(settings, services=services, thread_id="run-1")

            result = invoke(
                graph,
                {
                    "project_id": "run-1",
                    "topic": "LangGraph thesis automation",
                    "discipline": "Computer Science",
                    "language": "zh",
                    "format_standard": "GB/T 7714-2015",
                    "output_format": "markdown",
                },
                thread_id="run-1",
            )

            self.assertEqual(result["search_queries"], ["graph workflow", "langgraph thesis writing"])
            self.assertEqual(len(result["references"]), 1)
            self.assertIn("Graph Systems", result["literature_summary"])
            self.assertEqual(result["outline"]["chapters"][0]["chapter_id"], "ch_01")
            self.assertEqual(result["chapter_plans"][0]["chapter_title"], "绪论")

            checkpoint_path = Path(tmp) / "run-1" / "graph" / "checkpoints.sqlite"
            self.assertTrue(checkpoint_path.exists())

    def test_build_graph_default_classic_skips_global_review_node(self):
        from muse.graph.launcher import build_graph

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

            graph = build_graph(settings, services=_FakeServices(), thread_id="run-classic")
            graph_repr = graph.get_graph()
            edges = {(edge.source, edge.target) for edge in graph_repr.edges}

            self.assertNotIn("review_draft", graph_repr.nodes)
            self.assertNotIn("global_review", graph_repr.nodes)
            self.assertIn("prepare_next_chapter", graph_repr.nodes)
            self.assertIn("update_cross_chapter_state", graph_repr.nodes)
            self.assertIn("coherence_check", graph_repr.nodes)
            self.assertIn("citation_repair", graph_repr.nodes)
            self.assertIn(("approve_outline", "prepare_next_chapter"), edges)
            self.assertIn(("prepare_next_chapter", "chapter_subgraph"), edges)
            self.assertIn(("prepare_next_chapter", "merge_chapters"), edges)
            self.assertIn(("chapter_subgraph", "update_cross_chapter_state"), edges)
            self.assertIn(("update_cross_chapter_state", "prepare_next_chapter"), edges)
            self.assertIn(("merge_chapters", "coherence_check"), edges)
            self.assertIn(("coherence_check", "citation_subgraph"), edges)
            self.assertIn(("citation_subgraph", "citation_repair"), edges)
            self.assertIn(("citation_subgraph", "polish"), edges)
            self.assertIn(("citation_repair", "citation_subgraph"), edges)

    def test_build_graph_layered_mode_inserts_global_review_node(self):
        from muse.graph.launcher import build_graph

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
            object.__setattr__(settings, "review_mode", "layered")

            graph = build_graph(settings, services=_FakeServices(), thread_id="run-layered")
            graph_repr = graph.get_graph()
            edges = {(edge.source, edge.target) for edge in graph_repr.edges}

            self.assertIn("coherence_check", graph_repr.nodes)
            self.assertIn("global_review", graph_repr.nodes)
            self.assertIn("prepare_next_chapter", graph_repr.nodes)
            self.assertIn("update_cross_chapter_state", graph_repr.nodes)
            self.assertIn("citation_repair", graph_repr.nodes)
            self.assertIn(("approve_outline", "prepare_next_chapter"), edges)
            self.assertIn(("prepare_next_chapter", "chapter_subgraph"), edges)
            self.assertIn(("prepare_next_chapter", "merge_chapters"), edges)
            self.assertIn(("chapter_subgraph", "update_cross_chapter_state"), edges)
            self.assertIn(("update_cross_chapter_state", "prepare_next_chapter"), edges)
            self.assertIn(("merge_chapters", "coherence_check"), edges)
            self.assertIn(("coherence_check", "global_review"), edges)
            self.assertIn(("global_review", "citation_subgraph"), edges)
            self.assertIn(("citation_subgraph", "citation_repair"), edges)
            self.assertIn(("citation_subgraph", "polish"), edges)
            self.assertIn(("citation_repair", "citation_subgraph"), edges)


if __name__ == "__main__":
    unittest.main()
