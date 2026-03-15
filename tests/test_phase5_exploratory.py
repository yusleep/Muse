import json
import tempfile
import unittest

from muse.config import Settings, _yaml_to_settings


class _PerspectiveLLM:
    def structured(self, *, system, user, route="default", max_tokens=2500):
        del user, route, max_tokens
        system_lower = system.lower()
        if "expert personas" in system_lower:
            return {
                "personas": [
                    {"name": "Systems Researcher", "expertise": "distributed systems", "focus_area": "durability"},
                    {"name": "Performance Engineer", "expertise": "performance analysis", "focus_area": "tail latency"},
                    {"name": "Tooling Architect", "expertise": "developer tooling", "focus_area": "operator workflow"},
                ]
            }
        if "pairwise dialogues" in system_lower:
            return {
                "dialogues": [
                    {
                        "pair": ["Systems Researcher", "Performance Engineer"],
                        "questions": [
                            "How do checkpoint policies affect tail latency?",
                            "Which orchestration bottlenecks are under-studied?",
                        ],
                    }
                ],
                "search_queries": [
                    "checkpoint policy tail latency",
                    "chapter graph orchestration bottleneck",
                    "checkpoint policy tail latency",
                ],
            }
        raise AssertionError(system)


class _PerspectiveSearchClient:
    def __init__(self):
        self.calls = []

    def search_multi_source(self, topic, discipline, extra_queries=None):
        del topic, discipline
        queries = list(extra_queries or [])
        self.calls.append(queries)
        return (
            [
                {
                    "ref_id": "@keep1",
                    "title": "Existing Paper",
                    "authors": ["A"],
                    "year": 2023,
                    "doi": "10.1000/existing",
                    "venue": "Conf",
                    "abstract": "Existing reference.",
                    "source": "semantic_scholar",
                    "verified_metadata": True,
                },
                {
                    "ref_id": "@new2",
                    "title": "Fresh Angle",
                    "authors": ["B"],
                    "year": 2024,
                    "doi": "10.1000/fresh",
                    "venue": "Symp",
                    "abstract": "Fresh reference.",
                    "source": "semantic_scholar",
                    "verified_metadata": True,
                },
            ],
            queries,
        )


class _Services:
    def __init__(self, *, llm=None, search=None):
        self.llm = llm
        self.search = search
        self.local_refs = []
        self.rag_index = None


class _SinglePassLLM:
    def __init__(self):
        self.calls = []

    def text(self, *, system, user, route="default", temperature=0.2, max_tokens=2500):
        del system, route, temperature, max_tokens
        payload = json.loads(user)
        self.calls.append(payload)
        chapter_plan = payload["chapter_plan"]
        chapter_id = chapter_plan["chapter_id"]
        chapter_title = chapter_plan["chapter_title"]
        subtask = chapter_plan["subtask_plan"][0]
        claim = f"{chapter_title} improves thesis quality."
        return json.dumps(
            {
                "merged_text": f"{chapter_title} content with citation.",
                "quality_scores": {"coherence": 4, "logic": 4},
                "iterations_used": 1,
                "subtask_results": [
                    {
                        "subtask_id": subtask["subtask_id"],
                        "title": subtask["title"],
                        "target_words": subtask["target_words"],
                        "output_text": f"{chapter_title} subsection content.",
                        "actual_words": 5,
                        "citations_used": ["@smith2024graph"],
                        "key_claims": [claim],
                    }
                ],
            },
            ensure_ascii=False,
        )


class Phase5PerspectiveTests(unittest.TestCase):
    def test_perspective_node_generates_personas_and_queries(self):
        from muse.graph.nodes.perspective import build_perspective_node

        node = build_perspective_node(services=_Services(llm=_PerspectiveLLM()))
        result = node(
            {
                "topic": "LangGraph thesis automation",
                "discipline": "Computer Science",
                "references": [
                    {
                        "ref_id": "@smith2024graph",
                        "title": "Graph Systems",
                        "authors": ["Alice Smith"],
                        "year": 2024,
                        "abstract": "Graph-native orchestration study.",
                    }
                ],
            }
        )

        self.assertEqual(len(result["perspectives"]), 3)
        self.assertEqual(result["perspectives"][0]["name"], "Systems Researcher")
        self.assertEqual(
            result["perspective_queries"],
            [
                "checkpoint policy tail latency",
                "chapter graph orchestration bottleneck",
            ],
        )

    def test_second_round_search_uses_perspective_queries_and_returns_only_new_refs(self):
        from muse.graph.nodes.search import build_search_node

        search = _PerspectiveSearchClient()
        node = build_search_node(
            None,
            _Services(search=search),
            state_query_key="perspective_queries",
        )
        result = node(
            {
                "topic": "LangGraph thesis automation",
                "discipline": "Computer Science",
                "references": [
                    {
                        "ref_id": "@keep1",
                        "title": "Existing Paper",
                        "authors": ["A"],
                        "year": 2023,
                        "doi": "10.1000/existing",
                        "venue": "Conf",
                        "abstract": "Existing reference.",
                        "source": "semantic_scholar",
                        "verified_metadata": True,
                    }
                ],
                "search_queries": ["graph workflow"],
                "perspective_queries": [
                    "checkpoint policy tail latency",
                    "chapter graph orchestration bottleneck",
                ],
            }
        )

        self.assertEqual(
            search.calls,
            [["checkpoint policy tail latency", "chapter graph orchestration bottleneck"]],
        )
        self.assertEqual([ref["ref_id"] for ref in result["references"]], ["@new2"])
        self.assertEqual(
            result["search_queries"],
            [
                "graph workflow",
                "checkpoint policy tail latency",
                "chapter graph orchestration bottleneck",
            ],
        )
        self.assertIn("Existing Paper", result["literature_summary"])
        self.assertIn("Fresh Angle", result["literature_summary"])

    def test_graph_inserts_perspective_round_trip_before_outline(self):
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
            graph = build_graph(
                settings,
                services=_Services(llm=_PerspectiveLLM(), search=_PerspectiveSearchClient()),
                thread_id="run-perspective",
            )
            graph_repr = graph.get_graph()
            edges = {(edge.source, edge.target) for edge in graph_repr.edges}

            self.assertIn("perspective_discovery", graph_repr.nodes)
            self.assertIn("search_perspectives", graph_repr.nodes)
            self.assertIn(("review_refs", "perspective_discovery"), edges)
            self.assertIn(("perspective_discovery", "search_perspectives"), edges)
            self.assertIn(("search_perspectives", "outline"), edges)


class Phase5SinglePassTests(unittest.TestCase):
    def test_yaml_extracts_writing_mode(self):
        self.assertEqual(
            _yaml_to_settings({"writing": {"mode": "single_pass"}}, {}, None)["writing_mode"],
            "single_pass",
        )
        self.assertEqual(
            _yaml_to_settings({"writing_mode": "single_pass"}, {}, None)["writing_mode"],
            "single_pass",
        )

    def test_single_pass_writer_builds_chapter_results_for_merge_node(self):
        from muse.graph.nodes.merge import build_merge_chapters_node
        from muse.graph.nodes.single_pass import build_single_pass_node

        llm = _SinglePassLLM()
        node = build_single_pass_node(
            settings=Settings(
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
            ),
            services=_Services(llm=llm),
        )
        state = {
            "topic": "LangGraph thesis automation",
            "discipline": "Computer Science",
            "language": "zh",
            "references": [
                {
                    "ref_id": "@smith2024graph",
                    "title": "Graph Systems",
                    "authors": ["Alice Smith"],
                    "year": 2024,
                    "abstract": "Graph-native orchestration study.",
                }
            ],
            "chapter_plans": [
                {
                    "chapter_id": "ch_01",
                    "chapter_title": "绪论",
                    "target_words": 1200,
                    "subtask_plan": [{"subtask_id": "s1", "title": "研究背景", "target_words": 1200}],
                },
                {
                    "chapter_id": "ch_02",
                    "chapter_title": "系统设计",
                    "target_words": 1200,
                    "subtask_plan": [{"subtask_id": "s1", "title": "总体架构", "target_words": 1200}],
                },
            ],
        }

        result = node(state)

        self.assertEqual(set(result["chapters"].keys()), {"ch_01", "ch_02"})
        self.assertEqual(result["chapters"]["ch_01"]["iterations_used"], 1)
        self.assertTrue(result["chapters"]["ch_02"]["citation_uses"])
        self.assertIn("conversation_history", llm.calls[1])
        self.assertIn("previous_chapters", llm.calls[1])

        merge = build_merge_chapters_node(None, None)
        merged = merge({**state, **result})
        self.assertEqual(len(merged["paper_package"]["chapter_results"]), 2)
        self.assertIn("绪论 content", merged["final_text"])
        self.assertIn("系统设计 content", merged["final_text"])

    def test_graph_adds_single_pass_writer_route(self):
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
            object.__setattr__(settings, "writing_mode", "single_pass")

            graph = build_graph(
                settings,
                services=_Services(llm=_PerspectiveLLM(), search=_PerspectiveSearchClient()),
                thread_id="run-single-pass",
            )
            graph_repr = graph.get_graph()
            edges = {(edge.source, edge.target) for edge in graph_repr.edges}

            self.assertIn("single_pass_writer", graph_repr.nodes)
            self.assertIn(("ref_analysis", "single_pass_writer"), edges)
            self.assertIn(("single_pass_writer", "merge_chapters"), edges)


if __name__ == "__main__":
    unittest.main()
