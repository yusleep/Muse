import argparse
import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch

from muse.config import Settings
from muse.services.store import RunStore


class _ResumeSearch:
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


class _ResumeLLM:
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
        self.llm = _ResumeLLM()
        self.search = _ResumeSearch()
        self.local_refs = []
        self.rag_index = None


class _UnexpectedSearch:
    def search_multi_source(self, topic, discipline, extra_queries=None):
        raise AssertionError("search should not rerun during state-based resume fallback")


class _OutlineOnlyServices:
    def __init__(self):
        self.llm = _ResumeLLM()
        self.search = _UnexpectedSearch()
        self.local_refs = []
        self.rag_index = None


class _CliRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = RunStore(base_dir=settings.runs_dir)

    def build_graph(self, *, thread_id: str, auto_approve: bool = True):
        from muse.graph.launcher import build_graph

        return build_graph(
            self.settings,
            services=_OutlineOnlyServices(),
            thread_id=thread_id,
            auto_approve=auto_approve,
        )


class CheckpointResumeTests(unittest.TestCase):
    def test_resume_works_after_rebuilding_graph_from_same_sqlite_checkpoint(self):
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
            thread_id = "resume-run"
            first_graph = build_graph(settings, services=_Services(), thread_id=thread_id, auto_approve=False)
            first = invoke(
                first_graph,
                {
                    "project_id": thread_id,
                    "topic": "LangGraph thesis automation",
                    "discipline": "Computer Science",
                    "language": "zh",
                    "format_standard": "GB/T 7714-2015",
                    "output_format": "markdown",
                },
                thread_id=thread_id,
            )
            self.assertEqual(first["__interrupt__"][0].value["stage"], "research")

            second_graph = build_graph(settings, services=_Services(), thread_id=thread_id, auto_approve=False)
            resumed = invoke(
                second_graph,
                None,
                thread_id=thread_id,
                resume={"stage": "research", "approved": True},
            )
            self.assertEqual(resumed["__interrupt__"][0].value["stage"], "outline")

    def test_cmd_resume_rehydrates_saved_state_when_checkpoint_db_is_missing(self):
        from muse.cli import cmd_resume

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
            runtime = _CliRuntime(settings)
            run_id = "resume-without-checkpoint"
            runtime.store.save_state(
                run_id,
                {
                    "project_id": run_id,
                    "topic": "LangGraph thesis automation",
                    "discipline": "Computer Science",
                    "language": "zh",
                    "format_standard": "GB/T 7714-2015",
                    "output_format": "markdown",
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
                    "search_queries": ["graph workflow"],
                    "literature_summary": "- Graph Systems (2024) GraphConf",
                    "outline": {},
                    "chapter_plans": [],
                    "chapters": {},
                    "citation_uses": [],
                    "citation_ledger": {},
                    "claim_text_by_id": {},
                    "thesis_summary": "",
                    "verified_citations": [],
                    "flagged_citations": [],
                    "paper_package": {},
                    "final_text": "",
                    "polish_notes": [],
                    "abstract_zh": "",
                    "abstract_en": "",
                    "keywords_zh": [],
                    "keywords_en": [],
                    "output_filepath": "",
                    "export_artifacts": {},
                    "export_warnings": [],
                    "review_feedback": [],
                    "rag_enabled": False,
                    "local_refs_count": 0,
                },
            )
            runtime.store.append_hitl_feedback(run_id, {"stage": "research", "approved": True, "comment": ""})

            output = io.StringIO()
            args = argparse.Namespace(run_id=run_id, refs_dir=None, auto_approve=False)
            with patch("muse.cli._runtime_from_args", return_value=runtime), contextlib.redirect_stdout(output):
                exit_code = cmd_resume(args)

            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["status"], "waiting_hitl")
            self.assertEqual(payload["stage"], "outline")
            self.assertTrue(runtime.store.load_state(run_id)["chapter_plans"])


if __name__ == "__main__":
    unittest.main()
