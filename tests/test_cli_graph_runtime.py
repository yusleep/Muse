import argparse
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from muse.cli import build_parser, cmd_resume
from muse.config import Settings
from muse.graph.launcher import invoke
from muse.runtime import Runtime


class GraphCliSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_resume_command_no_longer_accepts_start_stage(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["resume", "--run-id", "r1", "--start-stage", "3"])

    def test_review_stage_accepts_named_stage(self):
        args = self.parser.parse_args(["review", "--run-id", "r1", "--stage", "research", "--approve"])
        self.assertEqual(args.stage, "research")
        self.assertTrue(args.approve)


class GraphRuntimeSurfaceTests(unittest.TestCase):
    def test_runtime_build_graph_replaces_build_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Runtime(
                Settings(
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
            )

            graph = runtime.build_graph(thread_id="run-5", auto_approve=True)
            self.assertTrue(hasattr(graph, "invoke"))
            self.assertFalse(hasattr(runtime, "build_engine"))


class _ResumeFallbackSearch:
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


class _ResumeFallbackOutlineLLM:
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


class _ExplodingSearch:
    def search_multi_source(self, topic, discipline, extra_queries=None):
        raise AssertionError("resume fallback should not rerun search when only state.json remains")


class ResumeCommandFallbackTests(unittest.TestCase):
    def test_resume_uses_state_json_when_sqlite_checkpoint_is_missing(self):
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
            run_id = "resume-fallback"

            initial_runtime = Runtime(settings)
            initial_runtime.search = _ResumeFallbackSearch()
            initial_runtime.llm = _ResumeFallbackOutlineLLM()
            initial_runtime.local_refs = []
            initial_runtime.rag_index = None
            initial_runtime.store.create_run(topic="LangGraph thesis automation")
            run_dir = Path(tmp) / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            initial_runtime.store.append_hitl_feedback(run_id, {"stage": "research", "approved": True, "comment": ""})

            graph = initial_runtime.build_graph(thread_id=run_id, auto_approve=False)
            first = invoke(
                graph,
                {
                    "project_id": run_id,
                    "topic": "LangGraph thesis automation",
                    "discipline": "Computer Science",
                    "language": "zh",
                    "format_standard": "GB/T 7714-2015",
                    "output_format": "markdown",
                },
                thread_id=run_id,
            )
            initial_runtime.store.save_state(run_id, {key: value for key, value in first.items() if key != "__interrupt__"})

            checkpoint_db = Path(tmp) / run_id / "graph" / "checkpoints.sqlite"
            checkpoint_db.unlink()

            resume_runtime = Runtime(settings)
            resume_runtime.search = _ExplodingSearch()
            resume_runtime.llm = _ResumeFallbackOutlineLLM()
            resume_runtime.local_refs = []
            resume_runtime.rag_index = None

            args = argparse.Namespace(run_id=run_id, refs_dir=None, auto_approve=False)
            stdout = io.StringIO()
            with patch("muse.cli._runtime_from_args", return_value=resume_runtime):
                with redirect_stdout(stdout):
                    cmd_resume(args)

            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "waiting_hitl")
            self.assertEqual(payload["stage"], "outline")


class OptionalLanggraphImportTests(unittest.TestCase):
    def _run_without_langgraph(self, code: str) -> subprocess.CompletedProcess[str]:
        script = f"""
import importlib.abc
import pathlib
import sys

class _BlockLanggraph(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "langgraph" or fullname.startswith("langgraph."):
            raise ModuleNotFoundError("No module named 'langgraph'")
        return None

sys.meta_path.insert(0, _BlockLanggraph())
sys.path.insert(0, str(pathlib.Path.cwd()))
{code}
"""
        return subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )

    def test_imports_do_not_fail_until_graph_runtime_is_used(self):
        result = self._run_without_langgraph(
            """
import muse
from muse.cli import build_parser
from muse.config import Settings
from muse.runtime import Runtime

print("IMPORT_OK")
build_parser()
runtime = Runtime(
    Settings(
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
)
try:
    runtime.build_graph(thread_id="dry-run")
except RuntimeError as exc:
    print(type(exc).__name__)
    print(str(exc))
else:
    raise SystemExit("expected RuntimeError when langgraph is unavailable")
"""
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("IMPORT_OK", result.stdout)
        self.assertIn("RuntimeError", result.stdout)
        self.assertIn("langgraph", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
