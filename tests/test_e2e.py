"""Minimal end-to-end smoke test for the full 6-stage pipeline.

Runs all stages through the real Runtime → ThesisEngine → stage functions,
but replaces the three network clients (LLM, search, metadata) with in-process
stubs that return the smallest valid JSON each stage needs.

No network calls, no GPU, no optional dependencies required.
Completes in < 1 second.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from thesis_agent.config import Settings
from thesis_agent.runtime import Runtime
from thesis_agent.schemas import new_thesis_state


# ---------------------------------------------------------------------------
# Minimal stub clients
# ---------------------------------------------------------------------------

class _MinimalLLM:
    """Returns the bare-minimum valid JSON for every pipeline route."""

    _CHAPTER = {
        "chapter_id": "ch_01",
        "chapter_title": "Introduction",
        "target_words": 200,
        "complexity": "low",
        "subsections": [{"title": "Background"}],
    }

    def structured(self, *, system, user, route, max_tokens):
        if route == "outline":
            if "queries" in system.lower():          # _generate_search_queries
                return {"queries": ["test query 1", "test query 2"]}
            if "Analyze" in system:                   # _analyze_topic
                return {
                    "research_gaps": [],
                    "core_concepts": [],
                    "methodology_domain": "general",
                    "suggested_contributions": [],
                }
            return {"chapters": [self._CHAPTER]}     # stage2_outline

        if route == "writing":
            return {
                "text": "Minimal e2e test content.",
                "citations_used": [],   # no citations → stage 4 is a no-op
                "key_claims": ["claim1"],
                "transition_out": "",
                "glossary_additions": {},
                "self_assessment": {
                    "confidence": 0.9,
                    "weak_spots": [],
                    "needs_revision": False,
                },
            }

        if route == "review":
            return {
                "scores": {
                    "coherence": 5, "logic": 5, "citation": 5,
                    "term_consistency": 5, "balance": 5, "redundancy": 5,
                },
                "review_notes": [],
            }

        if route == "polish":
            import json
            payload = json.loads(user)
            return {"final_text": payload.get("text", ""), "polish_notes": []}

        if route == "reasoning":
            return "ENTAILS"

        return {}

    # used by stage4_verify for entailment checks (skipped when no citations)
    def entailment(self, *, premise, hypothesis, route):
        return "ENTAILS"


class _MinimalSearch:
    def search_multi_source(self, topic, discipline, extra_queries=None):
        refs = [
            {
                "ref_id": "@e2e_ref1",
                "title": "E2E Test Paper",
                "authors": ["Author A"],
                "year": 2024,
                "doi": None,
                "venue": None,
                "abstract": "A paper for smoke testing.",
                "source": "test",
                "verified_metadata": True,
            }
        ]
        return refs, [topic]


class _MinimalMetadata:
    def verify_doi(self, doi):
        return True

    def crosscheck_metadata(self, ref):
        return {"match": True}


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------

class E2EPipelineTests(unittest.TestCase):

    def _make_runtime(self, runs_dir: str) -> Runtime:
        """Build a Runtime with stub clients and no real API credentials."""
        settings = Settings(
            llm_api_key="x",
            llm_base_url="http://localhost",
            llm_model="stub",
            model_router_config={},
            runs_dir=runs_dir,
            semantic_scholar_api_key=None,
            openalex_email=None,
            crossref_mailto=None,
            refs_dir=None,
        )
        runtime = Runtime(settings)
        # Replace the real network clients with stubs
        runtime.llm = _MinimalLLM()
        runtime.search = _MinimalSearch()
        runtime.metadata = _MinimalMetadata()
        return runtime

    def test_full_pipeline_produces_markdown_output(self):
        """All 6 stages complete and output/thesis.md is written."""
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._make_runtime(tmp)
            run_id = runtime.store.create_run(topic="Byzantine Fault Tolerance")

            state = new_thesis_state(
                project_id=run_id,
                topic="Byzantine Fault Tolerance",
                discipline="Computer Science",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            runtime.store.save_state(run_id, state)

            engine = runtime.build_engine(run_id=run_id, output_format="markdown")
            result = engine.run(run_id=run_id, start_stage=1, auto_approve=True)

            # Pipeline ran to completion
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["stage"], 6)

            # State persisted correctly
            final = runtime.store.load_state(run_id)
            self.assertEqual(final["stage1_status"], "hitl_review")
            self.assertEqual(final["stage6_status"], "completed")
            self.assertTrue(len(final["references"]) >= 1)
            self.assertTrue(len(final["chapter_plans"]) >= 1)
            self.assertTrue(len(final["chapter_results"]) >= 1)
            self.assertFalse(final["flagged_citations"])   # no citations → nothing flagged

            # Output file exists and contains written content
            output_path = final["output_filepath"]
            self.assertTrue(os.path.isfile(output_path), f"Missing: {output_path}")
            content = open(output_path, encoding="utf-8").read()
            self.assertIn("Minimal e2e test content", content)

    def test_pipeline_sets_rag_disabled_when_no_refs_dir(self):
        """Without --refs-dir, rag_enabled is False and local_refs_count is 0."""
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._make_runtime(tmp)
            run_id = runtime.store.create_run(topic="Test")

            state = new_thesis_state(
                project_id=run_id,
                topic="Test",
                discipline="CS",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            runtime.store.save_state(run_id, state)

            engine = runtime.build_engine(run_id=run_id, output_format="markdown")
            engine.run(run_id=run_id, start_stage=1, auto_approve=True)

            final = runtime.store.load_state(run_id)
            self.assertEqual(final.get("local_refs_count", 0), 0)
            self.assertFalse(final.get("rag_enabled", True))

    def test_pipeline_with_local_refs_sets_rag_enabled(self):
        """With a refs_dir containing .md files, local_refs_count > 0 and rag_enabled = True."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create a refs directory with one markdown file
            refs_dir = os.path.join(tmp, "refs")
            os.makedirs(refs_dir)
            with open(os.path.join(refs_dir, "smith_2023_bft.md"), "w") as f:
                f.write("# Byzantine Fault Tolerance\n\nThis paper discusses BFT protocols.\n" * 20)

            settings = Settings(
                llm_api_key="x",
                llm_base_url="http://localhost",
                llm_model="stub",
                model_router_config={},
                runs_dir=tmp,
                semantic_scholar_api_key=None,
                openalex_email=None,
                crossref_mailto=None,
                refs_dir=refs_dir,
            )
            runtime = Runtime(settings)
            runtime.llm = _MinimalLLM()
            runtime.search = _MinimalSearch()
            runtime.metadata = _MinimalMetadata()

            # RAG index built during __init__
            self.assertGreater(len(runtime.local_refs), 0)

            run_id = runtime.store.create_run(topic="BFT")
            state = new_thesis_state(
                project_id=run_id, topic="BFT", discipline="CS",
                language="zh", format_standard="GB/T 7714-2015",
            )
            runtime.store.save_state(run_id, state)

            engine = runtime.build_engine(run_id=run_id, output_format="markdown")
            engine.run(run_id=run_id, start_stage=1, auto_approve=True)

            final = runtime.store.load_state(run_id)
            self.assertGreater(final.get("local_refs_count", 0), 0)
            # Local refs appear first in references list
            self.assertEqual(final["references"][0]["source"], "local")
