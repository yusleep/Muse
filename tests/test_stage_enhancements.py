"""Tests for stage enhancements: Stage 1 local refs, Stage 2 topic analysis,
Stage 3 refs_snapshot with abstract, Stage 5 per-chapter polish."""

import json
import unittest
from unittest.mock import MagicMock

from thesis_agent.schemas import new_thesis_state
from thesis_agent.stages import (
    _generate_search_queries,
    stage1_literature,
    stage5_polish,
)


# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides):
    state = new_thesis_state(
        project_id="test",
        topic="Byzantine fault tolerance",
        discipline="Computer Science",
        language="zh",
        format_standard="GB/T 7714-2015",
    )
    state.update(overrides)
    return state


class _FakeSearchClient:
    def __init__(self, refs=None):
        self._refs = refs or [
            {"ref_id": "@online1", "title": "Online Paper", "authors": [], "year": 2022,
             "doi": None, "venue": None, "abstract": "Online abstract",
             "source": "semantic_scholar", "verified_metadata": True}
        ]
        self.last_extra_queries = None

    def search_multi_source(self, topic, discipline, extra_queries=None):
        self.last_extra_queries = extra_queries
        return list(self._refs), [topic]


class _FakeLLMClient:
    """Minimal LLM double that returns predictable JSON."""

    def structured(self, *, system, user, route, max_tokens):
        if "search queries" in system.lower() or "queries" in system.lower():
            return {"queries": ["query A", "query B", "query C"]}
        if "analyze" in system.lower():
            return {
                "research_gaps": ["gap1"],
                "core_concepts": ["concept1"],
                "methodology_domain": "systems",
                "suggested_contributions": ["contrib1"],
            }
        if "polish" in system.lower():
            payload = json.loads(user)
            return {
                "final_text": f"[polished] {payload.get('text', '')}",
                "polish_notes": ["Note A"],
            }
        return {}


# ---------------------------------------------------------------------------
# Stage 1 tests
# ---------------------------------------------------------------------------

class TestStage1LocalRefs(unittest.TestCase):
    def _make_local_refs(self, n=2):
        return [
            {
                "ref_id": f"@local_{i}",
                "title": f"Local Paper {i}",
                "authors": [],
                "year": 2023,
                "doi": None,
                "venue": None,
                "abstract": f"Local abstract {i}",
                "source": "local",
                "filepath": f"/tmp/paper{i}.pdf",
                "full_text": f"Full text {i}",
                "verified_metadata": False,
            }
            for i in range(n)
        ]

    def test_local_refs_prepended_before_online(self):
        state = _make_state()
        search = _FakeSearchClient()
        local = self._make_local_refs(2)

        stage1_literature(state, search, local_refs=local)

        refs = state["references"]
        # Local refs come first
        self.assertEqual(refs[0]["ref_id"], "@local_0")
        self.assertEqual(refs[1]["ref_id"], "@local_1")
        # Online ref appended
        self.assertEqual(refs[2]["ref_id"], "@online1")

    def test_deduplication_by_ref_id(self):
        """If online search returns a ref_id that matches a local ref, it's dropped."""
        state = _make_state()
        overlapping_online = [
            {"ref_id": "@local_0", "title": "Duplicate", "authors": [], "year": 2022,
             "doi": None, "venue": None, "abstract": "", "source": "semantic_scholar",
             "verified_metadata": True}
        ]
        search = _FakeSearchClient(refs=overlapping_online)
        local = self._make_local_refs(1)

        stage1_literature(state, search, local_refs=local)

        refs = state["references"]
        # Only one entry with ref_id @local_0 (the local one)
        ids = [r["ref_id"] for r in refs]
        self.assertEqual(ids.count("@local_0"), 1)
        self.assertEqual(refs[0]["source"], "local")

    def test_no_local_refs_uses_online_only(self):
        state = _make_state()
        search = _FakeSearchClient()
        stage1_literature(state, search)
        self.assertEqual(len(state["references"]), 1)
        self.assertEqual(state["references"][0]["ref_id"], "@online1")

    def test_extra_queries_passed_when_llm_provided(self):
        state = _make_state()
        search = _FakeSearchClient()
        llm = _FakeLLMClient()

        stage1_literature(state, search, llm_client=llm)

        # LLM generated queries should have been passed to search
        self.assertIsNotNone(search.last_extra_queries)
        self.assertIsInstance(search.last_extra_queries, list)
        self.assertGreater(len(search.last_extra_queries), 0)

    def test_no_extra_queries_without_llm(self):
        state = _make_state()
        search = _FakeSearchClient()

        stage1_literature(state, search)  # no llm_client

        self.assertIsNone(search.last_extra_queries)


class TestGenerateSearchQueries(unittest.TestCase):
    def test_returns_list_of_strings(self):
        llm = _FakeLLMClient()
        queries = _generate_search_queries(llm, "BFT", "CS")
        self.assertIsInstance(queries, list)
        self.assertTrue(all(isinstance(q, str) for q in queries))

    def test_llm_failure_returns_empty_list(self):
        llm = MagicMock()
        llm.structured.side_effect = RuntimeError("LLM down")
        queries = _generate_search_queries(llm, "topic", "discipline")
        self.assertEqual(queries, [])


# ---------------------------------------------------------------------------
# Stage 3 refs_snapshot test (abstract field)
# ---------------------------------------------------------------------------

class TestRefsSnapshotHasAbstract(unittest.TestCase):
    """Verify that refs_snapshot built in _write_subtasks includes abstract."""

    def test_abstract_in_snapshot(self):
        from thesis_agent.stages import _write_subtasks

        refs = [
            {
                "ref_id": "@r1",
                "title": "Paper",
                "year": 2023,
                "abstract": "Important abstract content for verification",
            }
        ]

        received_payloads = []

        class _CaptureLLM:
            def structured(self, *, system, user, route, max_tokens):
                payload = json.loads(user)
                received_payloads.append(payload)
                return {
                    "text": "Subsection text.",
                    "citations_used": ["@r1"],
                    "key_claims": [],
                    "transition_out": "",
                    "glossary_additions": {},
                    "self_assessment": {"confidence": 0.8, "weak_spots": [], "needs_revision": False},
                }

        state = _make_state()
        state["references"] = refs
        subtask_plan = [
            {"subtask_id": "ch01_s01", "title": "Intro", "target_words": 500}
        ]

        _write_subtasks(
            llm_client=_CaptureLLM(),
            state=state,
            chapter_title="Chapter 1",
            subtask_plan=subtask_plan,
            revision_instructions={},
            previous=[],
        )

        self.assertTrue(len(received_payloads) > 0)
        snapshot = received_payloads[0].get("available_references", [])
        self.assertTrue(len(snapshot) > 0)
        self.assertIn("abstract", snapshot[0])
        self.assertIn("Important abstract", snapshot[0]["abstract"])


# ---------------------------------------------------------------------------
# Stage 5 per-chapter polish tests
# ---------------------------------------------------------------------------

class TestStage5PerChapterPolish(unittest.TestCase):
    def _make_state_with_chapters(self, n: int = 3):
        state = _make_state()
        state["chapter_results"] = [
            {
                "chapter_id": f"ch_{i:02d}",
                "chapter_title": f"Chapter {i}",
                "merged_text": f"Original text for chapter {i}.",
            }
            for i in range(1, n + 1)
        ]
        return state

    def test_polish_called_once_per_chapter(self):
        call_count = [0]
        chapter_titles_received = []

        class _CountingLLM:
            def structured(self, *, system, user, route, max_tokens):
                call_count[0] += 1
                payload = json.loads(user)
                chapter_titles_received.append(payload.get("chapter_title", ""))
                return {
                    "final_text": f"[polished] {payload.get('text', '')}",
                    "polish_notes": ["note"],
                }

        state = self._make_state_with_chapters(3)
        stage5_polish(state, _CountingLLM())

        self.assertEqual(call_count[0], 3)
        self.assertIn("Chapter 1", chapter_titles_received)
        self.assertIn("Chapter 2", chapter_titles_received)
        self.assertIn("Chapter 3", chapter_titles_received)

    def test_failed_chapter_falls_back_to_original(self):
        call_count = [0]

        class _FailingLLM:
            def structured(self, *, system, user, route, max_tokens):
                call_count[0] += 1
                raise RuntimeError("LLM error")

        state = self._make_state_with_chapters(2)
        stage5_polish(state, _FailingLLM())

        # Should fall back to original text, not crash
        final = state["final_text"]
        self.assertIn("Original text for chapter 1", final)
        self.assertIn("Original text for chapter 2", final)

    def test_polish_notes_include_chapter_prefix(self):
        llm = _FakeLLMClient()
        state = self._make_state_with_chapters(2)
        stage5_polish(state, llm)

        notes = state["polish_notes"]
        self.assertTrue(any("Chapter 1" in n for n in notes))
        self.assertTrue(any("Chapter 2" in n for n in notes))

    def test_empty_chapter_text_skipped(self):
        state = _make_state()
        state["chapter_results"] = [
            {"chapter_id": "ch_01", "chapter_title": "Chapter 1", "merged_text": "   "},
        ]
        llm = MagicMock()
        stage5_polish(state, llm)
        # LLM should not have been called for empty chapter
        llm.structured.assert_not_called()
