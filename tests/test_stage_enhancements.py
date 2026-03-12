"""Tests for stage enhancements: Stage 1 local refs, Stage 2 topic analysis,
Stage 3 refs_snapshot with abstract, Stage 5 per-chapter polish."""

import json
import unittest
from unittest.mock import MagicMock

from muse.graph.nodes.outline import _analyze_topic, build_outline_node
from muse.graph.nodes.polish import _run_polish
from muse.graph.nodes.search import _generate_search_queries, build_search_node
from muse.schemas import new_thesis_state
from muse.graph.helpers.draft_support import write_subtasks


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


class _SearchServices:
    def __init__(self, *, search=None, llm=None, local_refs=None, rag_index=None):
        self.search = search or _FakeSearchClient()
        self.llm = llm
        self.local_refs = list(local_refs or [])
        self.rag_index = rag_index


class _OutlineServices:
    def __init__(self, llm):
        self.llm = llm


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
        node = build_search_node(None, _SearchServices(search=search, local_refs=local))
        result = node({"topic": state["topic"], "discipline": state["discipline"]})

        refs = result["references"]
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
        node = build_search_node(None, _SearchServices(search=search, local_refs=local))
        result = node({"topic": state["topic"], "discipline": state["discipline"]})

        refs = result["references"]
        # Only one entry with ref_id @local_0 (the local one)
        ids = [r["ref_id"] for r in refs]
        self.assertEqual(ids.count("@local_0"), 1)
        self.assertEqual(refs[0]["source"], "local")

    def test_no_local_refs_uses_online_only(self):
        state = _make_state()
        search = _FakeSearchClient()
        node = build_search_node(None, _SearchServices(search=search))
        result = node({"topic": state["topic"], "discipline": state["discipline"]})
        self.assertEqual(len(result["references"]), 1)
        self.assertEqual(result["references"][0]["ref_id"], "@online1")

    def test_extra_queries_passed_when_llm_provided(self):
        state = _make_state()
        search = _FakeSearchClient()
        llm = _FakeLLMClient()
        node = build_search_node(None, _SearchServices(search=search, llm=llm))
        node({"topic": state["topic"], "discipline": state["discipline"]})

        # LLM generated queries should have been passed to search
        self.assertIsNotNone(search.last_extra_queries)
        self.assertIsInstance(search.last_extra_queries, list)
        self.assertGreater(len(search.last_extra_queries), 0)

    def test_no_extra_queries_without_llm(self):
        state = _make_state()
        search = _FakeSearchClient()
        node = build_search_node(None, _SearchServices(search=search))
        node({"topic": state["topic"], "discipline": state["discipline"]})  # no llm_client

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

    def test_write_subtasks_includes_abstract_in_available_references_snapshot(self):
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

        write_subtasks(
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
        outputs = _run_polish(state, _CountingLLM())

        self.assertEqual(call_count[0], 5)  # 3 polish + 2 abstract generation
        self.assertIn("[polished]", outputs["final_text"])
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
        outputs = _run_polish(state, _FailingLLM())

        # Should fall back to original text, not crash
        final = outputs["final_text"]
        self.assertIn("Original text for chapter 1", final)
        self.assertIn("Original text for chapter 2", final)

    def test_polish_notes_include_chapter_prefix(self):
        llm = _FakeLLMClient()
        state = self._make_state_with_chapters(2)
        outputs = _run_polish(state, llm)

        notes = outputs["polish_notes"]
        self.assertTrue(any("Chapter 1" in n for n in notes))
        self.assertTrue(any("Chapter 2" in n for n in notes))

    def test_empty_chapter_text_skipped(self):
        state = _make_state()
        state["chapter_results"] = [
            {"chapter_id": "ch_01", "chapter_title": "Chapter 1", "merged_text": "   "},
        ]
        llm = MagicMock()
        _run_polish(state, llm)
        # LLM should not have been called for empty chapter
        llm.structured.assert_not_called()


# ---------------------------------------------------------------------------
# Stage 2 topic analysis tests
# ---------------------------------------------------------------------------

class TestStage2TopicAnalysis(unittest.TestCase):
    """Verify that _analyze_topic output is injected into the outline prompt context."""

    def _make_dual_llm(self):
        """LLM that handles both topic-analysis and outline calls."""
        received = []

        class _DualLLM:
            def structured(self, *, system, user, route, max_tokens):
                payload = json.loads(user)
                received.append(payload)
                if "Analyze" in system:
                    return {
                        "research_gaps": ["gap1"],
                        "core_concepts": ["BFT"],
                        "methodology_domain": "systems",
                        "suggested_contributions": ["contrib1"],
                    }
                # outline call
                return {
                    "chapters": [
                        {
                            "chapter_id": "ch_01",
                            "chapter_title": "Introduction",
                            "target_words": 2000,
                            "complexity": "low",
                            "subsections": [{"title": "Background"}],
                        }
                    ]
                }

        return _DualLLM(), received

    def test_topic_analysis_present_in_outline_context(self):
        llm, received = self._make_dual_llm()
        state = _make_state()
        state["literature_summary"] = "Some literature."
        node = build_outline_node(None, _OutlineServices(llm))
        result = node(
            {
                "topic": state["topic"],
                "discipline": state["discipline"],
                "language": state["language"],
                "literature_summary": state["literature_summary"],
            }
        )

        # The outline call payload should contain topic_analysis
        outline_calls = [c for c in received if "topic_analysis" in c]
        self.assertTrue(len(outline_calls) > 0, "No outline call with topic_analysis found")
        ta = outline_calls[0]["topic_analysis"]
        self.assertIn("research_gaps", ta)
        self.assertIn("methodology_domain", ta)
        self.assertEqual(ta["methodology_domain"], "systems")
        self.assertTrue(result["chapter_plans"])

    def test_topic_analysis_failure_uses_safe_defaults(self):
        """If _analyze_topic LLM call fails, outline still proceeds."""

        class _FailingAnalyzeLLM:
            def structured(self, *, system, user, route, max_tokens):
                if "Analyze" in system:
                    raise RuntimeError("LLM down")
                return {
                    "chapters": [
                        {
                            "chapter_id": "ch_01",
                            "chapter_title": "Introduction",
                            "target_words": 2000,
                            "complexity": "low",
                            "subsections": [],
                        }
                    ]
                }

        state = _make_state()
        state["literature_summary"] = "Some literature."
        # Should not raise even if topic analysis fails
        node = build_outline_node(None, _OutlineServices(_FailingAnalyzeLLM()))
        result = node(
            {
                "topic": state["topic"],
                "discipline": state["discipline"],
                "language": state["language"],
                "literature_summary": state["literature_summary"],
            }
        )
        self.assertTrue(len(result["chapter_plans"]) > 0)

    def test_analyze_topic_returns_default_when_research_gaps_empty(self):
        """LLM returning empty research_gaps list should still be accepted (not fall back)."""
        class _EmptyGapsLLM:
            def structured(self, *, system, user, route, max_tokens):
                return {
                    "research_gaps": [],
                    "core_concepts": ["BFT"],
                    "methodology_domain": "systems",
                    "suggested_contributions": [],
                }

        result = _analyze_topic(_EmptyGapsLLM(), "BFT", "CS", "summary")
        # research_gaps key is present → should return the dict, not the default
        self.assertEqual(result["methodology_domain"], "systems")
        self.assertEqual(result["core_concepts"], ["BFT"])
