"""Tests for the ReAct-based chapter subgraph."""

from __future__ import annotations

import unittest


class _FakeAgentLLM:
    """Simulates an agent-capable LLM used by the chapter subgraph."""

    def structured(self, *, system, user, route="default", max_tokens=2500):
        if "Write one thesis subsection" in system:
            return {
                "text": "Drafted subsection on graph orchestration.",
                "citations_used": ["@smith2024graph"],
                "key_claims": ["Graph orchestration improves reliability."],
                "transition_out": "",
                "glossary_additions": {},
                "self_assessment": {
                    "confidence": 0.8,
                    "weak_spots": [],
                    "needs_revision": False,
                },
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
        return {}


class _FakeServices:
    def __init__(self):
        self.llm = _FakeAgentLLM()
        self.rag_index = None
        self.search = None


class ChapterReActAgentTests(unittest.TestCase):
    def test_build_chapter_agent_returns_callable(self):
        from muse.graph.subgraphs.chapter import build_chapter_subgraph_node

        node_fn = build_chapter_subgraph_node(services=_FakeServices())
        self.assertTrue(callable(node_fn))

    def test_build_chapter_agent_accepts_optional_settings(self):
        from muse.graph.subgraphs.chapter import build_chapter_subgraph_node

        node_fn = build_chapter_subgraph_node(
            services=_FakeServices(),
            settings=object(),
        )
        self.assertTrue(callable(node_fn))

    def test_chapter_state_schema_has_required_fields(self):
        from muse.graph.subgraphs.chapter import ChapterState

        hints = ChapterState.__annotations__
        self.assertIn("chapter_plan", hints)
        self.assertIn("merged_text", hints)
        self.assertIn("quality_scores", hints)

    def test_chapter_agent_system_prompt_exists(self):
        from muse.prompts.chapter_agent import chapter_agent_system_prompt

        prompt = chapter_agent_system_prompt(
            topic="LangGraph thesis automation",
            language="zh",
            chapter_title="Introduction",
            chapter_plan={"chapter_id": "ch_01", "subtask_plan": []},
            references_summary="5 references available",
        )
        self.assertIn("chapter", prompt.lower())
        self.assertIn("submit", prompt.lower())


if __name__ == "__main__":
    unittest.main()
