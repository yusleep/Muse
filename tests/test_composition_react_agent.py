"""Tests for the ReAct-based composition subgraph."""

from __future__ import annotations

import unittest


class CompositionReActTests(unittest.TestCase):
    def test_build_composition_subgraph_node_returns_callable(self):
        from muse.graph.subgraphs.composition import build_composition_subgraph_node

        fn = build_composition_subgraph_node()
        self.assertTrue(callable(fn))

    def test_build_composition_subgraph_node_accepts_optional_settings(self):
        from muse.graph.subgraphs.composition import build_composition_subgraph_node

        fn = build_composition_subgraph_node(settings=object(), services=object())
        self.assertTrue(callable(fn))

    def test_composition_graph_fixed_flow_still_works(self):
        from muse.graph.subgraphs.composition import build_composition_graph

        graph = build_composition_graph()
        result = graph.invoke(
            {
                "final_text": "Chapter 1 text. Chapter 2 text.",
                "abstract_zh": "摘要",
                "abstract_en": "Abstract",
                "paper_package": {},
            }
        )
        self.assertTrue(result.get("paper_package", {}).get("terminology_normalized"))
        self.assertTrue(result.get("paper_package", {}).get("cross_refs_aligned"))

    def test_composition_agent_system_prompt_exists(self):
        from muse.prompts.composition_agent import composition_agent_system_prompt

        prompt = composition_agent_system_prompt(
            chapter_count=5,
            total_words=25000,
            language="zh",
        )
        self.assertIn("composition", prompt.lower())
        self.assertIn("submit", prompt.lower())


if __name__ == "__main__":
    unittest.main()
