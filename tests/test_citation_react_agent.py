"""Tests for the ReAct-based citation subgraph."""

from __future__ import annotations

import unittest


class _FakeNLILLM:
    def entailment(self, *, premise, hypothesis, route="reasoning"):
        return "entailment"

    def structured(self, *, system, user, route="default", max_tokens=2500):
        return {}


class _FakeMetadata:
    def verify_doi(self, doi):
        return True

    def crosscheck_metadata(self, ref):
        return True


class _FakeServices:
    def __init__(self):
        self.llm = _FakeNLILLM()
        self.metadata = _FakeMetadata()
        self.search = None
        self.rag_index = None


class CitationReActTests(unittest.TestCase):
    def test_build_citation_subgraph_node_returns_callable(self):
        from muse.graph.subgraphs.citation import build_citation_subgraph_node

        fn = build_citation_subgraph_node(services=_FakeServices())
        self.assertTrue(callable(fn))

    def test_build_citation_subgraph_node_accepts_optional_settings(self):
        from muse.graph.subgraphs.citation import build_citation_subgraph_node

        fn = build_citation_subgraph_node(
            services=_FakeServices(),
            settings=object(),
        )
        self.assertTrue(callable(fn))

    def test_citation_graph_fixed_flow_still_works(self):
        from muse.graph.subgraphs.citation import build_citation_graph

        graph = build_citation_graph(services=_FakeServices())
        result = graph.invoke(
            {
                "references": [
                    {
                        "ref_id": "@a",
                        "title": "Paper A",
                        "doi": "10.1/a",
                        "authors": ["A"],
                        "year": 2024,
                    }
                ],
                "citation_uses": [{"cite_key": "@a", "claim_id": "c1"}],
                "claim_text_by_id": {"c1": "Claim text."},
            }
        )
        self.assertIn("citation_ledger", result)

    def test_citation_agent_system_prompt_exists(self):
        from muse.prompts.citation_agent import citation_agent_system_prompt

        prompt = citation_agent_system_prompt(
            total_citations=5,
            total_claims=10,
            references_summary="5 references",
        )
        self.assertIn("citation", prompt.lower())
        self.assertIn("submit", prompt.lower())


if __name__ == "__main__":
    unittest.main()
