"""Tests for the ReAct-based citation subgraph."""

from __future__ import annotations

import unittest
from unittest.mock import patch


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


class _FinalizingReactAgent:
    def __init__(self, *, partial: bool = False, skip_finalize: bool = False):
        self.partial = partial
        self.skip_finalize = skip_finalize

    def invoke(self, agent_input, config, **kwargs):
        del config, kwargs
        from muse.tools.citation import finalize_citation_review, record_citation_assessment

        worklist = agent_input.get("citation_worklist")
        if not isinstance(worklist, list):
            raise AssertionError("citation_worklist missing from agent input")

        items = worklist[:1] if self.partial else worklist
        for item in items:
            record_citation_assessment.invoke(
                {
                    "cite_key": item["cite_key"],
                    "claim_id": item["claim_id"],
                    "verdict": "verified",
                    "support_score": 0.95,
                    "confidence": "high",
                    "reason": "supported",
                    "detail": "Verified against current reference set.",
                    "evidence_excerpt": item.get("evidence", ""),
                }
            )

        if not self.skip_finalize:
            finalize_citation_review.invoke({"summary": "citation review complete"})

        return {"messages": []}


class CitationReActTests(unittest.TestCase):
    def _sample_state(self):
        return {
            "references": [
                {
                    "ref_id": "@a",
                    "title": "Paper A",
                    "doi": "10.1/a",
                    "authors": ["A"],
                    "year": 2024,
                    "abstract": "Paper A supports claim A.",
                },
                {
                    "ref_id": "@b",
                    "title": "Paper B",
                    "doi": "10.1/b",
                    "authors": ["B"],
                    "year": 2024,
                    "abstract": "Paper B supports claim B.",
                },
            ],
            "citation_uses": [
                {"cite_key": "@a", "claim_id": "c1"},
                {"cite_key": "@b", "claim_id": "c2"},
            ],
            "claim_text_by_id": {
                "c1": "Claim A.",
                "c2": "Claim B.",
            },
            "citation_ledger": {},
            "verified_citations": [],
            "flagged_citations": [],
        }

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
        result = graph.invoke(self._sample_state())
        self.assertIn("citation_ledger", result)

    def test_react_agent_toolset_is_citation_only(self):
        from muse.graph.subgraphs.citation import _build_react_citation_agent

        captured: dict[str, object] = {}

        def fake_create_agent(*, model, tools, middleware, state_schema, name):
            captured["tools"] = tools
            captured["middleware"] = middleware
            captured["state_schema"] = state_schema
            captured["name"] = name
            return object()

        with patch("langchain.agents.create_agent", side_effect=fake_create_agent), patch(
            "muse.graph.subgraphs.citation._create_react_model",
            return_value=object(),
        ):
            agent = _build_react_citation_agent(services=_FakeServices(), settings=object())

        self.assertIsNotNone(agent)
        tool_names = [tool.name for tool in captured["tools"]]
        self.assertEqual(
            tool_names,
            [
                "verify_doi",
                "crosscheck_metadata",
                "entailment_check",
                "record_citation_assessment",
                "finalize_citation_review",
            ],
        )
        self.assertEqual(len(captured["middleware"]), 1)

    def test_react_unavailable_raises_instead_of_fallback(self):
        from muse.graph.subgraphs.citation import CitationAgentExecutionError, build_citation_subgraph_node

        with patch(
            "muse.graph.subgraphs.citation._build_react_citation_agent",
            return_value=None,
        ):
            node_fn = build_citation_subgraph_node(services=_FakeServices(), settings=object())

        with self.assertRaises(CitationAgentExecutionError):
            node_fn(self._sample_state())

    def test_react_path_returns_structured_result_after_finalize(self):
        from muse.graph.subgraphs.citation import build_citation_subgraph_node

        with patch(
            "muse.graph.subgraphs.citation._build_react_citation_agent",
            return_value=_FinalizingReactAgent(),
        ):
            node_fn = build_citation_subgraph_node(services=_FakeServices(), settings=object())

        result = node_fn(self._sample_state())
        self.assertEqual(result["verified_citations"], ["@a", "@b"])
        self.assertEqual(result["flagged_citations"], [])
        self.assertIn("c1", result["citation_ledger"])
        self.assertIn("c2", result["citation_ledger"])

    def test_react_path_raises_when_finalize_missing(self):
        from muse.graph.subgraphs.citation import CitationAgentExecutionError, build_citation_subgraph_node

        with patch(
            "muse.graph.subgraphs.citation._build_react_citation_agent",
            return_value=_FinalizingReactAgent(skip_finalize=True),
        ):
            node_fn = build_citation_subgraph_node(services=_FakeServices(), settings=object())

        with self.assertRaises(CitationAgentExecutionError):
            node_fn(self._sample_state())

    def test_react_path_raises_on_partial_coverage(self):
        from muse.graph.subgraphs.citation import CitationAgentExecutionError, build_citation_subgraph_node

        with patch(
            "muse.graph.subgraphs.citation._build_react_citation_agent",
            return_value=_FinalizingReactAgent(partial=True),
        ):
            node_fn = build_citation_subgraph_node(services=_FakeServices(), settings=object())

        with self.assertRaises(CitationAgentExecutionError):
            node_fn(self._sample_state())

    def test_citation_agent_system_prompt_exists(self):
        from muse.prompts.citation_agent import citation_agent_system_prompt

        prompt = citation_agent_system_prompt(
            worklist_json='[{"cite_key":"@a","claim_id":"c1"}]',
            total_citations=1,
            total_claims=1,
            references_summary="1 references",
        )
        self.assertIn("citation", prompt.lower())
        self.assertIn("record_citation_assessment", prompt)
        self.assertIn("finalize_citation_review", prompt)


if __name__ == "__main__":
    unittest.main()
