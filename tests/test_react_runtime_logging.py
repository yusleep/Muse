"""Regression tests for ReAct runtime logging around chapter/citation subgraphs."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch


class _ChapterLLM:
    def structured(self, *, system, user, route="default", max_tokens=2500):
        del user, route, max_tokens
        if "Write one thesis subsection" in system:
            return {
                "text": "Generated subsection.",
                "citations_used": [],
                "key_claims": [],
                "transition_out": "",
                "glossary_additions": {},
                "self_assessment": {
                    "confidence": 0.9,
                    "weak_spots": [],
                    "needs_revision": False,
                },
            }
        if "strict thesis reviewer" in system:
            return {
                "scores": {
                    "coherence": 5,
                    "logic": 5,
                    "citation": 5,
                    "term_consistency": 5,
                    "balance": 5,
                    "redundancy": 5,
                },
                "review_notes": [],
            }
        return {}

    def text(self, *, system, user, route="default", max_tokens=2500):
        del system, user, route, max_tokens
        return "Generated text."


class _ChapterServices:
    def __init__(self):
        self.llm = _ChapterLLM()
        self.search = None
        self.metadata = None
        self.rag_index = None
        self.local_refs = []
        self.subagent_executor = None


class _ChapterNoSubmitAgent:
    def invoke(self, agent_input, config, **kwargs):
        del agent_input, config, kwargs
        return {
            "messages": [
                {
                    "tool_calls": [
                        {
                            "name": "update_plan",
                            "args": {"progress_pct": 40, "current_step": "drafting"},
                        }
                    ]
                }
            ]
        }


class _CitationServices:
    def __init__(self):
        self.search = None
        self.rag_index = None
        self.local_refs = []
        self.subagent_executor = None

        class _Metadata:
            def verify_doi(self, doi):
                del doi
                return True

            def crosscheck_metadata(self, ref):
                del ref
                return True

        class _LLM:
            def entailment(self, *, premise, hypothesis, route="reasoning"):
                del premise, hypothesis, route
                return "entailment"

        self.metadata = _Metadata()
        self.llm = _LLM()


class _CitationSkipFinalizeAgent:
    def invoke(self, agent_input, config, **kwargs):
        del config, kwargs
        from muse.tools.citation import record_citation_assessment

        worklist = agent_input.get("citation_worklist", [])
        for item in worklist:
            record_citation_assessment.invoke(
                {
                    "cite_key": item["cite_key"],
                    "claim_id": item["claim_id"],
                    "verdict": "verified",
                    "support_score": 0.91,
                    "confidence": "high",
                    "reason": "supported",
                    "detail": "Recorded but not finalized.",
                    "evidence_excerpt": item.get("evidence", ""),
                }
            )
        return {"messages": []}


class ReactRuntimeLoggingTests(unittest.TestCase):
    def _chapter_state(self):
        return {
            "chapter_plan": {
                "chapter_id": "ch_01",
                "chapter_title": "Introduction",
                "subtask_plan": [
                    {
                        "subtask_id": "sub_01",
                        "title": "Background",
                        "target_words": 300,
                    }
                ],
            },
            "references": [],
            "topic": "React citation runtime",
            "discipline": "Computer Science",
            "language": "zh",
            "subtask_results": [],
            "merged_text": "",
            "quality_scores": {},
            "review_notes": [],
            "revision_instructions": {},
            "iteration": 0,
            "max_iterations": 1,
            "citation_uses": [],
            "claim_text_by_id": {},
        }

    def _citation_state(self):
        return {
            "references": [
                {
                    "ref_id": "@a",
                    "title": "Paper A",
                    "doi": "10.1/a",
                    "authors": ["A"],
                    "year": 2024,
                    "abstract": "Paper A supports claim A.",
                }
            ],
            "citation_uses": [{"cite_key": "@a", "claim_id": "c1"}],
            "claim_text_by_id": {"c1": "Claim A."},
            "citation_ledger": {},
            "verified_citations": [],
            "flagged_citations": [],
        }

    def test_chapter_react_logs_missing_submit_fallback_reason(self):
        from muse.graph.subgraphs.chapter import build_chapter_subgraph_node

        with patch(
            "muse.graph.subgraphs.chapter._build_react_chapter_agent",
            return_value=_ChapterNoSubmitAgent(),
        ):
            node_fn = build_chapter_subgraph_node(services=_ChapterServices(), settings=object())

        with self.assertLogs("muse.chapter", level="INFO") as logs:
            result = node_fn(self._chapter_state())

        joined = "\n".join(logs.output)
        self.assertIn("chapter react start chapter_id=ch_01", joined)
        self.assertIn("chapter react invoke_return chapter_id=ch_01", joined)
        self.assertIn("reason=missing_submit_result", joined)
        self.assertIn("chapters", result)

    def test_citation_react_logs_finalize_status_before_validation_error(self):
        from muse.graph.subgraphs.citation import CitationAgentExecutionError, build_citation_subgraph_node

        with patch(
            "muse.graph.subgraphs.citation._build_react_citation_agent",
            return_value=_CitationSkipFinalizeAgent(),
        ):
            node_fn = build_citation_subgraph_node(services=_CitationServices(), settings=object())

        with self.assertLogs("muse.citation", level="INFO") as logs:
            with self.assertRaises(CitationAgentExecutionError):
                node_fn(self._citation_state())

        joined = "\n".join(logs.output)
        self.assertIn("citation react start", joined)
        self.assertIn("citation react invoke_return", joined)
        self.assertIn("finalized=False", joined)


if __name__ == "__main__":
    unittest.main()
