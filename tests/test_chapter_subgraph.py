import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class _DraftReviewLLM:
    def __init__(self):
        self.review_calls = 0

    def structured(self, *, system, user, route="default", max_tokens=2500):
        if "Write one thesis subsection with citations" in system:
            return {
                "text": "Drafted subsection content.",
                "citations_used": ["@smith2024graph"],
                "key_claims": ["Graph orchestration improves durability."],
                "transition_out": "",
                "glossary_additions": {},
                "self_assessment": {"confidence": 0.6, "weak_spots": ["transition"], "needs_revision": True},
            }

        if "strict thesis reviewer" in system:
            self.review_calls += 1
            score = 3 if self.review_calls == 1 else 4
            return {
                "scores": {
                    "coherence": score,
                    "logic": 4,
                    "citation": 4,
                    "term_consistency": 4,
                    "balance": 4,
                    "redundancy": 4,
                },
                "review_notes": [
                    {
                        "subtask_id": "sub_01",
                        "issue": "衔接不足",
                        "instruction": "补充过渡段。",
                        "severity": 2,
                    }
                ] if self.review_calls == 1 else [],
            }

        raise AssertionError(f"unexpected prompt: {system}")


class _Services:
    def __init__(self):
        self.llm = _DraftReviewLLM()
        self.rag_index = None
        self.search = None
        self.subagent_executor = None


class _RecoveryLLM:
    def structured(self, *, system, user, route="default", max_tokens=2500):
        del route, max_tokens
        if "Write one thesis subsection" not in system:
            raise AssertionError(f"unexpected prompt: {system}")
        payload = json.loads(user)
        title = payload["subtask"]["title"]
        return {
            "text": f"{title} generated content.",
            "citations_used": ["@smith2024graph"],
            "key_claims": [f"{title} claim."],
            "transition_out": "",
            "glossary_additions": {},
            "self_assessment": {"confidence": 0.8, "weak_spots": [], "needs_revision": False},
        }


class _RecoveryServices:
    def __init__(self):
        self.llm = _RecoveryLLM()
        self.rag_index = None
        self.search = None
        self.subagent_executor = None


class _SubmittedThenCrashAgent:
    def invoke(self, agent_input, config, **kwargs):
        del agent_input, config, kwargs
        from muse.tools.orchestration import submit_result

        submit_result.invoke(
            {
                "result_json": json.dumps(
                    {
                        "merged_text": "Submitted chapter content.",
                        "quality_scores": {"coherence": 4},
                        "subtask_results": [
                            {
                                "subtask_id": "sub_01",
                                "title": "Research Background",
                                "output_text": "Submitted chapter content.",
                                "actual_words": 3,
                                "citations_used": [],
                                "key_claims": [],
                            }
                        ],
                        "citation_uses": [],
                        "claim_text_by_id": {},
                        "iterations_used": 1,
                    },
                    ensure_ascii=False,
                ),
                "summary": "submitted before crash",
            }
        )
        raise RuntimeError("boom after submit")


class _PartialWriteThenCrashAgent:
    def invoke(self, agent_input, config, **kwargs):
        del config, kwargs
        from muse.tools.writing import write_section

        subtask = agent_input["chapter_plan"]["subtask_plan"][0]
        write_section.func(
            chapter_title=agent_input["chapter_plan"]["chapter_title"],
            subtask_id=subtask["subtask_id"],
            subtask_title=subtask["title"],
            target_words=subtask["target_words"],
            topic=agent_input["topic"],
            language=agent_input["language"],
            references_json=json.dumps(agent_input["references"], ensure_ascii=False),
            runtime=None,
        )
        raise RuntimeError("boom during react loop")


class _WriteAndSubmitAgent:
    def invoke(self, agent_input, config, **kwargs):
        del config, kwargs
        from muse.tools.orchestration import submit_result
        from muse.tools.writing import write_section

        subtask = agent_input["chapter_plan"]["subtask_plan"][0]
        tool_result = json.loads(
            write_section.func(
                chapter_title=agent_input["chapter_plan"]["chapter_title"],
                subtask_id=subtask["subtask_id"],
                subtask_title=subtask["title"],
                target_words=subtask["target_words"],
                topic=agent_input["topic"],
                language=agent_input["language"],
                references_json=json.dumps(agent_input["references"], ensure_ascii=False),
                runtime=None,
            )
        )
        submit_result.invoke(
            {
                "result_json": json.dumps(
                    {
                        "merged_text": tool_result["text"],
                        "quality_scores": {"coherence": 4},
                        "subtask_results": [
                            {
                                "subtask_id": subtask["subtask_id"],
                                "title": subtask["title"],
                                "output_text": tool_result["text"],
                                "actual_words": len(tool_result["text"].split()),
                                "citations_used": tool_result["citations_used"],
                                "key_claims": tool_result["key_claims"],
                            }
                        ],
                        "citation_uses": [],
                        "claim_text_by_id": {},
                        "iterations_used": 1,
                    },
                    ensure_ascii=False,
                ),
                "summary": "done",
            }
        )
        return {"messages": []}


class _CrashNoProgressAgent:
    def invoke(self, agent_input, config, **kwargs):
        del agent_input, config, kwargs
        raise RuntimeError("hard failure")


class _ManualPartialThenCrashAgent:
    def invoke(self, agent_input, config, **kwargs):
        del agent_input, config, kwargs
        from muse.tools.orchestration import append_partial_subtask_result

        append_partial_subtask_result(
            {
                "subtask_id": "sub_01",
                "title": "Background",
                "target_words": 500,
                "output_text": "Background generated content.",
                "actual_words": 3,
                "citations_used": ["@smith2024graph"],
                "key_claims": ["Background claim."],
                "confidence": 0.3,
                "weak_spots": ["transition"],
                "needs_revision": True,
            }
        )
        raise RuntimeError("boom after partial write")


class ChapterSubgraphTests(unittest.TestCase):
    def test_references_summary_lists_all_ref_ids_but_only_top_20_summaries(self):
        from muse.graph.subgraphs.chapter import _references_summary

        references = [
            {
                "ref_id": f"@ref{i:02d}",
                "title": f"Reference Title {i}",
                "year": 2020 + (i % 5),
            }
            for i in range(1, 51)
        ]

        summary = _references_summary(references)

        self.assertIn("50 references available.", summary)
        self.assertIn("All ref_ids:", summary)
        self.assertIn("@ref01", summary)
        self.assertIn("@ref50", summary)
        self.assertIn("Top 20 summaries:", summary)
        self.assertIn("- @ref20:", summary)
        self.assertNotIn("- @ref21:", summary)

    def test_references_summary_handles_empty_list(self):
        from muse.graph.subgraphs.chapter import _references_summary

        self.assertEqual(_references_summary([]), "0 references available.")

    def test_assemble_chapter_result_orders_subtasks_and_builds_claim_links(self):
        from muse.graph.subgraphs.chapter import _assemble_chapter_result

        result = _assemble_chapter_result(
            [
                {
                    "subtask_id": "sub_02",
                    "title": "Methods",
                    "output_text": "Methods content.",
                    "actual_words": 2,
                    "citations_used": ["@smith2024graph"],
                    "key_claims": ["Methods claim."],
                },
                {
                    "subtask_id": "sub_01",
                    "title": "Background",
                    "output_text": "Background content.",
                    "actual_words": 2,
                    "citations_used": ["@smith2024graph"],
                    "key_claims": ["Background claim."],
                },
            ],
            {
                "chapter_plan": {
                    "chapter_id": "ch_01",
                    "chapter_title": "绪论",
                    "subtask_plan": [
                        {"subtask_id": "sub_01", "title": "Background", "target_words": 1200},
                        {"subtask_id": "sub_02", "title": "Methods", "target_words": 1200},
                    ],
                },
            }["chapter_plan"],
        )

        chapter = result["chapters"]["ch_01"]
        self.assertEqual(
            [item["subtask_id"] for item in chapter["subtask_results"]],
            ["sub_01", "sub_02"],
        )
        self.assertIn("ch_01_sub_01_c01", chapter["claim_text_by_id"])
        self.assertEqual(chapter["citation_uses"][0]["chapter_id"], "ch_01")

    def test_chapter_subgraph_prefers_submitted_result_when_agent_crashes(self):
        from muse.graph.subgraphs.chapter import build_chapter_subgraph_node

        with patch(
            "muse.graph.subgraphs.chapter._build_react_chapter_agent",
            return_value=_SubmittedThenCrashAgent(),
        ):
            node_fn = build_chapter_subgraph_node(services=_RecoveryServices(), settings=object())

        result = node_fn(
            {
                "chapter_plan": {
                    "chapter_id": "ch_01",
                    "chapter_title": "Introduction",
                    "subtask_plan": [
                        {"subtask_id": "sub_01", "title": "Research Background", "target_words": 500}
                    ],
                },
                "references": [],
                "topic": "Test",
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
        )

        self.assertEqual(result["chapters"]["ch_01"]["merged_text"], "Submitted chapter content.")

    def test_chapter_subgraph_recovers_missing_subtasks_from_partial_results(self):
        from muse.graph.subgraphs.chapter import build_chapter_subgraph_node

        with patch(
            "muse.graph.subgraphs.chapter._build_react_chapter_agent",
            return_value=_PartialWriteThenCrashAgent(),
        ):
            node_fn = build_chapter_subgraph_node(services=_RecoveryServices(), settings=object())

        result = node_fn(
            {
                "chapter_plan": {
                    "chapter_id": "ch_01",
                    "chapter_title": "Introduction",
                    "subtask_plan": [
                        {"subtask_id": "sub_01", "title": "Background", "target_words": 500},
                        {"subtask_id": "sub_02", "title": "Methods", "target_words": 500},
                    ],
                },
                "references": [{"ref_id": "@smith2024graph", "title": "Graph Systems", "year": 2024}],
                "topic": "Test",
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
        )

        chapter = result["chapters"]["ch_01"]
        self.assertEqual(len(chapter["subtask_results"]), 2)
        self.assertEqual(
            [item["subtask_id"] for item in chapter["subtask_results"]],
            ["sub_01", "sub_02"],
        )
        self.assertIn("Background generated content.", chapter["merged_text"])
        self.assertIn("Methods generated content.", chapter["merged_text"])
        self.assertTrue(all(item["needs_revision"] for item in chapter["subtask_results"]))

    def test_chapter_subgraph_raises_when_no_recoverable_progress_exists(self):
        from muse.graph.subgraphs.chapter import ChapterAgentExecutionError, build_chapter_subgraph_node

        with patch(
            "muse.graph.subgraphs.chapter._build_react_chapter_agent",
            return_value=_CrashNoProgressAgent(),
        ):
            node_fn = build_chapter_subgraph_node(services=_RecoveryServices(), settings=object())

        with self.assertRaises(ChapterAgentExecutionError):
            node_fn(
                {
                    "chapter_plan": {
                        "chapter_id": "ch_01",
                        "chapter_title": "Introduction",
                        "subtask_plan": [
                            {"subtask_id": "sub_01", "title": "Background", "target_words": 500}
                        ],
                    },
                    "references": [],
                    "topic": "Test",
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
            )

    def test_chapter_subgraph_clears_partial_accumulator_after_success(self):
        from muse.graph.subgraphs.chapter import build_chapter_subgraph_node
        from muse.tools.orchestration import get_partial_subtask_results

        with patch(
            "muse.graph.subgraphs.chapter._build_react_chapter_agent",
            return_value=_WriteAndSubmitAgent(),
        ):
            node_fn = build_chapter_subgraph_node(services=_RecoveryServices(), settings=object())

        node_fn(
            {
                "chapter_plan": {
                    "chapter_id": "ch_01",
                    "chapter_title": "Introduction",
                    "subtask_plan": [
                        {"subtask_id": "sub_01", "title": "Background", "target_words": 500}
                    ],
                },
                "references": [{"ref_id": "@smith2024graph", "title": "Graph Systems", "year": 2024}],
                "topic": "Test",
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
        )

        self.assertEqual(get_partial_subtask_results(), [])

    def test_chapter_subgraph_recovers_missing_subtasks_from_settings_backed_writer(self):
        from muse.graph.subgraphs.chapter import build_chapter_subgraph_node

        services = SimpleNamespace(
            llm=None,
            rag_index=None,
            search=None,
            subagent_executor=None,
        )

        with patch(
            "muse.graph.subgraphs.chapter._build_react_chapter_agent",
            return_value=_ManualPartialThenCrashAgent(),
        ), patch(
            "muse.graph.subgraphs.chapter._create_react_model",
            return_value=SimpleNamespace(llm_client=_RecoveryLLM()),
        ):
            node_fn = build_chapter_subgraph_node(services=services, settings=object())
            result = node_fn(
                {
                    "chapter_plan": {
                        "chapter_id": "ch_01",
                        "chapter_title": "Introduction",
                        "subtask_plan": [
                            {"subtask_id": "sub_01", "title": "Background", "target_words": 500},
                            {"subtask_id": "sub_02", "title": "Methods", "target_words": 500},
                        ],
                    },
                    "references": [{"ref_id": "@smith2024graph", "title": "Graph Systems", "year": 2024}],
                    "topic": "Test",
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
            )

        chapter = result["chapters"]["ch_01"]
        self.assertEqual(
            [item["subtask_id"] for item in chapter["subtask_results"]],
            ["sub_01", "sub_02"],
        )
        self.assertIn("Methods generated content.", chapter["merged_text"])


if __name__ == "__main__":
    unittest.main()
