import unittest


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

    def test_chapter_graph_revises_until_scores_reach_threshold(self):
        from muse.graph.subgraphs.chapter import build_chapter_graph

        graph = build_chapter_graph(services=_Services())
        result = graph.invoke(
            {
                "chapter_plan": {
                    "chapter_id": "ch_01",
                    "chapter_title": "绪论",
                    "subtask_plan": [{"subtask_id": "sub_01", "title": "研究背景", "target_words": 1200}],
                },
                "references": [
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
                "topic": "LangGraph thesis automation",
                "language": "zh",
                "subtask_results": [],
                "merged_text": "",
                "quality_scores": {},
                "review_notes": [],
                "revision_instructions": {},
                "iteration": 0,
                "max_iterations": 3,
                "citation_uses": [],
                "claim_text_by_id": {},
            }
        )

        self.assertEqual(result["iteration"], 2)
        self.assertEqual(result["quality_scores"]["coherence"], 4)
        self.assertIn("Drafted subsection content.", result["merged_text"])
        self.assertTrue(result["citation_uses"])
        self.assertTrue(result["claim_text_by_id"])


if __name__ == "__main__":
    unittest.main()
