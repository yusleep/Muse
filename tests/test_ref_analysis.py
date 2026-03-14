import unittest


class RefAnalysisNodeTests(unittest.TestCase):
    def test_ref_analysis_generates_briefs_and_gaps_for_each_chapter(self):
        from muse.graph.nodes.ref_analysis import build_ref_analysis_node

        class _CaptureLLM:
            def __init__(self):
                self.calls = 0

            def structured(self, *, system, user, route="default", max_tokens=1500):
                del system, user, route, max_tokens
                self.calls += 1
                if self.calls == 1:
                    return {
                        "key_references": [
                            {
                                "ref_id": "@smith2024",
                                "relevance": "directly addresses method",
                                "key_finding": "Table 2 shows a 15% improvement.",
                                "how_to_cite": "Use as the main evidence for the method choice.",
                            }
                        ],
                        "evidence_gaps": ["No source covers deployment constraints."],
                    }
                return {
                    "key_references": [
                        {
                            "ref_id": "@jones2023",
                            "relevance": "provides background context",
                            "key_finding": "Figure 3 summarizes the system pipeline.",
                            "how_to_cite": "Use as background context.",
                        }
                    ],
                    "evidence_gaps": [],
                }

        services = type("_Services", (), {"llm": _CaptureLLM()})()
        node = build_ref_analysis_node(services=services)

        result = node(
            {
                "chapter_plans": [
                    {
                        "chapter_id": "ch_01",
                        "chapter_title": "绪论",
                        "subtask_plan": [{"title": "研究背景", "description": "问题背景"}],
                    },
                    {
                        "chapter_id": "ch_02",
                        "chapter_title": "方法",
                        "subtask_plan": [{"title": "方法设计", "description": "系统方法"}],
                    },
                ],
                "references": [
                    {"ref_id": "@smith2024", "title": "Graph Systems", "abstract": "Method details."},
                    {"ref_id": "@jones2023", "title": "Pipeline Design", "abstract": "Pipeline background."},
                ],
            }
        )

        self.assertEqual(result["reference_briefs"]["ch_01"][0]["ref_id"], "@smith2024")
        self.assertEqual(result["reference_briefs"]["ch_01_gaps"], ["No source covers deployment constraints."])
        self.assertEqual(result["reference_briefs"]["ch_02"][0]["ref_id"], "@jones2023")

    def test_ref_analysis_returns_empty_when_outline_or_references_missing(self):
        from muse.graph.nodes.ref_analysis import build_ref_analysis_node

        services = type("_Services", (), {"llm": None})()
        node = build_ref_analysis_node(services=services)

        self.assertEqual(node({"chapter_plans": [], "references": []}), {})


if __name__ == "__main__":
    unittest.main()
