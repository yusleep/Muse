import unittest

from thesis_agent.chapter import apply_chapter_review


class ChapterFlowTests(unittest.TestCase):
    def test_apply_chapter_review_routes_to_revise_with_targeted_instructions(self):
        state = {
            "quality_scores": {},
            "review_notes": [],
            "revision_instructions": {},
            "current_iteration": 0,
            "max_iterations": 3,
        }
        review = {
            "scores": {
                "coherence": 3,
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
                },
                {
                    "subtask_id": "sub_02",
                    "issue": "轻微措辞",
                    "instruction": "替换不规范表述。",
                    "severity": 1,
                },
            ],
        }

        route, updated = apply_chapter_review(state, review, score_threshold=4, min_severity=2)

        self.assertEqual(route, "revise")
        self.assertEqual(updated["current_iteration"], 1)
        self.assertEqual(list(updated["revision_instructions"].keys()), ["sub_01"])
        self.assertEqual(len(updated["review_notes"]), 2)

    def test_apply_chapter_review_routes_done_when_threshold_met(self):
        state = {
            "quality_scores": {},
            "review_notes": [],
            "revision_instructions": {},
            "current_iteration": 1,
            "max_iterations": 3,
        }
        review = {
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

        route, updated = apply_chapter_review(state, review, score_threshold=4, min_severity=2)

        self.assertEqual(route, "done")
        self.assertEqual(updated["current_iteration"], 2)
        self.assertEqual(updated["revision_instructions"], {})


if __name__ == "__main__":
    unittest.main()
