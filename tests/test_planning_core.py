import unittest

from thesis_agent.planning import plan_subtasks
from thesis_agent.chapter import should_iterate


class PlanSubtasksTests(unittest.TestCase):
    def test_allocates_at_least_two_subtasks_with_word_budget_in_range(self):
        subsections = [
            {"title": "背景", "relevant_refs": ["@a"]},
            {"title": "问题", "relevant_refs": ["@b"]},
            {"title": "方法", "relevant_refs": ["@c"]},
        ]

        subtasks = plan_subtasks(target_words=3200, complexity="low", subsections=subsections)

        self.assertGreaterEqual(len(subtasks), 2)
        for subtask in subtasks:
            self.assertGreaterEqual(subtask["target_words"], 1000)
            self.assertLessEqual(subtask["target_words"], 2000)

    def test_high_complexity_uses_smaller_chunks(self):
        subsections = [
            {"title": f"section-{i}", "relevant_refs": [f"@r{i}"]} for i in range(1, 8)
        ]

        subtasks = plan_subtasks(target_words=7000, complexity="high", subsections=subsections)

        # high complexity should split more aggressively than medium/low defaults
        self.assertGreaterEqual(len(subtasks), 5)


class IterationRoutingTests(unittest.TestCase):
    def test_done_when_min_score_meets_threshold(self):
        state = {
            "quality_scores": {
                "coherence": 4,
                "logic": 4,
                "citation": 5,
                "term_consistency": 4,
                "balance": 4,
                "redundancy": 4,
            },
            "current_iteration": 1,
            "max_iterations": 3,
        }

        self.assertEqual(should_iterate(state), "done")

    def test_revise_when_score_below_threshold_and_iterations_left(self):
        state = {
            "quality_scores": {
                "coherence": 3,
                "logic": 4,
                "citation": 4,
                "term_consistency": 4,
                "balance": 4,
                "redundancy": 4,
            },
            "current_iteration": 1,
            "max_iterations": 3,
        }

        self.assertEqual(should_iterate(state), "revise")

    def test_done_when_iteration_limit_hit(self):
        state = {
            "quality_scores": {
                "coherence": 2,
                "logic": 3,
                "citation": 2,
                "term_consistency": 3,
                "balance": 3,
                "redundancy": 2,
            },
            "current_iteration": 3,
            "max_iterations": 3,
        }

        self.assertEqual(should_iterate(state), "done")


if __name__ == "__main__":
    unittest.main()
