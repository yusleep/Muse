import unittest


class ReflectionBankTests(unittest.TestCase):
    def test_add_reflection_extracts_positive_and_regression_entries(self):
        from muse.graph.helpers.reflection_bank import ReflectionBank

        bank = ReflectionBank()
        bank.add_reflection(
            review_history=[
                {
                    "iteration": 1,
                    "scores": {"logic": 2, "citation": 4},
                    "top_instructions": [
                        "Clarify the core argument before introducing implementation details.",
                        "Connect the problem statement to the research gap.",
                    ],
                },
                {
                    "iteration": 2,
                    "scores": {"logic": 4, "citation": 2},
                    "top_instructions": [
                        "Add stronger support for the method comparison.",
                    ],
                },
            ],
            chapter_id="ch_01",
        )

        entries = bank.to_dict()["entries"]
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["outcome"], "positive")
        self.assertEqual(entries[0]["dimension"], "logic")
        self.assertIn("Clarify the core argument", entries[0]["instruction"])
        self.assertEqual(entries[1]["outcome"], "regression")
        self.assertEqual(entries[1]["dimension"], "citation")

    def test_get_relevant_reflections_prefers_positive_entries_for_weak_dimensions(self):
        from muse.graph.helpers.reflection_bank import ReflectionBank

        bank = ReflectionBank.from_dict(
            {
                "entries": [
                    {
                        "chapter_id": "ch_01",
                        "dimension": "logic",
                        "outcome": "positive",
                        "instruction": "Clarify the core argument before implementation details.",
                        "score_delta": 2,
                    },
                    {
                        "chapter_id": "ch_01",
                        "dimension": "citation",
                        "outcome": "regression",
                        "instruction": "Add stronger support for the method comparison.",
                        "score_delta": -2,
                    },
                ]
            }
        )

        reflections = bank.get_relevant_reflections(
            {"logic": 2, "citation": 2},
            max_reflections=3,
        )

        self.assertEqual(len(reflections), 1)
        self.assertEqual(reflections[0]["dimension"], "logic")

    def test_get_writing_tips_returns_distilled_positive_tips(self):
        from muse.graph.helpers.reflection_bank import ReflectionBank

        bank = ReflectionBank.from_dict(
            {
                "entries": [
                    {
                        "chapter_id": "ch_01",
                        "dimension": "logic",
                        "outcome": "positive",
                        "instruction": "Clarify the core argument before implementation details.",
                        "score_delta": 2,
                    }
                ]
            }
        )

        tips = bank.get_writing_tips(max_tips=2)

        self.assertEqual(len(tips), 1)
        self.assertIn("logic", tips[0].lower())
        self.assertIn("Clarify the core argument", tips[0])


if __name__ == "__main__":
    unittest.main()
