"""Tests for muse/tools/review.py"""

from __future__ import annotations

import json
import unittest


class _FakeReviewLLM:
    def structured(self, *, system, user, route="default", max_tokens=2500):
        return {
            "scores": {
                "coherence": 4,
                "logic": 3,
                "citation": 5,
                "term_consistency": 4,
                "balance": 4,
                "redundancy": 4,
            },
            "review_notes": [
                {
                    "subtask_id": "sub_01",
                    "issue": "Weak logic flow",
                    "instruction": "Add transition.",
                    "severity": 3,
                }
            ],
        }


class SelfReviewToolTests(unittest.TestCase):
    def test_self_review_returns_json_with_scores_and_notes(self):
        from muse.tools.review import self_review

        result_str = self_review.invoke(
            {
                "chapter_title": "Introduction",
                "merged_text": "This is the chapter text.",
                "lenses": "logic,style,citation,structure",
            }
        )
        result = json.loads(result_str)
        self.assertIn("scores", result)
        self.assertIn("review_notes", result)
        self.assertIn("revision_instructions", result)

    def test_self_review_with_single_lens(self):
        from muse.tools.review import self_review

        result_str = self_review.invoke(
            {
                "chapter_title": "Methods",
                "merged_text": "Methods text.",
                "lenses": "logic",
            }
        )
        result = json.loads(result_str)
        self.assertIn("scores", result)


if __name__ == "__main__":
    unittest.main()
