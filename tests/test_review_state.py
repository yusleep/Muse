"""Tests for review-state stall detection helpers."""

from __future__ import annotations

import hashlib
import unittest

from muse.graph.helpers.review_state import should_iterate


class ReviewStateTests(unittest.TestCase):
    def test_should_iterate_stops_when_score_stalls_after_second_round(self):
        state = {
            "quality_scores": {"coherence": 2, "logic": 3},
            "current_iteration": 2,
            "max_iterations": 4,
        }

        route = should_iterate(
            state,
            threshold=4,
            previous_min_score=2,
        )

        self.assertEqual(route, "done")

    def test_should_iterate_keeps_revising_when_score_improves(self):
        state = {
            "quality_scores": {"coherence": 3, "logic": 4},
            "current_iteration": 2,
            "max_iterations": 4,
        }

        route = should_iterate(
            state,
            threshold=4,
            previous_min_score=2,
        )

        self.assertEqual(route, "revise")

    def test_should_iterate_stops_when_text_hash_is_unchanged(self):
        text = "Draft text that did not change."
        state = {
            "quality_scores": {"coherence": 3, "logic": 4},
            "current_iteration": 1,
            "max_iterations": 4,
        }

        route = should_iterate(
            state,
            threshold=4,
            previous_text_hash=hashlib.md5(text.encode()).hexdigest(),
            current_text=text,
        )

        self.assertEqual(route, "done")

    def test_should_iterate_remains_backward_compatible_without_stall_inputs(self):
        state = {
            "quality_scores": {"coherence": 3, "logic": 4},
            "current_iteration": 1,
            "max_iterations": 4,
        }

        self.assertEqual(should_iterate(state, threshold=4), "revise")


if __name__ == "__main__":
    unittest.main()
