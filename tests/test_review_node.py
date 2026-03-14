"""Tests for review-node self-assessment integration."""

from __future__ import annotations

import unittest


class _ReviewLLM:
    def structured(self, *, system, user, route="default", max_tokens=2500):
        del system, user, route, max_tokens
        return {"scores": {"logic": 4}, "review_notes": []}


class _Services:
    llm = _ReviewLLM()


class ReviewNodeTests(unittest.TestCase):
    def test_build_self_assessment_notes_flags_low_confidence_subtasks(self):
        from muse.graph.nodes.review import _build_self_assessment_notes

        notes = _build_self_assessment_notes(
            {
                "chapters": {
                    "ch_01": {
                        "subtask_results": [
                            {
                                "subtask_id": "sub_01",
                                "confidence": 0.3,
                                "weak_spots": ["transition", "evidence"],
                                "needs_revision": True,
                            }
                        ]
                    }
                }
            }
        )

        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["subtask_id"], "sub_01")
        self.assertEqual(notes[0]["lens"], "self_assessment")
        self.assertIn("confidence=0.30", notes[0]["instruction"])
        self.assertIn("transition", notes[0]["instruction"])

    def test_build_self_assessment_notes_ignores_confident_subtasks(self):
        from muse.graph.nodes.review import _build_self_assessment_notes

        notes = _build_self_assessment_notes(
            {
                "chapters": {
                    "ch_01": {
                        "subtask_results": [
                            {
                                "subtask_id": "sub_01",
                                "confidence": 0.8,
                                "weak_spots": ["none"],
                                "needs_revision": False,
                            }
                        ]
                    }
                }
            }
        )

        self.assertEqual(notes, [])

    def test_chapter_review_node_is_unchanged_when_self_assessment_missing(self):
        from muse.graph.nodes.review import build_chapter_review_node

        node = build_chapter_review_node(_Services())
        result = node(
            {
                "chapter_plan": {"chapter_title": "绪论"},
                "merged_text": "已有草稿。",
            }
        )

        self.assertEqual(result["review_notes"], [])
        self.assertEqual(result["revision_instructions"], {})


if __name__ == "__main__":
    unittest.main()
