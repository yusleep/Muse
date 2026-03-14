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

    def test_global_review_node_uses_base_prompt_for_first_iteration(self):
        from muse.graph.nodes.review import build_global_review_node

        seen_systems = []

        class _CaptureLLM:
            def structured(self, *, system, user, route="default", max_tokens=2500):
                del user, route, max_tokens
                seen_systems.append(system)
                return {
                    "scores": {"logic": 4},
                    "review_notes": [
                        {
                            "section": "Introduction",
                            "severity": 2,
                            "instruction": "Tighten the transition into the problem statement.",
                            "lens": "logic",
                            "is_recurring": False,
                        }
                    ],
                }

        class _GlobalServices:
            llm = _CaptureLLM()

        node = build_global_review_node(_GlobalServices())
        result = node({"final_text": "Merged thesis draft."})

        self.assertEqual(len(seen_systems), 4)
        self.assertTrue(all("full merged thesis draft" in system for system in seen_systems))
        self.assertTrue(all("Previous review round" not in system for system in seen_systems))
        self.assertEqual(result["review_iteration"], 2)
        self.assertEqual(result["review_history"][0]["iteration"], 1)
        self.assertIn("problem statement", result["review_history"][0]["notes_summary"])
        self.assertFalse(result["review_notes"][0]["is_recurring"])

    def test_global_review_node_uses_adaptive_prompt_after_first_iteration(self):
        from muse.graph.nodes.review import build_global_review_node

        seen_systems = []

        class _AdaptiveLLM:
            def structured(self, *, system, user, route="default", max_tokens=2500):
                del user, route, max_tokens
                seen_systems.append(system)
                return {
                    "scores": {"logic": 3},
                    "review_notes": [
                        {
                            "section": "Methods",
                            "severity": 4,
                            "instruction": "The evidence gap from the prior draft is still unresolved.",
                            "lens": "logic",
                            "is_recurring": True,
                        }
                    ],
                }

        class _GlobalServices:
            llm = _AdaptiveLLM()

        node = build_global_review_node(_GlobalServices())
        result = node(
            {
                "final_text": "Merged thesis draft, revised.",
                "review_iteration": 2,
                "review_history": [
                    {
                        "iteration": 1,
                        "scores": {"logic": 2},
                        "notes_summary": "The evidence gap from the prior draft is still unresolved.",
                    }
                ],
            }
        )

        self.assertEqual(len(seen_systems), 4)
        self.assertTrue(all("Previous review round (iteration 1)" in system for system in seen_systems))
        self.assertTrue(all('"logic": 2' in system for system in seen_systems))
        self.assertEqual(result["review_iteration"], 3)
        self.assertTrue(result["review_notes"][0]["is_recurring"])
        self.assertEqual(result["review_history"][0]["iteration"], 2)


if __name__ == "__main__":
    unittest.main()
