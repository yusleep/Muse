"""Tests for review-node self-assessment integration."""

from __future__ import annotations

import json
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

    def test_global_review_node_persona_mode_uses_judge_route_and_filters_dimensions(self):
        from muse.graph.nodes.review import build_global_review_node

        class _PersonaJudgeLLM:
            def __init__(self):
                self.calls = []

            def structured(self, *, system, user, route="default", max_tokens=2500):
                del max_tokens
                self.calls.append({"system": system, "user": user, "route": route})
                if len(self.calls) == 1:
                    return {
                        "scores": {"logic": 4, "structure": 3, "balance": 2, "citation": 1},
                        "review_notes": [
                            {
                                "section": "Introduction",
                                "severity": 3,
                                "instruction": "Clarify the core argument.",
                                "lens": "logic",
                            }
                        ],
                    }
                if len(self.calls) == 2:
                    return {
                        "scores": {"citation": 5, "coverage": 4, "depth": 3, "style": 1},
                        "review_notes": [
                            {
                                "section": "Related Work",
                                "severity": 2,
                                "instruction": "Add the missing baseline citation.",
                                "lens": "citation",
                            }
                        ],
                    }
                if len(self.calls) == 3:
                    return {
                        "scores": {"style": 4, "term_consistency": 5, "redundancy": 2, "logic": 1},
                        "review_notes": [
                            {
                                "section": "Conclusion",
                                "severity": 2,
                                "instruction": "Reduce repetitive phrasing.",
                                "lens": "readability",
                            }
                        ],
                    }
                assert route == "review_judge"
                packets = json.loads(user)
                assert set(packets[0]["result"]["scores"].keys()) == {"logic", "structure", "balance"}
                assert set(packets[1]["result"]["scores"].keys()) == {"citation", "coverage", "depth"}
                assert set(packets[2]["result"]["scores"].keys()) == {
                    "style",
                    "term_consistency",
                    "redundancy",
                }
                return {
                    "final_scores": {
                        "logic": 4,
                        "structure": 3,
                        "balance": 2,
                        "citation": 5,
                        "coverage": 4,
                        "depth": 3,
                        "style": 4,
                        "term_consistency": 5,
                        "redundancy": 2,
                    },
                    "unified_notes": [
                        {
                            "section": "Introduction",
                            "severity": 3,
                            "instruction": "Clarify the core argument.",
                            "lens": "judge",
                        }
                    ],
                    "conflicts_resolved": [
                        {"topic": "severity", "ruling": "Accepted the stricter critique for logic."}
                    ],
                }

        llm = _PersonaJudgeLLM()

        services = type("_GlobalServices", (), {"llm": llm})()
        node = build_global_review_node(services, mode="persona")
        result = node({"final_text": "Merged thesis draft."})

        self.assertEqual(len(llm.calls), 4)
        self.assertEqual(llm.calls[-1]["route"], "review_judge")
        self.assertEqual(len(result["quality_scores"]), 9)
        self.assertEqual(result["review_notes"][0]["lens"], "judge")
        self.assertEqual(result["review_iteration"], 2)

    def test_merge_persona_results_detects_divergent_reviewer_strictness(self):
        from muse.graph.nodes.review import _merge_persona_results

        merged = _merge_persona_results(
            [
                {
                    "persona": "logic",
                    "result": {"scores": {"logic": 1, "structure": 2, "balance": 1}, "review_notes": []},
                },
                {
                    "persona": "citation",
                    "result": {"scores": {"citation": 5, "coverage": 4, "depth": 4}, "review_notes": []},
                },
                {
                    "persona": "readability",
                    "result": {
                        "scores": {"style": 4, "term_consistency": 4, "redundancy": 5},
                        "review_notes": [],
                    },
                },
            ]
        )

        self.assertEqual(len(merged["final_scores"]), 9)
        self.assertTrue(merged["conflicts_resolved"])

    def test_global_review_node_persona_mode_falls_back_when_judge_fails(self):
        from muse.graph.nodes.review import build_global_review_node

        class _JudgeFailureLLM:
            def __init__(self):
                self.count = 0

            def structured(self, *, system, user, route="default", max_tokens=2500):
                del system, user, max_tokens
                self.count += 1
                if route == "review_judge":
                    raise RuntimeError("judge unavailable")
                if self.count == 1:
                    return {
                        "scores": {"logic": 2, "structure": 2, "balance": 3},
                        "review_notes": [{"section": "Intro", "severity": 4, "instruction": "Fix logic gaps."}],
                    }
                if self.count == 2:
                    return {
                        "scores": {"citation": 4, "coverage": 4, "depth": 3},
                        "review_notes": [{"section": "Related Work", "severity": 5, "instruction": "Add citations."}],
                    }
                return {
                    "scores": {"style": 3, "term_consistency": 2, "redundancy": 1},
                    "review_notes": [{"section": "Conclusion", "severity": 2, "instruction": "Trim repetition."}],
                }

        llm = _JudgeFailureLLM()

        services = type("_GlobalServices", (), {"llm": llm})()
        node = build_global_review_node(services, mode="persona")
        result = node({"final_text": "Merged thesis draft."})

        self.assertEqual(result["quality_scores"]["logic"], 2)
        self.assertEqual(result["quality_scores"]["citation"], 4)
        self.assertEqual(result["quality_scores"]["style"], 3)
        self.assertEqual(result["review_notes"][0]["severity"], 5)
        self.assertEqual(result["review_history"][0]["note_count"], 3)


if __name__ == "__main__":
    unittest.main()
