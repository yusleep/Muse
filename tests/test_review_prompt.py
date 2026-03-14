from __future__ import annotations

import json
import unittest


class ChapterReviewPromptTests(unittest.TestCase):
    def test_lens_prompt_includes_logic_rubric_and_output_contract(self):
        from muse.prompts.chapter_review import chapter_review_prompt_for_lens

        system, user = chapter_review_prompt_for_lens(
            chapter_title="Methods",
            merged_text="Draft text.",
            lens="logic",
        )

        self.assertIn("Primary review lens: logic", system)
        self.assertIn("1 point:", system)
        self.assertIn("5 point:", system)
        self.assertIn("review_notes", system)
        self.assertIn('"chapter_title": "Methods"', user)

    def test_lens_prompts_are_distinct(self):
        from muse.prompts.chapter_review import chapter_review_prompt_for_lens

        systems = {
            lens: chapter_review_prompt_for_lens("Intro", "Text", lens)[0]
            for lens in ("logic", "style", "citation", "structure")
        }

        self.assertEqual(len(set(systems.values())), 4)

    def test_legacy_prompt_combines_all_lenses_for_backward_compatibility(self):
        from muse.prompts.chapter_review import chapter_review_prompt

        system, _ = chapter_review_prompt("Intro", "Text")

        self.assertIn("Available review lenses", system)
        self.assertIn("logic", system)
        self.assertIn("style", system)
        self.assertIn("citation", system)
        self.assertIn("structure", system)


class GlobalReviewPromptTests(unittest.TestCase):
    def test_global_review_prompt_targets_full_draft_and_review_contract(self):
        from muse.prompts.global_review import global_review_prompt_for_lens

        system, user = global_review_prompt_for_lens(
            merged_text="Full thesis draft.",
            lens="logic",
        )

        self.assertIn("full merged thesis draft", system)
        self.assertIn("Primary review lens: logic", system)
        self.assertIn('"section"', system)
        self.assertIn('"is_recurring"', system)
        self.assertIn('"text": "Full thesis draft."', user)

    def test_adaptive_review_prompt_injects_previous_scores_and_notes(self):
        from muse.prompts.adaptive_review import adaptive_review_prompt

        system, user = adaptive_review_prompt(
            merged_text="Updated thesis draft.",
            lens="logic",
            review_history=[
                {
                    "iteration": 1,
                    "scores": {"logic": 2, "citation": 3},
                    "notes_summary": "Still missing support for the transition paragraph.",
                }
            ],
            iteration=2,
        )

        self.assertIn("Previous review round (iteration 1)", system)
        self.assertIn('"logic": 2', system)
        self.assertIn("Still missing support for the transition paragraph.", system)
        self.assertIn("escalate the severity", system)
        self.assertIn('"text": "Updated thesis draft."', user)

    def test_reviewer_persona_prompts_are_scoped_to_owned_dimensions(self):
        from muse.prompts.reviewer_personas import reviewer_persona_prompt

        logic_system, _ = reviewer_persona_prompt("logic", merged_text="Draft.")
        citation_system, _ = reviewer_persona_prompt("citation", merged_text="Draft.")
        readability_system, _ = reviewer_persona_prompt("readability", merged_text="Draft.")

        self.assertIn("logic, structure, balance", logic_system)
        self.assertIn("citation, coverage, depth", citation_system)
        self.assertIn("style, term_consistency, redundancy", readability_system)
        self.assertIn('"scores": {"logic":', logic_system)
        self.assertIn('"scores": {"citation":', citation_system)
        self.assertIn('"scores": {"style":', readability_system)

    def test_review_judge_prompt_defines_unified_output_contract(self):
        from muse.prompts.review_judge import JUDGE_SYSTEM

        self.assertIn('"final_scores"', JUDGE_SYSTEM)
        self.assertIn('"unified_notes"', JUDGE_SYSTEM)
        self.assertIn('"conflicts_resolved"', JUDGE_SYSTEM)
        payload = json.dumps(
            [
                {"persona": "logic", "result": {"scores": {"logic": 3}, "review_notes": []}},
                {"persona": "citation", "result": {"scores": {"citation": 4}, "review_notes": []}},
            ],
            ensure_ascii=False,
        )
        self.assertIn('"persona": "logic"', payload)


if __name__ == "__main__":
    unittest.main()
