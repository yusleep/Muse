from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
