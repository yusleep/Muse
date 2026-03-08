"""Tests for muse/tools/orchestration.py"""

from __future__ import annotations

import unittest


class SubmitResultTests(unittest.TestCase):
    def test_submit_result_returns_confirmation(self):
        from muse.tools.orchestration import submit_result

        result = submit_result.invoke(
            {
                "result_json": '{"merged_text": "Chapter content.", "quality_scores": {"coherence": 4}}',
                "summary": "Chapter draft completed with 2 revisions.",
            }
        )
        self.assertIn("submitted", result.lower())

    def test_submit_result_invalid_json(self):
        from muse.tools.orchestration import submit_result

        result = submit_result.invoke(
            {
                "result_json": "not valid json{",
                "summary": "Bad data.",
            }
        )
        self.assertIn("error", result.lower())


class UpdatePlanTests(unittest.TestCase):
    def test_update_plan_returns_confirmation(self):
        from muse.tools.orchestration import update_plan

        result = update_plan.invoke(
            {
                "status": "drafting",
                "progress_pct": 45,
                "current_step": "Writing section 2.3",
                "notes": "References loaded, outline stable.",
            }
        )
        self.assertIn("updated", result.lower())


if __name__ == "__main__":
    unittest.main()
