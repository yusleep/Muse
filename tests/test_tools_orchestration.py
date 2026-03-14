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


class PartialSubtaskAccumulatorTests(unittest.TestCase):
    def setUp(self):
        from muse.tools.orchestration import clear_partial_subtask_results

        clear_partial_subtask_results()

    def tearDown(self):
        from muse.tools.orchestration import clear_partial_subtask_results

        clear_partial_subtask_results()

    def test_append_partial_subtask_results_preserves_order(self):
        from muse.tools.orchestration import (
            append_partial_subtask_result,
            get_partial_subtask_results,
        )

        append_partial_subtask_result({"subtask_id": "sub_01", "output_text": "First"})
        append_partial_subtask_result({"subtask_id": "sub_02", "output_text": "Second"})

        results = get_partial_subtask_results()
        self.assertEqual([item["subtask_id"] for item in results], ["sub_01", "sub_02"])

    def test_clear_partial_subtask_results_resets_accumulator(self):
        from muse.tools.orchestration import (
            append_partial_subtask_result,
            clear_partial_subtask_results,
            get_partial_subtask_results,
        )

        append_partial_subtask_result({"subtask_id": "sub_01", "output_text": "First"})
        clear_partial_subtask_results()

        self.assertEqual(get_partial_subtask_results(), [])


if __name__ == "__main__":
    unittest.main()
