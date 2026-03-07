import unittest

from muse.orchestrator import can_advance_to_stage, gate_export
from muse.schemas import new_thesis_state


class OrchestratorGateTests(unittest.TestCase):
    def test_blocks_export_when_flagged_citations_exist(self):
        state = new_thesis_state(
            project_id="p3",
            topic="topic",
            discipline="discipline",
            language="zh",
            format_standard="GB/T 7714-2015",
        )
        state["current_stage"] = 5
        state["flagged_citations"] = [
            {
                "cite_key": "@x",
                "reason": "unsupported_claim",
                "claim_id": "c1",
                "detail": "entailment result=contradiction",
            }
        ]

        allowed, reason = gate_export(state)

        self.assertFalse(allowed)
        self.assertTrue(reason)  # any non-empty reason string is acceptable

    def test_allows_export_when_no_flagged_citations(self):
        state = new_thesis_state(
            project_id="p4",
            topic="topic",
            discipline="discipline",
            language="zh",
            format_standard="GB/T 7714-2015",
        )
        state["current_stage"] = 5

        allowed, reason = gate_export(state)

        self.assertTrue(allowed)
        self.assertEqual(reason, "ok")

    def test_requires_approved_outline_before_stage3(self):
        state = new_thesis_state(
            project_id="p5",
            topic="topic",
            discipline="discipline",
            language="zh",
            format_standard="GB/T 7714-2015",
        )
        state["current_stage"] = 2
        state["outline_json"] = {}
        state["chapter_plans"] = []

        allowed, reason = can_advance_to_stage(state, 3)

        self.assertFalse(allowed)
        self.assertIn("outline", reason)


if __name__ == "__main__":
    unittest.main()
