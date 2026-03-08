import unittest

from muse.graph.nodes.export import _gate_export
from muse.schemas import new_thesis_state


class GraphExportGateTests(unittest.TestCase):
    def test_blocks_export_when_claim_is_explicitly_contradicted(self):
        state = new_thesis_state(
            project_id="p3",
            topic="topic",
            discipline="discipline",
            language="zh",
            format_standard="GB/T 7714-2015",
        )
        state["flagged_citations"] = [
            {
                "cite_key": "@x",
                "reason": "unsupported_claim",
                "claim_id": "c1",
                "detail": "entailment result=contradiction",
            }
        ]

        allowed, reason = _gate_export(state)

        self.assertFalse(allowed)
        self.assertTrue(reason)

    def test_allows_export_when_no_flagged_citations_exist(self):
        state = new_thesis_state(
            project_id="p4",
            topic="topic",
            discipline="discipline",
            language="zh",
            format_standard="GB/T 7714-2015",
        )

        allowed, reason = _gate_export(state)

        self.assertTrue(allowed)
        self.assertEqual(reason, "ok")

    def test_allows_export_for_metadata_only_flags(self):
        state = new_thesis_state(
            project_id="p5",
            topic="topic",
            discipline="discipline",
            language="zh",
            format_standard="GB/T 7714-2015",
        )
        state["flagged_citations"] = [
            {"cite_key": "@x", "reason": "metadata_mismatch", "detail": "metadata_mismatch"},
            {"cite_key": "@y", "reason": "not_found", "detail": "not_found"},
        ]

        allowed, reason = _gate_export(state)

        self.assertTrue(allowed)
        self.assertEqual(reason, "ok")


if __name__ == "__main__":
    unittest.main()
