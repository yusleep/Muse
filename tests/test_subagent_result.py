"""Tests for SubagentResult."""

from __future__ import annotations

import unittest


class SubagentResultTests(unittest.TestCase):
    def test_result_has_all_fields(self):
        from muse.agents.result import SubagentResult

        result = SubagentResult(
            status="completed",
            accomplishments=["done"],
            key_findings=["finding"],
            files_created=["out.txt"],
            issues=["warning"],
            citations=[{"ref_id": "@a"}],
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.accomplishments, ["done"])
        self.assertEqual(result.key_findings, ["finding"])
        self.assertEqual(result.files_created, ["out.txt"])
        self.assertEqual(result.issues, ["warning"])
        self.assertEqual(result.citations, [{"ref_id": "@a"}])

    def test_result_defaults(self):
        from muse.agents.result import SubagentResult

        result = SubagentResult()
        self.assertEqual(result.accomplishments, [])
        self.assertEqual(result.key_findings, [])
        self.assertEqual(result.files_created, [])
        self.assertEqual(result.issues, [])
        self.assertEqual(result.citations, [])

    def test_result_to_dict(self):
        from muse.agents.result import SubagentResult

        result = SubagentResult(
            status="completed",
            accomplishments=["done"],
            key_findings=["finding"],
            files_created=["out.txt"],
            issues=["warning"],
            citations=[{"ref_id": "@a"}],
        )
        payload = result.to_dict()
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["files_created"], ["out.txt"])

    def test_result_from_dict(self):
        from muse.agents.result import SubagentResult

        payload = {
            "status": "failed",
            "accomplishments": [],
            "key_findings": ["finding"],
            "files_created": [],
            "issues": ["boom"],
            "citations": [],
        }
        result = SubagentResult.from_dict(payload)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.issues, ["boom"])

    def test_result_summary_string(self):
        from muse.agents.result import SubagentResult

        result = SubagentResult(
            status="completed",
            accomplishments=["a", "b"],
        )
        summary = result.summary()
        self.assertIn("completed", summary)
        self.assertIn("2", summary)


if __name__ == "__main__":
    unittest.main()
