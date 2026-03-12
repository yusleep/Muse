"""Tests for muse/tools/citation.py."""

from __future__ import annotations

import json
import unittest


class CitationToolTests(unittest.TestCase):
    def test_verify_doi_returns_result(self):
        from muse.tools.citation import verify_doi

        result = verify_doi.invoke({"doi": "10.1000/test"})
        self.assertIsInstance(result, str)

    def test_crosscheck_metadata_returns_result(self):
        from muse.tools.citation import crosscheck_metadata

        result = crosscheck_metadata.invoke(
            {
                "reference_json": '{"ref_id": "@test", "title": "Test Paper", "doi": "10.1000/test", "authors": ["A. Test"], "year": 2024}'
            }
        )
        self.assertIsInstance(result, str)

    def test_entailment_check_returns_result(self):
        from muse.tools.citation import entailment_check

        result = entailment_check.invoke(
            {
                "premise": "Neural networks can approximate any function.",
                "hypothesis": "Deep learning has universal approximation capability.",
            }
        )
        self.assertIsInstance(result, str)
        self.assertIn(
            json.loads(result)["entailment"],
            ["entailment", "neutral", "contradiction", "skipped"],
        )

    def test_flag_citation_returns_json(self):
        from muse.tools.citation import flag_citation

        result = flag_citation.invoke(
            {
                "cite_key": "@smith2024",
                "reason": "unsupported_claim",
                "claim_id": "ch01_sub01_c01",
                "detail": "Claim not supported by reference abstract.",
            }
        )
        parsed = json.loads(result)
        self.assertEqual(parsed["cite_key"], "@smith2024")

    def test_repair_citation_returns_json(self):
        from muse.tools.citation import repair_citation

        result = repair_citation.invoke(
            {
                "claim_id": "ch01_sub01_c01",
                "action": "replace_source",
                "new_cite_key": "@jones2023",
                "justification": "Jones 2023 directly addresses the claim.",
            }
        )
        parsed = json.loads(result)
        self.assertEqual(parsed["action"], "replace_source")

    def test_record_and_finalize_citation_review(self):
        from muse.tools.citation import (
            clear_citation_review_session,
            finalize_citation_review,
            get_finalized_citation_review,
            prepare_citation_review_session,
            record_citation_assessment,
        )

        prepare_citation_review_session(
            [
                {
                    "cite_key": "@smith2024",
                    "claim_id": "c1",
                    "claim": "Claim text.",
                    "evidence": "Supporting evidence.",
                }
            ]
        )
        try:
            record_result = record_citation_assessment.invoke(
                {
                    "cite_key": "@smith2024",
                    "claim_id": "c1",
                    "verdict": "verified",
                    "support_score": 0.95,
                    "confidence": "high",
                    "reason": "supported",
                    "detail": "Evidence matches claim.",
                    "evidence_excerpt": "Supporting evidence.",
                }
            )
            self.assertIn("recorded", record_result.lower())

            finalize_result = finalize_citation_review.invoke({"summary": "done"})
            self.assertIn("completed", finalize_result.lower())

            payload = get_finalized_citation_review()
            self.assertIsInstance(payload, dict)
            self.assertEqual(payload["verified_citations"], ["@smith2024"])
            self.assertEqual(payload["flagged_citations"], [])
            self.assertIn("c1", payload["citation_ledger"])
        finally:
            clear_citation_review_session()

    def test_finalize_citation_review_rejects_missing_assessments(self):
        from muse.tools.citation import (
            clear_citation_review_session,
            finalize_citation_review,
            prepare_citation_review_session,
        )

        prepare_citation_review_session(
            [
                {
                    "cite_key": "@smith2024",
                    "claim_id": "c1",
                    "claim": "Claim text.",
                    "evidence": "Supporting evidence.",
                }
            ]
        )
        try:
            result = finalize_citation_review.invoke({"summary": "done"})
            self.assertIn("error", result.lower())
        finally:
            clear_citation_review_session()


if __name__ == "__main__":
    unittest.main()
