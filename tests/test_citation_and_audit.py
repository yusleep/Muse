import json
import os
import tempfile
import unittest

from muse.audit import JsonlAuditSink, build_event
from muse.citation import verify_all_citations


class CitationVerificationTests(unittest.TestCase):
    def test_classifies_all_failure_reasons_and_verified(self):
        references = [
            {
                "ref_id": "@ok",
                "doi": "10.1000/ok",
                "title": "ok",
                "verified_metadata": True,
            },
            {
                "ref_id": "@bad_doi",
                "doi": "10.1000/bad",
                "title": "bad doi",
                "verified_metadata": True,
            },
            {
                "ref_id": "@bad_meta",
                "doi": "10.1000/meta",
                "title": "bad metadata",
                "verified_metadata": False,
            },
            {
                "ref_id": "@neutral",
                "doi": "10.1000/neutral",
                "title": "neutral support",
                "verified_metadata": True,
            },
        ]

        citation_uses = [
            {"cite_key": "@missing", "claim_id": "c0", "chapter_id": "ch1", "subtask_id": "s1"},
            {"cite_key": "@bad_doi", "claim_id": "c1", "chapter_id": "ch1", "subtask_id": "s1"},
            {"cite_key": "@bad_meta", "claim_id": "c2", "chapter_id": "ch1", "subtask_id": "s1"},
            {"cite_key": "@neutral", "claim_id": "c3", "chapter_id": "ch1", "subtask_id": "s1"},
            {"cite_key": "@ok", "claim_id": "c4", "chapter_id": "ch1", "subtask_id": "s1"},
        ]

        claim_text = {
            "c0": "missing citation claim",
            "c1": "doi invalid claim",
            "c2": "metadata mismatch claim",
            "c3": "unsupported claim",
            "c4": "supported claim",
        }

        def verify_doi(doi: str) -> bool:
            return doi != "10.1000/bad"

        def crosscheck_metadata(ref: dict) -> bool:
            return bool(ref.get("verified_metadata"))

        def retrieve_passage(ref: dict, claim: str) -> str:
            return f"passage for {ref['ref_id']} / {claim}"

        def check_entailment(premise: str, hypothesis: str) -> str:
            if "unsupported" in hypothesis:
                return "neutral"
            return "entailment"

        verified, flagged = verify_all_citations(
            references=references,
            citation_uses=citation_uses,
            claim_text_by_id=claim_text,
            verify_doi=verify_doi,
            crosscheck_metadata=crosscheck_metadata,
            retrieve_passage=retrieve_passage,
            check_entailment=check_entailment,
        )

        self.assertIn("@ok", verified)
        reasons = {(item["cite_key"], item["reason"]) for item in flagged}
        self.assertIn(("@missing", "not_found"), reasons)
        self.assertIn(("@bad_doi", "doi_invalid"), reasons)
        self.assertIn(("@bad_meta", "metadata_mismatch"), reasons)
        self.assertIn(("@neutral", "unsupported_claim"), reasons)


class AuditSinkTests(unittest.TestCase):
    def test_jsonl_sink_is_append_only_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "audit.jsonl")
            sink = JsonlAuditSink(path)

            event = build_event(
                stage=3,
                agent="SubAgent-sub_01",
                event_type="llm_call",
                model="test-model",
                tokens=123,
                latency_ms=456,
                cost_estimate=0.02,
            )

            sink.append(event)
            sink.append(event)  # duplicate should be ignored by event_id

            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["event_id"], event["event_id"])
            self.assertEqual(payload["stage"], 3)

    def test_rejects_event_without_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "audit.jsonl")
            sink = JsonlAuditSink(path)

            with self.assertRaises(ValueError):
                sink.append({"event_type": "llm_call"})


if __name__ == "__main__":
    unittest.main()
