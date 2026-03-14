"""Tests for muse/tools/writing.py"""

from __future__ import annotations

import json
import unittest


class _FakeLLM:
    def structured(self, *, system, user, route="default", max_tokens=2500):
        return {
            "text": "Generated section text about the topic.",
            "citations_used": ["@smith2024"],
            "key_claims": ["Claim A."],
            "transition_out": "",
            "glossary_additions": {},
            "self_assessment": {
                "confidence": 0.7,
                "weak_spots": [],
                "needs_revision": False,
            },
        }

    def text(self, *, system, user, route="default", max_tokens=2500):
        return "Revised section text."


class WriteSectionToolTests(unittest.TestCase):
    def test_write_section_returns_text_and_citations(self):
        from muse.tools._context import set_services
        from muse.tools.writing import write_section

        class _Services:
            llm = _FakeLLM()

        set_services(_Services())
        result = write_section.func(
            chapter_title="Introduction",
            subtask_id="sub_01",
            subtask_title="Background",
            target_words=1200,
            topic="LangGraph thesis automation",
            language="zh",
            references_json='[{"ref_id": "@smith2024", "title": "Graph Systems", "year": 2024, "abstract": "A study."}]',
            runtime=None,
        )
        self.assertIsInstance(result, str)
        self.assertIn("text", result)

    def test_write_section_filters_citations_to_allowed_refs(self):
        from muse.tools._context import set_services
        from muse.tools.writing import write_section

        class _HallucinatingLLM(_FakeLLM):
            def structured(self, *, system, user, route="default", max_tokens=2500):
                del system, user, route, max_tokens
                return {
                    "text": "Generated section text about the topic.",
                    "citations_used": ["@smith2024", "@fake2024"],
                    "key_claims": ["Claim A."],
                    "transition_out": "",
                    "glossary_additions": {},
                    "self_assessment": {
                        "confidence": 0.7,
                        "weak_spots": [],
                        "needs_revision": False,
                    },
                }

        class _Services:
            llm = _HallucinatingLLM()

        set_services(_Services())
        result = write_section.func(
            chapter_title="Introduction",
            subtask_id="sub_01",
            subtask_title="Background",
            target_words=1200,
            topic="LangGraph thesis automation",
            language="zh",
            references_json='[{"ref_id": "@smith2024", "title": "Graph Systems", "year": 2024, "abstract": "A study."}]',
            runtime=None,
        )

        payload = json.loads(result)
        self.assertEqual(payload["citations_used"], ["@smith2024"])

    def test_write_section_does_not_double_serialize_string_output(self):
        from muse.tools._context import set_services
        from muse.tools.writing import write_section

        class _StringLLM:
            def structured(self, *, system, user, route="default", max_tokens=2500):
                del system, user, route, max_tokens
                return '{"text":"already serialized","citations_used":["@smith2024"],"key_claims":[]}'

        class _Services:
            llm = _StringLLM()

        set_services(_Services())
        result = write_section.func(
            chapter_title="Introduction",
            subtask_id="sub_01",
            subtask_title="Background",
            target_words=1200,
            topic="LangGraph thesis automation",
            language="zh",
            references_json='[{"ref_id": "@smith2024", "title": "Graph Systems", "year": 2024, "abstract": "A study."}]',
            runtime=None,
        )

        self.assertEqual(
            result,
            '{"text":"already serialized","citations_used":["@smith2024"],"key_claims":[]}',
        )

    def test_write_section_prompt_includes_scope_guard(self):
        from muse.tools._context import set_services
        from muse.tools.writing import write_section

        seen_systems = []

        class _CaptureLLM:
            def structured(self, *, system, user, route="default", max_tokens=2500):
                del user, route, max_tokens
                seen_systems.append(system)
                return {
                    "text": "Generated section text about the topic.",
                    "citations_used": ["@smith2024"],
                    "key_claims": [],
                }

        class _Services:
            llm = _CaptureLLM()

        set_services(_Services())
        write_section.func(
            chapter_title="Introduction",
            subtask_id="sub_01",
            subtask_title="Background",
            target_words=1200,
            topic="LangGraph thesis automation",
            language="zh",
            references_json='[{"ref_id": "@smith2024", "title": "Graph Systems", "year": 2024, "abstract": "A study."}]',
            runtime=None,
        )

        self.assertEqual(len(seen_systems), 1)
        self.assertIn("SCOPE GUARD", seen_systems[0])

    def test_write_section_keeps_full_abstract_and_caps_snapshot_at_50(self):
        from muse.tools._context import set_services
        from muse.tools.writing import write_section

        long_abstract = "A" * 500
        seen_payloads = []

        class _CaptureLLM:
            def structured(self, *, system, user, route="default", max_tokens=2500):
                del system, route, max_tokens
                seen_payloads.append(json.loads(user))
                return {
                    "text": "Generated section text about the topic.",
                    "citations_used": ["@r01"],
                    "key_claims": [],
                }

        class _Services:
            llm = _CaptureLLM()

        references_json = json.dumps(
            [
                {
                    "ref_id": f"@r{i:02d}",
                    "title": f"Reference {i}",
                    "year": 2024,
                    "abstract": long_abstract if i == 1 else f"Abstract {i}",
                }
                for i in range(1, 52)
            ],
            ensure_ascii=False,
        )

        set_services(_Services())
        write_section.func(
            chapter_title="Introduction",
            subtask_id="sub_01",
            subtask_title="Background",
            target_words=1200,
            topic="LangGraph thesis automation",
            language="zh",
            references_json=references_json,
            runtime=None,
        )

        snapshot = seen_payloads[0]["available_references"]
        self.assertEqual(len(snapshot), 50)
        self.assertEqual(snapshot[0]["abstract"], long_abstract)

    def test_revise_section_returns_revised_text(self):
        from muse.tools.writing import revise_section

        result = revise_section.func(
            section_text="Original text here.",
            instruction="Improve transitions between paragraphs.",
            chapter_title="Introduction",
            language="zh",
            runtime=None,
        )
        self.assertIsInstance(result, str)

    def test_apply_patch_replaces_old_with_new(self):
        from muse.tools.writing import apply_patch

        result = apply_patch.invoke(
            {
                "section_text": "The quick brown fox jumps over the lazy dog.",
                "old_string": "quick brown fox",
                "new_string": "slow red cat",
            }
        )
        self.assertIn("slow red cat", result)
        self.assertNotIn("quick brown fox", result)

    def test_apply_patch_reports_not_found(self):
        from muse.tools.writing import apply_patch

        result = apply_patch.invoke(
            {
                "section_text": "Hello world.",
                "old_string": "nonexistent string",
                "new_string": "replacement",
            }
        )
        self.assertIn("not found", result.lower())


if __name__ == "__main__":
    unittest.main()
