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
        from muse.tools.orchestration import (
            clear_partial_subtask_results,
            get_partial_subtask_results,
        )
        from muse.tools.writing import write_section

        class _StringLLM:
            def structured(self, *, system, user, route="default", max_tokens=2500):
                del system, user, route, max_tokens
                return '{"text":"already serialized","citations_used":["@smith2024"],"key_claims":[]}'

        class _Services:
            llm = _StringLLM()

        clear_partial_subtask_results()
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
        self.assertEqual(get_partial_subtask_results()[0]["output_text"], "already serialized")

    def test_write_section_accumulates_plain_text_success_output(self):
        from muse.tools._context import set_services
        from muse.tools.orchestration import (
            clear_partial_subtask_results,
            get_partial_subtask_results,
        )
        from muse.tools.writing import write_section

        class _PlainTextLLM:
            def structured(self, *, system, user, route="default", max_tokens=2500):
                del system, user, route, max_tokens
                return "Plain text subsection."

        class _Services:
            llm = _PlainTextLLM()

        clear_partial_subtask_results()
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

        partial_results = get_partial_subtask_results()
        self.assertEqual(result, "Plain text subsection.")
        self.assertEqual(len(partial_results), 1)
        self.assertEqual(partial_results[0]["output_text"], "Plain text subsection.")

    def test_write_section_does_not_accumulate_failed_placeholder_output(self):
        from muse.tools._context import set_services
        from muse.tools.orchestration import (
            clear_partial_subtask_results,
            get_partial_subtask_results,
        )
        from muse.tools.writing import write_section

        class _FailingLLM:
            def structured(self, *, system, user, route="default", max_tokens=2500):
                del system, user, route, max_tokens
                raise RuntimeError("boom")

        class _Services:
            llm = _FailingLLM()

        clear_partial_subtask_results()
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

        self.assertIn("LLM call failed", result)
        self.assertEqual(get_partial_subtask_results(), [])

    def test_write_section_prompt_includes_scope_guard(self):
        from muse.tools._context import set_services
        from muse.tools._context import set_state
        from muse.tools.writing import write_section

        seen_systems = []
        seen_payloads = []

        class _CaptureLLM:
            def structured(self, *, system, user, route="default", max_tokens=2500):
                del route, max_tokens
                seen_systems.append(system)
                seen_payloads.append(json.loads(user))
                return {
                    "text": "Generated section text about the topic.",
                    "citations_used": ["@smith2024"],
                    "key_claims": [],
                }

        class _Services:
            llm = _CaptureLLM()

        set_state(
            {
                "indexed_papers": {
                    "@smith2024": {
                        "source": "local",
                        "indexed": True,
                        "available_sections": ["Results"],
                    }
                }
            }
        )
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
        self.assertIn("source=local", seen_systems[0])
        snapshot = seen_payloads[0]["available_references"][0]
        self.assertEqual(snapshot["source"], "local")
        self.assertTrue(snapshot["indexed"])
        self.assertEqual(snapshot["available_sections"], ["Results"])

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

    def test_write_section_accumulates_partial_result_for_recovery(self):
        from muse.tools._context import set_services
        from muse.tools.orchestration import (
            clear_partial_subtask_results,
            get_partial_subtask_results,
        )
        from muse.tools.writing import write_section

        class _Services:
            llm = _FakeLLM()

        clear_partial_subtask_results()
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

        partial_results = get_partial_subtask_results()
        self.assertEqual(len(partial_results), 1)
        self.assertEqual(partial_results[0]["subtask_id"], "sub_01")
        self.assertEqual(partial_results[0]["title"], "Background")
        self.assertEqual(partial_results[0]["confidence"], 0.3)
        self.assertTrue(partial_results[0]["needs_revision"])

    def test_write_section_injects_consistency_context_from_runtime_state(self):
        from muse.tools._context import set_services
        from muse.tools._context import set_state
        from muse.tools.orchestration import clear_partial_subtask_results
        from muse.tools.writing import write_section

        seen_payloads = []

        class _CaptureLLM:
            def structured(self, *, system, user, route="default", max_tokens=2500):
                del system, route, max_tokens
                seen_payloads.append(json.loads(user))
                return {
                    "text": "Generated section text about the topic.",
                    "citations_used": [],
                    "key_claims": [],
                    "transition_out": "",
                    "glossary_additions": {},
                    "self_assessment": {"confidence": 0.7, "weak_spots": [], "needs_revision": False},
                }

        class _Services:
            llm = _CaptureLLM()

        set_state(
            {
                "consistency_data": {
                    "glossary": {"Agent Runtime": "智能体运行时"},
                    "citation_counts": {"@smith2024": 2},
                    "notation": {},
                    "chapter_summaries": {"ch_01": "Prior chapter summary."},
                }
            }
        )
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

        consistency = seen_payloads[0]["consistency_context"]
        self.assertEqual(consistency["glossary"]["Agent Runtime"], "智能体运行时")
        self.assertEqual(consistency["citation_counts"]["@smith2024"], 2)
        clear_partial_subtask_results()

    def test_write_section_injects_reflection_tips_from_runtime_state(self):
        from muse.tools._context import set_services
        from muse.tools._context import set_state
        from muse.tools.orchestration import clear_partial_subtask_results
        from muse.tools.writing import write_section

        seen_payloads = []

        class _CaptureLLM:
            def structured(self, *, system, user, route="default", max_tokens=2500):
                del system, route, max_tokens
                seen_payloads.append(json.loads(user))
                return {
                    "text": "Generated section text about the topic.",
                    "citations_used": [],
                    "key_claims": [],
                    "transition_out": "",
                    "glossary_additions": {},
                    "self_assessment": {"confidence": 0.7, "weak_spots": [], "needs_revision": False},
                }

        class _Services:
            llm = _CaptureLLM()

        set_state(
            {
                "reflection_data": {
                    "entries": [
                        {
                            "chapter_id": "ch_01",
                            "dimension": "logic",
                            "outcome": "positive",
                            "instruction": "Clarify the core argument before implementation details.",
                            "score_delta": 2,
                        }
                    ]
                }
            }
        )
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

        tips = seen_payloads[0]["writing_tips_from_experience"]
        self.assertEqual(len(tips), 1)
        self.assertIn("Clarify the core argument", tips[0])
        clear_partial_subtask_results()

    def test_write_section_injects_reference_briefs_from_runtime_state(self):
        from muse.tools._context import set_services
        from muse.tools._context import set_state
        from muse.tools.orchestration import clear_partial_subtask_results
        from muse.tools.writing import write_section

        seen_payloads = []

        class _CaptureLLM:
            def structured(self, *, system, user, route="default", max_tokens=2500):
                del system, route, max_tokens
                seen_payloads.append(json.loads(user))
                return {
                    "text": "Generated section text about the topic.",
                    "citations_used": [],
                    "key_claims": [],
                    "transition_out": "",
                    "glossary_additions": {},
                    "self_assessment": {"confidence": 0.7, "weak_spots": [], "needs_revision": False},
                }

        class _Services:
            llm = _CaptureLLM()

        set_state(
            {
                "chapter_plan": {"chapter_id": "ch_02"},
                "reference_briefs": {
                    "ch_02": [
                        {
                            "ref_id": "@smith2024",
                            "relevance": "directly addresses method",
                            "key_finding": "Table 2 shows a 15% improvement.",
                            "how_to_cite": "Use as main evidence.",
                        }
                    ],
                    "ch_02_gaps": ["No source covers deployment constraints."],
                },
            }
        )
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

        self.assertEqual(seen_payloads[0]["reference_briefs"][0]["ref_id"], "@smith2024")
        self.assertEqual(seen_payloads[0]["evidence_gaps"], ["No source covers deployment constraints."])
        clear_partial_subtask_results()

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
