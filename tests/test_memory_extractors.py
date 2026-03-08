"""Tests for memory extraction functions (muse.memory.extractors)."""

from __future__ import annotations

from muse.memory.extractors import (
    extract_from_citation_subgraph,
    extract_from_hitl_feedback,
    extract_from_initialize,
    extract_from_review,
)


class TestExtractFromInitialize:
    def test_extracts_topic(self):
        entries = extract_from_initialize({"topic": "Deep Learning"})
        assert any("Deep Learning" in entry.content for entry in entries)
        assert any(entry.category == "fact" for entry in entries)

    def test_extracts_discipline(self):
        entries = extract_from_initialize({"discipline": "Computer Science"})
        assert any("Computer Science" in entry.content for entry in entries)

    def test_extracts_language(self):
        entries = extract_from_initialize({"language": "zh"})
        assert any(entry.category == "user_pref" for entry in entries)
        assert any("zh" in entry.content for entry in entries)

    def test_extracts_format_standard(self):
        entries = extract_from_initialize({"format_standard": "GB/T 7714"})
        assert any("GB/T 7714" in entry.content for entry in entries)

    def test_empty_state(self):
        assert extract_from_initialize({}) == []

    def test_passes_run_id(self):
        entries = extract_from_initialize({"topic": "X"}, run_id="run_123")
        assert all(entry.source_run == "run_123" for entry in entries)


class TestExtractFromHITLFeedback:
    def test_extracts_long_notes(self):
        result = {
            "review_feedback": [{"notes": "Please use more formal language in the introduction section"}]
        }
        entries = extract_from_hitl_feedback("review_draft", result)
        assert len(entries) == 1
        assert "formal language" in entries[0].content

    def test_skips_short_notes(self):
        result = {"review_feedback": [{"notes": "ok"}]}
        assert extract_from_hitl_feedback("review_draft", result) == []

    def test_classifies_style_feedback(self):
        result = {
            "review_feedback": [{"notes": "Please adopt a more formal academic tone throughout the paper"}]
        }
        entries = extract_from_hitl_feedback("review_draft", result)
        assert len(entries) == 1
        assert entries[0].category == "writing_style"
        assert entries[0].confidence == 0.7

    def test_handles_empty_feedback(self):
        assert extract_from_hitl_feedback("review_draft", {}) == []
        assert extract_from_hitl_feedback("review_draft", {"review_feedback": []}) == []

    def test_handles_non_dict_entries(self):
        result = {"review_feedback": ["string_entry", None, 42]}
        assert extract_from_hitl_feedback("review_draft", result) == []


class TestExtractFromCitationSubgraph:
    def test_extracts_verified_citation(self):
        state = {
            "references": [
                {
                    "ref_id": "@smith2024dl",
                    "title": "Deep Learning",
                    "doi": "10.1234/test",
                    "year": 2024,
                }
            ]
        }
        result = {"verified_citations": ["@smith2024dl"]}
        entries = extract_from_citation_subgraph(state, result)
        assert len(entries) == 1
        assert entries[0].category == "citation"
        assert "Deep Learning" in entries[0].content
        assert "10.1234/test" in entries[0].content
        assert entries[0].confidence == 0.9

    def test_includes_year(self):
        state = {"references": [{"ref_id": "@a", "title": "T", "year": 2024}]}
        result = {"verified_citations": ["@a"]}
        entries = extract_from_citation_subgraph(state, result)
        assert "(2024)" in entries[0].content

    def test_no_verified_returns_empty(self):
        assert extract_from_citation_subgraph({}, {"verified_citations": []}) == []

    def test_missing_reference_uses_cite_key(self):
        entries = extract_from_citation_subgraph({"references": []}, {"verified_citations": ["@unknown2024"]})
        assert len(entries) == 1
        assert "@unknown2024" in entries[0].content


class TestExtractFromReview:
    def test_flags_low_quality_dimensions(self):
        entries = extract_from_review({}, {"quality_scores": {"logic": 2, "style": 4, "citation": 1}})
        assert len(entries) == 2
        contents = {entry.content for entry in entries}
        assert any("logic" in content for content in contents)
        assert any("citation" in content for content in contents)

    def test_ignores_good_scores(self):
        assert extract_from_review({}, {"quality_scores": {"logic": 4, "style": 5}}) == []

    def test_empty_scores(self):
        assert extract_from_review({}, {}) == []
        assert extract_from_review({}, {"quality_scores": {}}) == []
