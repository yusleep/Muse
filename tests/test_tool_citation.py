"""Tests for citation verification LangChain tools."""

from __future__ import annotations

import unittest
from typing import Any


class _FakeMetadataClient:
    def __init__(self, doi_valid: bool = True, metadata_match: bool = True):
        self._doi_valid = doi_valid
        self._metadata_match = metadata_match

    def verify_doi(self, doi: str) -> bool:
        return self._doi_valid

    def crosscheck_metadata(self, ref: dict[str, Any]) -> bool:
        return self._metadata_match


class VerifyDoiToolTests(unittest.TestCase):
    def test_verify_doi_returns_string(self):
        from muse.tools.citation import make_verify_doi_tool

        tool = make_verify_doi_tool(_FakeMetadataClient(doi_valid=True))
        result = tool.invoke({"doi": "10.1000/test"})
        self.assertIsInstance(result, str)
        self.assertIn("valid", result.lower())

    def test_verify_doi_invalid(self):
        from muse.tools.citation import make_verify_doi_tool

        tool = make_verify_doi_tool(_FakeMetadataClient(doi_valid=False))
        result = tool.invoke({"doi": "10.9999/fake"})
        self.assertIn("invalid", result.lower())

    def test_verify_doi_tool_name(self):
        from muse.tools.citation import make_verify_doi_tool

        tool = make_verify_doi_tool(_FakeMetadataClient())
        self.assertEqual(tool.name, "verify_doi")

    def test_verify_doi_is_base_tool(self):
        from langchain_core.tools import BaseTool
        from muse.tools.citation import make_verify_doi_tool

        tool = make_verify_doi_tool(_FakeMetadataClient())
        self.assertIsInstance(tool, BaseTool)


class CrosscheckMetadataToolTests(unittest.TestCase):
    def test_crosscheck_returns_string(self):
        from muse.tools.citation import make_crosscheck_metadata_tool

        tool = make_crosscheck_metadata_tool(_FakeMetadataClient(metadata_match=True))
        result = tool.invoke(
            {
                "title": "Graph Neural Networks",
                "authors": "Smith, Jones",
                "year": "2024",
                "doi": "10.1000/test",
            }
        )
        self.assertIsInstance(result, str)
        self.assertIn("verified", result.lower())

    def test_crosscheck_mismatch(self):
        from muse.tools.citation import make_crosscheck_metadata_tool

        tool = make_crosscheck_metadata_tool(_FakeMetadataClient(metadata_match=False))
        result = tool.invoke(
            {
                "title": "Fake Paper",
                "authors": "Nobody",
                "year": "2024",
                "doi": "",
            }
        )
        self.assertIn("mismatch", result.lower())

    def test_crosscheck_tool_name(self):
        from muse.tools.citation import make_crosscheck_metadata_tool

        tool = make_crosscheck_metadata_tool(_FakeMetadataClient())
        self.assertEqual(tool.name, "crosscheck_metadata")

    def test_crosscheck_is_base_tool(self):
        from langchain_core.tools import BaseTool
        from muse.tools.citation import make_crosscheck_metadata_tool

        tool = make_crosscheck_metadata_tool(_FakeMetadataClient())
        self.assertIsInstance(tool, BaseTool)


if __name__ == "__main__":
    unittest.main()
