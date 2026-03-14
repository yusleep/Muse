"""Regression tests for ReAct tool schema generation."""

from __future__ import annotations

import unittest

from langchain_core.utils.function_calling import convert_to_openai_tool


class ReactToolSchemaTests(unittest.TestCase):
    def test_runtime_injected_tools_produce_openai_schema(self):
        from muse.tools.citation import crosscheck_metadata, entailment_check, verify_doi
        from muse.tools.composition import check_terminology, check_transitions, rewrite_passage
        from muse.tools.research import academic_search, retrieve_local_refs, web_search
        from muse.tools.review import self_review
        from muse.tools.writing import revise_section, write_section

        tools = [
            verify_doi,
            crosscheck_metadata,
            entailment_check,
            academic_search,
            retrieve_local_refs,
            web_search,
            write_section,
            revise_section,
            self_review,
            check_terminology,
            check_transitions,
            rewrite_passage,
        ]

        for tool in tools:
            with self.subTest(tool=tool.name):
                schema = convert_to_openai_tool(tool)
                properties = schema["function"]["parameters"].get("properties", {})
                self.assertNotIn("runtime", properties)


if __name__ == "__main__":
    unittest.main()
