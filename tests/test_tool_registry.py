"""Tests for the ToolRegistry."""

from __future__ import annotations

import unittest

from langchain_core.tools import tool


@tool
def search_papers(query: str) -> str:
    """Search for academic papers by query."""

    return f"results for {query}"


@tool
def verify_doi(doi: str) -> bool:
    """Verify a DOI exists via CrossRef."""

    return True


@tool
def write_section(outline: str, references: str) -> str:
    """Write a thesis section from outline and references."""

    return "written section"


class ToolRegistryTests(unittest.TestCase):
    def test_register_and_get_by_group(self):
        from muse.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(search_papers, group="research")
        tools = registry.get_tools(groups=["research"])
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "search_papers")

    def test_register_multiple_groups(self):
        from muse.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(search_papers, group="research")
        registry.register(verify_doi, group="review")
        registry.register(write_section, group="writing")

        research_tools = registry.get_tools(groups=["research"])
        self.assertEqual(len(research_tools), 1)

        review_tools = registry.get_tools(groups=["review"])
        self.assertEqual(len(review_tools), 1)

    def test_get_tools_multiple_groups(self):
        from muse.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(search_papers, group="research")
        registry.register(verify_doi, group="review")
        registry.register(write_section, group="writing")

        tools = registry.get_tools(groups=["research", "review"])
        names = {tool_item.name for tool_item in tools}
        self.assertEqual(names, {"search_papers", "verify_doi"})

    def test_get_tools_for_profile(self):
        from muse.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(search_papers, group="research")
        registry.register(verify_doi, group="review")
        registry.register(write_section, group="writing")

        registry.define_profile("chapter", groups=["research", "writing"])
        registry.define_profile("citation", groups=["review"])

        chapter_tools = registry.get_tools_for_profile("chapter")
        names = {tool_item.name for tool_item in chapter_tools}
        self.assertEqual(names, {"search_papers", "write_section"})

        citation_tools = registry.get_tools_for_profile("citation")
        self.assertEqual(len(citation_tools), 1)
        self.assertEqual(citation_tools[0].name, "verify_doi")

    def test_unknown_profile_returns_empty(self):
        from muse.tools.registry import ToolRegistry

        registry = ToolRegistry()
        tools = registry.get_tools_for_profile("nonexistent")
        self.assertEqual(tools, [])

    def test_unknown_group_returns_empty(self):
        from muse.tools.registry import ToolRegistry

        registry = ToolRegistry()
        tools = registry.get_tools(groups=["nonexistent"])
        self.assertEqual(tools, [])

    def test_list_groups(self):
        from muse.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(search_papers, group="research")
        registry.register(verify_doi, group="review")
        self.assertEqual(sorted(registry.list_groups()), ["research", "review"])

    def test_list_profiles(self):
        from muse.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.define_profile("chapter", groups=["research", "writing"])
        registry.define_profile("citation", groups=["review"])
        self.assertEqual(sorted(registry.list_profiles()), ["chapter", "citation"])

    def test_no_duplicate_tools_across_groups(self):
        from muse.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(search_papers, group="research")
        registry.register(search_papers, group="extra")
        registry.define_profile("all", groups=["research", "extra"])
        tools = registry.get_tools_for_profile("all")
        names = [tool_item.name for tool_item in tools]
        self.assertEqual(len(names), len(set(names)), "tools should be deduplicated")


if __name__ == "__main__":
    unittest.main()
