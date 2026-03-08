"""LangChain tool wrapping Muse's academic search client."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field


class AcademicSearchInput(BaseModel):
    """Input schema for the academic literature search tool."""

    query: str = Field(description="Search query for academic papers")


class AcademicSearchTool(BaseTool):
    """Search Semantic Scholar, OpenAlex, and arXiv for academic literature."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "academic_search"
    description: str = (
        "Search academic databases for papers relevant to a query. "
        "Returns titles, authors, year, DOI, venue, and abstracts."
    )
    args_schema: type[BaseModel] = AcademicSearchInput
    search_client: Any = None
    default_discipline: str = ""

    def _run(self, query: str) -> str:
        papers, _queries = self.search_client.search_multi_source(
            topic=query,
            discipline=self.default_discipline,
        )
        if not papers:
            return "No papers found for the given query."

        lines = [f"Found {len(papers)} paper(s):"]
        for index, paper in enumerate(papers[:20], start=1):
            authors = ", ".join(paper.get("authors", [])[:3]) or "Unknown authors"
            title = paper.get("title", "Untitled")
            year = paper.get("year", "n/a")
            venue = paper.get("venue") or "unknown venue"
            doi = paper.get("doi") or "no DOI"
            abstract = (paper.get("abstract") or "")[:200]
            lines.append(
                f"\n{index}. [{paper.get('ref_id', '')}] {title}\n"
                f"   Authors: {authors}\n"
                f"   Year: {year} | Venue: {venue} | DOI: {doi}\n"
                f"   Abstract: {abstract}"
            )
        return "\n".join(lines)


def make_academic_search_tool(search_client: Any, default_discipline: str = "") -> BaseTool:
    """Create the academic_search tool from an existing search client."""

    return AcademicSearchTool(
        search_client=search_client,
        default_discipline=default_discipline,
    )
