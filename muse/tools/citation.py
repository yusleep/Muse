"""LangChain tools for citation verification and metadata cross-checking."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field


class VerifyDoiInput(BaseModel):
    """Input schema for DOI verification."""

    doi: str = Field(description="The DOI string to verify")


class VerifyDoiTool(BaseTool):
    """Check whether a DOI resolves in CrossRef."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "verify_doi"
    description: str = (
        "Check whether a DOI is valid and exists in CrossRef. "
        "Returns whether the DOI is valid or invalid."
    )
    args_schema: type[BaseModel] = VerifyDoiInput
    metadata_client: Any = None

    def _run(self, doi: str) -> str:
        is_valid = self.metadata_client.verify_doi(doi)
        if is_valid:
            return f"DOI '{doi}' is valid and exists in CrossRef."
        return f"DOI '{doi}' is invalid or not found in CrossRef."


def make_verify_doi_tool(metadata_client: Any) -> BaseTool:
    """Create the ``verify_doi`` tool from a metadata client."""

    return VerifyDoiTool(metadata_client=metadata_client)


class CrosscheckMetadataInput(BaseModel):
    """Input schema for citation metadata cross-checking."""

    title: str = Field(description="Paper title")
    authors: str = Field(default="", description="Comma-separated author names")
    year: str = Field(default="", description="Publication year")
    doi: str = Field(default="", description="DOI if available")


class CrosscheckMetadataTool(BaseTool):
    """Verify citation metadata against CrossRef-like records."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "crosscheck_metadata"
    description: str = (
        "Verify a citation's metadata against CrossRef records and return "
        "whether it is verified or has a mismatch."
    )
    args_schema: type[BaseModel] = CrosscheckMetadataInput
    metadata_client: Any = None

    def _run(self, title: str, authors: str = "", year: str = "", doi: str = "") -> str:
        reference = {
            "title": title,
            "authors": [author.strip() for author in authors.split(",") if author.strip()],
            "year": int(year) if year.isdigit() else None,
            "doi": doi or None,
        }
        is_match = self.metadata_client.crosscheck_metadata(reference)
        if is_match:
            return f"Citation metadata verified: '{title}' matches CrossRef records."
        return f"Citation metadata mismatch: '{title}' does not match CrossRef records."


def make_crosscheck_metadata_tool(metadata_client: Any) -> BaseTool:
    """Create the ``crosscheck_metadata`` tool from a metadata client."""

    return CrosscheckMetadataTool(metadata_client=metadata_client)
