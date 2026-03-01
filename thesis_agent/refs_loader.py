"""Local reference file ingestion for the thesis agent.

Scans a refs_dir for PDF, Markdown/TXT, and DOCX files, extracts text and
metadata, and returns them in the same dict shape used by AcademicSearchClient.

Public API
----------
load_local_refs(refs_dir: str) -> list[dict]
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

_SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".docx"}


def load_local_refs(refs_dir: str) -> list[dict[str, Any]]:
    """Scan refs_dir and return a list of reference dicts.

    Each dict has the same shape as API-sourced references:
        ref_id          str     e.g. "@local_smith2021security"
        title           str     inferred from filename (title-cased stem)
        authors         list    always [] for local files
        year            int|None  extracted from filename YYYY pattern
        doi             None
        venue           None
        abstract        str     first 1000 chars of extracted text
        source          str     "local"
        filepath        str     absolute path to source file
        full_text       str     complete extracted text (used by RAG)
        verified_metadata bool  False
    """
    refs_dir_path = Path(refs_dir).resolve()
    if not refs_dir_path.is_dir():
        return []

    results: list[dict[str, Any]] = []
    for path in sorted(refs_dir_path.iterdir()):
        if path.is_dir():
            continue  # skip .index/ and any subdirectory
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            continue
        try:
            ref = _ingest_file(path)
            if ref is not None:
                results.append(ref)
        except Exception:  # noqa: BLE001
            # Silently skip unreadable files so one bad file doesn't break the pipeline
            pass
    return results


def _ingest_file(path: Path) -> dict[str, Any] | None:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        full_text = _extract_pdf(path)
    elif suffix in {".md", ".txt"}:
        full_text = _extract_text(path)
    elif suffix == ".docx":
        full_text = _extract_docx(path)
    else:
        return None

    if not full_text or not full_text.strip():
        return None

    stem = path.stem
    ref_id = _local_ref_id(stem)
    title = _stem_to_title(stem)
    year = _extract_year_from_stem(stem)
    abstract = full_text.strip()[:1000]

    return {
        "ref_id": ref_id,
        "title": title,
        "authors": [],
        "year": year,
        "doi": None,
        "venue": None,
        "abstract": abstract,
        "source": "local",
        "filepath": str(path),
        "full_text": full_text,
        "verified_metadata": False,
    }


def _local_ref_id(stem: str) -> str:
    """Convert a filename stem to a safe ref_id like @local_smith2021security."""
    safe = re.sub(r"[^a-zA-Z0-9]", "_", stem).lower()
    safe = re.sub(r"_+", "_", safe).strip("_")
    return f"@local_{safe}"


def _stem_to_title(stem: str) -> str:
    """Convert a filename stem to a readable title."""
    cleaned = re.sub(r"[_\-]+", " ", stem)
    return cleaned.strip().title()


def _extract_year_from_stem(stem: str) -> int | None:
    """Try to extract a 4-digit year (1900–2099) from the filename stem."""
    # Use digit-boundary lookaheads instead of \b so underscores don't interfere
    matches = re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", stem)
    return int(matches[-1]) if matches else None


def _extract_pdf(path: Path) -> str:
    """Extract text from a PDF. Tries pdfminer.six first, then pypdf."""
    try:
        from pdfminer.high_level import extract_text as _pdfminer_extract  # type: ignore
        return _pdfminer_extract(str(path)) or ""
    except ImportError:
        pass
    try:
        import pypdf  # type: ignore
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        pass
    raise RuntimeError(
        f"Cannot read PDF '{path.name}': install pdfminer.six or pypdf.\n"
        "  pip install pdfminer.six"
    )


def _extract_text(path: Path) -> str:
    """Read a plain-text or Markdown file."""
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_docx(path: Path) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        import docx  # type: ignore
        doc = docx.Document(str(path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        raise RuntimeError(
            f"Cannot read DOCX '{path.name}': install python-docx.\n"
            "  pip install python-docx"
        )
