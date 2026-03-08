"""Research tools: web search, academic search, PDF reading, and local retrieval."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for general knowledge relevant to thesis writing."""

    return f"[web_search] No web search provider configured. Query: {query}"


@tool
def web_fetch(url: str, prompt: str = "Extract the main content.") -> str:
    """Fetch a web page and return a truncated text body."""

    import urllib.request

    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Muse/1.0"})
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read().decode("utf-8", errors="replace")
        return raw[:8000]
    except Exception as exc:  # noqa: BLE001
        return f"[web_fetch] Error fetching {url}: {exc}"


@tool
def academic_search(query: str, max_results: int = 10) -> str:
    """Search academic databases for papers and return a JSON list."""

    from muse.tools._context import get_services

    services = get_services()
    search_client = getattr(services, "search", None)
    if search_client is None:
        return json.dumps([])

    try:
        try:
            results, _queries = search_client.search_multi_source(
                topic=query,
                discipline="",
            )
        except TypeError:
            results = search_client.search_multi_source(query, max_results=max_results)
        if not isinstance(results, list):
            results = []
        return json.dumps(results[:max_results], ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@tool
def retrieve_local_refs(query: str, top_k: int = 5) -> str:
    """Retrieve relevant local reference passages via RAG."""

    from muse.tools._context import get_services

    services = get_services()
    rag_index = getattr(services, "rag_index", None)
    if rag_index is None:
        return json.dumps([])

    try:
        results = rag_index.retrieve(query, top_k=top_k)
        return json.dumps(results, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return json.dumps([])


@tool
def read_pdf(file_path: str, pages: str = "1-5") -> str:
    """Read text from selected PDF pages."""

    if not os.path.isfile(file_path):
        return f"[read_pdf] Error: file not found: {file_path}"

    try:
        import fitz
    except ImportError:
        return "[read_pdf] Error: PyMuPDF (fitz) not installed. Install with: pip install PyMuPDF"

    try:
        document = fitz.open(file_path)
    except Exception as exc:  # noqa: BLE001
        return f"[read_pdf] Error opening {file_path}: {exc}"

    start, end = 0, min(5, len(document))
    if "-" in pages:
        start_text, end_text = pages.split("-", 1)
        try:
            start = max(0, int(start_text) - 1)
            end = min(len(document), int(end_text))
        except ValueError:
            pass
    else:
        try:
            start = max(0, int(pages) - 1)
            end = start + 1
        except ValueError:
            pass

    text_parts = []
    for page_num in range(start, min(end, len(document))):
        page = document[page_num]
        text_parts.append(f"--- Page {page_num + 1} ---\n{page.get_text()}")
    document.close()
    return "\n".join(text_parts)[:12000]


@tool
def image_search(query: str) -> str:
    """Search for relevant images/figures."""

    return f"[image_search] No image search provider configured. Query: {query}"
