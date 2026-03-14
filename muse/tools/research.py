"""Research tools: web search, academic search, PDF reading, and local retrieval."""

import json
import os
import re
from typing import Annotated
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg
from langchain_core.tools import tool

from muse.tools._context import AgentRuntimeContext

MuseToolRuntime = ToolRuntime[AgentRuntimeContext, Any]


def _tokenize_query(query: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", str(query or "").lower())


def _reference_text(reference: dict[str, Any]) -> str:
    authors = reference.get("authors", [])
    if isinstance(authors, list):
        author_text = " ".join(str(author) for author in authors)
    else:
        author_text = str(authors or "")

    return " ".join(
        str(part or "")
        for part in (
            reference.get("ref_id"),
            reference.get("title"),
            reference.get("abstract"),
            author_text,
            reference.get("venue"),
        )
    ).lower()


def _search_state_references(
    *,
    query: str,
    top_k: int,
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    references = state.get("references", [])
    if not isinstance(references, list):
        return []

    tokens = _tokenize_query(query)
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for index, item in enumerate(references):
        if not isinstance(item, dict):
            continue
        haystack = _reference_text(item)
        score = 0
        for token in tokens:
            if token and token in haystack:
                score += max(1, len(token))
        if score > 0:
            scored.append((score, -index, item))

    if scored:
        scored.sort(reverse=True)
        return [item for _score, _neg_index, item in scored[:top_k]]

    # If the current state already has references but lexical matching fails,
    # expose the leading entries instead of returning [] and encouraging loops.
    fallback_refs = [item for item in references if isinstance(item, dict)]
    return fallback_refs[:top_k]


def _services_from_runtime(runtime: MuseToolRuntime | None) -> Any:
    from muse.tools._context import get_services, services_from_runtime

    services = services_from_runtime(runtime)
    return services if services is not None else get_services()


def _state_from_runtime(runtime: MuseToolRuntime | None) -> dict[str, Any]:
    if runtime is not None and isinstance(getattr(runtime, "state", None), dict):
        return runtime.state

    from muse.tools._context import get_state

    state = get_state()
    return state if isinstance(state, dict) else {}


@tool
def web_search(
    query: str,
    *,
    runtime: Annotated[MuseToolRuntime, InjectedToolArg],
) -> str:
    """Search the web for general knowledge relevant to thesis writing."""

    services = _services_from_runtime(runtime)
    web_search_client = getattr(services, "web_search_client", None)
    if web_search_client is None:
        return f"[web_search] No web search provider configured. Query: {query}"

    try:
        results = web_search_client.search(query)
    except Exception as exc:  # noqa: BLE001
        return f"[web_search] Search failed: {exc}. Query: {query}"

    return json.dumps(results, ensure_ascii=False)


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
def academic_search(
    query: str,
    max_results: int = 10,
    *,
    runtime: Annotated[MuseToolRuntime, InjectedToolArg],
) -> str:
    """Search academic databases for papers and return a JSON list."""

    services = _services_from_runtime(runtime)
    tool_state = _state_from_runtime(runtime)
    search_client = getattr(services, "search", None)
    if search_client is None:
        return json.dumps([])

    try:
        try:
            results, _queries = search_client.search_multi_source(
                topic=query,
                discipline=str(tool_state.get("discipline", "")).strip(),
            )
        except TypeError:
            results = search_client.search_multi_source(query, max_results=max_results)
        if not isinstance(results, list):
            results = []
        return json.dumps(results[:max_results], ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@tool
def retrieve_local_refs(
    query: str,
    top_k: int = 5,
    *,
    runtime: Annotated[MuseToolRuntime, InjectedToolArg],
) -> str:
    """Retrieve relevant local reference passages via RAG."""

    services = _services_from_runtime(runtime)
    tool_state = _state_from_runtime(runtime)
    rag_index = getattr(services, "rag_index", None)

    results: list[dict[str, Any]] = []
    if rag_index is not None:
        try:
            raw_results = rag_index.retrieve(query, top_k=top_k)
            if isinstance(raw_results, list):
                results = raw_results
        except Exception:  # noqa: BLE001
            results = []

    if not results and isinstance(tool_state, dict):
        results = _search_state_references(
            query=query,
            top_k=top_k,
            state=tool_state,
        )

    return json.dumps(results, ensure_ascii=False)


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
