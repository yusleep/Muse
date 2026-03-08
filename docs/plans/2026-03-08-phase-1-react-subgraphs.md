# Phase 1: Sub-graph ReAct Conversion

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert chapter, citation, and composition sub-graphs from fixed node flows to LangChain ReAct agents with tool-calling.

**Architecture:** Each sub-graph becomes a create_react_agent with a curated tool set. Top-level pipeline unchanged. Fan-out pattern preserved.

**Tech Stack:** LangChain (create_react_agent), LangGraph, Python 3.10

**Depends on:** Phase 0-A (MuseChatModel, ToolRegistry), Phase 0-C (Skills injection)

---

## Task 1 — Writing Tools (`muse/tools/writing.py`)

**Files:**
- Create: `muse/tools/writing.py`
- Read (context only): `muse/graph/helpers/draft_support.py`, `muse/prompts/section_write.py`

**TDD — write tests first in `tests/test_tools_writing.py`:**

```python
"""Tests for muse/tools/writing.py"""
import unittest
from unittest.mock import MagicMock


class _FakeLLM:
    def structured(self, *, system, user, route="default", max_tokens=2500):
        return {
            "text": "Generated section text about the topic.",
            "citations_used": ["@smith2024"],
            "key_claims": ["Claim A."],
            "transition_out": "",
            "glossary_additions": {},
            "self_assessment": {"confidence": 0.7, "weak_spots": [], "needs_revision": False},
        }

    def text(self, *, system, user, route="default", max_tokens=2500):
        return "Revised section text."


class WriteSectionToolTests(unittest.TestCase):
    def test_write_section_returns_text_and_citations(self):
        from muse.tools.writing import write_section
        tool = write_section
        result = tool.invoke({
            "chapter_title": "Introduction",
            "subtask_id": "sub_01",
            "subtask_title": "Background",
            "target_words": 1200,
            "topic": "LangGraph thesis automation",
            "language": "zh",
            "references_json": '[{"ref_id": "@smith2024", "title": "Graph Systems", "year": 2024, "abstract": "A study."}]',
        })
        self.assertIsInstance(result, str)
        self.assertIn("text", result)

    def test_revise_section_returns_revised_text(self):
        from muse.tools.writing import revise_section
        result = revise_section.invoke({
            "section_text": "Original text here.",
            "instruction": "Improve transitions between paragraphs.",
            "chapter_title": "Introduction",
            "language": "zh",
        })
        self.assertIsInstance(result, str)

    def test_apply_patch_replaces_old_with_new(self):
        from muse.tools.writing import apply_patch
        result = apply_patch.invoke({
            "section_text": "The quick brown fox jumps over the lazy dog.",
            "old_string": "quick brown fox",
            "new_string": "slow red cat",
        })
        self.assertIn("slow red cat", result)
        self.assertNotIn("quick brown fox", result)

    def test_apply_patch_reports_not_found(self):
        from muse.tools.writing import apply_patch
        result = apply_patch.invoke({
            "section_text": "Hello world.",
            "old_string": "nonexistent string",
            "new_string": "replacement",
        })
        self.assertIn("not found", result.lower())


if __name__ == "__main__":
    unittest.main()
```

**Implementation — `muse/tools/writing.py`:**

```python
"""Writing tools for the chapter ReAct agent."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool


@tool
def write_section(
    chapter_title: str,
    subtask_id: str,
    subtask_title: str,
    target_words: int,
    topic: str,
    language: str,
    references_json: str,
    revision_instruction: str = "",
    previous_subsection: str = "",
) -> str:
    """Write a thesis subsection given an outline subtask.

    Args:
        chapter_title: Title of the chapter this section belongs to.
        subtask_id: Unique ID of the subtask (e.g. "sub_01").
        subtask_title: Human-readable title of the subtask.
        target_words: Target word count for this section.
        topic: Overall thesis topic.
        language: Writing language ("zh" or "en").
        references_json: JSON array of reference objects with ref_id, title, year, abstract.
        revision_instruction: Optional revision instruction from a prior review.
        previous_subsection: Text of the preceding subsection for continuity.

    Returns:
        JSON string with keys: text, citations_used, key_claims, self_assessment.
    """
    from muse.tools._context import get_services

    services = get_services()
    llm = getattr(services, "llm", None)
    if llm is None:
        return json.dumps({"text": f"[{chapter_title}] {subtask_title}\n\n(No LLM available.)", "citations_used": [], "key_claims": []})

    try:
        refs = json.loads(references_json)
    except (json.JSONDecodeError, TypeError):
        refs = []

    refs_snapshot = [
        {"ref_id": r.get("ref_id", ""), "title": r.get("title", ""), "year": r.get("year"), "abstract": (r.get("abstract") or "")[:300]}
        for r in refs if isinstance(r, dict) and r.get("ref_id")
    ][:30]

    system = (
        "Write one thesis subsection with citations. "
        "IMPORTANT: for citations_used, use ONLY ref_id values from the available_references list. "
        "Do not invent citation keys not in that list. "
        "Include specific technical details, mathematical notation where appropriate, "
        "and reference concrete experimental results. "
        "Return JSON with keys: text, citations_used (list of ref_id strings), key_claims (list), "
        "transition_out, glossary_additions (object), "
        "self_assessment (object with confidence, weak_spots, needs_revision)."
    )
    user_payload: dict[str, Any] = {
        "topic": topic,
        "chapter_title": chapter_title,
        "subtask": {"subtask_id": subtask_id, "title": subtask_title, "target_words": target_words},
        "language": language,
        "available_references": refs_snapshot,
        "allowed_refs": [r["ref_id"] for r in refs_snapshot],
        "previous_subsection": previous_subsection,
        "revision_instruction": revision_instruction or None,
    }
    user = json.dumps(user_payload, ensure_ascii=False)

    try:
        out = llm.structured(system=system, user=user, route="writing", max_tokens=2800)
    except Exception:
        out = {"text": f"[{chapter_title}] {subtask_title}\n\n(LLM call failed.)", "citations_used": [], "key_claims": []}

    return json.dumps(out, ensure_ascii=False)


@tool
def revise_section(
    section_text: str,
    instruction: str,
    chapter_title: str,
    language: str,
) -> str:
    """Revise an existing thesis section per a review instruction.

    Args:
        section_text: The current text of the section to revise.
        instruction: Specific revision instruction (e.g. "improve transitions").
        chapter_title: Title of the chapter for context.
        language: Writing language ("zh" or "en").

    Returns:
        The revised section text.
    """
    from muse.tools._context import get_services

    services = get_services()
    llm = getattr(services, "llm", None)
    if llm is None:
        return section_text

    system = (
        f"Revise the following thesis section from chapter '{chapter_title}'. "
        f"Language: {language}. Follow the instruction precisely. "
        "Return ONLY the revised text, no JSON wrapping."
    )
    user = json.dumps({"instruction": instruction, "text": section_text}, ensure_ascii=False)

    try:
        return llm.text(system=system, user=user, route="writing", max_tokens=2800)
    except Exception:
        return section_text


@tool
def apply_patch(
    section_text: str,
    old_string: str,
    new_string: str,
) -> str:
    """Apply a targeted text replacement within a section.

    Args:
        section_text: The full section text.
        old_string: The exact substring to find and replace.
        new_string: The replacement string.

    Returns:
        The updated section text, or an error message if old_string was not found.
    """
    if old_string not in section_text:
        return f"ERROR: old_string not found in section_text. Ensure the old_string matches exactly."
    return section_text.replace(old_string, new_string, 1)
```

**Also create the service-context helper — `muse/tools/_context.py`:**

```python
"""Thread-local service context for tool functions."""
from __future__ import annotations

import threading
from typing import Any


_local = threading.local()


class _NullServices:
    llm = None
    search = None
    metadata = None
    rag_index = None
    local_refs: list = []


def set_services(services: Any) -> None:
    """Set the services object for the current thread."""
    _local.services = services


def get_services() -> Any:
    """Get the services object for the current thread."""
    return getattr(_local, "services", _NullServices())
```

**Also create `muse/tools/__init__.py` (update existing):**

```python
"""Muse tool definitions for ReAct sub-graph agents."""
```

**Steps:**
1. Create `muse/tools/_context.py` with thread-local services
2. Update `muse/tools/__init__.py` docstring
3. Create `muse/tools/writing.py` with `write_section`, `revise_section`, `apply_patch`
4. Create `tests/test_tools_writing.py`
5. Run: `cd /home/planck/gradute/Muse && python -m pytest tests/test_tools_writing.py -v`
6. Verify all 4 tests pass

---

## Task 2 — Review Tools (`muse/tools/review.py`)

**Files:**
- Create: `muse/tools/review.py`
- Read (context only): `muse/graph/nodes/review.py`, `muse/prompts/chapter_review.py`

**TDD — write tests first in `tests/test_tools_review.py`:**

```python
"""Tests for muse/tools/review.py"""
import unittest
import json


class _FakeReviewLLM:
    def structured(self, *, system, user, route="default", max_tokens=2500):
        return {
            "scores": {"coherence": 4, "logic": 3, "citation": 5, "term_consistency": 4, "balance": 4, "redundancy": 4},
            "review_notes": [
                {"subtask_id": "sub_01", "issue": "Weak logic flow", "instruction": "Add transition.", "severity": 3}
            ],
        }


class SelfReviewToolTests(unittest.TestCase):
    def test_self_review_returns_json_with_scores_and_notes(self):
        from muse.tools.review import self_review
        result_str = self_review.invoke({
            "chapter_title": "Introduction",
            "merged_text": "This is the chapter text.",
            "lenses": "logic,style,citation,structure",
        })
        result = json.loads(result_str)
        self.assertIn("scores", result)
        self.assertIn("review_notes", result)
        self.assertIn("revision_instructions", result)

    def test_self_review_with_single_lens(self):
        from muse.tools.review import self_review
        result_str = self_review.invoke({
            "chapter_title": "Methods",
            "merged_text": "Methods text.",
            "lenses": "logic",
        })
        result = json.loads(result_str)
        self.assertIn("scores", result)


if __name__ == "__main__":
    unittest.main()
```

**Implementation — `muse/tools/review.py`:**

```python
"""Review tools for multi-lens chapter evaluation."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from muse.graph.helpers.review_state import build_revision_instructions
from muse.prompts.chapter_review import chapter_review_prompt


@tool
def self_review(
    chapter_title: str,
    merged_text: str,
    lenses: str = "logic,style,citation,structure",
) -> str:
    """Run a multi-lens quality review on a chapter draft.

    Evaluates the chapter text across multiple quality dimensions and returns
    structured scores, review notes, and revision instructions.

    Args:
        chapter_title: Title of the chapter being reviewed.
        merged_text: The full chapter text to review.
        lenses: Comma-separated review lenses (e.g. "logic,style,citation,structure").

    Returns:
        JSON string with keys: scores, review_notes, revision_instructions.
    """
    from muse.tools._context import get_services

    services = get_services()
    llm = getattr(services, "llm", None)
    lens_list = [lens.strip() for lens in lenses.split(",") if lens.strip()]
    if not lens_list:
        lens_list = ["logic", "style", "citation", "structure"]

    packets: list[dict[str, Any]] = []

    if llm is not None:
        for lens in lens_list:
            system, user = chapter_review_prompt(chapter_title, merged_text)
            system = f"{system} Focus primarily on {lens}."
            try:
                payload = llm.structured(system=system, user=user, route="review", max_tokens=1800)
            except Exception:
                payload = {}
            if isinstance(payload, dict):
                packets.append(payload)

    scores: dict[str, int] = {}
    review_notes: list[dict[str, Any]] = []
    for packet in packets:
        packet_scores = packet.get("scores", {})
        if isinstance(packet_scores, dict):
            for key, value in packet_scores.items():
                if isinstance(value, (int, float)):
                    scores[key] = min(scores.get(key, int(value)), int(value)) if key in scores else int(value)
        packet_notes = packet.get("review_notes", [])
        if isinstance(packet_notes, list):
            review_notes.extend(note for note in packet_notes if isinstance(note, dict))

    revision_instructions = build_revision_instructions(review_notes, min_severity=2)

    return json.dumps({
        "scores": scores,
        "review_notes": review_notes,
        "revision_instructions": revision_instructions,
    }, ensure_ascii=False)
```

**Steps:**
1. Create `muse/tools/review.py`
2. Create `tests/test_tools_review.py`
3. Run: `cd /home/planck/gradute/Muse && python -m pytest tests/test_tools_review.py -v`
4. Verify both tests pass

---

## Task 3 — Research Tools (`muse/tools/research.py`)

**Files:**
- Create: `muse/tools/research.py`
- Read (context only): `muse/services/search.py`, `muse/services/providers.py`, `muse/adapters/__init__.py`

**TDD — write tests first in `tests/test_tools_research.py`:**

```python
"""Tests for muse/tools/research.py"""
import unittest
import json


class ResearchToolTests(unittest.TestCase):
    def test_academic_search_returns_json_list(self):
        from muse.tools.research import academic_search
        result_str = academic_search.invoke({
            "query": "graph neural networks",
            "max_results": 3,
        })
        result = json.loads(result_str)
        self.assertIsInstance(result, list)

    def test_retrieve_local_refs_without_index(self):
        from muse.tools.research import retrieve_local_refs
        result_str = retrieve_local_refs.invoke({
            "query": "transformer architecture",
            "top_k": 5,
        })
        result = json.loads(result_str)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_web_search_returns_string(self):
        from muse.tools.research import web_search
        result = web_search.invoke({"query": "LangGraph documentation"})
        self.assertIsInstance(result, str)

    def test_web_fetch_returns_string(self):
        from muse.tools.research import web_fetch
        result = web_fetch.invoke({"url": "https://example.com", "prompt": "summarize"})
        self.assertIsInstance(result, str)

    def test_read_pdf_returns_string(self):
        from muse.tools.research import read_pdf
        result = read_pdf.invoke({"file_path": "/nonexistent/file.pdf", "pages": "1-3"})
        self.assertIn("error", result.lower())

    def test_image_search_returns_string(self):
        from muse.tools.research import image_search
        result = image_search.invoke({"query": "neural network architecture diagram"})
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
```

**Implementation — `muse/tools/research.py`:**

```python
"""Research tools: web search, academic search, PDF reading, local RAG retrieval."""
from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for general knowledge relevant to thesis writing.

    Args:
        query: Search query string.

    Returns:
        Search results as text, or an error message.
    """
    # Placeholder: will integrate with actual web search API in Phase 4
    return f"[web_search] No web search provider configured. Query: {query}"


@tool
def web_fetch(url: str, prompt: str = "Extract the main content.") -> str:
    """Fetch a web page and extract content as markdown.

    Args:
        url: The URL to fetch.
        prompt: Instruction for what to extract from the page.

    Returns:
        Extracted content as text, or an error message.
    """
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Muse/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        # Truncate to 8000 chars for LLM context budget
        return raw[:8000]
    except Exception as exc:
        return f"[web_fetch] Error fetching {url}: {exc}"


@tool
def academic_search(query: str, max_results: int = 10) -> str:
    """Search academic databases (Semantic Scholar, OpenAlex, arXiv) for papers.

    Args:
        query: Academic search query.
        max_results: Maximum number of results to return.

    Returns:
        JSON array of reference objects with ref_id, title, authors, year, doi, abstract.
    """
    from muse.tools._context import get_services

    services = get_services()
    search_client = getattr(services, "search", None)
    if search_client is None:
        return json.dumps([])

    try:
        results = search_client.search_multi_source(query, max_results=max_results)
        return json.dumps(results[:max_results], ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
def retrieve_local_refs(query: str, top_k: int = 5) -> str:
    """Retrieve relevant passages from locally indexed reference PDFs via RAG.

    Args:
        query: Semantic query to match against local references.
        top_k: Number of top results to return.

    Returns:
        JSON array of matching passages with metadata.
    """
    from muse.tools._context import get_services

    services = get_services()
    rag_index = getattr(services, "rag_index", None)
    if rag_index is None:
        return json.dumps([])

    try:
        results = rag_index.retrieve(query, top_k=top_k)
        return json.dumps(results, ensure_ascii=False)
    except Exception:
        return json.dumps([])


@tool
def read_pdf(file_path: str, pages: str = "1-5") -> str:
    """Read specific pages from a PDF file and return text content.

    Args:
        file_path: Absolute path to the PDF file.
        pages: Page range to read (e.g. "1-5", "3", "10-20").

    Returns:
        Extracted text content from the specified pages.
    """
    if not os.path.isfile(file_path):
        return f"[read_pdf] Error: file not found: {file_path}"

    try:
        import fitz  # PyMuPDF
    except ImportError:
        return f"[read_pdf] Error: PyMuPDF (fitz) not installed. Install with: pip install PyMuPDF"

    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        return f"[read_pdf] Error opening {file_path}: {exc}"

    # Parse page range
    start, end = 0, min(5, len(doc))
    if "-" in pages:
        parts = pages.split("-", 1)
        try:
            start = max(0, int(parts[0]) - 1)
            end = min(len(doc), int(parts[1]))
        except ValueError:
            pass
    else:
        try:
            start = max(0, int(pages) - 1)
            end = start + 1
        except ValueError:
            pass

    text_parts = []
    for page_num in range(start, end):
        page = doc[page_num]
        text_parts.append(f"--- Page {page_num + 1} ---\n{page.get_text()}")
    doc.close()

    result = "\n".join(text_parts)
    return result[:12000]  # Truncate for LLM context budget


@tool
def image_search(query: str) -> str:
    """Search for images relevant to thesis figures and diagrams.

    Args:
        query: Descriptive query for the image (e.g. "transformer architecture diagram").

    Returns:
        Search results or a placeholder message.
    """
    # Placeholder: will integrate with image search API in Phase 4
    return f"[image_search] No image search provider configured. Query: {query}"
```

**Steps:**
1. Create `muse/tools/research.py`
2. Create `tests/test_tools_research.py`
3. Run: `cd /home/planck/gradute/Muse && python -m pytest tests/test_tools_research.py -v`
4. Verify all 6 tests pass

---

## Task 4 — File Tools (`muse/tools/file.py`)

**Files:**
- Create: `muse/tools/file.py`

**TDD — write tests first in `tests/test_tools_file.py`:**

```python
"""Tests for muse/tools/file.py"""
import os
import tempfile
import unittest


class FileToolTests(unittest.TestCase):
    def test_read_file_existing(self):
        from muse.tools.file import read_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line1\nline2\nline3\n")
            path = f.name
        try:
            result = read_file.invoke({"file_path": path})
            self.assertIn("line1", result)
            self.assertIn("line2", result)
        finally:
            os.unlink(path)

    def test_read_file_nonexistent(self):
        from muse.tools.file import read_file
        result = read_file.invoke({"file_path": "/tmp/_nonexistent_file_xyz.txt"})
        self.assertIn("error", result.lower())

    def test_read_file_with_offset_limit(self):
        from muse.tools.file import read_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for i in range(20):
                f.write(f"line {i}\n")
            path = f.name
        try:
            result = read_file.invoke({"file_path": path, "offset": 5, "limit": 3})
            self.assertIn("line 5", result)
            self.assertNotIn("line 0", result)
        finally:
            os.unlink(path)

    def test_write_file_creates_and_writes(self):
        from muse.tools.file import write_file
        path = os.path.join(tempfile.gettempdir(), "_muse_test_write.txt")
        try:
            result = write_file.invoke({"file_path": path, "content": "hello world"})
            self.assertIn("ok", result.lower())
            with open(path) as f:
                self.assertEqual(f.read(), "hello world")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_edit_file_replaces_string(self):
        from muse.tools.file import edit_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("The quick brown fox.")
            path = f.name
        try:
            result = edit_file.invoke({"file_path": path, "old_string": "quick brown", "new_string": "slow red"})
            self.assertIn("ok", result.lower())
            with open(path) as f:
                self.assertIn("slow red", f.read())
        finally:
            os.unlink(path)

    def test_edit_file_old_string_not_found(self):
        from muse.tools.file import edit_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello world.")
            path = f.name
        try:
            result = edit_file.invoke({"file_path": path, "old_string": "xyz", "new_string": "abc"})
            self.assertIn("not found", result.lower())
        finally:
            os.unlink(path)

    def test_glob_finds_files(self):
        from muse.tools.file import glob_files
        result = glob_files.invoke({"pattern": "*.py", "directory": os.path.dirname(__file__)})
        self.assertIsInstance(result, str)

    def test_grep_searches_content(self):
        from muse.tools.file import grep
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello_world():\n    pass\n")
            path = f.name
        try:
            result = grep.invoke({"pattern": "hello_world", "path": os.path.dirname(path)})
            self.assertIn("hello_world", result)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
```

**Implementation — `muse/tools/file.py`:**

```python
"""File manipulation tools for ReAct agents."""
from __future__ import annotations

import fnmatch
import os
import re

from langchain_core.tools import tool


@tool
def read_file(file_path: str, offset: int = 0, limit: int = 2000) -> str:
    """Read a file from the filesystem with optional line offset and limit.

    Args:
        file_path: Absolute path to the file to read.
        offset: Line number to start reading from (0-indexed).
        limit: Maximum number of lines to read.

    Returns:
        File contents with line numbers, or an error message.
    """
    if not os.path.isfile(file_path):
        return f"[read_file] Error: file not found: {file_path}"

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as exc:
        return f"[read_file] Error reading {file_path}: {exc}"

    selected = lines[offset : offset + limit]
    numbered = []
    for i, line in enumerate(selected, start=offset + 1):
        truncated = line.rstrip("\n")[:2000]
        numbered.append(f"{i:>6}\t{truncated}")
    return "\n".join(numbered)


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed.

    Args:
        file_path: Absolute path to the file to write.
        content: The content to write to the file.

    Returns:
        OK message on success, or an error message.
    """
    try:
        parent = os.path.dirname(file_path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"OK: wrote {len(content)} bytes to {file_path}"
    except Exception as exc:
        return f"[write_file] Error writing {file_path}: {exc}"


@tool
def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """Perform an exact string replacement in a file.

    Args:
        file_path: Absolute path to the file to edit.
        old_string: The exact substring to find and replace.
        new_string: The replacement string.

    Returns:
        OK message on success, or an error message.
    """
    if not os.path.isfile(file_path):
        return f"[edit_file] Error: file not found: {file_path}"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as exc:
        return f"[edit_file] Error reading {file_path}: {exc}"

    if old_string not in content:
        return f"[edit_file] Error: old_string not found in {file_path}"

    new_content = content.replace(old_string, new_string, 1)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"OK: replaced in {file_path}"
    except Exception as exc:
        return f"[edit_file] Error writing {file_path}: {exc}"


@tool
def glob_files(pattern: str, directory: str = ".") -> str:
    """Find files matching a glob pattern in a directory.

    Args:
        pattern: Glob pattern (e.g. "*.py", "**/*.md").
        directory: Directory to search in (default: current directory).

    Returns:
        Newline-separated list of matching file paths.
    """
    matches: list[str] = []
    try:
        for root, _dirs, files in os.walk(directory):
            for fname in files:
                if fnmatch.fnmatch(fname, pattern):
                    matches.append(os.path.join(root, fname))
            if len(matches) > 200:
                break
    except Exception as exc:
        return f"[glob] Error: {exc}"

    if not matches:
        return f"[glob] No files matching '{pattern}' in {directory}"
    return "\n".join(sorted(matches)[:200])


@tool
def grep(pattern: str, path: str = ".", file_glob: str = "*") -> str:
    """Search file contents for a regex pattern.

    Args:
        pattern: Regular expression pattern to search for.
        path: File or directory to search in.
        file_glob: Glob pattern to filter files (e.g. "*.py").

    Returns:
        Matching lines with file path and line number prefixes.
    """
    results: list[str] = []
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return f"[grep] Invalid regex: {exc}"

    def search_file(fpath: str) -> None:
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f, 1):
                    if compiled.search(line):
                        results.append(f"{fpath}:{line_num}: {line.rstrip()[:200]}")
                    if len(results) > 100:
                        return
        except Exception:
            pass

    if os.path.isfile(path):
        search_file(path)
    elif os.path.isdir(path):
        for root, _dirs, files in os.walk(path):
            for fname in files:
                if fnmatch.fnmatch(fname, file_glob):
                    search_file(os.path.join(root, fname))
                if len(results) > 100:
                    break
            if len(results) > 100:
                break
    else:
        return f"[grep] Error: path not found: {path}"

    if not results:
        return f"[grep] No matches for '{pattern}' in {path}"
    return "\n".join(results[:100])
```

**Steps:**
1. Create `muse/tools/file.py`
2. Create `tests/test_tools_file.py`
3. Run: `cd /home/planck/gradute/Muse && python -m pytest tests/test_tools_file.py -v`
4. Verify all 8 tests pass

---

## Task 5 — Orchestration Tools (`muse/tools/orchestration.py`)

**Files:**
- Create: `muse/tools/orchestration.py`

**TDD — write tests first in `tests/test_tools_orchestration.py`:**

```python
"""Tests for muse/tools/orchestration.py"""
import unittest
import json


class SubmitResultTests(unittest.TestCase):
    def test_submit_result_returns_confirmation(self):
        from muse.tools.orchestration import submit_result
        result = submit_result.invoke({
            "result_json": '{"merged_text": "Chapter content.", "quality_scores": {"coherence": 4}}',
            "summary": "Chapter draft completed with 2 revisions.",
        })
        self.assertIn("submitted", result.lower())

    def test_submit_result_invalid_json(self):
        from muse.tools.orchestration import submit_result
        result = submit_result.invoke({
            "result_json": "not valid json{",
            "summary": "Bad data.",
        })
        self.assertIn("error", result.lower())


class UpdatePlanTests(unittest.TestCase):
    def test_update_plan_returns_confirmation(self):
        from muse.tools.orchestration import update_plan
        result = update_plan.invoke({
            "status": "drafting",
            "progress_pct": 45,
            "current_step": "Writing section 2.3",
            "notes": "References loaded, outline stable.",
        })
        self.assertIn("updated", result.lower())


if __name__ == "__main__":
    unittest.main()
```

**Implementation — `muse/tools/orchestration.py`:**

```python
"""Orchestration tools for ReAct agent control flow."""
from __future__ import annotations

import json
import logging
import threading
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Thread-local storage for the most recent submit_result payload.
_local = threading.local()


def get_submitted_result() -> dict[str, Any] | None:
    """Retrieve the most recent submit_result payload (used by subgraph wrapper)."""
    return getattr(_local, "submitted_result", None)


def clear_submitted_result() -> None:
    """Clear the stored result after consumption."""
    _local.submitted_result = None


@tool
def submit_result(
    result_json: str,
    summary: str,
) -> str:
    """Submit the final result of this sub-graph agent and signal completion.

    This is the TERMINATION tool. Call this when you are satisfied with the output.
    The result_json must be valid JSON containing the deliverables of this stage.

    Args:
        result_json: JSON string containing the stage result (e.g. merged_text, scores).
        summary: Brief human-readable summary of what was accomplished.

    Returns:
        Confirmation message.
    """
    try:
        payload = json.loads(result_json)
    except (json.JSONDecodeError, TypeError) as exc:
        return f"[submit_result] Error: invalid JSON — {exc}. Fix your JSON and try again."

    _local.submitted_result = {"payload": payload, "summary": summary}
    logger.info("submit_result: %s", summary)
    return f"SUBMITTED. Summary: {summary}"


@tool
def update_plan(
    status: str,
    progress_pct: int,
    current_step: str,
    notes: str = "",
) -> str:
    """Update the current progress of this agent's work for monitoring.

    Call periodically to report progress. Does NOT terminate the agent.

    Args:
        status: Current status (e.g. "researching", "drafting", "reviewing", "revising").
        progress_pct: Estimated completion percentage (0-100).
        current_step: Description of the current step being performed.
        notes: Optional notes about decisions or issues encountered.

    Returns:
        Confirmation message.
    """
    logger.info("update_plan: [%d%%] %s — %s %s", progress_pct, status, current_step, notes)
    return f"Plan updated: {status} ({progress_pct}%) — {current_step}"
```

**Steps:**
1. Create `muse/tools/orchestration.py`
2. Create `tests/test_tools_orchestration.py`
3. Run: `cd /home/planck/gradute/Muse && python -m pytest tests/test_tools_orchestration.py -v`
4. Verify all 3 tests pass

---

## Task 6 — Rewrite Chapter Subgraph as ReAct Agent

**Files:**
- Modify: `muse/graph/subgraphs/chapter.py`
- Create: `muse/prompts/chapter_agent.py`
- Read (context only): `muse/graph/state.py`, `muse/tools/writing.py`, `muse/tools/review.py`, `muse/tools/research.py`, `muse/tools/file.py`, `muse/tools/orchestration.py`

**TDD — write tests first in `tests/test_chapter_react_agent.py`:**

```python
"""Tests for the ReAct-based chapter subgraph."""
import json
import unittest
from unittest.mock import MagicMock, patch


class _FakeAgentLLM:
    """Simulates a ReAct agent that writes, reviews, then submits."""
    def __init__(self):
        self.call_count = 0

    def structured(self, *, system, user, route="default", max_tokens=2500):
        if "Write one thesis subsection" in system:
            return {
                "text": "Drafted subsection on graph orchestration.",
                "citations_used": ["@smith2024graph"],
                "key_claims": ["Graph orchestration improves reliability."],
                "transition_out": "",
                "glossary_additions": {},
                "self_assessment": {"confidence": 0.8, "weak_spots": [], "needs_revision": False},
            }
        if "strict thesis reviewer" in system:
            return {
                "scores": {"coherence": 4, "logic": 4, "citation": 4, "term_consistency": 4, "balance": 4, "redundancy": 4},
                "review_notes": [],
            }
        return {}


class _FakeServices:
    def __init__(self):
        self.llm = _FakeAgentLLM()
        self.rag_index = None
        self.search = None


class ChapterReActAgentTests(unittest.TestCase):
    def test_build_chapter_agent_returns_callable(self):
        from muse.graph.subgraphs.chapter import build_chapter_subgraph_node
        node_fn = build_chapter_subgraph_node(services=_FakeServices())
        self.assertTrue(callable(node_fn))

    def test_chapter_state_schema_has_required_fields(self):
        from muse.graph.subgraphs.chapter import ChapterState
        hints = ChapterState.__annotations__
        self.assertIn("chapter_plan", hints)
        self.assertIn("merged_text", hints)
        self.assertIn("quality_scores", hints)

    def test_chapter_agent_system_prompt_exists(self):
        from muse.prompts.chapter_agent import chapter_agent_system_prompt
        prompt = chapter_agent_system_prompt(
            topic="LangGraph thesis automation",
            language="zh",
            chapter_title="Introduction",
            chapter_plan={"chapter_id": "ch_01", "subtask_plan": []},
            references_summary="5 references available",
        )
        self.assertIn("chapter", prompt.lower())
        self.assertIn("submit", prompt.lower())


if __name__ == "__main__":
    unittest.main()
```

**Create system prompt — `muse/prompts/chapter_agent.py`:**

```python
"""System prompt for the chapter-writing ReAct agent."""
from __future__ import annotations

import json
from typing import Any


def chapter_agent_system_prompt(
    *,
    topic: str,
    language: str,
    chapter_title: str,
    chapter_plan: dict[str, Any],
    references_summary: str,
) -> str:
    """Build the system prompt for the chapter ReAct agent.

    The prompt guides the agent through a flexible workflow:
    research -> write sections -> self-review -> revise -> submit.
    """
    subtask_plan = chapter_plan.get("subtask_plan", [])
    subtask_summary = "\n".join(
        f"  - {s.get('subtask_id', '?')}: {s.get('title', '?')} (~{s.get('target_words', 1200)} words)"
        for s in subtask_plan
    )

    return f"""You are a thesis chapter writing agent. Your task is to produce a high-quality
chapter for an academic thesis.

## Context
- Topic: {topic}
- Language: {language}
- Chapter: {chapter_title}
- Chapter ID: {chapter_plan.get('chapter_id', 'chapter')}

## Subtask Plan
{subtask_summary}

## Available References
{references_summary}

## Workflow (suggested, not mandatory)
1. **Research**: Use `academic_search` or `retrieve_local_refs` to gather relevant material.
2. **Write**: Use `write_section` for each subtask in order. Pass the previous subsection text
   to maintain continuity.
3. **Review**: Use `self_review` to evaluate the merged chapter draft across multiple lenses
   (logic, style, citation, structure).
4. **Revise**: If any score < 4, use `revise_section` or `apply_patch` to address review notes.
   Repeat review-revise at most 3 times.
5. **Submit**: When all scores >= 4 (or max iterations reached), call `submit_result` with
   a JSON object containing:
   - "merged_text": the full chapter text
   - "subtask_results": list of per-subtask outputs
   - "quality_scores": final review scores
   - "citation_uses": list of {{"cite_key", "claim_id", "chapter_id", "subtask_id"}}
   - "claim_text_by_id": mapping of claim_id to claim text
   - "iterations_used": number of review-revise cycles

## Rules
- Use ONLY ref_id values from the provided references. Never invent citations.
- Write in {language} (zh = Chinese, en = English).
- Each subtask should produce ~target_words words.
- You MUST call `submit_result` to finish. The agent will be terminated after 30 turns if
  you do not submit.
- Use `update_plan` periodically to report progress.
"""
```

**Modify — `muse/graph/subgraphs/chapter.py`:**

```python
"""Chapter-level LangGraph subgraph — ReAct agent with tool-calling."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from muse.graph.helpers.review_state import should_iterate
from muse.graph.nodes.draft import build_chapter_draft_node
from muse.graph.nodes.review import build_chapter_review_node


class ChapterState(TypedDict, total=False):
    chapter_plan: dict[str, Any]
    references: list[dict[str, Any]]
    topic: str
    language: str
    subtask_results: list[dict[str, Any]]
    merged_text: str
    quality_scores: dict[str, int]
    review_notes: list[dict[str, Any]]
    revision_instructions: dict[str, str]
    iteration: int
    max_iterations: int
    citation_uses: list[dict[str, Any]]
    claim_text_by_id: dict[str, str]


def _chapter_route(state: ChapterState) -> Literal["revise", "done"]:
    route = should_iterate(
        {
            "quality_scores": state.get("quality_scores", {}),
            "current_iteration": state.get("iteration", 0),
            "max_iterations": state.get("max_iterations", 3),
        },
        threshold=4,
    )
    return "revise" if route == "revise" else "done"


def _chapter_revise(_: ChapterState) -> dict[str, Any]:
    return {}


def build_chapter_graph(*, services: Any):
    """Build the fixed-flow chapter graph (legacy, used as fallback)."""
    builder = StateGraph(ChapterState)
    builder.add_node("chapter_draft", build_chapter_draft_node(services))
    builder.add_node("chapter_review", build_chapter_review_node(services))
    builder.add_node("chapter_revise", _chapter_revise)
    builder.add_edge(START, "chapter_draft")
    builder.add_edge("chapter_draft", "chapter_review")
    builder.add_conditional_edges(
        "chapter_review",
        _chapter_route,
        {"revise": "chapter_revise", "done": END},
    )
    builder.add_edge("chapter_revise", "chapter_draft")
    return builder.compile()


def _build_react_chapter_agent(*, services: Any, settings: Any = None):
    """Build a ReAct agent for chapter writing using create_react_agent.

    Falls back to build_chapter_graph if create_react_agent or MuseChatModel
    are not available (Phase 0-A not yet integrated).
    """
    try:
        from langgraph.prebuilt import create_react_agent
    except ImportError:
        return None

    try:
        from muse.models.factory import create_chat_model
    except ImportError:
        return None

    if settings is None:
        return None

    from muse.tools.writing import write_section, revise_section, apply_patch
    from muse.tools.review import self_review
    from muse.tools.research import academic_search, retrieve_local_refs, web_search, web_fetch, read_pdf, image_search
    from muse.tools.file import read_file, write_file, edit_file, glob_files, grep
    from muse.tools.orchestration import submit_result, update_plan

    tools = [
        write_section, revise_section, apply_patch,
        self_review,
        academic_search, retrieve_local_refs, web_search, web_fetch, read_pdf, image_search,
        read_file, write_file, edit_file, glob_files, grep,
        submit_result, update_plan,
    ]

    try:
        model = create_chat_model(settings, route="writing")
    except Exception:
        return None

    agent = create_react_agent(
        model=model,
        tools=tools,
        state_schema=ChapterState,
    )
    return agent


def build_chapter_subgraph_node(*, services: Any, settings: Any = None):
    """Build the chapter subgraph node for the main pipeline.

    Attempts to use the ReAct agent if Phase 0-A is available; falls back
    to the fixed-flow StateGraph otherwise.
    """
    react_agent = _build_react_chapter_agent(services=services, settings=settings)

    if react_agent is not None:
        # ReAct agent path
        def run_react_chapter(state: dict[str, Any]) -> dict[str, Any]:
            from muse.tools._context import set_services
            from muse.tools.orchestration import get_submitted_result, clear_submitted_result
            from muse.prompts.chapter_agent import chapter_agent_system_prompt

            set_services(services)
            clear_submitted_result()

            chapter_plan = state.get("chapter_plan", {})
            chapter_id = chapter_plan.get("chapter_id", "chapter")
            refs = state.get("references", [])
            refs_summary = f"{len(refs)} references available. Top refs: " + ", ".join(
                r.get("ref_id", "?") for r in refs[:10]
            )

            system_prompt = chapter_agent_system_prompt(
                topic=state.get("topic", ""),
                language=state.get("language", "zh"),
                chapter_title=chapter_plan.get("chapter_title", ""),
                chapter_plan=chapter_plan,
                references_summary=refs_summary,
            )

            agent_input = {
                "messages": [{"role": "system", "content": system_prompt}],
                "chapter_plan": chapter_plan,
                "references": refs,
                "topic": state.get("topic", ""),
                "language": state.get("language", "zh"),
            }

            try:
                result = react_agent.invoke(agent_input, {"recursion_limit": 60})
            except Exception:
                # Fallback to fixed-flow on agent failure
                fallback = build_chapter_graph(services=services)
                result = fallback.invoke(state)
                return _extract_chapter_result(result, chapter_plan)

            submitted = get_submitted_result()
            clear_submitted_result()
            if submitted:
                payload = submitted["payload"]
                return {
                    "chapters": {chapter_id: {
                        "chapter_id": chapter_id,
                        "chapter_title": chapter_plan.get("chapter_title", chapter_id),
                        "merged_text": payload.get("merged_text", ""),
                        "quality_scores": payload.get("quality_scores", {}),
                        "iterations_used": payload.get("iterations_used", 0),
                        "subtask_results": payload.get("subtask_results", []),
                        "citation_uses": payload.get("citation_uses", []),
                        "claim_text_by_id": payload.get("claim_text_by_id", {}),
                    }},
                    "claim_text_by_id": payload.get("claim_text_by_id", {}),
                }

            # Agent did not call submit_result — extract what we can from state
            return _extract_chapter_result(result, chapter_plan)

        return run_react_chapter

    # Fallback: fixed-flow chapter graph
    chapter_graph = build_chapter_graph(services=services)

    def run_chapter_subgraph(state: dict[str, Any]) -> dict[str, Any]:
        result = chapter_graph.invoke(state)
        return _extract_chapter_result(result, state.get("chapter_plan", {}))

    return run_chapter_subgraph


def _extract_chapter_result(result: dict[str, Any], chapter_plan: dict[str, Any]) -> dict[str, Any]:
    """Extract standardized chapter result dict from graph output."""
    chapter_id = chapter_plan.get("chapter_id", "chapter")
    chapter_result = {
        "chapter_id": chapter_id,
        "chapter_title": chapter_plan.get("chapter_title", chapter_id),
        "merged_text": result.get("merged_text", ""),
        "quality_scores": result.get("quality_scores", {}),
        "iterations_used": result.get("iteration", 0),
        "subtask_results": result.get("subtask_results", []),
        "citation_uses": result.get("citation_uses", []),
        "claim_text_by_id": result.get("claim_text_by_id", {}),
    }
    return {
        "chapters": {chapter_id: chapter_result},
        "claim_text_by_id": result.get("claim_text_by_id", {}),
    }
```

**Steps:**
1. Create `muse/prompts/chapter_agent.py`
2. Replace `muse/graph/subgraphs/chapter.py` with dual-mode version (ReAct + fallback)
3. Create `tests/test_chapter_react_agent.py`
4. Run: `cd /home/planck/gradute/Muse && python -m pytest tests/test_chapter_react_agent.py tests/test_chapter_subgraph.py -v`
5. Verify existing `test_chapter_subgraph.py` still passes (backward compat via fallback)
6. Verify new tests pass

---

## Task 7 — Update `fan_out_chapters` for New Chapter Agent

**Files:**
- Modify: `muse/graph/nodes/draft.py`
- Modify: `muse/graph/main_graph.py`

**TDD — add tests to `tests/test_chapter_react_agent.py`:**

```python
class FanOutChaptersTests(unittest.TestCase):
    def test_fan_out_returns_send_objects(self):
        from muse.graph.nodes.draft import fan_out_chapters
        state = {
            "chapter_plans": [
                {"chapter_id": "ch_01", "chapter_title": "Intro", "subtask_plan": []},
                {"chapter_id": "ch_02", "chapter_title": "Methods", "subtask_plan": []},
            ],
            "references": [],
            "topic": "Test topic",
            "language": "zh",
        }
        sends = fan_out_chapters(state)
        self.assertEqual(len(sends), 2)
        for send in sends:
            self.assertEqual(send.node, "chapter_subgraph")
            self.assertIn("chapter_plan", send.arg)

    def test_fan_out_preserves_all_required_keys(self):
        from muse.graph.nodes.draft import fan_out_chapters
        state = {
            "chapter_plans": [{"chapter_id": "ch_01", "chapter_title": "Intro", "subtask_plan": []}],
            "references": [{"ref_id": "@a", "title": "A"}],
            "topic": "Topic",
            "language": "en",
        }
        sends = fan_out_chapters(state)
        arg = sends[0].arg
        self.assertIn("references", arg)
        self.assertIn("topic", arg)
        self.assertIn("language", arg)
```

**Modify — `muse/graph/nodes/draft.py`** (no changes needed to `fan_out_chapters` itself since the chapter subgraph node interface is preserved; only ensure the Send payload is stable):

The existing `fan_out_chapters` already produces the correct `Send` payloads. The `build_chapter_subgraph_node` wrapper handles both ReAct and fallback paths, and both accept the same state dict. No changes to `draft.py` or `main_graph.py` are required.

**Steps:**
1. Append the `FanOutChaptersTests` class to `tests/test_chapter_react_agent.py`
2. Run: `cd /home/planck/gradute/Muse && python -m pytest tests/test_chapter_react_agent.py -v`
3. Verify new tests pass
4. Run full graph tests: `cd /home/planck/gradute/Muse && python -m pytest tests/test_graph.py tests/test_chapter_subgraph.py -v`
5. Verify no regressions

---

## Task 8 — Citation Tools (`muse/tools/citation.py`)

**Files:**
- Create: `muse/tools/citation.py`
- Read (context only): `muse/services/citation.py`, `muse/graph/subgraphs/citation.py`

**TDD — write tests first in `tests/test_tools_citation.py`:**

```python
"""Tests for muse/tools/citation.py"""
import unittest
import json


class CitationToolTests(unittest.TestCase):
    def test_verify_doi_returns_result(self):
        from muse.tools.citation import verify_doi
        result = verify_doi.invoke({"doi": "10.1000/test"})
        self.assertIsInstance(result, str)

    def test_crosscheck_metadata_returns_result(self):
        from muse.tools.citation import crosscheck_metadata
        result = crosscheck_metadata.invoke({
            "reference_json": '{"ref_id": "@test", "title": "Test Paper", "doi": "10.1000/test", "authors": ["A. Test"], "year": 2024}'
        })
        self.assertIsInstance(result, str)

    def test_entailment_check_returns_result(self):
        from muse.tools.citation import entailment_check
        result = entailment_check.invoke({
            "premise": "Neural networks can approximate any function.",
            "hypothesis": "Deep learning has universal approximation capability.",
        })
        self.assertIsInstance(result, str)
        self.assertIn(json.loads(result)["entailment"], ["entailment", "neutral", "contradiction", "skipped"])

    def test_flag_citation_returns_json(self):
        from muse.tools.citation import flag_citation
        result = flag_citation.invoke({
            "cite_key": "@smith2024",
            "reason": "unsupported_claim",
            "claim_id": "ch01_sub01_c01",
            "detail": "Claim not supported by reference abstract.",
        })
        parsed = json.loads(result)
        self.assertEqual(parsed["cite_key"], "@smith2024")

    def test_repair_citation_returns_json(self):
        from muse.tools.citation import repair_citation
        result = repair_citation.invoke({
            "claim_id": "ch01_sub01_c01",
            "action": "replace_source",
            "new_cite_key": "@jones2023",
            "justification": "Jones 2023 directly addresses the claim.",
        })
        parsed = json.loads(result)
        self.assertEqual(parsed["action"], "replace_source")


if __name__ == "__main__":
    unittest.main()
```

**Implementation — `muse/tools/citation.py`:**

```python
"""Citation verification tools for the citation ReAct agent."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool


@tool
def verify_doi(doi: str) -> str:
    """Verify that a DOI resolves to a valid record via CrossRef or DOI.org.

    Args:
        doi: The DOI string to verify (e.g. "10.1000/test").

    Returns:
        JSON with keys: doi, valid (bool), detail.
    """
    from muse.tools._context import get_services

    services = get_services()
    metadata_client = getattr(services, "metadata", None)

    if metadata_client is None:
        # Fallback: basic format check
        valid = bool(doi and doi.startswith("10."))
        return json.dumps({"doi": doi, "valid": valid, "detail": "format_check_only"})

    try:
        valid = metadata_client.verify_doi(doi)
        return json.dumps({"doi": doi, "valid": valid, "detail": "crossref_verified" if valid else "doi_not_found"})
    except Exception as exc:
        return json.dumps({"doi": doi, "valid": False, "detail": f"error: {exc}"})


@tool
def crosscheck_metadata(reference_json: str) -> str:
    """Cross-check a reference's metadata (title, authors, year, DOI) for consistency.

    Args:
        reference_json: JSON string of the reference object.

    Returns:
        JSON with keys: ref_id, consistent (bool), issues (list).
    """
    from muse.tools._context import get_services

    try:
        ref = json.loads(reference_json)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"ref_id": "unknown", "consistent": False, "issues": ["invalid JSON input"]})

    services = get_services()
    metadata_client = getattr(services, "metadata", None)

    issues: list[str] = []
    if not ref.get("title"):
        issues.append("missing title")
    if not ref.get("authors"):
        issues.append("missing authors")
    if not ref.get("year"):
        issues.append("missing year")

    if metadata_client is not None:
        try:
            if not metadata_client.crosscheck_metadata(ref):
                issues.append("metadata_mismatch")
        except Exception as exc:
            issues.append(f"crosscheck error: {exc}")

    return json.dumps({
        "ref_id": ref.get("ref_id", "unknown"),
        "consistent": len(issues) == 0,
        "issues": issues,
    })


@tool
def entailment_check(premise: str, hypothesis: str) -> str:
    """Check if a reference passage (premise) entails a claim (hypothesis) using NLI.

    Args:
        premise: The source text (e.g. abstract or passage from the reference).
        hypothesis: The claim made in the thesis that cites this reference.

    Returns:
        JSON with keys: entailment ("entailment"|"neutral"|"contradiction"|"skipped"), confidence.
    """
    from muse.tools._context import get_services

    services = get_services()
    llm = getattr(services, "llm", None)

    if llm is None or not hasattr(llm, "entailment"):
        return json.dumps({"entailment": "skipped", "confidence": 0.0})

    try:
        result = llm.entailment(premise=premise, hypothesis=hypothesis, route="reasoning")
        return json.dumps({"entailment": result, "confidence": 0.8 if result == "entailment" else 0.3})
    except Exception:
        return json.dumps({"entailment": "skipped", "confidence": 0.0})


@tool
def flag_citation(
    cite_key: str,
    reason: str,
    claim_id: str = "",
    detail: str = "",
) -> str:
    """Flag a citation as problematic for later repair or removal.

    Args:
        cite_key: The citation key (e.g. "@smith2024").
        reason: Reason for flagging ("not_found"|"doi_invalid"|"metadata_mismatch"|"unsupported_claim").
        claim_id: ID of the claim associated with this citation.
        detail: Additional detail about the issue.

    Returns:
        JSON confirmation of the flagged citation.
    """
    entry = {
        "cite_key": cite_key,
        "reason": reason,
        "claim_id": claim_id or None,
        "detail": detail or None,
        "status": "flagged",
    }
    return json.dumps(entry)


@tool
def repair_citation(
    claim_id: str,
    action: str,
    new_cite_key: str = "",
    justification: str = "",
) -> str:
    """Propose a repair action for a flagged citation.

    Args:
        claim_id: ID of the claim needing citation repair.
        action: Repair action ("replace_source"|"weaken_claim"|"remove_claim"|"add_evidence").
        new_cite_key: New citation key if action is "replace_source".
        justification: Reason for the repair decision.

    Returns:
        JSON confirmation of the repair action.
    """
    entry = {
        "claim_id": claim_id,
        "action": action,
        "new_cite_key": new_cite_key or None,
        "justification": justification,
        "status": "repair_proposed",
    }
    return json.dumps(entry)
```

**Steps:**
1. Create `muse/tools/citation.py`
2. Create `tests/test_tools_citation.py`
3. Run: `cd /home/planck/gradute/Muse && python -m pytest tests/test_tools_citation.py -v`
4. Verify all 5 tests pass

---

## Task 9 — Rewrite Citation Subgraph as ReAct Agent

**Files:**
- Modify: `muse/graph/subgraphs/citation.py`
- Create: `muse/prompts/citation_agent.py`

**TDD — write tests first in `tests/test_citation_react_agent.py`:**

```python
"""Tests for the ReAct-based citation subgraph."""
import unittest


class _FakeNLILLM:
    def entailment(self, *, premise, hypothesis, route="reasoning"):
        return "entailment"

    def structured(self, *, system, user, route="default", max_tokens=2500):
        return {}


class _FakeMetadata:
    def verify_doi(self, doi):
        return True

    def crosscheck_metadata(self, ref):
        return True


class _FakeServices:
    def __init__(self):
        self.llm = _FakeNLILLM()
        self.metadata = _FakeMetadata()
        self.search = None
        self.rag_index = None


class CitationReActTests(unittest.TestCase):
    def test_build_citation_subgraph_node_returns_callable(self):
        from muse.graph.subgraphs.citation import build_citation_subgraph_node
        fn = build_citation_subgraph_node(services=_FakeServices())
        self.assertTrue(callable(fn))

    def test_citation_graph_fixed_flow_still_works(self):
        """Backward compat: the fixed-flow graph must still function."""
        from muse.graph.subgraphs.citation import build_citation_graph
        graph = build_citation_graph(services=_FakeServices())
        result = graph.invoke({
            "references": [{"ref_id": "@a", "title": "Paper A", "doi": "10.1/a", "authors": ["A"], "year": 2024}],
            "citation_uses": [{"cite_key": "@a", "claim_id": "c1"}],
            "claim_text_by_id": {"c1": "Claim text."},
        })
        self.assertIn("citation_ledger", result)

    def test_citation_agent_system_prompt_exists(self):
        from muse.prompts.citation_agent import citation_agent_system_prompt
        prompt = citation_agent_system_prompt(
            total_citations=5,
            total_claims=10,
            references_summary="5 references",
        )
        self.assertIn("citation", prompt.lower())
        self.assertIn("submit", prompt.lower())


if __name__ == "__main__":
    unittest.main()
```

**Create system prompt — `muse/prompts/citation_agent.py`:**

```python
"""System prompt for the citation verification ReAct agent."""
from __future__ import annotations


def citation_agent_system_prompt(
    *,
    total_citations: int,
    total_claims: int,
    references_summary: str,
) -> str:
    """Build the system prompt for the citation ReAct agent."""
    return f"""You are a citation verification agent for an academic thesis.
Your task is to verify that every citation properly supports its associated claim.

## Context
- Total citation uses to verify: {total_citations}
- Total unique claims: {total_claims}
- {references_summary}

## Workflow
1. For each citation use, run these checks in order:
   a. `verify_doi` — confirm the DOI resolves to a real record.
   b. `crosscheck_metadata` — verify title/authors/year consistency.
   c. `entailment_check` — verify the reference content supports the claim.
2. If any check fails, use `flag_citation` to record the issue.
3. For flagged citations with medium confidence, use `repair_citation` to propose fixes.
4. When all citations are processed, call `submit_result` with a JSON object containing:
   - "citation_ledger": dict mapping claim_id to verification result
   - "verified_citations": list of cite_keys that passed all checks
   - "flagged_citations": list of flagged citation objects

## Rules
- Process critical claims (high-impact, central arguments) with full 3-layer verification.
- For minor citations (acknowledgments, background), DOI-only check is acceptable.
- You MUST call `submit_result` to finish. Max 20 turns.
"""
```

**Modify — `muse/graph/subgraphs/citation.py`** — add ReAct path with fallback (append after existing code, modify `build_citation_subgraph_node`):

The same dual-mode pattern as chapter: try `create_react_agent` with citation tools, fall back to the existing `build_citation_graph` StateGraph.

```python
# Add to end of citation.py, replacing build_citation_subgraph_node:

def _build_react_citation_agent(*, services: Any, settings: Any = None):
    """Build a ReAct agent for citation verification."""
    try:
        from langgraph.prebuilt import create_react_agent
        from muse.models.factory import create_chat_model
    except ImportError:
        return None

    if settings is None:
        return None

    from muse.tools.citation import verify_doi, crosscheck_metadata, entailment_check, flag_citation, repair_citation
    from muse.tools.research import academic_search
    from muse.tools.file import read_file
    from muse.tools.orchestration import submit_result, update_plan

    tools = [
        verify_doi, crosscheck_metadata, entailment_check,
        flag_citation, repair_citation,
        academic_search, read_file,
        submit_result, update_plan,
    ]

    try:
        model = create_chat_model(settings, route="reasoning")
    except Exception:
        return None

    return create_react_agent(model=model, tools=tools, state_schema=CitationState)


def build_citation_subgraph_node(*, services: Any, settings: Any = None):
    react_agent = _build_react_citation_agent(services=services, settings=settings)

    if react_agent is not None:
        def run_react_citation(state: dict[str, Any]) -> dict[str, Any]:
            from muse.tools._context import set_services
            from muse.tools.orchestration import get_submitted_result, clear_submitted_result
            from muse.prompts.citation_agent import citation_agent_system_prompt

            set_services(services)
            clear_submitted_result()

            refs = state.get("references", [])
            citation_uses = state.get("citation_uses", [])
            claim_text_by_id = state.get("claim_text_by_id", {})

            prompt = citation_agent_system_prompt(
                total_citations=len(citation_uses),
                total_claims=len(claim_text_by_id),
                references_summary=f"{len(refs)} references available",
            )

            agent_input = {
                "messages": [{"role": "system", "content": prompt}],
                "references": refs,
                "citation_uses": citation_uses,
                "claim_text_by_id": claim_text_by_id,
            }

            try:
                react_agent.invoke(agent_input, {"recursion_limit": 40})
            except Exception:
                pass

            submitted = get_submitted_result()
            clear_submitted_result()
            if submitted:
                p = submitted["payload"]
                return {
                    "citation_ledger": p.get("citation_ledger", {}),
                    "verified_citations": p.get("verified_citations", []),
                    "flagged_citations": p.get("flagged_citations", []),
                }

            # Fallback to fixed-flow
            graph = build_citation_graph(services=services)
            result = graph.invoke({
                "references": refs,
                "citation_uses": citation_uses,
                "claim_text_by_id": claim_text_by_id,
                "citation_ledger": state.get("citation_ledger", {}),
                "verified_citations": state.get("verified_citations", []),
                "flagged_citations": state.get("flagged_citations", []),
            })
            return {
                "citation_ledger": result.get("citation_ledger", {}),
                "verified_citations": result.get("verified_citations", []),
                "flagged_citations": result.get("flagged_citations", []),
            }

        return run_react_citation

    # Pure fallback
    graph = build_citation_graph(services=services)

    def run_citation_subgraph(state: dict[str, Any]) -> dict[str, Any]:
        result = graph.invoke({
            "references": state.get("references", []),
            "citation_uses": state.get("citation_uses", []),
            "claim_text_by_id": state.get("claim_text_by_id", {}),
            "citation_ledger": state.get("citation_ledger", {}),
            "verified_citations": state.get("verified_citations", []),
            "flagged_citations": state.get("flagged_citations", []),
        })
        return {
            "citation_ledger": result.get("citation_ledger", {}),
            "verified_citations": result.get("verified_citations", []),
            "flagged_citations": result.get("flagged_citations", []),
        }

    return run_citation_subgraph
```

**Steps:**
1. Create `muse/prompts/citation_agent.py`
2. Modify `muse/graph/subgraphs/citation.py` — replace `build_citation_subgraph_node`, add `_build_react_citation_agent`
3. Create `tests/test_citation_react_agent.py`
4. Run: `cd /home/planck/gradute/Muse && python -m pytest tests/test_citation_react_agent.py tests/test_citation_ledger.py -v`
5. Verify all tests pass (new + existing)

---

## Task 10 — Composition Tools (`muse/tools/composition.py`)

**Files:**
- Create: `muse/tools/composition.py`

**TDD — write tests first in `tests/test_tools_composition.py`:**

```python
"""Tests for muse/tools/composition.py"""
import unittest
import json


class CompositionToolTests(unittest.TestCase):
    def test_check_terminology_returns_json(self):
        from muse.tools.composition import check_terminology
        result_str = check_terminology.invoke({
            "text": "We use deep learning and DL interchangeably. The neural net processes data.",
        })
        result = json.loads(result_str)
        self.assertIn("issues", result)
        self.assertIsInstance(result["issues"], list)

    def test_align_cross_refs_returns_json(self):
        from muse.tools.composition import align_cross_refs
        result_str = align_cross_refs.invoke({
            "text": "As shown in Figure 1 and discussed in Section 2.3, the results in Table 5 confirm our hypothesis.",
        })
        result = json.loads(result_str)
        self.assertIn("cross_refs_found", result)

    def test_check_transitions_returns_json(self):
        from muse.tools.composition import check_transitions
        result_str = check_transitions.invoke({
            "chapter_texts_json": '[{"chapter_id": "ch1", "ending": "In summary, the method works."}, {"chapter_id": "ch2", "opening": "This chapter explores results."}]',
        })
        result = json.loads(result_str)
        self.assertIn("transitions", result)

    def test_rewrite_passage_returns_text(self):
        from muse.tools.composition import rewrite_passage
        result = rewrite_passage.invoke({
            "passage": "The thing works good because of reasons.",
            "instruction": "Improve academic tone and specificity.",
            "context": "Methods section of a CS thesis.",
        })
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
```

**Implementation — `muse/tools/composition.py`:**

```python
"""Composition tools for terminology, cross-references, transitions, and rewriting."""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.tools import tool


@tool
def check_terminology(text: str) -> str:
    """Scan text for terminology inconsistencies (synonyms, abbreviation mismatches).

    Args:
        text: The full text to scan for terminology issues.

    Returns:
        JSON with keys: issues (list of {term, variants, suggestion}).
    """
    from muse.tools._context import get_services

    services = get_services()
    llm = getattr(services, "llm", None)

    if llm is None:
        # Fallback: basic abbreviation check
        issues: list[dict[str, str]] = []
        # Detect common pattern: full term + abbreviation used inconsistently
        abbr_pattern = re.findall(r"\b([A-Z]{2,})\b", text)
        seen = set()
        for abbr in abbr_pattern:
            if abbr not in seen:
                seen.add(abbr)
        return json.dumps({"issues": issues, "abbreviations_found": sorted(seen)})

    system = (
        "Scan the following text for terminology inconsistencies. "
        "Find cases where the same concept uses different terms or abbreviations. "
        "Return JSON with key 'issues': list of {term, variants, suggestion}."
    )
    try:
        result = llm.structured(system=system, user=text[:6000], route="review", max_tokens=1200)
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False)
    except Exception:
        pass

    return json.dumps({"issues": []})


@tool
def align_cross_refs(text: str) -> str:
    """Check that all cross-references (Figure X, Table Y, Section Z) in the text are valid.

    Args:
        text: The full thesis text to check for cross-reference consistency.

    Returns:
        JSON with keys: cross_refs_found (list), dangling (list of unresolved refs).
    """
    # Extract cross-references
    patterns = [
        r"(?:Figure|Fig\.?|图)\s*(\d+[\.\d]*)",
        r"(?:Table|Tab\.?|表)\s*(\d+[\.\d]*)",
        r"(?:Section|Sec\.?|节|章)\s*(\d+[\.\d]*)",
        r"(?:Equation|Eq\.?|式|公式)\s*[\(（]?(\d+[\.\d]*)[\)）]?",
    ]
    cross_refs: list[dict[str, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            cross_refs.append({"type": pattern.split("(")[0].strip("(?:").split("|")[0], "number": match.group(1), "position": match.start()})

    return json.dumps({
        "cross_refs_found": cross_refs[:100],
        "dangling": [],  # Full validation requires index of defined figures/tables
        "total_count": len(cross_refs),
    }, ensure_ascii=False)


@tool
def check_transitions(chapter_texts_json: str) -> str:
    """Check the quality of transitions between chapters.

    Args:
        chapter_texts_json: JSON array of objects with chapter_id, ending (last paragraph), opening (first paragraph of next chapter).

    Returns:
        JSON with keys: transitions (list of {from_chapter, to_chapter, quality, suggestion}).
    """
    from muse.tools._context import get_services

    try:
        chapters = json.loads(chapter_texts_json)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"transitions": [], "error": "invalid JSON"})

    services = get_services()
    llm = getattr(services, "llm", None)

    transitions: list[dict[str, Any]] = []
    for i in range(len(chapters) - 1):
        current = chapters[i]
        next_ch = chapters[i + 1]
        transition = {
            "from_chapter": current.get("chapter_id", f"ch{i+1}"),
            "to_chapter": next_ch.get("chapter_id", f"ch{i+2}"),
            "quality": "unknown",
            "suggestion": "",
        }

        if llm is not None:
            system = (
                "Evaluate the transition between these two chapter segments. "
                "Return JSON: {quality: 'smooth'|'abrupt'|'missing', suggestion: str}"
            )
            user = json.dumps({
                "ending": current.get("ending", "")[:500],
                "opening": next_ch.get("opening", "")[:500],
            }, ensure_ascii=False)
            try:
                result = llm.structured(system=system, user=user, route="review", max_tokens=500)
                if isinstance(result, dict):
                    transition["quality"] = result.get("quality", "unknown")
                    transition["suggestion"] = result.get("suggestion", "")
            except Exception:
                pass

        transitions.append(transition)

    return json.dumps({"transitions": transitions}, ensure_ascii=False)


@tool
def rewrite_passage(
    passage: str,
    instruction: str,
    context: str = "",
) -> str:
    """Rewrite a passage according to specific instructions (tone, clarity, conciseness).

    Args:
        passage: The text passage to rewrite.
        instruction: Specific rewriting instruction.
        context: Additional context about where this passage appears.

    Returns:
        The rewritten passage text.
    """
    from muse.tools._context import get_services

    services = get_services()
    llm = getattr(services, "llm", None)

    if llm is None:
        return passage

    system = f"Rewrite the following passage. Context: {context}. Return ONLY the rewritten text."
    user = json.dumps({"instruction": instruction, "passage": passage}, ensure_ascii=False)

    try:
        return llm.text(system=system, user=user, route="polish", max_tokens=2000)
    except Exception:
        return passage
```

**Steps:**
1. Create `muse/tools/composition.py`
2. Create `tests/test_tools_composition.py`
3. Run: `cd /home/planck/gradute/Muse && python -m pytest tests/test_tools_composition.py -v`
4. Verify all 4 tests pass

---

## Task 11 — Rewrite Composition Subgraph as ReAct Agent

**Files:**
- Modify: `muse/graph/subgraphs/composition.py`
- Create: `muse/prompts/composition_agent.py`

**TDD — write tests first in `tests/test_composition_react_agent.py`:**

```python
"""Tests for the ReAct-based composition subgraph."""
import unittest


class CompositionReActTests(unittest.TestCase):
    def test_build_composition_subgraph_node_returns_callable(self):
        from muse.graph.subgraphs.composition import build_composition_subgraph_node
        fn = build_composition_subgraph_node()
        self.assertTrue(callable(fn))

    def test_composition_graph_fixed_flow_still_works(self):
        from muse.graph.subgraphs.composition import build_composition_graph
        graph = build_composition_graph()
        result = graph.invoke({
            "final_text": "Chapter 1 text. Chapter 2 text.",
            "abstract_zh": "摘要",
            "abstract_en": "Abstract",
            "paper_package": {},
        })
        self.assertTrue(result.get("paper_package", {}).get("terminology_normalized"))
        self.assertTrue(result.get("paper_package", {}).get("cross_refs_aligned"))

    def test_composition_agent_system_prompt_exists(self):
        from muse.prompts.composition_agent import composition_agent_system_prompt
        prompt = composition_agent_system_prompt(
            chapter_count=5,
            total_words=25000,
            language="zh",
        )
        self.assertIn("composition", prompt.lower())
        self.assertIn("submit", prompt.lower())


if __name__ == "__main__":
    unittest.main()
```

**Create system prompt — `muse/prompts/composition_agent.py`:**

```python
"""System prompt for the composition/coherence ReAct agent."""
from __future__ import annotations


def composition_agent_system_prompt(
    *,
    chapter_count: int,
    total_words: int,
    language: str,
) -> str:
    """Build the system prompt for the composition ReAct agent."""
    return f"""You are a thesis composition and coherence agent. Your task is to ensure
the final thesis reads as a unified, polished document.

## Context
- Chapters: {chapter_count}
- Approximate total words: {total_words}
- Language: {language}

## Workflow
1. **Terminology**: Use `check_terminology` to scan for inconsistent terms and abbreviations.
   Fix issues with `rewrite_passage` or `apply_patch`.
2. **Cross-references**: Use `align_cross_refs` to verify all Figure/Table/Section references
   are valid. Fix dangling references with `edit_file` or `apply_patch`.
3. **Transitions**: Use `check_transitions` to evaluate chapter-to-chapter flow.
   Improve abrupt transitions with `rewrite_passage`.
4. **Submit**: When all checks pass, call `submit_result` with a JSON object containing:
   - "final_text": the polished full text
   - "paper_package": updated package dict with terminology_normalized, cross_refs_aligned, composed_text
   - "changes_made": list of changes applied

## Rules
- Do NOT change the substantive content or arguments. Only improve consistency and flow.
- Preserve all citations and cross-references.
- You MUST call `submit_result` to finish. Max 15 turns.
"""
```

**Modify — `muse/graph/subgraphs/composition.py`** — add ReAct path with fallback:

```python
# Replace build_composition_subgraph_node with dual-mode version:

def _build_react_composition_agent(*, settings: Any = None):
    """Build a ReAct agent for composition/coherence."""
    try:
        from langgraph.prebuilt import create_react_agent
        from muse.models.factory import create_chat_model
    except ImportError:
        return None

    if settings is None:
        return None

    from muse.tools.composition import check_terminology, align_cross_refs, check_transitions, rewrite_passage
    from muse.tools.writing import apply_patch, revise_section
    from muse.tools.file import read_file, write_file, edit_file, glob_files, grep
    from muse.tools.orchestration import submit_result, update_plan

    tools = [
        check_terminology, align_cross_refs, check_transitions, rewrite_passage,
        apply_patch, revise_section,
        read_file, write_file, edit_file, glob_files, grep,
        submit_result, update_plan,
    ]

    try:
        model = create_chat_model(settings, route="polish")
    except Exception:
        return None

    return create_react_agent(model=model, tools=tools, state_schema=CompositionState)


def build_composition_subgraph_node(*, settings: Any = None, services: Any = None):
    react_agent = _build_react_composition_agent(settings=settings)

    if react_agent is not None:
        def run_react_composition(state: dict[str, Any]) -> dict[str, Any]:
            from muse.tools._context import set_services
            from muse.tools.orchestration import get_submitted_result, clear_submitted_result
            from muse.prompts.composition_agent import composition_agent_system_prompt

            if services is not None:
                set_services(services)
            clear_submitted_result()

            final_text = state.get("final_text", "")
            prompt = composition_agent_system_prompt(
                chapter_count=len(state.get("paper_package", {}).get("chapters", {})),
                total_words=len(final_text.split()),
                language=state.get("language", "zh"),
            )

            agent_input = {
                "messages": [{"role": "system", "content": prompt}],
                "final_text": final_text,
                "abstract_zh": state.get("abstract_zh", ""),
                "abstract_en": state.get("abstract_en", ""),
                "paper_package": state.get("paper_package", {}),
            }

            try:
                react_agent.invoke(agent_input, {"recursion_limit": 30})
            except Exception:
                pass

            submitted = get_submitted_result()
            clear_submitted_result()
            if submitted:
                p = submitted["payload"]
                return {
                    "final_text": p.get("final_text", final_text),
                    "paper_package": p.get("paper_package", state.get("paper_package", {})),
                }

            # Fallback to fixed-flow
            graph = build_composition_graph()
            result = graph.invoke({
                "final_text": final_text,
                "abstract_zh": state.get("abstract_zh", ""),
                "abstract_en": state.get("abstract_en", ""),
                "paper_package": state.get("paper_package", {}),
            })
            return {
                "final_text": result.get("final_text", final_text),
                "paper_package": result.get("paper_package", state.get("paper_package", {})),
            }

        return run_react_composition

    # Pure fallback
    graph = build_composition_graph()

    def run_composition_subgraph(state: dict[str, Any]) -> dict[str, Any]:
        result = graph.invoke({
            "final_text": state.get("final_text", ""),
            "abstract_zh": state.get("abstract_zh", ""),
            "abstract_en": state.get("abstract_en", ""),
            "paper_package": state.get("paper_package", {}),
        })
        return {
            "final_text": result.get("final_text", state.get("final_text", "")),
            "paper_package": result.get("paper_package", state.get("paper_package", {})),
        }

    return run_composition_subgraph
```

**IMPORTANT:** Update `main_graph.py` to pass `settings` and `services` to `build_composition_subgraph_node`:

```python
# In main_graph.py, change:
#   builder.add_node("composition_subgraph", build_composition_subgraph_node())
# To:
#   builder.add_node("composition_subgraph", build_composition_subgraph_node(settings=settings, services=services))
```

**Steps:**
1. Create `muse/prompts/composition_agent.py`
2. Modify `muse/graph/subgraphs/composition.py` — add `_build_react_composition_agent`, replace `build_composition_subgraph_node`
3. Update `muse/graph/main_graph.py` — pass `settings` and `services` to composition node builder
4. Create `tests/test_composition_react_agent.py`
5. Run: `cd /home/planck/gradute/Muse && python -m pytest tests/test_composition_react_agent.py -v`
6. Verify all tests pass
7. Run: `cd /home/planck/gradute/Muse && python -m pytest tests/test_graph.py -v` (regression check)

---

## Task 12 — Integration Test: Full Pipeline with ReAct Sub-graphs

**Files:**
- Create: `tests/test_react_integration.py`

**Implementation — `tests/test_react_integration.py`:**

```python
"""Integration test: full pipeline with ReAct sub-graphs (fallback mode).

This test verifies that the main graph runs end-to-end with the new dual-mode
subgraph nodes. Since Phase 0-A (MuseChatModel) is not yet implemented, all
sub-graphs will use their fixed-flow fallback paths. The test confirms:
1. The main graph compiles without errors.
2. Fan-out chapters work with the new build_chapter_subgraph_node.
3. Citation subgraph runs with the new build_citation_subgraph_node.
4. Composition subgraph runs with the new build_composition_subgraph_node.
5. The full pipeline produces an output state with expected keys.
"""
import unittest


class _IntegrationLLM:
    """Minimal LLM stub for full-pipeline integration."""
    def __init__(self):
        self._review_calls = 0

    def structured(self, *, system, user, route="default", max_tokens=2500):
        if "search queries" in system.lower() or "topic analysis" in system.lower():
            return {"queries": ["test query"], "analysis": "test analysis"}
        if "outline" in system.lower():
            return {
                "chapter_plans": [
                    {
                        "chapter_id": "ch_01",
                        "chapter_title": "Introduction",
                        "subtask_plan": [{"subtask_id": "sub_01", "title": "Background", "target_words": 500}],
                    }
                ]
            }
        if "Write one thesis subsection" in system:
            return {
                "text": "Test subsection content.",
                "citations_used": [],
                "key_claims": ["Test claim."],
                "transition_out": "",
                "glossary_additions": {},
                "self_assessment": {"confidence": 0.9, "weak_spots": [], "needs_revision": False},
            }
        if "strict thesis reviewer" in system:
            self._review_calls += 1
            return {
                "scores": {"coherence": 5, "logic": 5, "citation": 5, "term_consistency": 5, "balance": 5, "redundancy": 5},
                "review_notes": [],
            }
        if "polish" in system.lower():
            return {"text": "Polished text.", "notes": []}
        if "abstract" in system.lower():
            return {"abstract": "Test abstract."}
        return {}

    def text(self, *, system, user, route="default", max_tokens=2500):
        return "Generated text."

    def entailment(self, *, premise, hypothesis, route="reasoning"):
        return "entailment"


class _IntegrationMetadata:
    def verify_doi(self, doi):
        return True

    def crosscheck_metadata(self, ref):
        return True


class _IntegrationServices:
    def __init__(self):
        self.llm = _IntegrationLLM()
        self.metadata = _IntegrationMetadata()
        self.search = None
        self.rag_index = None
        self.local_refs = []


class FullPipelineIntegrationTest(unittest.TestCase):
    def test_main_graph_compiles(self):
        """Verify that build_graph succeeds with new subgraph nodes."""
        from muse.graph.main_graph import build_graph
        graph = build_graph(services=_IntegrationServices(), auto_approve=True)
        self.assertIsNotNone(graph)

    def test_chapter_subgraph_fallback_path(self):
        """Chapter subgraph falls back to fixed-flow when Phase 0-A is absent."""
        from muse.graph.subgraphs.chapter import build_chapter_subgraph_node
        node_fn = build_chapter_subgraph_node(services=_IntegrationServices())
        result = node_fn({
            "chapter_plan": {
                "chapter_id": "ch_01",
                "chapter_title": "Introduction",
                "subtask_plan": [{"subtask_id": "sub_01", "title": "Background", "target_words": 500}],
            },
            "references": [],
            "topic": "Test",
            "language": "zh",
            "subtask_results": [],
            "merged_text": "",
            "quality_scores": {},
            "review_notes": [],
            "revision_instructions": {},
            "iteration": 0,
            "max_iterations": 1,
            "citation_uses": [],
            "claim_text_by_id": {},
        })
        self.assertIn("chapters", result)
        self.assertIn("ch_01", result["chapters"])
        self.assertIn("merged_text", result["chapters"]["ch_01"])

    def test_citation_subgraph_fallback_path(self):
        """Citation subgraph falls back to fixed-flow when Phase 0-A is absent."""
        from muse.graph.subgraphs.citation import build_citation_subgraph_node
        node_fn = build_citation_subgraph_node(services=_IntegrationServices())
        result = node_fn({
            "references": [{"ref_id": "@a", "title": "A", "doi": "10.1/a", "authors": ["X"], "year": 2024}],
            "citation_uses": [{"cite_key": "@a", "claim_id": "c1"}],
            "claim_text_by_id": {"c1": "Test claim."},
            "citation_ledger": {},
            "verified_citations": [],
            "flagged_citations": [],
        })
        self.assertIn("citation_ledger", result)

    def test_composition_subgraph_fallback_path(self):
        """Composition subgraph falls back to fixed-flow when Phase 0-A is absent."""
        from muse.graph.subgraphs.composition import build_composition_subgraph_node
        node_fn = build_composition_subgraph_node()
        result = node_fn({
            "final_text": "Test text.",
            "abstract_zh": "摘要",
            "abstract_en": "Abstract",
            "paper_package": {},
        })
        self.assertIn("final_text", result)
        self.assertIn("paper_package", result)
        self.assertTrue(result["paper_package"].get("terminology_normalized"))


if __name__ == "__main__":
    unittest.main()
```

**Steps:**
1. Create `tests/test_react_integration.py`
2. Run: `cd /home/planck/gradute/Muse && python -m pytest tests/test_react_integration.py -v`
3. Verify all 4 tests pass
4. Run full test suite: `cd /home/planck/gradute/Muse && python -m pytest tests/ -v --tb=short`
5. Verify zero regressions across all existing tests

---

## File Inventory

### New Files (13)

| File | Task | Purpose |
|------|------|---------|
| `muse/tools/_context.py` | 1 | Thread-local service context for tools |
| `muse/tools/writing.py` | 1 | write_section, revise_section, apply_patch |
| `muse/tools/review.py` | 2 | self_review (multi-lens) |
| `muse/tools/research.py` | 3 | web_search, web_fetch, academic_search, retrieve_local_refs, read_pdf, image_search |
| `muse/tools/file.py` | 4 | read_file, write_file, edit_file, glob_files, grep |
| `muse/tools/orchestration.py` | 5 | submit_result, update_plan |
| `muse/tools/citation.py` | 8 | verify_doi, crosscheck_metadata, entailment_check, flag_citation, repair_citation |
| `muse/tools/composition.py` | 10 | check_terminology, align_cross_refs, check_transitions, rewrite_passage |
| `muse/prompts/chapter_agent.py` | 6 | Chapter ReAct agent system prompt |
| `muse/prompts/citation_agent.py` | 9 | Citation ReAct agent system prompt |
| `muse/prompts/composition_agent.py` | 11 | Composition ReAct agent system prompt |
| `tests/test_tools_writing.py` | 1 | Tests for writing tools |
| `tests/test_tools_review.py` | 2 | Tests for review tools |
| `tests/test_tools_research.py` | 3 | Tests for research tools |
| `tests/test_tools_file.py` | 4 | Tests for file tools |
| `tests/test_tools_orchestration.py` | 5 | Tests for orchestration tools |
| `tests/test_tools_citation.py` | 8 | Tests for citation tools |
| `tests/test_tools_composition.py` | 10 | Tests for composition tools |
| `tests/test_chapter_react_agent.py` | 6, 7 | Tests for chapter ReAct agent + fan-out |
| `tests/test_citation_react_agent.py` | 9 | Tests for citation ReAct agent |
| `tests/test_composition_react_agent.py` | 11 | Tests for composition ReAct agent |
| `tests/test_react_integration.py` | 12 | Full pipeline integration tests |

### Modified Files (4)

| File | Task | Change |
|------|------|--------|
| `muse/tools/__init__.py` | 1 | Update docstring |
| `muse/graph/subgraphs/chapter.py` | 6 | Add ReAct path + `_extract_chapter_result` helper |
| `muse/graph/subgraphs/citation.py` | 9 | Add ReAct path + fallback |
| `muse/graph/subgraphs/composition.py` | 11 | Add ReAct path + fallback, change function signature |
| `muse/graph/main_graph.py` | 11 | Pass `settings`/`services` to composition builder |

### Unmodified Files

| File | Reason |
|------|--------|
| `muse/graph/nodes/draft.py` | `fan_out_chapters` already produces correct Send payloads |
| `muse/graph/state.py` | MuseState schema unchanged |
| `muse/graph/nodes/review.py` | Review node preserved for fixed-flow fallback |
| `muse/graph/helpers/*` | Helpers preserved for fixed-flow fallback |
| `muse/prompts/chapter_review.py` | Used by both self_review tool and fallback path |

---

## Execution Order

```
Task 1  (writing tools)        ~5 min   — no dependencies
Task 2  (review tools)         ~3 min   — no dependencies
Task 3  (research tools)       ~5 min   — no dependencies
Task 4  (file tools)           ~5 min   — no dependencies
Task 5  (orchestration tools)  ~3 min   — no dependencies
  ↓ (all tools exist)
Task 6  (chapter ReAct agent)  ~5 min   — depends on Tasks 1-5
Task 7  (fan-out update)       ~2 min   — depends on Task 6
Task 8  (citation tools)       ~4 min   — no dependencies (can parallel with 6-7)
Task 9  (citation ReAct agent) ~4 min   — depends on Tasks 5, 8
Task 10 (composition tools)    ~4 min   — no dependencies (can parallel with 9)
Task 11 (composition ReAct)    ~4 min   — depends on Tasks 5, 10
Task 12 (integration test)     ~3 min   — depends on all above
```

Tasks 1-5 and 8 and 10 can run in parallel. Total sequential critical path: ~22 min.
