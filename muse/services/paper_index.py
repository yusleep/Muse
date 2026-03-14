"""Paper full-text indexing with lazy LlamaIndex/LlamaParse imports."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "-", str(value or "").strip()).strip("-").lower()
    return text or "paper"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", str(text or "").lower())


def _section_candidates(chunks: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for chunk in chunks:
        title = str(chunk.get("section_title", "")).strip()
        if title and title not in seen:
            seen.append(title)
    return seen


class PaperIndexService:
    """Manage full-text paper ingestion, persistence, and retrieval."""

    def __init__(
        self,
        llamaparse_api_key: str,
        cache_dir: Path,
        index_dir: Path,
        *,
        local_priority: bool = True,
    ) -> None:
        self.llamaparse_api_key = str(llamaparse_api_key or "").strip()
        self.cache_dir = Path(cache_dir)
        self.index_dir = Path(index_dir)
        self.local_priority = bool(local_priority)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._papers: dict[str, dict[str, Any]] = {}
        self._chunks: list[dict[str, Any]] = []
        self._index_file = self.index_dir / "paper_index.json"
        self._import_error: Exception | None = None
        self._llamaparse_cls: Any = None
        self._hierarchical_node_parser_cls: Any = None
        self._import_llama_dependencies()
        self._load_persisted()

    def _import_llama_dependencies(self) -> None:
        try:
            from llama_index.core.node_parser import HierarchicalNodeParser
            from llama_parse import LlamaParse
        except Exception as exc:  # noqa: BLE001
            self._import_error = exc
            self._llamaparse_cls = None
            self._hierarchical_node_parser_cls = None
            return

        self._llamaparse_cls = LlamaParse
        self._hierarchical_node_parser_cls = HierarchicalNodeParser
        self._import_error = None

    def _load_persisted(self) -> None:
        if not self._index_file.is_file():
            return
        try:
            payload = json.loads(self._index_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        papers = payload.get("papers", {})
        chunks = payload.get("chunks", [])
        if isinstance(papers, dict):
            self._papers = {
                str(ref_id): dict(meta)
                for ref_id, meta in papers.items()
                if isinstance(meta, dict)
            }
        if isinstance(chunks, list):
            self._chunks = [dict(chunk) for chunk in chunks if isinstance(chunk, dict)]

    def _persist(self) -> None:
        payload = {
            "papers": self._papers,
            "chunks": self._chunks,
        }
        self._index_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _resolve_ref_id(self, reference: dict[str, Any], fallback_stem: str) -> str:
        ref_id = str(reference.get("ref_id", "")).strip()
        if ref_id:
            return ref_id
        return f"@{_slug(fallback_stem)}"

    def _resolve_pdf_url(self, reference: dict[str, Any]) -> str:
        direct = str(reference.get("pdf_url", "")).strip()
        if direct:
            return direct

        open_access = reference.get("openAccessPdf")
        if isinstance(open_access, dict):
            open_access_url = str(open_access.get("url", "")).strip()
            if open_access_url:
                return open_access_url

        arxiv_id = str(reference.get("arxiv_id", "")).strip()
        if arxiv_id:
            return f"https://export.arxiv.org/pdf/{arxiv_id}"

        return ""

    def _download_pdf(self, url: str, dest_path: Path) -> bool:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Muse/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                dest_path.write_bytes(response.read())
        except Exception:  # noqa: BLE001
            return False
        return True

    def _markdown_to_chunks(
        self,
        markdown_text: str,
        *,
        paper_id: str,
        ref_id: str,
        paper_title: str,
        source: str,
        source_priority: int,
    ) -> list[dict[str, Any]]:
        if not markdown_text.strip():
            return []

        sections: list[tuple[str, str]] = []
        current_title = "Document"
        current_lines: list[str] = []
        for raw_line in markdown_text.splitlines():
            line = raw_line.strip()
            if line.startswith("#"):
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines).strip()))
                    current_lines = []
                current_title = line.lstrip("#").strip() or "Document"
                continue
            current_lines.append(raw_line)
        if current_lines:
            sections.append((current_title, "\n".join(current_lines).strip()))

        if not sections:
            sections = [("Document", markdown_text.strip())]

        chunks: list[dict[str, Any]] = []
        for section_title, body in sections:
            body = body.strip()
            if not body:
                continue
            for start in range(0, len(body), 1200):
                chunk_text = body[start : start + 1200].strip()
                if not chunk_text:
                    continue
                chunks.append(
                    {
                        "paper_id": paper_id,
                        "ref_id": ref_id,
                        "paper_title": paper_title,
                        "section_title": section_title,
                        "page_label": "",
                        "text": chunk_text,
                        "source": source,
                        "source_priority": source_priority,
                    }
                )
        return chunks

    def _parse_pdf_to_chunks(
        self,
        pdf_path: Path,
        *,
        paper_id: str,
        ref_id: str,
        paper_title: str,
        source: str,
        source_priority: int,
    ) -> list[dict[str, Any]]:
        if not self.llamaparse_api_key or self._llamaparse_cls is None:
            return []

        try:
            parser = self._llamaparse_cls(
                api_key=self.llamaparse_api_key,
                result_type="markdown",
            )
            documents = parser.load_data(str(pdf_path))
        except Exception:  # noqa: BLE001
            return []

        markdown_parts: list[str] = []
        for document in documents or []:
            text = getattr(document, "text", "")
            if not isinstance(text, str):
                text = str(text or "")
            if text.strip():
                markdown_parts.append(text)

        if not markdown_parts:
            return []

        return self._markdown_to_chunks(
            "\n\n".join(markdown_parts),
            paper_id=paper_id,
            ref_id=ref_id,
            paper_title=paper_title,
            source=source,
            source_priority=source_priority,
        )

    def _register_ingested_paper(
        self,
        paper_meta: dict[str, Any],
        chunks: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        ref_id = str(paper_meta.get("ref_id", "")).strip()
        if not ref_id or not chunks:
            return {}

        self._chunks = [
            chunk
            for chunk in self._chunks
            if str(chunk.get("ref_id", "")).strip() != ref_id
        ]
        cleaned_chunks = [dict(chunk) for chunk in chunks if isinstance(chunk, dict)]
        self._chunks.extend(cleaned_chunks)

        metadata = dict(paper_meta)
        metadata["indexed"] = True
        metadata["available_sections"] = _section_candidates(cleaned_chunks)
        self._papers[ref_id] = metadata
        self._persist()
        return {ref_id: metadata}

    def _ingest_pdf(
        self,
        pdf_path: Path,
        *,
        reference: dict[str, Any],
        source: str,
        source_priority: int,
    ) -> dict[str, dict[str, Any]]:
        ref_id = self._resolve_ref_id(reference, pdf_path.stem)
        paper_id = _slug(str(reference.get("paper_id", "")).strip() or ref_id.lstrip("@"))
        paper_title = str(reference.get("title", "")).strip() or pdf_path.stem
        chunks = self._parse_pdf_to_chunks(
            pdf_path,
            paper_id=paper_id,
            ref_id=ref_id,
            paper_title=paper_title,
            source=source,
            source_priority=source_priority,
        )
        if not chunks:
            return {}
        return self._register_ingested_paper(
            {
                "ref_id": ref_id,
                "paper_id": paper_id,
                "paper_title": paper_title,
                "source": source,
                "source_priority": source_priority,
            },
            chunks,
        )

    def ingest_local(self, dir_path: Path) -> dict[str, dict[str, Any]]:
        directory = Path(dir_path)
        if not directory.is_dir():
            return {}

        indexed: dict[str, dict[str, Any]] = {}
        for pdf_path in sorted(directory.glob("*.pdf")):
            indexed.update(
                self._ingest_pdf(
                    pdf_path,
                    reference={"title": pdf_path.stem},
                    source="local",
                    source_priority=1,
                )
            )
        return indexed

    def ingest_online(self, references: list[dict[str, Any]], http: Any = None) -> dict[str, dict[str, Any]]:
        del http
        indexed: dict[str, dict[str, Any]] = {}
        for reference in references:
            if not isinstance(reference, dict):
                continue
            url = self._resolve_pdf_url(reference)
            if not url:
                continue
            ref_id = self._resolve_ref_id(reference, str(reference.get("title", "paper")))
            dest_path = self.cache_dir / f"{_slug(ref_id)}.pdf"
            if not dest_path.is_file() and not self._download_pdf(url, dest_path):
                continue
            indexed.update(
                self._ingest_pdf(
                    dest_path,
                    reference=reference,
                    source="online",
                    source_priority=2,
                )
            )
        return indexed

    def _score_chunk(self, query: str, chunk: dict[str, Any]) -> float:
        haystack = " ".join(
            [
                str(chunk.get("paper_title", "")),
                str(chunk.get("section_title", "")),
                str(chunk.get("text", "")),
            ]
        ).lower()
        score = 0.0
        for token in _tokenize(query):
            if token and token in haystack:
                score += float(len(token))
        return score

    def query(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        scored: list[tuple[float, dict[str, Any]]] = []
        for chunk in self._chunks:
            score = self._score_chunk(text, chunk)
            if self.local_priority and int(chunk.get("source_priority", 2)) == 1:
                score += 0.1
            if score <= 0:
                continue
            scored.append((score, dict(chunk)))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _score, chunk in scored[:top_k]]

    def get_section(
        self,
        paper_id: str,
        section_title: str,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        filtered = [
            dict(chunk)
            for chunk in self._chunks
            if str(chunk.get("paper_id", "")).strip() == str(paper_id).strip()
            and str(section_title).strip().lower() in str(chunk.get("section_title", "")).strip().lower()
        ]
        scored: list[tuple[float, dict[str, Any]]] = []
        for chunk in filtered:
            score = self._score_chunk(query or section_title, chunk)
            if self.local_priority and int(chunk.get("source_priority", 2)) == 1:
                score += 0.1
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _score, chunk in scored[:top_k]]
