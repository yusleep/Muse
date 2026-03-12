"""Runtime wiring for Muse."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .config import Settings
from .providers import AcademicSearchClient, CitationMetadataClient, HttpClient, LLMClient
from .store import RunStore


_PUBLIC_EXPORT_FORMATS = {"md", "markdown", "latex", "pdf"}
_LANGGRAPH_INSTALL_HINT = (
    "LangGraph runtime dependencies are not installed. "
    "Install them with `pip install -r requirements.txt` or "
    "`pip install langgraph langgraph-checkpoint-sqlite`."
)


def _log(msg: str) -> None:
    print(f"[muse] {msg}", file=sys.stderr, flush=True)


def _warn(msg: str) -> None:
    print(f"[muse] WARNING: {msg}", file=sys.stderr, flush=True)


def _is_missing_langgraph(exc: ModuleNotFoundError) -> bool:
    missing = str(exc.name or "")
    message = str(exc)
    return "langgraph" in missing or "langgraph" in message


def _load_graph_builder():
    try:
        from .graph.launcher import build_graph as build_langgraph
    except ModuleNotFoundError as exc:
        if _is_missing_langgraph(exc):
            raise RuntimeError(_LANGGRAPH_INSTALL_HINT) from exc
        raise
    return build_langgraph


class Runtime:
    def __init__(self, settings: Settings) -> None:
        from .agents.executor import SubagentExecutor
        from .memory.store import MemoryStore
        from .sandbox.local import LocalSandbox

        self.settings = settings
        self.store = RunStore(base_dir=settings.runs_dir)
        self.llm_http = HttpClient(timeout_seconds=120)
        self.api_http = HttpClient(timeout_seconds=10)
        self.http = self.llm_http
        runtime_dir = Path(settings.runs_dir) / "_runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self.memory_store = MemoryStore(runtime_dir / "memory.sqlite")
        self.subagent_executor = SubagentExecutor(max_concurrent=3)
        self.sandbox = LocalSandbox(runtime_dir / "sandbox")
        self.llm = LLMClient(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            http=self.llm_http,
            model_router_config=settings.model_router_config,
            env=dict(os.environ),
        )
        self.search = AcademicSearchClient(
            http=self.api_http,
            semantic_scholar_api_key=settings.semantic_scholar_api_key,
            openalex_email=settings.openalex_email,
        )
        self.metadata = CitationMetadataClient(
            http=self.api_http,
            crossref_mailto=settings.crossref_mailto,
        )

        self.local_refs: list[dict[str, Any]] = []
        self.rag_index: Any = None
        if settings.refs_dir:
            self._init_local_refs(settings.refs_dir)

    def _init_local_refs(self, refs_dir: str) -> None:
        try:
            from .refs_loader import load_local_refs

            self.local_refs = load_local_refs(refs_dir)
            _log(f"Loaded {len(self.local_refs)} local reference(s) from {refs_dir}")
        except Exception as exc:  # noqa: BLE001
            _warn(f"Failed to load local refs from {refs_dir!r}: {exc}")
            self.local_refs = []

        if self.local_refs:
            try:
                from .rag import RagIndex

                self.rag_index = RagIndex.build(self.local_refs, refs_dir)
                _log(f"RAG index built ({len(self.local_refs)} refs)")
            except Exception as exc:  # noqa: BLE001
                _warn(f"Failed to build RAG index: {exc}")
                self.rag_index = None

    def build_graph(self, *, thread_id: str, auto_approve: bool = True):
        build_langgraph = _load_graph_builder()
        return build_langgraph(
            self.settings,
            services=self,
            thread_id=thread_id,
            auto_approve=auto_approve,
        )

    def connectivity_check(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "llm": False,
            "semantic_scholar": None,
            "openalex": False,
            "crossref": False,
        }

        try:
            self.llm.text(system="Reply with 'ok'.", user="ok", max_tokens=8)
            result["llm"] = True
        except Exception:  # noqa: BLE001
            result["llm"] = False

        if self.settings.semantic_scholar_api_key:
            try:
                self.search.search_semantic_scholar("test", limit=1)
                result["semantic_scholar"] = True
            except Exception:  # noqa: BLE001
                result["semantic_scholar"] = False

        try:
            self.search.search_openalex("test", limit=1)
            result["openalex"] = True
        except Exception:  # noqa: BLE001
            result["openalex"] = False

        try:
            result["crossref"] = self.metadata.verify_doi("10.1038/nphys1170")
        except Exception:  # noqa: BLE001
            result["crossref"] = False

        semantic_ok = result["semantic_scholar"] is not False
        result["ok"] = bool(result["llm"] and result["openalex"] and result["crossref"] and semantic_ok)
        return result

    def debug_llm(self, route: str = "default") -> dict[str, Any]:
        return self.llm.debug_probe(route=route)
