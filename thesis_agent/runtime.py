"""Runtime wiring for full v3 thesis agent."""

from __future__ import annotations

import os
import sys
from typing import Any

from .audit import JsonlAuditSink, build_event
from .config import Settings
from .engine import ThesisEngine
from .providers import AcademicSearchClient, CitationMetadataClient, HttpClient, LLMClient
from .stages import (
    stage1_literature,
    stage2_outline,
    stage3_write,
    stage4_verify,
    stage5_polish,
    stage6_export,
)
from .store import RunStore


def _log(msg: str) -> None:
    print(f"[thesis-agent] {msg}", file=sys.stderr, flush=True)


def _warn(msg: str) -> None:
    print(f"[thesis-agent] WARNING: {msg}", file=sys.stderr, flush=True)


class Runtime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = RunStore(base_dir=settings.runs_dir)
        # LLM streaming calls need a longer timeout; external REST APIs (Crossref,
        # Semantic Scholar, OpenAlex) can fail fast so we use a shorter one.
        self.llm_http = HttpClient(timeout_seconds=120)
        self.api_http = HttpClient(timeout_seconds=10)
        self.http = self.llm_http  # backwards compat alias
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
        """Load local reference files and build RAG index (failures only warn)."""
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

    def build_engine(self, run_id: str, output_format: str) -> ThesisEngine:
        audit_path = self.store.artifact_path(run_id, "audit.jsonl")
        audit_sink = JsonlAuditSink(audit_path)

        local_refs = self.local_refs or None
        rag_index = self.rag_index

        def _audit(stage: int, event_type: str, input_summary: str = "", output_summary: str = "") -> None:
            event = build_event(
                stage=stage,
                agent="orchestrator",
                event_type=event_type,
                model=self.settings.llm_model,
                tokens=0,
                latency_ms=0,
                cost_estimate=0.0,
                input_summary=input_summary,
                output_summary=output_summary,
            )
            audit_sink.append(event)

        def s1(ctx: Any) -> str:
            _audit(1, "stage_start", input_summary=ctx.state.get("topic", ""))
            result = stage1_literature(
                ctx.state, self.search, llm_client=self.llm, local_refs=local_refs
            )
            ctx.state["local_refs_count"] = len(self.local_refs)
            ctx.state["rag_enabled"] = rag_index is not None
            _audit(1, "stage_end", output_summary=f"references={len(ctx.state.get('references', []))}")
            return result

        def s2(ctx: Any) -> str:
            _audit(2, "stage_start")
            result = stage2_outline(ctx.state, self.llm)
            _audit(2, "stage_end", output_summary=f"chapters={len(ctx.state.get('chapter_plans', []))}")
            return result

        def s3(ctx: Any) -> str:
            _audit(3, "stage_start")
            result = stage3_write(ctx.state, self.llm, rag_index=rag_index)
            _audit(3, "stage_end", output_summary=f"chapter_results={len(ctx.state.get('chapter_results', []))}")
            return result

        def s4(ctx: Any) -> str:
            _audit(4, "stage_start")
            result = stage4_verify(ctx.state, self.metadata, self.llm)
            _audit(4, "stage_end", output_summary=f"flagged={len(ctx.state.get('flagged_citations', []))}")
            return result

        def s5(ctx: Any) -> str:
            _audit(5, "stage_start")
            result = stage5_polish(ctx.state, self.llm)
            _audit(5, "stage_end")
            return result

        def s6(ctx: Any) -> str:
            _audit(6, "stage_start", input_summary=output_format)
            result = stage6_export(ctx.state, self.store, run_id, output_format=output_format)
            _audit(6, "stage_end", output_summary=ctx.state.get("output_filepath", ""))
            return result

        return ThesisEngine(
            store=self.store,
            stages={1: s1, 2: s2, 3: s3, 4: s4, 5: s5, 6: s6},
        )

    def connectivity_check(self) -> dict[str, Any]:
        result: dict[str, Any] = {"llm": False, "semantic_scholar": False, "openalex": False, "crossref": False}

        try:
            self.llm.text(system="Reply with 'ok'.", user="ok", max_tokens=8)
            result["llm"] = True
        except Exception:  # noqa: BLE001
            result["llm"] = False

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

        result["ok"] = all(result.values())
        return result

    def debug_llm(self, route: str = "default") -> dict[str, Any]:
        return self.llm.debug_probe(route=route)
