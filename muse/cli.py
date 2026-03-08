"""CLI entrypoint for the Muse runtime."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sqlite3
from typing import Any

from .config import load_settings
from .runtime import Runtime


_STAGE_CHOICES = ["research", "outline", "draft", "final"]
_STAGE_TO_ANCHOR_NODE = {
    "research": "review_refs",
    "outline": "approve_outline",
    "draft": "review_draft",
    "final": "approve_final",
}


def _is_missing_langgraph(exc: ModuleNotFoundError) -> bool:
    missing = str(exc.name or "")
    message = str(exc)
    return "langgraph" in missing or "langgraph" in message


def _load_graph_invoker():
    try:
        from .graph.launcher import invoke as invoke_graph
    except ModuleNotFoundError as exc:
        if _is_missing_langgraph(exc):
            raise RuntimeError(
                "LangGraph runtime dependencies are not installed. "
                "Install them with `pip install -r requirements.txt` or "
                "`pip install langgraph langgraph-checkpoint-sqlite`."
            ) from exc
        raise
    return invoke_graph


def _load_export_node_builder():
    from .graph.nodes.export import build_export_node

    return build_export_node


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="muse")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Validate provider configuration and connectivity")
    check.set_defaults(func=cmd_check)

    debug_llm = sub.add_parser("debug-llm", help="Debug LLM routing attempts and errors")
    debug_llm.add_argument("--route", default="default")
    debug_llm.set_defaults(func=cmd_debug_llm)

    run = sub.add_parser("run", help="Create a new run and execute pipeline")
    run.add_argument("--topic", required=True)
    run.add_argument("--discipline", default="general")
    run.add_argument("--language", default="zh")
    run.add_argument("--format-standard", default="GB/T 7714-2015")
    run.add_argument("--output-format", default="markdown", choices=["markdown", "latex", "pdf"])
    run.add_argument("--refs-dir", default=None, help="Local reference files directory (overrides auto-detection)")
    run.add_argument("--auto-approve", action="store_true", help="Skip HITL pauses")
    run.set_defaults(func=cmd_run)

    resume = sub.add_parser("resume", help="Resume an existing run")
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--refs-dir", default=None, help="Local reference files directory (overrides auto-detection)")
    resume.add_argument("--auto-approve", action="store_true")
    resume.set_defaults(func=cmd_resume)

    review = sub.add_parser("review", help="Record HITL decision for a run")
    review.add_argument("--run-id", required=True)
    review.add_argument("--stage", required=True, choices=_STAGE_CHOICES)
    review.add_argument("--approve", action="store_true")
    review.add_argument("--comment", default="")
    review.set_defaults(func=cmd_review)

    export = sub.add_parser("export", help="Run export stage only")
    export.add_argument("--run-id", required=True)
    export.add_argument("--output-format", default="markdown", choices=["markdown", "latex", "pdf"])
    export.set_defaults(func=cmd_export)

    return parser


def _runtime_from_args(args: argparse.Namespace) -> Runtime:
    settings = load_settings()
    if getattr(args, "refs_dir", None):
        resolved = os.path.abspath(args.refs_dir)
        settings = dataclasses.replace(settings, refs_dir=resolved if os.path.isdir(resolved) else None)
    return Runtime(settings)


def _interrupt_stage(result: dict[str, Any]) -> str:
    interrupts = result.get("__interrupt__") or []
    if not interrupts:
        return "unknown"
    value = getattr(interrupts[0], "value", {})
    if isinstance(value, dict) and value.get("stage"):
        return str(value["stage"])
    return "unknown"


def _persist_graph_state(runtime: Runtime, run_id: str, result: dict[str, Any]) -> dict[str, Any]:
    snapshot = {key: value for key, value in result.items() if key != "__interrupt__"}
    runtime.store.save_state(run_id, snapshot)
    return snapshot


def _graph_response(result: dict[str, Any], thread_id: str) -> dict[str, Any]:
    if result.get("__interrupt__"):
        return {
            "status": "waiting_hitl",
            "stage": _interrupt_stage(result),
            "thread_id": thread_id,
        }
    return {
        "status": "completed",
        "thread_id": thread_id,
        "output_filepath": result.get("output_filepath", ""),
    }


def _has_checkpoint_state(graph: Any, thread_id: str) -> bool:
    db_path = getattr(graph, "_muse_checkpoint_db", "")
    if not isinstance(db_path, str) or not db_path:
        return False
    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return False

    try:
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            if row and row[0]
        }
        if "checkpoints" not in tables:
            return False
        row = connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()
        return bool(row and int(row[0]) > 0)
    except sqlite3.Error:
        return False
    finally:
        connection.close()


def _load_saved_state(runtime: Runtime, run_id: str) -> dict[str, Any] | None:
    try:
        state = runtime.store.load_state(run_id)
    except FileNotFoundError:
        return None
    return state if isinstance(state, dict) else None


def _chapters_from_results(chapter_results: Any) -> dict[str, Any]:
    chapters: dict[str, Any] = {}
    if not isinstance(chapter_results, list):
        return chapters
    for result in chapter_results:
        if not isinstance(result, dict):
            continue
        chapter_id = str(result.get("chapter_id") or "").strip()
        if chapter_id:
            chapters[chapter_id] = result
    return chapters


def _normalize_saved_state(run_id: str, state: dict[str, Any]) -> dict[str, Any]:
    chapter_results = state.get("chapter_results", [])
    paper_package = state.get("paper_package")
    if not isinstance(paper_package, dict):
        paper_package = {}
    if chapter_results and "chapter_results" not in paper_package:
        paper_package = {
            **paper_package,
            "chapter_results": chapter_results,
            "thesis_summary": state.get("thesis_summary", paper_package.get("thesis_summary", "")),
        }

    chapters = state.get("chapters")
    if not isinstance(chapters, dict):
        chapters = _chapters_from_results(chapter_results)

    outline = state.get("outline")
    if not isinstance(outline, dict):
        legacy_outline = state.get("outline_json")
        outline = legacy_outline if isinstance(legacy_outline, dict) else {}

    review_feedback = state.get("review_feedback")
    if not isinstance(review_feedback, list):
        legacy_feedback = state.get("hitl_feedback")
        review_feedback = legacy_feedback if isinstance(legacy_feedback, list) else []

    return {
        "project_id": state.get("project_id", run_id),
        "topic": state.get("topic", ""),
        "discipline": state.get("discipline", "general"),
        "language": state.get("language", "zh"),
        "format_standard": state.get("format_standard", "GB/T 7714-2015"),
        "output_format": state.get("output_format", "markdown"),
        "references": state.get("references", []),
        "search_queries": state.get("search_queries", []),
        "literature_summary": state.get("literature_summary", ""),
        "outline": outline,
        "chapter_plans": state.get("chapter_plans", []),
        "chapters": chapters,
        "citation_uses": state.get("citation_uses", []),
        "citation_ledger": state.get("citation_ledger", {}),
        "claim_text_by_id": state.get("claim_text_by_id", {}),
        "thesis_summary": state.get("thesis_summary", ""),
        "verified_citations": state.get("verified_citations", []),
        "flagged_citations": state.get("flagged_citations", []),
        "paper_package": paper_package,
        "final_text": state.get("final_text", ""),
        "polish_notes": state.get("polish_notes", []),
        "abstract_zh": state.get("abstract_zh", ""),
        "abstract_en": state.get("abstract_en", ""),
        "keywords_zh": state.get("keywords_zh", []),
        "keywords_en": state.get("keywords_en", []),
        "output_filepath": state.get("output_filepath", ""),
        "export_artifacts": state.get("export_artifacts", {}),
        "export_warnings": state.get("export_warnings", []),
        "review_feedback": review_feedback,
        "rag_enabled": bool(state.get("rag_enabled", False)),
        "local_refs_count": int(state.get("local_refs_count", 0)),
    }


def _infer_resume_stage(state: dict[str, Any], feedback: dict[str, Any]) -> str:
    stage = feedback.get("stage")
    if isinstance(stage, str) and stage in _STAGE_TO_ANCHOR_NODE:
        return stage

    review_feedback = state.get("review_feedback")
    if isinstance(review_feedback, list):
        for entry in reversed(review_feedback):
            if isinstance(entry, dict):
                entry_stage = entry.get("stage")
                if isinstance(entry_stage, str) and entry_stage in _STAGE_TO_ANCHOR_NODE:
                    return entry_stage

    raw_stage = state.get("current_stage")
    try:
        current_stage = int(raw_stage)
    except (TypeError, ValueError):
        current_stage = 0
    stage_status = state.get(f"stage{current_stage}_status") if current_stage else None
    legacy_map = {1: "research", 2: "outline", 3: "draft", 5: "final"}
    if stage_status == "hitl_review" and current_stage in legacy_map:
        return legacy_map[current_stage]

    if state.get("output_filepath"):
        return "final"
    if state.get("abstract_zh") or state.get("abstract_en") or state.get("polish_notes"):
        return "final"
    if state.get("paper_package", {}).get("chapter_results") or state.get("chapters"):
        return "draft"
    if state.get("chapter_plans"):
        return "outline"
    return "research"


def _merge_review_feedback(state: dict[str, Any], feedback: dict[str, Any]) -> dict[str, Any]:
    restored = dict(state)
    history = restored.get("review_feedback")
    feedback_items = list(history) if isinstance(history, list) else []
    feedback_items.append(dict(feedback))
    restored["review_feedback"] = feedback_items
    return restored


def _resume_from_saved_state(
    graph: Any,
    state: dict[str, Any],
    *,
    thread_id: str,
    feedback: dict[str, Any],
    stage_hint: str | None = None,
) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    stage = stage_hint or _infer_resume_stage(state, feedback)
    anchor = _STAGE_TO_ANCHOR_NODE[stage]
    restored = _merge_review_feedback(state, feedback)
    graph.update_state(config, restored, as_node=anchor)
    return graph.invoke(None, config=config)


def cmd_check(args: argparse.Namespace) -> int:
    runtime = _runtime_from_args(args)
    result = runtime.connectivity_check()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


def cmd_debug_llm(args: argparse.Namespace) -> int:
    runtime = _runtime_from_args(args)
    result = runtime.debug_llm(route=str(args.route))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 1


def cmd_run(args: argparse.Namespace) -> int:
    invoke_graph = _load_graph_invoker()
    runtime = _runtime_from_args(args)
    run_id = runtime.store.create_run(topic=args.topic)
    initial_state = {
        "project_id": run_id,
        "topic": args.topic,
        "discipline": args.discipline,
        "language": args.language,
        "format_standard": args.format_standard,
        "output_format": args.output_format,
    }
    graph = runtime.build_graph(thread_id=run_id, auto_approve=bool(args.auto_approve))
    result = invoke_graph(graph, initial_state, thread_id=run_id)
    _persist_graph_state(runtime, run_id, result)
    print(json.dumps({"run_id": run_id, **_graph_response(result, run_id)}, ensure_ascii=False, indent=2))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    invoke_graph = _load_graph_invoker()
    runtime = _runtime_from_args(args)
    feedback_items = runtime.store.load_hitl_feedback(args.run_id)
    feedback = feedback_items[-1] if feedback_items else {"approved": True}
    graph = runtime.build_graph(thread_id=args.run_id, auto_approve=bool(args.auto_approve))
    saved_state = _load_saved_state(runtime, args.run_id)
    if _has_checkpoint_state(graph, args.run_id) or saved_state is None:
        result = invoke_graph(graph, None, thread_id=args.run_id, resume=feedback)
    else:
        stage_hint = _infer_resume_stage(saved_state, feedback)
        result = _resume_from_saved_state(
            graph,
            _normalize_saved_state(args.run_id, saved_state),
            thread_id=args.run_id,
            feedback=feedback,
            stage_hint=stage_hint,
        )
    _persist_graph_state(runtime, args.run_id, result)
    print(json.dumps(_graph_response(result, args.run_id), ensure_ascii=False, indent=2))
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    runtime = _runtime_from_args(args)
    feedback = {
        "stage": str(args.stage),
        "approved": bool(args.approve),
        "comment": args.comment,
    }
    runtime.store.append_hitl_feedback(args.run_id, feedback)
    print(json.dumps({"run_id": args.run_id, "saved": True, "feedback": feedback}, ensure_ascii=False, indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    runtime = _runtime_from_args(args)
    state = runtime.store.load_state(args.run_id)
    state["output_format"] = args.output_format
    build_export_node = _load_export_node_builder()
    export_node = build_export_node(runtime.settings)
    updated = export_node(state)
    state.update(updated)
    runtime.store.save_state(args.run_id, state)
    print(
        json.dumps(
            {
                "output_filepath": state.get("output_filepath"),
                "export_artifacts": state.get("export_artifacts", {}),
                "export_warnings": state.get("export_warnings", []),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if state.get("output_filepath") else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
