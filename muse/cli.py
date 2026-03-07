"""CLI entrypoint for the Muse runtime."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from typing import Any

from .config import load_settings
from .runtime import Runtime
from .schemas import hydrate_thesis_state, new_thesis_state, validate_thesis_state


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
    resume.add_argument("--start-stage", type=int, default=0)
    resume.add_argument("--output-format", default="markdown", choices=["markdown", "latex", "pdf"])
    resume.add_argument("--refs-dir", default=None, help="Local reference files directory (overrides auto-detection)")
    resume.add_argument("--auto-approve", action="store_true")
    resume.set_defaults(func=cmd_resume)

    review = sub.add_parser("review", help="Record HITL decision for a run")
    review.add_argument("--run-id", required=True)
    review.add_argument("--stage", type=int, required=True)
    review.add_argument("--approve", action="store_true")
    review.add_argument("--comment", default="")
    review.set_defaults(func=cmd_review)

    export = sub.add_parser("export", help="Run export stage only")
    export.add_argument("--run-id", required=True)
    export.add_argument("--output-format", default="markdown", choices=["markdown", "latex", "pdf"])
    export.set_defaults(func=cmd_export)

    return parser


def cmd_check(args: argparse.Namespace) -> int:
    settings = load_settings()
    runtime = Runtime(settings)
    result = runtime.connectivity_check()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


def cmd_debug_llm(args: argparse.Namespace) -> int:
    settings = load_settings()
    runtime = Runtime(settings)
    result = runtime.debug_llm(route=str(args.route))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 1


def cmd_run(args: argparse.Namespace) -> int:
    settings = load_settings()
    # CLI --refs-dir overrides env var / auto-detection
    if args.refs_dir:
        resolved = os.path.abspath(args.refs_dir)
        settings = dataclasses.replace(settings, refs_dir=resolved if os.path.isdir(resolved) else None)
    runtime = Runtime(settings)
    run_id = runtime.store.create_run(topic=args.topic)

    state = new_thesis_state(
        project_id=run_id,
        topic=args.topic,
        discipline=args.discipline,
        language=args.language,
        format_standard=args.format_standard,
    )
    state["output_format"] = args.output_format
    runtime.store.save_state(run_id, state)

    engine = runtime.build_engine(run_id=run_id, output_format=args.output_format)
    result = engine.run(run_id=run_id, start_stage=1, auto_approve=bool(args.auto_approve))

    print(json.dumps({"run_id": run_id, **result}, ensure_ascii=False, indent=2))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    settings = load_settings()
    # CLI --refs-dir overrides env var / auto-detection
    if args.refs_dir:
        resolved = os.path.abspath(args.refs_dir)
        settings = dataclasses.replace(settings, refs_dir=resolved if os.path.isdir(resolved) else None)
    runtime = Runtime(settings)

    state = runtime.store.load_state(args.run_id)
    state = hydrate_thesis_state(state)
    validate_thesis_state(state)
    runtime.store.save_state(args.run_id, state)

    start_stage = args.start_stage if args.start_stage > 0 else int(state.get("current_stage", 1)) + 1
    engine = runtime.build_engine(run_id=args.run_id, output_format=args.output_format)
    result = engine.run(run_id=args.run_id, start_stage=start_stage, auto_approve=bool(args.auto_approve))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    settings = load_settings()
    runtime = Runtime(settings)

    feedback = {
        "stage": int(args.stage),
        "approved": bool(args.approve),
        "comment": args.comment,
    }
    runtime.store.append_hitl_feedback(args.run_id, feedback)

    state = hydrate_thesis_state(runtime.store.load_state(args.run_id))
    state.setdefault("hitl_feedback", []).append(feedback)
    runtime.store.save_state(args.run_id, state)

    print(json.dumps({"run_id": args.run_id, "saved": True, "feedback": feedback}, ensure_ascii=False, indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    settings = load_settings()
    runtime = Runtime(settings)

    state = hydrate_thesis_state(runtime.store.load_state(args.run_id))
    engine = runtime.build_engine(run_id=args.run_id, output_format=args.output_format)

    # Execute stage 6 only.
    runtime.store.save_state(args.run_id, state)
    result = engine.run(run_id=args.run_id, start_stage=6, auto_approve=True)
    latest = runtime.store.load_state(args.run_id)

    print(
        json.dumps(
            {
                "result": result,
                "output_filepath": latest.get("output_filepath"),
                "export_artifacts": latest.get("export_artifacts", {}),
                "export_warnings": latest.get("export_warnings", []),
                "stage6_status": latest.get("stage6_status"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if latest.get("stage6_status") == "completed" else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
