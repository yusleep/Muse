"""Export helpers and node that persist the composed paper package."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from muse.graph.helpers.prompt_optimizer import PromptOptimizer
from muse.prompts.section_write import BASE_SECTION_WRITE_SYSTEM_PROMPT
from muse.services.store import RunStore


def _chapter_results_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    chapter_results = state.get("chapter_results")
    if isinstance(chapter_results, list) and chapter_results:
        return chapter_results
    paper_package = state.get("paper_package", {})
    if isinstance(paper_package, dict):
        nested = paper_package.get("chapter_results")
        if isinstance(nested, list) and nested:
            return nested
    # Fallback: reconstruct from chapters dict (uses _merge_dict, always preserved)
    chapters = state.get("chapters")
    if isinstance(chapters, dict) and chapters:
        chapter_plans = state.get("chapter_plans", [])
        if chapter_plans:
            return [chapters[str(p["chapter_id"])] for p in chapter_plans if str(p.get("chapter_id", "")) in chapters]
        return list(chapters.values())
    return []


def _gate_export(state: dict[str, Any]) -> tuple[bool, str]:
    flagged = state.get("flagged_citations", [])
    if not isinstance(flagged, list):
        return True, "ok"

    contradictions = [
        flagged_item
        for flagged_item in flagged
        if isinstance(flagged_item, dict)
        and flagged_item.get("reason") == "unsupported_claim"
        and "contradiction" in str(flagged_item.get("detail", ""))
    ]
    if contradictions:
        return False, f"{len(contradictions)} claims contradicted by cited sources"

    return True, "ok"


def _write_export_artifacts(
    state: dict[str, Any],
    store: Any,
    run_id: str,
    *,
    output_format: str = "markdown",
) -> dict[str, Any]:
    text = state.get("final_text") or "\n\n".join(
        chapter.get("merged_text", "") for chapter in _chapter_results_from_state(state)
    )

    fmt = output_format.lower().strip()
    export_artifacts: dict[str, Any] = {}
    export_warnings: list[str] = []

    md_path = store.artifact_path(run_id, "output/thesis.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(text)

    if fmt in {"md", "markdown"}:
        path = md_path
    elif fmt == "latex":
        from muse.latex_export import export_latex_project

        export_state = dict(state)
        export_state["export_artifacts"] = export_artifacts
        export_state["export_warnings"] = export_warnings
        path = export_latex_project(export_state, store, run_id)
        export_artifacts = export_state.get("export_artifacts", {})
        export_warnings = export_state.get("export_warnings", [])
    elif fmt == "pdf":
        path = store.artifact_path(run_id, "output/thesis.pdf")
        _pandoc_export(md_path, path, "pdf")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    return {
        "output_format": fmt,
        "output_filepath": path,
        "export_artifacts": export_artifacts,
        "export_warnings": export_warnings,
    }


def _aggregate_quality_scores(chapter_results: list[dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for chapter in chapter_results:
        if not isinstance(chapter, dict):
            continue
        scores = chapter.get("quality_scores", {})
        if not isinstance(scores, dict):
            continue
        for key, value in scores.items():
            if isinstance(value, (int, float)):
                buckets.setdefault(str(key), []).append(float(value))
    return {
        key: sum(values) / len(values)
        for key, values in buckets.items()
        if values
    }


def _run_prompt_optimizer(
    *,
    state: dict[str, Any],
    settings: Any,
    services: Any | None,
    run_id: str,
) -> tuple[dict[str, Any], list[str]]:
    llm = getattr(services, "llm", None) if services is not None else None
    chapter_results = _chapter_results_from_state(state)
    aggregated_scores = _aggregate_quality_scores(chapter_results)
    if not aggregated_scores:
        return {}, []

    optimizer = PromptOptimizer(Path(settings.runs_dir) / "_prompt_bank")
    selected_prompt = optimizer.select_prompt(
        "section_write",
        BASE_SECTION_WRITE_SYSTEM_PROMPT,
    )
    selected_prompt_id = optimizer.record_result(
        "section_write",
        selected_prompt,
        aggregated_scores,
        run_id=run_id,
        baseline_prompt=BASE_SECTION_WRITE_SYSTEM_PROMPT,
    )
    weaknesses = optimizer.analyze_weakness(aggregated_scores)
    summary = {
        "prompt_name": "section_write",
        "active_prompt_id": selected_prompt_id,
        "scores": aggregated_scores,
        "weaknesses": weaknesses,
    }
    warnings: list[str] = []

    if not weaknesses:
        return summary, warnings

    try:
        candidate_prompt = optimizer.generate_improvement(
            "section_write",
            selected_prompt,
            weaknesses,
            llm,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"prompt optimizer skipped: {exc}")
        return summary, warnings

    if candidate_prompt and candidate_prompt.strip() != selected_prompt.strip():
        candidate_id = optimizer.add_candidate(
            "section_write",
            candidate_prompt,
            weaknesses,
            source_prompt_id=selected_prompt_id,
            source_run_id=run_id,
            baseline_prompt=BASE_SECTION_WRITE_SYSTEM_PROMPT,
        )
        summary["candidate_prompt_id"] = candidate_id
    return summary, warnings


def _run_export(
    state: dict[str, Any],
    store: Any,
    run_id: str,
    *,
    output_format: str = "markdown",
) -> tuple[str, dict[str, Any]]:
    allowed, _ = _gate_export(state)
    if not allowed:
        return "blocked", {
            "stage6_status": "blocked",
            "output_filepath": "",
            "export_artifacts": {},
            "export_warnings": [],
        }

    outputs = _write_export_artifacts(
        state,
        store,
        run_id,
        output_format=output_format,
    )
    outputs["stage6_status"] = "completed"
    outputs["current_stage"] = 6
    return "done", outputs


def _pandoc_export(md_path: str, output_path: str, fmt: str) -> None:
    """Convert markdown thesis sources to `latex` or `pdf` via pandoc."""

    if not shutil.which("pandoc"):
        raise RuntimeError("pandoc is not installed. Install with: sudo apt-get install pandoc")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cjk_args = [
        "--from", "markdown+tex_math_single_backslash",
        "-V", "documentclass=ctexart",
        "-V", "classoption=UTF8,a4paper,12pt",
        "-V", "geometry:top=2.5cm, bottom=2.5cm, left=3cm, right=2.5cm",
        "-V", "CJKmainfont=Noto Serif CJK SC",
        "-V", "CJKsansfont=Noto Sans CJK SC",
        "-V", "CJKmonofont=Noto Sans CJK SC",
        "-V", "linestretch=1.5",
        "--toc", "--toc-depth=3", "--number-sections",
    ]

    def _run(args: list[str]) -> None:
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"pandoc failed (exit {result.returncode}):\n" + (result.stderr or result.stdout)[-2000:]
            )

    if fmt == "latex":
        _run(["pandoc", md_path] + cjk_args + ["-o", output_path])
        return

    if fmt == "pdf":
        if not shutil.which("xelatex"):
            raise RuntimeError(
                "xelatex is not installed. Install TeX Live XeLaTeX, e.g. sudo apt-get install texlive-xetex"
            )
        tex_path = os.path.splitext(output_path)[0] + ".tex"
        _run(["pandoc", md_path] + cjk_args + ["--pdf-engine=xelatex", "-o", output_path])
        if not os.path.exists(tex_path):
            _run(["pandoc", md_path] + cjk_args + ["-o", tex_path])
        return

    raise ValueError(f"_pandoc_export: unsupported fmt {fmt!r}")


def build_export_node(settings: Any, services: Any | None = None):
    store = RunStore(base_dir=settings.runs_dir)

    def export(state: dict[str, Any]) -> dict[str, Any]:
        run_id = state.get("project_id", "run")
        temp_state = {
            "final_text": state.get("final_text", ""),
            "chapter_results": state.get("paper_package", {}).get("chapter_results", []),
            "chapters": state.get("chapters", {}),
            "chapter_plans": state.get("chapter_plans", []),
            "flagged_citations": state.get("flagged_citations", []),
            "references": state.get("references", []),
            "citation_uses": state.get("citation_uses", []),
            "output_format": state.get("output_format", "markdown"),
            "abstract_zh": state.get("abstract_zh", ""),
            "abstract_en": state.get("abstract_en", ""),
            "keywords_zh": state.get("keywords_zh", []),
            "keywords_en": state.get("keywords_en", []),
        }
        _, outputs = _run_export(
            temp_state,
            store,
            run_id,
            output_format=state.get("output_format", "markdown"),
        )
        optimizer_summary, optimizer_warnings = _run_prompt_optimizer(
            state=temp_state,
            settings=settings,
            services=services,
            run_id=run_id,
        )
        export_warnings = list(outputs.get("export_warnings", []))
        export_warnings.extend(optimizer_warnings)
        paper_package = {
            **state.get("paper_package", {}),
            "export_artifacts": outputs.get("export_artifacts", {}),
            "export_warnings": export_warnings,
        }
        if optimizer_summary:
            paper_package["prompt_optimizer"] = optimizer_summary
        return {
            "output_filepath": outputs.get("output_filepath", ""),
            "export_artifacts": outputs.get("export_artifacts", {}),
            "export_warnings": export_warnings,
            "paper_package": paper_package,
        }

    return export
