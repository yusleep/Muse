"""Stage implementations for full v3 thesis pipeline."""

from __future__ import annotations

import json
import os
import sys
from typing import Any


def _log(msg: str) -> None:
    print(f"[thesis-agent] {msg}", flush=True, file=sys.stderr)

from .chapter import apply_chapter_review
from .citation import verify_all_citations
from .orchestrator import gate_export
from .planning import plan_subtasks


def _generate_search_queries(llm_client: Any, topic: str, discipline: str, count: int = 7) -> list[str]:
    """Ask the LLM to generate diverse academic search queries for the topic."""
    system = (
        f"Generate {count} diverse English academic search queries for the given research topic. "
        "Cover: core concepts, methodology, sub-topics, related fields, and key debates. "
        "Return JSON with key: queries (list of strings)."
    )
    user = json.dumps({"topic": topic, "discipline": discipline}, ensure_ascii=False)
    try:
        out = llm_client.structured(system=system, user=user, route="outline", max_tokens=500)
        queries = out.get("queries", []) if isinstance(out, dict) else []
        if isinstance(queries, list) and queries:
            return [str(q).strip() for q in queries if str(q).strip()][:count]
    except Exception:  # noqa: BLE001
        pass
    return []


def stage1_literature(
    state: dict[str, Any],
    search_client: Any,
    llm_client: Any = None,
    local_refs: list[dict[str, Any]] | None = None,
) -> str:
    _log("Stage 1: searching literature sources...")

    extra_queries: list[str] | None = None
    if llm_client is not None:
        extra_queries = _generate_search_queries(
            llm_client, state["topic"], state.get("discipline", "")
        ) or None

    references, queries = search_client.search_multi_source(
        topic=state["topic"],
        discipline=state.get("discipline", ""),
        extra_queries=extra_queries,
    )

    # Prepend local refs (higher priority) and deduplicate by ref_id
    if local_refs:
        seen_ids = {r["ref_id"] for r in local_refs}
        online_deduped = [r for r in references if r.get("ref_id") not in seen_ids]
        references = local_refs + online_deduped
        _log(f"Stage 1: {len(local_refs)} local + {len(online_deduped)} online references")
    else:
        _log(f"Stage 1: {len(references)} online references")

    state["search_queries"] = queries
    state["references"] = references
    state["literature_summary"] = _summarize_references(references)
    state["stage1_status"] = "hitl_review"
    state["current_stage"] = 1
    _log(f"Stage 1 done: {len(references)} references total")
    return "hitl"


def _analyze_topic(
    llm_client: Any, topic: str, discipline: str, literature_summary: str
) -> dict[str, Any]:
    """Pre-analyze the topic to surface research gaps and domain context."""
    system = (
        "Analyze this research topic and return JSON with keys: "
        "research_gaps (list), core_concepts (list), methodology_domain (string), "
        "suggested_contributions (list). "
        "Be specific to the discipline."
    )
    user = json.dumps(
        {"topic": topic, "discipline": discipline, "literature_summary": literature_summary},
        ensure_ascii=False,
    )
    try:
        out = llm_client.structured(system=system, user=user, route="outline", max_tokens=800)
        if isinstance(out, dict) and out.get("research_gaps"):
            return out
    except Exception:  # noqa: BLE001
        pass
    return {
        "research_gaps": [],
        "core_concepts": [],
        "methodology_domain": "general",
        "suggested_contributions": [],
    }


def stage2_outline(state: dict[str, Any], llm_client: Any) -> str:
    _log("Stage 2: generating outline...")

    topic_analysis = _analyze_topic(
        llm_client,
        state["topic"],
        state.get("discipline", ""),
        state.get("literature_summary", ""),
    )
    _log(f"Stage 2: topic analysis done, methodology_domain={topic_analysis.get('methodology_domain')}")

    prompt = (
        "Generate a thesis outline as JSON with keys: chapters (list). Each chapter must include "
        "chapter_id, chapter_title, target_words, complexity, subsections (list of {title}). "
        "Use the topic_analysis to create a discipline-specific, non-generic structure. "
        "For CS/systems topics include: background, related work, system design, evaluation, conclusion. "
        "For social science topics include: literature review, theory, methods, findings, discussion."
    )
    context = {
        "topic": state["topic"],
        "discipline": state.get("discipline", ""),
        "language": state.get("language", "zh"),
        "literature_summary": state.get("literature_summary", ""),
        "topic_analysis": topic_analysis,
    }

    result = llm_client.structured(
        system=prompt,
        user=json.dumps(context, ensure_ascii=False),
        route="outline",
        max_tokens=3000,
    )
    chapters = result.get("chapters", []) if isinstance(result, dict) else []
    if not chapters:
        chapters = _fallback_outline(state["topic"])

    chapter_plans = []
    for idx, chapter in enumerate(chapters, start=1):
        target_words = int(chapter.get("target_words", 3000))
        complexity = str(chapter.get("complexity", "medium"))
        subsections = chapter.get("subsections", [])
        subtasks = plan_subtasks(target_words=target_words, complexity=complexity, subsections=subsections)
        chapter_plans.append(
            {
                "chapter_id": chapter.get("chapter_id", f"ch_{idx:02d}"),
                "chapter_title": chapter.get("chapter_title", f"Chapter {idx}"),
                "target_words": target_words,
                "complexity": complexity,
                "subtask_plan": subtasks,
            }
        )

    state["outline_json"] = {"chapters": chapters}
    state["chapter_plans"] = chapter_plans
    state["stage2_status"] = "hitl_review"
    state["current_stage"] = 2
    _log(f"Stage 2 done: {len(chapter_plans)} chapters planned")
    return "hitl"


def stage3_write(state: dict[str, Any], llm_client: Any, rag_index: Any = None) -> str:
    total_chapters = len(state.get("chapter_plans", []))
    _log(f"Stage 3: writing {total_chapters} chapters...")
    chapter_results = []
    all_citation_uses = []
    claim_text_by_id: dict[str, str] = {}

    for ch_idx, chapter_plan in enumerate(state.get("chapter_plans", []), start=1):
        chapter_title = chapter_plan["chapter_title"]
        subtask_plan = chapter_plan.get("subtask_plan", [])
        _log(f"Stage 3: chapter {ch_idx}/{total_chapters} '{chapter_title}' ({len(subtask_plan)} subtasks)")
        subtask_results = []
        current_iteration = 0
        max_iterations = 3
        revision_instructions: dict[str, str] = {}

        while current_iteration < max_iterations:
            subtask_results = _write_subtasks(
                llm_client=llm_client,
                state=state,
                chapter_title=chapter_title,
                subtask_plan=subtask_plan,
                revision_instructions=revision_instructions,
                previous=subtask_results,
                rag_index=rag_index,
            )

            merged = "\n\n".join(item["output_text"] for item in subtask_results)
            review = _review_chapter(llm_client=llm_client, chapter_title=chapter_title, merged_text=merged)

            chapter_state = {
                "quality_scores": review.get("scores", {}),
                "review_notes": review.get("review_notes", []),
                "revision_instructions": revision_instructions,
                "current_iteration": current_iteration,
                "max_iterations": max_iterations,
            }
            route, updated = apply_chapter_review(chapter_state, review, score_threshold=4, min_severity=2)
            revision_instructions = updated.get("revision_instructions", {})
            current_iteration = int(updated.get("current_iteration", current_iteration + 1))
            if route == "done":
                break

        for sub in subtask_results:
            for claim_idx, claim in enumerate(sub.get("key_claims", []), start=1):
                claim_id = f"{chapter_plan['chapter_id']}_{sub['subtask_id']}_c{claim_idx:02d}"
                claim_text_by_id[claim_id] = claim
                for cite in sub.get("citations_used", []):
                    all_citation_uses.append(
                        {
                            "cite_key": cite,
                            "claim_id": claim_id,
                            "chapter_id": chapter_plan["chapter_id"],
                            "subtask_id": sub["subtask_id"],
                        }
                    )

        chapter_results.append(
            {
                "chapter_id": chapter_plan["chapter_id"],
                "chapter_title": chapter_title,
                "target_words": chapter_plan.get("target_words", 0),
                "complexity": chapter_plan.get("complexity", "medium"),
                "subtask_results": subtask_results,
                "merged_text": "\n\n".join(item["output_text"] for item in subtask_results),
                "quality_scores": review.get("scores", {}),
                "iterations_used": current_iteration,
            }
        )

    state["chapter_results"] = chapter_results
    state["citation_uses"] = all_citation_uses
    state["claim_text_by_id"] = claim_text_by_id
    state["thesis_summary"] = _build_thesis_summary(chapter_results)
    state["stage3_status"] = "hitl_review"
    state["current_stage"] = 3
    return "hitl"


def stage4_verify(state: dict[str, Any], metadata_client: Any, llm_client: Any) -> str:
    citation_uses = state.get("citation_uses", [])
    unique_keys = len({u.get("cite_key") for u in citation_uses if u.get("cite_key")})
    _log(f"Stage 4: verifying {len(citation_uses)} citation uses ({unique_keys} unique keys)...")
    references = state.get("references", [])

    def retrieve_passage(ref: dict, claim: str) -> str:
        abstract = ref.get("abstract") or ""
        return str(abstract)[:2000] if abstract else f"Title: {ref.get('title', '')}"

    def check_entailment(premise: str, hypothesis: str) -> str:
        return llm_client.entailment(premise=premise, hypothesis=hypothesis, route="reasoning")

    verified, flagged = verify_all_citations(
        references=references,
        citation_uses=state.get("citation_uses", []),
        claim_text_by_id=state.get("claim_text_by_id", {}),
        verify_doi=metadata_client.verify_doi,
        crosscheck_metadata=metadata_client.crosscheck_metadata,
        retrieve_passage=retrieve_passage,
        check_entailment=check_entailment,
    )

    state["verified_citations"] = verified
    state["flagged_citations"] = flagged
    state["stage4_status"] = "completed"
    state["current_stage"] = 4
    _log(f"Stage 4 done: {len(verified)} verified, {len(flagged)} flagged")
    return "ok"


def stage5_polish(state: dict[str, Any], llm_client: Any) -> str:
    _log("Stage 5: polishing chapters...")
    chapter_results = state.get("chapter_results", [])
    polished_chapters: list[str] = []
    all_notes: list[str] = []

    system = (
        "Polish the academic thesis chapter for consistency and clarity. "
        "Do not alter core claims. Return JSON with keys: final_text, polish_notes (list)."
    )

    for ch in chapter_results:
        chapter_title = ch.get("chapter_title", "")
        chapter_text = ch.get("merged_text", "")
        if not chapter_text.strip():
            polished_chapters.append(chapter_text)
            continue

        user = json.dumps(
            {
                "language": state.get("language", "zh"),
                "format_standard": state.get("format_standard", "GB/T 7714-2015"),
                "chapter_title": chapter_title,
                "text": chapter_text,
            },
            ensure_ascii=False,
        )
        try:
            out = llm_client.structured(system=system, user=user, route="polish", max_tokens=4500)
            polished = out.get("final_text") if isinstance(out, dict) else None
            notes = out.get("polish_notes", []) if isinstance(out, dict) else []
            if isinstance(polished, str) and polished.strip():
                polished_chapters.append(polished)
            else:
                polished_chapters.append(chapter_text)
            if isinstance(notes, list):
                all_notes.extend(f"[{chapter_title}] {n}" for n in notes)
        except Exception:  # noqa: BLE001
            polished_chapters.append(chapter_text)

    state["final_text"] = "\n\n".join(polished_chapters)
    state["polish_notes"] = all_notes
    state["stage5_status"] = "hitl_review"
    state["current_stage"] = 5
    _log(f"Stage 5 done: polished {len(polished_chapters)} chapters")
    return "hitl"


def stage6_export(state: dict[str, Any], store: Any, run_id: str, output_format: str = "markdown") -> str:
    _log(f"Stage 6: exporting ({output_format})...")
    allowed, _ = gate_export(state)
    if not allowed:
        state["stage6_status"] = "blocked"
        return "blocked"

    text = state.get("final_text") or "\n\n".join(
        ch.get("merged_text", "") for ch in state.get("chapter_results", [])
    )

    fmt = output_format.lower().strip()

    # Always write the markdown source first – it's the pandoc input and a
    # useful artefact on its own.
    md_path = store.artifact_path(run_id, "output/thesis.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(text)

    if fmt in {"md", "markdown"}:
        path = md_path
    elif fmt == "latex":
        path = store.artifact_path(run_id, "output/thesis.tex")
        _pandoc_export(md_path, path, "latex")
    elif fmt == "pdf":
        path = store.artifact_path(run_id, "output/thesis.pdf")
        _pandoc_export(md_path, path, "pdf")
    elif fmt == "docx":
        path = store.artifact_path(run_id, "output/thesis.docx")
        _pandoc_export(md_path, path, "docx")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    state["output_format"] = fmt
    state["output_filepath"] = path
    state["stage6_status"] = "completed"
    state["current_stage"] = 6
    return "done"


def _write_subtasks(
    *,
    llm_client: Any,
    state: dict[str, Any],
    chapter_title: str,
    subtask_plan: list[dict[str, Any]],
    revision_instructions: dict[str, str],
    previous: list[dict[str, Any]],
    rag_index: Any = None,
) -> list[dict[str, Any]]:
    results = []
    prev_text = ""
    prev_by_id = {item["subtask_id"]: item for item in previous}

    for subtask in subtask_plan:
        sid = subtask["subtask_id"]
        if sid in prev_by_id and sid not in revision_instructions:
            kept = dict(prev_by_id[sid])
            results.append(kept)
            prev_text = kept.get("output_text", "")
            continue

        # Build a compact snapshot of available references so the LLM can
        # cite using the actual ref_id keys stored in the reference index.
        refs_snapshot = [
            {
                "ref_id": r["ref_id"],
                "title": r.get("title", ""),
                "year": r.get("year"),
                "abstract": (r.get("abstract") or "")[:300],
            }
            for r in state.get("references", [])
            if isinstance(r, dict) and r.get("ref_id")
        ][:30]

        # RAG: retrieve locally-indexed chunks relevant to this subtask
        local_context: list[dict[str, Any]] = []
        if rag_index is not None:
            query = f"{chapter_title} {subtask.get('title', '')} {state.get('topic', '')}"
            try:
                local_context = rag_index.retrieve(query, top_k=5)
            except Exception:  # noqa: BLE001
                local_context = []

        system = (
            "Write one thesis subsection with citations. "
            "IMPORTANT: for citations_used, use ONLY ref_id values from the available_references list. "
            "Do not invent citation keys not in that list. "
            "Include specific technical details, mathematical notation where appropriate, "
            "and reference concrete experimental results. "
            "Return JSON with keys: "
            "text, citations_used (list of ref_id strings), key_claims (list), transition_out, "
            "glossary_additions (object), "
            "self_assessment (object with confidence, weak_spots, needs_revision)."
        )
        user_payload: dict[str, Any] = {
            "topic": state.get("topic", ""),
            "chapter_title": chapter_title,
            "subtask": subtask,
            "language": state.get("language", "zh"),
            "available_references": refs_snapshot,
            "allowed_refs": [r["ref_id"] for r in refs_snapshot],
            "previous_subsection": prev_text,
            "revision_instruction": revision_instructions.get(sid),
        }
        if local_context:
            user_payload["local_context"] = local_context
        user = json.dumps(user_payload, ensure_ascii=False)

        out = llm_client.structured(system=system, user=user, route="writing", max_tokens=2800)
        text = str(out.get("text", "")).strip()
        if not text:
            text = f"[{chapter_title}] {subtask['title']}\n\n(LLM returned empty content.)"

        citations_used = out.get("citations_used", [])
        if not isinstance(citations_used, list):
            citations_used = []

        key_claims = out.get("key_claims", [])
        if not isinstance(key_claims, list):
            key_claims = []

        assessment = out.get("self_assessment", {})
        if not isinstance(assessment, dict):
            assessment = {}

        results.append(
            {
                "subtask_id": sid,
                "title": subtask.get("title", ""),
                "target_words": subtask.get("target_words", 1200),
                "output_text": text,
                "actual_words": len(text.split()),
                "citations_used": [str(c).strip() for c in citations_used if str(c).strip()],
                "key_claims": [str(c).strip() for c in key_claims if str(c).strip()],
                "transition_out": str(out.get("transition_out", "")),
                "glossary_additions": out.get("glossary_additions", {})
                if isinstance(out.get("glossary_additions", {}), dict)
                else {},
                "confidence": float(assessment.get("confidence", 0.5)),
                "weak_spots": assessment.get("weak_spots", [])
                if isinstance(assessment.get("weak_spots", []), list)
                else [],
                "needs_revision": bool(assessment.get("needs_revision", False)),
            }
        )
        prev_text = text

    return results


def _review_chapter(*, llm_client: Any, chapter_title: str, merged_text: str) -> dict[str, Any]:
    system = (
        "You are a strict thesis reviewer. Return JSON with keys: scores (object) and review_notes (list). "
        "scores keys: coherence, logic, citation, term_consistency, balance, redundancy; values 1-5."
    )
    user = json.dumps(
        {
            "chapter_title": chapter_title,
            "text": merged_text,
        },
        ensure_ascii=False,
    )

    out = llm_client.structured(system=system, user=user, route="review", max_tokens=1800)
    if not isinstance(out, dict):
        return {"scores": {}, "review_notes": []}

    scores = out.get("scores", {})
    if not isinstance(scores, dict):
        scores = {}
    review_notes = out.get("review_notes", [])
    if not isinstance(review_notes, list):
        review_notes = []
    return {"scores": scores, "review_notes": review_notes}


def _summarize_references(references: list[dict[str, Any]]) -> str:
    if not references:
        return "No references found."
    top = references[:12]
    bullets = []
    for ref in top:
        title = ref.get("title", "Untitled")
        year = ref.get("year", "n.d.")
        venue = ref.get("venue", "")
        bullets.append(f"- {title} ({year}) {venue}".strip())
    return "\n".join(bullets)


def _build_thesis_summary(chapters: list[dict[str, Any]]) -> str:
    lines = []
    for chapter in chapters:
        title = chapter.get("chapter_title", "")
        text = chapter.get("merged_text", "")
        excerpt = " ".join(text.split()[:80])
        lines.append(f"[{title}] {excerpt}")
    return "\n\n".join(lines)


def _fallback_outline(topic: str) -> list[dict[str, Any]]:
    return [
        {
            "chapter_id": "ch_01",
            "chapter_title": "绪论",
            "target_words": 3000,
            "complexity": "low",
            "subsections": [{"title": "研究背景"}, {"title": "研究问题与目标"}],
        },
        {
            "chapter_id": "ch_02",
            "chapter_title": "文献综述",
            "target_words": 7000,
            "complexity": "high",
            "subsections": [{"title": "核心概念"}, {"title": "国内外研究进展"}, {"title": "研究缺口"}],
        },
        {
            "chapter_id": "ch_03",
            "chapter_title": "研究方法",
            "target_words": 4000,
            "complexity": "medium",
            "subsections": [{"title": "研究设计"}, {"title": "数据与方法"}],
        },
        {
            "chapter_id": "ch_04",
            "chapter_title": "结果与分析",
            "target_words": 6000,
            "complexity": "medium",
            "subsections": [{"title": "主要发现"}, {"title": "分析与讨论"}],
        },
        {
            "chapter_id": "ch_05",
            "chapter_title": "结论与展望",
            "target_words": 3000,
            "complexity": "low",
            "subsections": [{"title": "研究结论"}, {"title": "局限与未来工作"}],
        },
    ]


def _pandoc_export(md_path: str, output_path: str, fmt: str) -> None:
    """Convert the markdown thesis to *fmt* using pandoc.

    fmt values:
      "latex"  → thesis.tex  (ctexart, CJK fonts, math-aware)
      "pdf"    → thesis.pdf  (xelatex) + thesis.tex alongside it
      "docx"   → thesis.docx

    Raises RuntimeError with an actionable message when pandoc / xelatex is
    not found, or when pandoc exits with a non-zero status.
    """
    import shutil
    import subprocess

    if not shutil.which("pandoc"):
        raise RuntimeError(
            "pandoc is not installed. Install with: sudo apt-get install pandoc"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Shared args for CJK academic output (latex / pdf)
    _cjk = [
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
                f"pandoc failed (exit {result.returncode}):\n"
                + (result.stderr or result.stdout)[-2000:]
            )

    if fmt == "latex":
        _run(["pandoc", md_path] + _cjk + ["-o", output_path])

    elif fmt == "pdf":
        if not shutil.which("xelatex"):
            raise RuntimeError(
                "xelatex is not installed. Install with: "
                "sudo apt-get install texlive-xetex texlive-lang-chinese"
            )
        # Emit .tex alongside the PDF so the source is available
        tex_path = os.path.splitext(output_path)[0] + ".tex"
        _run(["pandoc", md_path] + _cjk + ["-o", tex_path])
        _run(["pandoc", md_path] + _cjk + ["--pdf-engine=xelatex", "-o", output_path])

    elif fmt == "docx":
        _run([
            "pandoc", md_path,
            "--from", "markdown+tex_math_single_backslash",
            "--toc", "--toc-depth=3", "--number-sections",
            "-o", output_path,
        ])

    else:
        raise ValueError(f"_pandoc_export: unsupported fmt {fmt!r}")
