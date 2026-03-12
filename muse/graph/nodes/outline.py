"""Outline node for graph-native planning."""

from __future__ import annotations

from typing import Any

from muse.prompts.outline_gen import outline_gen_prompt
from muse.prompts.topic_analysis import topic_analysis_prompt
from muse.services.planning import plan_subtasks


def _analyze_topic(
    llm_client: Any,
    topic: str,
    discipline: str,
    literature_summary: str,
) -> dict[str, Any]:
    analysis = {
        "research_gaps": [],
        "core_concepts": [],
        "methodology_domain": "general",
        "suggested_contributions": [],
    }
    system, user = topic_analysis_prompt(topic, discipline, literature_summary)
    try:
        payload = llm_client.structured(system=system, user=user, route="outline", max_tokens=800)
        if isinstance(payload, dict) and "research_gaps" in payload:
            analysis.update(payload)
    except Exception:
        pass
    return analysis


def _fallback_outline(topic: str) -> list[dict[str, Any]]:
    return [
        {
            "chapter_id": "ch_01",
            "chapter_title": "绪论",
            "target_words": 3000,
            "complexity": "medium",
            "subsections": [{"title": f"{topic} 研究背景"}],
        }
    ]


def build_outline_node(settings: Any, services: Any):
    def outline(state: dict[str, Any]) -> dict[str, Any]:
        llm = getattr(services, "llm", None)
        topic = state.get("topic", "")
        discipline = state.get("discipline", "")
        language = state.get("language", "zh")
        literature_summary = state.get("literature_summary", "")

        analysis = {
            "research_gaps": [],
            "core_concepts": [],
            "methodology_domain": "general",
            "suggested_contributions": [],
        }
        if llm is not None:
            analysis = _analyze_topic(llm, topic, discipline, literature_summary)

        chapters = _fallback_outline(topic)
        if llm is not None:
            system, user = outline_gen_prompt(topic, discipline, language, literature_summary, analysis)
            try:
                payload = llm.structured(system=system, user=user, route="outline", max_tokens=3000)
                generated = payload.get("chapters", []) if isinstance(payload, dict) else []
                if generated:
                    chapters = generated
            except Exception:
                pass

        chapter_plans = []
        for index, chapter in enumerate(chapters, start=1):
            target_words = int(chapter.get("target_words", 3000))
            complexity = str(chapter.get("complexity", "medium"))
            subtasks = plan_subtasks(target_words=target_words, complexity=complexity, subsections=chapter.get("subsections", []))
            chapter_plans.append(
                {
                    "chapter_id": chapter.get("chapter_id", f"ch_{index:02d}"),
                    "chapter_title": chapter.get("chapter_title", f"Chapter {index}"),
                    "target_words": target_words,
                    "complexity": complexity,
                    "subtask_plan": subtasks,
                }
            )

        return {
            "outline": {"chapters": chapters, "topic_analysis": analysis},
            "chapter_plans": chapter_plans,
        }

    return outline
