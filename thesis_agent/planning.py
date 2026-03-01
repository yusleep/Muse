"""Planning utilities for chapter subtask allocation."""

from __future__ import annotations

from typing import Any


def _desired_words_per_task(complexity: str) -> int:
    mapping = {
        "low": 1500,
        "medium": 1500,
        "high": 1200,
    }
    return mapping.get(complexity, 1500)


def _normalize_subsections(subsections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not subsections:
        return [{"title": "内容主体", "relevant_refs": [], "instructions": "围绕章节目标写作。"}]

    normalized = []
    for item in subsections:
        normalized.append(
            {
                "title": item.get("title", "未命名小节"),
                "relevant_refs": list(item.get("relevant_refs", [])),
                "instructions": item.get("instructions", "围绕小节标题展开论证，保持学术风格。"),
            }
        )
    return normalized


def _calc_subtask_count(target_words: int, complexity: str) -> int:
    words_per_task = _desired_words_per_task(complexity)
    base_count = max(2, round(target_words / words_per_task))

    while target_words // base_count > 2000 and base_count < 12:
        base_count += 1
    while target_words // base_count < 1000 and base_count > 2:
        base_count -= 1

    return max(2, base_count)


def _distribute_subsections(
    subsections: list[dict[str, Any]],
    subtask_count: int,
    target_words: int,
) -> list[dict[str, Any]]:
    buckets = [[] for _ in range(subtask_count)]
    for idx, subsection in enumerate(subsections):
        buckets[idx % subtask_count].append(subsection)

    base_words = target_words // subtask_count
    remainder = target_words % subtask_count

    result = []
    for i, bucket in enumerate(buckets):
        words = base_words + (1 if i < remainder else 0)
        words = max(1000, min(2000, words))

        if bucket:
            title = " / ".join(item["title"] for item in bucket)
            refs = sorted({ref for item in bucket for ref in item["relevant_refs"]})
            instructions = "\n".join(item["instructions"] for item in bucket)
        else:
            title = f"补充分节 {i + 1}"
            refs = []
            instructions = "补充论证与过渡内容，避免重复。"

        result.append(
            {
                "subtask_id": f"sub_{i + 1:02d}",
                "title": title,
                "target_words": words,
                "relevant_refs": refs,
                "instructions": instructions,
                "predecessor_hint": "",
                "successor_hint": "",
            }
        )

    return result


def plan_subtasks(target_words: int, complexity: str, subsections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute subtask plan with bounded word budget per subtask."""

    if target_words <= 0:
        raise ValueError("target_words must be positive")

    normalized = _normalize_subsections(subsections)
    subtask_count = _calc_subtask_count(target_words, complexity)
    subtasks = _distribute_subsections(normalized, subtask_count, target_words)

    # Populate adjacency hints for smoother transitions.
    for idx, task in enumerate(subtasks):
        if idx > 0:
            task["predecessor_hint"] = subtasks[idx - 1]["title"]
        if idx < len(subtasks) - 1:
            task["successor_hint"] = subtasks[idx + 1]["title"]

    return subtasks
