"""Persistent reflection helpers derived from review history."""

from __future__ import annotations

from typing import Any


class ReflectionBank:
    def __init__(self, entries: list[dict[str, Any]] | None = None) -> None:
        self.entries: list[dict[str, Any]] = [
            dict(entry)
            for entry in (entries or [])
            if isinstance(entry, dict)
        ]

    def add_reflection(self, review_history: list[dict[str, Any]], chapter_id: str) -> None:
        if not isinstance(review_history, list):
            return

        seen = {
            self._entry_key(entry)
            for entry in self.entries
        }
        for previous, current in zip(review_history, review_history[1:]):
            if not isinstance(previous, dict) or not isinstance(current, dict):
                continue
            previous_scores = previous.get("scores", {})
            current_scores = current.get("scores", {})
            if not isinstance(previous_scores, dict) or not isinstance(current_scores, dict):
                continue

            instructions = [
                str(item).strip()
                for item in previous.get("top_instructions", [])
                if str(item).strip()
            ]
            instruction = instructions[0] if instructions else str(previous.get("notes_summary", "")).strip()
            if not instruction:
                continue

            for dimension, old_value in previous_scores.items():
                new_value = current_scores.get(dimension)
                if not isinstance(old_value, (int, float)) or not isinstance(new_value, (int, float)):
                    continue
                score_delta = int(new_value) - int(old_value)
                if score_delta == 0:
                    continue

                outcome = "positive" if score_delta > 0 else "regression"
                entry = {
                    "chapter_id": str(chapter_id).strip() or "thesis",
                    "from_iteration": int(previous.get("iteration", 0) or 0),
                    "to_iteration": int(current.get("iteration", 0) or 0),
                    "dimension": str(dimension).strip(),
                    "outcome": outcome,
                    "instruction": instruction,
                    "score_delta": score_delta,
                }
                entry_key = self._entry_key(entry)
                if entry_key in seen:
                    continue
                seen.add(entry_key)
                self.entries.append(entry)

    def get_relevant_reflections(
        self,
        current_scores: dict[str, Any] | None,
        max_reflections: int = 5,
    ) -> list[dict[str, Any]]:
        weak_dimensions = {
            str(key).strip()
            for key, value in (current_scores or {}).items()
            if isinstance(value, (int, float)) and int(value) <= 3
        }

        candidates = [
            dict(entry)
            for entry in self.entries
            if isinstance(entry, dict) and entry.get("outcome") == "positive"
        ]
        if weak_dimensions:
            candidates = [
                entry
                for entry in candidates
                if str(entry.get("dimension", "")).strip() in weak_dimensions
            ]
        candidates.sort(
            key=lambda entry: (
                -int(entry.get("score_delta", 0)),
                str(entry.get("dimension", "")),
                str(entry.get("instruction", "")),
            )
        )
        return candidates[:max_reflections]

    def get_writing_tips(self, max_tips: int = 3) -> list[str]:
        tips: list[str] = []
        seen: set[str] = set()
        candidates = self.get_relevant_reflections({}, max_reflections=max_tips * 3)
        for entry in candidates:
            dimension = str(entry.get("dimension", "")).strip() or "quality"
            instruction = str(entry.get("instruction", "")).strip()
            if not instruction:
                continue
            tip = f"For {dimension}: {instruction}"
            if tip in seen:
                continue
            seen.add(tip)
            tips.append(tip)
            if len(tips) >= max_tips:
                break
        return tips

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [dict(entry) for entry in self.entries]}

    @classmethod
    def from_dict(cls, data: Any) -> "ReflectionBank":
        if not isinstance(data, dict):
            return cls()
        entries = data.get("entries", [])
        if not isinstance(entries, list):
            entries = []
        return cls(entries=[entry for entry in entries if isinstance(entry, dict)])

    @staticmethod
    def _entry_key(entry: dict[str, Any]) -> tuple[str, int, int, str, str, str]:
        return (
            str(entry.get("chapter_id", "")).strip(),
            int(entry.get("from_iteration", 0) or 0),
            int(entry.get("to_iteration", 0) or 0),
            str(entry.get("dimension", "")).strip(),
            str(entry.get("outcome", "")).strip(),
            str(entry.get("instruction", "")).strip(),
        )
