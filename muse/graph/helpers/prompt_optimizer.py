"""Persistent prompt bank for exploratory prompt optimization."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4
from typing import Any

from muse.prompts.optimizer_meta import optimizer_meta_prompt


def _average_score(scores: dict[str, Any]) -> float:
    numeric = [
        float(value)
        for value in scores.values()
        if isinstance(value, (int, float))
    ]
    if not numeric:
        return 0.0
    return sum(numeric) / len(numeric)


class PromptOptimizer:
    def __init__(self, bank_dir: Path):
        self.bank_dir = Path(bank_dir)
        self.bank_dir.mkdir(parents=True, exist_ok=True)

    def _bank_path(self, prompt_name: str) -> Path:
        return self.bank_dir / f"{prompt_name}.json"

    def _load_bank(self, prompt_name: str, baseline_prompt: str) -> dict[str, Any]:
        path = self._bank_path(prompt_name)
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                bank = json.load(handle)
        else:
            bank = {
                "prompt_name": prompt_name,
                "baseline": {
                    "id": "baseline",
                    "prompt": baseline_prompt,
                    "runs": 0,
                    "avg_score": 0.0,
                    "last_scores": {},
                },
                "variants": [],
            }

        if not isinstance(bank.get("baseline"), dict):
            bank["baseline"] = {
                "id": "baseline",
                "prompt": baseline_prompt,
                "runs": 0,
                "avg_score": 0.0,
                "last_scores": {},
            }
        bank["baseline"]["prompt"] = baseline_prompt
        if not isinstance(bank.get("variants"), list):
            bank["variants"] = []
        return bank

    def _save_bank(self, prompt_name: str, bank: dict[str, Any]) -> None:
        path = self._bank_path(prompt_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(bank, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def analyze_weakness(scores: dict[str, Any]) -> list[str]:
        ranked = sorted(
            (
                (str(key), float(value))
                for key, value in scores.items()
                if isinstance(value, (int, float))
            ),
            key=lambda item: item[1],
        )
        return [key for key, value in ranked if value <= 3.0]

    def select_prompt(self, prompt_name: str, baseline_prompt: str) -> str:
        bank = self._load_bank(prompt_name, baseline_prompt)
        pending = [
            variant
            for variant in bank["variants"]
            if isinstance(variant, dict) and variant.get("status") == "trial_pending"
        ]
        if pending:
            return str(pending[0].get("prompt", baseline_prompt))

        baseline_avg = float(bank["baseline"].get("avg_score", 0.0) or 0.0)
        best_prompt = str(bank["baseline"].get("prompt", baseline_prompt))
        best_score = baseline_avg
        for variant in bank["variants"]:
            if not isinstance(variant, dict):
                continue
            runs = int(variant.get("runs", 0) or 0)
            avg_score = float(variant.get("avg_score", 0.0) or 0.0)
            if runs > 0 and avg_score >= baseline_avg and avg_score >= best_score:
                best_prompt = str(variant.get("prompt", best_prompt))
                best_score = avg_score
        return best_prompt

    def record_result(
        self,
        prompt_name: str,
        prompt_text: str,
        scores: dict[str, Any],
        *,
        run_id: str,
        baseline_prompt: str | None = None,
    ) -> str:
        baseline_prompt = baseline_prompt or prompt_text
        bank = self._load_bank(prompt_name, baseline_prompt)
        entry: dict[str, Any] | None = None

        if prompt_text == bank["baseline"].get("prompt"):
            entry = bank["baseline"]
        else:
            for variant in bank["variants"]:
                if isinstance(variant, dict) and str(variant.get("prompt", "")) == prompt_text:
                    entry = variant
                    break
        if entry is None:
            entry = {
                "id": f"variant-{uuid4().hex[:8]}",
                "prompt": prompt_text,
                "weaknesses": [],
                "runs": 0,
                "avg_score": 0.0,
                "last_scores": {},
                "status": "validated",
            }
            bank["variants"].append(entry)

        previous_runs = int(entry.get("runs", 0) or 0)
        previous_avg = float(entry.get("avg_score", 0.0) or 0.0)
        current_avg = _average_score(scores)
        entry["runs"] = previous_runs + 1
        entry["avg_score"] = ((previous_avg * previous_runs) + current_avg) / entry["runs"]
        entry["last_scores"] = scores
        entry["last_run_id"] = run_id
        if entry is not bank["baseline"]:
            entry["status"] = "validated"

        self._save_bank(prompt_name, bank)
        return str(entry.get("id", "baseline"))

    def add_candidate(
        self,
        prompt_name: str,
        prompt_text: str,
        weaknesses: list[str],
        *,
        source_prompt_id: str,
        source_run_id: str,
        baseline_prompt: str | None = None,
    ) -> str:
        baseline_prompt = baseline_prompt or prompt_text
        bank = self._load_bank(prompt_name, baseline_prompt)
        for variant in bank["variants"]:
            if isinstance(variant, dict) and str(variant.get("prompt", "")) == prompt_text:
                return str(variant.get("id", ""))

        variant_id = f"variant-{uuid4().hex[:8]}"
        bank["variants"].append(
            {
                "id": variant_id,
                "prompt": prompt_text,
                "weaknesses": list(weaknesses),
                "runs": 0,
                "avg_score": 0.0,
                "last_scores": {},
                "status": "trial_pending",
                "source_prompt_id": source_prompt_id,
                "source_run_id": source_run_id,
            }
        )
        self._save_bank(prompt_name, bank)
        return variant_id

    def generate_improvement(
        self,
        prompt_name: str,
        current_prompt: str,
        weaknesses: list[str],
        llm: Any,
    ) -> str | None:
        if llm is None or not hasattr(llm, "structured") or not weaknesses:
            return None
        system, user = optimizer_meta_prompt(prompt_name, current_prompt, weaknesses)
        payload = llm.structured(
            system=system,
            user=user,
            route="default",
            max_tokens=1200,
        )
        improved_prompt = str(payload.get("improved_prompt", "")).strip() if isinstance(payload, dict) else ""
        return improved_prompt or None
