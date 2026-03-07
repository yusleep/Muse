"""Runtime engine for executing staged thesis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .schemas import hydrate_thesis_state
from .store import RunStore

StageFn = Callable[["EngineContext"], str]


@dataclass
class EngineContext:
    run_id: str
    state: dict
    store: RunStore


class ThesisEngine:
    """Executes stage functions in ascending stage order."""

    def __init__(self, store: RunStore, stages: dict[int, StageFn]) -> None:
        self.store = store
        self.stages = dict(sorted(stages.items(), key=lambda item: item[0]))

    def run(self, run_id: str, start_stage: int, auto_approve: bool = False) -> dict:
        state = hydrate_thesis_state(self.store.load_state(run_id))
        stage_numbers = [s for s in self.stages.keys() if s >= start_stage]
        current_stage = start_stage

        for stage in stage_numbers:
            ctx = EngineContext(run_id=run_id, state=state, store=self.store)
            result = self.stages[stage](ctx)

            self.store.save_state(run_id, state)
            current_stage = stage

            if result == "hitl" and not auto_approve:
                return {
                    "status": "waiting_hitl",
                    "stage": stage,
                    "run_id": run_id,
                }
            if result == "done":
                return {
                    "status": "completed",
                    "stage": stage,
                    "run_id": run_id,
                }
            if result in {"blocked", "failed", "error"}:
                return {
                    "status": result,
                    "stage": stage,
                    "run_id": run_id,
                }

        return {
            "status": "completed",
            "stage": current_stage,
            "run_id": run_id,
        }
