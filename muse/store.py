"""Persistent run storage for checkpoints, feedback, and outputs."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunStore:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        return self.base_dir / run_id

    def create_run(self, topic: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        run_id = f"run_{ts}_{suffix}"
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=False)

        metadata = {
            "run_id": run_id,
            "topic": topic,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_json(run_dir / "metadata.json", metadata)
        self._write_json(run_dir / "hitl_feedback.json", [])

        return run_id

    def save_state(self, run_id: str, state: dict[str, Any]) -> None:
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(run_dir / "state.json", state)

    def load_state(self, run_id: str) -> dict[str, Any]:
        return self._read_json(self._run_dir(run_id) / "state.json")

    def append_hitl_feedback(self, run_id: str, feedback: dict[str, Any]) -> None:
        path = self._run_dir(run_id) / "hitl_feedback.json"
        data = self._read_json(path) if path.exists() else []
        if not isinstance(data, list):
            data = []
        data.append(feedback)
        self._write_json(path, data)

    def load_hitl_feedback(self, run_id: str) -> list[dict[str, Any]]:
        path = self._run_dir(run_id) / "hitl_feedback.json"
        data = self._read_json(path) if path.exists() else []
        return data if isinstance(data, list) else []

    def artifact_path(self, run_id: str, name: str) -> str:
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / name
        os.makedirs(path.parent, exist_ok=True)
        return str(path)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _read_json(path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
