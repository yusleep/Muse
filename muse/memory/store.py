"""SQLite-backed memory store for persistent cross-session memory."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

CATEGORIES = frozenset(
    {
        "user_pref",
        "writing_style",
        "citation",
        "feedback_pattern",
        "fact",
    }
)


@dataclass
class MemoryEntry:
    """A single memory record."""

    id: str
    key: str
    category: str
    content: str
    confidence: float
    source_run: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(
                f"Invalid category '{self.category}'. Must be one of: {', '.join(sorted(CATEGORIES))}"
            )
        self.confidence = max(0.0, min(1.0, self.confidence))


class MemoryStore:
    """SQLite-backed persistent memory store."""

    _SCHEMA = """\
    CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        key TEXT NOT NULL,
        category TEXT NOT NULL,
        content TEXT NOT NULL,
        confidence REAL NOT NULL DEFAULT 0.5,
        source_run TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
    CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key);
    CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence);
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_dir = Path.home() / ".muse"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "memory.sqlite"
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cursor = self._conn.cursor()
        cursor.executescript(self._SCHEMA)
        self._conn.commit()

    def upsert(self, entry: MemoryEntry) -> None:
        """Insert or update a memory entry by dedup key."""

        now = datetime.now(timezone.utc).isoformat()
        existing = self._find_by_key(entry.key)
        if existing is not None:
            self._conn.execute(
                """\
                UPDATE memories
                SET content = ?, confidence = ?, category = ?, source_run = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    entry.content,
                    entry.confidence,
                    entry.category,
                    entry.source_run,
                    now,
                    existing.id,
                ),
            )
        else:
            entry_id = entry.id or uuid.uuid4().hex
            self._conn.execute(
                """\
                INSERT INTO memories (
                    id, key, category, content, confidence, source_run, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    entry.key,
                    entry.category,
                    entry.content,
                    entry.confidence,
                    entry.source_run,
                    now,
                    now,
                ),
            )
        self._conn.commit()

    def query(
        self,
        *,
        category: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 50,
    ) -> list[MemoryEntry]:
        """Query memories by category and confidence threshold."""

        conditions = ["confidence >= ?"]
        params: list[Any] = [min_confidence]
        if category is not None:
            conditions.append("category = ?")
            params.append(category)
        where_clause = " AND ".join(conditions)
        params.append(limit)

        rows = self._conn.execute(
            f"""\
            SELECT id, key, category, content, confidence, source_run, created_at, updated_at
            FROM memories
            WHERE {where_clause}
            ORDER BY confidence DESC, updated_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def get(self, memory_id: str) -> MemoryEntry | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def delete(self, memory_id: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE id = ?",
            (memory_id,),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def update_confidence(self, memory_id: str, delta: float) -> None:
        entry = self.get(memory_id)
        if entry is None:
            return
        new_confidence = max(0.0, min(1.0, entry.confidence + delta))
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE memories SET confidence = ?, updated_at = ? WHERE id = ?",
            (new_confidence, now, memory_id),
        )
        self._conn.commit()

    def set_confidence(self, memory_id: str, value: float) -> None:
        now = datetime.now(timezone.utc).isoformat()
        confidence = max(0.0, min(1.0, value))
        self._conn.execute(
            "UPDATE memories SET confidence = ?, updated_at = ? WHERE id = ?",
            (confidence, now, memory_id),
        )
        self._conn.commit()

    def decay_old_memories(self, days: int = 90, factor: float = 0.9) -> int:
        threshold = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cursor = self._conn.execute(
            """\
            UPDATE memories
            SET confidence = confidence * ?, updated_at = updated_at
            WHERE updated_at < ? AND confidence > 0.01
            """,
            (factor, threshold),
        )
        self._conn.commit()
        return cursor.rowcount

    def count(self, *, category: str | None = None) -> int:
        if category is not None:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE category = ?",
                (category,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self._conn.close()

    def _find_by_key(self, key: str) -> MemoryEntry | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE key = ?",
            (key,),
        ).fetchone()
        return self._row_to_entry(row) if row else None

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            key=row["key"],
            category=row["category"],
            content=row["content"],
            confidence=row["confidence"],
            source_run=row["source_run"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
