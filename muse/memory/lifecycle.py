"""Memory confidence lifecycle management."""

from __future__ import annotations

import logging

from muse.memory.store import MemoryStore

logger = logging.getLogger(__name__)


def confirm_memory(store: MemoryStore, key: str, *, delta: float = 0.1) -> bool:
    """Increment confidence for a memory identified by key."""

    for entry in store.query(min_confidence=0.0, limit=1000):
        if entry.key == key:
            store.update_confidence(entry.id, delta)
            logger.debug("Confirmed memory '%s': confidence += %.2f", key, delta)
            return True
    return False


def deny_memory(store: MemoryStore, key: str) -> bool:
    """Delete a memory identified by key."""

    for entry in store.query(min_confidence=0.0, limit=1000):
        if entry.key == key:
            store.delete(entry.id)
            logger.info("Denied and deleted memory '%s'", key)
            return True
    return False


def run_decay(
    store: MemoryStore,
    *,
    days: int = 90,
    factor: float = 0.9,
) -> int:
    """Decay stale memories that have not been updated recently."""

    count = store.decay_old_memories(days=days, factor=factor)
    if count:
        logger.info("Decayed %d memories (>%d days old, factor=%.2f)", count, days, factor)
    return count


def cleanup_dead_memories(
    store: MemoryStore,
    *,
    threshold: float = 0.01,
) -> int:
    """Delete memories whose confidence has effectively reached zero."""

    deleted = 0
    for entry in store.query(min_confidence=0.0, limit=1000):
        if entry.confidence <= threshold:
            store.delete(entry.id)
            deleted += 1
    if deleted:
        logger.info("Cleaned up %d dead memories (confidence <= %.3f)", deleted, threshold)
    return deleted


def run_maintenance(store: MemoryStore) -> dict[str, int]:
    """Run the decay + cleanup maintenance cycle."""

    decayed = run_decay(store)
    cleaned = cleanup_dead_memories(store)
    return {"decayed": decayed, "cleaned": cleaned}
