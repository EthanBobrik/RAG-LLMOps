"""Pluggable chat-history store.

Default is process-local (InMemorySessionStore). If REDIS_URL is set, history is kept
in Redis so multiple uvicorn workers share session state — making the app horizontally
scalable. (The FAISS index is already on disk; each worker rebuilds its own RAG from it,
so only the serializable chat history needs sharing.)
"""

from __future__ import annotations
import os
import json
from typing import Dict, List


class InMemorySessionStore:
    """Process-local store (single-worker default)."""

    def __init__(self) -> None:
        self._d: Dict[str, List[dict]] = {}

    def exists(self, session_id: str) -> bool:
        return session_id in self._d

    def create(self, session_id: str) -> None:
        self._d.setdefault(session_id, [])

    def history(self, session_id: str) -> List[dict]:
        return list(self._d.get(session_id, []))

    def append(self, session_id: str, role: str, content: str) -> None:
        self._d.setdefault(session_id, []).append({"role": role, "content": content})

    def clear(self) -> None:
        self._d.clear()


class RedisSessionStore:
    """Redis-backed store shared across workers.

    Existence is tracked in a SET; per-session history in a LIST. RPUSH is atomic, so
    concurrent appends don't race (unlike a read-modify-write on a JSON blob).
    """

    _SET = "mdc:sessions"
    _PREFIX = "mdc:session:"

    def __init__(self, url: str) -> None:
        import redis  # lazy: only needed when REDIS_URL is configured
        self._r = redis.Redis.from_url(url, decode_responses=True)

    def _key(self, session_id: str) -> str:
        return self._PREFIX + session_id

    def exists(self, session_id: str) -> bool:
        return bool(self._r.sismember(self._SET, session_id))

    def create(self, session_id: str) -> None:
        self._r.sadd(self._SET, session_id)

    def history(self, session_id: str) -> List[dict]:
        return [json.loads(x) for x in self._r.lrange(self._key(session_id), 0, -1)]

    def append(self, session_id: str, role: str, content: str) -> None:
        self._r.sadd(self._SET, session_id)
        self._r.rpush(self._key(session_id), json.dumps({"role": role, "content": content}))

    def clear(self) -> None:
        for sid in self._r.smembers(self._SET):
            self._r.delete(self._key(sid))
        self._r.delete(self._SET)


def get_session_store():
    """Return the Redis store if REDIS_URL is set, else the in-memory store."""
    url = os.getenv("REDIS_URL")
    return RedisSessionStore(url) if url else InMemorySessionStore()
