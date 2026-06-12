"""Redis exact query cache."""

import hashlib
import json
import logging
from typing import Any, Optional

import redis

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class RedisCache:
    """Exact query cache: normalized query -> answer + citations."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._client = redis.Redis(
            host=self.settings.redis_host,
            port=self.settings.redis_port,
            db=self.settings.redis_db,
            decode_responses=True,
        )

    def _cache_key(self, query: str) -> str:
        query_hash = hashlib.sha256(query.strip().lower().encode()).hexdigest()[:32]
        return f"cache:query:{query_hash}"

    def get(self, query: str) -> Optional[dict[str, Any]]:
        key = self._cache_key(query)
        hit = self._client.get(key)
        if hit:
            logger.info("Cache hit for query")
            return json.loads(hit)
        return None

    def set(
        self,
        query: str,
        answer: str,
        citations: list[dict[str, Any]],
        chunk_ids: list[str],
    ) -> None:
        entry = {
            "query": query,
            "answer": answer,
            "citations": citations,
            "chunk_ids": chunk_ids,
        }
        key = self._cache_key(query)
        self._client.setex(key, 86400 * 7, json.dumps(entry))
        logger.info("Cached response for query")

    def flush_all(self) -> None:
        deleted = 0
        for key in self._client.scan_iter(match="cache:query:*", count=100):
            self._client.delete(key)
            deleted += 1
        logger.info("Flushed %d cache entries", deleted)
