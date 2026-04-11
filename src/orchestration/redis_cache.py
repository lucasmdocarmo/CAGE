"""Redis cache helpers.

We use Redis as an optional centralized cache baseline.

Important:
- This repo does NOT store raw vLLM KV-cache blocks in Redis.
- Redis is used to cache *retrieval artifacts* (e.g., query -> retrieved doc ids)
  or other metadata to simulate a centralized cache server baseline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


@dataclass
class RedisConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    key_prefix: str = "cage"


class RedisClient:
    """Small wrapper around redis-py with JSON helpers."""

    def __init__(self, cfg: RedisConfig):
        try:
            import redis
        except ImportError as e:
            raise ImportError("redis package not installed. Install redis>=5.0.1") from e

        self.cfg = cfg
        self._redis = redis.Redis(
            host=cfg.host,
            port=cfg.port,
            db=cfg.db,
            decode_responses=True,
        )

    def ping(self) -> bool:
        try:
            return bool(self._redis.ping())
        except Exception:
            return False

    def _key(self, namespace: str, key: str) -> str:
        return f"{self.cfg.key_prefix}:{namespace}:{key}"

    def get_json(self, namespace: str, key: str) -> Optional[Any]:
        raw = self._redis.get(self._key(namespace, key))
        if raw is None:
            return None
        return json.loads(raw)

    def set_json(self, namespace: str, key: str, value: Any, *, ttl_seconds: Optional[int] = None) -> None:
        k = self._key(namespace, key)
        payload = json.dumps(value, ensure_ascii=False)
        if ttl_seconds:
            self._redis.setex(k, ttl_seconds, payload)
        else:
            self._redis.set(k, payload)

    def delete_namespace(self, namespace: str) -> int:
        pattern = self._key(namespace, "*")
        deleted = 0
        for key in self._redis.scan_iter(match=pattern):
            deleted += int(self._redis.delete(key))
        return deleted


class RetrievalCache:
    """Cache for retrieval hits keyed by (dataset, embedding_model, top_k, query)."""

    NAMESPACE = "retrieval"

    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client

    def make_key(self, *, dataset: str, embedding_model: str, top_k: int, query: str) -> str:
        model = embedding_model.replace("/", "_")
        qh = _sha1(query)
        return f"{dataset}:{model}:{top_k}:{qh}"

    def get(self, *, dataset: str, embedding_model: str, top_k: int, query: str) -> Optional[List[Dict[str, Any]]]:
        key = self.make_key(dataset=dataset, embedding_model=embedding_model, top_k=top_k, query=query)
        return self.redis.get_json(self.NAMESPACE, key)

    def set(
        self,
        *,
        dataset: str,
        embedding_model: str,
        top_k: int,
        query: str,
        hits: List[Dict[str, Any]],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        key = self.make_key(dataset=dataset, embedding_model=embedding_model, top_k=top_k, query=query)
        self.redis.set_json(self.NAMESPACE, key, hits, ttl_seconds=ttl_seconds)

    def clear(self) -> int:
        return self.redis.delete_namespace(self.NAMESPACE)
