"""Redis cache helpers.

We use Redis as an optional centralized cache baseline.

Important:
- This repo does NOT store raw vLLM KV-cache blocks in Redis.
- Redis is used to cache *retrieval artifacts* (e.g., query -> retrieved doc ids)
  or other metadata to simulate a centralized cache server baseline.

Backlog F5b (2026-09-17): the retrieval-cache key folds the corpus
fingerprint of the index that produced the hits (``corpus_doc_ids_sha1``,
the same hash ``ensure_ir_index`` checks). A rebuilt index (different
distractor pool, re-drawn manifest, re-chunked corpus) therefore never serves
hits cached from a previous build, and a cache that cannot learn the corpus
hash refuses (``RetrievalCacheKeyError``) instead of composing a hash-free
key. Backlog F5a's per-cell namespaces (``RedisConfig.key_prefix``, minted by
the campaign driver as ``cage:<sha1(row_key)[:12]>``) isolate CELLS; the
corpus segment isolates INDEX BUILDS within a cell.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.orchestration.ir import corpus_doc_ids_sha1


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


class RetrievalCacheKeyError(ValueError):
    """Typed refusal: a retrieval-cache key cannot fold the corpus hash (backlog F5b).

    The key must carry the fingerprint of the corpus the index was built over;
    without it a rebuilt index could serve stale hits under the same key and
    nothing would label the row. Raised when the index exposes no corpus
    (no ``documents``) or when the hash handed to ``make_key`` is not a
    non-empty string.
    """


def corpus_sha1_of_index(ir_index: Any) -> str:
    """The corpus fingerprint of a loaded IR index (F5b key ingredient).

    Reads the index's ``documents`` and hashes their ids exactly as
    ``FaissIRIndex.save`` stamps ``doc_ids_sha1`` into meta.json
    (``corpus_doc_ids_sha1``: sha1 over the SORTED doc ids). Refuses (typed)
    when the index exposes no corpus: the cache must never fall back to a
    hash-free key.
    """
    documents: Optional[Sequence[Any]] = getattr(ir_index, "documents", None)
    if documents is None:
        raise RetrievalCacheKeyError(
            f"IR index {type(ir_index).__name__} exposes no corpus (no 'documents'): "
            "the retrieval-cache key must fold the corpus hash (backlog F5b) and "
            "there is nothing to hash; refusing to serve cached hits without it"
        )
    try:
        docs = list(documents)
    except TypeError as exc:
        raise RetrievalCacheKeyError(
            f"IR index {type(ir_index).__name__}.documents is not iterable "
            f"({type(documents).__name__}): cannot derive the corpus hash (backlog F5b)"
        ) from exc
    return corpus_doc_ids_sha1(docs)


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
    """Cache for retrieval hits keyed by (dataset, embedding_model, top_k, [pool], corpus, query).

    ADR-0104 (rerank pool): when the runner retrieves a candidate POOL of
    ``pool`` hits to rerank before serving ``top_k``, the key also carries
    the pool size (a pool-10 hit list is not a top-3 hit list) and the cached
    payload is the pool itself, in dense order, BEFORE the rerank. ``pool``
    None omits the pool segment.

    Backlog F5b: every key folds ``corpus_sha1``, the fingerprint of the
    corpus the index was built over (``corpus_sha1_of_index``), so a rebuilt
    index never serves hits cached from a previous build. The hash is a
    REQUIRED keyword: there is no hash-free key shape.
    """

    NAMESPACE = "retrieval"

    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client

    @staticmethod
    def _checked_corpus_sha1(corpus_sha1: Any) -> str:
        if not isinstance(corpus_sha1, str) or not corpus_sha1.strip():
            raise RetrievalCacheKeyError(
                f"corpus_sha1={corpus_sha1!r} must be a non-empty string: the "
                "retrieval-cache key folds the index's corpus hash (backlog F5b) "
                "and refuses to compose a key without one"
            )
        return corpus_sha1.strip()

    def make_key(
        self,
        *,
        dataset: str,
        embedding_model: str,
        top_k: int,
        query: str,
        corpus_sha1: str,
        pool: Optional[int] = None,
    ) -> str:
        model = embedding_model.replace("/", "_")
        corpus = self._checked_corpus_sha1(corpus_sha1)
        qh = _sha1(query)
        if pool is None:
            return f"{dataset}:{model}:{top_k}:corpus{corpus}:{qh}"
        return f"{dataset}:{model}:{top_k}:pool{int(pool)}:corpus{corpus}:{qh}"

    def get(
        self,
        *,
        dataset: str,
        embedding_model: str,
        top_k: int,
        query: str,
        corpus_sha1: str,
        pool: Optional[int] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        key = self.make_key(
            dataset=dataset,
            embedding_model=embedding_model,
            top_k=top_k,
            query=query,
            corpus_sha1=corpus_sha1,
            pool=pool,
        )
        return self.redis.get_json(self.NAMESPACE, key)

    def set(
        self,
        *,
        dataset: str,
        embedding_model: str,
        top_k: int,
        query: str,
        hits: List[Dict[str, Any]],
        corpus_sha1: str,
        ttl_seconds: Optional[int] = None,
        pool: Optional[int] = None,
    ) -> None:
        key = self.make_key(
            dataset=dataset,
            embedding_model=embedding_model,
            top_k=top_k,
            query=query,
            corpus_sha1=corpus_sha1,
            pool=pool,
        )
        self.redis.set_json(self.NAMESPACE, key, hits, ttl_seconds=ttl_seconds)

    def clear(self) -> int:
        return self.redis.delete_namespace(self.NAMESPACE)
