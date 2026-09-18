"""Backlog F5b: the retrieval-cache key folds the index's corpus hash.

src/orchestration/redis_cache.py caches retrieval artifacts (query -> hits)
for the B7 retrieval-reuse arm. Before F5b the key carried the dataset, the
embedding model, top_k, the ADR-0104 pool size and the query hash, but NOT
the corpus the index was built over: a rebuilt index (distractor pool
changed, manifest re-drawn, corpus re-chunked) could serve stale hits from
a previous build under the same key, and nothing would label the row. The
IR index already fingerprints its corpus (``corpus_doc_ids_sha1``, checked
by ``ensure_ir_index``), so the key now folds that fingerprint, and a cache
that cannot learn the corpus hash refuses with a typed error instead of
composing a hash-free key.

Pins:

- ``make_key`` REQUIRES ``corpus_sha1`` and folds it: two corpora, same
  query, different keys; the embedding model, top_k and pool stay folded;
  an empty/non-string hash refuses (typed).
- ``corpus_sha1_of_index`` derives the hash from the index's ``documents``
  (identical to ``corpus_doc_ids_sha1`` over the same documents) and refuses
  (typed) when the index exposes no corpus.
- ``get``/``set`` hand the composed key to the client under the
  ``retrieval`` namespace.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.orchestration.ir import IRDocument, corpus_doc_ids_sha1  # noqa: E402
from src.orchestration.redis_cache import (  # noqa: E402
    RetrievalCache,
    RetrievalCacheKeyError,
    corpus_sha1_of_index,
)

_SHA_A = "a" * 40
_SHA_B = "b" * 40


class _FakeClient:
    """RedisClient stand-in recording (namespace, key) of every call."""

    def __init__(self) -> None:
        self.store: Dict[tuple, Any] = {}
        self.gets: List[tuple] = []
        self.sets: List[tuple] = []

    def get_json(self, namespace: str, key: str) -> Optional[Any]:
        self.gets.append((namespace, key))
        return self.store.get((namespace, key))

    def set_json(self, namespace: str, key: str, value: Any, *, ttl_seconds=None) -> None:
        self.sets.append((namespace, key, ttl_seconds))
        self.store[(namespace, key)] = value


class _IndexWithDocs:
    def __init__(self, docs: List[IRDocument]) -> None:
        self.documents = docs


class _IndexWithoutDocs:
    """An index object exposing no corpus at all."""


def _docs(*ids: str) -> List[IRDocument]:
    return [IRDocument(doc_id=i, text=f"text {i}", metadata={}) for i in ids]


class TestKeyComposition:
    def test_key_folds_the_corpus_hash(self):
        rc = RetrievalCache(redis_client=None)
        common = dict(dataset="squad_v2", embedding_model="intfloat/e5-large-v2", top_k=3, query="q")
        key_a = rc.make_key(corpus_sha1=_SHA_A, **common)
        key_b = rc.make_key(corpus_sha1=_SHA_B, **common)
        assert key_a != key_b
        assert _SHA_A in key_a and _SHA_B in key_b
        # The pre-F5b prefix (dataset:model:top_k:) is preserved.
        assert key_a.startswith("squad_v2:intfloat_e5-large-v2:3:")

    def test_key_still_folds_model_top_k_and_pool(self):
        rc = RetrievalCache(redis_client=None)
        base = dict(dataset="d", embedding_model="m", top_k=3, query="q", corpus_sha1=_SHA_A)
        k = rc.make_key(**base)
        assert rc.make_key(**{**base, "embedding_model": "m2"}) != k
        assert rc.make_key(**{**base, "top_k": 5}) != k
        assert rc.make_key(**{**base, "query": "q2"}) != k
        pooled = rc.make_key(**base, pool=10)
        assert pooled != k
        assert "pool10" in pooled and "pool" not in k

    def test_same_inputs_same_key(self):
        rc = RetrievalCache(redis_client=None)
        base = dict(dataset="d", embedding_model="m", top_k=3, query="q", corpus_sha1=_SHA_A)
        assert rc.make_key(**base) == rc.make_key(**base)

    def test_corpus_hash_is_required(self):
        rc = RetrievalCache(redis_client=None)
        with pytest.raises(TypeError):
            rc.make_key(dataset="d", embedding_model="m", top_k=3, query="q")  # type: ignore[call-arg]

    @pytest.mark.parametrize("bad", ["", "   ", None, 12])
    def test_empty_or_untyped_corpus_hash_refuses(self, bad):
        rc = RetrievalCache(redis_client=None)
        with pytest.raises(RetrievalCacheKeyError, match="corpus"):
            rc.make_key(dataset="d", embedding_model="m", top_k=3, query="q", corpus_sha1=bad)

    def test_error_is_typed_and_cites_f5b(self):
        assert issubclass(RetrievalCacheKeyError, ValueError)
        assert "F5b" in (RetrievalCacheKeyError.__doc__ or "")


class TestCorpusHashOfIndex:
    def test_matches_the_ir_fingerprint(self):
        docs = _docs("d3", "d1", "d2")
        assert corpus_sha1_of_index(_IndexWithDocs(docs)) == corpus_doc_ids_sha1(docs)
        # Order-independent: the fingerprint is over SORTED ids.
        assert corpus_sha1_of_index(_IndexWithDocs(list(reversed(docs)))) == corpus_doc_ids_sha1(docs)

    def test_index_without_corpus_refuses(self):
        with pytest.raises(RetrievalCacheKeyError, match="corpus"):
            corpus_sha1_of_index(_IndexWithoutDocs())

    def test_index_with_none_corpus_refuses(self):
        with pytest.raises(RetrievalCacheKeyError, match="corpus"):
            corpus_sha1_of_index(_IndexWithDocs(None))  # type: ignore[arg-type]

    def test_none_index_refuses(self):
        with pytest.raises(RetrievalCacheKeyError):
            corpus_sha1_of_index(None)


class TestGetSetRoundTrip:
    def test_get_and_set_use_the_composed_key(self):
        client = _FakeClient()
        rc = RetrievalCache(client)  # type: ignore[arg-type]
        kw = dict(dataset="d", embedding_model="m", top_k=3, query="q", corpus_sha1=_SHA_A, pool=10)
        assert rc.get(**kw) is None
        rc.set(hits=[{"doc_id": "d1", "score": 1.0}], ttl_seconds=5, **kw)
        assert rc.get(**kw) == [{"doc_id": "d1", "score": 1.0}]
        expected = rc.make_key(**kw)
        assert client.gets == [("retrieval", expected)] * 2
        assert client.sets == [("retrieval", expected, 5)]

    def test_rebuilt_corpus_misses(self):
        client = _FakeClient()
        rc = RetrievalCache(client)  # type: ignore[arg-type]
        kw = dict(dataset="d", embedding_model="m", top_k=3, query="q")
        rc.set(hits=[{"doc_id": "old"}], corpus_sha1=_SHA_A, **kw)
        # Same query against a rebuilt index: never the stale hits.
        assert rc.get(corpus_sha1=_SHA_B, **kw) is None
        assert rc.get(corpus_sha1=_SHA_A, **kw) == [{"doc_id": "old"}]
