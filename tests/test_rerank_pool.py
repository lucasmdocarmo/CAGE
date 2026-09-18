"""ADR-0104 (owner decision 2026-09-16): rerank POOL then SERVE top-k.

B6 and every arm inheriting the ranked pipeline retrieve a dense candidate
pool of RERANK_POOL (10) hits, rerank the whole pool with the pinned
cross-encoder, and serve the top ``--top-k`` (3). B5 keeps serving the dense
top 3 unranked. Pins, on the runner side (scripts/3_run/run_experiment.py):

- pool -> rerank -> truncate ORDERING: the served top-3 of a reranked
  pool-10 differs from the reranked top-3 when the reranker promotes a
  rank-7 doc (the legacy path, ``--rerank-pool`` unset, reranks exactly
  the top_k hits, so the promoted doc can never be served);
- a pool WITHOUT a reranker is refused with a typed error (a silent
  pool on B5 would change the ablation's control arm);
- the Redis retrieval-cache key carries the pool size when a pool is used
  (a pool-10 hit list is not a top-3 hit list), the cached payload is the
  POOL hits, and the rerank runs AFTER the cache;
- per-row telemetry carries retrieval_pool_size and retrieval_reranked;
- ``--rerank-pool`` is parsed by the runner CLI and refused before any
  dataset or engine work when the reranker is off;
- backlog F5b: the cache key ALSO folds the index's corpus hash
  (``corpus_sha1``), so a rebuilt index never serves stale hits, and
  ``dense_retrieve`` refuses (typed) a cache without a corpus hash before
  touching the index.

The driver side (RERANK_POOL, RETRIEVER_ARGV) is pinned in
tests/test_run_campaign.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.orchestration.ir import IRDocument, IRHit  # noqa: E402
from src.orchestration.redis_cache import RetrievalCache, RetrievalCacheKeyError  # noqa: E402

RUNNER_PATH = REPO_ROOT / "scripts" / "3_run" / "run_experiment.py"

#: A fixed corpus fingerprint (backlog F5b: every cache call carries one).
_CORPUS_SHA = "c" * 40


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_experiment_rerank_pool", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


# --------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------


class _StubIndex:
    """Dense index stand-in: d1..d10 with strictly descending dense scores."""

    def __init__(self, n: int = 10) -> None:
        self.docs = [
            IRDocument(doc_id=f"d{i}", text=f"text of d{i}", metadata={})
            for i in range(1, n + 1)
        ]
        self.search_calls: List[int] = []

    def search(self, query: str, *, top_k: int = 5) -> List[IRHit]:
        self.search_calls.append(top_k)
        return [
            IRHit(doc_id=d.doc_id, score=float(len(self.docs) - i))
            for i, d in enumerate(self.docs[:top_k])
        ]

    def resolve_hits(self, hits: Sequence[IRHit]) -> List[IRDocument]:
        by_id = {d.doc_id: d for d in self.docs}
        return [by_id[h.doc_id] for h in hits if h.doc_id in by_id]


class _PromoteD7Reranker:
    """Cross-encoder stand-in: d7 scores highest, the rest keep dense order."""

    def __init__(self) -> None:
        self.calls: List[List[str]] = []

    def rerank(self, query: str, hits: Sequence[IRHit], index: Any) -> List[IRHit]:
        self.calls.append([h.doc_id for h in hits])
        scored = [
            (h.doc_id, 100.0 if h.doc_id == "d7" else h.score) for h in hits
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [IRHit(doc_id=d, score=s) for d, s in scored]


class _RecordingCache:
    """RetrievalCache stand-in recording every get/set keyword set."""

    def __init__(self) -> None:
        self.store: Dict[tuple, List[Dict[str, Any]]] = {}
        self.get_calls: List[Dict[str, Any]] = []
        self.set_calls: List[Dict[str, Any]] = []

    @staticmethod
    def _key(kw: Dict[str, Any]) -> tuple:
        return (
            kw["dataset"], kw["embedding_model"], kw["top_k"], kw.get("pool"),
            kw["corpus_sha1"], kw["query"],
        )

    def get(self, **kw: Any) -> Optional[List[Dict[str, Any]]]:
        self.get_calls.append(dict(kw))
        return self.store.get(self._key(kw))

    def set(self, **kw: Any) -> None:
        self.set_calls.append(dict(kw))
        self.store[self._key(kw)] = list(kw["hits"])


# --------------------------------------------------------------------------
# resolve_rerank_pool: the fail-closed validator
# --------------------------------------------------------------------------


class TestResolveRerankPool:
    def test_unset_is_legacy(self):
        assert runner.resolve_rerank_pool(None, top_k=3, reranker_active=True) is None
        assert runner.resolve_rerank_pool(None, top_k=3, reranker_active=False) is None

    def test_pool_with_reranker(self):
        assert runner.resolve_rerank_pool(10, top_k=3, reranker_active=True) == 10

    def test_pool_without_reranker_refuses(self):
        with pytest.raises(runner.RerankPoolError, match="reranker"):
            runner.resolve_rerank_pool(10, top_k=3, reranker_active=False)

    def test_pool_smaller_than_served_refuses(self):
        with pytest.raises(runner.RerankPoolError, match="top_k"):
            runner.resolve_rerank_pool(2, top_k=3, reranker_active=True)

    def test_non_positive_pool_refuses(self):
        with pytest.raises(runner.RerankPoolError):
            runner.resolve_rerank_pool(0, top_k=3, reranker_active=True)

    def test_error_is_typed_value_error(self):
        assert issubclass(runner.RerankPoolError, ValueError)
        assert "ADR-0104" in (runner.RerankPoolError.__doc__ or "")


# --------------------------------------------------------------------------
# dense_retrieve: pool -> rerank -> truncate
# --------------------------------------------------------------------------


class TestDenseRetrieveOrdering:
    def test_pool_rerank_truncate_serves_promoted_doc(self):
        index = _StubIndex()
        reranker = _PromoteD7Reranker()
        out = runner.dense_retrieve(
            "q",
            ir_index=index,
            top_k=3,
            reranker=reranker,
            rerank_pool=10,
            retrieval_cache=None,
            dataset="squad_v2",
            embedding_model="intfloat/e5-large-v2",
        )
        # The pool is the dense top-10, reranked WHOLE, then cut to top_k.
        assert index.search_calls == [10]
        assert reranker.calls == [[f"d{i}" for i in range(1, 11)]]
        assert [h.doc_id for h in out.hits] == ["d7", "d1", "d2"]
        assert out.reranked is True
        assert out.cached is False
        assert out.pool_size == 10

    def test_legacy_reranks_exactly_top_k(self):
        index = _StubIndex()
        reranker = _PromoteD7Reranker()
        out = runner.dense_retrieve(
            "q",
            ir_index=index,
            top_k=3,
            reranker=reranker,
            rerank_pool=None,
            retrieval_cache=None,
            dataset="squad_v2",
            embedding_model="intfloat/e5-large-v2",
        )
        # Legacy behavior byte-for-byte: search top_k, rerank those, serve
        # all of them. d7 was never a candidate.
        assert index.search_calls == [3]
        assert reranker.calls == [["d1", "d2", "d3"]]
        assert [h.doc_id for h in out.hits] == ["d1", "d2", "d3"]
        assert out.reranked is True
        assert out.pool_size is None

    def test_served_count_unchanged_by_the_pool(self):
        index = _StubIndex()
        pooled = runner.dense_retrieve(
            "q", ir_index=index, top_k=3, reranker=_PromoteD7Reranker(),
            rerank_pool=10, retrieval_cache=None, dataset="d", embedding_model="m",
        )
        legacy = runner.dense_retrieve(
            "q", ir_index=index, top_k=3, reranker=_PromoteD7Reranker(),
            rerank_pool=None, retrieval_cache=None, dataset="d", embedding_model="m",
        )
        assert len(pooled.hits) == len(legacy.hits) == 3

    def test_no_reranker_no_pool_is_b5(self):
        index = _StubIndex()
        out = runner.dense_retrieve(
            "q", ir_index=index, top_k=3, reranker=None,
            rerank_pool=None, retrieval_cache=None, dataset="d", embedding_model="m",
        )
        assert index.search_calls == [3]
        assert [h.doc_id for h in out.hits] == ["d1", "d2", "d3"]
        assert out.reranked is False
        assert out.pool_size is None

    def test_pool_without_reranker_refuses_at_retrieval(self):
        index = _StubIndex()
        with pytest.raises(runner.RerankPoolError):
            runner.dense_retrieve(
                "q", ir_index=index, top_k=3, reranker=None,
                rerank_pool=10, retrieval_cache=None, dataset="d", embedding_model="m",
            )
        assert index.search_calls == []


# --------------------------------------------------------------------------
# Retrieval cache: the key carries the pool, the payload IS the pool
# --------------------------------------------------------------------------


class TestRetrievalCacheWithPool:
    def test_cache_key_and_payload_carry_the_pool(self):
        index = _StubIndex()
        reranker = _PromoteD7Reranker()
        cache = _RecordingCache()
        kw = dict(
            ir_index=index, top_k=3, reranker=reranker, rerank_pool=10,
            retrieval_cache=cache, dataset="squad_v2",
            embedding_model="intfloat/e5-large-v2", ttl_seconds=60,
            corpus_sha1=_CORPUS_SHA,
        )
        miss = runner.dense_retrieve("q", **kw)
        assert miss.cached is False
        assert cache.get_calls[0]["top_k"] == 3
        assert cache.get_calls[0]["pool"] == 10
        assert cache.get_calls[0]["corpus_sha1"] == _CORPUS_SHA
        assert len(cache.set_calls) == 1
        assert cache.set_calls[0]["pool"] == 10
        assert cache.set_calls[0]["corpus_sha1"] == _CORPUS_SHA
        assert cache.set_calls[0]["top_k"] == 3
        assert cache.set_calls[0]["ttl_seconds"] == 60
        # The cached payload is the POOL (dense order, pre-rerank), not the
        # served list.
        assert [h["doc_id"] for h in cache.set_calls[0]["hits"]] == [
            f"d{i}" for i in range(1, 11)
        ]

        hit = runner.dense_retrieve("q", **kw)
        assert hit.cached is True
        assert hit.reranked is True
        assert hit.pool_size == 10
        # The rerank runs AFTER the cache: the served list is identical.
        assert [h.doc_id for h in hit.hits] == [h.doc_id for h in miss.hits] == ["d7", "d1", "d2"]
        assert index.search_calls == [10]
        assert len(reranker.calls) == 2

    def test_legacy_cache_calls_carry_no_pool(self):
        index = _StubIndex()
        cache = _RecordingCache()
        runner.dense_retrieve(
            "q", ir_index=index, top_k=3, reranker=None, rerank_pool=None,
            retrieval_cache=cache, dataset="d", embedding_model="m",
            corpus_sha1=_CORPUS_SHA,
        )
        assert cache.get_calls[0].get("pool") is None
        assert cache.set_calls[0].get("pool") is None
        assert [h["doc_id"] for h in cache.set_calls[0]["hits"]] == ["d1", "d2", "d3"]

    def test_redis_key_includes_pool(self):
        rc = RetrievalCache(redis_client=None)  # make_key never touches redis
        base = rc.make_key(
            dataset="squad_v2", embedding_model="intfloat/e5-large-v2", top_k=3, query="q",
            corpus_sha1=_CORPUS_SHA,
        )
        pooled = rc.make_key(
            dataset="squad_v2", embedding_model="intfloat/e5-large-v2", top_k=3, query="q",
            pool=10, corpus_sha1=_CORPUS_SHA,
        )
        assert base != pooled
        assert "pool10" in pooled
        assert "pool" not in base  # the pool segment rides pooled keys only
        assert base.startswith("squad_v2:intfloat_e5-large-v2:3:")

    # ---- backlog F5b: the corpus hash rides every cache call ----------

    def test_cache_without_corpus_hash_refuses_before_search(self):
        index = _StubIndex()
        cache = _RecordingCache()
        with pytest.raises(RetrievalCacheKeyError, match="corpus"):
            runner.dense_retrieve(
                "q", ir_index=index, top_k=3, reranker=None, rerank_pool=None,
                retrieval_cache=cache, dataset="d", embedding_model="m",
            )
        assert index.search_calls == []
        assert cache.get_calls == [] and cache.set_calls == []

    def test_no_cache_needs_no_corpus_hash(self):
        index = _StubIndex()
        out = runner.dense_retrieve(
            "q", ir_index=index, top_k=3, reranker=None, rerank_pool=None,
            retrieval_cache=None, dataset="d", embedding_model="m",
        )
        assert [h.doc_id for h in out.hits] == ["d1", "d2", "d3"]

    def test_rebuilt_corpus_never_serves_stale_hits(self):
        index = _StubIndex()
        cache = _RecordingCache()
        kw = dict(
            ir_index=index, top_k=3, reranker=None, rerank_pool=None,
            retrieval_cache=cache, dataset="d", embedding_model="m",
        )
        first = runner.dense_retrieve("q", corpus_sha1="a" * 40, **kw)
        assert first.cached is False
        # Same query, rebuilt index (new corpus hash): a MISS, re-searched.
        second = runner.dense_retrieve("q", corpus_sha1="b" * 40, **kw)
        assert second.cached is False
        assert index.search_calls == [3, 3]
        # Same corpus again: a HIT.
        third = runner.dense_retrieve("q", corpus_sha1="a" * 40, **kw)
        assert third.cached is True
        assert index.search_calls == [3, 3]


# --------------------------------------------------------------------------
# Runner CLI: --rerank-pool parsed, refused early without a reranker
# --------------------------------------------------------------------------


class TestRunnerCli:
    def test_rerank_pool_without_reranker_refuses_before_any_work(
        self, monkeypatch, tmp_path, capsys
    ):
        # main() turns any run_experiment exception into SystemExit(1) after
        # printing it, so the pin is: exit 1, the RerankPoolError text on
        # stdout, and the dataset loader NEVER reached (its sentinel would
        # otherwise be the printed error).
        argv = [
            "run_experiment.py",
            "--baseline", "rag",
            "--model", "stub/model",
            "--dataset", "squad_v2",
            "--num-queries", "1",
            "--output-dir", str(tmp_path),
            "--retriever", "dense",
            "--reranker-model", "none",
            "--rerank-pool", "10",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        def _no_loader(*a, **k):
            raise AssertionError("dataset loader reached: the refusal was not early")

        monkeypatch.setattr(runner, "get_loader", _no_loader)
        with pytest.raises(SystemExit) as ei:
            runner.main()
        assert ei.value.code == 1
        out = capsys.readouterr().out
        assert "--rerank-pool 10 requires an active reranker" in out
        assert "dataset loader reached" not in out

    def test_rerank_pool_default_is_none(self):
        src = RUNNER_PATH.read_text(encoding="utf-8")
        assert '"--rerank-pool"' in src
        assert "rerank_pool=args.rerank_pool" in src
