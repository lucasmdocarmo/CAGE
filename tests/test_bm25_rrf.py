"""Tests for the in-repo Okapi BM25 index, RRF fusion, and stage-tagged retrieval.

Charter refs: sec. 7.2 (BM25 offline gate, RRF optional-if-gap-found) and D8 sec. 8.2
(stage-tagged pool/reranked/served outputs for Layer-0 scoring).

Pure-Python units: no FAISS, no SentenceTransformers, no network. The reranker used in
the stage-tagged tests is a fake that only exercises the duck-typed resolve_hits path.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence

import pytest

from src.orchestration.ir import (
    BM25IRIndex,
    IRDocument,
    IRHit,
    RRFHit,
    StageTaggedRetrieval,
    bm25_tokenize,
    default_bm25_index_dir,
    ensure_bm25_index,
    rrf_fuse,
    stable_text_id,
    stage_tagged_search,
)


def _doc(text: str) -> IRDocument:
    return IRDocument(doc_id=stable_text_id(text), text=text, metadata={})


def _corpus() -> List[IRDocument]:
    # N=3, avgdl=(3+2+4)/3=3
    return [
        _doc("apple banana apple"),
        _doc("banana cherry"),
        _doc("cherry cherry cherry durian"),
    ]


# --------------------------------------------------------------------------- #
# BM25 (Robertson & Zaragoza 2009)
# --------------------------------------------------------------------------- #


def test_bm25_tokenize_lowercase_whitespace():
    assert bm25_tokenize("Apple  BANANA\ncherry") == ["apple", "banana", "cherry"]


def test_bm25_hand_computed_score():
    """Verify the exact Okapi BM25 value for a single-term query (hand-derived).

    Query 'apple': df=1, N=3 -> idf = ln(1 + (3-1+0.5)/(1+0.5)) = ln(8/3).
    Doc 'apple banana apple': tf=2, dl=3, avgdl=3, k1=1.5, b=0.75 ->
    denom = 2 + 1.5*(1 - 0.75 + 0.75*1) = 3.5; score = idf * 2*2.5/3.5.
    """
    docs = _corpus()
    idx = BM25IRIndex()  # standard defaults k1=1.5, b=0.75
    idx.build(docs)

    hits = idx.search("apple", top_k=5)
    assert [h.doc_id for h in hits] == [docs[0].doc_id]  # only d1 contains 'apple'

    expected = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5)) * (2 * (1.5 + 1)) / 3.5
    assert hits[0].score == pytest.approx(expected, rel=1e-9)
    assert hits[0].score == pytest.approx(1.401184, abs=1e-4)  # frozen numeric anchor


def test_bm25_ranking_order_and_multi_term():
    docs = _corpus()
    idx = BM25IRIndex()
    idx.build(docs)

    # 'cherry': d3 (tf=3) must outrank d2 (tf=1) despite d3 being longer.
    hits = idx.search("cherry", top_k=5)
    assert [h.doc_id for h in hits] == [docs[2].doc_id, docs[1].doc_id]
    assert hits[0].score > hits[1].score > 0.0

    # Multi-term query accumulates per-term scores; d2 matches both terms.
    hits2 = idx.search("banana cherry", top_k=5)
    assert hits2[0].doc_id == docs[1].doc_id


def test_bm25_no_match_and_top_k_truncation():
    docs = _corpus()
    idx = BM25IRIndex()
    idx.build(docs)

    assert idx.search("zzz-unseen-term", top_k=5) == []  # zero-score docs never returned
    assert len(idx.search("cherry", top_k=1)) == 1


def test_bm25_deterministic_tie_break_by_doc_id():
    # Two docs with identical text-statistics (same tokens, same length) tie exactly;
    # ordering must be ascending doc_id, stable across runs.
    d_a = IRDocument(doc_id="a" * 40, text="same tokens here", metadata={})
    d_b = IRDocument(doc_id="b" * 40, text="same tokens here", metadata={})
    idx = BM25IRIndex()
    idx.build([d_b, d_a])  # insertion order deliberately reversed

    hits = idx.search("tokens", top_k=5)
    assert [h.doc_id for h in hits] == [d_a.doc_id, d_b.doc_id]
    assert hits[0].score == pytest.approx(hits[1].score)


def test_bm25_parameter_validation_and_fail_closed():
    with pytest.raises(ValueError):
        BM25IRIndex(k1=-0.1)
    with pytest.raises(ValueError):
        BM25IRIndex(b=1.5)
    with pytest.raises(ValueError):
        BM25IRIndex().build([])  # empty corpus is an error, never a silent empty index

    idx = BM25IRIndex()
    with pytest.raises(ValueError):
        idx.search("apple")  # search before build
    idx.build(_corpus())
    with pytest.raises(ValueError):
        idx.search("apple", top_k=0)


def test_bm25_save_before_build_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        BM25IRIndex().save(tmp_path)


def test_bm25_resolve_hits_order():
    docs = _corpus()
    idx = BM25IRIndex()
    idx.build(docs)
    hits = [IRHit(doc_id=docs[2].doc_id, score=1.0), IRHit(doc_id=docs[0].doc_id, score=0.5)]
    resolved = idx.resolve_hits(hits)
    assert [d.doc_id for d in resolved] == [docs[2].doc_id, docs[0].doc_id]


def test_bm25_save_load_roundtrip(tmp_path: Path):
    docs = _corpus()
    idx = BM25IRIndex(k1=1.2, b=0.6)
    idx.build(docs)
    idx.save(tmp_path / "bm25")

    loaded = BM25IRIndex.load(tmp_path / "bm25")
    assert loaded.k1 == 1.2 and loaded.b == 0.6
    for query in ("apple", "cherry", "banana cherry"):
        assert loaded.search(query, top_k=5) == idx.search(query, top_k=5)


def test_bm25_load_fail_closed(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        BM25IRIndex.load(tmp_path / "missing")

    # A non-BM25 meta.json must be refused, not silently reinterpreted.
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "meta.json").write_text('{"index_type": "faiss", "k1": 1.5, "b": 0.75}')
    with pytest.raises(ValueError):
        BM25IRIndex.load(foreign)

    # Content-hash mismatch (tampered documents.jsonl) must raise.
    good = tmp_path / "good"
    idx = BM25IRIndex()
    idx.build(_corpus())
    idx.save(good)
    tampered = _doc("totally different corpus content")
    (good / "documents.jsonl").write_text(
        '{"doc_id": "%s", "text": "%s", "metadata": {}}\n' % (tampered.doc_id, tampered.text)
    )
    with pytest.raises(ValueError):
        BM25IRIndex.load(good)


def test_ensure_bm25_index_build_reload_and_staleness(tmp_path: Path):
    docs = _corpus()
    index_dir = default_bm25_index_dir(base_dir=tmp_path, dataset_name="unit")
    assert index_dir.name == "ir_unit_bm25"

    idx1 = ensure_bm25_index(index_dir=index_dir, documents=docs)
    assert (index_dir / "meta.json").exists()
    baseline_hits = idx1.search("cherry", top_k=5)

    # Reload path (same corpus, same params) returns identical rankings.
    idx2 = ensure_bm25_index(index_dir=index_dir, documents=docs)
    assert idx2.search("cherry", top_k=5) == baseline_hits

    # Parameter change forces a rebuild with the new parameters persisted.
    idx3 = ensure_bm25_index(index_dir=index_dir, documents=docs, k1=2.0)
    assert idx3.k1 == 2.0
    import json

    assert json.loads((index_dir / "meta.json").read_text())["k1"] == 2.0

    # Content change (same count, different docs) forces a rebuild (M2 guard).
    new_docs = [_doc("alpha beta"), _doc("beta gamma"), _doc("gamma delta")]
    idx4 = ensure_bm25_index(index_dir=index_dir, documents=new_docs, k1=2.0)
    assert idx4.search("alpha", top_k=1)  # retrievable only from the new corpus


# --------------------------------------------------------------------------- #
# RRF (Cormack, Clarke & Buettcher 2009)
# --------------------------------------------------------------------------- #


def _hit(doc_id: str) -> IRHit:
    return IRHit(doc_id=doc_id, score=0.0)


def test_rrf_hand_computed_scores_and_order():
    ranking_a = [_hit("d1"), _hit("d2"), _hit("d3")]
    ranking_b = [_hit("d3"), _hit("d1")]

    fused = rrf_fuse([ranking_a, ranking_b], k=60)
    by_id = {h.doc_id: h for h in fused}

    assert by_id["d1"].score == pytest.approx(1 / 61 + 1 / 62)
    assert by_id["d2"].score == pytest.approx(1 / 62)
    assert by_id["d3"].score == pytest.approx(1 / 63 + 1 / 61)
    assert [h.doc_id for h in fused] == ["d1", "d3", "d2"]


def test_rrf_contribution_trace():
    fused = rrf_fuse(
        [[_hit("d1"), _hit("d2")], [_hit("d2")]],
        k=60,
        names=["dense", "bm25"],
    )
    d2 = next(h for h in fused if h.doc_id == "d2")
    trace = {(c.ranking, c.rank): c.contribution for c in d2.contributions}
    assert trace == {
        ("dense", 2): pytest.approx(1 / 62),
        ("bm25", 1): pytest.approx(1 / 61),
    }
    assert d2.score == pytest.approx(1 / 62 + 1 / 61)

    d1 = next(h for h in fused if h.doc_id == "d1")
    assert [(c.ranking, c.rank) for c in d1.contributions] == [("dense", 1)]


def test_rrf_hits_are_irhits():
    fused = rrf_fuse([[_hit("d1")]])
    assert isinstance(fused[0], IRHit)
    assert isinstance(fused[0], RRFHit)


def test_rrf_tie_break_by_doc_id():
    # d_b and d_a each appear at rank 1 of one ranking -> exact tie -> doc_id order.
    fused = rrf_fuse([[_hit("d_b")], [_hit("d_a")]])
    assert [h.doc_id for h in fused] == ["d_a", "d_b"]


def test_rrf_duplicate_doc_within_one_ranking_uses_best_rank():
    fused = rrf_fuse([[_hit("d1"), _hit("d1")]], k=60)
    assert len(fused) == 1
    assert fused[0].score == pytest.approx(1 / 61)  # rank-2 duplicate ignored
    assert len(fused[0].contributions) == 1


def test_rrf_top_k_truncation():
    fused = rrf_fuse([[_hit("d1"), _hit("d2"), _hit("d3")]], top_k=2)
    assert [h.doc_id for h in fused] == ["d1", "d2"]


def test_rrf_validation_fail_closed():
    with pytest.raises(ValueError):
        rrf_fuse([])
    with pytest.raises(ValueError):
        rrf_fuse([[_hit("d1")]], k=0)
    with pytest.raises(ValueError):
        rrf_fuse([[_hit("d1")]], names=["a", "b"])
    with pytest.raises(ValueError):
        rrf_fuse([[_hit("d1")], [_hit("d2")]], names=["same", "same"])
    with pytest.raises(ValueError):
        rrf_fuse([[_hit("d1")]], top_k=0)


# --------------------------------------------------------------------------- #
# Stage-tagged retrieval (charter D8 sec. 8.2)
# --------------------------------------------------------------------------- #


class _FakeIndex:
    """Duck-typed index: fixed ranking, resolve_hits over an in-memory doc store."""

    def __init__(self, docs: Sequence[IRDocument]) -> None:
        self._docs = list(docs)

    def search(self, query: str, *, top_k: int = 5) -> List[IRHit]:
        return [
            IRHit(doc_id=d.doc_id, score=1.0 / (i + 1)) for i, d in enumerate(self._docs)
        ][:top_k]

    def resolve_hits(self, hits: Sequence[IRHit]) -> List[IRDocument]:
        by_id = {d.doc_id: d for d in self._docs}
        return [by_id[h.doc_id] for h in hits if h.doc_id in by_id]


class _ReversingReranker:
    """Fake reranker: resolves texts via the duck-typed index, reverses the order."""

    def __init__(self) -> None:
        self.resolved_texts: List[str] = []

    def rerank(self, query: str, hits: Sequence[IRHit], index) -> List[IRHit]:
        docs = index.resolve_hits(hits)
        self.resolved_texts = [d.text for d in docs]
        reordered = list(reversed(docs))
        return [IRHit(doc_id=d.doc_id, score=float(len(reordered) - i)) for i, d in enumerate(reordered)]


def test_stage_tagged_dense_default_no_reranker():
    docs = [_doc(f"passage number {i}") for i in range(6)]
    index = _FakeIndex(docs)

    result = stage_tagged_search("q", index=index, pool_k=5, served_k=2)
    assert isinstance(result, StageTaggedRetrieval)
    assert result.retriever == "dense"
    assert len(result.pool) == 5
    assert result.reranked is None  # stage did not run -> None, never a silent copy
    assert result.served == result.pool[:2]

    ranks = result.stage_ranks()
    assert set(ranks) == {"pool", "served"}  # no 'reranked' key when the stage is absent
    assert [r["rank"] for r in ranks["pool"]] == [1, 2, 3, 4, 5]
    assert ranks["served"][0]["doc_id"] == result.pool[0].doc_id


def test_stage_tagged_with_reranker_orders_served_by_rerank():
    docs = [_doc(f"passage number {i}") for i in range(4)]
    index = _FakeIndex(docs)
    reranker = _ReversingReranker()

    result = stage_tagged_search("q", index=index, pool_k=4, served_k=2, reranker=reranker)
    assert result.reranked is not None
    assert [h.doc_id for h in result.reranked] == [d.doc_id for d in reversed(docs)]
    assert [h.doc_id for h in result.served] == [docs[3].doc_id, docs[2].doc_id]
    assert reranker.resolved_texts  # texts really were resolved via the index
    assert "reranked" in result.stage_ranks()


def test_stage_tagged_prebuilt_pool_rrf_path():
    dense = [_hit("d1"), _hit("d2"), _hit("d3")]
    bm25 = [_hit("d3"), _hit("d1")]
    fused = rrf_fuse([dense, bm25], names=["dense", "bm25"])

    result = stage_tagged_search("q", pool=fused, pool_k=2, served_k=1, retriever_label="rrf")
    assert result.retriever == "rrf"
    assert len(result.pool) == 2  # truncated to pool_k
    assert [h.doc_id for h in result.served] == ["d1"]


def test_stage_tagged_bm25_index_with_reranker_duck_typing():
    idx = BM25IRIndex()
    idx.build(_corpus())
    reranker = _ReversingReranker()

    result = stage_tagged_search(
        "cherry", index=idx, pool_k=2, served_k=1, reranker=reranker, retriever_label="bm25"
    )
    assert result.retriever == "bm25"
    assert len(result.pool) == 2
    assert result.reranked is not None and len(result.reranked) == 2
    assert reranker.resolved_texts  # BM25IRIndex.resolve_hits satisfied the reranker


def test_stage_tagged_validation_fail_closed():
    index = _FakeIndex([_doc("x")])
    with pytest.raises(ValueError):
        stage_tagged_search("q")  # neither index nor pool
    with pytest.raises(ValueError):
        stage_tagged_search("q", index=index, pool=[_hit("d1")])  # both
    with pytest.raises(ValueError):
        stage_tagged_search("q", index=index, pool_k=0)
    with pytest.raises(ValueError):
        stage_tagged_search("q", index=index, served_k=0)
    with pytest.raises(ValueError):
        stage_tagged_search("q", index=index, pool_k=3, served_k=4)  # served > pool
    with pytest.raises(ValueError):
        # Reranker over a pre-built pool with no resolver: cannot resolve texts.
        stage_tagged_search("q", pool=[_hit("d1")], served_k=1, reranker=_ReversingReranker())
