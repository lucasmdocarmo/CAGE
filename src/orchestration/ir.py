"""Information Retrieval (IR) module for RAG baselines.

This module provides:
- Building a local corpus from dataset contexts
- Embedding documents and queries (SentenceTransformers)
- Vector search (FAISS)
- Optional persistence to disk for reuse across runs

Design goals:
- Keep the implementation simple and transparent (baseline-quality, not production IR)
- Make local experimentation reproducible
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class IRDocument:
    doc_id: str
    text: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class IRHit:
    doc_id: str
    score: float


def stable_text_id(text: str) -> str:
    """Deterministic ID for a document text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def corpus_doc_ids_sha1(documents: Sequence["IRDocument"]) -> str:
    """sha1 over the SORTED doc ids -- the corpus content fingerprint persisted in
    meta.json and checked by ensure_ir_index (audit 2026-07-16 M2). Doc ids are
    stable_text_id hashes of the passage text, so this fingerprints content, not order.
    """
    joined = "\n".join(sorted(d.doc_id for d in documents))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def build_corpus_from_contexts(
    examples: Sequence[Any],
    *,
    dataset_name: str,
) -> List[IRDocument]:
    """Build a deduplicated document corpus from CAGExample.context entries."""
    docs_by_id: dict[str, IRDocument] = {}

    for ex in examples:
        contexts = getattr(ex, "context", None) or []
        for ctx in contexts:
            if not ctx:
                continue
            doc_id = stable_text_id(ctx)
            if doc_id in docs_by_id:
                continue

            docs_by_id[doc_id] = IRDocument(
                doc_id=doc_id,
                text=str(ctx),
                metadata={
                    "dataset": dataset_name,
                    "source": "dataset_context",
                },
            )

    return list(docs_by_id.values())


class FaissIRIndex:
    """FAISS-backed IR index using SentenceTransformers embeddings."""

    def __init__(
        self,
        *,
        embedding_model: str = "intfloat/e5-large-v2",
        normalize_embeddings: bool = True,
        device: str = "cpu",
    ):
        self.embedding_model = embedding_model
        self.normalize_embeddings = normalize_embeddings
        self.device = device

        # E5 / BGE-style models REQUIRE asymmetric "query:"/"passage:" prefixes.
        # Omitting them runs the encoder out-of-distribution and silently degrades
        # retrieval (depressed Hit@k). Auto-enable for the model families that need it.
        model_lc = embedding_model.lower()
        self.uses_e5_prefixes = ("e5" in model_lc) or ("bge" in model_lc and "reranker" not in model_lc)

        self._st_model = None
        self._faiss = None
        self._index = None
        self._documents: list[IRDocument] = []

    def _format_passage(self, text: str) -> str:
        return f"passage: {text}" if self.uses_e5_prefixes else text

    def _format_query(self, text: str) -> str:
        return f"query: {text}" if self.uses_e5_prefixes else text

    def _ensure_deps(self) -> None:
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self.embedding_model, device=self.device)

        if self._faiss is None:
            import faiss

            self._faiss = faiss

    @property
    def documents(self) -> Sequence[IRDocument]:
        return self._documents

    def build(self, documents: Sequence[IRDocument], *, batch_size: int = 64) -> None:
        """Build the FAISS index from documents."""
        self._ensure_deps()
        if not documents:
            raise ValueError("No documents provided to build IR index")

        self._documents = list(documents)
        texts = [self._format_passage(d.text) for d in self._documents]

        # SentenceTransformers returns np.ndarray if convert_to_numpy=True
        embeddings = self._st_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        ).astype("float32")

        dim = embeddings.shape[1]

        # Cosine similarity: use inner product on normalized vectors.
        if self.normalize_embeddings:
            index = self._faiss.IndexFlatIP(dim)
        else:
            index = self._faiss.IndexFlatL2(dim)

        index.add(embeddings)
        self._index = index

    def search(self, query: str, *, top_k: int = 5) -> List[IRHit]:
        """Search for the top_k most similar documents to the query."""
        self._ensure_deps()
        if self._index is None:
            raise ValueError("IR index not built/loaded")

        q_emb = self._st_model.encode(
            [self._format_query(query)],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        ).astype("float32")

        scores, idxs = self._index.search(q_emb, top_k)
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()

        hits: list[IRHit] = []
        for score, idx in zip(scores, idxs):
            if idx < 0 or idx >= len(self._documents):
                continue
            hits.append(IRHit(doc_id=self._documents[idx].doc_id, score=float(score)))

        return hits

    def resolve_hits(self, hits: Sequence[IRHit]) -> List[IRDocument]:
        """Return IRDocument objects for the given hits (in order)."""
        by_id = {d.doc_id: d for d in self._documents}
        docs: list[IRDocument] = []
        for h in hits:
            d = by_id.get(h.doc_id)
            if d is not None:
                docs.append(d)
        return docs

    def save(self, directory: Path) -> None:
        """Persist index + documents to disk."""
        self._ensure_deps()
        if self._index is None:
            raise ValueError("IR index not built")

        directory.mkdir(parents=True, exist_ok=True)

        meta = {
            "embedding_model": self.embedding_model,
            "normalize_embeddings": self.normalize_embeddings,
            "num_documents": len(self._documents),
            # Content-hash staleness guard (audit 2026-07-16 M2): sha1 over the SORTED
            # doc ids, so ensure_ir_index can detect same-COUNT/different-CONTENT corpora
            # (two trials with equal-size corpora previously reused the wrong index).
            "doc_ids_sha1": corpus_doc_ids_sha1(self._documents),
            "uses_e5_prefixes": self.uses_e5_prefixes,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Documents
        with (directory / "documents.jsonl").open("w", encoding="utf-8") as f:
            for d in self._documents:
                f.write(
                    json.dumps(
                        {"doc_id": d.doc_id, "text": d.text, "metadata": d.metadata},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        # FAISS index
        self._faiss.write_index(self._index, str(directory / "faiss.index"))

    @classmethod
    def load(cls, directory: Path, *, device: str = "cpu") -> "FaissIRIndex":
        """Load a persisted index from disk."""
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing IR meta.json at {meta_path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        inst = cls(
            embedding_model=meta["embedding_model"],
            normalize_embeddings=bool(meta["normalize_embeddings"]),
            device=device,
        )
        # Respect how THIS index was built. Indices built before the e5-prefix fix
        # have no flag -> default False so queries match the (un-prefixed) passages.
        # Rebuild with --rebuild-ir-index to get the corrected, prefixed retrieval.
        inst.uses_e5_prefixes = bool(meta.get("uses_e5_prefixes", False))
        model_lc = str(meta.get("embedding_model", "")).lower()
        wants_prefixes = ("e5" in model_lc) or ("bge" in model_lc and "reranker" not in model_lc)
        if wants_prefixes and not inst.uses_e5_prefixes:
            print(
                f"WARNING: IR index at {directory} was built BEFORE the e5/bge "
                f"query:/passage: prefix fix (model={meta.get('embedding_model')}). "
                f"Retrieval is out-of-distribution and RAG/hybrid quality is degraded. "
                f"Rebuild with --rebuild-ir-index."
            )
        inst._ensure_deps()

        # Documents
        docs: list[IRDocument] = []
        docs_path = directory / "documents.jsonl"
        with docs_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                docs.append(
                    IRDocument(
                        doc_id=row["doc_id"],
                        text=row["text"],
                        metadata=row.get("metadata") or {},
                    )
                )
        inst._documents = docs

        # Index
        inst._index = inst._faiss.read_index(str(directory / "faiss.index"))
        return inst


def bm25_tokenize(text: str) -> List[str]:
    """Deterministic BM25 tokenization: lowercase + simple whitespace split.

    Deliberately matches the corpus-side text convention already used in this
    module (``_normalize_passage``: lowercase + whitespace collapse) so the
    lexical index and the passage-comparison helpers see the same token stream.
    Punctuation is NOT stripped (transparent, baseline-quality tokenization --
    the charter's offline gate compares retrievers under one fixed, documented
    tokenizer, not a tuned analyzer chain).
    """
    return text.lower().split()


class BM25IRIndex:
    """In-repo Okapi BM25 lexical index over the same chunk store as the dense retriever.

    Implements Okapi BM25 per Robertson & Zaragoza (2009), "The Probabilistic
    Relevance Framework: BM25 and Beyond", Foundations and Trends in IR 3(4)
    [robertson2009bm25]:

        score(q, d) = sum_{t in q} idf(t) * tf(t,d)*(k1+1) / (tf(t,d) + k1*(1 - b + b*|d|/avgdl))

    with the Robertson-Sparck Jones IDF (Robertson & Sparck Jones 1976, as
    presented in Robertson & Zaragoza 2009 sec. 3.3), in the standard
    non-negative "+1" smoothing variant (the Lucene convention):

        idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

    Design notes:
    - Pure Python, deterministic, NO new dependency (charter sec. 7.2 offline
      gate: the BM25 characterization table must be reproducible locally at $0).
    - Ties are broken by ascending doc_id so rankings are stable across runs.
    - Keyed by the same ``stable_text_id`` doc ids as ``FaissIRIndex`` so it
      reuses the existing qrels/content-hash machinery unchanged.
    - Interface mirrors ``FaissIRIndex``: build / search / resolve_hits / save / load.
    """

    META_INDEX_TYPE = "bm25"

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError(f"BM25 k1 must be >= 0, got {k1}")
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"BM25 b must be in [0, 1], got {b}")
        self.k1 = float(k1)
        self.b = float(b)

        self._documents: list[IRDocument] = []
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        # term -> list of (doc_index, term_frequency); postings kept sorted by doc_index.
        self._postings: Dict[str, List[Tuple[int, int]]] = {}
        self._idf: Dict[str, float] = {}
        self._built = False

    @property
    def documents(self) -> Sequence[IRDocument]:
        return self._documents

    def build(self, documents: Sequence[IRDocument]) -> None:
        """Build the inverted index + IDF table from documents (fail-closed on empty)."""
        if not documents:
            raise ValueError("No documents provided to build BM25 index")

        self._documents = list(documents)
        self._doc_lens = []
        self._postings = {}

        for doc_idx, doc in enumerate(self._documents):
            tokens = bm25_tokenize(doc.text)
            self._doc_lens.append(len(tokens))
            tf: Dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            for term, freq in tf.items():
                self._postings.setdefault(term, []).append((doc_idx, freq))

        n_docs = len(self._documents)
        self._avgdl = sum(self._doc_lens) / n_docs if n_docs else 0.0
        # RSJ IDF, "+1" non-negative smoothing variant (see class docstring).
        self._idf = {
            term: float(np.log(1.0 + (n_docs - len(postings) + 0.5) / (len(postings) + 0.5)))
            for term, postings in self._postings.items()
        }
        self._built = True

    def search(self, query: str, *, top_k: int = 5) -> List[IRHit]:
        """Top-k BM25 search. Only positive-scoring documents (>=1 matching term) return."""
        if not self._built:
            raise ValueError("BM25 index not built/loaded")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        scores: Dict[int, float] = {}
        for term in set(bm25_tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for doc_idx, tf in postings:
                dl = self._doc_lens[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self._avgdl if self._avgdl > 0 else 0.0))
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * (tf * (self.k1 + 1.0)) / denom

        # Deterministic ordering: descending score, then ascending doc_id on ties.
        ranked = sorted(
            ((self._documents[i].doc_id, s) for i, s in scores.items() if s > 0.0),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return [IRHit(doc_id=doc_id, score=float(score)) for doc_id, score in ranked[:top_k]]

    def resolve_hits(self, hits: Sequence[IRHit]) -> List[IRDocument]:
        """Return IRDocument objects for the given hits (in order); mirrors FaissIRIndex."""
        by_id = {d.doc_id: d for d in self._documents}
        docs: list[IRDocument] = []
        for h in hits:
            d = by_id.get(h.doc_id)
            if d is not None:
                docs.append(d)
        return docs

    def save(self, directory: Path) -> None:
        """Persist parameters + documents; postings are rebuilt deterministically on load."""
        if not self._built:
            raise ValueError("BM25 index not built")

        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "index_type": self.META_INDEX_TYPE,
            "k1": self.k1,
            "b": self.b,
            "num_documents": len(self._documents),
            # Same content-hash staleness guard as FaissIRIndex (audit 2026-07-16 M2).
            "doc_ids_sha1": corpus_doc_ids_sha1(self._documents),
            "tokenizer": "lowercase_whitespace",
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        with (directory / "documents.jsonl").open("w", encoding="utf-8") as f:
            for d in self._documents:
                f.write(
                    json.dumps(
                        {"doc_id": d.doc_id, "text": d.text, "metadata": d.metadata},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    @classmethod
    def load(cls, directory: Path) -> "BM25IRIndex":
        """Load a persisted BM25 index (fail-closed on missing/mismatched metadata)."""
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing BM25 meta.json at {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        index_type = meta.get("index_type")
        if index_type != cls.META_INDEX_TYPE:
            raise ValueError(
                f"Index at {directory} has index_type={index_type!r}, expected "
                f"{cls.META_INDEX_TYPE!r} -- refusing to load a non-BM25 index as BM25."
            )

        inst = cls(k1=float(meta["k1"]), b=float(meta["b"]))
        docs: list[IRDocument] = []
        docs_path = directory / "documents.jsonl"
        if not docs_path.exists():
            raise FileNotFoundError(f"Missing BM25 documents.jsonl at {docs_path}")
        with docs_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                docs.append(
                    IRDocument(
                        doc_id=row["doc_id"],
                        text=row["text"],
                        metadata=row.get("metadata") or {},
                    )
                )
        inst.build(docs)

        # Content-hash verification: the rebuilt corpus must match what was saved.
        stored_sha = meta.get("doc_ids_sha1")
        rebuilt_sha = corpus_doc_ids_sha1(docs)
        if stored_sha is not None and stored_sha != rebuilt_sha:
            raise ValueError(
                f"BM25 index at {directory} content hash mismatch: meta.json has "
                f"{stored_sha[:12]} but documents.jsonl rebuilds to {rebuilt_sha[:12]}."
            )
        return inst


class CrossEncoderReranker:
    """Optional cross-encoder reranker for retrieved hits."""

    def __init__(self, model_name: str, *, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.device = device
        self._model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        hits: Sequence[IRHit],
        index: "SupportsHitResolution",
    ) -> List[IRHit]:
        if not hits:
            return list(hits)

        docs = index.resolve_hits(hits)
        if not docs:
            return list(hits)

        pairs = [(query, d.text) for d in docs]
        scores = self._model.predict(pairs)

        scored = []
        for doc, score in zip(docs, scores):
            try:
                scored.append((doc.doc_id, float(score)))
            except Exception:
                scored.append((doc.doc_id, 0.0))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [IRHit(doc_id=doc_id, score=score) for doc_id, score in scored]


# --------------------------------------------------------------------------- #
# Reciprocal Rank Fusion (charter sec. 7.2 `rrf` retriever variant)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RRFContribution:
    """One source ranking's contribution to a fused hit's RRF score."""

    ranking: str  # label of the source ranking (e.g. 'bm25', 'dense')
    rank: int  # 1-based rank of the doc in that source ranking
    contribution: float  # 1 / (k + rank)


@dataclass(frozen=True)
class RRFHit(IRHit):
    """A fused hit: IRHit plus the per-ranking contribution trace."""

    contributions: Tuple[RRFContribution, ...] = ()


def rrf_fuse(
    rankings: Sequence[Sequence[IRHit]],
    *,
    k: int = 60,
    names: Optional[Sequence[str]] = None,
    top_k: Optional[int] = None,
) -> List[RRFHit]:
    """Reciprocal Rank Fusion of two or more rankings (e.g. BM25 + dense).

    Implements RRF per Cormack, Clarke & Buettcher (2009), "Reciprocal Rank
    Fusion outperforms Condorcet and individual Rank Learning Methods",
    SIGIR '09 [cormack2009rrf]:

        RRFscore(d) = sum_{r in rankings} 1 / (k + rank_r(d))

    with rank_r 1-based and k = 60, the paper's fixed constant. Documents absent
    from a ranking simply contribute nothing for it (no penalty term). If a doc
    appears more than once inside ONE ranking, only its best (first) rank counts.

    Returns ``RRFHit`` objects (an ``IRHit`` subclass, so downstream IRHit
    consumers work unchanged) carrying a per-hit contribution trace: which
    source ranking placed the doc at which rank and how much score that added.
    Ties are broken by ascending doc_id for run-to-run determinism.

    Args:
        rankings: two or more rank-ordered hit lists (fail-closed on <1 or on
            an empty list-of-lists; a single ranking is allowed and degenerates
            to a monotone rescoring).
        k: the RRF constant (default 60 per Cormack et al. 2009); must be >= 1.
        names: optional labels for the source rankings (default 'ranking_<i>');
            must be unique and match len(rankings).
        top_k: optionally truncate the fused ranking.
    """
    if not rankings:
        raise ValueError("rrf_fuse requires at least one ranking, got 0")
    if k < 1:
        raise ValueError(f"RRF k must be >= 1, got {k}")
    if names is None:
        names = [f"ranking_{i}" for i in range(len(rankings))]
    if len(names) != len(rankings):
        raise ValueError(
            f"names length {len(names)} does not match rankings length {len(rankings)}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"ranking names must be unique, got {list(names)}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    contributions: Dict[str, List[RRFContribution]] = {}
    for name, ranking in zip(names, rankings):
        seen: set[str] = set()
        for rank, hit in enumerate(ranking, start=1):
            if hit.doc_id in seen:
                continue  # best (first) rank wins within one ranking
            seen.add(hit.doc_id)
            contributions.setdefault(hit.doc_id, []).append(
                RRFContribution(ranking=name, rank=rank, contribution=1.0 / (k + rank))
            )

    fused = [
        RRFHit(
            doc_id=doc_id,
            score=sum(c.contribution for c in contribs),
            contributions=tuple(contribs),
        )
        for doc_id, contribs in contributions.items()
    ]
    # Deterministic ordering: descending fused score, ascending doc_id on ties.
    fused.sort(key=lambda h: (-h.score, h.doc_id))
    return fused[:top_k] if top_k is not None else fused


# --------------------------------------------------------------------------- #
# Stage-tagged retrieval (charter D8 sec. 8.2: pool / reranked / served stages)
# --------------------------------------------------------------------------- #


class SupportsHitResolution:
    """Structural interface: any index exposing search + resolve_hits.

    Satisfied by ``FaissIRIndex`` and ``BM25IRIndex`` (duck-typed; kept as a
    plain marker class rather than typing.Protocol to avoid a runtime-checkable
    dependency in hot paths).
    """

    def search(self, query: str, *, top_k: int = 5) -> List[IRHit]:  # pragma: no cover
        raise NotImplementedError

    def resolve_hits(self, hits: Sequence[IRHit]) -> List[IRDocument]:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class StageTaggedRetrieval:
    """Charter D8 sec. 8.2 stage-tagged retrieval output for Layer-0 scoring.

    Exposes the three stages the L0 scorer needs, each in rank order:
    - ``pool``:     pre-rerank candidate pool (top pool_k from the retriever /
                    fuser) -> feeds first-stage pool recall@100;
    - ``reranked``: the cross-encoder-reordered pool (None when no reranker is
                    configured -- the stage did not run; never silently equal
                    to the pool) -> feeds reranked nDCG@10;
    - ``served``:   what is actually placed in the prompt (top served_k of the
                    final stage) -> feeds served-context recall@k_served.
    """

    query: str
    retriever: str  # 'dense' | 'bm25' | 'rrf' | caller-defined label
    pool: Tuple[IRHit, ...]
    reranked: Optional[Tuple[IRHit, ...]]
    served: Tuple[IRHit, ...]
    pool_k: int
    served_k: int

    def stage_ranks(self) -> Dict[str, List[Dict[str, Any]]]:
        """Stage -> [{doc_id, rank, score}] with 1-based ranks.

        The exact shape downstream L0 scoring needs to build per-stage runs
        (e.g. ranx ``Run`` dicts) for pool recall@100 / reranked nDCG@10 /
        served recall@k_served per charter sec. 8.2. ``reranked`` is omitted
        (not emitted as an empty list) when the stage did not run.
        """
        out: Dict[str, List[Dict[str, Any]]] = {
            "pool": [
                {"doc_id": h.doc_id, "rank": r, "score": h.score}
                for r, h in enumerate(self.pool, start=1)
            ],
            "served": [
                {"doc_id": h.doc_id, "rank": r, "score": h.score}
                for r, h in enumerate(self.served, start=1)
            ],
        }
        if self.reranked is not None:
            out["reranked"] = [
                {"doc_id": h.doc_id, "rank": r, "score": h.score}
                for r, h in enumerate(self.reranked, start=1)
            ]
        return out


def stage_tagged_search(
    query: str,
    *,
    index: Optional[SupportsHitResolution] = None,
    pool: Optional[Sequence[IRHit]] = None,
    pool_k: int = 100,
    served_k: int = 5,
    reranker: Optional[Any] = None,
    resolve_index: Optional[SupportsHitResolution] = None,
    retriever_label: str = "dense",
) -> StageTaggedRetrieval:
    """Run retrieval with charter sec. 8.2 stage tags: pool -> (reranked) -> served.

    ADDITIVE API: the existing dense+cross-encoder path in the runner is
    untouched; this function packages the same primitives (index.search,
    reranker.rerank) with explicit stage outputs so downstream Layer-0 scoring
    can compute pool recall@100, reranked nDCG@10, and served recall@k_served.

    Exactly one of ``index`` / ``pool`` must be provided:
    - ``index``: any object with ``search``/``resolve_hits`` (FaissIRIndex or
      BM25IRIndex); the pool is ``index.search(query, top_k=pool_k)``.
    - ``pool``: a pre-built candidate list (e.g. the output of ``rrf_fuse`` for
      the 'rrf' retriever variant), truncated to pool_k.

    When ``reranker`` is given, texts are resolved via ``resolve_index``
    (default: ``index``); fail-closed if neither can resolve.
    """
    if (index is None) == (pool is None):
        raise ValueError("stage_tagged_search requires exactly one of index= or pool=")
    if pool_k < 1:
        raise ValueError(f"pool_k must be >= 1, got {pool_k}")
    if served_k < 1:
        raise ValueError(f"served_k must be >= 1, got {served_k}")
    if served_k > pool_k:
        raise ValueError(
            f"served_k ({served_k}) cannot exceed pool_k ({pool_k}): the served "
            f"context is a subset of the candidate pool by construction."
        )

    if index is not None:
        pool_hits: List[IRHit] = list(index.search(query, top_k=pool_k))
    else:
        pool_hits = list(pool or [])[:pool_k]

    reranked_hits: Optional[List[IRHit]] = None
    if reranker is not None:
        resolver = resolve_index if resolve_index is not None else index
        if resolver is None:
            raise ValueError(
                "stage_tagged_search with a reranker over a pre-built pool needs "
                "resolve_index= (an index able to resolve hit texts)."
            )
        reranked_hits = list(reranker.rerank(query, pool_hits, resolver))

    final = reranked_hits if reranked_hits is not None else pool_hits
    return StageTaggedRetrieval(
        query=query,
        retriever=retriever_label,
        pool=tuple(pool_hits),
        reranked=tuple(reranked_hits) if reranked_hits is not None else None,
        served=tuple(final[:served_k]),
        pool_k=pool_k,
        served_k=served_k,
    )


def default_index_dir(
    *,
    base_dir: Path,
    dataset_name: str,
    embedding_model: str,
) -> Path:
    safe_model = embedding_model.replace("/", "_")
    return base_dir / f"ir_{dataset_name}_{safe_model}"


def default_bm25_index_dir(
    *,
    base_dir: Path,
    dataset_name: str,
) -> Path:
    return base_dir / f"ir_{dataset_name}_bm25"


def ensure_ir_index(
    *,
    index_dir: Path,
    documents: Sequence[IRDocument],
    embedding_model: str,
    rebuild: bool = False,
    device: str = "cpu",
) -> FaissIRIndex:
    """Load an existing index if present AND current, otherwise build and persist one.

    Staleness guard: the repo ships tiny STUB indices (e.g. 17 docs) and a committed index
    can also lag the corpus. Loading such an index silently makes every RAG/redis/hybrid
    baseline retrieve from the wrong corpus (invalid retrieval + quality metrics, no error).
    Guarded by CONTENT (audit 2026-07-16 M2): meta.json stores doc_ids_sha1 (sha1 of the
    sorted doc ids) at build time; a hash mismatch REBUILDS. The count-only check that
    preceded it let two same-size/different-content trial corpora silently share an index
    (the 100x3 run escaped only because its per-trial corpora were 31/30/32 docs).
    Backward compat: an old meta.json without doc_ids_sha1 triggers ONE rebuild, which
    persists the hash (a rebuild also restores the correct e5 prefixes).
    """
    meta_path = index_dir / "meta.json"
    if meta_path.exists() and not rebuild:
        stale = False
        try:
            import json as _json
            _meta = _json.loads(meta_path.read_text())
            _n = _meta.get("num_documents")
            # `documents` (truthy) also guards the empty-corpus case: an empty corpus must NOT
            # trigger a rebuild (idx.build([]) raises); fall through to load the existing index.
            if _n is not None and documents and int(_n) != len(documents):
                print(
                    f"[ir] index at {index_dir} has {_n} docs but the corpus has "
                    f"{len(documents)}; rebuilding (stale/stub index)."
                )
                stale = True
            elif documents:
                _stored_sha = _meta.get("doc_ids_sha1")
                _corpus_sha = corpus_doc_ids_sha1(documents)
                if _stored_sha is None:
                    print(
                        f"[ir] index at {index_dir} predates the content-hash guard "
                        f"(no doc_ids_sha1 in meta.json); rebuilding once to stamp it."
                    )
                    stale = True
                elif _stored_sha != _corpus_sha:
                    print(
                        f"[ir] index at {index_dir} content hash {_stored_sha[:12]} != "
                        f"corpus {_corpus_sha[:12]} (same count, different documents); "
                        f"rebuilding (stale index)."
                    )
                    stale = True
        except Exception:
            stale = False  # unreadable meta -> fall through to load (prior behavior)
        if not stale:
            return FaissIRIndex.load(index_dir, device=device)

    idx = FaissIRIndex(embedding_model=embedding_model, device=device)
    idx.build(documents)
    idx.save(index_dir)
    return idx


def ensure_bm25_index(
    *,
    index_dir: Path,
    documents: Sequence[IRDocument],
    k1: float = 1.5,
    b: float = 0.75,
    rebuild: bool = False,
) -> BM25IRIndex:
    """Load a persisted BM25 index if present AND current, else build and persist one.

    Mirrors ``ensure_ir_index``'s content-hash staleness guard (audit 2026-07-16
    M2): meta.json's ``doc_ids_sha1`` must match the live corpus, and the stored
    k1/b must match the requested parameters (a parameter change invalidates the
    ranking semantics even over identical documents, so it forces a rebuild).
    """
    meta_path = index_dir / "meta.json"
    if meta_path.exists() and not rebuild:
        stale = False
        try:
            _meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if _meta.get("index_type") != BM25IRIndex.META_INDEX_TYPE:
                stale = True  # foreign index dir -> rebuild as BM25
            elif float(_meta.get("k1", -1)) != float(k1) or float(_meta.get("b", -1)) != float(b):
                print(
                    f"[ir] BM25 index at {index_dir} has k1={_meta.get('k1')}/b={_meta.get('b')} "
                    f"but k1={k1}/b={b} requested; rebuilding (parameter change)."
                )
                stale = True
            elif documents:
                _stored_sha = _meta.get("doc_ids_sha1")
                _corpus_sha = corpus_doc_ids_sha1(documents)
                if _stored_sha != _corpus_sha:
                    print(
                        f"[ir] BM25 index at {index_dir} content hash "
                        f"{str(_stored_sha)[:12]} != corpus {_corpus_sha[:12]}; "
                        f"rebuilding (stale index)."
                    )
                    stale = True
        except Exception:
            stale = True  # unreadable/corrupt meta -> rebuild (BM25 rebuilds are cheap)
        if not stale:
            return BM25IRIndex.load(index_dir)

    idx = BM25IRIndex(k1=k1, b=b)
    idx.build(documents)
    idx.save(index_dir)
    return idx


def _normalize_passage(text: str) -> str:
    """Lowercase + whitespace-collapse for robust passage-text comparison."""
    return " ".join(text.lower().split())


def retrieval_hit_rate(
    *,
    gold_doc_ids: Sequence[str],
    retrieved_doc_ids: Sequence[str],
    gold_texts: Optional[Sequence[str]] = None,
    retrieved_texts: Optional[Sequence[str]] = None,
) -> float:
    """LENIENT hit@k coverage indicator: 1.0 if a gold passage is present anywhere in the
    retrieved set, else 0.0.

    Primary check is an exact doc-id match. A normalized-TEXT fallback runs when the ids
    do not match but texts are supplied, because the corpus passage and the gold passage
    can hash to different stable_text_id values if they differ only in whitespace or
    encoding. In Phase 2 this made the metric a false 0.0 for every row even when top-1
    similarity was ~0.99; the text fallback fixes that. Returns 0.0 if gold is unknown.

    Fix #5 (option B): this is intentionally a LENIENT presence check -- the bidirectional
    substring fallback can rubber-stamp 1.0 on a closed corpus, so read it as "was a gold
    passage present in the retrieved set", NOT as graded retrieval quality. The false-0.0 bug
    the fallback fixes is a worse failure than a lenient 1.0, so the logic is kept as-is. For
    a GRADED signal use retrieval_rank_of_gold (MRR, below) and the retrieval_top1_score the
    runner already records; both discriminate where this binary indicator saturates.
    """
    gold = set(gold_doc_ids)
    if gold and any(d in gold for d in retrieved_doc_ids):
        return 1.0
    if gold_texts and retrieved_texts:
        gold_norm = {_normalize_passage(t) for t in gold_texts if t and t.strip()}
        if gold_norm:
            for r in retrieved_texts:
                rn = _normalize_passage(r)
                if not rn:
                    continue
                if rn in gold_norm or any(g in rn or rn in g for g in gold_norm):
                    return 1.0
            return 0.0
    return 0.0


def retrieval_rank_of_gold(
    *,
    gold_doc_ids: Sequence[str],
    retrieved_doc_ids: Sequence[str],
    gold_texts: Optional[Sequence[str]] = None,
    retrieved_texts: Optional[Sequence[str]] = None,
) -> Optional[int]:
    """1-based rank of the FIRST retrieved passage that matches gold, else None (miss).

    Graded companion to retrieval_hit_rate (fix #5, option C). Where hit@k saturates at 1.0
    on a closed corpus, the rank discriminates a top-1 retrieval from a rank-8 one, giving
    Mean Reciprocal Rank (MRR = mean of 1/rank over queries; a miss contributes 0). It mirrors
    the hit matcher exactly -- exact doc-id first, normalized-text fallback second -- but walks
    ``retrieved_doc_ids`` / ``retrieved_texts`` IN ORDER so position is preserved. Both id and
    text lists are assumed to be in retrieval-rank order (the runner passes them straight from
    the scored hit list). Returns None when gold is unknown so callers can distinguish a miss
    from an unmeasurable row.
    """
    gold = set(gold_doc_ids)
    if not gold and not (gold_texts and any((t or "").strip() for t in gold_texts)):
        return None  # gold unknown -> unmeasurable, not a miss

    gold_norm = (
        {_normalize_passage(t) for t in gold_texts if t and t.strip()}
        if gold_texts
        else set()
    )
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold:
            return rank
    # Text fallback preserves rank: index into retrieved_texts positionally.
    if gold_norm and retrieved_texts is not None:
        for rank, r in enumerate(retrieved_texts, start=1):
            rn = _normalize_passage(r)
            if not rn:
                continue
            if rn in gold_norm or any(g in rn or rn in g for g in gold_norm):
                return rank
    return None  # a gold passage was defined but never retrieved -> reciprocal rank 0
