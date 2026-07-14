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
        index: FaissIRIndex,
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


def default_index_dir(
    *,
    base_dir: Path,
    dataset_name: str,
    embedding_model: str,
) -> Path:
    safe_model = embedding_model.replace("/", "_")
    return base_dir / f"ir_{dataset_name}_{safe_model}"


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
    So if the persisted meta.json's num_documents does not match the corpus actually being
    indexed, REBUILD instead of loading (a rebuild also restores the correct e5 prefixes).
    """
    meta_path = index_dir / "meta.json"
    if meta_path.exists() and not rebuild:
        stale = False
        try:
            import json as _json
            _n = _json.loads(meta_path.read_text()).get("num_documents")
            # `documents` (truthy) also guards the empty-corpus case: an empty corpus must NOT
            # trigger a rebuild (idx.build([]) raises); fall through to load the existing index.
            if _n is not None and documents and int(_n) != len(documents):
                print(
                    f"[ir] index at {index_dir} has {_n} docs but the corpus has "
                    f"{len(documents)}; rebuilding (stale/stub index)."
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
