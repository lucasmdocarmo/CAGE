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
    """Load an existing index if present, otherwise build and persist one."""
    if (index_dir / "meta.json").exists() and not rebuild:
        return FaissIRIndex.load(index_dir, device=device)

    idx = FaissIRIndex(embedding_model=embedding_model, device=device)
    idx.build(documents)
    idx.save(index_dir)
    return idx


def retrieval_hit_rate(
    *,
    gold_doc_ids: Sequence[str],
    retrieved_doc_ids: Sequence[str],
) -> float:
    """Compute a simple hit indicator (1.0 if any gold doc is retrieved, else 0.0)."""
    gold = set(gold_doc_ids)
    if not gold:
        return 0.0
    return 1.0 if any(d in gold for d in retrieved_doc_ids) else 0.0
