"""Layer-0 stage-tagged evidence/retrieval scoring (PUBLICATION.md D8 §8.2).

The charter's Layer 0 runs pre-campaign, once per dataset × retriever variant,
and its columns are JOINED per cell by reference — cells are keyed by the
``CellSpec`` row key (cloud/RESULTS_LAYOUT.md §2). The metrics are STAGE-TAGGED
(AMENDED 2026-08-01 per IR audit):

- **first-stage recall@100** — the candidate pool's job is to not lose the
  answer (the B5-vs-B6 contrast requires this row: a reranker cannot rank what
  the pool never contained);
- **reranked nDCG@10** — the ordering job;
- **served-context recall@k_served** — what the model actually saw;
- **MRR@10** — single-gold sets only;
- **complete-evidence@k** — the multi-hop ceiling metric: fraction of queries
  with EVERY gold paragraph in the served context (recall@k averages over hops
  and hides unanswerability — 1-of-2 hops = recall 0.5, answerability 0).

Recall/nDCG/MRR are computed with **ranx** (Bassani 2022, "ranx: A Blazing-Fast
Python Library for Ranking Evaluation and Comparison", ECIR — the library the
charter names in §8.2; paired significance in-library). Complete-evidence@k is
CAGE-native (defined by charter §8.2, amendment 2026-08-01) and is computed
directly here.

Fail-closed doctrine (mirrors ``src.evaluation.quality.InstrumentUnavailableError``):
a missing ``ranx`` raises ``RanxUnavailableError`` at scoring time — never a
silent skip and never a hand-rolled substitute for the registered library.
``ranx`` is pinned in requirements.txt (``ranx==0.3.21``, Tier 1); this module
imports lazily so the rest of ``src.analysis`` stays importable in an
environment that was not installed from it.

Qrels sources (charter §8.2: "Qrels derive from gold evidence annotations"):
- ``qrels_from_manifest`` — the uniform-yardstick query manifest
  (``src.data.manifest.build_manifest`` output): each question's assigned corpus
  block is its relevant document (doc id ``block-<block_id>``).
- ``qrels_from_gold_docs`` — loader-metadata route: an explicit mapping of
  query id → gold document ids (e.g. HotpotQA supporting-fact titles, MuSiQue
  supporting paragraphs, Qasper evidence spans), pre-resolved to the retrieval
  corpus's doc ids by the caller (the §8.2 chunk↔qrel containment rule is
  applied where the chunks live, not here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from src.analysis.cellspec import CellSpec, CellSpecError

#: charter §8.2 stage cutoffs.
DEFAULT_K_POOL: int = 100
DEFAULT_K_RERANK: int = 10

Qrels = Mapping[str, Mapping[str, int]]
Run = Mapping[str, Mapping[str, float]]


class RetrievalScoringError(ValueError):
    """Invalid Layer-0 input (malformed qrels/run, bad k, illegal cell key)."""


class RanxUnavailableError(RuntimeError):
    """The registered Layer-0 instrument (ranx) failed to import.

    Raised INSTEAD of substituting a hand-rolled recall/nDCG implementation:
    §8.2 names ranx (paired significance in-library), and a substitute would
    score cells with a different implementation under the same column names.
    Mirrors ``src.evaluation.quality.InstrumentUnavailableError``.
    """

    def __init__(self, cause: str) -> None:
        self.cause = cause
        super().__init__(
            "Layer-0 instrument unavailable: ranx failed to import "
            f"({cause}). ranx is pinned in requirements.txt (ranx==0.3.21; "
            "charter §8.2 names it) — a failing import means this environment "
            "was not installed from requirements.txt. This scorer does not "
            "degrade to a hand-rolled substitute."
        )


def _import_ranx() -> Any:
    """Lazy, fail-closed import of the registered instrument."""
    try:
        import ranx  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise RanxUnavailableError(str(exc)) from exc
    return ranx


# ---------------------------------------------------------------------------
# Qrels constructors
# ---------------------------------------------------------------------------


def qrels_from_manifest(manifest: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Block-level qrels from a query-manifest dict (src.data.manifest).

    Each question's assigned corpus block is its (single) relevant document,
    with doc id ``block-<block_id>`` — matching how a block-serving retriever
    must key its runs. Fails loud on a manifest without assignments.
    """
    q2b = manifest.get("question_to_block")
    if not isinstance(q2b, Mapping) or not q2b:
        raise RetrievalScoringError(
            "manifest has no non-empty 'question_to_block' mapping — cannot "
            "derive qrels (is this a src.data.manifest.build_manifest output?)"
        )
    qrels: dict[str, dict[str, int]] = {}
    for query_id, block_id in q2b.items():
        if not isinstance(query_id, str) or not query_id:
            raise RetrievalScoringError(
                f"question_to_block key {query_id!r} is not a non-empty string"
            )
        if isinstance(block_id, bool) or not isinstance(block_id, int):
            raise RetrievalScoringError(
                f"question_to_block[{query_id!r}]={block_id!r} is not an int block id"
            )
        qrels[query_id] = {f"block-{block_id}": 1}
    return qrels


def qrels_from_gold_docs(
    gold_docs: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, int]]:
    """Qrels from loader metadata: query id -> gold document ids (binary rel).

    The §8.2 containment rule (answer-span / ≥50%-token coverage) must already
    have been applied by the caller when mapping gold annotations onto corpus
    chunk ids — this constructor only validates shape and fails loud on a
    query with zero gold documents (a qrel row that can never score).
    """
    if not gold_docs:
        raise RetrievalScoringError("gold_docs is empty — no queries to score")
    qrels: dict[str, dict[str, int]] = {}
    for query_id, docs in gold_docs.items():
        if not isinstance(query_id, str) or not query_id:
            raise RetrievalScoringError(
                f"gold_docs key {query_id!r} is not a non-empty string"
            )
        if isinstance(docs, str) or not isinstance(docs, Sequence) or not docs:
            raise RetrievalScoringError(
                f"gold_docs[{query_id!r}] must be a non-empty sequence of doc "
                f"ids, got {docs!r}"
            )
        row: dict[str, int] = {}
        for doc in docs:
            if not isinstance(doc, str) or not doc:
                raise RetrievalScoringError(
                    f"gold_docs[{query_id!r}] contains a non-string doc id: {doc!r}"
                )
            row[doc] = 1
        qrels[query_id] = row
    return qrels


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_qrels(qrels: Qrels) -> None:
    if not qrels:
        raise RetrievalScoringError("qrels are empty — nothing to score")
    for query_id, docs in qrels.items():
        if not isinstance(query_id, str) or not query_id:
            raise RetrievalScoringError(f"qrels key {query_id!r} is not a non-empty string")
        if not isinstance(docs, Mapping) or not docs:
            raise RetrievalScoringError(
                f"qrels[{query_id!r}] must be a non-empty mapping doc_id -> relevance"
            )
        n_relevant = 0
        for doc_id, rel in docs.items():
            if not isinstance(doc_id, str) or not doc_id:
                raise RetrievalScoringError(
                    f"qrels[{query_id!r}] doc id {doc_id!r} is not a non-empty string"
                )
            if isinstance(rel, bool) or not isinstance(rel, int) or rel < 0:
                raise RetrievalScoringError(
                    f"qrels[{query_id!r}][{doc_id!r}]={rel!r} must be an int >= 0"
                )
            n_relevant += int(rel > 0)
        if n_relevant == 0:
            raise RetrievalScoringError(
                f"qrels[{query_id!r}] has no relevant document (all rel=0) — "
                "an unscoreable qrel row is a join bug, not a zero"
            )


def _validate_run(name: str, run: Run, qrels: Qrels) -> None:
    if not isinstance(run, Mapping) or not run:
        raise RetrievalScoringError(f"{name} run is empty — nothing was retrieved")
    missing = sorted(set(qrels) - set(run))
    if missing:
        raise RetrievalScoringError(
            f"{name} run is missing {len(missing)} qrels query id(s) "
            f"(first: {missing[:3]}) — a missing query is a JOIN bug, not a "
            "zero-recall retrieval (fail closed)"
        )
    for query_id in qrels:
        docs = run[query_id]
        if not isinstance(docs, Mapping):
            raise RetrievalScoringError(
                f"{name} run[{query_id!r}] must be a mapping doc_id -> score"
            )
        for doc_id, score in docs.items():
            if not isinstance(doc_id, str) or not doc_id:
                raise RetrievalScoringError(
                    f"{name} run[{query_id!r}] doc id {doc_id!r} is not a non-empty string"
                )
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise RetrievalScoringError(
                    f"{name} run[{query_id!r}][{doc_id!r}]={score!r} must be numeric"
                )


def _check_k(name: str, k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise RetrievalScoringError(f"{name}={k!r} must be an int >= 1")


# ---------------------------------------------------------------------------
# Stage scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageRuns:
    """The three stage-tagged runs of one dataset × retriever variant.

    - ``pool``: the FIRST-STAGE candidate pool ranking (pre-rerank, ≥ k_pool
      candidates where available).
    - ``reranked``: the post-reranker ordering.
    - ``served``: the documents actually placed in the served context, ranked
      in served order (what the model saw).
    """

    pool: Run
    reranked: Run
    served: Run


@dataclass(frozen=True)
class StageScores:
    """One dataset × retriever variant's §8.2 stage-tagged score row."""

    n_queries: int
    k_pool: int
    k_rerank: int
    k_served: int
    pool_recall: float
    reranked_ndcg: float
    reranked_mrr: float | None
    served_recall: float
    complete_evidence: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_queries": self.n_queries,
            "k_pool": self.k_pool,
            "k_rerank": self.k_rerank,
            "k_served": self.k_served,
            f"pool_recall_at_{self.k_pool}": self.pool_recall,
            f"reranked_ndcg_at_{self.k_rerank}": self.reranked_ndcg,
            f"reranked_mrr_at_{self.k_rerank}": self.reranked_mrr,
            f"served_recall_at_{self.k_served}": self.served_recall,
            f"complete_evidence_at_{self.k_served}": self.complete_evidence,
        }


def _is_single_gold(qrels: Qrels) -> bool:
    """MRR@10 is registered for single-gold sets only (§8.2)."""
    return all(
        sum(1 for rel in docs.values() if rel > 0) == 1 for docs in qrels.values()
    )


def _ranx_evaluate(ranx: Any, qrels: Qrels, run: Run, metrics: list[str]) -> dict[str, float]:
    """One ranx evaluation call, normalized to a metric->float mapping."""
    result = ranx.evaluate(
        ranx.Qrels({q: dict(d) for q, d in qrels.items()}),
        ranx.Run({q: {doc: float(s) for doc, s in docs.items()} for q, docs in run.items()}),
        metrics,
    )
    if isinstance(result, Mapping):
        return {m: float(result[m]) for m in metrics}
    if len(metrics) != 1:  # pragma: no cover - defensive against API drift
        raise RetrievalScoringError(
            f"ranx returned a scalar for multi-metric request {metrics}"
        )
    return {metrics[0]: float(result)}


def complete_evidence_at_k(qrels: Qrels, served: Run, k: int) -> float:
    """Charter §8.2 (AMENDED 2026-08-01) complete-evidence@k.

    Fraction of queries whose EVERY relevant document appears in the served
    run's top-k. This is the multi-hop ceiling metric: recall@k averages over
    hops and hides unanswerability, complete-evidence does not. CAGE-native —
    defined by the charter, not by ranx — hence computed directly.
    """
    _validate_qrels(qrels)
    _validate_run("served", served, qrels)
    _check_k("k", k)
    n_complete = 0
    for query_id, docs in qrels.items():
        relevant = {doc for doc, rel in docs.items() if rel > 0}
        ranked = sorted(served[query_id].items(), key=lambda kv: (-float(kv[1]), kv[0]))
        top_k = {doc for doc, _ in ranked[:k]}
        n_complete += int(relevant <= top_k)
    return n_complete / len(qrels)


def score_stages(
    qrels: Qrels,
    runs: StageRuns,
    *,
    k_served: int,
    k_pool: int = DEFAULT_K_POOL,
    k_rerank: int = DEFAULT_K_RERANK,
    multi_hop: bool = False,
) -> StageScores:
    """Score one dataset × retriever variant's three stages (§8.2).

    ``k_served`` is the arm's served-context depth (what the model actually
    saw) — there is no defensible default, so it is required. ``multi_hop``
    turns on complete-evidence@k_served (HotpotQA/MuSiQue/Qasper); on
    single-gold sets MRR@k_rerank is emitted instead (``None`` otherwise —
    a labeled absence, never a silent zero).
    """
    _validate_qrels(qrels)
    _check_k("k_pool", k_pool)
    _check_k("k_rerank", k_rerank)
    _check_k("k_served", k_served)
    _validate_run("pool", runs.pool, qrels)
    _validate_run("reranked", runs.reranked, qrels)
    _validate_run("served", runs.served, qrels)

    ranx = _import_ranx()
    pool_scores = _ranx_evaluate(ranx, qrels, runs.pool, [f"recall@{k_pool}"])
    rerank_metrics = [f"ndcg@{k_rerank}"]
    single_gold = _is_single_gold(qrels)
    if single_gold:
        rerank_metrics.append(f"mrr@{k_rerank}")
    rerank_scores = _ranx_evaluate(ranx, qrels, runs.reranked, rerank_metrics)
    served_scores = _ranx_evaluate(ranx, qrels, runs.served, [f"recall@{k_served}"])

    return StageScores(
        n_queries=len(qrels),
        k_pool=k_pool,
        k_rerank=k_rerank,
        k_served=k_served,
        pool_recall=pool_scores[f"recall@{k_pool}"],
        reranked_ndcg=rerank_scores[f"ndcg@{k_rerank}"],
        reranked_mrr=(
            rerank_scores[f"mrr@{k_rerank}"] if single_gold else None
        ),
        served_recall=served_scores[f"recall@{k_served}"],
        complete_evidence=(
            complete_evidence_at_k(qrels, runs.served, k_served) if multi_hop else None
        ),
    )


# ---------------------------------------------------------------------------
# Per-cell join (keyed by CellSpec row key)
# ---------------------------------------------------------------------------


def _validate_row_key(row_key: str) -> CellSpec:
    """Parse + round-trip one CellSpec row key (RESULTS_LAYOUT §2 discipline).

    Layer 0 is defined only for cells whose arm consumes a ranked retrieved
    list (§8.2) — a gold/corpus cell (retriever='none') has evidence quality
    by construction and scoring it here would be a category error.
    """
    parts = row_key.split("|")
    if len(parts) < 7:
        raise RetrievalScoringError(
            f"row key {row_key!r} has {len(parts)} segment(s); expected the 7 "
            "axes (arm|retriever|policy|topology|engine|model|family) + "
            "optional coords"
        )
    budget_r: float | None = None
    rate_frac: float | None = None
    for coord in parts[7:]:
        try:
            if coord.startswith("lam"):
                rate_frac = float(coord[3:])
            elif coord.startswith("r"):
                budget_r = float(coord[1:])
            else:
                raise RetrievalScoringError(
                    f"row key {row_key!r}: unrecognized coord segment {coord!r}"
                )
        except ValueError as exc:
            raise RetrievalScoringError(
                f"row key {row_key!r}: malformed coord segment {coord!r}: {exc}"
            ) from exc
    try:
        spec = CellSpec(*parts[:7], budget_r=budget_r, rate_frac=rate_frac)  # type: ignore[arg-type]
    except (CellSpecError, ValueError) as exc:
        raise RetrievalScoringError(
            f"row key {row_key!r} is not a valid CellSpec: {exc}"
        ) from exc
    if spec.to_row_key() != row_key:
        raise RetrievalScoringError(
            f"row key {row_key!r} does not round-trip "
            f"(canonical: {spec.to_row_key()!r})"
        )
    if spec.retriever == "none":
        raise RetrievalScoringError(
            f"cell {row_key!r} has retriever='none' — Layer 0 is defined only "
            "for arms consuming a ranked retrieved list (§8.2); gold/corpus "
            "arms have evidence quality by construction"
        )
    return spec


def score_cells(
    qrels: Qrels,
    runs_by_cell: Mapping[str, StageRuns],
    *,
    k_served: int,
    k_pool: int = DEFAULT_K_POOL,
    k_rerank: int = DEFAULT_K_RERANK,
    multi_hop: bool = False,
) -> pd.DataFrame:
    """Stage-tagged scores JOINED per cell, keyed by the CellSpec row key.

    One output row per cell (§8.2: "cells join the columns by reference"):
    ``row_key`` + identity axes + the ``StageScores.to_dict()`` columns. Every
    key is validated through a CellSpec round-trip; unknown/illegal keys and
    non-retrieval cells FAIL LOUD, listed together, none skipped.
    """
    if not runs_by_cell:
        raise RetrievalScoringError("runs_by_cell is empty — no cells to join")
    problems: list[str] = []
    specs: dict[str, CellSpec] = {}
    for row_key in runs_by_cell:
        try:
            specs[row_key] = _validate_row_key(row_key)
        except RetrievalScoringError as exc:
            problems.append(str(exc))
    if problems:
        raise RetrievalScoringError(
            f"{len(problems)} illegal cell key(s):\n" + "\n".join(problems)
        )
    rows: list[dict[str, Any]] = []
    for row_key in sorted(runs_by_cell):
        spec = specs[row_key]
        scores = score_stages(
            qrels,
            runs_by_cell[row_key],
            k_served=k_served,
            k_pool=k_pool,
            k_rerank=k_rerank,
            multi_hop=multi_hop,
        )
        rows.append(
            {
                "row_key": row_key,
                "arm": spec.arm,
                "retriever": spec.retriever,
                "engine": spec.engine,
                "model": spec.model,
                "family": spec.family,
                **scores.to_dict(),
            }
        )
    return pd.DataFrame(rows)
