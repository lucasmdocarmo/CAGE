#!/usr/bin/env python3
"""Order:     preflight — charter-MANDATORY BEFORE any campaign dispatch (§7.2(a))
Objective: Produce the §7.2(a) offline retrieval gate table (bm25 vs dense vs
           rrf, first-stage pool recall@100) that decides registered contrast #6
Cloud:     local

$0 by construction — no GPU, no LLM; the dense leg loads the pinned
SentenceTransformers checkpoint on CPU.

The charter's self-deciding retriever gate (PUBLICATION.md §7.2, S3 pattern,
metric + ε PINNED 2026-08-02): the offline table is MANDATORY, and the gate
metric is **first-stage pool recall@100 per the §8.2 stage-tagged
definitions** — one stage, one metric. nDCG@10 / MRR@10 belong to the
reranked stage and are NEVER gate inputs; this producer emits the gate metric
only (reranked-stage context columns are Layer 0's job once rerankers run —
src.analysis.l0_retrieval.score_stages). Registered contrast #6
(src.analysis.stats.families) "fires only if the §7.2 offline gate shows a
≥5pp pool-recall@100 gap" — the artifact written here is the thing that
sentence reads.

Dataset scope (derived, recorded): §7.2(a) runs "per dataset, against
gold-passage qrels"; §8.2 names the qrels sources — SQuAD gold paragraph,
HotpotQA supporting facts, MuSiQue supporting paragraphs, Qasper evidence
spans — and sizes the layer at "≈ 4 datasets × 3 variants … Feeds the §7.2
gate table". Those four are D5 items 1-4 (the quality-instrumented QA
datasets): squad_v2, hotpotqa, musique, qasper. RULER (instrument), SCBench
(external slice), ShareGPT (load donor) and CRAG (cite-only) carry no
gold-passage-qrels role and are not gate datasets.

Retrieval systems — the three §7.2 variants over IDENTICAL chunk ids (the
context-paragraph chunk store from src.orchestration.ir.build_corpus_from_contexts,
keyed by stable_text_id; at this granularity a chunk IS a gold paragraph, so
the §8.2 chunk↔qrel containment rule reduces to identity):

- ``bm25``  — src.orchestration.ir.BM25IRIndex (Robertson & Zaragoza 2009);
- ``dense`` — src.orchestration.ir.FaissIRIndex, default intfloat/e5-large-v2,
  revision CONSUMED from the frozen registration artifact
  (MyDocs/registration/freeze_resolutions.json →
  INSTRUMENT_REVISIONS.dense_retriever — the retriever's OWN slot; the
  artifact's existing INSTRUMENT_REVISIONS.embedding entry registers a
  DIFFERENT instrument, the quality module's similarity embedder
  (src.evaluation.quality, enforced via CAGE_EMBEDDING_REVISION, §9 prereg
  content), and is never consumed, matched against, or steered-at here — one
  slot per instrument, or two instruments end up sharing one registered pin);
  a missing artifact/entry, or an entry pinning a DIFFERENT model than the one
  requested, REFUSES — a revision hash belongs to one repo, and stamping
  another model's hash onto this leg would be fabricated provenance;
- ``rrf``   — src.orchestration.ir.rrf_fuse of the bm25 + dense pools
  (Cormack et al. 2009, k=60).

Scoring reuses src.analysis.l0_retrieval (qrels_from_gold_docs + pool_recall:
same validation, same registered instrument ranx, same metric string) — never
a hand-rolled recall.

Fail-closed doctrine: unknown/duplicate datasets, an absent freeze pin, a
model↔pin mismatch, empty corpora/qrels, duplicate query ids, and a gold id
outside the chunk store all REFUSE with the missing piece named. Queries with
no gold evidence (e.g. Qasper questions whose annotators supplied none) are
SKIPPED and counted in the artifact — a labeled absence, never a silent drop
and never a fabricated zero; the gold route is metadata-KEY-sensitive, so an
evidence_doc_ids / supporting_titles key present-but-EMPTY is that recorded
absence and never falls through to whole-context gold (that fallback belongs
only to loaders that emit neither key, i.e. SQuAD v2). An existing output
REFUSES without --force.

Test-only seams (unit tests must not download datasets or load real
encoders): ``examples_by_dataset_for_tests_only`` injects tiny synthetic
example lists, and ``dense_index_factory_for_tests_only`` injects a
deterministic dense-leg stand-in. BOTH are explicit keyword parameters named
for what they are; the production path (factory=None) always resolves the
freeze pin and builds the real FaissIRIndex, and an injected dense leg stamps
``dense_leg_test_only: true`` plus a None-revision provenance row into the
artifact so a test table can never masquerade as the registered gate table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.l0_retrieval import (  # noqa: E402
    DEFAULT_K_POOL,
    RanxUnavailableError,
    RetrievalScoringError,
    pool_recall,
    qrels_from_gold_docs,
)
from src.data.loader import DatasetUnavailableError, gold_only  # noqa: E402
from src.orchestration.ir import (  # noqa: E402
    BM25IRIndex,
    FaissIRIndex,
    IRDocument,
    build_corpus_from_contexts,
    corpus_doc_ids_sha1,
    rrf_fuse,
    stable_text_id,
)

__all__ = [
    "GateTableError",
    "SCHEMA",
    "GATE_DATASETS",
    "GATE_EPSILON_PP",
    "GATE_METRIC",
    "VARIANTS",
    "FREEZE_DENSE_KEY",
    "resolve_dense_retriever_revision",
    "gate_decision",
    "build_retrieval_gate_table",
]

SCHEMA: Final[str] = "retrieval-gate-table-v1"
#: §7.2(a): the ONE gate metric (first-stage pool recall@100, §8.2 stage tags).
GATE_METRIC: Final[str] = "first_stage_pool_recall_at_100"
#: §7.2(b): ε = 5 percentage points, either direction (PINNED 2026-08-02).
GATE_EPSILON_PP: Final[float] = 5.0
#: D5 items 1-4 — the four gold-passage-qrels datasets (derivation: module doc).
GATE_DATASETS: Final[tuple[str, ...]] = ("squad_v2", "hotpotqa", "musique", "qasper")
#: The three §7.2 retriever variants, table row order.
VARIANTS: Final[tuple[str, ...]] = ("bm25", "dense", "rrf")
#: Signed pairwise deltas rendered in the table (a − b, percentage points).
DELTA_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("dense", "bm25"),
    ("rrf", "bm25"),
    ("rrf", "dense"),
)
#: Percentage-point deltas are rounded here before the ε comparison: binary
#: float dust (e.g. (0.90-0.85)*100 = 4.999999999999993) must never flip the
#: registered 5.0pp boundary. Six decimals is far below any honest recall
#: resolution and far above double-precision noise.
DELTA_DECIMALS: Final[int] = 6

DEFAULT_EMBEDDING_MODEL: Final[str] = "intfloat/e5-large-v2"
DEFAULT_SEED: Final[int] = 42
DEFAULT_SPLIT: Final[str] = "validation"
#: BM25/RRF constants match the ir.py defaults (documented there).
BM25_K1: Final[float] = 1.5
BM25_B: Final[float] = 0.75
RRF_K: Final[int] = 60

DEFAULT_OUT: Final[Path] = REPO_ROOT / "results" / "preflight" / "retrieval_gate_table.json"
#: Same freeze-artifact resolution chain as build_predicate_table.py (T6.1).
FREEZE_ENV_VAR: Final[str] = "CAGE_FREEZE_RESOLUTIONS"
DEFAULT_FREEZE_FILE: Final[Path] = (
    REPO_ROOT / "MyDocs" / "registration" / "freeze_resolutions.json"
)
#: The dense leg's OWN freeze slot. Deliberately NOT the artifact's existing
#: 'embedding' key: that one registers the QUALITY module's similarity
#: embedder (src.evaluation.quality, CAGE_EMBEDDING_REVISION) — a different
#: instrument whose registered pin this producer must neither consume nor
#: instruct anyone to edit.
FREEZE_DENSE_KEY: Final[str] = "dense_retriever"

#: Provenance labels for the dense leg's revision pin.
PIN_SOURCE_FREEZE: Final[str] = "freeze-file"
PIN_SOURCE_TEST_INJECTION: Final[str] = "dense_index_factory_for_tests_only"


class GateTableError(RuntimeError):
    """Any refusal in the gate-table build (fail loud, missing piece named)."""


# --------------------------------------------------------------------------- #
# Freeze-artifact pin resolution (fail-closed)
# --------------------------------------------------------------------------- #


def resolve_dense_retriever_revision(
    freeze_file: Path, embedding_model: str
) -> dict[str, Any]:
    """Resolve the dense leg's pinned HF revision from the freeze artifact.

    Consumes ``INSTRUMENT_REVISIONS.dense_retriever`` of
    MyDocs/registration/freeze_resolutions.json — the retriever's OWN slot.
    The artifact's ``INSTRUMENT_REVISIONS.embedding`` entry is a DIFFERENT
    instrument's registered pin (the quality module's similarity embedder,
    enforced via CAGE_EMBEDDING_REVISION): it is never consumed here, never a
    fallback, and no refusal below may steer the owner into editing it —
    overwriting a registered quality-instrument pin to unblock the retrieval
    gate would corrupt §9 prereg provenance. Refuses (named) when the
    artifact is missing/unreadable/invalid, when the entry or its
    model/revision fields are absent, and when the entry pins a DIFFERENT
    model than ``embedding_model`` — a commit hash is only meaningful for the
    repo it was resolved from, so a mismatch must never silently re-pin.
    """
    slot = f"INSTRUMENT_REVISIONS.{FREEZE_DENSE_KEY}"
    fix = (
        f"point --freeze-file / ${FREEZE_ENV_VAR} at the frozen registration "
        f"artifact (default {DEFAULT_FREEZE_FILE}); the §7.2 gate table never "
        "runs its dense leg with an unpinned retriever revision"
    )
    if not freeze_file.is_file():
        raise GateTableError(
            f"{freeze_file}: freeze artifact missing — the dense leg's "
            f"retriever revision pin ({slot}) cannot be consumed; {fix}"
        )
    try:
        data = json.loads(freeze_file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GateTableError(
            f"{freeze_file}: freeze artifact exists but cannot be read: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise GateTableError(
            f"{freeze_file}: freeze artifact is not valid JSON: {exc}"
        ) from exc
    revisions = data.get("INSTRUMENT_REVISIONS") if isinstance(data, dict) else None
    if not isinstance(revisions, Mapping):
        raise GateTableError(
            f"{freeze_file}: no INSTRUMENT_REVISIONS mapping — the artifact "
            f"carries no instrument pins to consume; {fix}"
        )
    entry = revisions.get(FREEZE_DENSE_KEY)
    if not isinstance(entry, Mapping):
        raise GateTableError(
            f"{freeze_file}: INSTRUMENT_REVISIONS has no "
            f"{FREEZE_DENSE_KEY!r} entry — the dense leg's revision pin is "
            f"absent. Fix: resolve the frozen HF commit hash for "
            f"{embedding_model!r} and ADD it as {slot} "
            f"{{model, revision, resolved}}. Do NOT repurpose "
            f"INSTRUMENT_REVISIONS.embedding for this — that slot registers "
            f"the quality module's similarity embedder "
            f"(CAGE_EMBEDDING_REVISION), a different instrument whose pin "
            f"must stay untouched ({fix})"
        )
    model = entry.get("model")
    revision = entry.get("revision")
    if not isinstance(model, str) or not model.strip():
        raise GateTableError(
            f"{freeze_file}: {slot}.model is "
            f"{model!r} — a pin without a model pins nothing"
        )
    if not isinstance(revision, str) or not revision.strip():
        raise GateTableError(
            f"{freeze_file}: {slot}.revision is "
            f"{revision!r} — the dense leg's revision pin is absent; resolve "
            f"and record the HF commit hash in the {slot} entry"
        )
    if model != embedding_model:
        raise GateTableError(
            f"dense-retriever model/pin mismatch: the gate table was asked "
            f"to run dense={embedding_model!r} but {freeze_file} pins "
            f"{slot}.model={model!r} "
            f"(revision {revision}). A revision hash belongs to ONE repo — "
            f"refusing to stamp {model!r}'s hash onto {embedding_model!r}. "
            f"Fix: record the frozen revision for {embedding_model!r} in the "
            f"{slot} entry, or pass --embedding-model {model} to run the "
            f"retriever that entry pins"
        )
    return {
        "model": model,
        "revision": revision,
        "source": PIN_SOURCE_FREEZE,
        "freeze_file": str(freeze_file.resolve()),
    }


# --------------------------------------------------------------------------- #
# Gate decision (§7.2(b) trigger, pure)
# --------------------------------------------------------------------------- #


def gate_decision(
    recall_by_variant: Mapping[str, float], *, epsilon_pp: float = GATE_EPSILON_PP
) -> dict[str, Any]:
    """Charter §7.2(b) trigger for ONE dataset's {variant: recall@100} row.

    Returns the signed pairwise deltas in percentage points (rounded to
    ``DELTA_DECIMALS`` — see the constant's comment), the max absolute delta,
    and ``fires_contrast_6`` = (max |Δ| ≥ ε). "Either direction" per charter,
    hence the absolute value; ε defaults to the registered 5.0pp.
    """
    if not isinstance(epsilon_pp, (int, float)) or isinstance(epsilon_pp, bool) or epsilon_pp <= 0:
        raise GateTableError(f"epsilon_pp must be a number > 0, got {epsilon_pp!r}")
    if set(recall_by_variant) != set(VARIANTS):
        raise GateTableError(
            f"gate_decision needs recall for exactly the variants "
            f"{list(VARIANTS)}, got {sorted(recall_by_variant)}"
        )
    for variant, value in recall_by_variant.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GateTableError(
                f"recall[{variant!r}]={value!r} is not a number"
            )
        if not (0.0 <= float(value) <= 1.0):
            raise GateTableError(
                f"recall[{variant!r}]={value!r} is outside [0, 1] — not a recall"
            )
    deltas = {
        f"{a}_minus_{b}": round(
            (float(recall_by_variant[a]) - float(recall_by_variant[b])) * 100.0,
            DELTA_DECIMALS,
        )
        for a, b in DELTA_PAIRS
    }
    max_abs = max(abs(d) for d in deltas.values())
    return {
        "pairwise_delta_pp": deltas,
        "max_abs_delta_pp": max_abs,
        "epsilon_pp": float(epsilon_pp),
        "fires_contrast_6": max_abs >= float(epsilon_pp),
    }


# --------------------------------------------------------------------------- #
# Qrels derivation (gold evidence from the loaders)
# --------------------------------------------------------------------------- #


def _gold_context_texts(example: Any) -> list[str]:
    """The example's gold paragraph texts, per its loader's metadata.

    KEY-sensitive, not truthiness-sensitive: a gold-evidence metadata key that
    is PRESENT with an empty value is a recorded absence — the loader looked
    for gold and found none — and returns [] so the caller counts a skip.
    Falling through to "the whole context is gold" there would fabricate
    qrels (a real Qasper no-evidence/unanswerable question carries
    ``evidence_doc_ids=[]`` beside the full paper context). Routes:

    - ``metadata["evidence_doc_ids"]`` present (Qasper): the precise route —
      the ids index the exact context docs holding the human gold evidence
      (the loader documents that title-prefix matching over-selects on
      duplicate section names); empty ⇒ recorded absence.
    - ``metadata["supporting_titles"]`` present (HotpotQA/MuSiQue): the
      ``src.data.loader.gold_only`` title filter; empty ⇒ recorded absence —
      gold_only's whole-context fallback exists for loaders WITHOUT the key,
      never for a recorded empty list.
    - neither key (SQuAD v2): the context already IS its gold paragraph.
    """
    metadata = getattr(example, "metadata", None) or {}
    context = getattr(example, "context", None) or []
    if "evidence_doc_ids" in metadata:
        return [
            context[i]
            for i in (metadata.get("evidence_doc_ids") or [])
            if isinstance(i, int) and 0 <= i < len(context) and context[i]
        ]
    if "supporting_titles" in metadata:
        if not metadata.get("supporting_titles"):
            return []  # recorded absence — a counted skip, never fabricated gold
        return [t for t in gold_only(example) if t]
    # No gold-evidence metadata keys at all: the loader's context IS the gold
    # (SQuAD v2's single gold paragraph). gold_only returns it unchanged.
    return [t for t in gold_only(example) if t]


def _qrels_sha1(qrels: Mapping[str, Mapping[str, int]]) -> str:
    """Content fingerprint of the gold mapping (sorted qid/doc-id pairs)."""
    joined = "\n".join(
        f"{qid}\t{doc_id}" for qid in sorted(qrels) for doc_id in sorted(qrels[qid])
    )
    return sha1(joined.encode("utf-8")).hexdigest()


def _build_qrels(
    dataset: str, examples: Sequence[Any], corpus_doc_ids: set[str]
) -> tuple[dict[str, dict[str, int]], dict[str, str], int]:
    """(qrels, query id -> question text, n skipped-no-gold) for one dataset.

    Refuses on duplicate query ids and on a gold id absent from the chunk
    store (a containment bug, never a zero); an example with NO gold evidence
    is skipped and counted — absence stays absence.
    """
    gold_map: dict[str, list[str]] = {}
    questions: dict[str, str] = {}
    seen: set[str] = set()
    n_skipped = 0
    for example in examples:
        query_id = str(getattr(example, "id", ""))
        if not query_id:
            raise GateTableError(
                f"dataset {dataset!r}: an example has no id — qrels need a "
                "stable query key"
            )
        if query_id in seen:
            raise GateTableError(
                f"dataset {dataset!r}: duplicate query id {query_id!r} — one "
                "qrel row per query; a colliding id silently merges two "
                "queries' gold sets"
            )
        seen.add(query_id)
        gold_ids = list(
            dict.fromkeys(stable_text_id(t) for t in _gold_context_texts(example))
        )
        if not gold_ids:
            n_skipped += 1
            continue
        missing = [g for g in gold_ids if g not in corpus_doc_ids]
        if missing:
            raise GateTableError(
                f"dataset {dataset!r}, query {query_id!r}: {len(missing)} gold "
                f"doc id(s) absent from the chunk store (first: {missing[:2]}) "
                "— a chunk↔qrel containment bug (§8.2), not a zero-recall row"
            )
        gold_map[query_id] = gold_ids
        questions[query_id] = str(getattr(example, "question", "") or "")
    if not gold_map:
        raise GateTableError(
            f"dataset {dataset!r}: no query has gold evidence "
            f"({n_skipped} skipped) — an all-skip dataset cannot gate anything"
        )
    return qrels_from_gold_docs(gold_map), questions, n_skipped


# --------------------------------------------------------------------------- #
# The three first-stage systems over identical chunk ids
# --------------------------------------------------------------------------- #


def _build_real_dense_index(
    documents: Sequence[IRDocument], *, embedding_model: str, device: str
) -> FaissIRIndex:
    """The production dense leg: the real pinned SentenceTransformers model."""
    index = FaissIRIndex(embedding_model=embedding_model, device=device)
    index.build(documents)
    return index


def _score_dataset(
    dataset: str,
    examples: Sequence[Any],
    *,
    dense_index_factory: Callable[[str, Sequence[IRDocument]], Any] | None,
    embedding_model: str,
    device: str,
) -> dict[str, Any]:
    """One per-dataset gate-table row (corpus → qrels → 3 runs → decision)."""
    if not examples:
        raise GateTableError(f"dataset {dataset!r}: loader returned 0 examples")
    documents = build_corpus_from_contexts(examples, dataset_name=dataset)
    if not documents:
        raise GateTableError(
            f"dataset {dataset!r}: chunk store is empty (no non-empty "
            "contexts) — nothing to retrieve from"
        )
    qrels, questions, n_skipped = _build_qrels(
        dataset, examples, {d.doc_id for d in documents}
    )

    bm25 = BM25IRIndex(k1=BM25_K1, b=BM25_B)
    bm25.build(documents)
    if dense_index_factory is not None:
        dense = dense_index_factory(dataset, documents)
    else:
        dense = _build_real_dense_index(
            documents, embedding_model=embedding_model, device=device
        )

    runs: dict[str, dict[str, dict[str, float]]] = {v: {} for v in VARIANTS}
    for query_id, question in questions.items():
        bm25_hits = bm25.search(question, top_k=DEFAULT_K_POOL)
        dense_hits = dense.search(question, top_k=DEFAULT_K_POOL)
        fused_hits = rrf_fuse(
            [bm25_hits, dense_hits],
            k=RRF_K,
            names=("bm25", "dense"),
            top_k=DEFAULT_K_POOL,
        )
        # A query with zero hits keeps an EMPTY run row: a measured
        # zero-recall retrieval, never a dropped query (l0 validation would
        # treat a missing qid as a join bug — correctly).
        runs["bm25"][query_id] = {h.doc_id: float(h.score) for h in bm25_hits}
        runs["dense"][query_id] = {h.doc_id: float(h.score) for h in dense_hits}
        runs["rrf"][query_id] = {h.doc_id: float(h.score) for h in fused_hits}

    recalls = {
        variant: pool_recall(qrels, runs[variant], k_pool=DEFAULT_K_POOL)
        for variant in VARIANTS
    }
    decision = gate_decision(recalls)
    return {
        "n_examples": len(examples),
        "n_queries_scored": len(questions),
        "n_queries_skipped_no_gold": n_skipped,
        "corpus": {
            "n_docs": len(documents),
            "doc_ids_sha1": corpus_doc_ids_sha1(documents),
        },
        "qrels_sha1": _qrels_sha1(qrels),
        "recall_at_100": {v: recalls[v] for v in VARIANTS},
        **decision,
    }


# --------------------------------------------------------------------------- #
# Artifact assembly + rendering
# --------------------------------------------------------------------------- #


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _render_markdown(artifact: dict[str, Any]) -> str:
    """Human table beside the machine artifact (same numbers, no recompute)."""
    dense = artifact["retrievers"]["dense"]
    lines = [
        "# §7.2(a) offline retrieval gate table",
        "",
        f"- schema: `{artifact['schema']}`  ·  created: {artifact['created_utc']}",
        f"- gate metric: **{artifact['gate_metric']}** "
        f"(ε = {artifact['epsilon_pp']}pp, either direction; §7.2(b))",
        f"- nDCG@10 / MRR@10 are reranked-stage context and are **never** gate "
        f"inputs (§7.2(a)); this producer emits the gate metric only",
        f"- datasets: {', '.join(artifact['datasets'])}  ·  split: "
        f"{artifact['split']}  ·  seed: {artifact['seed']}  ·  max_examples: "
        f"{artifact['max_examples']}",
        f"- dense leg: `{dense['model']}` @ revision `{dense['revision']}` "
        f"(source: {dense['source']})",
        f"- bm25: k1={artifact['retrievers']['bm25']['k1']}, "
        f"b={artifact['retrievers']['bm25']['b']}  ·  rrf: "
        f"k={artifact['retrievers']['rrf']['k']} over bm25+dense",
    ]
    if artifact["dense_leg_test_only"]:
        lines += [
            "",
            "**WARNING: dense leg is a TEST-ONLY injected stand-in — this is "
            "NOT the registered gate table.**",
        ]
    lines += [
        "",
        "| dataset | n_q | bm25 | dense | rrf | dense−bm25 (pp) | "
        "rrf−bm25 (pp) | rrf−dense (pp) | max \\|Δ\\| (pp) | fires #6 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, row in artifact["datasets"].items():
        recall = row["recall_at_100"]
        delta = row["pairwise_delta_pp"]
        lines.append(
            f"| {name} | {row['n_queries_scored']} "
            f"| {recall['bm25']:.4f} | {recall['dense']:.4f} "
            f"| {recall['rrf']:.4f} | {delta['dense_minus_bm25']:+.2f} "
            f"| {delta['rrf_minus_bm25']:+.2f} | {delta['rrf_minus_dense']:+.2f} "
            f"| {row['max_abs_delta_pp']:.2f} "
            f"| {'FIRES' if row['fires_contrast_6'] else 'no'} |"
        )
    overall = artifact["overall"]
    verdict = (
        f"**FIRES** on {', '.join(overall['fired_datasets'])} — run "
        "`retr-fresh · bm25` downstream there, anchor model only (§7.2(b))"
        if overall["fires_contrast_6"]
        else "does **not** fire — |gap| < ε everywhere; no downstream bm25 "
        "cells run and the identical-pools argument closes the question "
        "(§7.2(b))"
    )
    lines += ["", f"Contrast #6 {verdict}.", ""]
    return "\n".join(lines)


def build_retrieval_gate_table(
    *,
    datasets: Sequence[str],
    out_json: Path,
    force: bool = False,
    seed: int = DEFAULT_SEED,
    split: str = DEFAULT_SPLIT,
    max_examples: int | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    freeze_file: Path = DEFAULT_FREEZE_FILE,
    device: str = "cpu",
    examples_by_dataset_for_tests_only: Mapping[str, Sequence[Any]] | None = None,
    dense_index_factory_for_tests_only: Callable[[str, Sequence[IRDocument]], Any] | None = None,
) -> dict[str, Any]:
    """Build + write the §7.2(a) gate table (JSON + sibling .md); returns it.

    Order of refusals (deliberate — cheap checks before any model/dataset
    load): dataset-name validation → output overwrite → freeze revision pin →
    per-dataset load/score. The two ``*_for_tests_only`` seams are documented
    in the module docstring; production callers leave both as None.
    """
    if not datasets:
        raise GateTableError("no datasets requested — the gate table gates nothing")
    unknown = [d for d in datasets if d not in GATE_DATASETS]
    if unknown:
        raise GateTableError(
            f"unknown gate dataset(s) {unknown}: §7.2 gates over the D5 "
            f"gold-passage-qrels datasets {list(GATE_DATASETS)}; "
            "ruler/scbench/sharegpt/crag have no gold-passage-qrels role "
            "(instrument / external slice / load donor / cite-only)"
        )
    if len(set(datasets)) != len(datasets):
        raise GateTableError(f"duplicate dataset(s) in {list(datasets)}")
    if examples_by_dataset_for_tests_only is not None:
        uncovered = [
            d for d in datasets if d not in examples_by_dataset_for_tests_only
        ]
        if uncovered:
            raise GateTableError(
                f"examples_by_dataset_for_tests_only carries no examples for "
                f"{uncovered} — the test seam must cover every requested "
                "dataset (no silent fallback to real loaders in tests)"
            )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise GateTableError(f"seed must be an int, got {seed!r}")

    out_json = Path(out_json)
    if out_json.suffix != ".json":
        raise GateTableError(
            f"--out must end in .json (the .md sibling is derived from it), "
            f"got {out_json}"
        )
    out_md = out_json.with_suffix(".md")
    existing = [str(p) for p in (out_json, out_md) if p.exists()]
    if existing and not force:
        raise GateTableError(
            f"output already exists: {', '.join(existing)} — a silently "
            "rewritten gate table would re-decide a registered trigger; "
            "re-run with --force to overwrite deliberately"
        )

    if dense_index_factory_for_tests_only is None:
        embedding_prov = resolve_dense_retriever_revision(
            Path(freeze_file), embedding_model
        )
        dense_impl = "src.orchestration.ir.FaissIRIndex"
    else:
        # Test seam: the artifact says so LOUDLY (None revision with named
        # source — absence stays absence) and can never pass as registered.
        embedding_prov = {
            "model": "TEST-ONLY injected dense stand-in",
            "revision": None,
            "source": PIN_SOURCE_TEST_INJECTION,
            "freeze_file": None,
        }
        dense_impl = "test-only injected index (dense_index_factory_for_tests_only)"
        print(
            "[build_retrieval_gate_table] WARNING: dense leg injected via "
            "dense_index_factory_for_tests_only — NOT the registered gate table",
            file=sys.stderr,
        )

    rows: dict[str, dict[str, Any]] = {}
    for name in datasets:
        if examples_by_dataset_for_tests_only is not None:
            examples = list(examples_by_dataset_for_tests_only[name])
        else:
            from src.data.loader import get_loader  # lazy: needs `datasets`

            examples = get_loader(name, split=split, seed=seed).load(max_examples)
        rows[name] = _score_dataset(
            name,
            examples,
            dense_index_factory=dense_index_factory_for_tests_only,
            embedding_model=embedding_model,
            device=device,
        )

    fired = [name for name, row in rows.items() if row["fires_contrast_6"]]
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "charter": (
            "PUBLICATION.md §7.2(a) offline retrieval gate; gate metric = "
            "first-stage pool recall@100 (§8.2 stage-tagged definitions); "
            "decides registered contrast #6 (families.py)"
        ),
        "gate_metric": GATE_METRIC,
        "epsilon_pp": GATE_EPSILON_PP,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "split": split,
        "max_examples": max_examples,
        "k_pool": DEFAULT_K_POOL,
        "retrievers": {
            "bm25": {
                "impl": "src.orchestration.ir.BM25IRIndex",
                "k1": BM25_K1,
                "b": BM25_B,
                "tokenizer": "lowercase_whitespace",
            },
            "dense": {"impl": dense_impl, **embedding_prov},
            "rrf": {
                "impl": "src.orchestration.ir.rrf_fuse",
                "k": RRF_K,
                "fused": ["bm25", "dense"],
            },
        },
        "dense_leg_test_only": dense_index_factory_for_tests_only is not None,
        "datasets": rows,
        "overall": {
            "fires_contrast_6": bool(fired),
            "fired_datasets": fired,
            "rule": (
                "fires iff ANY dataset's max |pairwise Δ| ≥ ε; |gap| < ε "
                "everywhere ⇒ no downstream bm25 cells run (§7.2(b))"
            ),
        },
    }
    _atomic_write_text(out_json, json.dumps(artifact, indent=2) + "\n")
    _atomic_write_text(out_md, _render_markdown(artifact))
    return artifact


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Produce the charter-mandatory §7.2(a) offline retrieval "
                    "gate table (bm25 vs dense vs rrf, first-stage pool "
                    "recall@100) deciding registered contrast #6."
    )
    parser.add_argument("--datasets", nargs="+", default=list(GATE_DATASETS),
                        metavar="NAME",
                        help=f"gate datasets (default: all of {list(GATE_DATASETS)})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output JSON path (.md sibling derived); "
                             f"default {DEFAULT_OUT}")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="loader sampling seed, recorded in the artifact "
                             f"(default {DEFAULT_SEED})")
    parser.add_argument("--split", default=DEFAULT_SPLIT,
                        help=f"dataset split (default {DEFAULT_SPLIT!r})")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="cap examples per dataset (default: full split — "
                             "the registered gate run)")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
                        help="dense-leg SentenceTransformers model; must match "
                             f"the freeze artifact's {FREEZE_DENSE_KEY} pin "
                             f"(default {DEFAULT_EMBEDDING_MODEL!r})")
    parser.add_argument("--freeze-file", type=Path, default=None,
                        help="frozen registration artifact carrying "
                             f"INSTRUMENT_REVISIONS.{FREEZE_DENSE_KEY}; default "
                             f"${FREEZE_ENV_VAR}, then {DEFAULT_FREEZE_FILE}")
    parser.add_argument("--device", default="cpu",
                        help="dense-leg encode device (default cpu — the gate "
                             "is a local $0 pass)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing gate table deliberately")
    args = parser.parse_args(argv)

    freeze_file = args.freeze_file
    if freeze_file is not None and not freeze_file.is_file():
        # An EXPLICIT --freeze-file that does not exist is a typo, not an
        # opt-out (same rule as build_predicate_table.py).
        print(
            f"ERROR: --freeze-file {freeze_file} does not exist — fix the "
            f"path, or drop the flag to use ${FREEZE_ENV_VAR} / the repo "
            "default",
            file=sys.stderr,
        )
        return 2
    if freeze_file is None:
        env_path = os.environ.get(FREEZE_ENV_VAR, "").strip()
        freeze_file = Path(env_path) if env_path else DEFAULT_FREEZE_FILE

    try:
        artifact = build_retrieval_gate_table(
            datasets=args.datasets,
            out_json=args.out,
            force=args.force,
            seed=args.seed,
            split=args.split,
            max_examples=args.max_examples,
            embedding_model=args.embedding_model,
            freeze_file=freeze_file,
            device=args.device,
        )
    except (GateTableError, RetrievalScoringError, RanxUnavailableError,
            DatasetUnavailableError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out_json = Path(args.out)
    overall = artifact["overall"]
    print(f"[build_retrieval_gate_table] table : {out_json}")
    print(f"[build_retrieval_gate_table] human : {out_json.with_suffix('.md')}")
    for name, row in artifact["datasets"].items():
        print(
            f"[build_retrieval_gate_table] {name}: max |Δ| = "
            f"{row['max_abs_delta_pp']:.2f}pp -> "
            f"{'FIRES' if row['fires_contrast_6'] else 'no fire'}"
        )
    print(
        "[build_retrieval_gate_table] contrast #6 overall: "
        + ("FIRES on " + ", ".join(overall["fired_datasets"])
           if overall["fires_contrast_6"] else "does not fire (<5pp everywhere)")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
