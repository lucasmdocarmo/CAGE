"""W4.1 — the §7.2(a) offline retrieval gate-table PRODUCER, pinned.

WHAT is pinned and WHY (PUBLICATION.md §7.2; registered contrast #6 in
src.analysis.stats.families reads this artifact):

- REAL first-stage code paths: BM25IRIndex and rrf_fuse run for real on tiny
  synthetic corpora — the bm25/rrf legs are never mocked. Only two seams are
  test-injected, both by their explicit ``*_for_tests_only`` names: the
  example lists (no HF downloads) and the dense leg (a deterministic canned
  stand-in — the production path must resolve the freeze revision pin and
  build the real FaissIRIndex). ranx is stubbed at the sys.modules import
  seam (the tests/test_l0_retrieval.py instrument pattern) with a REAL
  recall@k implementation so the scored means are verifiable by hand.
- Freeze-pin refusals: a missing freeze artifact, a missing
  INSTRUMENT_REVISIONS/dense_retriever entry, a missing revision, and a
  model↔pin MISMATCH all refuse with the missing piece named — the dense leg
  never runs unpinned and never wears another model's commit hash. The slot
  is the retriever's OWN (dense_retriever): the artifact's 'embedding' entry
  registers the QUALITY module's similarity embedder (MiniLM, enforced via
  CAGE_EMBEDDING_REVISION) and is never consumed as a fallback, and the
  refusal text steers the owner to ADD dense_retriever — never to edit the
  registered quality pin.
- Overwrite refusal: an existing table refuses without --force (a silently
  rewritten gate table re-decides a registered trigger).
- Gold route is metadata-KEY-sensitive (repair round): an evidence_doc_ids /
  supporting_titles key present-but-EMPTY is a recorded absence — a counted
  skip, never whole-context fabricated gold; the whole-context route belongs
  only to loaders emitting neither key (SQuAD v2).
- The 5pp boundary: 4.9pp does not fire, 5.0pp fires — including the binary
  float-dust case (0.90 vs 0.85) that naive subtraction would leave at
  4.999999999999993pp.
- Artifact schema: per-dataset {bm25, dense, rrf} recall rows, signed
  pairwise deltas in pp, per-dataset + overall fires_contrast_6, embedding
  model+revision, corpus/qrels fingerprints, seed, timestamp; a test-injected
  dense leg stamps dense_leg_test_only=true with a None-revision provenance
  row (absence stays absence).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_retrieval_gate_table as gate  # noqa: E402
from src.data.loader import CAGExample  # noqa: E402
from src.orchestration.ir import IRDocument, IRHit  # noqa: E402


# ---------------------------------------------------------------------------
# ranx stand-in (import seam, per tests/test_l0_retrieval.py) — REAL recall@k
# ---------------------------------------------------------------------------


class _StubQrels:
    def __init__(self, d: dict[str, dict[str, int]]) -> None:
        self.d = d


class _StubRun:
    def __init__(self, d: dict[str, dict[str, float]]) -> None:
        self.d = d


def _make_real_recall_ranx_stub() -> types.ModuleType:
    """A ranx stand-in whose recall@k is computed FOR REAL (same tie rule as
    l0_retrieval.complete_evidence_at_k: descending score, ascending doc id),
    so the gate table's means are hand-verifiable — not canned constants."""
    stub = types.ModuleType("ranx")
    stub.Qrels = _StubQrels  # type: ignore[attr-defined]
    stub.Run = _StubRun  # type: ignore[attr-defined]

    def evaluate(qrels: _StubQrels, run: _StubRun, metrics: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for metric in metrics:
            assert metric.startswith("recall@"), f"unexpected metric {metric!r}"
            k = int(metric.split("@", 1)[1])
            per_query: list[float] = []
            for query_id, rels in qrels.d.items():
                relevant = {doc for doc, rel in rels.items() if rel > 0}
                ranked = sorted(
                    run.d.get(query_id, {}).items(),
                    key=lambda kv: (-float(kv[1]), kv[0]),
                )
                top_k = {doc for doc, _ in ranked[:k]}
                per_query.append(len(relevant & top_k) / len(relevant))
            out[metric] = sum(per_query) / len(per_query)
        return out

    stub.evaluate = evaluate  # type: ignore[attr-defined]
    return stub


@pytest.fixture()
def ranx_stub(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    stub = _make_real_recall_ranx_stub()
    monkeypatch.setitem(sys.modules, "ranx", stub)
    return stub


# ---------------------------------------------------------------------------
# Tiny synthetic corpora + a TEST-ONLY canned dense stand-in
# ---------------------------------------------------------------------------

# hotpotqa-shaped: q h1 lexically reaches its gold (bm25 hit); q h2 shares no
# token with anything (bm25 zero-hit) -> bm25 recall 0.5, dense 1.0 => 50pp gap.
_HOTPOT_GOLD_1 = "Alpha: the alpha protocol was designed by vance"
_HOTPOT_DISTRACT = "Beta: unrelated filler nobody asked about"
_HOTPOT_GOLD_2 = "Gamma: cryptic glyphs qqq www eee"

_MUSIQUE_GOLD = "Delta: the delta river is in wonderland"


def _hotpot_examples() -> list[CAGExample]:
    return [
        CAGExample(
            id="h1",
            question="who designed the alpha protocol",
            context=[_HOTPOT_GOLD_1, _HOTPOT_DISTRACT],
            answer="vance",
            metadata={"supporting_titles": ["Alpha"]},
        ),
        CAGExample(
            id="h2",
            question="zzz yyy xxx",
            context=[_HOTPOT_GOLD_2, _HOTPOT_DISTRACT],
            answer="glyphs",
            metadata={"supporting_titles": ["Gamma"]},
        ),
    ]


def _musique_examples() -> list[CAGExample]:
    return [
        CAGExample(
            id="m1",
            question="where is the delta river",
            context=[_MUSIQUE_GOLD],
            answer="wonderland",
            metadata={"supporting_titles": ["Delta"]},
        ),
        # No context at all -> no gold evidence -> a COUNTED skip, never a row.
        CAGExample(
            id="m2",
            question="question with no gold",
            context=[],
            answer="",
            metadata={},
        ),
    ]


# qasper-shaped, REAL loader metadata shape (src/data/loader.py QasperLoader):
# EVERY example carries the evidence_doc_ids and supporting_titles KEYS; a
# no-evidence/unanswerable question carries them EMPTY beside the FULL paper
# context — a recorded absence, never "the whole paper is gold".
_QASPER_INTRO = "Introduction: transformers are popular in nlp"
_QASPER_GOLD = "Method: the encoder uses rotary embeddings"
_QASPER_RESULTS = "Results: accuracy improved on every benchmark"
_QASPER_PAPER = [_QASPER_INTRO, _QASPER_GOLD, _QASPER_RESULTS]


def _qasper_examples() -> list[CAGExample]:
    return [
        CAGExample(
            id="p1_q1",
            question="rotary embeddings in the encoder",
            context=list(_QASPER_PAPER),
            answer="rotary embeddings",
            metadata={"dataset": "qasper", "is_impossible": False,
                      "evidence_doc_ids": [1],
                      "supporting_titles": ["Method"]},
        ),
        # Annotators supplied NO evidence: keys PRESENT and EMPTY, context
        # full — the exact shape the old truthiness fall-through fabricated
        # whole-paper gold for (repair-round blocker).
        CAGExample(
            id="p1_q2",
            question="what color is the dataset",
            context=list(_QASPER_PAPER),
            answer="",
            metadata={"dataset": "qasper", "is_impossible": True,
                      "evidence_doc_ids": [],
                      "supporting_titles": []},
        ),
    ]


_EXAMPLES = {
    "hotpotqa": _hotpot_examples(),
    "musique": _musique_examples(),
    "qasper": _qasper_examples(),
}

#: query -> ranked gold-first doc TEXTS per dataset (the canned dense answers).
_DENSE_RANKINGS: dict[str, dict[str, list[str]]] = {
    "hotpotqa": {
        "who designed the alpha protocol": [_HOTPOT_GOLD_1, _HOTPOT_DISTRACT],
        "zzz yyy xxx": [_HOTPOT_GOLD_2],
    },
    "musique": {
        "where is the delta river": [_MUSIQUE_GOLD],
    },
    "qasper": {
        "rotary embeddings in the encoder": [_QASPER_GOLD],
    },
}


class _TestOnlyCannedDenseIndex:
    """TEST-ONLY dense-leg stand-in: a canned query -> ranked-texts table.

    Deterministic and dependency-free; exists so unit tests never load
    sentence-transformers/faiss. Never a production retriever — the builder
    accepts it only through the explicitly named
    ``dense_index_factory_for_tests_only`` seam.
    """

    def __init__(self, documents: Sequence[IRDocument],
                 ranking_by_query: dict[str, list[str]]) -> None:
        self._by_text = {d.text: d for d in documents}
        self._ranking = ranking_by_query

    def search(self, query: str, *, top_k: int = 5) -> list[IRHit]:
        texts = self._ranking.get(query, [])
        hits = [
            IRHit(doc_id=self._by_text[t].doc_id, score=float(len(texts) - i))
            for i, t in enumerate(texts)
            if t in self._by_text
        ]
        return hits[:top_k]

    def resolve_hits(self, hits: Sequence[IRHit]) -> list[IRDocument]:
        by_id = {d.doc_id: d for d in self._by_text.values()}
        return [by_id[h.doc_id] for h in hits if h.doc_id in by_id]


def _dense_factory(dataset: str, documents: Sequence[IRDocument]) -> _TestOnlyCannedDenseIndex:
    return _TestOnlyCannedDenseIndex(documents, _DENSE_RANKINGS[dataset])


def _build(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        datasets=("hotpotqa", "musique"),
        out_json=tmp_path / "retrieval_gate_table.json",
        examples_by_dataset_for_tests_only=_EXAMPLES,
        dense_index_factory_for_tests_only=_dense_factory,
    )
    kwargs.update(overrides)
    return gate.build_retrieval_gate_table(**kwargs)


# ---------------------------------------------------------------------------
# End-to-end on the real bm25/rrf paths
# ---------------------------------------------------------------------------


def test_gate_table_end_to_end_real_bm25_and_rrf(
    tmp_path: Path, ranx_stub: types.ModuleType
) -> None:
    artifact = _build(tmp_path)

    hotpot = artifact["datasets"]["hotpotqa"]
    # bm25 (REAL BM25IRIndex): h1 reaches its gold lexically, h2 shares no
    # token with any doc -> zero hits -> measured recall 0 -> mean 0.5.
    assert hotpot["recall_at_100"]["bm25"] == pytest.approx(0.5)
    # dense (canned): gold ranked first for both queries.
    assert hotpot["recall_at_100"]["dense"] == pytest.approx(1.0)
    # rrf (REAL rrf_fuse of both pools): gold present in the fusion for both.
    assert hotpot["recall_at_100"]["rrf"] == pytest.approx(1.0)
    assert hotpot["pairwise_delta_pp"]["dense_minus_bm25"] == pytest.approx(50.0)
    assert hotpot["pairwise_delta_pp"]["rrf_minus_bm25"] == pytest.approx(50.0)
    assert hotpot["pairwise_delta_pp"]["rrf_minus_dense"] == pytest.approx(0.0)
    assert hotpot["max_abs_delta_pp"] == pytest.approx(50.0)
    assert hotpot["fires_contrast_6"] is True

    musique = artifact["datasets"]["musique"]
    assert musique["recall_at_100"] == {"bm25": 1.0, "dense": 1.0, "rrf": 1.0}
    assert musique["fires_contrast_6"] is False
    # The gold-less m2 is a COUNTED skip, never a scored row or a silent drop.
    assert musique["n_queries_scored"] == 1
    assert musique["n_queries_skipped_no_gold"] == 1
    assert musique["n_examples"] == 2

    assert artifact["overall"]["fires_contrast_6"] is True
    assert artifact["overall"]["fired_datasets"] == ["hotpotqa"]

    # Both artifacts landed; the .md carries the verdict + test-only banner.
    out_json = tmp_path / "retrieval_gate_table.json"
    out_md = tmp_path / "retrieval_gate_table.md"
    assert json.loads(out_json.read_text(encoding="utf-8")) == artifact
    md = out_md.read_text(encoding="utf-8")
    assert "hotpotqa" in md and "FIRES" in md
    assert "TEST-ONLY" in md  # injected dense leg can't masquerade as registered


def test_artifact_schema_and_provenance(
    tmp_path: Path, ranx_stub: types.ModuleType
) -> None:
    artifact = _build(tmp_path, seed=7)
    for key in ("schema", "charter", "gate_metric", "epsilon_pp", "created_utc",
                "seed", "split", "max_examples", "k_pool", "retrievers",
                "dense_leg_test_only", "datasets", "overall"):
        assert key in artifact, key
    assert artifact["schema"] == gate.SCHEMA
    assert artifact["gate_metric"] == "first_stage_pool_recall_at_100"
    assert artifact["epsilon_pp"] == 5.0
    assert artifact["k_pool"] == 100
    assert artifact["seed"] == 7

    dense = artifact["retrievers"]["dense"]
    # Injected dense leg: None revision WITH a named source — never fabricated.
    assert artifact["dense_leg_test_only"] is True
    assert dense["revision"] is None
    assert dense["source"] == gate.PIN_SOURCE_TEST_INJECTION
    assert artifact["retrievers"]["bm25"]["k1"] == 1.5
    assert artifact["retrievers"]["rrf"]["k"] == 60

    for row in artifact["datasets"].values():
        for key in ("n_examples", "n_queries_scored", "n_queries_skipped_no_gold",
                    "corpus", "qrels_sha1", "recall_at_100", "pairwise_delta_pp",
                    "max_abs_delta_pp", "epsilon_pp", "fires_contrast_6"):
            assert key in row, key
        assert set(row["recall_at_100"]) == {"bm25", "dense", "rrf"}
        assert len(row["corpus"]["doc_ids_sha1"]) == 40  # sha1 hex fingerprint
        assert len(row["qrels_sha1"]) == 40


# ---------------------------------------------------------------------------
# Overwrite refusal (+ --force)
# ---------------------------------------------------------------------------


def test_overwrite_refuses_without_force(
    tmp_path: Path, ranx_stub: types.ModuleType
) -> None:
    _build(tmp_path)
    with pytest.raises(gate.GateTableError, match="already exists.*--force"):
        _build(tmp_path)
    # Deliberate rebuild goes through.
    artifact = _build(tmp_path, force=True)
    assert artifact["overall"]["fires_contrast_6"] is True


def test_overwrite_refuses_on_md_sibling_alone(
    tmp_path: Path, ranx_stub: types.ModuleType
) -> None:
    (tmp_path / "retrieval_gate_table.md").write_text("stale", encoding="utf-8")
    with pytest.raises(gate.GateTableError, match="retrieval_gate_table.md"):
        _build(tmp_path)


def test_out_must_be_json(tmp_path: Path) -> None:
    with pytest.raises(gate.GateTableError, match=r"\.json"):
        _build(tmp_path, out_json=tmp_path / "table.txt")


# ---------------------------------------------------------------------------
# Freeze-pin refusals (production dense leg only — no injection)
# ---------------------------------------------------------------------------


def _build_production_dense(tmp_path: Path, freeze_file: Path) -> None:
    """Production dense leg (factory=None): must refuse AT the pin, before
    any dataset/model load (examples are injected, so a pin that resolved
    would fail later on the real FaissIRIndex import — these tests never get
    that far)."""
    gate.build_retrieval_gate_table(
        datasets=("hotpotqa",),
        out_json=tmp_path / "retrieval_gate_table.json",
        freeze_file=freeze_file,
        examples_by_dataset_for_tests_only=_EXAMPLES,
        dense_index_factory_for_tests_only=None,
    )


def test_refuses_when_freeze_artifact_missing(tmp_path: Path) -> None:
    with pytest.raises(gate.GateTableError, match="freeze artifact missing"):
        _build_production_dense(tmp_path, tmp_path / "no_such_freeze.json")


#: Mirror of the REAL freeze artifact's 'embedding' entry — the QUALITY
#: module's similarity embedder (CAGE_EMBEDDING_REVISION). Present in the
#: fixtures below as a decoy: the gate table must never consume it.
_QUALITY_EMBEDDING_DECOY = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
}


def test_refuses_when_dense_retriever_entry_missing(tmp_path: Path) -> None:
    # The real repo artifact's shape TODAY: a quality 'embedding' pin exists,
    # 'dense_retriever' does not. The producer must refuse on the missing
    # retriever slot — never fall back to (or match against) the quality pin.
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps({"INSTRUMENT_REVISIONS": {
            "nli": {},
            "embedding": _QUALITY_EMBEDDING_DECOY,
        }}),
        encoding="utf-8",
    )
    with pytest.raises(gate.GateTableError, match="no 'dense_retriever' entry") as exc_info:
        _build_production_dense(tmp_path, freeze)
    message = str(exc_info.value)
    # The fix steers at ADDING the retriever's own slot...
    assert "ADD" in message and gate.FREEZE_DENSE_KEY in message
    assert gate.DEFAULT_EMBEDDING_MODEL in message
    # ...and explicitly warns OFF the registered quality-instrument pin.
    assert "Do NOT repurpose INSTRUMENT_REVISIONS.embedding" in message
    assert "quality" in message and "CAGE_EMBEDDING_REVISION" in message


def test_refuses_when_instrument_revisions_missing(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"QASPER_TAU": 0.9}), encoding="utf-8")
    with pytest.raises(gate.GateTableError, match="INSTRUMENT_REVISIONS"):
        _build_production_dense(tmp_path, freeze)


def test_refuses_when_revision_absent(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps({"INSTRUMENT_REVISIONS": {"dense_retriever": {
            "model": gate.DEFAULT_EMBEDDING_MODEL}}}),
        encoding="utf-8",
    )
    with pytest.raises(gate.GateTableError, match="revision.*absent"):
        _build_production_dense(tmp_path, freeze)


def test_refuses_on_model_pin_mismatch(tmp_path: Path) -> None:
    # dense_retriever pins the OTHER charter retriever (bge); asking for e5
    # must refuse — a hash never migrates across repos silently.
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps({"INSTRUMENT_REVISIONS": {
            "embedding": _QUALITY_EMBEDDING_DECOY,
            "dense_retriever": {
                "model": "BAAI/bge-large-en-v1.5",
                "revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09"},
        }}),
        encoding="utf-8",
    )
    with pytest.raises(gate.GateTableError) as exc_info:
        _build_production_dense(tmp_path, freeze)
    message = str(exc_info.value)
    # Both models NAMED, and the fix steers at the retriever's OWN slot.
    assert "BAAI/bge-large-en-v1.5" in message
    assert gate.DEFAULT_EMBEDDING_MODEL in message
    assert gate.FREEZE_DENSE_KEY in message


def test_freeze_happy_path_resolution(tmp_path: Path) -> None:
    # The quality 'embedding' decoy sits beside dense_retriever and is
    # IGNORED: resolution reads the retriever's own slot only.
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps({"INSTRUMENT_REVISIONS": {
            "embedding": _QUALITY_EMBEDDING_DECOY,
            "dense_retriever": {
                "model": "intfloat/e5-large-v2", "revision": "abc123"},
        }}),
        encoding="utf-8",
    )
    prov = gate.resolve_dense_retriever_revision(freeze, "intfloat/e5-large-v2")
    assert prov["model"] == "intfloat/e5-large-v2"
    assert prov["revision"] == "abc123"
    assert prov["source"] == gate.PIN_SOURCE_FREEZE
    assert prov["freeze_file"] == str(freeze.resolve())


# ---------------------------------------------------------------------------
# The registered 5pp boundary
# ---------------------------------------------------------------------------


def test_gate_boundary_4p9_does_not_fire() -> None:
    decision = gate.gate_decision({"bm25": 0.851, "dense": 0.90, "rrf": 0.90})
    assert decision["max_abs_delta_pp"] == pytest.approx(4.9)
    assert decision["fires_contrast_6"] is False


def test_gate_boundary_5p0_fires_despite_float_dust() -> None:
    # (0.90 - 0.85) * 100 == 4.999999999999993 in binary; the registered ε
    # boundary must still see exactly 5.0pp and FIRE.
    decision = gate.gate_decision({"bm25": 0.85, "dense": 0.90, "rrf": 0.85})
    assert decision["max_abs_delta_pp"] == 5.0
    assert decision["fires_contrast_6"] is True


def test_gate_fires_in_either_direction() -> None:
    # bm25 ABOVE dense by 6pp fires too (charter: "either direction").
    decision = gate.gate_decision({"bm25": 0.96, "dense": 0.90, "rrf": 0.96})
    assert decision["fires_contrast_6"] is True
    assert decision["pairwise_delta_pp"]["dense_minus_bm25"] == pytest.approx(-6.0)


def test_gate_decision_validates_inputs() -> None:
    with pytest.raises(gate.GateTableError, match="exactly the variants"):
        gate.gate_decision({"bm25": 0.5, "dense": 0.5})
    with pytest.raises(gate.GateTableError, match="outside"):
        gate.gate_decision({"bm25": 1.5, "dense": 0.5, "rrf": 0.5})
    with pytest.raises(gate.GateTableError, match="epsilon_pp"):
        gate.gate_decision({"bm25": 0.5, "dense": 0.5, "rrf": 0.5}, epsilon_pp=0)


# ---------------------------------------------------------------------------
# Input validation refusals
# ---------------------------------------------------------------------------


def test_unknown_dataset_refused(tmp_path: Path) -> None:
    with pytest.raises(gate.GateTableError, match="unknown gate dataset"):
        _build(tmp_path, datasets=("hotpotqa", "sharegpt"))


def test_duplicate_dataset_refused(tmp_path: Path) -> None:
    with pytest.raises(gate.GateTableError, match="duplicate dataset"):
        _build(tmp_path, datasets=("hotpotqa", "hotpotqa"))


def test_duplicate_query_ids_refused(tmp_path: Path) -> None:
    twin = _hotpot_examples()[0]
    with pytest.raises(gate.GateTableError, match="duplicate query id"):
        _build(
            tmp_path,
            datasets=("hotpotqa",),
            examples_by_dataset_for_tests_only={
                "hotpotqa": [twin, twin],
            },
        )


def test_injected_examples_must_cover_every_dataset(tmp_path: Path) -> None:
    # Coverage is validated UP FRONT — the refusal fires before any dataset
    # is scored (no half-built table behind a passing hotpotqa leg).
    with pytest.raises(gate.GateTableError, match=r"no examples for \['musique'\]"):
        _build(
            tmp_path,
            datasets=("hotpotqa", "musique"),
            examples_by_dataset_for_tests_only={"hotpotqa": _hotpot_examples()},
        )


def test_qasper_evidence_doc_ids_take_precedence() -> None:
    # The precise Qasper route: evidence_doc_ids indexes context docs directly
    # (the loader documents that title-prefix matching over-selects).
    example = CAGExample(
        id="p1_q1",
        question="q",
        context=["Intro: a", "Method: b", "Results: c"],
        answer="a",
        metadata={"evidence_doc_ids": [2], "supporting_titles": ["Intro"]},
    )
    assert gate._gold_context_texts(example) == ["Results: c"]


def test_qasper_goldless_query_is_recorded_absence() -> None:
    # Repair-round BLOCKER repro: a real Qasper no-evidence/unanswerable
    # question (evidence_doc_ids=[] AND supporting_titles=[], full paper
    # context) must yield NO gold. The old truthiness fall-through returned
    # the ENTIRE paper — 3 fabricated gold ids where n_gold must be 0.
    example = _qasper_examples()[1]
    assert gate._gold_context_texts(example) == []


def test_empty_supporting_titles_is_recorded_absence() -> None:
    # hotpotqa/musique shape with the supporting_titles KEY present and
    # empty: same rule — gold_only's whole-context fallback must not run.
    example = CAGExample(
        id="h_empty",
        question="q",
        context=[_HOTPOT_GOLD_1, _HOTPOT_DISTRACT],
        answer="",
        metadata={"supporting_titles": []},
    )
    assert gate._gold_context_texts(example) == []


def test_squad_context_is_gold_when_no_evidence_keys() -> None:
    # SQuAD v2 emits NEITHER gold-evidence key: its context already IS the
    # gold paragraph, so the whole-context route survives for exactly (and
    # only) that loader shape.
    example = CAGExample(
        id="s1",
        question="q",
        context=["The gold paragraph."],
        answer="gold",
        metadata={"dataset": "squad_v2", "is_impossible": False},
    )
    assert gate._gold_context_texts(example) == ["The gold paragraph."]


def test_qasper_goldless_end_to_end_counted_skip(
    tmp_path: Path, ranx_stub: types.ModuleType
) -> None:
    # End to end on the real bm25/rrf paths: the gold-less p1_q2 is a COUNTED
    # skip, so qrels cover p1_q1 alone and every leg scores its clean 1.0 —
    # not recall against an "every paragraph is relevant" fabricated gold set
    # (which would drag the row and could flip the registered gate decision).
    artifact = _build(tmp_path, datasets=("qasper",))
    row = artifact["datasets"]["qasper"]
    assert row["n_examples"] == 2
    assert row["n_queries_scored"] == 1
    assert row["n_queries_skipped_no_gold"] == 1
    assert row["recall_at_100"] == {"bm25": 1.0, "dense": 1.0, "rrf": 1.0}
    assert row["fires_contrast_6"] is False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_unknown_dataset_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = gate.main(["--datasets", "bogus", "--out", str(tmp_path / "t.json")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "unknown gate dataset" in err and "squad_v2" in err


def test_cli_explicit_missing_freeze_file_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = gate.main([
        "--out", str(tmp_path / "t.json"),
        "--freeze-file", str(tmp_path / "nope.json"),
    ])
    assert rc == 2
    assert "does not exist" in capsys.readouterr().err


def test_cli_pin_mismatch_exits_1_before_any_load(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps({"INSTRUMENT_REVISIONS": {"dense_retriever": {
            "model": "BAAI/bge-large-en-v1.5",
            "revision": "deadbeef"}}}),
        encoding="utf-8",
    )
    rc = gate.main([
        "--out", str(tmp_path / "t.json"),
        "--freeze-file", str(freeze),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "mismatch" in err and "e5-large-v2" in err
    assert gate.FREEZE_DENSE_KEY in err
