"""Tests for src/analysis/l0_retrieval.py (charter D8 §8.2 Layer-0 scorer).

ranx is pinned in requirements.txt (ranx==0.3.21) but these tests stay
hermetic: they mock the instrument at the import seam (sys.modules), the same
mock-the-transport pattern tests/test_inference.py uses for HTTP:

- fail-closed: an absent ranx raises RanxUnavailableError (never a silent
  degrade, never a hand-rolled substitute);
- wiring: score_stages passes the STAGE-CORRECT qrels/run dicts and metric
  strings (recall@100 on the pool, ndcg@10 + mrr@10 on the reranked run,
  recall@k_served on the served run) and maps results onto StageScores;
- MRR is emitted for single-gold sets only (§8.2), a labeled None otherwise;
- complete-evidence@k (CAGE-native, charter-defined) is computed for real:
  happy path + boundary (gold doc outside top-k) + ties;
- qrels constructors (manifest / gold-docs) validate fail-closed;
- score_cells joins per cell keyed by the CellSpec row key, refusing illegal
  keys and non-retrieval cells, all problems listed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis import l0_retrieval as l0  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures: a tiny 3-query dataset with stage-tagged runs
# ---------------------------------------------------------------------------

QRELS_SINGLE_GOLD: dict[str, dict[str, int]] = {
    "q1": {"d1": 1},
    "q2": {"d4": 1},
    "q3": {"d7": 1},
}

QRELS_MULTI_HOP: dict[str, dict[str, int]] = {
    "q1": {"d1": 1, "d2": 1},  # two gold hops
    "q2": {"d4": 1},
}


def _run(*rankings: tuple[str, list[str]]) -> dict[str, dict[str, float]]:
    """Build a run dict from per-query ranked doc-id lists (descending score)."""
    return {
        query: {doc: float(len(docs) - rank) for rank, doc in enumerate(docs)}
        for query, docs in rankings
    }


STAGE_RUNS = l0.StageRuns(
    pool=_run(("q1", ["d9", "d1", "d2"]), ("q2", ["d4", "d5"]), ("q3", ["d8", "d7"])),
    reranked=_run(("q1", ["d1", "d9"]), ("q2", ["d4"]), ("q3", ["d7", "d8"])),
    served=_run(("q1", ["d1"]), ("q2", ["d4"]), ("q3", ["d8"])),
)


class _StubQrels:
    def __init__(self, d: dict[str, dict[str, int]]) -> None:
        self.d = d


class _StubRun:
    def __init__(self, d: dict[str, dict[str, float]]) -> None:
        self.d = d


def _make_ranx_stub(canned: dict[str, float] | None = None) -> types.ModuleType:
    """A recording ranx stand-in: canned metric values + call capture."""
    stub = types.ModuleType("ranx")
    stub.Qrels = _StubQrels  # type: ignore[attr-defined]
    stub.Run = _StubRun  # type: ignore[attr-defined]
    stub.calls = []  # type: ignore[attr-defined]
    values = canned or {}

    def evaluate(qrels: _StubQrels, run: _StubRun, metrics: list[str]) -> dict[str, float]:
        stub.calls.append({"qrels": qrels.d, "run": run.d, "metrics": list(metrics)})  # type: ignore[attr-defined]
        return {m: values.get(m, 0.5) for m in metrics}

    stub.evaluate = evaluate  # type: ignore[attr-defined]
    return stub


@pytest.fixture()
def ranx_stub(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    stub = _make_ranx_stub(
        {"recall@100": 0.9, "ndcg@10": 0.8, "mrr@10": 0.7, "recall@5": 0.6}
    )
    monkeypatch.setitem(sys.modules, "ranx", stub)
    return stub


# ---------------------------------------------------------------------------
# Fail-closed instrument import
# ---------------------------------------------------------------------------


def test_absent_ranx_raises_instrument_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # sys.modules[name] = None makes `import ranx` raise ImportError even if
    # a future environment installs it — the seam under test stays the seam.
    monkeypatch.setitem(sys.modules, "ranx", None)
    with pytest.raises(l0.RanxUnavailableError) as exc_info:
        l0.score_stages(QRELS_SINGLE_GOLD, STAGE_RUNS, k_served=5)
    assert "ranx" in str(exc_info.value)
    assert "does not degrade" in str(exc_info.value)


def test_module_imports_without_ranx() -> None:
    # The lazy-import design: the module itself must import in a ranx-free env
    # (only SCORING is fail-closed).
    assert not hasattr(l0, "ranx")
    assert callable(l0.score_stages)


# ---------------------------------------------------------------------------
# Stage wiring through the (stubbed) instrument
# ---------------------------------------------------------------------------


def test_score_stages_stage_tagged_wiring(ranx_stub: types.ModuleType) -> None:
    scores = l0.score_stages(QRELS_SINGLE_GOLD, STAGE_RUNS, k_served=5)

    calls: list[dict[str, Any]] = ranx_stub.calls  # type: ignore[attr-defined]
    assert [c["metrics"] for c in calls] == [
        ["recall@100"],  # pool stage
        ["ndcg@10", "mrr@10"],  # reranked stage (single-gold -> MRR emitted)
        ["recall@5"],  # served stage at k_served
    ]
    # The RIGHT run feeds the RIGHT stage (the B5-vs-B6 pool-row discipline).
    assert calls[0]["run"] == {q: dict(d) for q, d in STAGE_RUNS.pool.items()}
    assert calls[1]["run"] == {q: dict(d) for q, d in STAGE_RUNS.reranked.items()}
    assert calls[2]["run"] == {q: dict(d) for q, d in STAGE_RUNS.served.items()}
    for call in calls:
        assert call["qrels"] == QRELS_SINGLE_GOLD

    assert scores.n_queries == 3
    assert scores.pool_recall == 0.9
    assert scores.reranked_ndcg == 0.8
    assert scores.reranked_mrr == 0.7
    assert scores.served_recall == 0.6
    assert scores.complete_evidence is None  # multi_hop not requested

    d = scores.to_dict()
    assert d["pool_recall_at_100"] == 0.9
    assert d["reranked_ndcg_at_10"] == 0.8
    assert d["served_recall_at_5"] == 0.6
    assert d["complete_evidence_at_5"] is None


def test_mrr_is_single_gold_only(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _make_ranx_stub()
    monkeypatch.setitem(sys.modules, "ranx", stub)
    runs = l0.StageRuns(
        pool=_run(("q1", ["d1", "d2"]), ("q2", ["d4"])),
        reranked=_run(("q1", ["d1", "d2"]), ("q2", ["d4"])),
        served=_run(("q1", ["d1", "d2"]), ("q2", ["d4"])),
    )
    scores = l0.score_stages(QRELS_MULTI_HOP, runs, k_served=2)
    assert scores.reranked_mrr is None  # labeled absence, never a silent zero
    rerank_call = stub.calls[1]  # type: ignore[attr-defined]
    assert rerank_call["metrics"] == ["ndcg@10"]  # mrr never requested


# ---------------------------------------------------------------------------
# complete-evidence@k (CAGE-native, computed for real)
# ---------------------------------------------------------------------------


def test_complete_evidence_happy_path() -> None:
    served = _run(("q1", ["d1", "d2", "d9"]), ("q2", ["d4"]))
    # q1: both gold hops in top-3 -> complete; q2: single gold present.
    assert l0.complete_evidence_at_k(QRELS_MULTI_HOP, served, k=3) == 1.0


def test_complete_evidence_partial_hop_is_zero_for_that_query() -> None:
    served = _run(("q1", ["d1", "d9", "d8"]), ("q2", ["d4"]))
    # q1 has 1 of 2 gold hops served: recall@k would say 0.5 — the ceiling
    # metric says INCOMPLETE (that is its whole point, §8.2).
    assert l0.complete_evidence_at_k(QRELS_MULTI_HOP, served, k=3) == 0.5


def test_complete_evidence_k_boundary_cuts_late_gold() -> None:
    served = _run(("q1", ["d9", "d1", "d2"]), ("q2", ["d4"]))
    # At k=2 the second gold hop (d2, rank 3) is outside the served cut.
    assert l0.complete_evidence_at_k(QRELS_MULTI_HOP, served, k=2) == 0.5
    assert l0.complete_evidence_at_k(QRELS_MULTI_HOP, served, k=3) == 1.0


def test_complete_evidence_score_ties_break_deterministically() -> None:
    served = {"q1": {"d1": 1.0, "d2": 1.0, "d0": 1.0}, "q2": {"d4": 1.0}}
    # Equal scores tie-break by doc id ascending: top-2 = d0, d1 -> d2 cut.
    assert l0.complete_evidence_at_k(QRELS_MULTI_HOP, served, k=2) == 0.5


def test_complete_evidence_bad_k_fails(ranx_stub: types.ModuleType) -> None:
    with pytest.raises(l0.RetrievalScoringError, match="k=0"):
        l0.complete_evidence_at_k(QRELS_MULTI_HOP, STAGE_RUNS.served, k=0)


# ---------------------------------------------------------------------------
# Qrels constructors
# ---------------------------------------------------------------------------


def test_qrels_from_manifest_happy_path() -> None:
    manifest = {"question_to_block": {"q-a": 0, "q-b": 3}}
    assert l0.qrels_from_manifest(manifest) == {
        "q-a": {"block-0": 1},
        "q-b": {"block-3": 1},
    }


def test_qrels_from_manifest_refuses_without_assignments() -> None:
    with pytest.raises(l0.RetrievalScoringError, match="question_to_block"):
        l0.qrels_from_manifest({"blocks": []})
    with pytest.raises(l0.RetrievalScoringError, match="question_to_block"):
        l0.qrels_from_manifest({"question_to_block": {}})


def test_qrels_from_manifest_refuses_bad_block_id() -> None:
    with pytest.raises(l0.RetrievalScoringError, match="int block id"):
        l0.qrels_from_manifest({"question_to_block": {"q1": "zero"}})


def test_qrels_from_gold_docs_happy_and_failures() -> None:
    assert l0.qrels_from_gold_docs({"q1": ["d1", "d2"]}) == {
        "q1": {"d1": 1, "d2": 1}
    }
    with pytest.raises(l0.RetrievalScoringError, match="empty"):
        l0.qrels_from_gold_docs({})
    with pytest.raises(l0.RetrievalScoringError, match="non-empty sequence"):
        l0.qrels_from_gold_docs({"q1": []})
    with pytest.raises(l0.RetrievalScoringError, match="non-empty sequence"):
        l0.qrels_from_gold_docs({"q1": "d1"})  # a bare string is not doc ids


# ---------------------------------------------------------------------------
# Input validation (fail closed)
# ---------------------------------------------------------------------------


def test_run_missing_a_qrels_query_is_a_join_bug(ranx_stub: types.ModuleType) -> None:
    runs = l0.StageRuns(
        pool=_run(("q1", ["d1"])),  # q2/q3 missing
        reranked=STAGE_RUNS.reranked,
        served=STAGE_RUNS.served,
    )
    with pytest.raises(l0.RetrievalScoringError, match="JOIN bug"):
        l0.score_stages(QRELS_SINGLE_GOLD, runs, k_served=5)


def test_qrels_with_no_relevant_doc_refuse(ranx_stub: types.ModuleType) -> None:
    with pytest.raises(l0.RetrievalScoringError, match="no relevant document"):
        l0.score_stages({"q1": {"d1": 0}}, STAGE_RUNS, k_served=5)


def test_bad_k_values_refuse(ranx_stub: types.ModuleType) -> None:
    with pytest.raises(l0.RetrievalScoringError, match="k_served"):
        l0.score_stages(QRELS_SINGLE_GOLD, STAGE_RUNS, k_served=0)
    with pytest.raises(l0.RetrievalScoringError, match="k_pool"):
        l0.score_stages(QRELS_SINGLE_GOLD, STAGE_RUNS, k_served=5, k_pool=-1)


# ---------------------------------------------------------------------------
# Per-cell join keyed by the CellSpec row key
# ---------------------------------------------------------------------------

B6_KEY = "retr-fresh|rerank|none|single|vllm|qwen3-14b|F1"
B5_KEY = "retr-fresh|dense|none|single|vllm|qwen3-14b|F1"
GOLD_KEY = "gold-fresh|none|none|single|vllm|qwen3-14b|F1"


def test_score_cells_joins_by_row_key(ranx_stub: types.ModuleType) -> None:
    df = l0.score_cells(
        QRELS_SINGLE_GOLD,
        {B6_KEY: STAGE_RUNS, B5_KEY: STAGE_RUNS},
        k_served=5,
    )
    assert list(df["row_key"]) == sorted([B6_KEY, B5_KEY])
    assert set(df.columns) >= {
        "row_key", "arm", "retriever", "engine", "model", "family",
        "pool_recall_at_100", "reranked_ndcg_at_10", "served_recall_at_5",
    }
    b6 = df[df["row_key"] == B6_KEY].iloc[0]
    assert b6["retriever"] == "rerank"
    assert b6["pool_recall_at_100"] == 0.9


def test_score_cells_refuses_non_retrieval_cells(ranx_stub: types.ModuleType) -> None:
    # Layer 0 is defined only for ranked-list-consuming arms (§8.2): a gold
    # cell has evidence quality by construction — scoring it is a category
    # error, refused with the cell named.
    with pytest.raises(l0.RetrievalScoringError, match="retriever='none'"):
        l0.score_cells(QRELS_SINGLE_GOLD, {GOLD_KEY: STAGE_RUNS}, k_served=5)


def test_score_cells_lists_every_illegal_key(ranx_stub: types.ModuleType) -> None:
    bad_short = "not-a-cell-key"
    bad_axis = "retr-fresh|rerank|none|single|vllm|qwen3-14b|NOPE"
    with pytest.raises(l0.RetrievalScoringError) as exc_info:
        l0.score_cells(
            QRELS_SINGLE_GOLD,
            {bad_short: STAGE_RUNS, bad_axis: STAGE_RUNS, GOLD_KEY: STAGE_RUNS},
            k_served=5,
        )
    message = str(exc_info.value)
    assert "3 illegal cell key(s)" in message
    assert bad_short in message and "NOPE" in message and "retriever='none'" in message


def test_score_cells_empty_refuses(ranx_stub: types.ModuleType) -> None:
    with pytest.raises(l0.RetrievalScoringError, match="no cells"):
        l0.score_cells(QRELS_SINGLE_GOLD, {}, k_served=5)
