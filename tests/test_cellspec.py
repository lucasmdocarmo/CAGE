"""Tests for src.analysis.cellspec — the D7 cell tuple, baseline map, alias map."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from src.analysis.cellspec import (
    BASELINES,
    FRESH_SET,
    LEGACY_ALIASES,
    REUSE_SET,
    CellSpec,
    CellSpecError,
    InvalidCellSpecError,
    UnknownBaselineError,
    from_legacy,
)

PILOT_RUN = (
    Path(__file__).resolve().parents[1]
    / "results/phase2/2026-07-16_full_qwen3-8b_100x3_squad_v2"
)

# The 20 distinct `baseline` values in the 100x3 squad_v2 pilot archive
# (pinned so the round-trip contract holds even where the archive is absent).
PINNED_PILOT_NAMES = [
    "cag_full",
    "cag_true_off",
    "cag_true_on",
    "compressed_cag",
    "compressed_rag",
    "hybrid_retrieval_cache_cold",
    "hybrid_retrieval_cache_warm",
    "lmcache_rag",
    "no_cache",
    "prefix_cache",
    "prefix_cache_grouped",
    "prefix_cache_multiturn",
    "prefix_cache_repeat",
    "rag",
    "rag_full",
    "redis_retrieval_cache_cold",
    "spec_qwen8b_eagle3_cag",
    "spec_qwen8b_eagle3_rag",
    "spec_qwen8b_ngram_cag",
    "spec_qwen8b_ngram_rag",
]


def _discover_pilot_names() -> set[str]:
    names: set[str] = set()
    for path in PILOT_RUN.rglob("results.csv"):
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                names.add(row["baseline"])
    return names


# --- legacy alias map ----------------------------------------------------------


@pytest.mark.parametrize("name", PINNED_PILOT_NAMES)
def test_pinned_pilot_name_round_trips(name: str) -> None:
    spec = from_legacy(name)
    assert isinstance(spec, CellSpec)
    assert CellSpec.from_flat_dict(spec.to_flat_dict()) == spec
    assert spec.to_row_key() == CellSpec.from_flat_dict(spec.to_flat_dict()).to_row_key()


@pytest.mark.skipif(not PILOT_RUN.exists(), reason="pilot archive not present locally")
def test_every_archived_pilot_name_round_trips() -> None:
    discovered = _discover_pilot_names()
    assert discovered == set(PINNED_PILOT_NAMES)  # catches pinned-list drift
    for name in discovered:
        spec = from_legacy(name)
        assert CellSpec.from_flat_dict(spec.to_flat_dict()) == spec


def test_unknown_legacy_name_fails_closed() -> None:
    with pytest.raises(UnknownBaselineError):
        from_legacy("eagle3")
    with pytest.raises(UnknownBaselineError):
        from_legacy("")


def test_legacy_merge_and_policy_translations() -> None:
    # §7.7a: pilot cag_full ≡ prefix_cache — one cell, gold-reuse
    assert from_legacy("cag_full") == from_legacy("prefix_cache")
    assert from_legacy("cag_full").arm == "gold-reuse"
    # §7.5: compressed_cag was fp8 KV dtype -> the compress POLICY (=> F3)
    spec = from_legacy("compressed_cag")
    assert spec.policy == "compress-fp8"
    assert spec.family == "F3"
    assert spec.arm == "corpus-reuse"
    # pilot RAG cells ran the ranked pipeline; the spec-matrix rag cells did not
    assert from_legacy("rag").retriever == "rerank"
    assert from_legacy("spec_qwen8b_eagle3_rag").retriever == "dense"


def test_legacy_alias_map_specs_are_valid() -> None:
    for name, spec in LEGACY_ALIASES.items():
        assert from_legacy(name) == spec


# --- baseline map --------------------------------------------------------------


def test_fresh_reuse_membership() -> None:
    assert FRESH_SET == {"B1", "B5", "B6", "B9", "B11"}
    assert REUSE_SET == {"B2", "B3", "B4", "B7", "B8", "B10", "B12"}
    assert FRESH_SET.isdisjoint(REUSE_SET)
    assert FRESH_SET | REUSE_SET == set(BASELINES)
    for bid in FRESH_SET:
        assert BASELINES[bid].reuse == "fresh"
    for bid in REUSE_SET:
        assert BASELINES[bid].reuse == "reuse"


def test_b12_is_reuse_corpus_trunc() -> None:
    assert "B12" in REUSE_SET
    assert BASELINES["B12"].arm == "corpus-trunc"
    assert BASELINES["B12"].retriever == "none"


def test_b4_reuse_bit_rides_f3() -> None:
    # B4 = corpus-fresh but its reuse-bit classification is REUSE (§7.6 Group B)
    assert BASELINES["B4"].arm == "corpus-fresh"
    assert "B4" in REUSE_SET
    assert CellSpec.from_baseline("B4", family="F3").family == "F3"


def test_ranking_ablated_exactly_once() -> None:
    assert BASELINES["B5"] != BASELINES["B6"]
    assert BASELINES["B5"].arm == BASELINES["B6"].arm == "retr-fresh"
    assert BASELINES["B5"].retriever == "dense"
    assert BASELINES["B6"].retriever == "rerank"
    # B7-B12 RAG-side baselines inherit the ranked pipeline
    for bid in ("B7", "B8", "B9", "B11"):
        assert BASELINES[bid].retriever == "rerank"


def test_from_baseline_unknown_id_raises() -> None:
    with pytest.raises(UnknownBaselineError):
        CellSpec.from_baseline("B13")


def test_from_baseline_defaults_are_anchor_f1() -> None:
    spec = CellSpec.from_baseline("B1")
    assert (spec.engine, spec.model, spec.family, spec.topology) == (
        "vllm",
        "qwen3-14b",
        "F1",
        "single",
    )


# --- validation ----------------------------------------------------------------


def _spec(**overrides: object) -> CellSpec:
    base: dict[str, object] = dict(
        arm="gold-fresh",
        retriever="none",
        policy="none",
        topology="single",
        engine="vllm",
        model="qwen3-14b",
        family="F1",
    )
    base.update(overrides)
    return CellSpec(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        # reuse arm in F2 (prefix OFF — reuse arms undefined)
        dict(arm="gold-reuse", family="F2"),
        dict(arm="corpus-reuse", family="F2"),
        dict(arm="corpus-fresh", family="F2"),  # B4's reuse-bit routes it to F3
        # hf under pressure / non-single
        dict(engine="hf", family="F2"),
        dict(engine="hf", family="F3", arm="corpus-reuse"),
        dict(engine="hf", topology="tp"),
        # policy while family=F1
        dict(policy="evict"),
        dict(policy="compress-fp8"),
        # truncate is NOT a policy value (§7.3: B11/B12 carry policy=none)
        dict(policy="truncate", family="F2"),
        # retriever consistency
        dict(arm="retr-fresh"),  # retrieval arm with retriever=none
        dict(retriever="dense"),  # non-retrieval arm with a retriever
        # topology rules
        dict(topology="pd"),  # pd outside DIST
        dict(family="DIST"),  # DIST at single topology
        dict(family="DIST", topology="pd", arm="retr-fresh", retriever="rerank"),
        # engine-model support
        dict(engine="lmdeploy", model="deepseek-v3"),
        dict(engine="lmdeploy", model="qwen3-next-80b"),
        # pressure coords
        dict(budget_r=0.5),  # F1 is sub-pressure
        dict(rate_frac=0.85),
        dict(family="F2", budget_r=-0.5),
        dict(family="F2", rate_frac=0.0),
        dict(family="F2", budget_r=float("nan")),
        dict(family="F2", budget_r=math.inf),
        dict(family="F2", budget_r=True),
        # unknown axis values (pilot names are not arms)
        dict(arm="prefix_cache"),
        dict(model="qwen3-8b"),
        dict(engine="ollama"),
        dict(family="F4"),
    ],
)
def test_illegal_combo_raises(overrides: dict[str, object]) -> None:
    with pytest.raises(InvalidCellSpecError):
        _spec(**overrides)


def test_legal_pressure_and_dist_cells() -> None:
    f2 = _spec(family="F2", budget_r=0.5, rate_frac=0.95)
    assert (f2.budget_r, f2.rate_frac) == (0.5, 0.95)
    f3 = _spec(
        arm="corpus-reuse",
        policy="evict",
        family="F3",
        engine="sglang",
        model="llama-3.3-70b",
        topology="tp",
        budget_r=0.25,
        rate_frac=1.05,
    )
    assert f3.policy == "evict"
    dist = _spec(family="DIST", topology="pd", model="deepseek-v3", arm="corpus-reuse")
    assert dist.topology == "pd"


def test_pressure_coords_coerced_to_float() -> None:
    spec = _spec(family="F2", budget_r=1, rate_frac=1)
    assert isinstance(spec.budget_r, float) and isinstance(spec.rate_frac, float)


# --- serialization -------------------------------------------------------------


def test_flat_dict_round_trip_with_coords() -> None:
    spec = _spec(family="F2", budget_r=0.25, rate_frac=1.05)
    flat = spec.to_flat_dict()
    assert flat["budget_r"] == 0.25
    assert CellSpec.from_flat_dict(flat) == spec


def test_from_flat_dict_accepts_missing_coord_encodings() -> None:
    base = _spec().to_flat_dict()
    for absent in (None, "", float("nan")):
        row = dict(base, budget_r=absent, rate_frac=absent)
        spec = CellSpec.from_flat_dict(row)
        assert spec.budget_r is None and spec.rate_frac is None
    # extra CSV columns are ignored
    row = dict(base, ttft_ms=123.0, baseline="no_cache")
    assert CellSpec.from_flat_dict(row) == _spec()


def test_from_flat_dict_missing_axis_raises() -> None:
    row = _spec().to_flat_dict()
    del row["engine"]
    with pytest.raises(CellSpecError, match="engine"):
        CellSpec.from_flat_dict(row)


def test_to_row_key_shape() -> None:
    assert (
        _spec().to_row_key()
        == "gold-fresh|none|none|single|vllm|qwen3-14b|F1"
    )
    keyed = _spec(family="F2", budget_r=0.5, rate_frac=0.95).to_row_key()
    assert keyed == "gold-fresh|none|none|single|vllm|qwen3-14b|F2|r0.5|lam0.95"
