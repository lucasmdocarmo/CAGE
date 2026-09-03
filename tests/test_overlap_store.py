"""Engineered-overlap store + measured overlap (src/data/overlap.py) — synthetic,
no datasets/torch. Charter D5 F1/F3: overlap ENGINEERED and REPORTED, never assumed.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import (  # noqa: E402  (OverlapError re-export exercised)
    OverlapError,
    build_manifest,
)
from src.data.overlap import (  # noqa: E402
    gold_set,
    pairwise_shared_fraction,
    plan_overlap_groups,
)


@dataclass
class FakeExample:
    id: str
    question: str
    context: List[str]
    answer: str = "x"
    metadata: Dict = field(default_factory=dict)


def _para(name: str, words: int = 30) -> str:
    return " ".join(f"{name}w{w}" for w in range(words))


def make_chain_pool(n_families: int = 8, q_per_family: int = 6) -> List[FakeExample]:
    """HotpotQA-shaped: 2-gold-paragraph questions, family k = {P_k, P_{k+1}}, so
    adjacent families share exactly one paragraph (pairwise Jaccard 1/3) and same-
    family questions share both (Jaccard 1)."""
    paras = [_para(f"p{k}") for k in range(n_families + 1)]
    out = []
    for k in range(n_families):
        for q in range(q_per_family):
            out.append(FakeExample(id=f"f{k}_q{q}", question=f"Q{k}.{q}?",
                                   context=[paras[k], paras[k + 1]]))
    return out


def make_disjoint_singletons(n: int = 10) -> List[FakeExample]:
    """One question per family, families pairwise disjoint: max achievable
    cross-question overlap is 0."""
    return [
        FakeExample(id=f"s{k}", question=f"S{k}?",
                    context=[_para(f"a{k}"), _para(f"b{k}")])
        for k in range(n)
    ]


BUDGET = 800  # ~30-word paragraphs: several 2-paragraph sets fit comfortably


# ---------------------------------------------------------------- primitives

def test_pairwise_shared_fraction_jaccard() -> None:
    a = frozenset({"x", "y"})
    assert pairwise_shared_fraction(a, a) == 1.0
    assert pairwise_shared_fraction(a, frozenset({"y", "z"})) == pytest.approx(1 / 3)
    assert pairwise_shared_fraction(a, frozenset({"z", "w"})) == 0.0
    assert pairwise_shared_fraction(frozenset(), frozenset()) == 0.0


# ---------------------------------------------------- engineered mode: hits target

def test_engineered_hits_full_overlap_target() -> None:
    pool = make_chain_pool()
    m = build_manifest(pool, num_queries=4, num_trials=1, seed=42, block_budget=BUDGET,
                       overlap_target=1.0, overlap_tolerance=0.05,
                       overlap_group_size=3)
    ov = m["overlap"]
    assert ov["mode"] == "engineered"
    assert ov["target_shared_fraction"] == 1.0
    assert ov["tolerance"] == 0.05
    measured = ov["measured"]
    # Groups of same-family questions: realized overlap is exactly 1.0.
    assert measured["mean_pairwise_shared_fraction"] == pytest.approx(1.0)
    assert measured["n_pairs"] > 0
    # Every member needs every block document: the whole block is shared prefix.
    for g in measured["per_group"]:
        assert g["n_questions"] <= 3
        if g["n_pairs"]:
            assert g["mean_pairwise_shared_fraction"] == pytest.approx(1.0)
            assert g["shared_prefix_tokens_est"] > 0
            assert g["mean_pairwise_shared_tokens_est"] > 0


def test_engineered_hits_partial_overlap_target() -> None:
    pool = make_chain_pool()
    target = 1 / 3  # reachable exactly: pair one question with a chain neighbor
    m = build_manifest(pool, num_queries=4, num_trials=1, seed=42, block_budget=BUDGET,
                       overlap_target=target, overlap_tolerance=0.05,
                       overlap_group_size=2)
    realized = m["overlap"]["measured"]["mean_pairwise_shared_fraction"]
    assert realized == pytest.approx(target, abs=0.05)
    # The manifest is still a full yardstick: blocks, pool, trials all intact.
    assert m["stats"]["pool_size"] >= 4
    assert len(m["trials"]["1"]) == 4
    assert all(b["token_count"] <= BUDGET for b in m["blocks"])
    for ex_id in m["trials"]["1"]:
        assert ex_id in m["question_to_block"]


def test_engineered_group_is_the_block() -> None:
    """One planned query group per corpus block: block members' gold sets are
    subsets of the block and the group sizes respect the cap."""
    pool = make_chain_pool()
    m = build_manifest(pool, num_queries=4, num_trials=1, seed=7, block_budget=BUDGET,
                       overlap_target=1.0, overlap_tolerance=0.05,
                       overlap_group_size=3)
    by_id = {ex.id: ex for ex in pool}
    members: Dict[int, List[str]] = {}
    for ex_id, b in m["question_to_block"].items():
        members.setdefault(b, []).append(ex_id)
    for b, ids in members.items():
        assert len(ids) <= 3
        text = m["blocks"][b]["text"]
        for ex_id in ids:
            for p in gold_set(by_id[ex_id]):
                assert p in text


# ------------------------------------------------------ natural mode: still measured

def make_squad_pool(n_paragraphs: int = 8, q_per_para: int = 5) -> List[FakeExample]:
    out = []
    for p in range(n_paragraphs):
        para = _para(f"n{p}", words=60)
        for q in range(q_per_para):
            out.append(FakeExample(id=f"ex_{p}_{q}", question=f"Q{p}.{q}?",
                                   context=[para]))
    return out


def test_natural_mode_measures_overlap_never_assumes() -> None:
    # 60-word paragraphs ≈ 80 tokens; budget 300 packs 3 paragraph-families/block:
    # per block 15 questions, 105 pairs, 30 same-family (J=1) → mean 30/105.
    m = build_manifest(make_squad_pool(), num_queries=10, num_trials=1, seed=42,
                       block_budget=300)
    ov = m["overlap"]
    assert ov["mode"] == "natural"
    assert ov["target_shared_fraction"] is None    # no knob: nothing was engineered
    assert ov["tolerance"] is None and ov["group_size"] is None
    measured = ov["measured"]
    assert measured["mean_pairwise_shared_fraction"] == pytest.approx(30 / 105)
    assert measured["mean_pairwise_shared_tokens_est"] > 0
    # No block document is needed by EVERY member (families differ) → no shared prefix.
    assert measured["mean_shared_prefix_tokens_est"] == 0.0
    assert len(measured["per_group"]) == measured["n_groups"] == len(m["blocks"])


def test_natural_mode_no_pairs_is_none_with_reason() -> None:
    """Absence stays absence: single-question groups yield None means + a named
    reason, never a fabricated 0.0."""
    pool = [FakeExample(id=f"u{k}", question=f"U{k}?", context=[_para(f"u{k}", 60)])
            for k in range(12)]
    # Budget 150 fits exactly one ~80-token paragraph per block → all singletons.
    m = build_manifest(pool, num_queries=4, num_trials=1, seed=42, block_budget=150)
    measured = m["overlap"]["measured"]
    assert measured["n_pairs"] == 0
    assert measured["mean_pairwise_shared_fraction"] is None
    assert measured["mean_pairwise_shared_tokens_est"] is None
    assert measured["mean_shared_prefix_tokens_est"] is None
    assert "no_pairs_reason" in measured


# ------------------------------------------------------------------- fail-closed

def test_unreachable_target_is_named_refusal() -> None:
    with pytest.raises(OverlapError, match="UNREACHABLE"):
        build_manifest(make_disjoint_singletons(), num_queries=3, num_trials=1,
                       seed=42, block_budget=BUDGET,
                       overlap_target=0.6, overlap_tolerance=0.05,
                       overlap_group_size=3)


def test_malformed_knobs_are_named_refusals() -> None:
    pool = make_chain_pool(n_families=3, q_per_family=3)
    with pytest.raises(OverlapError, match="not a fraction"):
        build_manifest(pool, num_queries=2, num_trials=1, seed=42,
                       block_budget=BUDGET, overlap_target=1.5)
    with pytest.raises(OverlapError, match="group_size"):
        build_manifest(pool, num_queries=2, num_trials=1, seed=42,
                       block_budget=BUDGET, overlap_target=0.5,
                       overlap_group_size=1)
    with pytest.raises(OverlapError, match="tolerance"):
        build_manifest(pool, num_queries=2, num_trials=1, seed=42,
                       block_budget=BUDGET, overlap_target=0.5,
                       overlap_tolerance=0.0)


def test_engineered_oversized_family_excluded_and_counted() -> None:
    pool = make_chain_pool(n_families=4, q_per_family=3)
    giant = _para("giant", words=1000)  # ~1333 tokens: exceeds the budget alone
    pool += [FakeExample(id=f"g{q}", question=f"G{q}?", context=[giant])
             for q in range(2)]
    m = build_manifest(pool, num_queries=4, num_trials=1, seed=42, block_budget=BUDGET,
                       overlap_target=1.0, overlap_tolerance=0.05,
                       overlap_group_size=3, pool_target=14)
    assert m["stats"]["examples_excluded"] == 2
    assert all(not i.startswith("g") for i in m["question_to_block"])


# ------------------------------------------------------------------ determinism

def test_engineered_determinism_under_seed() -> None:
    pool = make_chain_pool()
    kwargs = dict(num_queries=4, num_trials=2, block_budget=BUDGET,
                  overlap_target=1 / 3, overlap_tolerance=0.05,
                  overlap_group_size=2)
    m1 = build_manifest(pool, seed=42, **kwargs)
    m2 = build_manifest(pool, seed=42, **kwargs)
    assert m1 == m2
    m3 = build_manifest(pool, seed=43, **kwargs)
    assert m3 != m1


def test_planner_is_pure_and_deterministic() -> None:
    pool = make_chain_pool()
    g1, x1 = plan_overlap_groups(pool, target=1.0, group_size=3,
                                 block_budget=BUDGET, seed=42, pool_target=12)
    g2, x2 = plan_overlap_groups(pool, target=1.0, group_size=3,
                                 block_budget=BUDGET, seed=42, pool_target=12)
    assert [[e.id for e in g] for g in g1] == [[e.id for e in g] for g in g2]
    assert x1 == x2 == []
    assert sum(len(g) for g in g1) >= 12
