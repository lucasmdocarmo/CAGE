"""Uniform-yardstick manifest (src/data/manifest.py) — synthetic, no datasets/torch."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import (  # noqa: E402
    ManifestError,
    block_for,
    build_manifest,
    select_examples,
)


@dataclass
class FakeExample:
    id: str
    question: str
    context: List[str]
    answer: str = "x"
    metadata: Dict = field(default_factory=dict)


def make_pool(n_paragraphs: int = 8, q_per_para: int = 5,
              words_per_para: int = 60) -> List[FakeExample]:
    out = []
    for p in range(n_paragraphs):
        para = " ".join(f"p{p}w{w}" for w in range(words_per_para))
        for q in range(q_per_para):
            out.append(FakeExample(id=f"ex_{p}_{q}", question=f"Q{p}.{q}?", context=[para]))
    return out


# words*4//3 heuristic: 60 words ≈ 80 tokens/paragraph; budget 300 ≈ 3 paragraphs/block
BUDGET = 300


def test_determinism_same_seed_identical_manifest() -> None:
    pool = make_pool()
    m1 = build_manifest(pool, num_queries=10, num_trials=3, seed=42, block_budget=BUDGET)
    m2 = build_manifest(pool, num_queries=10, num_trials=3, seed=42, block_budget=BUDGET)
    assert m1 == m2


def test_different_seed_different_selection() -> None:
    pool = make_pool()
    m1 = build_manifest(pool, num_queries=10, num_trials=1, seed=42, block_budget=BUDGET)
    m2 = build_manifest(pool, num_queries=10, num_trials=1, seed=43, block_budget=BUDGET)
    assert m1["trials"]["1"] != m2["trials"]["1"]


def test_blocks_respect_budget_and_cover_every_selected_question() -> None:
    pool = make_pool()
    m = build_manifest(pool, num_queries=10, num_trials=3, seed=42, block_budget=BUDGET)
    assert all(b["token_count"] <= BUDGET for b in m["blocks"])
    for t in ("1", "2", "3"):
        assert len(m["trials"][t]) == 10
        for ex_id in m["trials"][t]:
            assert block_for(m, ex_id)["token_count"] <= BUDGET


def test_trials_disjoint_when_pool_allows() -> None:
    pool = make_pool(n_paragraphs=12)  # pool 60 >= 10*3
    m = build_manifest(pool, num_queries=10, num_trials=3, seed=42, block_budget=BUDGET,
                       pool_target=60)
    assert m["stats"]["trials_disjoint"] is True
    t1, t2, t3 = (set(m["trials"][t]) for t in ("1", "2", "3"))
    assert not (t1 & t2) and not (t1 & t3) and not (t2 & t3)


def test_oversized_paragraph_excluded_and_counted() -> None:
    pool = make_pool(n_paragraphs=6)
    giant_para = " ".join(f"g{w}" for w in range(1000))  # ~1333 tokens > budget alone
    pool += [FakeExample(id=f"giant_{q}", question=f"G{q}?", context=[giant_para])
             for q in range(3)]
    m = build_manifest(pool, num_queries=10, num_trials=2, seed=42, block_budget=BUDGET)
    assert m["stats"]["examples_excluded"] >= 3
    assert all(not i.startswith("giant_") for t in m["trials"].values() for i in t)


def test_pool_too_small_raises() -> None:
    pool = make_pool(n_paragraphs=1, q_per_para=3)
    with pytest.raises(ManifestError):
        build_manifest(pool, num_queries=10, num_trials=3, seed=42, block_budget=BUDGET)


def test_select_examples_order_and_missing_id() -> None:
    pool = make_pool()
    m = build_manifest(pool, num_queries=10, num_trials=2, seed=42, block_budget=BUDGET)
    picked = select_examples(m, 1, pool)
    assert [ex.id for ex in picked] == m["trials"]["1"]  # manifest order preserved
    with pytest.raises(ManifestError):
        select_examples(m, 1, pool[:5])  # dataset mismatch -> loud failure
    with pytest.raises(ManifestError):
        select_examples(m, 9, pool)      # no such trial


def test_same_paragraph_questions_share_a_block() -> None:
    pool = make_pool()
    m = build_manifest(pool, num_queries=10, num_trials=1, seed=42, block_budget=BUDGET)
    q2b = m["question_to_block"]
    by_para: Dict[str, set] = {}
    for ex_id, b in q2b.items():
        para = ex_id.split("_")[1]  # ex_<p>_<q>
        by_para.setdefault(para, set()).add(b)
    assert all(len(bs) == 1 for bs in by_para.values())


def _gold_only(ex: FakeExample) -> List[str]:
    titles = ex.metadata.get("supporting_titles") or []
    return [c for c in ex.context if any(c.startswith(f"{t}: ") for t in titles)]


def test_context_selector_strips_distractors_before_grouping_and_packing() -> None:
    """HotpotQA/MuSiQue-shaped regression: examples share the SAME gold paragraphs but
    each carries its OWN (unique, distractor) paragraphs in .context, exactly like the
    real loaders (loader.py keeps gold + distractors, gold recoverable via
    metadata['supporting_titles']). Without gold-filtering, build_manifest used to key
    grouping/packing off the raw (mostly-distractor) context, so these examples never
    grouped and each burned its own block (~1 example/block, defeating cross-question
    corpus reuse). With context_selector=gold_only, they must all land in ONE block
    built from gold paragraphs only.
    """
    gold1 = "Alpha: " + " ".join(f"a{w}" for w in range(20))
    gold2 = "Beta: " + " ".join(f"b{w}" for w in range(20))

    def make_ex(i: int) -> FakeExample:
        distractors = [
            "D" + str(i) + "_" + str(j) + ": " + " ".join(f"x{w}" for w in range(20))
            for j in range(8)
        ]
        return FakeExample(
            id=f"ex_{i}", question=f"Q{i}?", context=[gold1, gold2] + distractors,
            metadata={"supporting_titles": ["Alpha", "Beta"]},
        )

    pool = [make_ex(i) for i in range(6)]
    # One example's full raw context (2 gold + 8 unique distractors) is ~316 tokens;
    # two examples' worth is ~561. A 350 budget therefore fits exactly one example's
    # raw paragraphs per block -- reproducing the live-verified MuSiQue degeneration
    # (pool_size == n_blocks, a 1:1 example:block ratio) deterministically.
    RAW_BUDGET = 350

    # WITHOUT the selector: unique-per-question distractors mean no two examples
    # share a raw context tuple, so grouping is a no-op and every example fights for
    # its own block (degenerate ~1-example-per-block).
    m_raw = build_manifest(pool, num_queries=6, num_trials=1, seed=42, block_budget=RAW_BUDGET)
    assert m_raw["stats"]["n_blocks"] == 6
    assert m_raw["stats"]["pool_size"] == 6

    # WITH the selector: filtered down to the shared gold pair, all 6 group and pack
    # into ONE block, and that block's text carries no distractor content.
    m = build_manifest(pool, num_queries=6, num_trials=1, seed=42, block_budget=RAW_BUDGET,
                       context_selector=_gold_only)
    assert m["stats"]["n_blocks"] == 1
    assert m["stats"]["pool_size"] == 6
    assert m["question_to_block"] == {f"ex_{i}": 0 for i in range(6)}
    block_text = m["blocks"][0]["text"]
    assert "Alpha:" in block_text and "Beta:" in block_text
    assert "D0_" not in block_text and "D5_" not in block_text


def test_context_selector_defaults_to_unfiltered_context() -> None:
    """No context_selector -> unchanged behavior (SQuAD-shaped default)."""
    pool = make_pool()
    m_default = build_manifest(pool, num_queries=10, num_trials=1, seed=42, block_budget=BUDGET)
    m_identity = build_manifest(pool, num_queries=10, num_trials=1, seed=42,
                                block_budget=BUDGET, context_selector=lambda ex: ex.context)
    assert m_default == m_identity


# ---------------------------------------------------------------------------
# ADR-0106 / charter §7.7(d): B12 corpus-truncation rungs (descending budgets,
# same seed, same packing order, no repacking; every query labeled)
# ---------------------------------------------------------------------------

from src.data.manifest import trunc_rung_for  # noqa: E402

# words*4//3: header 7 words; each Document = 62 words. 3 docs -> 257 tokens
# (<= BUDGET 300); at rung 200 only 2 docs fit (174); at rung 100 only 1 (92).
RUNGS = (200, 100)


def test_trunc_rungs_default_is_empty_and_leaves_the_artifact_unchanged() -> None:
    pool = make_pool()
    plain = build_manifest(pool, num_queries=10, num_trials=2, seed=42, block_budget=BUDGET)
    assert plain["trunc_rungs"] == {}
    laddered = build_manifest(pool, num_queries=10, num_trials=2, seed=42,
                              block_budget=BUDGET, trunc_budgets=RUNGS)
    # The full-budget store, mapping and draws are byte-identical: the ladder
    # is derived FROM the packed blocks, never a repack.
    for key in ("blocks", "question_to_block", "trials", "overlap", "stats"):
        assert laddered[key] == plain[key], key


def test_trunc_rungs_keep_packing_order_and_label_every_query() -> None:
    pool = make_pool()
    by_id = {ex.id: ex for ex in pool}
    m = build_manifest(pool, num_queries=10, num_trials=2, seed=42,
                       block_budget=BUDGET, trunc_budgets=RUNGS)
    assert list(m["trunc_rungs"]) == ["200", "100"]  # descending, as registered
    for b_str, rung in m["trunc_rungs"].items():
        budget = int(b_str)
        assert rung["budget"] == budget
        assert len(rung["blocks"]) == len(m["blocks"])
        for full_block, rung_block in zip(m["blocks"], rung["blocks"]):
            assert rung_block["block_id"] == full_block["block_id"]
            assert rung_block["token_count"] <= budget
            assert 0 < rung_block["n_paragraphs"] < full_block["n_paragraphs"]
            # Packing order preserved: the rung text is a literal prefix of the
            # full block text (a prefix of the Documents, same numbering).
            assert full_block["text"].startswith(rung_block["text"])
        in_ids = set(rung["in_corpus_ids"])
        assert rung["n_in_corpus"] == len(in_ids) == len(rung["in_corpus_ids"])
        assert rung["n_in_corpus"] + rung["n_out_of_corpus"] == len(m["question_to_block"])
        assert rung["n_out_of_corpus"] > 0  # the ladder actually truncates
        # Label correctness by construction: in-corpus iff the gold paragraph
        # survived in the query's own rung block.
        for ex_id, block_id in m["question_to_block"].items():
            text = rung["blocks"][block_id]["text"]
            survived = all(p in text for p in by_id[ex_id].context)
            assert (ex_id in in_ids) == survived, (b_str, ex_id)
        # Every pool id is labeled exactly once (in or out), never dropped.
        assert in_ids <= set(m["question_to_block"])
    # Monotone ladder: the lower rung keeps a subset of the higher rung.
    assert set(m["trunc_rungs"]["100"]["in_corpus_ids"]) < set(
        m["trunc_rungs"]["200"]["in_corpus_ids"]
    )


def test_trunc_rungs_are_deterministic() -> None:
    pool = make_pool()
    m1 = build_manifest(pool, num_queries=10, num_trials=2, seed=42,
                        block_budget=BUDGET, trunc_budgets=RUNGS)
    m2 = build_manifest(pool, num_queries=10, num_trials=2, seed=42,
                        block_budget=BUDGET, trunc_budgets=RUNGS)
    assert m1["trunc_rungs"] == m2["trunc_rungs"]


@pytest.mark.parametrize(
    "bad",
    [
        (BUDGET,),          # a rung equal to the block budget is B3's own cell
        (BUDGET + 1,),      # above the block budget: not a truncation
        (0,),               # non-positive
        (200, 200),         # duplicate rung
        (100, 200),         # not descending
        (200.0,),           # not an int
    ],
)
def test_trunc_rungs_refuse_bad_ladders(bad: tuple) -> None:
    pool = make_pool()
    with pytest.raises(ManifestError):
        build_manifest(pool, num_queries=10, num_trials=2, seed=42,
                       block_budget=BUDGET, trunc_budgets=bad)


def test_trunc_rung_lookup_fails_closed() -> None:
    pool = make_pool()
    m = build_manifest(pool, num_queries=10, num_trials=2, seed=42,
                       block_budget=BUDGET, trunc_budgets=RUNGS)
    assert trunc_rung_for(m, 200) is m["trunc_rungs"]["200"]
    with pytest.raises(ManifestError, match="150"):
        trunc_rung_for(m, 150)
    plain = build_manifest(pool, num_queries=10, num_trials=2, seed=42, block_budget=BUDGET)
    with pytest.raises(ManifestError, match="no truncation rungs"):
        trunc_rung_for(plain, 200)


def test_trial_ids_are_prefix_stable_for_nested_cells() -> None:
    """A9 per-row N: a cell measuring n < N reads the FIRST n ids of each
    trial in manifest order, so smaller cells are tested prefix subsets of
    the manifest's draw (and stay pairwise disjoint across trials when the
    pool allows). The manifest's trial lists are therefore the ordered
    source of truth: select_examples must preserve that order verbatim."""
    pool = make_pool(n_paragraphs=12)  # pool 60 >= 10*3
    m = build_manifest(pool, num_queries=10, num_trials=3, seed=42, block_budget=BUDGET,
                       pool_target=60)
    for t in ("1", "2", "3"):
        ordered = [ex.id for ex in select_examples(m, int(t), pool)]
        assert ordered == m["trials"][t]
        for n in (1, 4, 10):
            assert ordered[:n] == m["trials"][t][:n]
    prefixes = [set(m["trials"][t][:4]) for t in ("1", "2", "3")]
    assert not (prefixes[0] & prefixes[1]) and not (prefixes[0] & prefixes[2])
    assert not (prefixes[1] & prefixes[2])
