"""Unit tests for the RULER charter-subset tasks (charter D5 item 5).

Covers the three tasks added on top of the original NIAH pair —
``niah_multiquery``, ``variable_tracking``, ``qa`` — plus the native
per-task scoring contract (``score_native`` / ``score_example``). Pure
stdlib like the base instrument: the ``qa`` task is exercised through a
TINY deterministic in-memory fixture pool injected via ``qa_source``
(clearly test-only; real runs use the already-local SQuAD v2 through the
loader default). Covers determinism under seed, the SHAPE-32K length pin,
per-task reporting fields, and scoring correctness including a
wrong-answer case per task — all fail-closed paths raise NAMED errors.
"""

import random

import pytest

from src.data.loader import CAGExample, get_loader
from src.data.ruler import (
    MAX_CONTEXT_TOKENS,
    NATIVE_METRICS,
    NOISE_SENTENCE,
    RulerLoader,
    _TASKS,
    score_example,
    score_native,
)


# ---------------------------------------------------------------------------
# TEST-ONLY qa fixture pool (deterministic stand-in for SquadV2Loader output)
# ---------------------------------------------------------------------------


def _paragraph(topic: str, words: int = 30) -> str:
    """Distinct deterministic filler paragraph (~``words`` whitespace tokens)."""
    return " ".join(f"{topic}w{i}" for i in range(words - 1)) + " end."


def _qa_fixture_pool(n_answerable: int = 10, words: int = 30):
    """Tiny SQuAD-v2-shaped pool: answerable rows + rows the filter must drop."""
    pool = [
        CAGExample(
            id=f"fix_{i}",
            question=f"What is the codeword for site number {i}?",
            context=[_paragraph(f"site{i}", words)],
            answer=f"magicword-{i}-x",
            metadata={
                "is_impossible": False,
                "all_answers": [f"magicword-{i}-x", f"aliasword-{i}-y"],
            },
        )
        for i in range(n_answerable)
    ]
    pool.append(  # unanswerable: empty answer, must be filtered out
        CAGExample(
            id="fix_unanswerable", question="Unanswerable?", context=[_paragraph("nowhere")],
            answer="", metadata={"is_impossible": True, "all_answers": []},
        )
    )
    pool.append(  # empty context: must be filtered out
        CAGExample(
            id="fix_empty_ctx", question="No context?", context=[""],
            answer="ghost", metadata={"is_impossible": False, "all_answers": ["ghost"]},
        )
    )
    return pool


def _loader(task: str, **kw) -> RulerLoader:
    kw.setdefault("context_length_tokens", 256)
    if task == "qa":
        kw.setdefault("qa_source", _qa_fixture_pool)
        kw.setdefault("qa_source_name", "test-fixture")
    return RulerLoader(task=task, **kw)


_NEW_TASKS = ("niah_multiquery", "variable_tracking", "qa")


# ---------------------------------------------------------------------------
# Registration: charter D5#5 subset
# ---------------------------------------------------------------------------


def test_charter_subset_is_registered():
    """4 charter tasks (NIAH-MK/MQ, VT, QA) + the kept niah_single default."""
    assert _TASKS == (
        "niah_single", "niah_multikey", "niah_multiquery",
        "variable_tracking", "qa",
    )
    assert set(NATIVE_METRICS) == set(_TASKS)


def test_env_var_selects_new_tasks_via_factory(monkeypatch):
    monkeypatch.setenv("CAGE_RULER_TASK", "variable_tracking")
    monkeypatch.setenv("CAGE_RULER_CONTEXT_TOKENS", "256")
    loader = get_loader("ruler", seed=5)
    assert isinstance(loader, RulerLoader)
    assert loader.task == "variable_tracking"


def test_qa_default_source_is_local_squad_v2():
    """Default (no injection) pins the already-local SQuAD v2 validation split."""
    loader = RulerLoader(task="qa", context_length_tokens=256)
    assert loader.qa_source_name == "squad_v2:validation"


# ---------------------------------------------------------------------------
# Determinism under seed (all new tasks)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", _NEW_TASKS)
def test_same_seed_reproduces_identical_items(task):
    a = _loader(task, seed=42).load(max_examples=3)
    b = _loader(task, seed=42).load(max_examples=3)

    assert [ex.id for ex in a] == [ex.id for ex in b]
    assert [ex.question for ex in a] == [ex.question for ex in b]
    assert [ex.answer for ex in a] == [ex.answer for ex in b]
    assert [ex.context for ex in a] == [ex.context for ex in b]
    assert [ex.metadata["gold_answers"] for ex in a] == [ex.metadata["gold_answers"] for ex in b]


@pytest.mark.parametrize("task", _NEW_TASKS)
def test_different_seeds_draw_different_items(task):
    a = _loader(task, seed=42).load(max_examples=3)
    b = _loader(task, seed=43).load(max_examples=3)

    assert [ex.context for ex in a] != [ex.context for ex in b]


@pytest.mark.parametrize("task", _NEW_TASKS)
def test_items_are_prefix_stable_under_max_examples(task):
    """Item i derives from (seed, i): drawing more never changes earlier ones."""
    few = _loader(task, seed=7).load(max_examples=2)
    many = _loader(task, seed=7).load(max_examples=5)

    assert [ex.id for ex in few] == [ex.id for ex in many[:2]]
    assert [ex.answer for ex in few] == [ex.answer for ex in many[:2]]
    assert [ex.context for ex in few] == [ex.context for ex in many[:2]]


# ---------------------------------------------------------------------------
# SHAPE-32K length pin (cap refused, target never exceeded)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", _NEW_TASKS)
def test_charter_input_cap_is_enforced_per_task(task):
    _loader(task, context_length_tokens=MAX_CONTEXT_TOKENS)  # cap itself: OK
    with pytest.raises(ValueError, match="32768"):
        _loader(task, context_length_tokens=MAX_CONTEXT_TOKENS + 1)


@pytest.mark.parametrize("task", ("niah_multiquery", "variable_tracking"))
def test_synthetic_tasks_hit_target_without_exceeding_it(task):
    target = 512
    for ex in _loader(task, seed=1, context_length_tokens=target).load(max_examples=3):
        actual = ex.metadata["actual_context_tokens"]
        assert actual == len(ex.context[0].split())  # whitespace proxy recorded
        assert actual <= target
        assert actual >= int(0.9 * target)
        assert NOISE_SENTENCE in ex.context[0]


def test_qa_haystack_fills_toward_target_without_exceeding_it():
    target = 256
    for ex in _loader("qa", seed=1, context_length_tokens=target).load(max_examples=3):
        actual = ex.metadata["actual_context_tokens"]
        assert actual <= target
        assert actual >= int(0.8 * target)  # 30-word paragraphs pack to >=80%


# ---------------------------------------------------------------------------
# niah_multiquery — RULER multi-query semantics
# ---------------------------------------------------------------------------


def test_multiquery_needles_and_single_query():
    ex = _loader("niah_multiquery", seed=3, num_queries=4).load(max_examples=1)[0]
    md = ex.metadata

    assert md["num_queries"] == 4
    assert len(md["needle_keys"]) == len(set(md["needle_keys"])) == 4
    assert len(md["needle_depths"]) == 4
    assert len(md["gold_answers"]) == 4
    # ONE query covering every needle key; every needle is in the haystack.
    for key, value in zip(md["needle_keys"], md["gold_answers"]):
        assert key in ex.question
        assert f"One of the special magic numbers for {key} is: {value}." in ex.context[0]
    assert ex.answer == ", ".join(md["gold_answers"])
    # NOT the alias key: one-of-N credit would be wrong for multiquery.
    assert "all_answers" not in md


def test_multiquery_scoring_is_mean_over_all_needles():
    ex = _loader("niah_multiquery", seed=3, num_queries=4).load(max_examples=1)[0]
    golds = ex.metadata["gold_answers"]

    assert score_example(ex, "the numbers are " + ", ".join(golds)) == 1.0
    assert score_example(ex, f"only {golds[0]} and {golds[1]}") == 0.5  # partial credit
    assert score_example(ex, "no magic numbers here 0000000") == 0.0  # wrong answer


def test_multiquery_refuses_degenerate_single_query():
    with pytest.raises(ValueError, match="num_queries"):
        _loader("niah_multiquery", num_queries=1)


# ---------------------------------------------------------------------------
# variable_tracking — chained hops, answer = final binding
# ---------------------------------------------------------------------------


def test_vt_chain_appears_in_order_and_binds_final_variable():
    ex = _loader("variable_tracking", seed=11, num_hops=4).load(max_examples=1)[0]
    md = ex.metadata
    chain = md["chain_variables"]
    text = ex.context[0]

    assert len(chain) == md["num_hops"] + 1 == 5
    statements = [f"VAR {chain[0]} = {ex.answer}."] + [
        f"VAR {chain[i]} = VAR {chain[i - 1]}." for i in range(1, len(chain))
    ]
    positions = [text.find(s) for s in statements]
    assert all(p >= 0 for p in positions)  # every hop present
    assert positions == sorted(positions)  # hops IN ORDER (traceable chain)
    # The query targets the LAST variable; the answer is the initial binding.
    assert md["query_variable"] == chain[-1]
    assert md["query_variable"] in ex.question
    assert ex.answer.isdigit() and len(ex.answer) == 7
    assert md["gold_answers"] == [ex.answer]


def test_vt_decoy_chains_never_alias_the_gold_value():
    ex = _loader("variable_tracking", seed=11, num_hops=2, num_chains=3).load(max_examples=1)[0]
    md = ex.metadata

    assert md["num_chains"] == 3
    assert len(md["decoy_variables"]) == 2 * 3  # 2 decoy chains x (hops+1) vars
    assert not set(md["decoy_variables"]) & set(md["chain_variables"])
    for var in md["decoy_variables"]:
        assert f"VAR {var} " in ex.context[0]
    # Decoy values differ from the gold: its binding appears exactly once.
    assert ex.context[0].count(ex.answer) == 1


def test_vt_scoring_right_and_wrong_answers():
    ex = _loader("variable_tracking", seed=11).load(max_examples=1)[0]

    assert score_example(ex, f"The value is {ex.answer}.") == 1.0
    assert score_example(ex, "The value is 0000000.") == 0.0  # wrong answer


def test_vt_refuses_invalid_chain_shape():
    with pytest.raises(ValueError, match="num_hops"):
        _loader("variable_tracking", num_hops=0)
    with pytest.raises(ValueError, match="num_chains"):
        _loader("variable_tracking", num_chains=0)


def test_vt_chain_order_holds_across_items_and_seeds_at_short_context():
    """Regression sweep: the in-order invariant must hold on EVERY item.

    The old ``_assemble`` recomputed slots against a list that grew during
    insertion, inverting depth pairs closer than ~1/n_noise: under these
    seeds it broke 22/120 items at ctx=256, 6/120 at 1024, 2/120 at the
    4096 env default (the audit measured 27/120 at 256 under its own
    seeds). One-seed spot checks cannot see it; 120 items can.
    """
    for seed in (0, 1, 2):
        loader = _loader("variable_tracking", seed=seed, num_hops=4)
        for ex in loader.load(max_examples=40):
            chain = ex.metadata["chain_variables"]
            text = ex.context[0]
            statements = [f"VAR {chain[0]} = {ex.answer}."] + [
                f"VAR {chain[i]} = VAR {chain[i - 1]}." for i in range(1, len(chain))
            ]
            positions = [text.find(s) for s in statements]
            assert all(p >= 0 for p in positions)
            assert positions == sorted(positions), (
                f"seed={seed} item={ex.metadata['item_index']}: hops out of order"
            )


def test_vt_every_chain_stays_in_order_with_decoys():
    """Decoy chains carry the same per-chain in-order property as the gold
    chain (12 statements over ~13 noise slots at ctx=256 — heavy slot
    collisions, exercising the fixed-slot tie-breaking)."""
    loader = _loader("variable_tracking", seed=5, num_hops=3, num_chains=3)
    for ex in loader.load(max_examples=10):
        md = ex.metadata
        text = ex.context[0]
        chain_len = md["num_hops"] + 1
        chains = [md["chain_variables"]] + [
            md["decoy_variables"][i:i + chain_len]
            for i in range(0, len(md["decoy_variables"]), chain_len)
        ]
        for chain in chains:
            # "VAR {name} = " matches only the assignment TO name (each
            # variable is assigned exactly once; RHS mentions end in ".").
            positions = [text.find(f"VAR {name} = ") for name in chain]
            assert all(p >= 0 for p in positions)
            assert positions == sorted(positions)


def test_assemble_never_inverts_adjacent_depths():
    """Direct regression on ``_assemble``: depth order must survive depths
    far closer than ~1/n_noise (the old growing-list scheme inverted such
    pairs; e.g. depths 0.415/0.42 at n_noise=13 came out swapped)."""
    loader = _loader("variable_tracking")  # ctx=256 -> n_noise ~ 13
    rng = random.Random(0)
    for _ in range(200):
        d = rng.random() * 0.9
        depths = [d, d + rng.random() * 0.08]
        text = loader._assemble(rng, ["XNEEDLEAX ONE.", "XNEEDLEBX TWO."], depths)
        assert -1 < text.find("XNEEDLEAX") < text.find("XNEEDLEBX")


def _assemble_grow_reference(loader, needles, depths):
    """TEST-ONLY replica of the pre-Wave-4 grow-during-insertion assembly
    (HEAD 1dea744 ``_assemble``): the byte-level baseline the fixed-slot
    rewrite's disclosed behavior change is measured against below."""
    per_noise = max(1, loader._count(NOISE_SENTENCE + " "))
    needle_tokens = sum(loader._count(n + " ") for n in needles)
    budget = loader.context_length_tokens - needle_tokens
    n_noise = max(1, budget // per_noise)
    sentences = [NOISE_SENTENCE] * n_noise
    placements = sorted(
        ((d, needle) for d, needle in zip(depths, needles)),
        key=lambda p: p[0], reverse=True,
    )
    for depth, needle in placements:
        idx = round(depth * len(sentences))
        sentences.insert(min(idx, len(sentences)), needle)
    text = " ".join(sentences)
    while loader._count(text) > loader.context_length_tokens and len(sentences) > len(needles):
        for i in range(len(sentences) - 1, -1, -1):
            if sentences[i] == NOISE_SENTENCE:
                del sentences[i]
                break
        else:
            break
        text = " ".join(sentences)
    return text


def test_single_needle_assembly_is_byte_identical_to_pre_wave4():
    """Byte-stability HOLDS exactly where claimed: with one needle the old
    list never grew before the slot was computed, so the fixed-slot rewrite
    reproduces pre-Wave-4 ``niah_single`` haystacks byte-for-byte at every
    depth (item-level check: 60/60 identical at ctx=256/1024/4096, seed 42)."""
    loader = _loader("niah_single")
    rng = random.Random(0)
    for _ in range(200):
        depth = rng.random()
        needles = ["XNEEDLEAX ONE."]
        assert loader._assemble(rng, needles, [depth]) == _assemble_grow_reference(
            loader, needles, [depth]
        )


def test_multikey_layout_change_vs_pre_wave4_is_disclosed_and_bounded():
    """DISCLOSURE PIN for the fixed-slot rewrite (see ``_assemble``): with
    multiple needles the old scheme slotted each against an already-grown
    list, so the rewrite MOVES needles on most multi-needle haystacks
    (item-level measurement vs HEAD: 57/60 differ at ctx=256, 54/60 at
    1024, 53/60 at 4096 under seed 42). Only the noise/needle interleaving
    moves — every needle survives exactly once in both layouts, and ids/
    questions/answers/RNG draws are untouched (draw order is verbatim)."""
    loader = _loader("niah_multikey")
    rng = random.Random(0)
    diffs = 0
    for _ in range(60):
        needles = [f"XNEEDLE{c}X {c}VALUE." for c in "ABCD"]  # 1 target + 3 distractors
        depths = [rng.random() for _ in needles]
        new = loader._assemble(rng, needles, depths)
        old = _assemble_grow_reference(loader, needles, depths)
        for needle in needles:
            assert new.count(needle) == 1 and old.count(needle) == 1
        if new != old:
            diffs += 1
    # The change is real and majority-scale; a silent revert to the old
    # bytes must trip this LOUDLY (order correctness itself is enforced by
    # test_assemble_never_inverts_adjacent_depths above).
    assert diffs > 30, f"expected majority layout divergence, saw {diffs}/60"


# ---------------------------------------------------------------------------
# qa — real QA pairs in a real-paragraph distractor haystack
# ---------------------------------------------------------------------------


def test_qa_embeds_gold_pair_among_real_distractor_paragraphs():
    pool = _qa_fixture_pool()
    by_id = {ex.id: ex for ex in pool}
    ex = _loader("qa", seed=2).load(max_examples=1)[0]
    md = ex.metadata

    source = by_id[md["qa_source_id"]]  # picked pair comes from the pool
    assert ex.question == source.question
    assert ex.answer == source.answer
    assert source.context[0] in ex.context[0]  # gold paragraph never trimmed
    assert md["num_distractor_paragraphs"] >= 1
    others = [p.context[0] for p in pool if p.id != md["qa_source_id"] and p.context[0].strip()]
    assert any(p in ex.context[0] for p in others)  # real distractor text present
    assert 0.0 <= md["gold_depth"] <= 1.0
    assert md["qa_source"] == "test-fixture"
    # Aliases: primary answer first, downstream max-over-golds key exported.
    assert md["gold_answers"][0] == source.answer
    assert set(source.metadata["all_answers"]) <= set(md["gold_answers"])
    assert md["all_answers"] == md["gold_answers"]


def test_qa_scoring_accepts_any_alias_and_rejects_wrong_answers():
    ex = _loader("qa", seed=2).load(max_examples=1)[0]
    golds = ex.metadata["gold_answers"]

    assert score_example(ex, f"Answer: {golds[0]}") == 1.0
    assert score_example(ex, f"it was {golds[-1].upper()}") == 1.0  # alias, any case
    assert score_example(ex, "completely unrelated output") == 0.0  # wrong answer


def test_qa_filters_out_unanswerable_and_empty_context_rows():
    ex = _loader("qa", seed=2, num_items=20).load()[0]
    loader = _loader("qa", seed=2)
    pool, paragraphs = loader._qa_pool()

    assert all(not p.metadata["is_impossible"] for p in pool)
    assert all(p.answer for p in pool)
    assert len(paragraphs) == len(set(paragraphs))
    assert "ghost" not in [p.answer for p in pool]
    assert ex.metadata["qa_source_id"] != "fix_unanswerable"


def test_qa_refuses_source_without_answerable_pairs():
    only_bad = lambda: _qa_fixture_pool(n_answerable=1)  # 1 gold, no distractor text
    with pytest.raises(ValueError, match="answerable"):
        _loader("qa", qa_source=only_bad).load(max_examples=1)


def test_qa_refuses_gold_paragraph_that_cannot_fit():
    giant = lambda: _qa_fixture_pool(n_answerable=4, words=400)
    with pytest.raises(ValueError, match="does not fit"):
        _loader("qa", qa_source=giant, context_length_tokens=256).load(max_examples=1)


def test_qa_refuses_haystack_with_zero_distractors():
    two_smalls = lambda: _qa_fixture_pool(n_answerable=2, words=50)
    with pytest.raises(ValueError, match="ZERO distractor"):
        _loader("qa", qa_source=two_smalls, context_length_tokens=64).load(max_examples=1)


# ---------------------------------------------------------------------------
# Per-task reporting contract (analysis splits by task, never aggregates)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", _TASKS)
def test_per_task_reporting_fields_present(task):
    ex = _loader(task, seed=4).load(max_examples=1)[0]
    md = ex.metadata

    assert md["dataset"] == "ruler"
    assert md["task"] == task
    assert md["native_metric"] == NATIVE_METRICS[task]
    assert md["native_metrics_only"] is True
    assert isinstance(md["gold_answers"], list) and md["gold_answers"]
    assert all(isinstance(g, str) and g for g in md["gold_answers"])
    assert ex.id == f"ruler_{task}_256_0000"


def test_native_metric_split_matches_ruler_reference():
    assert NATIVE_METRICS["niah_multiquery"] == "string_match_all"
    assert NATIVE_METRICS["variable_tracking"] == "string_match_all"
    assert NATIVE_METRICS["qa"] == "string_match_part"


# ---------------------------------------------------------------------------
# score_native — fail-closed scoring boundaries
# ---------------------------------------------------------------------------


def test_score_native_containment_is_case_insensitive():
    assert score_native("niah_single", ["1234567"], "... 1234567 ...") == 1.0
    assert score_native("qa", ["Paris"], "the answer is paris") == 1.0
    assert score_native("qa", ["Paris"], "the answer is London") == 0.0


def test_score_native_refuses_unknown_task():
    with pytest.raises(ValueError, match="Unknown RULER task"):
        score_native("common_words_extraction", ["x"], "x")


def test_score_native_refuses_empty_golds_and_absent_prediction():
    with pytest.raises(ValueError, match="empty gold_answers"):
        score_native("qa", [], "anything")
    with pytest.raises(ValueError, match="absence"):
        score_native("qa", ["x"], None)  # absence stays absence, never 0.0


def test_score_example_refuses_non_ruler_examples():
    alien = CAGExample(id="x", question="q", context=["c"], answer="a", metadata={})
    with pytest.raises(ValueError, match="scoring contract"):
        score_example(alien, "a")
