"""Unit tests for the RULER-style length instrument (charter D5 item 5).

src/data/ruler.py is pure stdlib (no `datasets`, no network, no GPU): the
generator is exercised directly. Covers happy path (determinism, needle
retrievability, length control), boundary (charter 32,768 cap, minimum
length, prefix-stable item derivation), and failure (typed ValueError on
cap/task violations — fail-closed, never silent clipping).
"""

import pytest

from src.data.loader import DatasetLoader, get_loader
from src.data.ruler import (
    MAX_CONTEXT_TOKENS,
    MIN_CONTEXT_TOKENS,
    NOISE_SENTENCE,
    OUTPUT_TOKENS_HINT,
    RulerLoader,
)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_reproduces_identical_items():
    a = RulerLoader(seed=42, context_length_tokens=512).load(max_examples=4)
    b = RulerLoader(seed=42, context_length_tokens=512).load(max_examples=4)

    assert [ex.id for ex in a] == [ex.id for ex in b]
    assert [ex.question for ex in a] == [ex.question for ex in b]
    assert [ex.answer for ex in a] == [ex.answer for ex in b]
    assert [ex.context for ex in a] == [ex.context for ex in b]


def test_different_seeds_draw_different_items():
    a = RulerLoader(seed=42, context_length_tokens=512).load(max_examples=4)
    b = RulerLoader(seed=43, context_length_tokens=512).load(max_examples=4)

    assert [ex.answer for ex in a] != [ex.answer for ex in b]


def test_items_are_prefix_stable_under_max_examples():
    """Item i derives from (seed, i): drawing more items never changes
    earlier ones."""
    few = RulerLoader(seed=7, context_length_tokens=512).load(max_examples=3)
    many = RulerLoader(seed=7, context_length_tokens=512).load(max_examples=6)

    assert [ex.id for ex in few] == [ex.id for ex in many[:3]]
    assert [ex.answer for ex in few] == [ex.answer for ex in many[:3]]


# ---------------------------------------------------------------------------
# Needle retrievability + length control
# ---------------------------------------------------------------------------


def test_single_needle_is_retrievable_from_context():
    for ex in RulerLoader(seed=1, context_length_tokens=512).load(max_examples=3):
        key = ex.metadata["needle_key"]
        needle = f"One of the special magic numbers for {key} is: {ex.answer}."
        assert len(ex.context) == 1
        assert needle in ex.context[0]
        assert key in ex.question
        assert ex.answer.isdigit() and len(ex.answer) == 7
        assert NOISE_SENTENCE in ex.context[0]


def test_context_length_hits_target_without_exceeding_it():
    target = 2048
    for ex in RulerLoader(seed=1, context_length_tokens=target).load(max_examples=3):
        actual = ex.metadata["actual_context_tokens"]
        assert actual == len(ex.context[0].split())  # whitespace proxy recorded
        assert actual <= target
        assert actual >= int(0.9 * target)
        assert ex.metadata["target_context_tokens"] == target


def test_multikey_task_adds_distractor_needles():
    loader = RulerLoader(seed=3, context_length_tokens=1024,
                         task="niah_multikey", num_distractors=3)
    ex = loader.load(max_examples=1)[0]

    distractors = ex.metadata["distractor_keys"]
    assert len(distractors) == 3
    assert ex.metadata["needle_key"] not in distractors
    for key in distractors:
        assert f"special magic numbers for {key} is:" in ex.context[0]
    # The query still targets exactly the one target key.
    assert ex.metadata["needle_key"] in ex.question
    assert all(k not in ex.question for k in distractors)


def test_injected_tokenizer_governs_length_and_is_recorded():
    counter = lambda text: max(1, len(text) // 4)  # chars/4 stand-in tokenizer
    ex = RulerLoader(seed=1, context_length_tokens=1024, tokenizer=counter,
                     tokenizer_name="chars-div-4").load(max_examples=1)[0]

    assert ex.metadata["tokenizer_name"] == "chars-div-4"
    assert ex.metadata["actual_context_tokens"] == counter(ex.context[0])
    assert ex.metadata["actual_context_tokens"] <= 1024


def test_instrument_metadata_contract():
    ex = RulerLoader(seed=1, context_length_tokens=256).load(max_examples=1)[0]
    md = ex.metadata

    assert md["dataset"] == "ruler"
    assert md["task"] == "niah_single"
    assert md["native_metrics_only"] is True  # never claim = real answer quality
    assert md["max_output_tokens_hint"] == OUTPUT_TOKENS_HINT
    assert 0.0 <= md["needle_depth"] <= 1.0
    assert md["tokenizer_name"] == "whitespace-proxy"
    assert ex.id == "ruler_niah_single_256_0000"


# ---------------------------------------------------------------------------
# Fail-closed boundaries (typed errors, no silent clipping)
# ---------------------------------------------------------------------------


def test_charter_cap_is_enforced():
    RulerLoader(context_length_tokens=MAX_CONTEXT_TOKENS)  # cap itself: OK
    with pytest.raises(ValueError, match="32768"):
        RulerLoader(context_length_tokens=MAX_CONTEXT_TOKENS + 1)


def test_shape_32k_input_cap_is_32512_not_32768():
    """SHAPE-32K pins input 32,512 + output 256 = 32,768 TOTAL (D5 §5.1).

    The cap applies to ``context_length_tokens`` — the INPUT side — so the
    natural "32k" grid point (32,768-token input) must be REFUSED: with the
    reserved 256-token output it would be 33,024 total, 256 over the pinned
    shape at the headline top-pressure point.
    """
    assert MAX_CONTEXT_TOKENS == 32768 - OUTPUT_TOKENS_HINT == 32512
    RulerLoader(context_length_tokens=32512)  # the pinned input size: OK
    with pytest.raises(ValueError, match="32768"):
        RulerLoader(context_length_tokens=32768)


def test_minimum_length_is_enforced():
    with pytest.raises(ValueError, match="minimum"):
        RulerLoader(context_length_tokens=MIN_CONTEXT_TOKENS - 1)


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="Unknown RULER task"):
        RulerLoader(task="vt")


def test_invalid_counts_are_rejected():
    with pytest.raises(ValueError, match="num_items"):
        RulerLoader(num_items=0)
    with pytest.raises(ValueError, match="num_distractors"):
        RulerLoader(num_distractors=0)


# ---------------------------------------------------------------------------
# Factory + env configuration (harness treats it like any dataset)
# ---------------------------------------------------------------------------


def test_registered_in_factory_with_seed_passthrough():
    loader = get_loader("ruler", split="validation", seed=9)
    assert isinstance(loader, RulerLoader)
    assert isinstance(loader, DatasetLoader)
    assert loader.seed == 9
    assert hasattr(loader, "load") and hasattr(loader, "sample")


def test_env_vars_configure_factory_instances(monkeypatch):
    monkeypatch.setenv("CAGE_RULER_CONTEXT_TOKENS", "512")
    monkeypatch.setenv("CAGE_RULER_TASK", "niah_multikey")
    monkeypatch.setenv("CAGE_RULER_NUM_ITEMS", "5")

    loader = get_loader("ruler", seed=1)
    assert loader.context_length_tokens == 512
    assert loader.task == "niah_multikey"
    assert loader.num_items == 5
    assert len(loader.load()) == 5


def test_explicit_args_beat_env_vars(monkeypatch):
    monkeypatch.setenv("CAGE_RULER_CONTEXT_TOKENS", "512")
    loader = RulerLoader(context_length_tokens=256)
    assert loader.context_length_tokens == 256
