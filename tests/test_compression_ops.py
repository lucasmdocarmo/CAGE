"""B3a/B3b compressor operating point + latency (2026-07-16 pre-run package).

llmlingua is MOCKED via sys.modules (the package is heavy and not needed to pin the
call contract). These tests pin:
- use_context_level_filter=False is passed (token-level only: the LLMLingua-2 default
  context filter STACKED on the token filter, driving a configured 2x to ~3.3x).
- target_token is computed from the ACTUAL token count of the concatenated context
  (the compressor's own tokenizer), anchoring the ratio in tokens.
- CAGE_COMPRESSION_RATE env knob (default 0.5) overrides the per-baseline ratio so
  multi-rate sweeps are a config knob.
- compression_latency_ms is measured (time.perf_counter) and emitted in the stats
  dict that feeds results.csv rows.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestration.compression import (  # noqa: E402
    CompressionStats,
    ContextCompressor,
    DEFAULT_COMPRESSION_RATE,
)


class _FakeLinguaTokenizer:
    """Whitespace tokenizer with the HF call convention."""

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": [0] * len(text.split())}


class _FakePromptCompressor:
    """Records compress_prompt kwargs; keeps the first rate-fraction of words."""

    last_init: dict | None = None
    last_kwargs: dict | None = None

    def __init__(self, model_name=None, use_llmlingua2=True, device_map="cpu"):
        _FakePromptCompressor.last_init = {
            "model_name": model_name, "use_llmlingua2": use_llmlingua2,
        }
        self.tokenizer = _FakeLinguaTokenizer()

    def compress_prompt(self, context, **kwargs):
        _FakePromptCompressor.last_kwargs = dict(kwargs)
        words = " ".join(context).split()
        keep = max(1, int(len(words) * kwargs.get("rate", 0.5)))
        return {"compressed_prompt": " ".join(words[:keep])}


@pytest.fixture()
def fake_llmlingua(monkeypatch):
    mod = types.ModuleType("llmlingua")
    mod.PromptCompressor = _FakePromptCompressor
    monkeypatch.setitem(sys.modules, "llmlingua", mod)
    for var in ("CAGE_DISABLE_COMPRESSION", "CAGE_ALLOW_NO_COMPRESSION", "CAGE_COMPRESSION_RATE"):
        monkeypatch.delenv(var, raising=False)
    _FakePromptCompressor.last_kwargs = None
    _FakePromptCompressor.last_init = None
    return mod


CONTEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank today.",
    "A second document with a dozen or so extra words to compress properly here.",
]


def test_context_level_filter_disabled(fake_llmlingua):
    c = ContextCompressor(method="llmlingua2")
    _, stats = c.compress(CONTEXTS, question="q", target_ratio=0.5)
    kw = _FakePromptCompressor.last_kwargs
    assert kw is not None
    assert kw["use_context_level_filter"] is False  # token-level only (B3a)
    assert stats.applied is True


def test_target_token_from_actual_token_count(fake_llmlingua):
    c = ContextCompressor(method="llmlingua2")
    _, stats = c.compress(CONTEXTS, target_ratio=0.5)
    kw = _FakePromptCompressor.last_kwargs
    # Actual tokens = fake tokenizer count of "\n\n".join(contexts) == whitespace words.
    actual = len("\n\n".join(CONTEXTS).split())
    assert kw["target_token"] == max(1, int(actual * 0.5))
    assert kw["rate"] == 0.5
    assert stats.target_ratio == 0.5


def test_env_rate_knob_overrides_argument(fake_llmlingua, monkeypatch):
    monkeypatch.setenv("CAGE_COMPRESSION_RATE", "0.25")
    c = ContextCompressor(method="llmlingua2")
    _, stats = c.compress(CONTEXTS, target_ratio=0.5)  # explicit arg loses to the sweep knob
    kw = _FakePromptCompressor.last_kwargs
    assert kw["rate"] == 0.25
    actual = len("\n\n".join(CONTEXTS).split())
    assert kw["target_token"] == max(1, int(actual * 0.25))
    assert stats.target_ratio == 0.25


def test_default_rate_is_half(fake_llmlingua):
    assert DEFAULT_COMPRESSION_RATE == 0.5
    c = ContextCompressor(method="llmlingua2")
    _, stats = c.compress(CONTEXTS)  # no arg, no env -> 0.5
    assert _FakePromptCompressor.last_kwargs["rate"] == 0.5
    assert stats.target_ratio == 0.5


def test_invalid_env_rate_falls_back(fake_llmlingua, monkeypatch):
    monkeypatch.setenv("CAGE_COMPRESSION_RATE", "not-a-float")
    c = ContextCompressor(method="llmlingua2")
    _, stats = c.compress(CONTEXTS, target_ratio=0.4)
    assert stats.target_ratio == 0.4


def test_latency_measured_and_in_stats_dict(fake_llmlingua):
    c = ContextCompressor(method="llmlingua2")
    _, stats = c.compress(CONTEXTS, target_ratio=0.5)
    assert stats.compression_latency_ms is not None
    assert stats.compression_latency_ms >= 0.0
    d = stats.to_dict()
    assert "compression_latency_ms" in d  # reaches results.csv via compression_stats
    assert d["compression_latency_ms"] == stats.compression_latency_ms


def test_latency_none_when_not_applied(monkeypatch):
    monkeypatch.setenv("CAGE_DISABLE_COMPRESSION", "1")
    monkeypatch.delenv("CAGE_COMPRESSION_RATE", raising=False)
    c = ContextCompressor(method="llmlingua2")
    docs, stats = c.compress(CONTEXTS, target_ratio=0.5)
    assert stats.applied is False
    assert stats.compression_latency_ms is None
    assert docs == CONTEXTS


def test_stats_dict_keeps_existing_contract(fake_llmlingua):
    _, stats = ContextCompressor(method="llmlingua2").compress(CONTEXTS, target_ratio=0.5)
    d = stats.to_dict()
    for key in (
        "compress_method", "compress_target_ratio", "original_tokens",
        "compressed_tokens", "compression_ratio", "compression_applied",
        "compression_note", "compression_latency_ms",
    ):
        assert key in d


def test_question_still_forwarded_for_v1_compat(fake_llmlingua):
    # The question kwarg is forwarded (question-aware v1 path) but the LLMLingua-2
    # path ignores it -- see the module docstring's honest-label note (Pan et al. 2024).
    c = ContextCompressor(method="llmlingua2")
    c.compress(CONTEXTS, question="what does the fox do?", target_ratio=0.5)
    assert _FakePromptCompressor.last_kwargs["question"] == "what does the fox do?"


def test_compressed_ratio_roughly_tracks_target(fake_llmlingua):
    _, stats = ContextCompressor(method="llmlingua2").compress(CONTEXTS, target_ratio=0.5)
    # Fake keeps exactly rate-fraction of words, so achieved == target here; the point
    # is that the stats math reports it from token counts, not a stacked-filter output.
    assert stats.compression_ratio == pytest.approx(0.5, abs=0.1)
