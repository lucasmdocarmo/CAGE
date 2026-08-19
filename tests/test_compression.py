"""Unit tests for the compression axis (stats + strictness).

`CompressionStats` is a pure dataclass and `ContextCompressor.__init__` is lazy (it does not
import llmlingua until the compressor is used), so these run without the llmlingua package.
They pin the ratio math and the strict-by-default contract that prevents `compressed_rag` from
silently measuring plain RAG.

Env hygiene (K-series note, task #140): every CAGE_* env manipulation goes
through pytest's ``monkeypatch`` (setenv/delenv) — the old direct
``os.environ`` writes leaked state between tests on failure, and
``test_strict_by_default`` destructively popped a developer shell's variables
from the whole test process instead of shielding itself from them.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestration.compression import CompressionStats, ContextCompressor  # noqa: E402


def test_compression_ratio_math():
    s = CompressionStats("llmlingua2", 0.5, original_tokens=100, compressed_tokens=50, applied=True)
    assert s.compression_ratio == 0.5


def test_compression_ratio_no_original_is_one():
    s = CompressionStats("llmlingua2", 0.5, original_tokens=0, compressed_tokens=0, applied=False)
    assert s.compression_ratio == 1.0


def test_compression_stats_to_dict_keys():
    d = CompressionStats("llmlingua2", 0.5, 100, 50, applied=True).to_dict()
    for key in ("compress_method", "compression_ratio", "compression_applied", "compress_target_ratio"):
        assert key in d


def test_disabled_compressor_is_passthrough(monkeypatch: pytest.MonkeyPatch):
    # CAGE_DISABLE_COMPRESSION makes the compressor a transparent pass-through (applied=False),
    # and strictness is moot (the no-op is intended), so compress() must NOT raise.
    monkeypatch.delenv("CAGE_ALLOW_NO_COMPRESSION", raising=False)
    monkeypatch.setenv("CAGE_DISABLE_COMPRESSION", "1")
    c = ContextCompressor(method="llmlingua2")
    assert c._strict is False
    docs, stats = c.compress(["some context text"], question="q", target_ratio=0.5)
    assert stats.applied is False
    assert docs == ["some context text"]


def test_strict_by_default(monkeypatch: pytest.MonkeyPatch):
    # With neither disable nor allow-noop set, the compressor is strict: a missing/failed
    # backend would raise rather than silently returning ratio 1.0.
    for var in ("CAGE_DISABLE_COMPRESSION", "CAGE_ALLOW_NO_COMPRESSION"):
        monkeypatch.delenv(var, raising=False)
    c = ContextCompressor(method="llmlingua2")
    assert c._strict is True


def test_allow_noop_relaxes_strict(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CAGE_DISABLE_COMPRESSION", raising=False)
    monkeypatch.setenv("CAGE_ALLOW_NO_COMPRESSION", "1")
    c = ContextCompressor(method="llmlingua2")
    assert c._strict is False
