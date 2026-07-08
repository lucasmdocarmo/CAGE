"""Unit tests for baseline configuration.

`src/orchestration/baselines.py` imports only stdlib, so these run without vLLM/faiss/redis.
They pin the baseline taxonomy (ten families), the staleness config contract (version mode,
serving path wired, TTL fields present but defaulted off), and the compression/speculative arms.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from src.orchestration.baselines import BaselineType, get_baseline_config  # noqa: E402


ALL_NAMES = [
    "no_cache", "prefix_cache", "redis", "rag", "distributed",
    "hybrid", "speculative", "compressed_rag", "compressed_cag", "staleness",
]


def test_enum_has_ten_families():
    assert len(list(BaselineType)) == 10


def test_get_baseline_config_for_each_family():
    for name in ALL_NAMES:
        cfg = get_baseline_config(name)
        assert cfg.baseline_type.value == name


def test_unknown_baseline_raises():
    with pytest.raises(ValueError):
        get_baseline_config("does_not_exist")


def test_staleness_config_version_mode_and_wired():
    cfg = get_baseline_config("staleness")
    assert cfg.stale_evidence_mode == "version"   # the only wired mode; "ttl" is not implemented
    assert cfg.stale_fraction == 0.0
    assert cfg.cache_ttl_seconds is None
    assert cfg.metadata.get("serving_path_wired") is True


def test_to_dict_serializes_staleness_fields():
    d = get_baseline_config("staleness").to_dict()
    for key in ("stale_fraction", "cache_ttl_seconds", "stale_evidence_mode", "evidence_version_field"):
        assert key in d


def test_speculative_defaults():
    cfg = get_baseline_config("speculative")
    assert cfg.num_speculative_tokens == 5
    assert cfg.enable_prefix_caching is True
    assert "eagle" in cfg.metadata.get("supported_methods", [])


def test_compressed_rag_uses_llmlingua2():
    cfg = get_baseline_config("compressed_rag")
    assert cfg.compress_method == "llmlingua2"
    assert cfg.use_faiss is True


def test_compressed_cag_uses_fp8():
    cfg = get_baseline_config("compressed_cag")
    assert cfg.kv_cache_dtype == "fp8"
    assert cfg.enable_prefix_caching is True


def test_overrides_apply():
    cfg = get_baseline_config("staleness", stale_fraction=0.5)
    assert cfg.stale_fraction == 0.5
