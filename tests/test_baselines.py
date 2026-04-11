"""Tests for baseline configuration.

Focuses on config defaults and requirement checks (unit level).
"""

import pytest

from src.orchestration.baselines import get_baseline_config


def test_baseline_config_contains_ir_fields():
    cfg = get_baseline_config("rag")
    d = cfg.to_dict()

    assert d["baseline_type"] == "rag"
    assert d["use_faiss"] is True
    assert "embedding_model" in d
    assert "top_k_retrieval" in d
    assert "ir_index_dir" in d
    assert "ir_rebuild" in d


def test_redis_baseline_has_retrieval_enabled():
    cfg = get_baseline_config("redis")
    assert cfg.use_faiss is True
    assert cfg.top_k_retrieval >= 1


def test_unknown_baseline_raises():
    with pytest.raises(ValueError):
        get_baseline_config("nope")
