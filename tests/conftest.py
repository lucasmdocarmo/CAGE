"""Pytest fixtures shared across tests.

This repo includes optional integration tests for vLLM.
They are automatically skipped unless a vLLM server is reachable.

Environment variables:
- VLLM_TEST_API_BASE (default: http://localhost:8000)
- VLLM_TEST_MODEL (default: Qwen/Qwen3-4B)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest


@pytest.fixture(scope="session")
def vllm_test_api_base() -> str:
    return os.getenv("VLLM_TEST_API_BASE", "http://localhost:8000").rstrip("/")


@pytest.fixture(scope="session")
def vllm_test_model() -> str:
    return os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen2.5-Coder-0.5B-Instruct")


@pytest.fixture(scope="session")
def vllm_available(vllm_test_api_base: str) -> bool:
    """Return True if a vLLM OpenAI-compatible server is reachable."""
    try:
        import requests

        r = requests.get(f"{vllm_test_api_base}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Hermetic freeze artifact (backlog A5, ADR-0099, ADR-0112)
# ---------------------------------------------------------------------------
# The campaign driver reads its dense-retriever pin from the registration
# artifact named by $CAGE_FREEZE_RESOLUTIONS (default: the gitignored
# MyDocs/registration/freeze_resolutions.json). Every plan a test builds must be
# hermetic: the ADR-0112 detached worktree at the registered SHA has no MyDocs,
# so a test that silently depended on that file would be red there. The values
# mirror the registration artifact's INSTRUMENT_REVISIONS.dense_retriever slot
# (tests/test_run_campaign.py pins the SAME literals, so a drift between the two
# copies fails there).

FREEZE_FILE_ENV_VAR: str = "CAGE_FREEZE_RESOLUTIONS"
FREEZE_EMBEDDING_MODEL: str = "intfloat/e5-large-v2"
FREEZE_EMBEDDING_REVISION: str = "f169b11e22de13617baa190a028a32f3493550b6"


def hermetic_freeze_doc() -> Dict[str, Any]:
    return {
        "INSTRUMENT_REVISIONS": {
            "dense_retriever": {
                "model": FREEZE_EMBEDDING_MODEL,
                "revision": FREEZE_EMBEDDING_REVISION,
                "resolved": "test mirror of the registration artifact",
            },
            # The QUALITY module's similarity embedder: a DIFFERENT
            # instrument's pin; the driver must never consume it.
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            },
        }
    }


@pytest.fixture(scope="session", autouse=True)
def _hermetic_freeze_env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Point $CAGE_FREEZE_RESOLUTIONS at a tmp mirror for the WHOLE session.

    Session-scoped so the 3,000+ tests pay one file write; a test that needs
    a different artifact overrides the variable with its own monkeypatch
    (restored afterwards), and a test that needs it ABSENT deletes it the
    same way.
    """
    path = tmp_path_factory.mktemp("freeze") / "freeze_resolutions.json"
    path.write_text(json.dumps(hermetic_freeze_doc()), encoding="utf-8")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(FREEZE_FILE_ENV_VAR, str(path))
        yield path


@pytest.fixture()
def freeze_file(_hermetic_freeze_env: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The hermetic freeze artifact's path (re-pinned per test, so a test that
    previously replaced the variable cannot leak into the next one)."""
    monkeypatch.setenv(FREEZE_FILE_ENV_VAR, str(_hermetic_freeze_env))
    return _hermetic_freeze_env
