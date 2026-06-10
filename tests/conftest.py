"""Pytest fixtures shared across tests.

This repo includes optional integration tests for vLLM.
They are automatically skipped unless a vLLM server is reachable.

Environment variables:
- VLLM_TEST_API_BASE (default: http://localhost:8000)
- VLLM_TEST_MODEL (default: Qwen/Qwen3-4B)
"""

from __future__ import annotations

import os

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
