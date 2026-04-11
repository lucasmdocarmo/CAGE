"""Integration tests for the CAGE router.

These tests validate that the router:
- responds to /health, /stats and /metrics
- proxies non-streaming and streaming /v1/completions
- sets x-router-replica on responses

They are skipped unless the router is reachable.

Run (example):
  ROUTER_TEST_API_BASE=http://localhost:9000 \
  pytest -m integration -k router
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

import pytest


pytestmark = [pytest.mark.integration]


def _is_router(base: str) -> bool:
    """Return True if the endpoint looks like the CAGE router."""
    try:
        import requests

        r = requests.get(f"{base}/health", timeout=2)
        if r.status_code != 200:
            return False
        body = r.json()
        return isinstance(body, dict) and "router_initialized" in body
    except Exception:
        return False


@pytest.fixture(scope="session")
def router_test_api_base() -> str:
    """Base URL of a running router (defaults to localhost)."""
    return os.getenv("ROUTER_TEST_API_BASE", "http://localhost:9000").rstrip("/")


@pytest.fixture(scope="session")
def router_available(router_test_api_base: str) -> bool:
    """Return True if a router is reachable at router_test_api_base."""
    return _is_router(router_test_api_base)


def test_router_health(router_test_api_base: str, router_available: bool):
    """/health should report router_initialized=true when replicas are configured."""
    if not router_available:
        pytest.skip(f"router not reachable at {router_test_api_base}")

    import requests

    r = requests.get(f"{router_test_api_base}/health", timeout=2)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, dict)
    assert body.get("status") == "healthy"
    assert body.get("router_initialized") is True


def test_router_stats(router_test_api_base: str, router_available: bool):
    """/stats should return basic routing stats."""
    if not router_available:
        pytest.skip(f"router not reachable at {router_test_api_base}")

    import requests

    r = requests.get(f"{router_test_api_base}/stats", timeout=2)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, dict)
    assert "total_requests" in body
    assert "replica_distribution" in body
    assert "num_replicas" in body


def test_router_metrics_endpoint(router_test_api_base: str, router_available: bool):
    """/metrics should expose Prometheus text and include router series."""
    if not router_available:
        pytest.skip(f"router not reachable at {router_test_api_base}")

    import requests

    r = requests.get(f"{router_test_api_base}/metrics", timeout=3)
    assert r.status_code == 200
    text = r.text

    # Minimal sanity checks.
    assert "cage_router_requests_total" in text
    assert "cage_router_request_latency_seconds" in text


def test_router_completions_non_streaming(router_test_api_base: str, router_available: bool):
    """Router should proxy a non-streaming /v1/completions request and set x-router-replica."""
    if not router_available:
        pytest.skip(f"router not reachable at {router_test_api_base}")

    import requests

    payload: Dict[str, Any] = {
        "model": "Qwen/Qwen3-4B",
        "prompt": "The capital of France is",
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": False,
    }

    r = requests.post(f"{router_test_api_base}/v1/completions", json=payload, timeout=60)
    if r.status_code != 200:
        pytest.skip(f"router upstream not ready: {r.status_code} {r.text[:200]}")

    assert r.headers.get("x-router-replica")
    body = r.json()
    assert isinstance(body, dict)
    assert "choices" in body


def test_router_streaming_passthrough_sse(router_test_api_base: str, router_available: bool):
    """Router should proxy SSE and terminate with [DONE]."""
    if not router_available:
        pytest.skip(f"router not reachable at {router_test_api_base}")

    import requests

    payload: Dict[str, Any] = {
        "model": "Qwen/Qwen3-4B",
        "prompt": "The capital of France is",
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    got_done = False
    saw_choice_delta = False
    saw_usage: Optional[dict] = None

    with requests.post(
        f"{router_test_api_base}/v1/completions",
        json=payload,
        stream=True,
        timeout=(5, 60),
    ) as r:
        if r.status_code != 200:
            pytest.skip(f"router upstream not ready: {r.status_code} {r.text[:200]}")

        assert r.headers.get("x-router-replica")

        content_type = r.headers.get("Content-Type", "")
        assert "text/event-stream" in content_type or "event-stream" in content_type

        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if not raw_line.startswith("data:"):
                continue

            data = raw_line[5:].strip()
            if data == "[DONE]":
                got_done = True
                break

            try:
                obj = json.loads(data)
            except Exception:
                continue

            if isinstance(obj, dict) and isinstance(obj.get("usage"), dict):
                saw_usage = obj["usage"]

            choices = obj.get("choices") if isinstance(obj, dict) else None
            if isinstance(choices, list) and choices:
                # vLLM completion streaming chunks use the 'text' field.
                if isinstance(choices[0], dict) and choices[0].get("text") is not None:
                    saw_choice_delta = True

    assert got_done is True
    assert saw_choice_delta is True

    # Usage is optional depending on upstream support.
    if saw_usage is not None:
        assert "prompt_tokens" in saw_usage
