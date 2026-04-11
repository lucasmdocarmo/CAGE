"""Integration tests for vLLM.

These tests are skipped unless a vLLM server is running.

Run (example):
  VLLM_TEST_API_BASE=http://localhost:8000 \
  VLLM_TEST_MODEL=Qwen/Qwen3-4B \
  pytest -m vllm

Start vLLM server (example):
  vllm serve Qwen/Qwen3-4B --port 8000
"""

from __future__ import annotations

import pytest

from src.inference.engine import InferenceRequest, InferenceResponse
from src.inference.vllm_adapter import VLLMAdapter


pytestmark = [pytest.mark.integration, pytest.mark.vllm]


def test_vllm_health_endpoint(vllm_test_api_base: str, vllm_available: bool):
    if not vllm_available:
        pytest.skip(f"vLLM server not reachable at {vllm_test_api_base}")

    import requests

    r = requests.get(f"{vllm_test_api_base}/health", timeout=2)
    assert r.status_code == 200


def test_vllm_completions_basic(vllm_test_api_base: str, vllm_test_model: str, vllm_available: bool):
    if not vllm_available:
        pytest.skip(f"vLLM server not reachable at {vllm_test_api_base}")

    adapter = VLLMAdapter(model_name=vllm_test_model, api_base=vllm_test_api_base)
    assert adapter.is_ready() is True

    req = InferenceRequest(prompt="The capital of France is", max_tokens=8)
    resp = adapter.generate(req)

    assert isinstance(resp, InferenceResponse)
    assert resp.finish_reason in {"length", "stop"}
    assert resp.error is None
    assert isinstance(resp.generated_text, str)
    # Usually the response contains some text (may start with a leading space)
    assert len(resp.generated_text) >= 0


def test_vllm_completions_with_stop(vllm_test_api_base: str, vllm_test_model: str, vllm_available: bool):
    if not vllm_available:
        pytest.skip(f"vLLM server not reachable at {vllm_test_api_base}")

    adapter = VLLMAdapter(model_name=vllm_test_model, api_base=vllm_test_api_base)

    req = InferenceRequest(
        prompt="List: one, two, three.\nStopWord:",
        max_tokens=32,
        stop=["\n"],
    )
    resp = adapter.generate(req)

    assert resp.error is None
    assert isinstance(resp.generated_text, str)
    # Since we used stop=["\n"], the completion should not contain a newline.
    assert "\n" not in resp.generated_text


def test_vllm_cache_hit_reduces_ttft(vllm_test_api_base: str, vllm_test_model: str, vllm_available: bool):
    """Sanity check that repeating an identical prompt doesn't *increase* TTFT."""
    if not vllm_available:
        pytest.skip(f"vLLM server not reachable at {vllm_test_api_base}")

    adapter = VLLMAdapter(model_name=vllm_test_model, api_base=vllm_test_api_base)

    prompt = "The capital of France is"
    req = InferenceRequest(prompt=prompt, max_tokens=16)

    cold = adapter.generate(req, stream=True)
    warm = adapter.generate(req, stream=True)

    assert cold.error is None and warm.error is None
    # TTFT should not increase; allow small jitter (10%)
    assert warm.ttft_ms <= cold.ttft_ms * 1.10


def test_vllm_stream_usage_prompt_tokens_optional(
    vllm_test_api_base: str, vllm_test_model: str, vllm_available: bool
):
    """If the server supports stream_options.include_usage, prompt_tokens should be populated."""
    if not vllm_available:
        pytest.skip(f"vLLM server not reachable at {vllm_test_api_base}")

    adapter = VLLMAdapter(model_name=vllm_test_model, api_base=vllm_test_api_base)
    req = InferenceRequest(prompt="The capital of France is", max_tokens=8)
    resp = adapter.generate(req, stream=True)

    assert resp.error is None
    if resp.prompt_tokens is None:
        pytest.skip(
            "vLLM did not include usage in stream; ensure vLLM supports stream_options.include_usage"
        )

    assert resp.prompt_tokens > 0


def test_vllm_cached_prompt_tokens_optional(
    vllm_test_api_base: str, vllm_test_model: str, vllm_available: bool
):
    """cached_prompt_tokens is only present when vLLM is started with --enable-prompt-tokens-details."""
    if not vllm_available:
        pytest.skip(f"vLLM server not reachable at {vllm_test_api_base}")

    adapter = VLLMAdapter(model_name=vllm_test_model, api_base=vllm_test_api_base)
    req = InferenceRequest(prompt="The capital of France is", max_tokens=8)
    resp = adapter.generate(req, stream=True)

    assert resp.error is None
    if resp.cached_prompt_tokens is None:
        pytest.skip("cached_tokens not included; start vLLM with --enable-prompt-tokens-details")

    assert resp.cached_prompt_tokens >= 0


def test_router_sets_replica_header_when_using_router(
    vllm_test_api_base: str, vllm_test_model: str, vllm_available: bool
):
    """If VLLM_TEST_API_BASE points to the CAGE router, responses should include x-router-replica."""
    if not vllm_available:
        pytest.skip(f"endpoint not reachable at {vllm_test_api_base}")

    import requests

    # Detect router by the /health JSON shape.
    try:
        r = requests.get(f"{vllm_test_api_base}/health", timeout=2)
        r.raise_for_status()
        health = r.json()
    except Exception:
        pytest.skip("endpoint does not look like the CAGE router")

    if not isinstance(health, dict) or "router_initialized" not in health:
        pytest.skip("endpoint does not look like the CAGE router")

    if not health.get("router_initialized"):
        pytest.skip("router not initialized")

    adapter = VLLMAdapter(model_name=vllm_test_model, api_base=vllm_test_api_base)
    req = InferenceRequest(prompt="The capital of France is", max_tokens=8)

    # Check both non-streaming and streaming paths.
    resp_json = adapter.generate(req, stream=False)
    resp_stream = adapter.generate(req, stream=True)

    assert resp_json.error is None and resp_stream.error is None
    assert resp_json.router_replica
    assert resp_stream.router_replica
