"""
Tests for inference engine module.
"""

import pytest
from src.inference.engine import InferenceRequest, InferenceResponse
from src.inference.vllm_adapter import VLLMAdapter
from src.inference.ollama_adapter import OllamaAdapter


def test_inference_request_creation():
    """Test creating InferenceRequest."""
    request = InferenceRequest(
        prompt="Test prompt",
        max_tokens=50,
        temperature=0.0,
        request_id="test_123",
    )

    assert request.prompt == "Test prompt"
    assert request.max_tokens == 50
    assert request.temperature == 0.0
    assert request.request_id == "test_123"


# vLLM integration tests live in tests/test_vllm_integration.py


# --- Regression tests: batch_generate must stream (real TTFT), not silently ---
# --- fall back to a non-streamed measurement. See review findings on         ---
# --- run_experiment.py's context-hash batching (workload-mode batched).      ---


def test_vllm_batch_generate_streams_every_request(monkeypatch):
    """batch_generate must call generate(req, stream=True) for every request.

    Before the fix, batch_generate called self.generate(req) with no stream
    kwarg, which defaults to False and silently falls into vllm_adapter's
    non-streamed branch (ttft_ms == total_time_ms) -- a different TTFT
    measurement methodology than the streamed single-request path, mixed into
    the same baseline's rows with no column to distinguish them.
    """
    adapter = VLLMAdapter(model_name="test-model")
    calls = []

    def fake_generate(request, *, stream=False):
        calls.append(stream)
        return InferenceResponse(
            request_id=request.request_id,
            generated_text="ok",
            ttft_ms=1.0,
            total_time_ms=2.0,
            num_tokens=1,
            model_name="test-model",
            finish_reason="stop",
        )

    monkeypatch.setattr(adapter, "generate", fake_generate)

    requests_in = [
        InferenceRequest(prompt=f"prompt {i}", request_id=f"r{i}") for i in range(3)
    ]
    responses = adapter.batch_generate(requests_in)

    assert len(responses) == 3
    assert calls == [True, True, True], (
        "VLLMAdapter.batch_generate must pass stream=True to every generate() "
        "call so batched rows get a real first-token TTFT, not a full-response "
        "fallback."
    )


def test_ollama_batch_generate_streams_every_request(monkeypatch):
    """Same contract as above, for OllamaAdapter.

    Before the fix, batch_generate's non-streamed default hit the fabricated
    ttft_ms = total_time_ms * 0.2 branch (see the dedicated fabricated-TTFT
    regression test below) for every row in a work unit of size > 1.
    """
    adapter = OllamaAdapter(model_name="test-model")
    calls = []

    def fake_generate(request, *, stream=False):
        calls.append(stream)
        return InferenceResponse(
            request_id=request.request_id,
            generated_text="ok",
            ttft_ms=1.0,
            total_time_ms=2.0,
            num_tokens=1,
            model_name="test-model",
            finish_reason="stop",
        )

    monkeypatch.setattr(adapter, "generate", fake_generate)

    requests_in = [
        InferenceRequest(prompt=f"prompt {i}", request_id=f"r{i}") for i in range(3)
    ]
    responses = adapter.batch_generate(requests_in)

    assert len(responses) == 3
    assert calls == [True, True, True], (
        "OllamaAdapter.batch_generate must pass stream=True to every generate() "
        "call so batched rows get a real first-token TTFT, not the fabricated "
        "0.2x-of-total-time fallback."
    )


def test_ollama_non_streaming_ttft_is_full_response_time_not_fabricated(monkeypatch):
    """OllamaAdapter.generate(stream=False) must report ttft_ms == total_time_ms.

    Before the fix, the non-streaming branch set ttft_ms = total_time_ms * 0.2,
    an uncited fabricated fraction with no measurement basis -- inconsistent
    with vllm_adapter.py's honest "unobservable -> report full response time"
    convention for the identical situation.
    """
    import src.inference.ollama_adapter as ollama_adapter_mod

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        @property
        def headers(self):
            return {}

        def json(self):
            return {
                "response": "Paris",
                "done": True,
                "prompt_eval_count": 5,
                "eval_count": 1,
            }

    def fake_post(url, json=None, timeout=None):
        return _FakeResponse()

    monkeypatch.setattr(ollama_adapter_mod.requests, "post", fake_post)

    adapter = OllamaAdapter(model_name="test-model")
    request = InferenceRequest(prompt="The capital of France is", request_id="r0")
    response = adapter.generate(request, stream=False)

    assert response.error is None
    assert response.ttft_ms == pytest.approx(response.total_time_ms), (
        "Non-streaming Ollama TTFT must equal total_time_ms (the honest "
        "'unobservable, report full response time' convention), not an "
        "arbitrary fraction of it."
    )
