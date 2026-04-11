"""
Tests for inference engine module.
"""

import pytest
from src.inference.engine import InferenceRequest, InferenceResponse, DummyEngine


def test_inference_request_creation():
    """Test creating InferenceRequest."""
    request = InferenceRequest(
        prompt="Test prompt",
        max_tokens=50,
        temperature=0.7,
        request_id="test_123",
    )
    
    assert request.prompt == "Test prompt"
    assert request.max_tokens == 50
    assert request.temperature == 0.7
    assert request.request_id == "test_123"


def test_dummy_engine_generate():
    """Test DummyEngine generation."""
    engine = DummyEngine(model_name="dummy-test")
    
    request = InferenceRequest(
        prompt="What is 2+2?",
        max_tokens=10,
        request_id="req_1",
    )
    
    response = engine.generate(request)
    
    assert isinstance(response, InferenceResponse)
    assert response.request_id == "req_1"
    assert response.model_name == "dummy-test"
    assert len(response.generated_text) > 0
    assert response.ttft_ms > 0
    assert response.total_time_ms > 0
    assert response.num_tokens > 0
    assert response.finish_reason == "length"
    assert response.error is None


def test_dummy_engine_batch_generate():
    """Test DummyEngine batch generation."""
    engine = DummyEngine()
    
    requests = [
        InferenceRequest(prompt=f"Prompt {i}", request_id=f"req_{i}")
        for i in range(3)
    ]
    
    responses = engine.batch_generate(requests)
    
    assert len(responses) == 3
    assert all(isinstance(r, InferenceResponse) for r in responses)
    assert [r.request_id for r in responses] == ["req_0", "req_1", "req_2"]


def test_dummy_engine_is_ready():
    """Test DummyEngine readiness check."""
    engine = DummyEngine()
    assert engine.is_ready() is True


def test_dummy_engine_shutdown():
    """Test DummyEngine shutdown."""
    engine = DummyEngine()
    engine.shutdown()  # Should not raise


# vLLM integration tests live in tests/test_vllm_integration.py
