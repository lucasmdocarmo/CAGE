"""
Inference engine abstraction for CAGE framework.

Provides a unified interface for different LLM serving backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import time


@dataclass
class InferenceRequest:
    """Single inference request."""
    
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.95
    stop: Optional[List[str]] = None
    truncate_prompt_tokens: Optional[int] = None
    
    # Metadata for tracking
    request_id: Optional[str] = None
    prefix_hash: Optional[str] = None  # For prefix-aware routing


@dataclass
class InferenceResponse:
    """Single inference response with metrics."""

    request_id: Optional[str]
    generated_text: str

    # Performance metrics
    ttft_ms: float  # Time to first token (milliseconds)
    total_time_ms: float  # End-to-end latency
    num_tokens: int  # Number of generated tokens

    # Metadata
    model_name: str
    finish_reason: str  # "length", "stop", "error"
    error: Optional[str] = None

    # Optional metadata (e.g., which router replica served the request)
    router_replica: Optional[str] = None

    # Optional telemetry from the backend (when available)
    prompt_tokens: Optional[int] = None
    cached_prompt_tokens: Optional[int] = None
    kv_transfer_params: Optional[Dict[str, Any]] = None


class InferenceEngine(ABC):
    """Abstract base class for inference engines.

    All backends must accept a `stream` keyword argument. Backends that do not
    support streaming should ignore it.
    """

    def __init__(self, model_name: str, **kwargs):
        """Initialize an inference engine wrapper."""
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(self, request: InferenceRequest, *, stream: bool = False) -> InferenceResponse:
        """Generate a single response.

        Args:
            request: Prompt + decoding parameters.
            stream: If True, the backend may use streaming to measure TTFT.
                Backends that do not support streaming should ignore this flag.
        """
        raise NotImplementedError
    
    @abstractmethod
    def batch_generate(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Generate responses for batch of requests."""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if engine is ready to serve requests."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Cleanup and shutdown engine."""
        pass


class DummyEngine(InferenceEngine):
    """Dummy engine for testing (returns placeholder responses)."""

    def __init__(self, model_name: str = "dummy-model", **kwargs):
        """Create a deterministic fake engine for unit tests."""
        super().__init__(model_name, **kwargs)
        self.call_count = 0

    def generate(self, request: InferenceRequest, *, stream: bool = False) -> InferenceResponse:
        """Generate a dummy response.

        Note:
            `stream` is accepted for API compatibility but ignored.
        """
        self.call_count += 1
        
        # Simulate processing time
        start_time = time.time()
        time.sleep(0.01)  # 10ms simulated TTFT
        ttft = (time.time() - start_time) * 1000
        
        # Generate dummy text
        dummy_text = f"This is a dummy response #{self.call_count} for prompt: {request.prompt[:50]}..."
        
        time.sleep(0.04)  # Additional 40ms for generation
        total_time = (time.time() - start_time) * 1000
        
        return InferenceResponse(
            request_id=request.request_id,
            generated_text=dummy_text,
            ttft_ms=ttft,
            total_time_ms=total_time,
            num_tokens=len(dummy_text.split()),
            model_name=self.model_name,
            finish_reason="length",
        )
    
    def batch_generate(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Generate responses for batch."""
        return [self.generate(req) for req in requests]
    
    def is_ready(self) -> bool:
        """Always ready."""
        return True
    
    def shutdown(self) -> None:
        """No cleanup needed."""
        pass
