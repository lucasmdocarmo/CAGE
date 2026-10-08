"""
Inference engine abstraction for CAGE framework.

Provides a unified interface for different LLM serving backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

#: ``num_tokens`` provenance labels (ADR-0118, Batch 2 W5), stamped by the
#: adapters as the plain attribute ``num_tokens_source`` and persisted per row
#: by run_experiment.adapter_honesty_columns. "usage": the engine's
#: usage.completion_tokens; "whitespace": the word count of the generated text
#: because no usage.completion_tokens arrived (words, not tokens); "token_ids": the
#: in-process engine's own generated ids. None on an error row and on an
#: adapter that does not label.
NUM_TOKENS_SOURCE_USAGE: str = "usage"
NUM_TOKENS_SOURCE_WHITESPACE: str = "whitespace"
NUM_TOKENS_SOURCE_TOKEN_IDS: str = "token_ids"


@dataclass
class InferenceRequest:
    """Single inference request."""
    
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.0  # greedy decoding is the campaign's determinism contract
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

    def capabilities(self) -> Dict[str, Any]:
        """Data-driven capability declaration for the charter-D2
        telemetry-parity preflight gate (ADR-0007).

        Conservative default: an engine declares NOTHING until its adapter
        overrides this -- absence of a declared capability must read as
        "unavailable", never as an implicit yes. Value convention used by the
        adapters: ``True`` (verified in this codebase), ``False``/``None``
        (absent or deliberately not implemented), ``"verify-live"``
        (documented upstream, unverified here -- charter D2.1 [VERIFY-LIVE]).
        """
        return {
            "engine": getattr(self, "engine_id", type(self).__name__.lower()),
            "serving_grade": False,
            "in_process": False,
            "streamed_ttft": False,
            "cached_token_telemetry": False,
            "kv_usage_gauge": False,
            "flush_endpoint": None,
            "kv_transfer_params": False,
        }
