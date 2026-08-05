"""vLLM adapters for the CAGE framework.

This module provides:
- VLLMAdapter: OpenAI-compatible HTTP client with optional streaming TTFT.
- VLLMOfflineAdapter: in-process vLLM execution for local debugging.

We also extract optional vLLM telemetry when available:
- usage.prompt_tokens
- usage.prompt_tokens_details.cached_tokens (requires vLLM flag --enable-prompt-tokens-details)

Refactor note (ADR-0007, 2026-08-04): the engine-agnostic SSE/TTFT/request
machinery now lives in OpenAIChatAdapter (src/inference/openai_chat_adapter.py);
VLLMAdapter keeps exactly the vLLM-specific semantics as overrides -- the
0.11.0 absent-cached-block-means-zero usage quirk, the pinned
chat_template_kwargs/logprobs chat extras, truncate_prompt_tokens support, and
kv_transfer_params (NIXL) telemetry. Externally observable behavior is
unchanged from the pre-refactor adapter.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import requests  # noqa: F401  -- kept importable as a module attribute for test monkeypatching compatibility

from .engine import InferenceEngine, InferenceRequest, InferenceResponse
from .openai_chat_adapter import OpenAIChatAdapter


class VLLMAdapter(OpenAIChatAdapter):
    """HTTP client adapter for a vLLM OpenAI-compatible server."""

    engine_id: str = "vllm"
    # vLLM P/D connectors (NIXL) attach kv_transfer_params to bodies/headers.
    _kv_transfer_telemetry: bool = True
    # Documented dev-only endpoint; requires the server to run with
    # VLLM_SERVER_DEV_MODE=1 (see scripts/2_serving/manage_vllm_server.sh).
    _flush_endpoint: Optional[str] = "/reset_prefix_cache"

    def __init__(
        self,
        model_name: str,
        api_base: str = "http://localhost:8000",
        timeout: int = 300,
        include_usage_in_stream: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create an adapter targeting a vLLM server.

        Args:
            model_name: Served model name.
            api_base: Base URL of a vLLM server or the CAGE router.
            timeout: Requests timeout (seconds).
            include_usage_in_stream: If True, request a final streaming usage
                chunk so we can extract prompt/cached token telemetry.
            **kwargs: Forwarded to OpenAIChatAdapter (e.g. max_retries,
                retry_backoff_s) then InferenceEngine.
        """
        super().__init__(
            model_name,
            api_base=api_base,
            timeout=timeout,
            include_usage_in_stream=include_usage_in_stream,
            **kwargs,
        )

    def _apply_engine_chat_extras(self, payload: Dict[str, Any]) -> None:
        """vLLM-specific /v1/chat/completions fields.

        - chat_template_kwargs {"enable_thinking": false}: disables Qwen3
          thinking mode. Verified against vLLM 0.11.0 docs (openai_compatible_
          server): ChatCompletionRequest exposes ``chat_template_kwargs:
          Optional[dict]`` ("Additional keyword args to pass to the template
          renderer. Will be accessible by the chat template."); Qwen3's chat
          template reads ``enable_thinking``. Other templates simply ignore
          the unused variable (Jinja semantics), so the field is always sent.
        - logprobs=true, top_logprobs=0: per-generated-token logprobs (OpenAI
          chat schema) -> mean/sum persisted for abstention risk-coverage
          curves. top_logprobs=0 returns only the chosen token's logprob.
        """
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload["logprobs"] = True
        payload["top_logprobs"] = 0

    def _apply_truncate_prompt_tokens(
        self, payload: Dict[str, Any], request: InferenceRequest
    ) -> None:
        """vLLM supports the truncate_prompt_tokens extension parameter."""
        payload["truncate_prompt_tokens"] = request.truncate_prompt_tokens

    def _extract_usage(
        self, usage: Dict[str, Any]
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Extract (prompt_tokens, cached_prompt_tokens, completion_tokens) from usage."""
        prompt_tokens_out, cached_out, completion_out = super()._extract_usage(usage)

        # Audit 2026-07-16 M6 (cached-zero-recorded-as-missing): vLLM 0.11.0 OMITS
        # usage.prompt_tokens_details whenever num_cached_tokens is falsy (cold request),
        # even with --enable-prompt-tokens-details. Recording those rows as None made every
        # cached_prompt_tokens/cached_prompt_ratio statistic silently conditional-on-hit
        # and left no_cache-family arms with no cache telemetry at all. When the usage
        # object itself is present (prompt_tokens parsed), an absent details block means
        # cached == 0, not missing. None is kept only when usage is missing entirely.
        # This is a VERIFIED vLLM-specific semantic -- deliberately NOT inherited by
        # other engines (ADR-0007), whose absent fields stay None-with-provenance.
        if cached_out is None and prompt_tokens_out is not None:
            cached_out = 0

        return (prompt_tokens_out, cached_out, completion_out)

    def capabilities(self) -> Dict[str, Any]:
        """Capability declaration driving the charter-D2 telemetry-parity gate.

        Values: True (verified in this codebase at the pinned version), False/
        None (absent or deliberately not implemented), "verify-live" (documented
        upstream, unverified here -- charter D2.1 [VERIFY-LIVE]).
        """
        return {
            "engine": self.engine_id,
            "serving_grade": True,
            "in_process": False,
            "streamed_ttft": True,
            "cached_token_telemetry": True,
            "cached_token_server_flag": "--enable-prompt-tokens-details",
            "cached_token_absent_means_zero": True,  # vLLM 0.11.0 quirk (audit M6)
            "kv_usage_gauge": False,  # adapter never scrapes /metrics (ADR-0007 item 5)
            "flush_endpoint": self._flush_endpoint,  # dev-mode gated (VLLM_SERVER_DEV_MODE=1)
            "kv_transfer_params": True,
            "chat_template_thinking_pin": True,  # verified against vLLM 0.11.0 schema
            "logprobs": True,
            "truncate_prompt_tokens": True,
        }


class VLLMOfflineAdapter(InferenceEngine):
    """Adapter for vLLM offline inference (in-process).

    This is useful for local debugging. It does not provide true TTFT or
    prompt-cache telemetry.
    """

    def __init__(self, model_name: str, **kwargs):
        """Initialize an in-process vLLM LLM() engine."""
        super().__init__(model_name, **kwargs)

        # Import vLLM here to make it optional.
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm")

        # Initialize vLLM engine
        self.llm = LLM(model=model_name, **kwargs)
        self.SamplingParams = SamplingParams

    def generate(self, request: InferenceRequest, *, stream: bool = False) -> InferenceResponse:
        """Generate using offline vLLM engine.

        Note:
            `stream` is accepted for interface compatibility but ignored.
        """
        start_time = time.time()

        sampling_params = self.SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
        )

        try:
            outputs = self.llm.generate([request.prompt], sampling_params)
            total_time = (time.time() - start_time) * 1000

            output = outputs[0]
            generated_text = output.outputs[0].text
            num_tokens = len(output.outputs[0].token_ids)
            finish_reason = output.outputs[0].finish_reason

            # vLLM offline non-streaming: TTFT unobservable -> report full response time.
            ttft_ms = total_time

            return InferenceResponse(
                request_id=request.request_id,
                generated_text=generated_text,
                ttft_ms=ttft_ms,
                total_time_ms=total_time,
                num_tokens=num_tokens,
                model_name=self.model_name,
                finish_reason=finish_reason,
            )

        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            return InferenceResponse(
                request_id=request.request_id,
                generated_text="",
                ttft_ms=0,
                total_time_ms=total_time,
                num_tokens=0,
                model_name=self.model_name,
                finish_reason="error",
                error=str(e),
            )

    def batch_generate(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Batch generation using vLLM offline engine."""
        start_time = time.time()

        # Use first request's params as default (or make configurable)
        first_req = requests[0] if requests else InferenceRequest(prompt="")
        sampling_params = self.SamplingParams(
            temperature=first_req.temperature,
            top_p=first_req.top_p,
            max_tokens=first_req.max_tokens,
            stop=first_req.stop,
        )

        prompts = [req.prompt for req in requests]

        try:
            outputs = self.llm.generate(prompts, sampling_params)

            responses = []
            for i, (output, request) in enumerate(zip(outputs, requests)):
                elapsed = (time.time() - start_time) * 1000
                generated_text = output.outputs[0].text
                num_tokens = len(output.outputs[0].token_ids)

                responses.append(InferenceResponse(
                    request_id=request.request_id,
                    generated_text=generated_text,
                    ttft_ms=elapsed,  # non-streaming: TTFT unobservable -> full response time
                    total_time_ms=elapsed,
                    num_tokens=num_tokens,
                    model_name=self.model_name,
                    finish_reason=output.outputs[0].finish_reason,
                ))

            return responses

        except Exception as e:
            # Return error responses for all requests
            return [
                InferenceResponse(
                    request_id=req.request_id,
                    generated_text="",
                    ttft_ms=0,
                    total_time_ms=0,
                    num_tokens=0,
                    model_name=self.model_name,
                    finish_reason="error",
                    error=str(e),
                )
                for req in requests
            ]

    def is_ready(self) -> bool:
        """Always ready once initialized."""
        return hasattr(self, 'llm') and self.llm is not None

    def shutdown(self) -> None:
        """Cleanup vLLM engine."""
        if hasattr(self, 'llm'):
            del self.llm
