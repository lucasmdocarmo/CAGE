"""SGLang adapter for the CAGE framework (ADR-0007).

SGLangAdapter targets an SGLang server's OpenAI-compatible surface
(``POST /v1/chat/completions`` / ``/v1/completions``). SGLang is the charter
D2 "first structurally different manager": RadixAttention organizes the KV
cache as a radix tree over token sequences -- reuse is automatic
longest-prefix matching on the tree (not block hashing), eviction is LRU over
tree nodes, and the scheduler is cache-aware (Zheng et al. 2024, "SGLang:
Efficient Execution of Structured Language Model Programs", arXiv:2312.07104
-- the RadixAttention paper).

Telemetry honesty (charter D2.1): SGLang "reports cached tokens / hit rate"
but per-request granularity is [VERIFY-LIVE] -- unverified in this codebase.
``cached_prompt_tokens`` is parsed from ``usage.prompt_tokens_details.
cached_tokens`` when the server provides it and is otherwise ``None`` with
``cached_token_telemetry_available=False`` -- NEVER fabricated and NEVER
coerced to zero (the absent-means-zero semantic is a verified vLLM-0.11.0
quirk, not an OpenAI-schema guarantee; inheriting it here would silently
misreport cached-token rates, corrupting the D2 telemetry-parity gate).

Cache flush: SGLang natively exposes ``POST /flush_cache`` (flushes the radix
cache), used for cold-start-per-trial (the vLLM analogue is the dev-gated
``/reset_prefix_cache``).

kv_transfer_params stays unparsed: the field is vLLM/NIXL-shaped; SGLang's
disaggregation metadata format (if any) is unconfirmed (ADR-0007 item 6).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .engine import InferenceRequest  # noqa: F401  (re-exported for type context)
from .openai_chat_adapter import OpenAIChatAdapter


class SGLangAdapter(OpenAIChatAdapter):
    """HTTP client adapter for an SGLang OpenAI-compatible server."""

    engine_id: str = "sglang"
    # NIXL-shaped kv_transfer_params are vLLM telemetry; not parsed here.
    _kv_transfer_telemetry: bool = False
    # SGLang's native radix-cache flush endpoint.
    _flush_endpoint: Optional[str] = "/flush_cache"

    def __init__(
        self,
        model_name: str,
        api_base: str = "http://localhost:30000",
        timeout: int = 300,
        include_usage_in_stream: bool = True,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        request_logprobs: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create an adapter targeting an SGLang server.

        Args:
            model_name: Served model name.
            api_base: Base URL of the SGLang server (SGLang's default port is
                30000).
            timeout: Requests timeout (seconds).
            include_usage_in_stream: If True, request a final streaming usage
                chunk (``stream_options.include_usage`` -- OpenAI schema).
            chat_template_kwargs: Extra kwargs for the server-side chat
                template renderer (e.g. ``{"enable_thinking": False}``). NOT
                sent by default: whether SGLang honors the same kwarg name and
                Jinja semantics as vLLM is [VERIFY-LIVE] (ADR-0007); pass it
                explicitly once verified at preflight.
            request_logprobs: If True, request per-token logprobs
                (``logprobs=true, top_logprobs=0``, OpenAI chat schema).
                Default False until verified live against the pinned SGLang.
            **kwargs: Forwarded to OpenAIChatAdapter (e.g. max_retries).
        """
        super().__init__(
            model_name,
            api_base=api_base,
            timeout=timeout,
            include_usage_in_stream=include_usage_in_stream,
            **kwargs,
        )
        self.chat_template_kwargs = dict(chat_template_kwargs) if chat_template_kwargs else None
        self.request_logprobs = request_logprobs

    def _apply_engine_chat_extras(self, payload: Dict[str, Any]) -> None:
        """SGLang chat extras -- opt-in only, nothing unverified sent silently."""
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        if self.request_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 0

    def _extract_usage(
        self, usage: Dict[str, Any]
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """SGLang usage extraction: None-with-provenance for absent fields.

        Parses ``prompt_tokens``/``completion_tokens`` and, when present,
        ``prompt_tokens_details.cached_tokens`` (SGLang's per-request cached-
        token reporting is charter D2.1 [VERIFY-LIVE]). An absent cached-token
        block stays ``None`` -- deliberately NOT the vLLM absent-means-zero
        coercion, whose semantics are unverified on SGLang.
        """
        return super()._extract_usage(usage)

    def capabilities(self) -> Dict[str, Any]:
        """Capability declaration driving the charter-D2 telemetry-parity gate.

        Values: True (verified in this codebase), False/None (absent or
        deliberately not implemented), "verify-live" (documented upstream,
        unverified here -- charter D2.1 [VERIFY-LIVE]).
        """
        return {
            "engine": self.engine_id,
            "serving_grade": True,
            "in_process": False,
            "streamed_ttft": True,
            "cached_token_telemetry": "verify-live",  # D2.1: per-request granularity unverified
            "cached_token_server_flag": None,
            "cached_token_absent_means_zero": False,  # absence stays None, never coerced
            "kv_usage_gauge": False,  # adapter never scrapes /metrics (ADR-0007 item 5)
            "flush_endpoint": self._flush_endpoint,
            "kv_transfer_params": False,  # NIXL-shaped; no SGLang parser (ADR-0007 item 6)
            "chat_template_thinking_pin": "verify-live",
            "logprobs": "verify-live",
            "truncate_prompt_tokens": False,  # vLLM extension; fails closed here
        }
