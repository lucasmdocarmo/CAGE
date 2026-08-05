"""LMDeploy adapter for the CAGE framework (ADR-0007).

LMDeployAdapter targets the LMDeploy ``api_server``'s OpenAI-compatible
surface (``POST /v1/chat/completions`` / ``/v1/completions``).

TurboMind notes (charter D2 + P7 policy, user-confirmed 2026-07-27):
LMDeploy ships two runtimes -- **TurboMind** (kernel-first: blocked KV cache
managed by a persistent-batching runtime with hand-tuned CUDA kernels; the
third KV-management family the charter selected LMDeploy to represent) and a
**PyTorch fallback engine** whose paged-KV manager duplicates vLLM's mechanism
family. The pinned policy is **TurboMind-pinned-where-supported, absent
elsewhere, never mixed**: LMDeploy cells run TurboMind exclusively on
Qwen3-14B and Llama-3.3-70B; models TurboMind does not support are reported
ABSENT (by policy) rather than silently served by the PyTorch engine. The
backend choice is made server-side at launch (``lmdeploy serve api_server
--backend turbomind``); this adapter is transport-only and CANNOT observe
which backend the server actually selected -- the preflight gate must verify
TurboMind is running (not the silent PyTorch fallback) before any measured
row is served (cloud/VLLM_COMPATIBILITY.md section 7 engine matrix).

Telemetry honesty (charter D2.1): LMDeploy is the "weakest documented
telemetry" engine and its cached-token fields are [VERIFY-LIVE] -- a failed
cache-telemetry parity gate demotes its cells to serving-only or excluded.
``cached_prompt_tokens`` is parsed from ``usage.prompt_tokens_details.
cached_tokens`` only if the server ever provides it and is otherwise ``None``
with ``cached_token_telemetry_available=False`` -- NEVER fabricated, NEVER
zero-coerced (the absent-means-zero semantic is a verified vLLM-0.11.0 quirk).

Cache flush: LMDeploy documents no cache-flush endpoint; ``flush_cache()``
fails closed with a typed error (cold-start-per-trial on LMDeploy requires a
server restart, which the run scripts own).

kv_transfer_params stays unparsed: vLLM/NIXL-shaped; LMDeploy's proxy-based
disaggregation is "experimental at best" per the charter and its metadata
format is unconfirmed (ADR-0007 item 6).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .engine import InferenceRequest  # noqa: F401  (re-exported for type context)
from .openai_chat_adapter import OpenAIChatAdapter


class LMDeployAdapter(OpenAIChatAdapter):
    """HTTP client adapter for an LMDeploy OpenAI-compatible api_server."""

    engine_id: str = "lmdeploy-turbomind"
    # NIXL-shaped kv_transfer_params are vLLM telemetry; not parsed here.
    _kv_transfer_telemetry: bool = False
    # No documented cache-flush endpoint -> flush_cache() fails closed.
    _flush_endpoint: Optional[str] = None

    def __init__(
        self,
        model_name: str,
        api_base: str = "http://localhost:23333",
        timeout: int = 300,
        include_usage_in_stream: bool = True,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        request_logprobs: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create an adapter targeting an LMDeploy api_server.

        Args:
            model_name: Served model name.
            api_base: Base URL of the LMDeploy api_server (LMDeploy's default
                port is 23333).
            timeout: Requests timeout (seconds).
            include_usage_in_stream: If True, request a final streaming usage
                chunk (``stream_options.include_usage`` -- OpenAI schema;
                LMDeploy support is [VERIFY-LIVE]).
            chat_template_kwargs: Extra kwargs for the server-side chat
                template renderer. NOT sent by default: whether LMDeploy
                honors vLLM's kwarg name/semantics is [VERIFY-LIVE]
                (ADR-0007); pass explicitly once verified at preflight.
            request_logprobs: If True, request per-token logprobs
                (``logprobs=true, top_logprobs=0``). Default False until
                verified live against the pinned LMDeploy.
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
        """LMDeploy chat extras -- opt-in only, nothing unverified sent silently."""
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        if self.request_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 0

    def _extract_usage(
        self, usage: Dict[str, Any]
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """LMDeploy usage extraction: None-with-provenance for absent fields.

        LMDeploy has the charter's "weakest documented telemetry": only
        ``prompt_tokens``/``completion_tokens`` are relied on. A cached-token
        details block is parsed opportunistically if the server ever emits
        one, and its absence stays ``None`` -- deliberately NOT the vLLM
        absent-means-zero coercion, whose semantics are unverified here.
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
            "cached_token_telemetry": "verify-live",  # D2.1: "weakest documented"; may fail the gate
            "cached_token_server_flag": None,
            "cached_token_absent_means_zero": False,  # absence stays None, never coerced
            "kv_usage_gauge": False,  # adapter never scrapes /metrics (ADR-0007 item 5)
            "flush_endpoint": None,  # none documented; flush_cache() fails closed
            "kv_transfer_params": False,  # NIXL-shaped; no LMDeploy parser (ADR-0007 item 6)
            "chat_template_thinking_pin": "verify-live",
            "logprobs": "verify-live",
            "truncate_prompt_tokens": False,  # vLLM extension; fails closed here
        }
