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
``/reset_prefix_cache``). The POST carries ``?timeout=20`` (S0F-54,
ADR-0149): SGLang 0.5.10.post1 then performs the flush once its scheduler
is fully idle and answers 400 past the deadline, so a refused flush is
still a typed error, never a silent warm window.

Cache report (S0F-88 U7): the launcher always passes --enable-cache-report,
and SGLang 0.5.10.post1 then sends ``prompt_tokens_details`` as
``{"cached_tokens": n}`` when n > 0 and as null when n == 0, the same null it
sends with the report off (usage_processor.py:13-15 at the tag). A null is
therefore read as 0 only after ``probe_cache_report`` read
``enable_cache_report: true`` from GET /server_info (http_server.py:601-618);
otherwise it stays None.

kv_transfer_params stays unparsed: the field is vLLM/NIXL-shaped; SGLang's
disaggregation metadata format (if any) is unconfirmed (ADR-0007 item 6).

Thinking pin (S0F-59, ADR-0151): every chat request carries
``chat_template_kwargs = {"enable_thinking": false}`` by default, the pin
vLLM sends. On the 2026-10-08 landing SGLang served 2,450 of 2,450 rows as
"<think>" in 2 tokens: Qwen3's template opened a thinking block, the
harness stop sequence ended the request, and no answer was ever written.
Whether SGLang honors the kwarg is proven live by
scripts/checks/probe_thinking_pin.py at the master's stage 4, which fails
the run when the served text still carries the marker.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import requests

from .engine import InferenceRequest  # noqa: F401  (re-exported for type context)
from .openai_chat_adapter import OpenAIChatAdapter

#: S0F-59 (ADR-0151): the chat-template pin sent by default, the one vLLM
#: sends (vllm_adapter._apply_engine_chat_extras). ``chat_template_kwargs=None``
#: at construction means this pin; an explicit ``{}`` sends nothing (the
#: pre-ADR-0151 behavior); the runner's CAGE_SGLANG_CHAT_TEMPLATE_KWARGS
#: override replaces it.
SGLANG_CHAT_TEMPLATE_KWARGS: Dict[str, Any] = {"enable_thinking": False}


class SGLangAdapter(OpenAIChatAdapter):
    """HTTP client adapter for an SGLang OpenAI-compatible server."""

    engine_id: str = "sglang"
    # NIXL-shaped kv_transfer_params are vLLM telemetry; not parsed here.
    _kv_transfer_telemetry: bool = False
    # SGLang's native radix-cache flush endpoint, posted in the deferred form:
    # with ?timeout=<s> SGLang 0.5.10.post1 performs the flush once its
    # scheduler is fully idle (a forward still in the overlap result queue
    # counts as busy and no gauge shows it; live 2026-10-08 the plain POST
    # was refused with "#queue-req: 0, #running-req: 0") and answers 400
    # "Timed out waiting for idle state." past the deadline, so fail-closed
    # stays (S0F-54, ADR-0149; read from sglang/srt/managers/scheduler.py on
    # the pod). 20 s stays under the 30 s POST timeout of flush_cache().
    _flush_endpoint: Optional[str] = "/flush_cache"
    _flush_query: Optional[str] = "timeout=20"

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
                template renderer. None (the default) sends
                SGLANG_CHAT_TEMPLATE_KWARGS, ``{"enable_thinking": False}``
                (S0F-59, ADR-0151); an explicit ``{}`` sends nothing. Whether
                SGLang honors the kwarg is proven live by the stage 4 probe
                (scripts/checks/probe_thinking_pin.py), never assumed here.
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
        self.chat_template_kwargs: Optional[Dict[str, Any]] = (
            dict(chat_template_kwargs)
            if chat_template_kwargs is not None
            else dict(SGLANG_CHAT_TEMPLATE_KWARGS)
        )
        self.request_logprobs = request_logprobs
        # S0F-88 U7: True only after probe_cache_report read the server's flag
        self.cache_report: bool = False

    def probe_cache_report(self) -> Optional[bool]:
        """Read ``enable_cache_report`` from GET /server_info (S0F-88 U7).

        Returns the server's boolean, or None when the endpoint does not
        answer with one; ``cache_report`` is set True only on a True reading,
        so an unread flag never turns a null into a fabricated 0."""
        try:
            resp = requests.get(f"{self.api_base}/server_info", timeout=10)
            resp.raise_for_status()
            value = resp.json().get("enable_cache_report")
        except (requests.exceptions.RequestException, ValueError, AttributeError):
            value = None
        flag = value if isinstance(value, bool) else None
        self.cache_report = flag is True
        return flag

    def _apply_engine_chat_extras(self, payload: Dict[str, Any]) -> None:
        """SGLang chat extras: the thinking pin by default (ADR-0151), an
        empty pin sends nothing; logprobs stay opt-in (unverified live)."""
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
        token reporting is charter D2.1 [VERIFY-LIVE]). An absent or null
        cached-token block stays ``None`` unless the server's cache report is
        verified on (``cache_report``, S0F-88 U7): SGLang then sends null for
        a request with no cached token, so a usage object with a prompt count
        and no details means 0 cached.
        """
        prompt_tokens, cached, completion_tokens = super()._extract_usage(usage)
        if self.cache_report and cached is None and prompt_tokens is not None:
            cached = 0
        return prompt_tokens, cached, completion_tokens

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
            # S0F-88 U7: the launcher flag; null reads as 0 only once verified on
            "cached_token_server_flag": "--enable-cache-report",
            "cached_token_absent_means_zero": self.cache_report,
            "kv_usage_gauge": False,  # adapter never scrapes /metrics (ADR-0007 item 5)
            "flush_endpoint": self._flush_endpoint,
            "kv_transfer_params": False,  # NIXL-shaped; no SGLang parser (ADR-0007 item 6)
            # sent on every chat request by default (ADR-0151); that the
            # engine honors it is proven live by the stage 4 probe
            "chat_template_thinking_pin": True,
            "logprobs": "verify-live",
            "truncate_prompt_tokens": False,  # vLLM extension; fails closed here
        }
