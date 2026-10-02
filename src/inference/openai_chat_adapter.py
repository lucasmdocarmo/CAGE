"""Shared OpenAI-compatible adapter machinery (ADR-0007, CAGE technical review 2026-08-04).

Extracted from ``VLLMAdapter`` (src/inference/vllm_adapter.py) because the
SSE-parsing / TTFT-by-our-clock plumbing is genuinely engine-agnostic: any
OpenAI-compatible endpoint (vLLM, SGLang, LMDeploy api_server) speaks the same
``POST /v1/chat/completions`` + ``data:`` SSE line protocol, the same
first-non-empty-content-delta convention, and (where supported) a final usage
chunk via ``stream_options.include_usage``.

What deliberately does NOT generalize (ADR-0007 consequences) and is therefore
a per-engine hook, never inherited base behavior:

- **usage / cached-token extraction** (``_extract_usage``): vLLM's
  ``usage.prompt_tokens_details.cached_tokens`` semantics (absent block means
  ZERO) are a verified vLLM-0.11.0 quirk behind a vLLM-specific server flag
  (``--enable-prompt-tokens-details``); the SGLang and LMDeploy equivalents
  are charter D2.1 [VERIFY-LIVE]. The base parses generically and returns
  ``None``-with-provenance when a field is absent -- never a fabricated
  number, never a coerced zero.
- **chat-template / thinking-mode handling** (``_apply_engine_chat_extras``):
  ``chat_template_kwargs={"enable_thinking": False}`` is verified against
  vLLM 0.11.0's ChatCompletionRequest schema only; other engines pin their
  own (or send nothing until verified live).
- **``truncate_prompt_tokens``** (``_apply_truncate_prompt_tokens``): a vLLM
  extension parameter. The base FAILS CLOSED (typed error) so an engine that
  would silently ignore it can never serve an untruncated prompt as if it
  were truncated.
- **``kv_transfer_params``**: vLLM/NIXL-shaped metadata; parsing is gated by
  the ``_kv_transfer_telemetry`` class flag (vLLM only).

Telemetry honesty contract: every response carries explicit availability
flags -- ``usage_telemetry_available`` / ``cached_token_telemetry_available``
plus ``engine_id`` -- as plain attributes (consumed via getattr, so the shared
``InferenceResponse`` dataclass schema in src/inference/engine.py is
untouched), and since ADR-0118 (Batch 2 W5) the ``num_tokens_source`` label
saying whether ``num_tokens`` is the engine's usage.completion_tokens or the
whitespace word count of the text (no usage chunk). ``capabilities()`` exposes each adapter's declared surface so the
charter-D2 telemetry-parity preflight gate can be driven from data.

Retry semantics: ``max_retries``/``retry_backoff_s`` (default 0 -- behavior
identical to the pre-refactor adapter) re-issue a request whose attempt came
back as a recorded transport/parse error row (``finish_reason == "error"``).
The returned row's clocks measure the FINAL attempt only; the attempt count
is stamped as the plain attribute ``retries``.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp
import asyncio
import requests

from .engine import (
    InferenceEngine,
    InferenceRequest,
    InferenceResponse,
    NUM_TOKENS_SOURCE_USAGE,
    NUM_TOKENS_SOURCE_WHITESPACE,
)
from .errors import EngineCapabilityUnavailableError

#: S0F-25 (ADR-0135): the finish reasons of a SERVED response. vLLM 0.19.1 has
#: five (stop, length, abort, error, repetition: v1/engine/__init__.py:29); a
#: response that ended any other way, or with none, becomes an error row in
#: ``_finalize``. An allow-list on purpose: a reason a later engine version
#: adds is refused loudly instead of being recorded as a served answer.
SERVED_FINISH_REASONS: frozenset = frozenset({"stop", "length"})
#: Stable prefixes of the error text of an unserved response, so an analysis
#: can count the kinds without a new column: an in-band engine error object,
#: a response that ended with no finish_reason, a finish_reason outside the
#: served set (abort, repetition, anything unknown).
ERROR_KIND_ENGINE = "engine_error"
ERROR_KIND_NO_FINISH = "no_finish_reason"
ERROR_KIND_UNSERVED = "unserved_finish_reason"


def _exc_text(exc: BaseException) -> str:
    """``str(exc)``, never empty: an exception with no message (an aiohttp
    total timeout raises ``asyncio.TimeoutError()``, whose text is "") is
    named by its type. An error row with empty text reads as a served row to
    every consumer that keys on the truthiness of ``error`` (S0F-25)."""
    return str(exc) or type(exc).__name__


def _engine_error_text(obj: Any) -> Optional[str]:
    """The error text of an in-band engine error object, or None for a normal
    chunk (S0F-25). Two shapes: ``{"error": {"message", "type", "param",
    "code"}}`` (vLLM 0.19.1 streaming error events, LMDeploy 0.17.0) and the
    flat ``{"object": "error", "message", "type", "code"}``. Never empty:
    vLLM's generate error carries message "", so type and code ride the text.
    """
    if not isinstance(obj, dict):
        return None
    err = obj.get("error")
    if not err and obj.get("object") == "error":
        err = obj
    if not err:
        return None
    if not isinstance(err, dict):
        return f"{ERROR_KIND_ENGINE}: {err}"
    head = " ".join(str(err[k]) for k in ("type", "code") if err.get(k) not in (None, ""))
    message = str(err.get("message") or "").strip()
    detail = ": ".join(part for part in (head, message) if part)
    return f"{ERROR_KIND_ENGINE}: {detail}" if detail else ERROR_KIND_ENGINE


class OpenAIChatAdapter(InferenceEngine):
    """Base HTTP client for OpenAI-compatible serving engines.

    Subclasses (VLLMAdapter, SGLangAdapter, LMDeployAdapter) override the
    per-engine hooks documented in the module docstring; the streaming/TTFT/
    request-construction machinery here is shared verbatim.
    """

    #: Engine family identifier stamped onto responses and capabilities().
    engine_id: str = "openai-chat"
    #: Parse vLLM/NIXL-shaped kv_transfer_params from bodies/headers (vLLM only).
    _kv_transfer_telemetry: bool = False
    #: Engine cache-flush endpoint path, or None when the engine has none.
    _flush_endpoint: Optional[str] = None
    #: Query string the flush POST carries, or None. Kept apart from the path
    #: so ``_flush_endpoint`` stays the bare route capabilities() declares and
    #: the pd proxy matches (S0F-27: vLLM's honest reset mode is a query).
    _flush_query: Optional[str] = None

    def __init__(
        self,
        model_name: str,
        api_base: str = "http://localhost:8000",
        timeout: int = 300,
        include_usage_in_stream: bool = True,
        max_retries: int = 0,
        retry_backoff_s: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Create an adapter targeting an OpenAI-compatible server.

        Args:
            model_name: Served model name.
            api_base: Base URL of the serving engine (or the CAGE router).
            timeout: Requests timeout (seconds).
            include_usage_in_stream: If True, request a final streaming usage
                chunk so prompt/cached token telemetry can be extracted.
            max_retries: Extra attempts for a request whose attempt produced a
                transport/parse error row. Default 0 preserves the historical
                single-attempt behavior exactly.
            retry_backoff_s: Base sleep between retry attempts (exponential:
                ``retry_backoff_s * 2**attempt``).
        """
        super().__init__(model_name, **kwargs)
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.include_usage_in_stream = include_usage_in_stream
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.completions_url = f"{self.api_base}/v1/completions"
        self.chat_completions_url = f"{self.api_base}/v1/chat/completions"

    # ------------------------------------------------------------------ #
    # Per-engine hooks (ADR-0007: never inherited silently)
    # ------------------------------------------------------------------ #

    def _apply_engine_chat_extras(self, payload: Dict[str, Any]) -> None:
        """Add engine-specific chat-completions fields (chat template kwargs,
        logprobs, ...) to the payload IN PLACE.

        Base default: nothing. Chat-template/thinking-mode handling is
        engine-specific by design (ADR-0007): vLLM's pinned
        ``chat_template_kwargs`` was verified against vLLM 0.11.0 only.
        """

    def _apply_truncate_prompt_tokens(
        self, payload: Dict[str, Any], request: InferenceRequest
    ) -> None:
        """Apply ``request.truncate_prompt_tokens`` to the payload.

        Base default FAILS CLOSED: ``truncate_prompt_tokens`` is a vLLM
        extension parameter; an engine that silently ignored it would serve an
        untruncated prompt while the row claims truncation.
        """
        raise EngineCapabilityUnavailableError(
            self.engine_id,
            "truncate_prompt_tokens",
            "vLLM extension parameter; unverified on this engine "
            "(charter D2.1 VERIFY-LIVE) -- refusing to send it silently",
        )

    def _extract_usage(
        self, usage: Dict[str, Any]
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Extract (prompt_tokens, cached_prompt_tokens, completion_tokens).

        Generic OpenAI-schema parse: ``prompt_tokens``/``completion_tokens``
        plus ``prompt_tokens_details.cached_tokens`` when the engine provides
        it. A field that is absent stays ``None`` (None-with-provenance, per
        charter D2) -- the base NEVER coerces an absent cached-token block to
        zero; that absent-means-zero semantic is a verified vLLM-0.11.0 quirk
        applied only in VLLMAdapter's override.
        """
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        cached_prompt_tokens = None
        details = usage.get("prompt_tokens_details") or {}
        if isinstance(details, dict):
            cached_prompt_tokens = details.get("cached_tokens")

        return (
            int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else None,
            int(cached_prompt_tokens)
            if isinstance(cached_prompt_tokens, (int, float))
            else None,
            int(completion_tokens)
            if isinstance(completion_tokens, (int, float))
            else None,
        )

    # ------------------------------------------------------------------ #
    # Shared machinery (engine-agnostic; extracted verbatim from VLLMAdapter)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _request_messages(request: InferenceRequest) -> Optional[List[Dict[str, str]]]:
        """Chat messages attached to the request (Decision 1B).

        The runner attaches ``request.messages`` (a plain attribute on the
        InferenceRequest dataclass) when CAGE_PROMPT_MODE=chat; their presence
        routes the call to /v1/chat/completions. Absent/empty -> the legacy
        raw /v1/completions path is used unchanged (CAGE_PROMPT_MODE=raw).
        """
        messages = getattr(request, "messages", None)
        if isinstance(messages, list) and messages:
            return messages
        return None

    def _build_chat_payload(
        self, request: InferenceRequest, messages: List[Dict[str, str]], *, stream: bool
    ) -> Dict[str, Any]:
        """Build an OpenAI-compatible /v1/chat/completions payload.

        Engine-specific fields (chat_template_kwargs, logprobs, ...) are added
        by the ``_apply_engine_chat_extras`` hook, preserving the exact key
        order the pre-refactor VLLMAdapter emitted.
        """
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": list(messages),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": stream,
        }
        self._apply_engine_chat_extras(payload)
        if request.stop:
            payload["stop"] = request.stop
        if request.truncate_prompt_tokens is not None:
            self._apply_truncate_prompt_tokens(payload, request)
        if stream and self.include_usage_in_stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _build_payload(self, request: InferenceRequest, *, stream: bool) -> Dict[str, Any]:
        """Build an OpenAI-compatible /v1/completions payload."""
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": stream,
        }
        if request.stop:
            payload["stop"] = request.stop
        if request.truncate_prompt_tokens is not None:
            self._apply_truncate_prompt_tokens(payload, request)

        # Streaming usage is only included if stream_options is provided.
        if stream and self.include_usage_in_stream:
            payload["stream_options"] = {"include_usage": True}

        return payload

    @staticmethod
    def _extract_chat_logprobs(choice: Dict[str, Any]) -> List[float]:
        """Chosen-token logprobs from a chat choice/delta ``logprobs.content`` list."""
        logprobs_obj = choice.get("logprobs")
        if not isinstance(logprobs_obj, dict):
            return []
        content = logprobs_obj.get("content")
        if not isinstance(content, list):
            return []
        out: List[float] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("logprob"), (int, float)):
                out.append(float(item["logprob"]))
        return out

    @staticmethod
    def _attach_logprob_stats(
        response: InferenceResponse, token_logprobs: List[float]
    ) -> InferenceResponse:
        """Attach mean/sum token logprob as plain attributes (consumed via getattr).

        Plain attributes rather than dataclass fields so the shared
        InferenceResponse schema (src/inference/engine.py) is untouched.
        """
        if token_logprobs:
            response.sum_token_logprob = float(sum(token_logprobs))
            response.mean_token_logprob = float(
                sum(token_logprobs) / len(token_logprobs)
            )
        else:
            response.sum_token_logprob = None
            response.mean_token_logprob = None
        return response

    @staticmethod
    def _completion_token_count(
        completion_tokens: Optional[int], generated_text: str
    ) -> Tuple[int, str]:
        """(num_tokens, provenance): the engine's usage.completion_tokens when
        the usage chunk carried one, else the whitespace word count of the
        text, labeled so a consumer can tell a counted token from a counted
        word (ADR-0118, Batch 2 W5: the decode-phase rule keys on it)."""
        if isinstance(completion_tokens, int):
            return completion_tokens, NUM_TOKENS_SOURCE_USAGE
        return len(generated_text.split()), NUM_TOKENS_SOURCE_WHITESPACE

    def _finalize(
        self,
        response: InferenceResponse,
        *,
        prompt_tokens: Optional[int],
        cached_prompt_tokens: Optional[int],
        num_tokens_source: Optional[str] = None,
    ) -> InferenceResponse:
        """Stamp telemetry-provenance flags as plain attributes.

        ``None`` telemetry is recorded as unavailable, never fabricated
        (charter D2: None-with-provenance). Plain attributes keep the shared
        InferenceResponse schema untouched. ``num_tokens_source`` is the
        ADR-0118 token-count label; error rows carry None.

        The served rule (S0F-25, ADR-0135) lives here because every return
        site of the six request paths passes through: a response without an
        error whose ``finish_reason`` is not in SERVED_FINISH_REASONS (none
        at all, ``abort``, ``repetition``, anything unknown) is rewritten to
        the error-row shape the transport failures already use (empty text,
        zero clocks and counts, no token telemetry). The number of characters
        received stays in the error text, so a failure before the first token
        and one after it remain distinguishable.
        """
        reason = response.finish_reason
        # isinstance first: a non-string reason (an object) is unhashable, and
        # a bare membership test would raise instead of refusing the row.
        served = isinstance(reason, str) and reason in SERVED_FINISH_REASONS
        if not response.error and not served:
            response.error = (
                f"{ERROR_KIND_NO_FINISH}: the response ended without a terminal "
                f"finish_reason after {len(response.generated_text or '')} character(s) of text"
                if reason is None
                else f"{ERROR_KIND_UNSERVED}: {reason!r} (served: {sorted(SERVED_FINISH_REASONS)})"
            )
            response.generated_text = ""
            response.ttft_ms = 0.0
            response.num_tokens = 0
            response.finish_reason = "error"
            response.router_replica = None
            response.prompt_tokens = None
            response.cached_prompt_tokens = None
            response.kv_transfer_params = None
            if hasattr(response, "mean_token_logprob"):
                response.mean_token_logprob = None
                response.sum_token_logprob = None
            prompt_tokens = cached_prompt_tokens = num_tokens_source = None
        response.engine_id = self.engine_id
        response.usage_telemetry_available = prompt_tokens is not None
        response.cached_token_telemetry_available = cached_prompt_tokens is not None
        response.num_tokens_source = num_tokens_source
        return response

    def _extract_header_kv_transfer_params(self, headers: Any) -> Optional[Dict[str, Any]]:
        """Extract simulated KV transfer metadata from response headers when present."""
        if headers is None:
            return None

        raw = headers.get("x-kv-transfer-params")
        if not raw:
            return None

        try:
            parsed = json.loads(raw)
        except Exception:
            return None

        return parsed if isinstance(parsed, dict) else None

    def _header_kv_transfer(self, headers: Any) -> Optional[Dict[str, Any]]:
        """Header KV-transfer metadata, gated by the per-engine telemetry flag."""
        if not self._kv_transfer_telemetry:
            return None
        return self._extract_header_kv_transfer_params(headers)

    def _body_kv_transfer(self, obj: Any) -> Optional[Dict[str, Any]]:
        """Body KV-transfer metadata, gated by the per-engine telemetry flag.

        The field is vLLM/NIXL-shaped (ADR-0007): non-vLLM engines return None
        unless/until their disaggregation metadata format is confirmed live.
        """
        if not self._kv_transfer_telemetry:
            return None
        if isinstance(obj, dict):
            kv_params = obj.get("kv_transfer_params")
            if isinstance(kv_params, dict):
                return kv_params
        return None

    # ------------------------------------------------------------------ #
    # Serving paths
    # ------------------------------------------------------------------ #

    def _stream_completion(self, request: InferenceRequest) -> InferenceResponse:
        """Stream a completion to measure TTFT and optionally collect usage telemetry."""
        start_time = time.time()
        first_token_time: Optional[float] = None
        full_text_parts: List[str] = []
        finish_reason: Optional[str] = None

        prompt_tokens: Optional[int] = None
        cached_prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        kv_transfer_params: Optional[Dict[str, Any]] = None

        router_replica = None
        try:
            with requests.post(
                self.completions_url,
                json=self._build_payload(request, stream=True),
                timeout=self.timeout,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                router_replica = resp.headers.get("x-router-replica")
                kv_transfer_params = self._header_kv_transfer(resp.headers)

                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if data == "[DONE]":
                        break

                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue

                    # S0F-25: an in-band error object is a failed request,
                    # never a chunk to skip (the except below records it).
                    engine_error = _engine_error_text(obj)
                    if engine_error is not None:
                        raise ValueError(engine_error)

                    # Optional KV transfer metadata (used by vLLM P/D connectors).
                    body_kv = self._body_kv_transfer(obj)
                    if body_kv is not None:
                        kv_transfer_params = body_kv

                    # Usage chunk. Its choices are usually empty (the check
                    # below skips it then); a chunk that carries usage BESIDE
                    # choices is read in full, text and finish_reason included
                    # (review 2026-10-02: skipping it dropped both).
                    if isinstance(obj, dict) and "usage" in obj and isinstance(obj["usage"], dict):
                        prompt_tokens, cached_prompt_tokens, completion_tokens = self._extract_usage(
                            obj["usage"]
                        )

                    choices = obj.get("choices") if isinstance(obj, dict) else None
                    if not choices:
                        continue

                    choice = choices[0] if isinstance(choices, list) else {}
                    text_delta = choice.get("text", "") if isinstance(choice, dict) else ""
                    if text_delta:
                        full_text_parts.append(text_delta)
                        if first_token_time is None:
                            first_token_time = time.time()

                    if isinstance(choice, dict):
                        finish_reason = choice.get("finish_reason") or finish_reason
        except (requests.exceptions.RequestException, ValueError) as e:
            # ValueError covers a malformed/truncated streamed chunk (json parse) so a bad
            # response becomes a recorded error row, not a run-ending crash.
            total_time = (time.time() - start_time) * 1000
            return self._finalize(
                InferenceResponse(
                    request_id=request.request_id,
                    generated_text="",
                    ttft_ms=0.0,
                    total_time_ms=total_time,
                    num_tokens=0,
                    model_name=self.model_name,
                    finish_reason="error",
                    error=_exc_text(e),
                ),
                prompt_tokens=None,
                cached_prompt_tokens=None,
            )

        total_time_ms = (time.time() - start_time) * 1000
        ttft_ms = ((first_token_time - start_time) * 1000) if first_token_time else total_time_ms
        generated_text = "".join(full_text_parts)

        num_tokens, num_tokens_source = self._completion_token_count(

            completion_tokens, generated_text

        )

        return self._finalize(
            InferenceResponse(
                request_id=request.request_id,
                generated_text=generated_text,
                ttft_ms=ttft_ms,
                total_time_ms=total_time_ms,
                num_tokens=num_tokens,
                model_name=self.model_name,
                finish_reason=finish_reason,
                router_replica=router_replica,
                prompt_tokens=prompt_tokens,
                cached_prompt_tokens=cached_prompt_tokens,
                kv_transfer_params=kv_transfer_params,
            ),
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            num_tokens_source=num_tokens_source,
        )

    def _stream_chat_completion(
        self, request: InferenceRequest, messages: List[Dict[str, str]]
    ) -> InferenceResponse:
        """Stream a chat completion (Decision 1B).

        Same TTFT semantics as the raw path: TTFT = wall-clock until the FIRST
        non-empty content delta chunk arrives.
        """
        start_time = time.time()
        first_token_time: Optional[float] = None
        full_text_parts: List[str] = []
        finish_reason: Optional[str] = None
        token_logprobs: List[float] = []

        prompt_tokens: Optional[int] = None
        cached_prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        kv_transfer_params: Optional[Dict[str, Any]] = None

        router_replica = None
        try:
            with requests.post(
                self.chat_completions_url,
                json=self._build_chat_payload(request, messages, stream=True),
                timeout=self.timeout,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                router_replica = resp.headers.get("x-router-replica")
                kv_transfer_params = self._header_kv_transfer(resp.headers)

                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if data == "[DONE]":
                        break

                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue

                    # S0F-25: an in-band error object is a failed request,
                    # never a chunk to skip (the except below records it).
                    engine_error = _engine_error_text(obj)
                    if engine_error is not None:
                        raise ValueError(engine_error)

                    body_kv = self._body_kv_transfer(obj)
                    if body_kv is not None:
                        kv_transfer_params = body_kv

                    # Usage chunk: same rule as the raw loop (a chunk with
                    # usage beside choices is read in full).
                    if isinstance(obj, dict) and "usage" in obj and isinstance(obj["usage"], dict):
                        prompt_tokens, cached_prompt_tokens, completion_tokens = self._extract_usage(
                            obj["usage"]
                        )

                    choices = obj.get("choices") if isinstance(obj, dict) else None
                    if not choices:
                        continue

                    choice = choices[0] if isinstance(choices, list) else {}
                    if not isinstance(choice, dict):
                        continue

                    delta = choice.get("delta") or {}
                    text_delta = delta.get("content") or "" if isinstance(delta, dict) else ""
                    if text_delta:
                        full_text_parts.append(text_delta)
                        if first_token_time is None:
                            first_token_time = time.time()

                    token_logprobs.extend(self._extract_chat_logprobs(choice))

                    finish_reason = choice.get("finish_reason") or finish_reason
        except (requests.exceptions.RequestException, ValueError) as e:
            # Same guard as the raw path: a malformed/failed stream becomes a
            # recorded error row, not a run-ending crash.
            total_time = (time.time() - start_time) * 1000
            return self._finalize(
                self._attach_logprob_stats(
                    InferenceResponse(
                        request_id=request.request_id,
                        generated_text="",
                        ttft_ms=0.0,
                        total_time_ms=total_time,
                        num_tokens=0,
                        model_name=self.model_name,
                        finish_reason="error",
                        error=_exc_text(e),
                    ),
                    [],
                ),
                prompt_tokens=None,
                cached_prompt_tokens=None,
            )

        total_time_ms = (time.time() - start_time) * 1000
        ttft_ms = ((first_token_time - start_time) * 1000) if first_token_time else total_time_ms
        generated_text = "".join(full_text_parts)

        num_tokens, num_tokens_source = self._completion_token_count(

            completion_tokens, generated_text

        )

        return self._finalize(
            self._attach_logprob_stats(
                InferenceResponse(
                    request_id=request.request_id,
                    generated_text=generated_text,
                    ttft_ms=ttft_ms,
                    total_time_ms=total_time_ms,
                    num_tokens=num_tokens,
                    model_name=self.model_name,
                    finish_reason=finish_reason,
                    router_replica=router_replica,
                    prompt_tokens=prompt_tokens,
                    cached_prompt_tokens=cached_prompt_tokens,
                    kv_transfer_params=kv_transfer_params,
                ),
                token_logprobs,
            ),
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            num_tokens_source=num_tokens_source,
        )

    def _chat_completion(
        self, request: InferenceRequest, messages: List[Dict[str, str]]
    ) -> InferenceResponse:
        """Non-streaming chat completion (TTFT unobservable -> full response time)."""
        start_time = time.time()

        router_replica = None
        try:
            resp = requests.post(
                self.chat_completions_url,
                json=self._build_chat_payload(request, messages, stream=False),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            router_replica = resp.headers.get("x-router-replica")

            result = resp.json()
            total_time_ms = (time.time() - start_time) * 1000

            choice = result.get("choices", [{}])[0]
            message = choice.get("message") or {}
            generated_text = (message.get("content") or "") if isinstance(message, dict) else ""
            finish_reason = choice.get("finish_reason")
            token_logprobs = self._extract_chat_logprobs(choice)

            usage = result.get("usage") or {}
            prompt_tokens, cached_prompt_tokens, completion_tokens = self._extract_usage(
                usage if isinstance(usage, dict) else {}
            )

            kv_transfer_params = self._body_kv_transfer(result)
            if kv_transfer_params is None:
                kv_transfer_params = self._header_kv_transfer(resp.headers)

            num_tokens, num_tokens_source = self._completion_token_count(

                completion_tokens, generated_text

            )

            return self._finalize(
                self._attach_logprob_stats(
                    InferenceResponse(
                        request_id=request.request_id,
                        generated_text=generated_text,
                        ttft_ms=total_time_ms,
                        total_time_ms=total_time_ms,
                        num_tokens=num_tokens,
                        model_name=self.model_name,
                        finish_reason=finish_reason,
                        router_replica=router_replica,
                        prompt_tokens=prompt_tokens,
                        cached_prompt_tokens=cached_prompt_tokens,
                        kv_transfer_params=kv_transfer_params,
                    ),
                    token_logprobs,
                ),
                prompt_tokens=prompt_tokens,
                cached_prompt_tokens=cached_prompt_tokens,
                num_tokens_source=num_tokens_source,
            )

        except (requests.exceptions.RequestException, ValueError) as e:
            total_time_ms = (time.time() - start_time) * 1000
            return self._finalize(
                self._attach_logprob_stats(
                    InferenceResponse(
                        request_id=request.request_id,
                        generated_text="",
                        ttft_ms=0.0,
                        total_time_ms=total_time_ms,
                        num_tokens=0,
                        model_name=self.model_name,
                        finish_reason="error",
                        error=_exc_text(e),
                    ),
                    [],
                ),
                prompt_tokens=None,
                cached_prompt_tokens=None,
            )

    def _raw_completion(self, request: InferenceRequest) -> InferenceResponse:
        """Non-streaming raw completion (TTFT unobservable -> full response time)."""
        start_time = time.time()
        payload = self._build_payload(request, stream=False)

        router_replica = None
        try:
            resp = requests.post(
                self.completions_url,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            router_replica = resp.headers.get("x-router-replica")

            result = resp.json()
            total_time_ms = (time.time() - start_time) * 1000

            choice = result.get("choices", [{}])[0]
            generated_text = choice.get("text", "")
            finish_reason = choice.get("finish_reason")

            usage = result.get("usage") or {}
            prompt_tokens, cached_prompt_tokens, completion_tokens = self._extract_usage(
                usage if isinstance(usage, dict) else {}
            )

            # Optional KV transfer metadata (used by vLLM P/D connectors).
            kv_transfer_params = self._body_kv_transfer(result)
            if kv_transfer_params is None:
                kv_transfer_params = self._header_kv_transfer(resp.headers)

            num_tokens, num_tokens_source = self._completion_token_count(

                completion_tokens, generated_text

            )

            # Non-streaming: TTFT is unobservable (full response arrives at once), so report
            # it as the full response time rather than a fabricated fraction. Use stream=True
            # for a real TTFT measurement.
            ttft_ms = total_time_ms

            return self._finalize(
                InferenceResponse(
                    request_id=request.request_id,
                    generated_text=generated_text,
                    ttft_ms=ttft_ms,
                    total_time_ms=total_time_ms,
                    num_tokens=num_tokens,
                    model_name=self.model_name,
                    finish_reason=finish_reason,
                    router_replica=router_replica,
                    prompt_tokens=prompt_tokens,
                    cached_prompt_tokens=cached_prompt_tokens,
                    kv_transfer_params=kv_transfer_params,
                ),
                prompt_tokens=prompt_tokens,
                cached_prompt_tokens=cached_prompt_tokens,
                num_tokens_source=num_tokens_source,
            )

        except (requests.exceptions.RequestException, ValueError) as e:
            # ValueError covers an HTTP-200 truncated/malformed body (resp.json() raises
            # json.JSONDecodeError, a ValueError) so it becomes a recorded error row rather
            # than propagating out of the unguarded measured loop and aborting the baseline.
            total_time_ms = (time.time() - start_time) * 1000
            return self._finalize(
                InferenceResponse(
                    request_id=request.request_id,
                    generated_text="",
                    ttft_ms=0.0,
                    total_time_ms=total_time_ms,
                    num_tokens=0,
                    model_name=self.model_name,
                    finish_reason="error",
                    error=_exc_text(e),
                ),
                prompt_tokens=None,
                cached_prompt_tokens=None,
            )

    def _generate_once(
        self, request: InferenceRequest, *, stream: bool
    ) -> InferenceResponse:
        """Single request attempt, routed to the chat or raw path."""
        messages = self._request_messages(request)
        if messages is not None:
            if stream:
                return self._stream_chat_completion(request, messages)
            return self._chat_completion(request, messages)

        if stream:
            return self._stream_completion(request)
        return self._raw_completion(request)

    def generate(self, request: InferenceRequest, *, stream: bool = False) -> InferenceResponse:
        """Generate a completion via the engine's OpenAI-compatible server.

        Requests carrying ``messages`` (Decision 1B chat mode) are served via
        /v1/chat/completions; all others use the legacy raw /v1/completions
        path unchanged (CAGE_PROMPT_MODE=raw escape hatch).

        Retry semantics: with ``max_retries > 0``, an attempt that came back as
        a recorded transport/parse error row (``finish_reason == "error"``) is
        re-issued after an exponential backoff. The returned row's clocks
        measure the final attempt only; the attempt count is stamped as the
        plain attribute ``retries``. Default ``max_retries=0`` is exactly the
        historical single-attempt behavior.
        """
        response = self._generate_once(request, stream=stream)
        attempt = 0
        while response.finish_reason == "error" and attempt < self.max_retries:
            time.sleep(self.retry_backoff_s * (2 ** attempt))
            attempt += 1
            response = self._generate_once(request, stream=stream)
        response.retries = attempt
        return response

    def batch_generate(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Generate responses for batch of requests (sequential for now)."""
        # The serving engine handles batching internally, so requests are sent
        # sequentially. For async batching, use async_batch_generate instead.
        # stream=True: a non-streamed call reports ttft_ms == total_time_ms (see the
        # comment in _raw_completion above), which silently mixes methodologies with the
        # streamed single-request path. Every row must use the same real-TTFT clock
        # regardless of work-unit size, so batch_generate streams too.
        return [self.generate(req, stream=True) for req in requests]

    async def async_generate(self, request: InferenceRequest) -> InferenceResponse:
        """Async (non-streaming) completion request.

        Note: retry semantics are NOT applied on the async path (single
        attempt), matching the pre-refactor behavior.
        """
        start_time = time.time()

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": False,
        }

        if request.stop:
            payload["stop"] = request.stop

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.completions_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    resp.raise_for_status()
                    result = await resp.json()

                    total_time_ms = (time.time() - start_time) * 1000

                    choice = result.get("choices", [{}])[0]
                    generated_text = choice.get("text", "")
                    finish_reason = choice.get("finish_reason")

                    usage = result.get("usage") or {}
                    prompt_tokens, cached_prompt_tokens, completion_tokens = self._extract_usage(
                        usage if isinstance(usage, dict) else {}
                    )

                    kv_transfer_params = self._body_kv_transfer(result)
                    if kv_transfer_params is None:
                        kv_transfer_params = self._header_kv_transfer(resp.headers)

                    num_tokens, num_tokens_source = self._completion_token_count(

                        completion_tokens, generated_text

                    )
                    # Non-streaming: TTFT unobservable -> report full response time.
                    ttft_ms = total_time_ms

                    response = self._finalize(
                        InferenceResponse(
                            request_id=request.request_id,
                            generated_text=generated_text,
                            ttft_ms=ttft_ms,
                            total_time_ms=total_time_ms,
                            num_tokens=num_tokens,
                            model_name=self.model_name,
                            finish_reason=finish_reason,
                            router_replica=resp.headers.get("x-router-replica"),
                            prompt_tokens=prompt_tokens,
                            cached_prompt_tokens=cached_prompt_tokens,
                            kv_transfer_params=kv_transfer_params,
                        ),
                        prompt_tokens=prompt_tokens,
                        cached_prompt_tokens=cached_prompt_tokens,
                        num_tokens_source=num_tokens_source,
                    )
                    # Provenance: this ttft_ms is the full-response PROXY, not
                    # a streamed first-token time. Measured open-loop rows must
                    # come from async_stream_generate instead (D6 §6.3).
                    if response.error is None:
                        response.ttft_methodology = "full-response-proxy"
                    return response

        except Exception as e:
            total_time_ms = (time.time() - start_time) * 1000
            return self._finalize(
                InferenceResponse(
                    request_id=request.request_id,
                    generated_text="",
                    ttft_ms=0.0,
                    total_time_ms=total_time_ms,
                    num_tokens=0,
                    model_name=self.model_name,
                    finish_reason="error",
                    error=_exc_text(e),
                ),
                prompt_tokens=None,
                cached_prompt_tokens=None,
            )

    async def async_stream_generate(
        self,
        request: InferenceRequest,
        *,
        on_first_token: Optional[Callable[[], None]] = None,
    ) -> InferenceResponse:
        """Async STREAMING completion — real first-token TTFT by our clock.

        The D6 open-loop dispatcher issues every measured request through this
        path: ``async_generate`` is non-streaming, so its ``ttft_ms`` is the
        FULL response time ('full-response-proxy'), and comparing that against
        the streamed single-stream baseline would silently mix TTFT
        methodologies — the exact failure the ``batch_generate`` stream=True
        comment (and its regression test) exists to prevent. Same SSE
        ``data:``-line parsing and first-non-empty-content-delta TTFT
        convention as the sync ``_stream_completion`` /
        ``_stream_chat_completion`` paths; requests carrying ``messages``
        (Decision 1B chat mode) go to /v1/chat/completions, all others to the
        raw /v1/completions path.

        ``on_first_token`` is invoked exactly once, at the first non-empty
        content delta: the open-loop dispatcher stamps ``first_token_ts`` on
        ITS OWN (injectable) clock there, so the §6.3 coordinated-omission-
        corrected TTFT — clocked from the INTENDED arrival — is
        reconstructible from the recorded columns.

        Retry semantics are NOT applied on the async path (single attempt),
        matching ``async_generate``. Transport/parse failures become recorded
        error rows (``finish_reason == "error"``), never raises — a failed
        request is data (D6 §6.1 attainment), not a run abort.
        """
        messages = self._request_messages(request)
        if messages is not None:
            url = self.chat_completions_url
            payload = self._build_chat_payload(request, messages, stream=True)
        else:
            url = self.completions_url
            payload = self._build_payload(request, stream=True)

        start_time = time.time()
        first_token_time: Optional[float] = None
        full_text_parts: List[str] = []
        finish_reason: Optional[str] = None
        token_logprobs: List[float] = []

        prompt_tokens: Optional[int] = None
        cached_prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        kv_transfer_params: Optional[Dict[str, Any]] = None
        router_replica = None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    resp.raise_for_status()
                    router_replica = resp.headers.get("x-router-replica")
                    kv_transfer_params = self._header_kv_transfer(resp.headers)

                    async for raw_line in resp.content:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except Exception:
                            continue

                        # S0F-25: same rule as the sync loops.
                        engine_error = _engine_error_text(obj)
                        if engine_error is not None:
                            raise ValueError(engine_error)

                        body_kv = self._body_kv_transfer(obj)
                        if body_kv is not None:
                            kv_transfer_params = body_kv

                        # Usage chunk: same rule as the sync loops (a chunk
                        # with usage beside choices is read in full).
                        if (
                            isinstance(obj, dict)
                            and "usage" in obj
                            and isinstance(obj["usage"], dict)
                        ):
                            (
                                prompt_tokens,
                                cached_prompt_tokens,
                                completion_tokens,
                            ) = self._extract_usage(obj["usage"])

                        choices = obj.get("choices") if isinstance(obj, dict) else None
                        if not choices:
                            continue
                        choice = choices[0] if isinstance(choices, list) else {}
                        if not isinstance(choice, dict):
                            continue

                        if messages is not None:
                            delta = choice.get("delta") or {}
                            text_delta = (
                                (delta.get("content") or "")
                                if isinstance(delta, dict)
                                else ""
                            )
                            token_logprobs.extend(self._extract_chat_logprobs(choice))
                        else:
                            text_delta = choice.get("text", "") or ""
                        finish_reason = choice.get("finish_reason") or finish_reason

                        if text_delta:
                            full_text_parts.append(text_delta)
                            if first_token_time is None:
                                first_token_time = time.time()
                                if on_first_token is not None:
                                    on_first_token()
        except Exception as e:
            # Same guard as async_generate: a transport/parse failure becomes
            # a recorded error row, never a dispatcher-visible exception type
            # change.
            total_time_ms = (time.time() - start_time) * 1000
            return self._finalize(
                self._attach_logprob_stats(
                    InferenceResponse(
                        request_id=request.request_id,
                        generated_text="",
                        ttft_ms=0.0,
                        total_time_ms=total_time_ms,
                        num_tokens=0,
                        model_name=self.model_name,
                        finish_reason="error",
                        error=_exc_text(e),
                    ),
                    [],
                ),
                prompt_tokens=None,
                cached_prompt_tokens=None,
            )

        total_time_ms = (time.time() - start_time) * 1000
        ttft_ms = (
            ((first_token_time - start_time) * 1000)
            if first_token_time
            else total_time_ms
        )
        generated_text = "".join(full_text_parts)
        num_tokens, num_tokens_source = self._completion_token_count(
            completion_tokens, generated_text
        )

        response = self._finalize(
            self._attach_logprob_stats(
                InferenceResponse(
                    request_id=request.request_id,
                    generated_text=generated_text,
                    ttft_ms=ttft_ms,
                    total_time_ms=total_time_ms,
                    num_tokens=num_tokens,
                    model_name=self.model_name,
                    finish_reason=finish_reason,
                    router_replica=router_replica,
                    prompt_tokens=prompt_tokens,
                    cached_prompt_tokens=cached_prompt_tokens,
                    kv_transfer_params=kv_transfer_params,
                ),
                token_logprobs,
            ),
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            num_tokens_source=num_tokens_source,
        )
        if response.error is None:
            response.ttft_methodology = "streamed-first-delta"
        return response

    async def async_batch_generate(
        self, requests: List[InferenceRequest]
    ) -> List[InferenceResponse]:
        """Async batch generation (concurrent requests)."""
        tasks = [self.async_generate(req) for req in requests]
        return await asyncio.gather(*tasks)

    # ------------------------------------------------------------------ #
    # Server management
    # ------------------------------------------------------------------ #

    def flush_cache(self) -> None:
        """POST the engine's cache-flush endpoint (fail-closed).

        Raises:
            EngineCapabilityUnavailableError: when the engine exposes no flush
                endpoint, or the POST fails. A flush that silently no-ops
                would let a "cold-start" trial serve warm-cache rows.
        """
        if not self._flush_endpoint:
            raise EngineCapabilityUnavailableError(
                self.engine_id,
                "flush_endpoint",
                "engine exposes no documented cache-flush endpoint",
            )
        url = f"{self.api_base}{self._flush_endpoint}"
        if self._flush_query:
            url = f"{url}?{self._flush_query}"
        try:
            resp = requests.post(url, timeout=30)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise EngineCapabilityUnavailableError(
                self.engine_id,
                "flush_endpoint",
                f"POST {url} failed: {e}",
            ) from e

    def is_ready(self) -> bool:
        """Check if the server is ready and serving the expected model."""
        try:
            health_url = f"{self.api_base}/health"
            response = requests.get(health_url, timeout=5)
            if response.status_code != 200:
                return False

            # Also verify the model is loaded
            models_url = f"{self.api_base}/v1/models"
            models_response = requests.get(models_url, timeout=5)
            if models_response.status_code != 200:
                return False

            models_data = models_response.json()
            loaded_models = [m.get("id") for m in models_data.get("data", [])]

            if self.model_name not in loaded_models:
                print(f"WARNING: Model '{self.model_name}' not loaded on server.")
                print(f"         Server has: {loaded_models}")
                return False

            return True
        except Exception:
            return False

    def get_loaded_model(self) -> Optional[str]:
        """Get the model currently loaded on the server."""
        try:
            models_url = f"{self.api_base}/v1/models"
            response = requests.get(models_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                if models:
                    return models[0].get("id")
        except Exception:
            pass
        return None

    def shutdown(self) -> None:
        """No cleanup needed (server is external)."""
        pass
