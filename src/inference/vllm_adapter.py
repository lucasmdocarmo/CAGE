"""vLLM adapters for the CAGE framework.

This module provides:
- VLLMAdapter: OpenAI-compatible HTTP client with optional streaming TTFT.
- VLLMOfflineAdapter: in-process vLLM execution for local debugging.

We also extract optional vLLM telemetry when available:
- usage.prompt_tokens
- usage.prompt_tokens_details.cached_tokens (requires vLLM flag --enable-prompt-tokens-details)
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import asyncio
import requests

from .engine import InferenceEngine, InferenceRequest, InferenceResponse


class VLLMAdapter(InferenceEngine):
    """HTTP client adapter for a vLLM OpenAI-compatible server."""

    def __init__(
        self,
        model_name: str,
        api_base: str = "http://localhost:8000",
        timeout: int = 300,
        include_usage_in_stream: bool = True,
        **kwargs,
    ):
        """Create an adapter targeting a vLLM server.

        Args:
            model_name: Served model name.
            api_base: Base URL of a vLLM server or the CAGE router.
            timeout: Requests timeout (seconds).
            include_usage_in_stream: If True, request a final streaming usage
                chunk so we can extract prompt/cached token telemetry.
        """
        super().__init__(model_name, **kwargs)
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.include_usage_in_stream = include_usage_in_stream
        self.completions_url = f"{self.api_base}/v1/completions"
        self.chat_completions_url = f"{self.api_base}/v1/chat/completions"

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
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": list(messages),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False},
            "logprobs": True,
            "top_logprobs": 0,
        }
        if request.stop:
            payload["stop"] = request.stop
        if request.truncate_prompt_tokens is not None:
            payload["truncate_prompt_tokens"] = request.truncate_prompt_tokens
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
            payload["truncate_prompt_tokens"] = request.truncate_prompt_tokens

        # vLLM only includes streaming usage if stream_options is provided.
        if stream and self.include_usage_in_stream:
            payload["stream_options"] = {"include_usage": True}

        return payload

    def _extract_usage(self, usage: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Extract (prompt_tokens, cached_prompt_tokens, completion_tokens) from usage."""
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        cached_prompt_tokens = None
        details = usage.get("prompt_tokens_details") or {}
        if isinstance(details, dict):
            cached_prompt_tokens = details.get("cached_tokens")

        prompt_tokens_out = (
            int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else None
        )
        cached_out = (
            int(cached_prompt_tokens)
            if isinstance(cached_prompt_tokens, (int, float))
            else None
        )
        # Audit 2026-07-16 M6 (cached-zero-recorded-as-missing): vLLM 0.11.0 OMITS
        # usage.prompt_tokens_details whenever num_cached_tokens is falsy (cold request),
        # even with --enable-prompt-tokens-details. Recording those rows as None made every
        # cached_prompt_tokens/cached_prompt_ratio statistic silently conditional-on-hit
        # and left no_cache-family arms with no cache telemetry at all. When the usage
        # object itself is present (prompt_tokens parsed), an absent details block means
        # cached == 0, not missing. None is kept only when usage is missing entirely.
        if cached_out is None and prompt_tokens_out is not None:
            cached_out = 0

        return (
            prompt_tokens_out,
            cached_out,
            int(completion_tokens) if isinstance(completion_tokens, (int, float)) else None,
        )

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

    def _stream_completion(self, request: InferenceRequest) -> InferenceResponse:
        """Stream a completion to measure TTFT and optionally collect usage telemetry."""
        start_time = time.time()
        first_token_time: Optional[float] = None
        full_text_parts: list[str] = []
        finish_reason = "length"

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
                kv_transfer_params = self._extract_header_kv_transfer_params(resp.headers)

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

                    # Optional KV transfer metadata (used by vLLM P/D connectors).
                    if isinstance(obj, dict):
                        kv_params = obj.get("kv_transfer_params")
                        if isinstance(kv_params, dict):
                            kv_transfer_params = kv_params

                    # Final usage chunk (choices may be empty)
                    if isinstance(obj, dict) and "usage" in obj and isinstance(obj["usage"], dict):
                        prompt_tokens, cached_prompt_tokens, completion_tokens = self._extract_usage(
                            obj["usage"]
                        )
                        continue

                    choices = obj.get("choices") if isinstance(obj, dict) else None
                    if not choices:
                        continue

                    choice = choices[0] if isinstance(choices, list) else {}
                    text_delta = choice.get("text", "") if isinstance(choice, dict) else ""
                    if text_delta:
                        full_text_parts.append(text_delta)
                        if first_token_time is None:
                            first_token_time = time.time()

                    finish_reason = (
                        choice.get("finish_reason", finish_reason)
                        if isinstance(choice, dict)
                        else finish_reason
                    )
        except (requests.exceptions.RequestException, ValueError) as e:
            # ValueError covers a malformed/truncated streamed chunk (json parse) so a bad
            # response becomes a recorded error row, not a run-ending crash.
            total_time = (time.time() - start_time) * 1000
            return InferenceResponse(
                request_id=request.request_id,
                generated_text="",
                ttft_ms=0.0,
                total_time_ms=total_time,
                num_tokens=0,
                model_name=self.model_name,
                finish_reason="error",
                error=str(e),
            )

        total_time_ms = (time.time() - start_time) * 1000
        ttft_ms = ((first_token_time - start_time) * 1000) if first_token_time else total_time_ms
        generated_text = "".join(full_text_parts)

        num_tokens = completion_tokens if isinstance(completion_tokens, int) else len(generated_text.split())

        return InferenceResponse(
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
        full_text_parts: list[str] = []
        finish_reason = "length"
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
                kv_transfer_params = self._extract_header_kv_transfer_params(resp.headers)

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

                    if isinstance(obj, dict):
                        kv_params = obj.get("kv_transfer_params")
                        if isinstance(kv_params, dict):
                            kv_transfer_params = kv_params

                    # Final usage chunk (choices may be empty)
                    if isinstance(obj, dict) and "usage" in obj and isinstance(obj["usage"], dict):
                        prompt_tokens, cached_prompt_tokens, completion_tokens = self._extract_usage(
                            obj["usage"]
                        )
                        continue

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
            return self._attach_logprob_stats(
                InferenceResponse(
                    request_id=request.request_id,
                    generated_text="",
                    ttft_ms=0.0,
                    total_time_ms=total_time,
                    num_tokens=0,
                    model_name=self.model_name,
                    finish_reason="error",
                    error=str(e),
                ),
                [],
            )

        total_time_ms = (time.time() - start_time) * 1000
        ttft_ms = ((first_token_time - start_time) * 1000) if first_token_time else total_time_ms
        generated_text = "".join(full_text_parts)

        num_tokens = completion_tokens if isinstance(completion_tokens, int) else len(generated_text.split())

        return self._attach_logprob_stats(
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
            finish_reason = choice.get("finish_reason") or "length"
            token_logprobs = self._extract_chat_logprobs(choice)

            usage = result.get("usage") or {}
            prompt_tokens, cached_prompt_tokens, completion_tokens = self._extract_usage(
                usage if isinstance(usage, dict) else {}
            )

            kv_transfer_params = result.get("kv_transfer_params")
            if kv_transfer_params is not None and not isinstance(kv_transfer_params, dict):
                kv_transfer_params = None
            if kv_transfer_params is None:
                kv_transfer_params = self._extract_header_kv_transfer_params(resp.headers)

            num_tokens = (
                completion_tokens
                if isinstance(completion_tokens, int)
                else len(generated_text.split())
            )

            return self._attach_logprob_stats(
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
            )

        except (requests.exceptions.RequestException, ValueError) as e:
            total_time_ms = (time.time() - start_time) * 1000
            return self._attach_logprob_stats(
                InferenceResponse(
                    request_id=request.request_id,
                    generated_text="",
                    ttft_ms=0.0,
                    total_time_ms=total_time_ms,
                    num_tokens=0,
                    model_name=self.model_name,
                    finish_reason="error",
                    error=str(e),
                ),
                [],
            )

    def generate(self, request: InferenceRequest, *, stream: bool = False) -> InferenceResponse:
        """Generate a completion via the vLLM OpenAI-compatible server.

        Requests carrying ``messages`` (Decision 1B chat mode) are served via
        /v1/chat/completions; all others use the legacy raw /v1/completions
        path unchanged (CAGE_PROMPT_MODE=raw escape hatch).
        """
        messages = self._request_messages(request)
        if messages is not None:
            if stream:
                return self._stream_chat_completion(request, messages)
            return self._chat_completion(request, messages)

        if stream:
            return self._stream_completion(request)

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
            finish_reason = choice.get("finish_reason", "length")

            usage = result.get("usage") or {}
            prompt_tokens, cached_prompt_tokens, completion_tokens = self._extract_usage(
                usage if isinstance(usage, dict) else {}
            )

            # Optional KV transfer metadata (used by vLLM P/D connectors).
            kv_transfer_params = result.get("kv_transfer_params")
            if kv_transfer_params is not None and not isinstance(kv_transfer_params, dict):
                kv_transfer_params = None
            if kv_transfer_params is None:
                kv_transfer_params = self._extract_header_kv_transfer_params(resp.headers)

            num_tokens = (
                completion_tokens
                if isinstance(completion_tokens, int)
                else len(generated_text.split())
            )

            # Non-streaming: TTFT is unobservable (full response arrives at once), so report
            # it as the full response time rather than a fabricated fraction. Use stream=True
            # for a real TTFT measurement.
            ttft_ms = total_time_ms

            return InferenceResponse(
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
            )

        except (requests.exceptions.RequestException, ValueError) as e:
            # ValueError covers an HTTP-200 truncated/malformed body (resp.json() raises
            # json.JSONDecodeError, a ValueError) so it becomes a recorded error row rather
            # than propagating out of the unguarded measured loop and aborting the baseline.
            total_time_ms = (time.time() - start_time) * 1000
            return InferenceResponse(
                request_id=request.request_id,
                generated_text="",
                ttft_ms=0.0,
                total_time_ms=total_time_ms,
                num_tokens=0,
                model_name=self.model_name,
                finish_reason="error",
                error=str(e),
            )

    def batch_generate(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Generate responses for batch of requests (sequential for now)."""
        # vLLM handles batching internally, so we can send requests sequentially
        # For async batching, use async_batch_generate instead
        return [self.generate(req) for req in requests]
    
    async def async_generate(self, request: InferenceRequest) -> InferenceResponse:
        """Async (non-streaming) completion request to vLLM."""
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
                    finish_reason = choice.get("finish_reason", "length")

                    usage = result.get("usage") or {}
                    prompt_tokens, cached_prompt_tokens, completion_tokens = self._extract_usage(
                        usage if isinstance(usage, dict) else {}
                    )

                    kv_transfer_params = result.get("kv_transfer_params")
                    if kv_transfer_params is not None and not isinstance(kv_transfer_params, dict):
                        kv_transfer_params = None
                    if kv_transfer_params is None:
                        kv_transfer_params = self._extract_header_kv_transfer_params(resp.headers)

                    num_tokens = (
                        completion_tokens
                        if isinstance(completion_tokens, int)
                        else len(generated_text.split())
                    )
                    # Non-streaming: TTFT unobservable -> report full response time.
                    ttft_ms = total_time_ms

                    return InferenceResponse(
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
                    )

        except Exception as e:
            total_time_ms = (time.time() - start_time) * 1000
            return InferenceResponse(
                request_id=request.request_id,
                generated_text="",
                ttft_ms=0.0,
                total_time_ms=total_time_ms,
                num_tokens=0,
                model_name=self.model_name,
                finish_reason="error",
                error=str(e),
            )
    
    async def async_batch_generate(
        self, requests: List[InferenceRequest]
    ) -> List[InferenceResponse]:
        """Async batch generation (concurrent requests)."""
        tasks = [self.async_generate(req) for req in requests]
        return await asyncio.gather(*tasks)
    
    def is_ready(self) -> bool:
        """Check if vLLM server is ready and serving the expected model."""
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
    
    def get_loaded_model(self) -> str | None:
        """Get the model currently loaded on the vLLM server."""
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
