"""Tests for the ADR-0007 engine-adapter generalization.

Covers:
- OpenAIChatAdapter shared machinery through VLLMAdapter (byte-identical
  vLLM semantics: absent-cached-block-means-zero, pinned chat extras,
  kv_transfer_params, truncate support);
- SGLangAdapter (None-with-provenance telemetry, /flush_cache, fail-closed
  truncate, no NIXL parsing);
- LMDeployAdapter (weakest-telemetry honesty, fail-closed flush);
- retry semantics (max_retries re-issues transport-error attempts);
- HFOracleAdapter (fail-closed lazy ML-stack import, T=0 greedy enforcement,
  DynamicCache corpus-prefix reuse with crop-after-every-query per
  Chan et al. 2024, arXiv:2412.15605);
- capabilities() declarations driving the charter-D2 telemetry-parity gate.

ALL transport is mocked (no live GPU/server/network), following the
tests/test_inference.py monkeypatch pattern.
"""

from __future__ import annotations

import contextlib
import json
import types
from typing import Any, Dict, List, Optional

import pytest
import requests

import src.inference.openai_chat_adapter as base_mod
import src.inference.hf_oracle_adapter as hf_mod
from src.inference.engine import DummyEngine, InferenceRequest, InferenceResponse
from src.inference.errors import (
    EngineCapabilityUnavailableError,
    EngineDependencyUnavailableError,
)
from src.inference.hf_oracle_adapter import HFOracleAdapter
from src.inference.lmdeploy_adapter import LMDeployAdapter
from src.inference.openai_chat_adapter import OpenAIChatAdapter
from src.inference.sglang_adapter import SGLangAdapter
from src.inference.vllm_adapter import VLLMAdapter


# ------------------------------------------------------------------------- #
# HTTP transport fakes (SSE streaming + JSON)
# ------------------------------------------------------------------------- #


def _sse(obj: Dict[str, Any]) -> str:
    return "data: " + json.dumps(obj)


def chat_stream_lines(
    deltas: List[str],
    usage: Optional[Dict[str, Any]] = None,
    kv: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """OpenAI-compatible chat SSE stream: content deltas, a finish chunk,
    an optional final usage chunk (choices empty), then [DONE]."""
    lines = [
        _sse({"choices": [{"delta": {"content": d}, "finish_reason": None}]})
        for d in deltas
    ]
    lines.append(_sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
    if usage is not None:
        chunk: Dict[str, Any] = {"choices": [], "usage": usage}
        if kv is not None:
            chunk["kv_transfer_params"] = kv
        lines.append(_sse(chunk))
    lines.append("data: [DONE]")
    return lines


class FakeStreamResponse:
    """Context-manager response mimicking requests.post(stream=True)."""

    def __init__(self, lines: List[str], headers: Optional[Dict[str, str]] = None):
        self._lines = list(lines)
        self.headers = headers or {}
        self.status_code = 200

    def __enter__(self) -> "FakeStreamResponse":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def raise_for_status(self) -> None:
        pass

    def iter_lines(self, decode_unicode: bool = True):
        return iter(self._lines)


class FakeJSONResponse:
    """Non-streaming JSON response."""

    def __init__(self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None):
        self._payload = payload
        self.headers = headers or {}
        self.status_code = 200

    def raise_for_status(self) -> None:
        pass

    def json(self) -> Dict[str, Any]:
        return self._payload


def install_post(monkeypatch: pytest.MonkeyPatch, factory) -> List[Dict[str, Any]]:
    """Patch requests.post (as resolved by the adapter modules); record calls."""
    calls: List[Dict[str, Any]] = []

    def fake_post(url: str, json=None, timeout=None, stream=False, **kw):
        calls.append({"url": url, "json": json, "timeout": timeout, "stream": stream})
        return factory(calls)

    monkeypatch.setattr(base_mod.requests, "post", fake_post)
    return calls


def chat_request(**overrides: Any) -> InferenceRequest:
    req = InferenceRequest(
        prompt="ignored in chat mode",
        max_tokens=overrides.pop("max_tokens", 64),
        temperature=overrides.pop("temperature", 0.0),
        request_id=overrides.pop("request_id", "r1"),
        **overrides,
    )
    req.messages = [{"role": "user", "content": "hi"}]
    return req


USAGE_WITH_CACHED = {
    "prompt_tokens": 100,
    "completion_tokens": 2,
    "prompt_tokens_details": {"cached_tokens": 7},
}
USAGE_NO_DETAILS = {"prompt_tokens": 100, "completion_tokens": 2}


# ------------------------------------------------------------------------- #
# VLLMAdapter (via the extracted base): byte-identical vLLM semantics
# ------------------------------------------------------------------------- #


def test_vllm_stream_chat_happy_path(monkeypatch):
    lines = chat_stream_lines(["Hello", " world"], usage=USAGE_WITH_CACHED)
    calls = install_post(monkeypatch, lambda _c: FakeStreamResponse(lines))

    adapter = VLLMAdapter(model_name="m")
    resp = adapter.generate(chat_request(), stream=True)

    assert resp.error is None
    assert resp.generated_text == "Hello world"
    assert resp.finish_reason == "stop"
    assert resp.prompt_tokens == 100
    assert resp.cached_prompt_tokens == 7
    assert resp.num_tokens == 2
    assert 0.0 <= resp.ttft_ms <= resp.total_time_ms
    assert resp.engine_id == "vllm"
    assert resp.usage_telemetry_available is True
    assert resp.cached_token_telemetry_available is True
    assert resp.retries == 0
    assert calls[0]["url"].endswith("/v1/chat/completions")


def test_vllm_chat_payload_pins_are_unchanged(monkeypatch):
    """The refactor must emit the exact pre-refactor payload (keys AND order)."""
    lines = chat_stream_lines(["x"], usage=USAGE_WITH_CACHED)
    calls = install_post(monkeypatch, lambda _c: FakeStreamResponse(lines))

    VLLMAdapter(model_name="m").generate(chat_request(), stream=True)

    payload = calls[0]["json"]
    assert list(payload.keys()) == [
        "model", "messages", "max_tokens", "temperature", "top_p", "stream",
        "chat_template_kwargs", "logprobs", "top_logprobs", "stream_options",
    ]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 0
    assert payload["stream_options"] == {"include_usage": True}


def test_vllm_absent_cached_block_means_zero(monkeypatch):
    """Audit 2026-07-16 M6 pin: usage present but no prompt_tokens_details
    -> cached == 0 (vLLM-verified semantic), telemetry flagged available."""
    lines = chat_stream_lines(["Hello"], usage=USAGE_NO_DETAILS)
    install_post(monkeypatch, lambda _c: FakeStreamResponse(lines))

    resp = VLLMAdapter(model_name="m").generate(chat_request(), stream=True)

    assert resp.cached_prompt_tokens == 0
    assert resp.cached_token_telemetry_available is True


def test_vllm_usage_entirely_missing_stays_none(monkeypatch):
    """No usage chunk at all -> None telemetry, flags False, word-count fallback."""
    lines = chat_stream_lines(["Hello", " world"], usage=None)
    install_post(monkeypatch, lambda _c: FakeStreamResponse(lines))

    resp = VLLMAdapter(model_name="m").generate(chat_request(), stream=True)

    assert resp.prompt_tokens is None
    assert resp.cached_prompt_tokens is None
    assert resp.num_tokens == 2  # len("Hello world".split()) fallback
    assert resp.usage_telemetry_available is False
    assert resp.cached_token_telemetry_available is False


def test_vllm_kv_transfer_params_parsed(monkeypatch):
    kv = {"remote_block_ids": [1, 2]}
    lines = chat_stream_lines(["x"], usage=USAGE_WITH_CACHED, kv=kv)
    install_post(monkeypatch, lambda _c: FakeStreamResponse(lines))

    resp = VLLMAdapter(model_name="m").generate(chat_request(), stream=True)
    assert resp.kv_transfer_params == kv


def test_vllm_header_kv_transfer_params_parsed(monkeypatch):
    kv = {"transfer": "simulated"}
    lines = chat_stream_lines(["x"], usage=USAGE_WITH_CACHED)
    headers = {"x-kv-transfer-params": json.dumps(kv)}
    install_post(monkeypatch, lambda _c: FakeStreamResponse(lines, headers=headers))

    resp = VLLMAdapter(model_name="m").generate(chat_request(), stream=True)
    assert resp.kv_transfer_params == kv


def test_vllm_truncate_prompt_tokens_supported(monkeypatch):
    lines = chat_stream_lines(["x"], usage=USAGE_WITH_CACHED)
    calls = install_post(monkeypatch, lambda _c: FakeStreamResponse(lines))

    req = chat_request(truncate_prompt_tokens=128)
    VLLMAdapter(model_name="m").generate(req, stream=True)
    assert calls[0]["json"]["truncate_prompt_tokens"] == 128


def test_vllm_non_stream_chat_ttft_is_full_response_time(monkeypatch):
    payload = {
        "choices": [{"message": {"content": "Paris"}, "finish_reason": "stop"}],
        "usage": USAGE_WITH_CACHED,
    }
    install_post(monkeypatch, lambda _c: FakeJSONResponse(payload))

    resp = VLLMAdapter(model_name="m").generate(chat_request(), stream=False)

    assert resp.generated_text == "Paris"
    assert resp.ttft_ms == pytest.approx(resp.total_time_ms)
    assert resp.cached_prompt_tokens == 7


def test_vllm_is_subclass_of_shared_base():
    assert issubclass(VLLMAdapter, OpenAIChatAdapter)


# ------------------------------------------------------------------------- #
# async_stream_generate (the D6 open-loop streaming path: real TTFT)
# ------------------------------------------------------------------------- #


class _FakeAsyncLines:
    """aiohttp StreamReader stand-in: async-iterates SSE lines as bytes."""

    def __init__(self, lines: List[str]) -> None:
        self._raw = [(line + "\n").encode("utf-8") for line in lines]

    def __aiter__(self):
        async def gen():
            for raw in self._raw:
                yield raw

        return gen()


class FakeAsyncStreamResponse:
    """Async context-manager response mimicking aiohttp session.post."""

    def __init__(self, lines: List[str], headers: Optional[Dict[str, str]] = None):
        self.content = _FakeAsyncLines(lines)
        self.headers = headers or {}

    async def __aenter__(self) -> "FakeAsyncStreamResponse":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def raise_for_status(self) -> None:
        pass


class FakeAsyncSession:
    def __init__(self, response: Any, calls: List[Dict[str, Any]]) -> None:
        self._response = response
        self._calls = calls

    async def __aenter__(self) -> "FakeAsyncSession":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def post(self, url: str, json=None, timeout=None, **kw):
        self._calls.append({"url": url, "json": json, "timeout": timeout})
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def install_async_session(
    monkeypatch: pytest.MonkeyPatch, response: Any
) -> List[Dict[str, Any]]:
    """Patch aiohttp.ClientSession (as resolved by the adapter module)."""
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        base_mod.aiohttp, "ClientSession", lambda: FakeAsyncSession(response, calls)
    )
    return calls


def _run_async(coro):
    import asyncio

    return asyncio.run(coro)


def test_async_stream_generate_chat_real_ttft(monkeypatch):
    lines = chat_stream_lines(["Hello", " world"], usage=USAGE_WITH_CACHED)
    calls = install_async_session(monkeypatch, FakeAsyncStreamResponse(lines))

    hook_calls: List[int] = []
    adapter = VLLMAdapter(model_name="m")
    resp = _run_async(
        adapter.async_stream_generate(
            chat_request(), on_first_token=lambda: hook_calls.append(1)
        )
    )

    assert resp.error is None
    assert resp.generated_text == "Hello world"
    assert resp.finish_reason == "stop"
    assert resp.prompt_tokens == 100
    assert resp.cached_prompt_tokens == 7
    assert resp.num_tokens == 2
    # Real streamed TTFT by our clock -- never the full-response proxy.
    assert 0.0 <= resp.ttft_ms <= resp.total_time_ms
    assert resp.ttft_methodology == "streamed-first-delta"
    # The first-token hook fired exactly once (at the first non-empty delta).
    assert hook_calls == [1]
    # Chat-mode requests stream via /v1/chat/completions with stream=True.
    assert calls[0]["url"].endswith("/v1/chat/completions")
    assert calls[0]["json"]["stream"] is True


def test_async_stream_generate_raw_path(monkeypatch):
    lines = [
        _sse({"choices": [{"text": "Par", "finish_reason": None}]}),
        _sse({"choices": [{"text": "is", "finish_reason": "stop"}]}),
        _sse({"choices": [], "usage": USAGE_NO_DETAILS}),
        "data: [DONE]",
    ]
    calls = install_async_session(monkeypatch, FakeAsyncStreamResponse(lines))

    req = InferenceRequest(
        prompt="capital?", max_tokens=8, temperature=0.0, request_id="r-raw"
    )
    resp = _run_async(VLLMAdapter(model_name="m").async_stream_generate(req))

    assert resp.generated_text == "Paris"
    assert resp.finish_reason == "stop"
    assert resp.prompt_tokens == 100
    assert resp.ttft_methodology == "streamed-first-delta"
    assert calls[0]["url"].endswith("/v1/completions")
    assert calls[0]["json"]["stream"] is True


def test_async_stream_generate_transport_error_is_recorded_row(monkeypatch):
    install_async_session(monkeypatch, RuntimeError("connection refused"))

    resp = _run_async(
        VLLMAdapter(model_name="m").async_stream_generate(chat_request())
    )

    assert resp.finish_reason == "error"
    assert "connection refused" in (resp.error or "")
    assert resp.generated_text == ""
    assert resp.ttft_ms == 0.0


def test_async_generate_proxy_ttft_is_labeled(monkeypatch):
    """The non-streaming async path stays available but its ttft_ms is now
    explicitly labeled as the full-response proxy so it can never silently
    pose as a streamed first-token time."""

    class FakeAsyncJSONResponse:
        def __init__(self, payload: Dict[str, Any]) -> None:
            self._payload = payload
            self.headers: Dict[str, str] = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

        def raise_for_status(self) -> None:
            pass

        async def json(self) -> Dict[str, Any]:
            return self._payload

    payload = {
        "choices": [{"text": "Paris", "finish_reason": "stop"}],
        "usage": USAGE_NO_DETAILS,
    }
    install_async_session(monkeypatch, FakeAsyncJSONResponse(payload))

    req = InferenceRequest(
        prompt="capital?", max_tokens=8, temperature=0.0, request_id="r-nb"
    )
    resp = _run_async(VLLMAdapter(model_name="m").async_generate(req))

    assert resp.generated_text == "Paris"
    assert resp.ttft_ms == pytest.approx(resp.total_time_ms)
    assert resp.ttft_methodology == "full-response-proxy"


# ------------------------------------------------------------------------- #
# SGLangAdapter
# ------------------------------------------------------------------------- #


def test_sglang_stream_chat_happy_path(monkeypatch):
    lines = chat_stream_lines(["Radix", " tree"], usage=USAGE_WITH_CACHED)
    calls = install_post(monkeypatch, lambda _c: FakeStreamResponse(lines))

    adapter = SGLangAdapter(model_name="m", api_base="http://sgl:1234")
    resp = adapter.generate(chat_request(), stream=True)

    assert resp.error is None
    assert resp.generated_text == "Radix tree"
    assert resp.prompt_tokens == 100
    assert resp.cached_prompt_tokens == 7  # parsed when the server provides it
    assert resp.engine_id == "sglang"
    assert calls[0]["url"] == "http://sgl:1234/v1/chat/completions"


def test_sglang_payload_sends_nothing_unverified_by_default(monkeypatch):
    lines = chat_stream_lines(["x"], usage=USAGE_NO_DETAILS)
    calls = install_post(monkeypatch, lambda _c: FakeStreamResponse(lines))

    SGLangAdapter(model_name="m").generate(chat_request(), stream=True)

    payload = calls[0]["json"]
    assert "chat_template_kwargs" not in payload
    assert "logprobs" not in payload
    assert "top_logprobs" not in payload


def test_sglang_explicit_chat_template_kwargs_sent(monkeypatch):
    lines = chat_stream_lines(["x"], usage=USAGE_NO_DETAILS)
    calls = install_post(monkeypatch, lambda _c: FakeStreamResponse(lines))

    adapter = SGLangAdapter(
        model_name="m",
        chat_template_kwargs={"enable_thinking": False},
        request_logprobs=True,
    )
    adapter.generate(chat_request(), stream=True)

    payload = calls[0]["json"]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 0


def test_sglang_absent_cached_tokens_stays_none_never_zero(monkeypatch):
    """The vLLM absent-means-zero coercion must NOT be inherited: SGLang's
    absence semantics are VERIFY-LIVE, so absence is None-with-provenance."""
    lines = chat_stream_lines(["Hello"], usage=USAGE_NO_DETAILS)
    install_post(monkeypatch, lambda _c: FakeStreamResponse(lines))

    resp = SGLangAdapter(model_name="m").generate(chat_request(), stream=True)

    assert resp.prompt_tokens == 100
    assert resp.cached_prompt_tokens is None  # NOT 0
    assert resp.usage_telemetry_available is True
    assert resp.cached_token_telemetry_available is False


def test_sglang_ignores_nixl_kv_transfer_params(monkeypatch):
    lines = chat_stream_lines(
        ["x"], usage=USAGE_WITH_CACHED, kv={"remote_block_ids": [1]}
    )
    headers = {"x-kv-transfer-params": json.dumps({"transfer": "sim"})}
    install_post(monkeypatch, lambda _c: FakeStreamResponse(lines, headers=headers))

    resp = SGLangAdapter(model_name="m").generate(chat_request(), stream=True)
    assert resp.kv_transfer_params is None  # vLLM/NIXL-shaped; not parsed (ADR-0007)


def test_sglang_flush_cache_posts_native_endpoint(monkeypatch):
    calls = install_post(monkeypatch, lambda _c: FakeJSONResponse({}))

    SGLangAdapter(model_name="m", api_base="http://sgl:1234").flush_cache()
    assert calls[0]["url"] == "http://sgl:1234/flush_cache"


def test_sglang_flush_cache_failure_raises_typed_error(monkeypatch):
    def fake_post(url, json=None, timeout=None, stream=False, **kw):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(base_mod.requests, "post", fake_post)

    with pytest.raises(EngineCapabilityUnavailableError) as exc:
        SGLangAdapter(model_name="m").flush_cache()
    assert exc.value.capability == "flush_endpoint"


def test_sglang_truncate_prompt_tokens_fails_closed(monkeypatch):
    install_post(monkeypatch, lambda _c: FakeStreamResponse([]))

    req = chat_request(truncate_prompt_tokens=128)
    with pytest.raises(EngineCapabilityUnavailableError) as exc:
        SGLangAdapter(model_name="m").generate(req, stream=True)
    assert exc.value.capability == "truncate_prompt_tokens"


# ------------------------------------------------------------------------- #
# LMDeployAdapter
# ------------------------------------------------------------------------- #


def test_lmdeploy_stream_chat_happy_path(monkeypatch):
    lines = chat_stream_lines(["Turbo", "Mind"], usage=USAGE_NO_DETAILS)
    calls = install_post(monkeypatch, lambda _c: FakeStreamResponse(lines))

    adapter = LMDeployAdapter(model_name="m", api_base="http://lmd:23333")
    resp = adapter.generate(chat_request(), stream=True)

    assert resp.error is None
    assert resp.generated_text == "TurboMind"
    assert resp.prompt_tokens == 100
    assert resp.num_tokens == 2
    assert resp.engine_id == "lmdeploy-turbomind"
    assert calls[0]["url"] == "http://lmd:23333/v1/chat/completions"


def test_lmdeploy_absent_cached_tokens_stays_none(monkeypatch):
    """Weakest documented telemetry (charter D2.1): absence is None + flag,
    never a fabricated number."""
    lines = chat_stream_lines(["Hello"], usage=USAGE_NO_DETAILS)
    install_post(monkeypatch, lambda _c: FakeStreamResponse(lines))

    resp = LMDeployAdapter(model_name="m").generate(chat_request(), stream=True)

    assert resp.cached_prompt_tokens is None
    assert resp.cached_token_telemetry_available is False


def test_lmdeploy_flush_cache_fails_closed():
    with pytest.raises(EngineCapabilityUnavailableError) as exc:
        LMDeployAdapter(model_name="m").flush_cache()
    assert exc.value.capability == "flush_endpoint"
    assert exc.value.engine == "lmdeploy-turbomind"


def test_lmdeploy_truncate_prompt_tokens_fails_closed_raw_path():
    req = InferenceRequest(prompt="p", truncate_prompt_tokens=64, request_id="r1")
    with pytest.raises(EngineCapabilityUnavailableError):
        LMDeployAdapter(model_name="m").generate(req, stream=False)


# ------------------------------------------------------------------------- #
# Retry semantics (shared base)
# ------------------------------------------------------------------------- #


@pytest.mark.parametrize("adapter_cls", [VLLMAdapter, SGLangAdapter, LMDeployAdapter])
def test_retry_reissues_transport_errors_then_succeeds(monkeypatch, adapter_cls):
    good_lines = chat_stream_lines(["ok"], usage=USAGE_NO_DETAILS)
    n_calls = {"count": 0}

    def fake_post(url, json=None, timeout=None, stream=False, **kw):
        n_calls["count"] += 1
        if n_calls["count"] <= 2:
            raise requests.exceptions.ConnectionError("transient")
        return FakeStreamResponse(good_lines)

    monkeypatch.setattr(base_mod.requests, "post", fake_post)

    adapter = adapter_cls(model_name="m", max_retries=2, retry_backoff_s=0.0)
    resp = adapter.generate(chat_request(), stream=True)

    assert n_calls["count"] == 3
    assert resp.error is None
    assert resp.generated_text == "ok"
    assert resp.retries == 2


def test_default_zero_retries_preserves_single_attempt_error_row(monkeypatch):
    n_calls = {"count": 0}

    def fake_post(url, json=None, timeout=None, stream=False, **kw):
        n_calls["count"] += 1
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(base_mod.requests, "post", fake_post)

    resp = VLLMAdapter(model_name="m").generate(chat_request(), stream=True)

    assert n_calls["count"] == 1  # historical single-attempt behavior
    assert resp.finish_reason == "error"
    assert resp.error is not None
    assert resp.retries == 0


def test_retries_exhausted_returns_error_row(monkeypatch):
    n_calls = {"count": 0}

    def fake_post(url, json=None, timeout=None, stream=False, **kw):
        n_calls["count"] += 1
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(base_mod.requests, "post", fake_post)

    adapter = SGLangAdapter(model_name="m", max_retries=2, retry_backoff_s=0.0)
    resp = adapter.generate(chat_request(), stream=True)

    assert n_calls["count"] == 3
    assert resp.finish_reason == "error"
    assert resp.retries == 2


# ------------------------------------------------------------------------- #
# capabilities(): the data the D2 telemetry-parity preflight gate consumes
# ------------------------------------------------------------------------- #


def test_capabilities_declarations():
    vllm = VLLMAdapter(model_name="m").capabilities()
    assert vllm["engine"] == "vllm"
    assert vllm["streamed_ttft"] is True
    assert vllm["cached_token_telemetry"] is True
    assert vllm["cached_token_absent_means_zero"] is True
    assert vllm["flush_endpoint"] == "/reset_prefix_cache"
    assert vllm["kv_transfer_params"] is True
    assert vllm["truncate_prompt_tokens"] is True

    sgl = SGLangAdapter(model_name="m").capabilities()
    assert sgl["engine"] == "sglang"
    assert sgl["cached_token_telemetry"] == "verify-live"
    assert sgl["cached_token_absent_means_zero"] is False
    assert sgl["flush_endpoint"] == "/flush_cache"
    assert sgl["kv_transfer_params"] is False
    assert sgl["truncate_prompt_tokens"] is False

    lmd = LMDeployAdapter(model_name="m").capabilities()
    assert lmd["engine"] == "lmdeploy-turbomind"
    assert lmd["cached_token_telemetry"] == "verify-live"
    assert lmd["flush_endpoint"] is None
    assert lmd["kv_transfer_params"] is False


def test_capabilities_conservative_default_on_base_engine():
    caps = DummyEngine().capabilities()
    assert caps["streamed_ttft"] is False
    assert caps["cached_token_telemetry"] is False
    assert caps["flush_endpoint"] is None


# ------------------------------------------------------------------------- #
# HFOracleAdapter fakes (no torch/transformers in tests)
# ------------------------------------------------------------------------- #


class _FakeTensor:
    def __init__(self, ids: List[int]):
        self.ids = list(ids)

    @property
    def shape(self):
        return (1, len(self.ids))

    def __getitem__(self, key):
        row, sl = key
        assert row == 0
        return _FakeTensor(self.ids[sl])


class _FakeEncoding(dict):
    def __init__(self, ids: List[int]):
        super().__init__(
            input_ids=_FakeTensor(ids),
            attention_mask=_FakeTensor([1] * len(ids)),
        )

    @property
    def input_ids(self) -> _FakeTensor:
        return self["input_ids"]

    @property
    def attention_mask(self) -> _FakeTensor:
        return self["attention_mask"]

    def to(self, device):
        return self


class _FakeTokenizer:
    """Whitespace tokenizer: one word == one token id (memoized vocab)."""

    pad_token_id = None
    eos_token_id = 0

    def __init__(self):
        self._word2id: Dict[str, int] = {}
        self._id2word: Dict[int, str] = {}

    def encode_words(self, text: str) -> List[int]:
        ids = []
        for w in text.split():
            wid = self._word2id.setdefault(w, 10 + len(self._word2id))
            self._id2word[wid] = w
            ids.append(wid)
        return ids

    def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = True):
        return _FakeEncoding(self.encode_words(text))

    def decode(self, tensor: _FakeTensor, skip_special_tokens: bool = True) -> str:
        return " ".join(self._id2word[i] for i in tensor.ids)


class _FakeCache:
    """Stands in for transformers.DynamicCache."""

    def __init__(self):
        self._len = 0
        self.crop_calls: List[int] = []

    def get_seq_length(self) -> int:
        return self._len

    def crop(self, n: int) -> None:
        self.crop_calls.append(n)
        self._len = n


class _FakeModel:
    def __init__(self):
        self.device = types.SimpleNamespace(type="cpu")
        self.generated_ids: List[int] = []
        self.generate_calls: List[Dict[str, Any]] = []
        self.raise_on_generate: Optional[Exception] = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None, past_key_values=None, use_cache=False):
        # Prefill: the cache absorbs the prefix tokens.
        if past_key_values is not None:
            past_key_values._len = len(input_ids.ids)

    def generate(self, input_ids=None, attention_mask=None, past_key_values=None,
                 max_new_tokens=None, do_sample=None, pad_token_id=None):
        self.generate_calls.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
        })
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        assert do_sample is False, "oracle must decode greedily"
        return _FakeTensor(input_ids.ids + self.generated_ids)


def _fake_torch():
    return types.SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        float32="fp32",
        long="long",
        no_grad=lambda: contextlib.nullcontext(),
        ones=lambda shape, dtype=None, device=None: _FakeTensor([1] * int(shape[1])),
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            synchronize=lambda: None,
            empty_cache=lambda: None,
        ),
    )


def _install_fake_stack(monkeypatch: pytest.MonkeyPatch, model: _FakeModel, tok: _FakeTokenizer):
    def _stack():
        return (
            _fake_torch(),
            types.SimpleNamespace(from_pretrained=lambda name, torch_dtype=None: model),
            types.SimpleNamespace(from_pretrained=lambda name: tok),
            _FakeCache,
        )

    monkeypatch.setattr(hf_mod, "_import_ml_stack", _stack)


def _oracle(monkeypatch: pytest.MonkeyPatch, generated_text: str = "Paris"):
    tok = _FakeTokenizer()
    model = _FakeModel()
    model.generated_ids = tok.encode_words(generated_text)
    _install_fake_stack(monkeypatch, model, tok)
    return HFOracleAdapter(model_name="fake-model"), model, tok


# ------------------------------------------------------------------------- #
# HFOracleAdapter tests
# ------------------------------------------------------------------------- #


def test_hf_oracle_missing_ml_stack_fails_closed(monkeypatch):
    def _raise():
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(hf_mod, "_import_ml_stack", _raise)

    with pytest.raises(EngineDependencyUnavailableError) as exc:
        HFOracleAdapter(model_name="m")
    assert exc.value.engine == "hf_reference"
    assert exc.value.dependency == "torch+transformers"


def test_hf_oracle_greedy_happy_path_reference_labeled(monkeypatch):
    adapter, model, _tok = _oracle(monkeypatch, generated_text="Paris")
    req = InferenceRequest(
        prompt="What is the capital of France ?",
        temperature=0.0,
        max_tokens=8,
        request_id="q1",
    )

    resp = adapter.generate(req)

    assert resp.error is None
    assert resp.generated_text == "Paris"
    assert resp.num_tokens == 1
    assert resp.prompt_tokens == 7  # whitespace-token prompt length
    assert resp.cached_prompt_tokens == 0  # no corpus prefix loaded: exactly zero
    assert resp.finish_reason == "stop"  # 1 < max_tokens
    # Reference engine: no streaming, TTFT honestly equals the whole call.
    assert resp.ttft_ms == pytest.approx(resp.total_time_ms)
    assert resp.engine_id == "hf_reference"
    assert resp.reference_engine is True
    assert resp.corpus_prefill_ms is None
    assert model.generate_calls[0]["do_sample"] is False


def test_hf_oracle_corpus_prefix_reuse_and_crop(monkeypatch):
    """The Chan et al. 2024 recipe: prefill once, serve suffix against the
    cache, crop back to corpus length after EVERY query."""
    adapter, model, tok = _oracle(monkeypatch, generated_text="42")
    prefix = "SYSTEM instructions plus the corpus block text"
    n_prefix = len(prefix.split())

    prefill_ms = adapter.preload_corpus_prefix(prefix)
    assert prefill_ms >= 0.0

    suffix = " Question : what ? Answer :"
    req = InferenceRequest(
        prompt=prefix + suffix, temperature=0.0, max_tokens=8, request_id="q1"
    )
    resp = adapter.generate(req)

    assert resp.error is None
    assert resp.generated_text == "42"
    # Only the suffix is tokenized and fed after the cached corpus.
    assert resp.prompt_tokens == len(suffix.split())
    # Self-instrumented cache telemetry: exactly the resident corpus length.
    assert resp.cached_prompt_tokens == n_prefix
    assert resp.corpus_prefill_ms == pytest.approx(prefill_ms)
    # The generate call served against the SAME preloaded cache object...
    cache = model.generate_calls[0]["past_key_values"]
    assert isinstance(cache, _FakeCache)
    # ...and cropped back to the corpus length after the query (NON-OPTIONAL).
    assert cache.crop_calls == [n_prefix]


def test_hf_oracle_crop_happens_even_on_generation_error(monkeypatch):
    adapter, model, _tok = _oracle(monkeypatch)
    prefix = "corpus block"
    adapter.preload_corpus_prefix(prefix)
    model.raise_on_generate = RuntimeError("CUDA out of memory (simulated)")

    req = InferenceRequest(
        prompt=prefix + " Question", temperature=0.0, max_tokens=8, request_id="q1"
    )
    resp = adapter.generate(req)

    assert resp.finish_reason == "error"
    assert "CUDA out of memory" in (resp.error or "")
    cache = model.generate_calls[0]["past_key_values"]
    assert cache.crop_calls == [len(prefix.split())]  # record, crop, continue


def test_hf_oracle_mismatched_prefix_fails_closed(monkeypatch):
    adapter, _model, _tok = _oracle(monkeypatch)
    adapter.preload_corpus_prefix("the corpus block")

    req = InferenceRequest(
        prompt="a completely different prompt", temperature=0.0, request_id="q1"
    )
    with pytest.raises(ValueError, match="corpus prefix"):
        adapter.generate(req)


def test_hf_oracle_rejects_sampling_temperature(monkeypatch):
    adapter, _model, _tok = _oracle(monkeypatch)
    # InferenceRequest defaults to temperature=0.7 -- the oracle must refuse.
    req = InferenceRequest(prompt="hello there", request_id="q1")
    with pytest.raises(ValueError, match="T=0 greedy"):
        adapter.generate(req)


def test_hf_oracle_rejects_stop_sequences(monkeypatch):
    adapter, _model, _tok = _oracle(monkeypatch)
    req = InferenceRequest(
        prompt="hello", temperature=0.0, stop=["\n"], request_id="q1"
    )
    with pytest.raises(EngineCapabilityUnavailableError) as exc:
        adapter.generate(req)
    assert exc.value.capability == "stop_sequences"


def test_hf_oracle_batch_generate_is_sequential_batch_1(monkeypatch):
    adapter, model, _tok = _oracle(monkeypatch, generated_text="ok")
    reqs = [
        InferenceRequest(prompt=f"prompt {i}", temperature=0.0, request_id=f"r{i}")
        for i in range(3)
    ]
    responses = adapter.batch_generate(reqs)

    assert [r.request_id for r in responses] == ["r0", "r1", "r2"]
    assert len(model.generate_calls) == 3  # strictly one generate() per request


def test_hf_oracle_unknown_dtype_fails_closed(monkeypatch):
    tok = _FakeTokenizer()
    model = _FakeModel()
    _install_fake_stack(monkeypatch, model, tok)
    with pytest.raises(ValueError, match="dtype"):
        HFOracleAdapter(model_name="m", dtype="int4")


def test_hf_oracle_capabilities(monkeypatch):
    adapter, _model, _tok = _oracle(monkeypatch)
    caps = adapter.capabilities()
    assert caps["engine"] == "hf_reference"
    assert caps["serving_grade"] is False
    assert caps["streamed_ttft"] is False
    assert caps["cached_token_telemetry"] is True  # self-instrumented
    assert caps["in_process"] is True
    assert caps["corpus_prefix_reuse"] is True


def test_hf_oracle_clear_corpus_prefix_returns_to_prefixfree_serving(monkeypatch):
    adapter, model, _tok = _oracle(monkeypatch, generated_text="ok")
    adapter.preload_corpus_prefix("some corpus")
    adapter.clear_corpus_prefix()

    req = InferenceRequest(prompt="standalone prompt", temperature=0.0, request_id="q1")
    resp = adapter.generate(req)

    assert resp.error is None
    assert resp.cached_prompt_tokens == 0
    assert model.generate_calls[-1]["past_key_values"] is None
