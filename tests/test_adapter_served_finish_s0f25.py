"""S0F-25 (ADR-0135): a response counts as served only when the engine ended
it with ``stop`` or ``length``.

Before this, the adapter started every path from an invented
``finish_reason = "length"`` and skipped any stream chunk without ``choices``,
so three failures were recorded as served rows: an in-band error object
(dead engine, generate error, late validation), a stream that ended with no
``finish_reason`` at all (the vLLM 0.19.1 shape of a failed NIXL KV load: a
usage chunk and ``[DONE]``, no error event), and an error or a cut after
some tokens (a "successful" partial answer). A client timeout on the async
path produced an error row whose error text was empty
(``str(asyncio.TimeoutError()) == ""``), which every consumer reads as "no
error".

The real adapter is driven over a local stub HTTP server (requests and
aiohttp, no monkeypatching), on all six paths.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference import openai_chat_adapter as oca  # noqa: E402
from src.inference.engine import InferenceRequest, InferenceResponse  # noqa: E402
from src.inference.sglang_adapter import SGLangAdapter  # noqa: E402
from src.inference.vllm_adapter import VLLMAdapter  # noqa: E402

ADAPTER_PY = REPO_ROOT / "src" / "inference" / "openai_chat_adapter.py"

USAGE = {"prompt_tokens": 7, "completion_tokens": 2}

#: scenario -> the event list a stream replays. ("tok", text), ("finish",
#: reason), ("usage",), ("raw", obj) for a verbatim SSE object, ("done",).
#: A scenario that ends without ("done",) closes the stream with no [DONE].
STREAMS: Dict[str, List[Tuple[Any, ...]]] = {
    "ok_stop": [("tok", "A"), ("tok", "B"), ("finish", "stop"), ("usage",), ("done",)],
    "ok_length": [("tok", "A"), ("tok", "B"), ("finish", "length"), ("done",)],
    "ok_empty_stop": [("finish", "stop"), ("usage",), ("done",)],
    # vLLM 0.19.1 in-band error event shape (engine/serving.py create_streaming_error_response)
    "error_first": [
        ("raw", {"error": {"message": "EngineCore died", "type": "InternalServerError",
                           "param": None, "code": 500}}),
        ("done",),
    ],
    "error_after_tokens": [
        ("tok", "A"),
        ("raw", {"error": {"message": "late failure", "type": "BadRequestError",
                           "param": None, "code": 400}}),
        ("done",),
    ],
    # EngineGenerateError carries an empty message
    "error_empty_message": [
        ("raw", {"error": {"message": "", "type": "InternalServerError", "param": None, "code": 500}}),
        ("done",),
    ],
    # LMDeploy / older engines: the flat error object
    "error_flat": [
        ("raw", {"object": "error", "message": "flat failure", "type": "invalid_request_error", "code": 400}),
        ("done",),
    ],
    # the failed NIXL KV load shape: usage chunk and [DONE], no finish chunk
    "no_finish_usage_only": [("usage",), ("done",)],
    # a cut after tokens: no finish chunk, no [DONE]
    "cut_after_tokens": [("tok", "A"), ("tok", "B")],
    "finish_abort": [("tok", "A"), ("finish", "abort"), ("done",)],
    "finish_repetition": [("tok", "A"), ("finish", "repetition"), ("done",)],
    "finish_unknown": [("tok", "A"), ("finish", "content_filter"), ("done",)],
    # review 2026-10-02 MEDIUM: the terminal chunk carries usage AND the
    # finish_reason (the shape of an engine that folds its usage into the last
    # chunk), and continuous usage on every chunk (vLLM continuous_usage_stats)
    "usage_with_finish": [("tok", "A"), ("tok", "B"), ("finish_usage", "stop"), ("done",)],
    "continuous_usage": [("tok_usage", "A"), ("tok_usage", "B"), ("finish_usage", "stop"), ("done",)],
    # a finish_reason that is not a string (an object): unhashable, so a bare
    # set-membership test would raise instead of refusing the row
    "finish_object": [
        ("tok", "A"),
        ("raw", {"choices": [{"text": "", "delta": {"content": ""}, "finish_reason": {"type": "stop"}}]}),
        ("done",),
    ],
}

#: scenario -> the finish_reason of a non-streamed body ("__absent__" omits the key).
BODIES: Dict[str, Any] = {
    "body_stop": "stop",
    "body_length": "length",
    "body_null": None,
    "body_absent": "__absent__",
    "body_abort": "abort",
}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a: Any) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or "0")
        payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        scenario = self.path.strip("/").split("/", 1)[0]
        chat = self.path.endswith("/chat/completions")
        if scenario == "hang":
            time.sleep(3.0)
            scenario = "ok_stop"

        def choice(text: str, finish: Optional[str]) -> Dict[str, Any]:
            if chat:
                return {"choices": [{"delta": {"content": text}, "finish_reason": finish}]}
            return {"choices": [{"text": text, "finish_reason": finish}]}

        if not payload.get("stream"):
            finish = BODIES[scenario]
            entry: Dict[str, Any] = (
                {"message": {"content": "AB"}} if chat else {"text": "AB"}
            )
            if finish != "__absent__":
                entry["finish_reason"] = finish
            body = json.dumps({"choices": [entry], "usage": USAGE}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(data: str) -> None:
            raw = f"data: {data}\n\n".encode("utf-8")
            self.wfile.write(f"{len(raw):x}\r\n".encode() + raw + b"\r\n")
            self.wfile.flush()

        for event in STREAMS[scenario]:
            kind = event[0]
            if kind == "tok":
                send(json.dumps(choice(event[1], None)))
            elif kind == "finish":
                send(json.dumps(choice("", event[1])))
            elif kind == "usage":
                send(json.dumps({"choices": [], "usage": USAGE}))
            elif kind == "tok_usage":
                send(json.dumps({**choice(event[1], None), "usage": USAGE}))
            elif kind == "finish_usage":
                send(json.dumps({**choice("", event[1]), "usage": USAGE}))
            elif kind == "raw":
                send(json.dumps(event[1]))
            elif kind == "done":
                send("[DONE]")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request: Any, client_address: Any) -> None:
        # the adapter hangs up as soon as it reads an error event or times
        # out; the stub's next write then fails, which is the expected path
        pass


@pytest.fixture(scope="module")
def base_url():
    server = _QuietServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _request(chat: bool) -> InferenceRequest:
    req = InferenceRequest(prompt="p", max_tokens=8, temperature=0.0, top_p=1.0, request_id="r1")
    if chat:
        req.messages = [{"role": "user", "content": "p"}]
    return req


#: The four streamed paths: (label, chat?, async?).
STREAM_PATHS = [
    ("sync-raw", False, False),
    ("sync-chat", True, False),
    ("async-raw", False, True),
    ("async-chat", True, True),
]


def _stream(base_url: str, scenario: str, chat: bool, use_async: bool, **kw: Any) -> InferenceResponse:
    adapter = VLLMAdapter(model_name="m", api_base=f"{base_url}/{scenario}", **kw)
    request = _request(chat)
    if use_async:
        return asyncio.run(adapter.async_stream_generate(request))
    return adapter.generate(request, stream=True)


def _assert_error_row(resp: InferenceResponse, kind: str) -> None:
    assert resp.finish_reason == "error"
    assert isinstance(resp.error, str) and resp.error.strip(), "an error row never has empty error text"
    assert resp.error.startswith(kind), resp.error
    # the existing error-row shape: no text, no clocks, no token telemetry
    assert resp.generated_text == ""
    assert resp.ttft_ms == 0.0
    assert resp.num_tokens == 0
    assert resp.prompt_tokens is None and resp.cached_prompt_tokens is None
    assert resp.kv_transfer_params is None
    assert resp.usage_telemetry_available is False
    assert resp.num_tokens_source is None


@pytest.mark.parametrize("label,chat,use_async", STREAM_PATHS, ids=[p[0] for p in STREAM_PATHS])
class TestStreamedPaths:
    def test_stop_is_served(self, base_url, label, chat, use_async):
        resp = _stream(base_url, "ok_stop", chat, use_async)
        assert resp.error is None
        assert resp.finish_reason == "stop"
        assert resp.generated_text == "AB"
        assert resp.prompt_tokens == 7 and resp.num_tokens == 2

    def test_length_is_served(self, base_url, label, chat, use_async):
        resp = _stream(base_url, "ok_length", chat, use_async)
        assert resp.error is None and resp.finish_reason == "length"
        assert resp.generated_text == "AB"

    def test_an_empty_answer_that_stopped_is_still_served(self, base_url, label, chat, use_async):
        # a real empty answer, not a failure: the runner marks it empty_generation
        resp = _stream(base_url, "ok_empty_stop", chat, use_async)
        assert resp.error is None and resp.finish_reason == "stop"
        assert resp.generated_text == ""

    @pytest.mark.parametrize("scenario", ["usage_with_finish", "continuous_usage"])
    def test_a_chunk_that_carries_usage_beside_choices_is_read_in_full(
        self, base_url, label, chat, use_async, scenario
    ):
        # Before the review fix the usage branch skipped the whole chunk: its
        # text was dropped and its finish_reason never read, so a healthy
        # stream of this shape became a no_finish_reason error row.
        resp = _stream(base_url, scenario, chat, use_async)
        assert resp.error is None and resp.finish_reason == "stop"
        assert resp.generated_text == "AB"
        assert resp.prompt_tokens == 7 and resp.num_tokens == 2

    def test_an_error_event_first_is_an_error_row(self, base_url, label, chat, use_async):
        resp = _stream(base_url, "error_first", chat, use_async)
        _assert_error_row(resp, oca.ERROR_KIND_ENGINE)
        assert "EngineCore died" in resp.error and "InternalServerError" in resp.error

    def test_an_error_event_after_tokens_is_an_error_row_not_a_partial_answer(
        self, base_url, label, chat, use_async
    ):
        resp = _stream(base_url, "error_after_tokens", chat, use_async)
        _assert_error_row(resp, oca.ERROR_KIND_ENGINE)
        assert "late failure" in resp.error

    def test_an_error_event_with_an_empty_message_still_has_text(self, base_url, label, chat, use_async):
        resp = _stream(base_url, "error_empty_message", chat, use_async)
        _assert_error_row(resp, oca.ERROR_KIND_ENGINE)
        assert "InternalServerError" in resp.error and "500" in resp.error

    def test_the_flat_error_object_is_an_error_row(self, base_url, label, chat, use_async):
        resp = _stream(base_url, "error_flat", chat, use_async)
        _assert_error_row(resp, oca.ERROR_KIND_ENGINE)
        assert "flat failure" in resp.error

    def test_a_stream_with_no_finish_reason_is_an_error_row(self, base_url, label, chat, use_async):
        # the vLLM 0.19.1 shape of a failed NIXL KV load: usage chunk + [DONE]
        resp = _stream(base_url, "no_finish_usage_only", chat, use_async)
        _assert_error_row(resp, oca.ERROR_KIND_NO_FINISH)

    def test_a_stream_cut_after_tokens_is_an_error_row(self, base_url, label, chat, use_async):
        resp = _stream(base_url, "cut_after_tokens", chat, use_async)
        _assert_error_row(resp, oca.ERROR_KIND_NO_FINISH)
        # the evidence of where it failed stays in the text
        assert "2 character" in resp.error

    @pytest.mark.parametrize(
        "scenario,reason",
        [("finish_abort", "abort"), ("finish_repetition", "repetition"), ("finish_unknown", "content_filter")],
    )
    def test_any_other_finish_reason_is_an_error_row(
        self, base_url, label, chat, use_async, scenario, reason
    ):
        resp = _stream(base_url, scenario, chat, use_async)
        _assert_error_row(resp, oca.ERROR_KIND_UNSERVED)
        assert reason in resp.error

    def test_a_finish_reason_that_is_not_a_string_is_an_error_row_not_a_crash(
        self, base_url, label, chat, use_async
    ):
        resp = _stream(base_url, "finish_object", chat, use_async)
        _assert_error_row(resp, oca.ERROR_KIND_UNSERVED)
        assert "type" in resp.error


#: The three non-streamed paths: (label, chat?, async?).
BODY_PATHS = [("sync-raw", False, False), ("sync-chat", True, False), ("async-raw", False, True)]


def _body(base_url: str, scenario: str, chat: bool, use_async: bool) -> InferenceResponse:
    adapter = VLLMAdapter(model_name="m", api_base=f"{base_url}/{scenario}")
    request = _request(chat)
    if use_async:
        return asyncio.run(adapter.async_generate(request))
    return adapter.generate(request, stream=False)


@pytest.mark.parametrize("label,chat,use_async", BODY_PATHS, ids=[p[0] for p in BODY_PATHS])
class TestNonStreamedPaths:
    @pytest.mark.parametrize("scenario,reason", [("body_stop", "stop"), ("body_length", "length")])
    def test_stop_and_length_are_served(self, base_url, label, chat, use_async, scenario, reason):
        resp = _body(base_url, scenario, chat, use_async)
        assert resp.error is None and resp.finish_reason == reason
        assert resp.generated_text == "AB"

    @pytest.mark.parametrize("scenario", ["body_null", "body_absent"])
    def test_a_body_without_a_finish_reason_is_an_error_row(
        self, base_url, label, chat, use_async, scenario
    ):
        # the invented "length" default lived on these three paths too
        resp = _body(base_url, scenario, chat, use_async)
        _assert_error_row(resp, oca.ERROR_KIND_NO_FINISH)

    def test_abort_is_an_error_row(self, base_url, label, chat, use_async):
        resp = _body(base_url, "body_abort", chat, use_async)
        _assert_error_row(resp, oca.ERROR_KIND_UNSERVED)


def test_an_async_timeout_row_carries_error_text(base_url) -> None:
    # str(asyncio.TimeoutError()) is "": before S0F-25 the row had finish_reason
    # "error" and error "", and every consumer keyed on the truthiness of
    # `error` read it as a served row (the pd ticket gate then failed the whole
    # window on one client timeout).
    assert str(asyncio.TimeoutError()) == ""
    adapter = VLLMAdapter(model_name="m", api_base=f"{base_url}/hang", timeout=0.3)
    resp = asyncio.run(adapter.async_stream_generate(_request(False)))
    assert resp.finish_reason == "error"
    assert isinstance(resp.error, str) and resp.error.strip()
    assert "Timeout" in resp.error


def test_served_rows_keep_the_streamed_ttft_label_and_error_rows_do_not(base_url) -> None:
    ok = _stream(base_url, "ok_stop", False, True)
    assert getattr(ok, "ttft_methodology", None) == "streamed-first-delta"
    bad = _stream(base_url, "no_finish_usage_only", False, True)
    assert getattr(bad, "ttft_methodology", None) is None
    # the non-streamed async path labels served rows only as well (review LOW)
    assert getattr(_body(base_url, "body_stop", False, True), "ttft_methodology", None) == "full-response-proxy"
    assert getattr(_body(base_url, "body_abort", False, True), "ttft_methodology", None) is None


def test_the_retry_loop_reissues_an_unserved_response(base_url, monkeypatch) -> None:
    # the sync retry keys on finish_reason == "error"; an unserved response is
    # one now, so CAGE_ADAPTER_MAX_RETRIES > 0 re-sends it (default 0: one try)
    adapter = VLLMAdapter(
        model_name="m", api_base=f"{base_url}/no_finish_usage_only", max_retries=2, retry_backoff_s=0.0
    )
    resp = adapter.generate(_request(False), stream=True)
    assert resp.finish_reason == "error" and resp.retries == 2
    adapter = VLLMAdapter(model_name="m", api_base=f"{base_url}/no_finish_usage_only")
    assert adapter.generate(_request(False), stream=True).retries == 0


def test_the_rule_is_engine_agnostic(base_url) -> None:
    adapter = SGLangAdapter(model_name="m", api_base=f"{base_url}/no_finish_usage_only")
    resp = adapter.generate(_request(True), stream=True)
    _assert_error_row(resp, oca.ERROR_KIND_NO_FINISH)
    assert resp.engine_id == "sglang"


def test_logprob_stats_are_cleared_on_a_converted_row(base_url) -> None:
    resp = _stream(base_url, "finish_abort", True, False)
    assert getattr(resp, "mean_token_logprob", "missing") is None
    assert getattr(resp, "sum_token_logprob", "missing") is None


def test_served_set_and_source_pins() -> None:
    assert oca.SERVED_FINISH_REASONS == frozenset({"stop", "length"})
    src = ADAPTER_PY.read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    # no path starts from, or falls back to, an invented finish reason
    assert 'finish_reason = "length"' not in code
    assert 'or "length"' not in code
    assert '"finish_reason", "length")' not in code
    # every exception text goes through the never-empty helper
    assert "error=str(e)" not in code
    assert code.count("error=_exc_text(e)") == 6
