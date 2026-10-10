"""C14 (2026-10-10): the sync SSE reader on an UNFRAMED stream.

The pd proxy answers HTTP/1.0 with no Content-Length and no chunked framing
(scripts pd_proxy.py). requests' iter_lines read that body with read(512),
which blocks until 512 bytes or the end of the stream, so the first token of a
short answer waited for the rest and ttft_ms landed late. The adapter now
reads such a body with read1. Chunked bodies keep iter_lines (the S0F-25 tests
cover them).
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.engine import InferenceRequest  # noqa: E402
from src.inference.vllm_adapter import VLLMAdapter  # noqa: E402

HOLD_S = 1.5


class _Http10Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # no length, no chunking: read until close

    def log_message(self, *a: Any) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or "0")
        self.rfile.read(length)
        chat = self.path.endswith("/chat/completions")

        def chunk(text: str, finish: Any = None) -> bytes:
            body = (
                {"choices": [{"delta": {"content": text}, "finish_reason": finish}]}
                if chat else {"choices": [{"text": text, "finish_reason": finish}]}
            )
            return f"data: {json.dumps(body, ensure_ascii=False)}\n\n".encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if chat:
            role = {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(role)}\n\n".encode("utf-8"))
        self.wfile.write(chunk("A"))
        self.wfile.flush()
        time.sleep(HOLD_S)  # the rest of the answer arrives later
        tail = chunk("Ü", None)  # a 2-byte character, split across writes
        cut = tail.index("Ü".encode("utf-8")) + 1
        self.wfile.write(tail[:cut])
        self.wfile.flush()
        time.sleep(0.05)
        self.wfile.write(tail[cut:])
        self.wfile.write(chunk("", "stop"))
        usage = {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2}}
        self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode("utf-8"))
        self.wfile.flush()


@pytest.fixture(scope="module")
def base_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Http10Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("chat", [False, True], ids=["sync-raw", "sync-chat"])
def test_first_token_is_clocked_when_it_arrives_on_an_unframed_stream(base_url, chat):
    adapter = VLLMAdapter(model_name="m", api_base=f"{base_url}/v1")
    request = InferenceRequest(prompt="p", max_tokens=8, temperature=0.0, request_id="r1")
    if chat:
        request.messages = [{"role": "user", "content": "p"}]
    response = adapter.generate(request, stream=True)
    assert response.error is None, response.error
    assert response.generated_text == "AÜ"
    assert response.finish_reason == "stop"
    # The first token left the server before the hold; its clock must not
    # wait for the held remainder (old reader: about HOLD_S late).
    assert response.ttft_ms < HOLD_S * 1000 / 2, response.ttft_ms
    assert response.total_time_ms >= HOLD_S * 1000 * 0.9


class _BrokenRaw:
    chunked = False

    def __init__(self) -> None:
        self.calls = 0

    def read1(self, amt: int, decode_content: bool = True) -> bytes:
        import urllib3

        self.calls += 1
        if self.calls == 1:
            return b'data: {"choices": [{"text": "A", "finish_reason": null}]}\n\n'
        raise urllib3.exceptions.ProtocolError("Connection broken: reset")


class _BrokenResponse:
    headers: dict[str, str] = {}

    def __init__(self) -> None:
        self.raw = _BrokenRaw()

    def raise_for_status(self) -> None:
        pass

    def __enter__(self) -> "_BrokenResponse":
        return self

    def __exit__(self, *exc: Any) -> None:
        pass


def test_a_urllib3_error_mid_stream_is_an_error_row_not_a_crash(monkeypatch):
    # Fresh review 2026-10-10 (C14-1): requests wraps urllib3 errors inside
    # iter_content; the read1 branch reads urllib3 directly, so an unwrapped
    # ProtocolError escaped the call sites' RequestException handler.
    import requests

    from src.inference import openai_chat_adapter as oca

    with pytest.raises(requests.exceptions.ConnectionError, match="Connection broken"):
        list(oca._sse_lines(_BrokenResponse()))
    monkeypatch.setattr(oca.requests, "post", lambda *a, **k: _BrokenResponse())
    adapter = VLLMAdapter(model_name="m", api_base="http://127.0.0.1:9/v1")
    request = InferenceRequest(prompt="p", max_tokens=8, temperature=0.0, request_id="r1")
    response = adapter.generate(request, stream=True)
    assert response.finish_reason == "error"
    assert isinstance(response.error, str) and "Connection broken" in response.error
