"""S0F-27 and S0F-28: the pd proxy's reset relay and its stream relay.

S0F-28 (ADR-0138): the proxy relayed the decode stream with ``read(8192)``,
which blocks until 8 KB or the end of the stream, so the client saw the first
token together with the last one and pd TTFT equaled total time.

S0F-27 (ADR-0137): vLLM answers 200 to ``POST /reset_prefix_cache`` even when
the block pool declined; with ``?reset_running_requests=true`` a declined
reset is an HTTP 500 (v0.19.1 scheduler.py:1895-1902 through the generic
exception handler). An idle NixlConnector role holds the blocks of its last
ticketed requests until its next engine step, so the proxy sends one plain
one-token completion to each role (no ticket) before the reset, and repeats
the cycle once when a role answers 5xx.

Offline only: stub role servers on localhost, the real proxy in-process.
"""
from __future__ import annotations

import http.client
import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PD_PROXY_PY = REPO_ROOT / "scripts" / "2_serving" / "pd_proxy.py"

_spec = importlib.util.spec_from_file_location("pd_proxy_s0f27", PD_PROXY_PY)
pd_proxy = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pd_proxy
_spec.loader.exec_module(pd_proxy)

TICKET = {
    "do_remote_prefill": True,
    "do_remote_decode": False,
    "remote_block_ids": [[1]],
    "remote_engine_id": "e",
    "remote_request_id": "r",
    "remote_host": "localhost",
    "remote_port": 5600,
    "tp_size": 1,
}

#: One journal entry per upstream POST: (role, path with query, parsed body or None).
Journal = List[Tuple[str, str, Optional[Dict[str, Any]]]]


class _Role:
    """A recording role instance. ``reset_statuses`` is consumed one status
    per reset call (the last one repeats); ``wake_status`` answers a plain
    completion; ``release`` gates the second SSE event of a streamed decode
    answer so a test can prove the first event crossed the proxy before it."""

    def __init__(self, role: str, journal: Journal) -> None:
        self.role = role
        self.journal = journal
        self.reset_statuses: List[int] = [200]
        self.reset_body = b"{}"
        self.wake_status = 200
        self.wake_truncated = False
        self.release = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:
                pass

            def _json(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                self._json(200, b'{"status":"ok"}')

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length) if length else b""
                payload = json.loads(raw.decode("utf-8")) if raw else None
                outer.journal.append((outer.role, self.path, payload))
                if self.path.startswith("/reset_prefix_cache"):
                    status = (
                        outer.reset_statuses.pop(0)
                        if len(outer.reset_statuses) > 1
                        else outer.reset_statuses[0]
                    )
                    self._json(status, outer.reset_body)
                    return
                if isinstance(payload, dict) and payload.get("stream"):
                    # decode leg of a streamed request: two SSE events, the
                    # second one only after the test releases it
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()

                    def chunk(data: bytes) -> None:
                        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                        self.wfile.flush()

                    chunk(b'data: {"choices":[{"text":"A","finish_reason":null}]}\n\n')
                    outer.release.wait(timeout=8.0)
                    chunk(b'data: {"choices":[{"text":"B","finish_reason":"stop"}]}\n\n')
                    chunk(b"data: [DONE]\n\n")
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                    return
                if (
                    outer.wake_truncated
                    and isinstance(payload, dict)
                    and "kv_transfer_params" not in payload
                ):
                    # a wake answered with a body shorter than its Content-Length
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", "100")
                    self.end_headers()
                    self.wfile.write(b'{"cho')
                    self.wfile.flush()
                    self.close_connection = True
                    return
                if isinstance(payload, dict) and "kv_transfer_params" not in payload:
                    # a plain completion: the wake request
                    self._json(outer.wake_status, b'{"choices":[{"text":"x","finish_reason":"length"}]}')
                    return
                doc: Dict[str, Any] = {"choices": [{"text": "x"}]}
                if outer.role == "prefill":
                    doc["kv_transfer_params"] = TICKET
                self._json(200, json.dumps(doc).encode("utf-8"))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def stack():
    journal: Journal = []
    prefill, decode = _Role("prefill", journal), _Role("decode", journal)
    proxy = pd_proxy.build_server(0, prefill.url, decode.url)
    port = proxy.server_address[1]
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        yield journal, prefill, decode, port
    finally:
        proxy.shutdown()
        proxy.server_close()
        prefill.close()
        decode.close()


def _post(port: int, path: str, body: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=40)
    try:
        raw = json.dumps(body or {}).encode("utf-8")
        conn.request("POST", path, body=raw, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read().decode("utf-8"))
    finally:
        conn.close()


def _resets(journal: Journal, role: str) -> List[str]:
    return [path for r, path, _ in journal if r == role and path.startswith("/reset_prefix_cache")]


def _wakes(journal: Journal, role: str) -> List[Dict[str, Any]]:
    return [
        body for r, path, body in journal
        if r == role and path == pd_proxy.WAKE_PATH and isinstance(body, dict)
    ]


# ---------------------------------------------------------------------------
# S0F-28: the first event crosses the proxy before the upstream sends more
# ---------------------------------------------------------------------------


def test_stream_relay_forwards_the_first_event_before_the_second_exists(stack) -> None:
    _, _, decode, port = stack
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request(
            "POST", "/v1/completions",
            body=json.dumps({"prompt": "p", "max_tokens": 8, "stream": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        assert resp.status == 200
        # The decode stub holds its second event until `release` is set, and
        # this test sets it only AFTER the first bytes arrived: a relay that
        # waits for a full buffer deadlocks here and the 5 s socket timeout
        # fails the read (S0F-28: pd TTFT equaled total time).
        first = resp.read1(65536)
        assert b'"text":"A"' in first
        assert b'"text":"B"' not in first, "the second event was not sent yet"
        decode.release.set()
        rest = resp.read()
        assert b'"text":"B"' in rest and b"[DONE]" in rest
    finally:
        decode.release.set()
        conn.close()


def test_stream_relay_reads_what_is_available() -> None:
    src = PD_PROXY_PY.read_text(encoding="utf-8")
    assert "resp.read1(_CHUNK)" in src
    assert "resp.read(_CHUNK)" not in src


# ---------------------------------------------------------------------------
# S0F-27: wake both roles, then the honest reset, one retry on 5xx
# ---------------------------------------------------------------------------


def test_reset_wakes_both_roles_then_resets_with_the_honest_query(stack) -> None:
    journal, _, _, port = stack
    status, doc = _post(port, "/reset_prefix_cache?reset_running_requests=true")
    assert status == 200
    honest = f"{pd_proxy.RESET_PATH}?{pd_proxy.RESET_QUERY}"
    assert honest == "/reset_prefix_cache?reset_running_requests=true"
    for role in ("prefill", "decode"):
        assert _resets(journal, role) == [honest]
        wakes = _wakes(journal, role)
        assert len(wakes) == 1
        # a plain one-token completion: no ticket (a ticket would create a new
        # hold on the prefill), no model field, not streamed
        assert wakes[0]["max_tokens"] == 1
        assert "kv_transfer_params" not in wakes[0]
        assert "model" not in wakes[0]
        assert wakes[0].get("stream") is False
        # the wake precedes the reset on each role
        order = [path for r, path, _ in journal if r == role]
        assert order.index(pd_proxy.WAKE_PATH) < order.index(honest)
    assert doc["roles"] == {"prefill": 200, "decode": 200}
    assert doc["cycles"] == 1
    assert doc["query"] == pd_proxy.RESET_QUERY


def test_reset_without_a_query_still_sends_the_honest_query_upstream(stack) -> None:
    journal, _, _, port = stack
    status, _ = _post(port, "/reset_prefix_cache")
    assert status == 200
    honest = f"{pd_proxy.RESET_PATH}?{pd_proxy.RESET_QUERY}"
    assert _resets(journal, "prefill") == [honest]
    assert _resets(journal, "decode") == [honest]


def test_reset_repeats_the_cycle_once_when_a_role_declines(stack, monkeypatch) -> None:
    journal, prefill, _, port = stack
    monkeypatch.setattr(pd_proxy, "_RESET_RETRY_PAUSE_S", 0.01)
    prefill.reset_statuses = [500, 200]  # declined, then flushed
    status, doc = _post(port, "/reset_prefix_cache?reset_running_requests=true")
    assert status == 200
    assert doc["cycles"] == 2
    assert doc["roles"] == {"prefill": 200, "decode": 200}
    # the whole cycle repeats on both roles: wake, reset, wake, reset
    for role in ("prefill", "decode"):
        assert len(_wakes(journal, role)) == 2
        assert len(_resets(journal, role)) == 2
    assert doc["attempts"][0]["reset"]["prefill"] == 500


def test_reset_refuses_after_two_declined_cycles_and_names_the_role(stack, monkeypatch) -> None:
    journal, prefill, _, port = stack
    monkeypatch.setattr(pd_proxy, "_RESET_RETRY_PAUSE_S", 0.01)
    prefill.reset_statuses = [500]
    prefill.reset_body = json.dumps(
        {"error": {"message": "Failed to reset KV cache even when all the running requests are preempted"}}
    ).encode("utf-8")
    status, doc = _post(port, "/reset_prefix_cache?reset_running_requests=true")
    assert status == 502
    assert doc["cycles"] == pd_proxy.RESET_CYCLES == 2
    assert "prefill" in doc["failed"] and "decode" not in doc["failed"]
    assert "http-500" in doc["failed"]["prefill"]
    assert "Failed to reset KV cache" in doc["failed"]["prefill"]
    assert len(_resets(journal, "prefill")) == 2


def test_reset_does_not_retry_an_unreachable_role(stack) -> None:
    journal, _, decode, port = stack
    decode.close()
    status, doc = _post(port, "/reset_prefix_cache?reset_running_requests=true")
    assert status == 502
    assert doc["cycles"] == 1
    assert "unreachable" in doc["failed"]["decode"]
    assert len(_resets(journal, "prefill")) == 1


def test_reset_does_not_retry_a_4xx(stack) -> None:
    # 404 is what a role answers when VLLM_SERVER_DEV_MODE is off: retrying
    # cannot help, the refusal names the status at once.
    journal, prefill, _, port = stack
    prefill.reset_statuses = [404]
    status, doc = _post(port, "/reset_prefix_cache")
    assert status == 502
    assert doc["cycles"] == 1
    assert len(_resets(journal, "prefill")) == 1


def test_a_failed_wake_does_not_decide_the_reset(stack) -> None:
    # The reset's own status is the authority: a wake that fails while the
    # role still flushes is recorded and the reset stands.
    _, prefill, _, port = stack
    prefill.wake_status = 400
    status, doc = _post(port, "/reset_prefix_cache")
    assert status == 200
    assert doc["attempts"][0]["wake"]["prefill"] == 400
    assert doc["attempts"][0]["wake"]["decode"] == 200


def test_a_truncated_wake_answer_does_not_decide_the_reset(stack) -> None:
    # Review 2026-10-02 LOW: http.client raises IncompleteRead (an
    # HTTPException, not an OSError) on a body shorter than its Content-Length;
    # uncaught it killed the role thread and the client got no answer at all,
    # although both roles answered the reset 200.
    _, prefill, _, port = stack
    prefill.wake_truncated = True
    status, doc = _post(port, "/reset_prefix_cache")
    assert status == 200
    assert doc["roles"] == {"prefill": 200, "decode": 200}
    assert doc["attempts"][0]["wake"]["prefill"] is None
    assert doc["attempts"][0]["wake"]["decode"] == 200


def test_reset_budget_fits_the_runner_flush_timeout() -> None:
    # http.client applies a timeout per socket operation, so one leg is
    # bounded by about twice its timeout; the two roles run concurrently.
    # Two cycles of wake + reset plus the pause must stay inside the
    # adapter's 30 s flush timeout (openai_chat_adapter.flush_cache).
    cycle = 2 * pd_proxy._WAKE_TIMEOUT + 2 * pd_proxy._RESET_TIMEOUT
    worst = pd_proxy.RESET_CYCLES * cycle + (pd_proxy.RESET_CYCLES - 1) * pd_proxy._RESET_RETRY_PAUSE_S
    assert worst < 30.0
    assert pd_proxy.RESET_CYCLES == 2


def test_other_paths_with_a_query_are_not_mistaken_for_the_reset(stack) -> None:
    _, _, _, port = stack
    status, doc = _post(port, "/reset_prefix_cache_other?reset_running_requests=true")
    assert status == 404
