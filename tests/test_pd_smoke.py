"""S2 block D DEC-D3 (ADR-0182): the PD preflight smoke, scripts/checks/pd_smoke.py.

Against in-process stub roles (the test_pd_launcher idiom) and the REAL
pd_proxy.py between them: R requests through the proxy move KV bytes on the
decode (the five ADR-0134 clauses read by pd_transfer_record), every
response carries the engine ticket in the proxy's header, the decode's usage
reports the pulled prompt as cached, and one request straight to the decode
(the negative control) moves nothing. PASS, FAIL on zero bytes, FAIL on a
failed transfer, FAIL on a control that moved bytes, FAIL when the proxy
refuses a counted request with HTTP 502 (a prefill without a usable ticket;
the owner's amendment of DEC-D3, 2026-11-01), ERROR on a 502 on the warm-up,
on an unreachable proxy, on a non-502 status and on an unreachable scrape;
the verdict file is written in every case.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT = REPO_ROOT / "scripts" / "checks" / "pd_smoke.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


smoke = _load("cage_pd_smoke", SCRIPT)
pd_proxy = _load("cage_pd_proxy_for_smoke", REPO_ROOT / "scripts" / "2_serving" / "pd_proxy.py")

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
#: the engine ticket shape vLLM 0.19.1's prefill returns (test_pd_launcher.STUB_KV_TRANSFER_PARAMS)
TICKET = {
    "do_remote_prefill": True, "do_remote_decode": False, "remote_block_ids": [[1, 2, 3]],
    "remote_engine_id": "stub-prefill-engine", "remote_request_id": "cmpl-stub-1",
    "remote_host": "localhost", "remote_port": 5600, "tp_size": 1,
}
_NO_TICKET = object()
BUCKETS = (1024.0, 65536.0, 1048576.0, 16777216.0, float("inf"))


def _counter(reg, name: str, labelnames: List[str], values: Dict[tuple, int]) -> None:
    c = Counter(name, "help", labelnames, registry=reg)
    for labels, value in values.items():
        c.labels(*labels).inc(value)


class _Role:
    """A stub role instance: OpenAI chat completions with usage (streamed or
    not), the engine ticket on the prefill, and a /metrics exposition whose
    NIXL and by-source families move the way vLLM 0.19.1's do when the decode
    pulls a prefilled prompt (external = the prompt minus its last token,
    which the decode computes locally; recomputed and local_compute +1 each).
    Prompt tokens = words of the last message (a stub tokenizer)."""

    def __init__(self, role: str, journal: List[Tuple[str, Dict[str, Any]]]):
        self.role = role
        self.journal = journal
        self.bytes_obs: List[float] = []
        self.failed_transfers = 0
        self.failed_notifications = 0
        self.external = 0
        self.local_compute = 0
        self.local_cache_hit = 0
        self.recomputed = 0
        self.preemptions = 0
        self.expired = 0
        # knobs
        self.transfer_bytes = 1_048_576   # bytes one pull moves; 0 observes a zero-byte pull
        self.fail_each_pull = False       # a failed transfer per pull
        self.leak_on_control = False      # a direct request moves bytes too (a broken control)
        self.prefill_ticket: Any = TICKET  # _NO_TICKET omits the key (the proxy answers 502)
        self.tickets_before_refusal: Any = None  # when set, the prefill carries a ticket on its first N answers only
        self.decode_status: int = 200     # the decode's status on a proxied request (the proxy relays it verbatim)
        self.decode_ok_posts_before_status: Any = None  # when set, the decode serves its first N proxied answers before decode_status
        self.prefill_status_after_posts: Any = None  # when set, the prefill answers 500 after its first N posts (the proxy wraps it in a 502)
        self.prefill_posts = 0
        self.decode_posts = 0
        self.lazy_children = False        # the NIXL and by-source children appear at the first pull, not at init
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/metrics":
                    self._send(200, outer.exposition().encode("utf-8"), "text/plain; version=0.0.4")
                elif self.path == "/health":
                    self._send(200, b'{"status":"ok"}')
                elif self.path == "/v1/models":
                    self._send(200, json.dumps({"object": "list", "data": [{"id": MODEL}]}).encode("utf-8"))
                else:
                    self._send(404, b"{}")

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                outer.journal.append((outer.role, body))
                if self.path != "/v1/chat/completions":
                    self._send(404, b"{}")
                    return
                n = len(body["messages"][-1]["content"].split())
                if outer.role == "prefill":
                    outer.prefill_posts += 1
                    if outer.prefill_status_after_posts is not None and outer.prefill_posts > outer.prefill_status_after_posts:
                        self._send(500, b'{"error":"prefill engine failure"}')
                        return
                    outer.local_compute += n
                    doc: Dict[str, Any] = {
                        "id": "pf", "object": "chat.completion", "model": body["model"],
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": "length"}],
                        "usage": {"prompt_tokens": n, "completion_tokens": 1, "total_tokens": n + 1,
                                  "prompt_tokens_details": {"cached_tokens": 0}},
                    }
                    refused = (outer.tickets_before_refusal is not None
                               and outer.prefill_posts > outer.tickets_before_refusal)
                    if outer.prefill_ticket is not _NO_TICKET and not refused:
                        doc["kv_transfer_params"] = outer.prefill_ticket
                    self._send(200, json.dumps(doc).encode("utf-8"))
                    return
                ticket = body.get("kv_transfer_params")
                if ticket is not None:
                    outer.decode_posts += 1
                    past_ok = (outer.decode_ok_posts_before_status is None
                               or outer.decode_posts > outer.decode_ok_posts_before_status)
                    if outer.decode_status != 200 and past_ok:
                        self._send(outer.decode_status, b'{"error":"engine failure"}')
                        return
                pulled = isinstance(ticket, dict) and ticket.get("do_remote_prefill") is True
                if pulled or outer.leak_on_control:
                    outer.bytes_obs.append(float(outer.transfer_bytes))
                    outer.external += n - 1   # the connector claims N minus the locally computed last token
                    outer.local_compute += 1
                    outer.recomputed += 1
                    if outer.fail_each_pull:
                        outer.failed_transfers += 1
                    cached = n - 1
                else:
                    outer.local_compute += n
                    cached = 0
                usage = {"prompt_tokens": n, "completion_tokens": 4, "total_tokens": n + 4,
                         "prompt_tokens_details": {"cached_tokens": cached}}
                if body.get("stream"):
                    chunks: List[Dict[str, Any]] = [
                        {"id": "dc", "object": "chat.completion.chunk", "model": body["model"],
                         "choices": [{"index": 0, "delta": {"content": "y"}, "finish_reason": None}], "usage": None},
                        {"id": "dc", "object": "chat.completion.chunk", "model": body["model"],
                         "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}], "usage": None},
                    ]
                    if (body.get("stream_options") or {}).get("include_usage"):
                        chunks.append({"id": "dc", "object": "chat.completion.chunk", "model": body["model"],
                                       "choices": [], "usage": usage})
                    payload = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                    self._send(200, payload.encode("utf-8"), "text/event-stream")
                    return
                doc = {"id": "dc", "object": "chat.completion", "model": body["model"],
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": "yyyy"}, "finish_reason": "length"}],
                       "usage": usage}
                self._send(200, json.dumps(doc).encode("utf-8"))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def exposition(self) -> str:
        reg = CollectorRegistry()
        if self.role == "decode":
            h = Histogram("vllm:nixl_bytes_transferred", "bytes", ["model_name", "engine"], registry=reg, buckets=BUCKETS)
            # whether vLLM 0.19.1 creates the labeled children at logger init (a
            # never-observed histogram reads 0, not absent) or at the first
            # observation is [?] on the Mac (review 2026-11-01 MEDIUM 2); the default
            # models the first reading, lazy_children the second, and the smoke's
            # warm-up request makes the verdict the same under both
            if not self.lazy_children or self.bytes_obs:
                child = h.labels(MODEL, "0")
                for b in self.bytes_obs:
                    child.observe(b)
            _counter(reg, "vllm:nixl_num_failed_transfers", ["model_name", "engine"], {(MODEL, "0"): self.failed_transfers})
            _counter(reg, "vllm:nixl_num_failed_notifications", ["model_name", "engine"], {(MODEL, "0"): self.failed_notifications})
            if not self.lazy_children or self.bytes_obs:
                _counter(reg, "vllm:prompt_tokens_by_source", ["model_name", "engine", "source"], {
                    (MODEL, "0", "external_kv_transfer"): self.external,
                    (MODEL, "0", "local_compute"): self.local_compute,
                    (MODEL, "0", "local_cache_hit"): self.local_cache_hit,
                })
            _counter(reg, "vllm:prompt_tokens_recomputed", ["model_name", "engine"], {(MODEL, "0"): self.recomputed})
            _counter(reg, "vllm:num_preemptions", ["model_name", "engine"], {(MODEL, "0"): self.preemptions})
        else:
            _counter(reg, "vllm:nixl_num_kv_expired_reqs", ["model_name", "engine"], {(MODEL, "0"): self.expired})
        return generate_latest(reg).decode("utf-8")

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def stack():
    """(journal, prefill stub, decode stub, live proxy url) with teardown."""
    journal: List[Tuple[str, Dict[str, Any]]] = []
    prefill, decode = _Role("prefill", journal), _Role("decode", journal)
    proxy = pd_proxy.build_server(0, prefill.url, decode.url)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        yield journal, prefill, decode, f"http://127.0.0.1:{proxy.server_address[1]}"
    finally:
        proxy.shutdown()
        proxy.server_close()
        prefill.close()
        decode.close()


def _run(tmp_path: Path, stack, *extra: str, requests: int = 4) -> Tuple[int, Dict[str, Any]]:
    _journal, prefill, decode, proxy_url = stack
    out = tmp_path / "pd_smoke.json"
    code = smoke.main([
        "--session", "cd-act1", "--model", MODEL, "--proxy-url", proxy_url,
        "--prefill-url", prefill.url, "--decode-url", decode.url,
        "--requests", str(requests), "--words", "30", "--out", str(out), *extra,
    ])
    return code, json.loads(out.read_text(encoding="utf-8"))


def test_pass_when_bytes_move_on_every_request_and_the_control_moves_none(tmp_path, stack, capsys):
    journal, _prefill, _decode, _ = stack
    code, doc = _run(tmp_path, stack)
    assert code == 0 and doc["verdict"] == "PASS" and doc["error"] is None, doc
    assert doc["schema"] == "cage.pd_smoke/1" and doc["adr"] == "ADR-0182"
    assert doc["session"] == "cd-act1" and doc["model"] == MODEL
    assert "PD_SMOKE PASS" in capsys.readouterr().out
    # the five clauses on the decode deltas across the R requests
    tr = doc["transfer"]
    assert tr["verified"] is True and tr["reasons"] == [] and tr["n_served_rows"] == 4
    assert tr["delta"]["transfer_count"] == 4 and tr["delta"]["bytes_sum"] == 4 * 1_048_576
    assert tr["delta"]["failed_transfers"] == 0 and tr["delta"]["external_kv_tokens"] > 0
    assert tr["prefill_delta"] == {"kv_expired_reqs": 0}
    # the ADR-0134 identities are recorded, never gated: on a pull the connector
    # claims N minus the one locally computed token (vllm_telemetry.py:835-838),
    # so the S1 identity over external plus cache hits reads False on the engine
    ids = tr["identities"]
    assert ids["local_compute_equals_served"] is True and ids["recomputed_equals_served"] is True
    assert ids["no_preemptions"] is True
    assert ids["external_plus_cache_hit_equals_prompt_tokens"] is False and tr["identities_hold"] is False
    # every request: the engine ticket relayed by the proxy, every second one
    # streamed, the decode's usage reporting the pulled prompt as cached
    reqs = doc["requests"]
    assert [r["stream"] for r in reqs] == [False, True, False, True]
    for r in reqs:
        assert r["ticket"]["do_remote_prefill"] is True and r["ticket_ok"] is True
        assert r["prompt_tokens"] > 30 and r["cached_tokens"] == r["prompt_tokens"] - 1
        assert r["completion_tokens"] == 4 and r["status"] == 200
    assert tr["prompt_tokens_sum"] == sum(r["prompt_tokens"] for r in reqs)
    # the negative control went straight to the decode and moved nothing
    ctl = doc["control"]
    assert ctl["no_transfer"] is True and ctl["cached_tokens"] == 0 and ctl["reasons"] == []
    assert ctl["delta"]["bytes_sum"] == 0 and ctl["delta"]["transfer_count"] == 0
    assert ctl["delta"]["external_kv_tokens"] == 0
    assert ctl["delta"]["local_compute_tokens"] == ctl["prompt_tokens"]
    # the journal: the warm-up and the 4 counted requests through the proxy (a
    # prefill leg at max_tokens 1 and a decode leg with the ticket each), then
    # the control straight to the decode without a ticket
    roles = [r for r, _ in journal]
    assert roles.count("prefill") == 5 and roles.count("decode") == 6
    assert roles[-1] == "decode" and "kv_transfer_params" not in journal[-1][1]
    assert all(b["max_tokens"] == 1 for r, b in journal if r == "prefill")
    decode_legs = [b for r, b in journal if r == "decode"][:5]
    assert all(b["kv_transfer_params"] == TICKET and b["max_tokens"] == 4 for b in decode_legs)
    assert [b["stream"] for b in decode_legs] == [False, False, True, False, True]
    # unique prompts: no two requests share a content, the warm-up included
    contents = [b["messages"][-1]["content"] for r, b in journal if r == "prefill"]
    assert len(set(contents)) == 5
    # the warm-up is recorded apart and counted nowhere (the baseline scrape follows it)
    assert doc["warmup"]["ticket_ok"] is True and doc["warmup"]["index"] == -1
    assert doc["reasons"] == []


def test_fail_when_no_bytes_move(tmp_path, stack, capsys):
    _, _, decode, _ = stack
    decode.transfer_bytes = 0
    code, doc = _run(tmp_path, stack)
    assert code == 0 and doc["verdict"] == "FAIL"
    assert any("bytes_sum > 0 violated" in r for r in doc["reasons"]), doc["reasons"]
    assert doc["transfer"]["verified"] is False
    assert "PD_SMOKE FAIL" in capsys.readouterr().out


def test_fail_on_a_failed_transfer(tmp_path, stack):
    _, _, decode, _ = stack
    decode.fail_each_pull = True
    code, doc = _run(tmp_path, stack)
    assert code == 0 and doc["verdict"] == "FAIL"
    assert any("failed_transfers == 0 violated" in r for r in doc["reasons"]), doc["reasons"]


def test_fail_when_the_control_moves_bytes(tmp_path, stack):
    _, _, decode, _ = stack
    decode.leak_on_control = True
    code, doc = _run(tmp_path, stack)
    assert code == 0 and doc["verdict"] == "FAIL"
    assert doc["transfer"]["verified"] is True, "the proxied requests were fine; the control is the finding"
    assert doc["control"]["no_transfer"] is False
    assert any("control" in r for r in doc["reasons"]), doc["reasons"]


def test_fail_when_the_prefill_returns_no_ticket(tmp_path, stack, capsys):
    # the owner's amendment of DEC-D3 (2026-11-01, review MEDIUM 3): the proxy
    # refuses a counted request with HTTP 502 when the prefill hands back no
    # usable ticket; that is a verdict on the stack (it was up and answered),
    # so it is a FAIL reason the owner can accept, not an ERROR with no path on
    _, prefill, _, _ = stack
    prefill.tickets_before_refusal = 1   # the warm-up gets its ticket; every counted request is refused
    code, doc = _run(tmp_path, stack)
    assert code == 0 and doc["verdict"] == "FAIL" and doc["error"] is None, doc
    assert doc["warmup"]["ticket_ok"] is True
    rows = doc["requests"]
    assert [r["index"] for r in rows] == [0, 1, 2, 3]
    assert all(r["status"] == 502 and r["ticket"] is None and r["ticket_ok"] is False for r in rows)
    assert all(r["prompt_tokens"] is None and r["cached_tokens"] is None for r in rows)
    assert all("no usable KV transfer ticket" in r["error"] for r in rows)
    refusals = [r for r in doc["reasons"] if "HTTP 502 from the proxy" in r]
    assert len(refusals) == 4 and refusals[0].startswith("request 0: HTTP 502 from the proxy: ")
    assert "no usable KV transfer ticket" in refusals[0]
    # the record is still written, over the served rows only (none here), and
    # the control still runs: the document is complete for the owner's reading
    tr = doc["transfer"]
    assert tr["n_served_rows"] == 0 and tr["prompt_tokens_sum"] is None and tr["delta"]["transfer_count"] == 0
    assert doc["control"]["no_transfer"] is True
    # review 2026-11-01 (amendment) LOW 3: the refusals lead the reasons (the
    # planner prints the first three) and the marker line names them apart from
    # the served rows, so "requests=4 bytes=0" cannot read as four served requests
    assert doc["reasons"][0].startswith("request 0: HTTP 502 from the proxy: ")
    assert "PD_SMOKE FAIL requests=4 served=0 refused=4 " in capsys.readouterr().out


def test_fail_when_some_requests_are_refused_judges_the_served_rows_on_their_own_count(tmp_path, stack):
    _, prefill, _, _ = stack
    prefill.tickets_before_refusal = 3   # the warm-up and two counted requests served, two refused
    code, doc = _run(tmp_path, stack)
    assert code == 0 and doc["verdict"] == "FAIL"
    rows = doc["requests"]
    assert [r["status"] for r in rows] == [200, 200, 502, 502]
    assert [r["ticket_ok"] for r in rows] == [True, True, False, False]
    assert [r["stream"] for r in rows] == [False, True, False, True]
    tr = doc["transfer"]
    assert tr["n_served_rows"] == 2 and tr["delta"]["transfer_count"] == 2
    assert tr["prompt_tokens_sum"] == rows[0]["prompt_tokens"] + rows[1]["prompt_tokens"]
    assert tr["verified"] is True, "the two served rows moved their bytes; the refusals are the finding"
    assert [r for r in doc["reasons"] if r.startswith("request 2: HTTP 502")] and \
           [r for r in doc["reasons"] if r.startswith("request 3: HTTP 502")]
    assert not any(r.startswith(("request 0", "request 1")) for r in doc["reasons"])


def test_error_on_a_502_on_the_warmup(tmp_path, stack, capsys):
    # the warm-up establishes that the stack is ready; a refusal there is not a
    # judgment of the transfer, so it stays ERROR (no acceptance path)
    _, prefill, _, _ = stack
    prefill.prefill_ticket = _NO_TICKET
    code, doc = _run(tmp_path, stack)
    assert code == 1 and doc["verdict"] == "ERROR"
    assert "HTTP 502" in doc["error"] and doc["warmup"] is None and doc["requests"] == []
    assert doc["transfer"] is None and doc["control"] is None
    assert "PD_SMOKE ERROR" in capsys.readouterr().out


def test_error_on_a_non_502_status_on_a_counted_request(tmp_path, stack):
    # the boundary of the amendment: only the proxy's own refusal (502) is a
    # FAIL reason; a 500 the proxy relays from the decode engine on a COUNTED
    # request stays ERROR. The decode serves the warm-up first, so the status
    # reaches the loop's catch and not the warm-up (review 2026-11-01 of the
    # amendment, HIGH 1: with the 500 on the warm-up this test had no power)
    _, _, decode, _ = stack
    decode.decode_status = 500
    decode.decode_ok_posts_before_status = 1
    code, doc = _run(tmp_path, stack)
    assert code == 1 and doc["verdict"] == "ERROR" and "HTTP 500" in doc["error"], doc
    assert doc["warmup"]["ticket_ok"] is True and doc["requests"] == []


def test_fail_when_the_prefill_errors_after_the_warmup(tmp_path, stack):
    # the rule is the proxy's status, not its cause: the proxy wraps a prefill
    # engine error in its own 502 ("prefill request failed"), and that stack
    # was up and answered too, so it is a FAIL the owner reads and accepts or
    # reruns (review of the amendment, MEDIUM 2, kept coarse on purpose; the
    # reason names the cause and the prefill's status)
    _, prefill, _, _ = stack
    prefill.prefill_status_after_posts = 1
    code, doc = _run(tmp_path, stack)
    assert code == 0 and doc["verdict"] == "FAIL"
    assert all(r["status"] == 502 for r in doc["requests"])
    assert "prefill request failed" in doc["reasons"][0] and '"prefill_status": 500' in doc["reasons"][0]


def test_error_on_a_502_whose_body_is_cut_short():
    # review of the amendment, LOW 4: the refusal body is read inside the
    # HTTPError handler; a body cut short there raised IncompleteRead past
    # run_smoke's except tuple and the smoke died without writing --out
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b'{"error":"x"}')

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(smoke.SmokeError, match="cut short"):
            smoke.chat_request(f"http://127.0.0.1:{srv.server_address[1]}", MODEL, "x",
                               stream=False, max_tokens=4, timeout_s=5.0)
    finally:
        srv.shutdown()
        srv.server_close()


def test_error_when_the_proxy_is_unreachable(tmp_path, stack):
    _journal, prefill, decode, _ = stack
    out = tmp_path / "pd_smoke.json"
    code = smoke.main([
        "--session", "cd-act1", "--model", MODEL, "--proxy-url", "http://127.0.0.1:9",
        "--prefill-url", prefill.url, "--decode-url", decode.url, "--requests", "2", "--words", "30",
        "--out", str(out),
    ])
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert code == 1 and doc["verdict"] == "ERROR" and "failed" in doc["error"]
    assert doc["warmup"] is None and doc["requests"] == []


def test_error_when_a_role_cannot_be_scraped(tmp_path, stack):
    journal, prefill, _decode, proxy_url = stack
    out = tmp_path / "pd_smoke.json"
    code = smoke.main([
        "--session", "cd-act1", "--model", MODEL, "--proxy-url", proxy_url, "--prefill-url", prefill.url,
        "--decode-url", "http://127.0.0.1:9", "--requests", "2", "--words", "30", "--out", str(out),
    ])
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert code == 1 and doc["verdict"] == "ERROR" and "scrape" in doc["error"]
    # review 2026-11-01 MEDIUM 2: the one warm-up request goes through the proxy
    # BEFORE the baseline scrape (so lazily created metric children exist); only
    # its two legs reach the stack, none of the smoke requests
    warm_prompt = smoke.build_prompts(2 + 2, 30)[2 + 1]
    assert [r for r, _ in journal] == ["prefill", "decode"], "only the warm-up legs run before the first scrape"
    assert all(req["messages"][0]["content"] == warm_prompt for _, req in journal)
    assert doc["warmup"]["ticket_ok"] is True and doc["requests"] == []


def test_prompts_are_unique_sized_and_byte_stable():
    a = smoke.build_prompts(8, 1200)
    b = smoke.build_prompts(8, 1200)
    assert a == b and len(set(a)) == 8
    assert all(len(p.split()) >= 1200 for p in a)
    # no shared text past the request header: prefix hits stay in the header
    heads = {p.split(".", 1)[1][:200] for p in a}
    assert len(heads) == 8


def test_defaults_name_the_registered_ports_and_shape():
    assert smoke.DEFAULT_REQUESTS == 8 and smoke.DEFAULT_WORDS == 1200 and smoke.MAX_TOKENS == 4
    assert (smoke.DEFAULT_PROXY_URL, smoke.DEFAULT_PREFILL_URL, smoke.DEFAULT_DECODE_URL) == (
        "http://localhost:8000", "http://localhost:8100", "http://localhost:8200")
    assert smoke.TICKET_HEADER == pd_proxy.TICKET_HEADER


def test_pass_when_the_metric_children_appear_only_after_the_first_pull(tmp_path, stack):
    # review 2026-11-01 MEDIUM 2: if the decode creates the NIXL and by-source
    # children at the first observation, a baseline scrape before any request
    # reads them as absent and the deltas are None (three FAIL clauses on a
    # stack that moved every byte); the warm-up request before the baseline
    # scrape makes the children exist under either reading
    _journal, _prefill, decode, _ = stack
    decode.lazy_children = True
    code, doc = _run(tmp_path, stack)
    assert code == 0 and doc["verdict"] == "PASS", doc["reasons"] + [doc["error"]]
    tr = doc["transfer"]
    assert tr["n_served_rows"] == 4 and tr["delta"]["transfer_count"] == 4 and tr["delta"]["bytes_sum"] == 4 * 1_048_576
    assert doc["warmup"]["ticket_ok"] is True and doc["control"]["no_transfer"] is True


def test_error_on_a_truncated_response_body():
    # review 2026-11-01 MEDIUM 1: a body cut short raises http.client.IncompleteRead,
    # an HTTPException and not an OSError; it must become a SmokeError (an ERROR
    # verdict written to --out), never an uncaught traceback that leaves the
    # previous attempt's verdict as the only file on the pod
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b"0123456789")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(smoke.SmokeError, match="failed"):
            smoke.chat_request(f"http://127.0.0.1:{srv.server_address[1]}", MODEL, "x",
                               stream=False, max_tokens=4, timeout_s=5.0)
    finally:
        srv.shutdown()
        srv.server_close()

