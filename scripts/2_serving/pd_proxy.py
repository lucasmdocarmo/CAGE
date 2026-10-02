#!/usr/bin/env python3
"""
Order:     stage 2 — launched by manage_vllm_pd.sh after both role instances are ready; clients hit THIS port
Objective: Stdlib-only prefill/decode disaggregation front-end (1P1D pattern: prefill request with max_tokens=1, then the full request to decode; streaming passthrough)
Cloud:     both

Wave-3 T3.2. The minimal front-end for the intra-node P/D topology: a client
sends ONE OpenAI-style request to this proxy; the proxy first sends it to the
PREFILL instance with ``max_tokens=1`` (so prefill computes/stages the KV and
generates nothing beyond the mandatory first token), then sends the FULL
request to the DECODE instance and streams the decode response back verbatim,
forwarding each piece as the decode sends it (``read1``; S0F-28, ADR-0138:
``read(n)`` held the first token until 8 KB or the end of the stream).

The ticket (S0F-22, ADR-0133, Batch 1; vLLM v0.19.1 source read 2026-10-01):
the prefill ENGINE writes its KV transfer ticket (``kv_transfer_params`` on
the non-stream response: ``do_remote_prefill``, ``do_remote_decode``,
``remote_block_ids`` nested per KV group, ``remote_engine_id``,
``remote_request_id``, ``remote_host``, ``remote_port``, ``tp_size``;
nixl_connector.py:989-998) ONLY when the prefill request asked for a remote
decode (``kv_transfer_params.do_remote_decode``, :958) and finished by its
length cap (an EOS first token finishes STOPPED and writes nothing, :960-966).
So the prefill leg carries PREFILL_REQUEST_TICKET and ``ignore_eos``, as
vLLM's own NIXL proxy does (tests/v1/kv_connector/nixl_integration/
toy_proxy_server.py:162-175 at v0.19.1), and drops ``stream_options``
(vLLM refuses it when ``stream`` is false, completion/protocol.py:409-414;
the runner streams every request with it). Before this, every request
through the proxy recomputed the prompt on the decode under a pd label.

The ticket is forwarded to the decode request VERBATIM and relayed to the
client VERBATIM in the response header TICKET_HEADER (the one the vLLM
adapter already parses; the decode's own response carries no ticket, its
request_finished returns none). This proxy NEVER stamps, invents or
normalizes a field inside it: the engine never writes a ``source`` key, and
run_experiment.py's pd gate checks the ticket's SHAPE (the keys the decode
reads, nixl_connector.py:818-828), never a stamp. A prefill response with
no usable ticket (absent, null, empty, an address key missing, or empty
block ids: the last one kills the decode engine, :855-856) is a LOUD 502
and the decode is never called.

[VERIFY-LIVE at Run-C-prime preflight] whether the pinned vLLM's
NixlConnector returns the ticket on this request shape, whether the decode
consumes it from the request body, and whether the transfer happens: the
Batch 2 per-window decode counters and checklist rows RC-13 and RC-14 are
the proof. /health reports this as PENDING, never PASS.

Fail-closed doctrine: an unparseable client body, a failed/unparseable
prefill response, a missing ticket, or an unreachable upstream is a LOUD
4xx/5xx; the proxy never silently degrades to decode-only serving (that
would measure a non-disaggregated path under a PD label).

GET /v1/models and GET /version are relayed from the DECODE role (the
instance that answers the client's generation): the runner's readiness check
and its engine-version capture dial them (integration audit distributed-1).

Cold start per window on the pd topology (ADR-0102 amendment 2026-09-19,
Batch 2 W2-R1): a pd cell dials THIS port for everything, including the
runner's strict per-window reset, which first reads the in-flight gauge from
``GET /metrics`` and then issues ``POST /reset_prefix_cache``. The proxy
relays both to the role instances: ``/metrics`` exposes ONE family only,
``vllm:num_requests_running`` labeled ``pd_role="prefill"`` / ``"decode"``
(each the sum of that role's own samples, the same rule the runner applies,
so the runner's probe reads the stack's total), and answers 503 when either
role is unreachable or lacks the gauge; ``/reset_prefix_cache`` is sent to
BOTH roles and answers 200 only when both flushed, else 502 naming the role
that did not (never a partial success under a cold-start label). Since
S0F-27 (ADR-0137) the reset carries ``?reset_running_requests=true``, the
vLLM mode in which a declined reset is an HTTP 500 instead of a 200, and is
preceded by one plain one-token completion sent straight to each role: an
idle NixlConnector role holds the blocks of its last ticketed requests until
its next engine step, so without the wake its reset is declined at every
window boundary. A 5xx repeats the cycle once, then the proxy refuses. No other
metric family is relayed: a sampler pointed at the proxy finds absence,
never a doubled occupancy (per-role telemetry rides CAGE_TELEMETRY_ENDPOINTS
against the instances themselves). manage_vllm_pd.sh launches both role
instances with VLLM_SERVER_DEV_MODE defaulting to 1, which is what enables
the flush endpoint on them; an ambient VLLM_SERVER_DEV_MODE=0 in the
operator's shell disables it on both roles and the strict reset then refuses
(both legs non-2xx, proxy 502, CacheResetError) [VERIFY-LIVE at Run-C-prime
preflight: the pinned vLLM exposes both paths on a NixlConnector-configured
instance]. The two role calls of each relay run CONCURRENTLY, so the proxy's
worst case is one leg; http.client applies a timeout per socket operation
(connect, then each read), so a leg is bounded by about twice its timeout,
and the per-leg values below keep that bound inside the runner's own probe
(10 s) and flush (30 s) budgets.
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlsplit

#: /health JSON value for the unverified transfer path — a pending check
#: reports PENDING, never PASS (fail-closed doctrine).
PD_DATA_PATH_STATUS = "PENDING [VERIFY-LIVE at Run-C-prime preflight]"

#: Streaming passthrough chunk size (decode response -> client), bytes.
_CHUNK = 8192

#: Upstream connect/read timeout, seconds. Generation can be slow; the proxy
#: must outwait the engine, not race it.
_UPSTREAM_TIMEOUT = 600.0

_HEALTH_TIMEOUT = 5.0

#: The in-flight gauge the runner's strict reset probes (run_experiment.py
#: COLD_START_RUNNING_GAUGE["vllm"]; the pd launcher is vLLM-only). Mirrored
#: literally, pinned by tests/test_pd_launcher.py against the runner source.
RUNNING_GAUGE = "vllm:num_requests_running"
#: The label the proxy stamps on each role's relayed gauge sample.
ROLE_LABEL = "pd_role"
#: The paths the runner's strict reset dials on a cell's endpoint (the flush
#: path mirrors src/inference/vllm_adapter.py _flush_endpoint).
METRICS_PATH = "/metrics"
RESET_PATH = "/reset_prefix_cache"
#: S0F-27 (ADR-0137): the query that makes a DECLINED reset visible. Without
#: it vLLM v0.19.1 answers 200 whatever the block pool did
#: (serve/cache/api_router.py:21-44); with it the scheduler raises when blocks
#: are still held (scheduler.py:1895-1902), the engine survives and the
#: generic handler answers 500. The proxy sends it to both roles on every
#: reset, whatever the client sent, so its own 200 means both roles flushed.
RESET_QUERY = "reset_running_requests=true"
#: S0F-27: the wake request sent straight to each role before the reset. An
#: idle NixlConnector role releases the blocks of its last ticketed requests
#: only on an engine step, so one plain one-token completion makes it step.
#: No kv_transfer_params (a ticket would create a new hold on the prefill),
#: no model field (CompletionRequest.model is optional, completion/protocol.py:45).
WAKE_PATH = "/v1/completions"
WAKE_BODY: Dict[str, Any] = {
    "prompt": "wake",
    "max_tokens": 1,
    "temperature": 0.0,
    "stream": False,
}
#: Upstream timeouts for the relayed paths, seconds, PER socket operation
#: (http.client semantics: connect, then each read), so one leg is bounded by
#: about twice the value. The roles are visited concurrently (_call_roles),
#: so the metrics relay's worst case is one leg: about 8 s against the
#: runner's 10 s probe timeout. A reset is RESET_CYCLES cycles of wake then
#: reset with _RESET_RETRY_PAUSE_S between them: 2 x (6 s + 6 s) + 1 s = 25 s
#: against the adapter's 30 s flush timeout (pinned by the proxy tests). The
#: reset leg was 14 s when the relay was one call; the bound now has to hold
#: two cycles. Provenance of 3 s: on the S0 single-instance vLLM the reset
#: request line and the engine's "Successfully reset prefix cache" line share
#: one second (17:36:42, 2026-09-30); no pd pair has been timed [A, RC-12].
#: A wake that outlasts its leg is recorded and the reset still decides.
_METRICS_TIMEOUT = 4.0
_WAKE_TIMEOUT = 3.0
_RESET_TIMEOUT = 3.0
#: S0F-27: a role that answers 5xx declined the reset (blocks still held: a
#: notification that had not reached the role when its wake request ran).
#: The whole cycle runs once more after the pause, then the proxy refuses.
RESET_CYCLES = 2
_RESET_RETRY_PAUSE_S = 1.0
#: GET paths relayed from the decode role (readiness + engine-version capture).
MODELS_PATH = "/v1/models"
VERSION_PATH = "/version"
_RELAY_GET_TIMEOUT = 5.0

#: The request-side kv_transfer_params that make the prefill engine write its
#: ticket: vLLM's own NIXL proxy sends exactly this (toy_proxy_server.py:162-169
#: at v0.19.1); NixlConnector reads do_remote_decode at request_finished
#: (nixl_connector.py:958). Mirrored literally, pinned by the proxy tests.
PREFILL_REQUEST_TICKET: Dict[str, Any] = {
    "do_remote_decode": True,
    "do_remote_prefill": False,
    "remote_engine_id": None,
    "remote_block_ids": None,
    "remote_host": None,
    "remote_port": None,
}
#: The ticket keys the DECODE reads before it pulls (nixl_connector.py:818-828);
#: a ticket missing one leaves the decode request waiting with no read
#: scheduled. Mirrored in run_experiment.PD_TICKET_REQUIRED_KEYS (pinned equal).
TICKET_REQUIRED_KEYS: Tuple[str, ...] = (
    "remote_block_ids",
    "remote_engine_id",
    "remote_request_id",
    "remote_host",
    "remote_port",
)
#: The response header the engine's ticket is relayed in, verbatim: the vLLM
#: adapter already parses it (openai_chat_adapter._extract_header_kv_transfer_params).
TICKET_HEADER = "x-kv-transfer-params"


def _split_url(url: str) -> Tuple[str, int]:
    """http://host:port -> (host, port); refuse anything else (fail closed)."""
    parts = urlsplit(url)
    if parts.scheme != "http" or not parts.hostname or not parts.port:
        raise ValueError(
            f"upstream url {url!r} must be http://host:port (explicit port; "
            "the proxy refuses to guess a default)"
        )
    return parts.hostname, parts.port


def _probe_health(url: str) -> str:
    """'ok' iff GET <url>/health answers 200, else a labeled failure string."""
    host, port = _split_url(url)
    try:
        conn = http.client.HTTPConnection(host, port, timeout=_HEALTH_TIMEOUT)
        try:
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            return "ok" if resp.status == 200 else f"http-{resp.status}"
        finally:
            conn.close()
    except OSError as exc:
        return f"unreachable ({exc.__class__.__name__})"


def sum_gauge(metrics_text: str, gauge: str) -> Optional[int]:
    """Sum every sample of ``gauge`` in a Prometheus text exposition (labeled
    or bare, optional trailing timestamp ignored); None when absent.

    The same rule as run_experiment.parse_running_requests (restated here:
    the proxy is stdlib-only and launched standalone), so the value each
    role contributes is exactly what the runner would read from that role.
    """
    total = 0.0
    found = False
    for raw in metrics_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(gauge + "{"):
            rest = line[line.index("}") + 1:] if "}" in line else ""
        elif line.startswith(gauge + " "):
            rest = line[len(gauge):]
        else:
            continue
        tokens = rest.split()
        if not tokens:
            continue
        try:
            total += float(tokens[0])
        except ValueError:
            continue
        found = True
    # A non-finite sample (NaN, +Inf) reads as ABSENT, never as a count and
    # never as an exception escaping into the handler (the runner's parser
    # applies the same guard).
    return int(round(total)) if found and math.isfinite(total) else None


def _call_roles(
    roles: Tuple[Tuple[str, str], ...], method: str, path: str, timeout: float,
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Tuple[Optional[int], str]]:
    """One upstream call per role, run CONCURRENTLY (one thread per role), so
    the relay's worst case is a single leg rather than the sum of both; each
    thread writes its own key of the result."""
    out: Dict[str, Tuple[Optional[int], str]] = {}

    def _one(role: str, url: str) -> None:
        out[role] = _upstream_call(url, method, path, timeout, body)

    threads = [
        threading.Thread(target=_one, args=(role, url), daemon=True)
        for role, url in roles
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return out


def _upstream_call(
    url: str, method: str, path: str, timeout: float,
    body: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[int], str]:
    """(status, body text) of one upstream call; (None, reason) when the role
    is unreachable or times out (an OSError, never an exception escaping into
    the handler). ``body`` is sent as JSON when given."""
    host, port = _split_url(url)
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        try:
            if body is None:
                conn.request(method, path)
            else:
                conn.request(
                    method, path, body=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
            resp = conn.getresponse()
            return resp.status, resp.read().decode("utf-8", "replace")
        finally:
            conn.close()
    except (OSError, http.client.HTTPException) as exc:
        # HTTPException too (review 2026-10-02): a body shorter than its
        # Content-Length raises IncompleteRead, which is not an OSError and
        # would otherwise kill the role thread and leave the client unanswered.
        return None, f"unreachable ({exc.__class__.__name__}: {exc})"


def _prefill_body(body: Dict[str, Any]) -> Dict[str, Any]:
    """The prefill-side request: same prompt, generation clamped to 1 token,
    asking the engine to stage the KV for a remote decode.

    ``stream`` is forced off (the proxy consumes this response itself and
    needs one JSON object; only the non-stream response carries the ticket)
    and ``stream_options`` is dropped with it (vLLM 0.19.1 refuses the pair
    stream=false + stream_options, completion/protocol.py:409-414).
    ``kv_transfer_params`` is PREFILL_REQUEST_TICKET, what makes
    request_finished write the ticket; ``ignore_eos`` keeps a prompt whose
    first sampled token is EOS from finishing STOPPED, which writes none
    (nixl_connector.py:960-966; sched/utils.py:104-117). The decode leg is
    the client's own request, untouched but for the forwarded ticket.
    [VERIFY-LIVE at Run-C-prime preflight] for the whole request shape.
    """
    out = dict(body)
    out["max_tokens"] = 1
    if "max_completion_tokens" in out:
        out["max_completion_tokens"] = 1  # chat-completions spelling
    out["stream"] = False
    out.pop("stream_options", None)
    out["kv_transfer_params"] = dict(PREFILL_REQUEST_TICKET)
    out["ignore_eos"] = True
    return out


def validate_ticket(ticket: Any) -> Optional[str]:
    """None when ``ticket`` is a usable prefill ticket, else the reason it is
    not: absent/null/empty, not an object, ``do_remote_prefill`` not true, a
    TICKET_REQUIRED_KEYS key missing, or ``remote_block_ids`` empty (a list
    with no ids in any group). The engine's own shape only; nothing is added.
    """
    if ticket is None or ticket == {} or ticket == "":
        return "ticket absent: the prefill returned no kv_transfer_params"
    if not isinstance(ticket, dict):
        return f"ticket is not an object ({type(ticket).__name__})"
    if ticket.get("do_remote_prefill") is not True:
        return (
            f"ticket do_remote_prefill is {ticket.get('do_remote_prefill')!r}, "
            "the decode pulls only when it is true"
        )
    missing = [k for k in TICKET_REQUIRED_KEYS if k not in ticket]
    if missing:
        return f"ticket lacks the key(s) the decode reads: {missing}"
    ids = ticket.get("remote_block_ids")
    if not isinstance(ids, list):
        return f"ticket remote_block_ids is {ids!r}, not a list of block id groups"
    flat = [i for group in ids for i in (group if isinstance(group, list) else [group])]
    if not flat:
        return "ticket remote_block_ids carries no block ids (nothing to pull)"
    return None


class PDProxyHandler(BaseHTTPRequestHandler):
    """One 1P1D exchange per POST; GET /health for the launcher's readiness."""

    # Filled in by build_server (subclassing keeps handler state process-global
    # and testable without module-level mutable config).
    prefill_url: str = ""
    decode_url: str = ""

    # HTTP/1.0 semantics: the decode response is streamed and close-delimited,
    # so no Content-Length is required for SSE passthrough.
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # stdout, not stderr
        print("[pd_proxy] " + fmt % args)

    # -- helpers -----------------------------------------------------------

    def _reply_json(self, status: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _upstream_post(
        self, url: str, path: str, body: Dict[str, Any]
    ) -> Tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
        host, port = _split_url(url)
        conn = http.client.HTTPConnection(host, port, timeout=_UPSTREAM_TIMEOUT)
        conn.request(
            "POST",
            path,
            body=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        return conn, conn.getresponse()

    # -- health ------------------------------------------------------------

    def _reply_text(self, status: int, text: str) -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _roles(self) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        return ("prefill", self.prefill_url), ("decode", self.decode_url)

    # -- cold start per window (ADR-0102 on the pd topology) ----------------

    def _relay_metrics(self) -> None:
        """GET /metrics: the in-flight gauge of EACH role, relabeled, and
        nothing else; 503 unless both roles are readable (a half-readable
        stack must never report a partial count as the whole)."""
        counts: Dict[str, int] = {}
        failures: Dict[str, str] = {}
        replies = _call_roles(self._roles(), "GET", METRICS_PATH, _METRICS_TIMEOUT)
        for role, _url in self._roles():
            status, body = replies[role]
            if status != 200:
                failures[role] = body if status is None else f"http-{status}"
                continue
            value = sum_gauge(body, RUNNING_GAUGE)
            if value is None:
                failures[role] = f"gauge {RUNNING_GAUGE} absent"
                continue
            counts[role] = value
        if failures:
            self._reply_json(
                503,
                {
                    "error": "pd stack in-flight count unreadable (fail closed)",
                    "gauge": RUNNING_GAUGE,
                    "roles": failures,
                },
            )
            return
        lines = [
            f"# HELP {RUNNING_GAUGE} In-flight requests per pd role, relayed by pd_proxy.",
            f"# TYPE {RUNNING_GAUGE} gauge",
        ]
        for role, _url in self._roles():
            lines.append(f'{RUNNING_GAUGE}{{{ROLE_LABEL}="{role}"}} {counts[role]}')
        self._reply_text(200, "\n".join(lines) + "\n")

    def _relay_reset(self) -> None:
        """Reset BOTH roles; 200 only when both flushed, else 502 naming the
        role(s) that did not (a flush that one role declined is not a cold
        start).

        S0F-27 (ADR-0137). One cycle is: wake both roles (WAKE_BODY straight
        to each role, never through the ticket path), then POST the reset
        with RESET_QUERY to both. With that query a role answers 5xx when
        its block pool declined, so 2xx from both means both flushed. A 5xx
        repeats the whole cycle once after _RESET_RETRY_PAUSE_S (a decode
        notification can reach the prefill after its first wake step); an
        unreachable role or a 4xx (no dev-mode route) is refused at once.
        The wake's own status is recorded and never decides the outcome: the
        reset status is the authority."""
        target = f"{RESET_PATH}?{RESET_QUERY}"
        attempts = []
        statuses: Dict[str, Any] = {}
        failed: Dict[str, str] = {}
        cycle = 0
        while cycle < RESET_CYCLES:
            cycle += 1
            wake = _call_roles(self._roles(), "POST", WAKE_PATH, _WAKE_TIMEOUT, WAKE_BODY)
            replies = _call_roles(self._roles(), "POST", target, _RESET_TIMEOUT)
            statuses, failed = {}, {}
            for role, _url in self._roles():
                status, body = replies[role]
                statuses[role] = status
                if status is None:
                    failed[role] = body
                elif not 200 <= status < 300:
                    failed[role] = f"http-{status}: {body[:200]}"
            attempts.append(
                {"wake": {role: wake[role][0] for role, _ in self._roles()}, "reset": dict(statuses)}
            )
            declined = all(
                statuses[role] is not None and 500 <= statuses[role] < 600 for role in failed
            )
            if not failed or not declined or cycle == RESET_CYCLES:
                break
            time.sleep(_RESET_RETRY_PAUSE_S)
        record = {"query": RESET_QUERY, "roles": statuses, "cycles": cycle, "attempts": attempts}
        if failed:
            self._reply_json(
                502,
                {
                    "error": "prefix cache reset failed on a pd role (fail closed)",
                    "failed": failed,
                    **record,
                },
            )
            return
        self._reply_json(200, {"reset": RESET_PATH, **record})

    # -- health ------------------------------------------------------------

    def _relay_decode_get(self, path: str) -> None:
        """GET ``path`` from the DECODE role, status and JSON body verbatim;
        503 when the role is unreachable (the readiness probe then reads
        not-ready, never a fabricated model list)."""
        status, body = _upstream_call(self.decode_url, "GET", path, _RELAY_GET_TIMEOUT)
        if status is None:
            self._reply_json(
                503, {"error": f"decode role unreachable for GET {path}: {body}"}
            )
            return
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path == METRICS_PATH:
            self._relay_metrics()
            return
        if self.path in (MODELS_PATH, VERSION_PATH):
            self._relay_decode_get(self.path)
            return
        if self.path != "/health":
            self._reply_json(404, {"error": f"unknown path {self.path!r}"})
            return
        prefill = _probe_health(self.prefill_url)
        decode = _probe_health(self.decode_url)
        healthy = prefill == "ok" and decode == "ok"
        # 503 unless BOTH upstreams answer: a proxy over a half-up pair must
        # never report ready (the launcher's readiness gate keys on this).
        self._reply_json(
            200 if healthy else 503,
            {
                "proxy": "ok",
                "prefill": prefill,
                "decode": decode,
                "pd_data_path": PD_DATA_PATH_STATUS,
            },
        )

    # -- the 1P1D data path -------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        # The vLLM adapter posts the reset WITH its query (S0F-27), so the
        # match is on the path alone; the relay sends RESET_QUERY upstream
        # whatever the client's query was.
        if urlsplit(self.path).path == RESET_PATH:
            self._relay_reset()
            return
        if not self.path.startswith("/v1/"):
            self._reply_json(404, {"error": f"unknown path {self.path!r}"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
        except (ValueError, UnicodeDecodeError) as exc:
            self._reply_json(400, {"error": f"unparseable request body: {exc}"})
            return

        # 1) PREFILL: same request, generation clamped to one token.
        try:
            conn, resp = self._upstream_post(
                self.prefill_url, self.path, _prefill_body(body)
            )
        except OSError as exc:
            self._reply_json(
                502, {"error": f"prefill upstream unreachable: {exc}"}
            )
            return
        try:
            prefill_raw = resp.read()
            prefill_status = resp.status
        finally:
            conn.close()
        if prefill_status != 200:
            # Fail closed: NEVER fall through to decode-only serving — that
            # would measure a non-disaggregated path under a PD label.
            self._reply_json(
                502,
                {
                    "error": "prefill request failed — refusing decode-only fallback",
                    "prefill_status": prefill_status,
                    "prefill_body": prefill_raw.decode("utf-8", "replace")[:512],
                },
            )
            return
        try:
            prefill_json = json.loads(prefill_raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            self._reply_json(
                502, {"error": f"prefill response unparseable: {exc}"}
            )
            return

        # 2) The TICKET (S0F-22): the engine's kv_transfer_params, validated
        # for the shape the decode reads and nothing else. No field inside it
        # is added, removed, or rewritten here (no "source" stamp: provenance
        # belongs to the engine alone, module docstring). A prefill answer
        # with no usable ticket is refused HERE, before the decode is called:
        # forwarding nothing would make the decode recompute the prompt under
        # a pd label (the silent S0 path), and forwarding an empty block list
        # with do_remote_prefill true kills the decode engine.
        ticket = prefill_json.get("kv_transfer_params") if isinstance(prefill_json, dict) else None
        reason = validate_ticket(ticket)
        if reason is not None:
            self._reply_json(
                502,
                {
                    "error": "prefill returned no usable KV transfer ticket -- "
                             "refusing decode-only fallback (S0F-22)",
                    "reason": reason,
                    "prefill_body": prefill_raw.decode("utf-8", "replace")[:512],
                },
            )
            return
        # 3) DECODE: the ORIGINAL request plus the ticket, VERBATIM.
        decode_body = dict(body)
        decode_body["kv_transfer_params"] = ticket
        try:
            conn, resp = self._upstream_post(self.decode_url, self.path, decode_body)
        except OSError as exc:
            self._reply_json(502, {"error": f"decode upstream unreachable: {exc}"})
            return
        try:
            # 4) Streaming passthrough: status + content-type + raw body
            # chunks, verbatim (SSE streams flow through untouched), plus the
            # engine's ticket relayed VERBATIM in TICKET_HEADER so the client
            # row carries what the prefill offered (the decode body never does).
            self.send_response(resp.status)
            ctype = resp.getheader("Content-Type")
            if ctype:
                self.send_header("Content-Type", ctype)
            self.send_header(TICKET_HEADER, json.dumps(ticket, separators=(",", ":")))
            self.end_headers()
            # read1, never read (S0F-28, ADR-0138): read(n) blocks until n
            # bytes or the end of the stream, so a short SSE answer reached
            # the client whole, at the end, and the client's TTFT equaled its
            # total time. read1 returns what the upstream has sent so far.
            while True:
                chunk = resp.read1(_CHUNK)
                if not chunk:
                    break
                self.wfile.write(chunk)
        finally:
            conn.close()


def build_server(
    port: int, prefill_url: str, decode_url: str, host: str = "127.0.0.1"
) -> ThreadingHTTPServer:
    """Construct the proxy server (separated from main() so the offline suite
    can run it in-process against stub upstreams — no network beyond
    localhost, no engine)."""
    # Validate BOTH upstream urls before binding anything (fail closed).
    _split_url(prefill_url)
    _split_url(decode_url)
    handler = type(
        "BoundPDProxyHandler",
        (PDProxyHandler,),
        {"prefill_url": prefill_url, "decode_url": decode_url},
    )
    return ThreadingHTTPServer((host, port), handler)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pd_proxy",
        description=(
            "Stdlib 1-prefill/1-decode disaggregation front-end (Wave-3 "
            "T3.2). Launched by manage_vllm_pd.sh; the data path is "
            "[VERIFY-LIVE at Run-C-prime preflight]."
        ),
    )
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--prefill-url", required=True, help="http://host:port of the prefill (kv producer) instance")
    parser.add_argument("--decode-url", required=True, help="http://host:port of the decode (kv consumer) instance")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    try:
        server = build_server(args.port, args.prefill_url, args.decode_url, host=args.host)
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(
        f"[pd_proxy] listening on {args.host}:{args.port} -> "
        f"prefill={args.prefill_url} decode={args.decode_url} "
        f"(pd_data_path: {PD_DATA_PATH_STATUS})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
