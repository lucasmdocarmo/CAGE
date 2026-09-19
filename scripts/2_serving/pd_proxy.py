#!/usr/bin/env python3
"""
Order:     stage 2 — launched by manage_vllm_pd.sh after both role instances are ready; clients hit THIS port
Objective: Stdlib-only prefill/decode disaggregation front-end (1P1D pattern: prefill request with max_tokens=1, then the full request to decode; streaming passthrough)
Cloud:     both

Wave-3 T3.2. The minimal front-end for the intra-node P/D topology: a client
sends ONE OpenAI-style request to this proxy; the proxy first sends it to the
PREFILL instance with ``max_tokens=1`` (so prefill computes/stages the KV and
generates nothing beyond the mandatory first token), then sends the FULL
request to the DECODE instance and streams the decode response back verbatim.

kv_transfer_params provenance (the T3.3 campaign-gate interplay — read this
before "fixing" a refusing run): any ``kv_transfer_params`` the prefill
ENGINE returns is forwarded to the decode request UNTOUCHED, and this proxy
NEVER stamps, invents, or normalizes a ``source`` field (or any other field)
inside it. run_experiment.py's campaign PD provenance gate accepts only
engine-real source stamps (allowlist: ``nixl``); until the live NIXL path is
verified at the Run-C-prime preflight, the CORRECT end-to-end outcome is that
gate REFUSING (absent/unknown source is not evidence) — a proxy that stamped
``source`` to appease the gate would be fabricating provenance, the exact
fail-closed violation the gate exists to catch.

[VERIFY-LIVE at Run-C-prime preflight] — the WHOLE data path: whether the
pinned vLLM's NixlConnector returns kv_transfer_params on the prefill
response, whether the decode instance consumes them from the request body,
whether ``max_tokens=1`` (+ ``stream=false``) is the correct prefill-side
request shape, and whether transfer actually happens. /health reports this as
PENDING, never PASS. No serving-behavior claim in this file is proven until
that smoke runs.

Fail-closed doctrine: an unparseable client body, a failed/unparseable
prefill response, or an unreachable upstream is a LOUD 4xx/5xx — the proxy
never silently degrades to decode-only serving (that would measure a
non-disaggregated path under a PD label).

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
that did not (never a partial success under a cold-start label). No other
metric family is relayed: a sampler pointed at the proxy finds absence,
never a doubled occupancy (per-role telemetry rides CAGE_TELEMETRY_ENDPOINTS
against the instances themselves). Both role instances are launched with
VLLM_SERVER_DEV_MODE=1 by manage_vllm_pd.sh, which is what enables the flush
endpoint on them [VERIFY-LIVE at Run-C-prime preflight: the pinned vLLM
exposes both paths on a NixlConnector-configured instance].
"""
from __future__ import annotations

import argparse
import http.client
import json
import sys
import threading  # noqa: F401  (documented seam: ThreadingHTTPServer below)
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
#: Upstream timeouts for the two relayed paths, seconds. The runner's own
#: flush timeout is 30 s and its probe timeout 10 s; the proxy visits the two
#: roles in sequence, so each leg stays well inside those budgets.
_METRICS_TIMEOUT = 4.0
_RESET_TIMEOUT = 10.0


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
    return int(round(total)) if found else None


def _upstream_call(
    url: str, method: str, path: str, timeout: float
) -> Tuple[Optional[int], str]:
    """(status, body text) of one upstream call; (None, reason) when the role
    is unreachable or times out (an OSError, never an exception escaping into
    the handler)."""
    host, port = _split_url(url)
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        try:
            conn.request(method, path)
            resp = conn.getresponse()
            return resp.status, resp.read().decode("utf-8", "replace")
        finally:
            conn.close()
    except OSError as exc:
        return None, f"unreachable ({exc.__class__.__name__}: {exc})"


def _prefill_body(body: Dict[str, Any]) -> Dict[str, Any]:
    """The prefill-side request: same prompt, generation clamped to 1 token.

    ``stream`` is forced off — the proxy consumes this response itself and
    needs one JSON object, not an SSE stream. Both the max_tokens=1 clamp and
    the stream-off override are request-shape assumptions
    [VERIFY-LIVE at Run-C-prime preflight].
    """
    out = dict(body)
    out["max_tokens"] = 1
    if "max_completion_tokens" in out:
        out["max_completion_tokens"] = 1  # chat-completions spelling
    out["stream"] = False
    return out


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
        for role, url in self._roles():
            status, body = _upstream_call(url, "GET", METRICS_PATH, _METRICS_TIMEOUT)
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
        """POST /reset_prefix_cache to BOTH roles; 200 only when both answered
        2xx, else 502 naming the role(s) that did not (a flush that one role
        declined is not a cold start)."""
        statuses: Dict[str, Any] = {}
        failed: Dict[str, str] = {}
        for role, url in self._roles():
            status, body = _upstream_call(url, "POST", RESET_PATH, _RESET_TIMEOUT)
            statuses[role] = status
            if status is None:
                failed[role] = body
            elif not 200 <= status < 300:
                failed[role] = f"http-{status}: {body[:200]}"
        if failed:
            self._reply_json(
                502,
                {
                    "error": "prefix cache reset failed on a pd role (fail closed)",
                    "roles": statuses,
                    "failed": failed,
                },
            )
            return
        self._reply_json(200, {"reset": RESET_PATH, "roles": statuses})

    # -- health ------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path == METRICS_PATH:
            self._relay_metrics()
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
        if self.path == RESET_PATH:
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

        # 2) DECODE: the ORIGINAL request, plus any engine-provided
        # kv_transfer_params forwarded VERBATIM. No field inside them is
        # added, removed, or rewritten here — in particular no "source"
        # stamp: provenance belongs to the engine alone, and the campaign
        # gate refusing engine-less provenance is correct (module docstring).
        decode_body = dict(body)
        if isinstance(prefill_json, dict) and "kv_transfer_params" in prefill_json:
            decode_body["kv_transfer_params"] = prefill_json["kv_transfer_params"]
        try:
            conn, resp = self._upstream_post(self.decode_url, self.path, decode_body)
        except OSError as exc:
            self._reply_json(502, {"error": f"decode upstream unreachable: {exc}"})
            return
        try:
            # 3) Streaming passthrough: status + content-type + raw body
            # chunks, verbatim (SSE streams flow through untouched).
            self.send_response(resp.status)
            ctype = resp.getheader("Content-Type")
            if ctype:
                self.send_header("Content-Type", ctype)
            self.end_headers()
            while True:
                chunk = resp.read(_CHUNK)
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
