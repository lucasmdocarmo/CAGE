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

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
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
