#!/usr/bin/env python3
"""Order:     stage 4 (validate), once per pd session, after the identity gate, on the PD stack the master starts from the stage 0 PD env, before stage 5 bills
Objective: Prove live that the prefill/decode stack moves KV bytes through the proxy on every request and none on a request straight to the decode (S2 block D DEC-D3, ADR-0182)
Cloud:     both

The S1 harness proved KV transfer only after the fact, per window, from the
decode's NIXL counters (ADR-0134). S2 block D moves that proof in front of the
money: before stage 5 bills, the master starts the control-rung PD stack and
this script sends one warm-up request through the proxy (so the decode's NIXL
and by-source metric children exist before the baseline scrape, whether the
engine creates them at logger init or at the first observation; review
2026-11-01 MEDIUM 2), then R unique prompts of about 1,200 words through the
proxy (half of them streamed, max_tokens 4), reads the engine ticket the proxy
relays in its response header and the decode's cached_tokens, scrapes the
decode and the prefill before and after, and judges the deltas with the same
pd_transfer_record the campaign writes per window. One more prompt then goes
straight to the decode, bypassing the proxy: the negative control, which
must move no bytes, count no pull and take no prompt token from the wire.

Verdicts: PASS when the five ADR-0134 clauses hold on the decode deltas,
every response carried a pull ticket, and the control moved nothing; FAIL
when any of those is violated or the proxy refused a counted request with
HTTP 502, the reasons naming each one with the proxy's error text. The
proxy's 502 covers five causes (pd_proxy.py): the prefill unreachable, the
prefill answering a non-200 status, its answer unparseable, no usable ticket
in its answer, the decode unreachable from the proxy. In every one the
stack had come up (the warm-up was served) and then could not serve a
proxied request, so the refusal is a verdict on the stack the owner reads
and either accepts or reruns (DEC-D3 as amended by the owner on 2026-11-01,
review MEDIUM 3; the rule is the proxy's status, the reason carries the
cause). The refusals lead the reasons list;
ERROR when the warm-up, a scrape, the control or the transport failed, or
a counted request got any other HTTP error (the stack did not answer; no
verdict on the transfer). The verdict document is written to --out in every case and the
last stdout line reads ``PD_SMOKE PASS|FAIL|ERROR ...``. The exit status is
1 on ERROR only: FAIL is a result the master records (DEC-D6) and stops on
until the owner accepts it. The identities pd_transfer_record computes are
recorded, not gated (spec section 11).
"""
from __future__ import annotations

import argparse
import http.client
import json
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.monitoring.vllm_telemetry import (  # noqa: E402
    PD_PREFILL_SERIES, PD_TRANSFER_SERIES, pd_transfer_record, scrape_counter_series,
)

SCHEMA = "cage.pd_smoke/1"
ADR = "ADR-0182"
MARKER = "PD_SMOKE"
#: pd_proxy.TICKET_HEADER: the proxy relays the prefill's kv_transfer_params
#: on every POST response under this header (tests/test_pd_smoke.py pins the
#: equality so the two files cannot drift).
TICKET_HEADER = "x-kv-transfer-params"
#: the planner's port table (run_campaign.PD_PROXY_PORT / PD_PREFILL_PORT /
#: PD_DECODE_PORT), which manage_vllm_pd.sh serves on localhost
DEFAULT_PROXY_URL = "http://localhost:8000"
DEFAULT_PREFILL_URL = "http://localhost:8100"
DEFAULT_DECODE_URL = "http://localhost:8200"
DEFAULT_REQUESTS = 8
DEFAULT_WORDS = 1200
MAX_TOKENS = 4
SEED = 20261101
#: the control's clauses: a request without a ticket must move no bytes,
#: count no pull and take no prompt token from the wire
CONTROL_SERIES = ("bytes_sum", "transfer_count", "external_kv_tokens")
VOCAB = (
    "the", "of", "and", "to", "in", "is", "that", "it", "was", "for", "on", "are", "as", "with",
    "his", "they", "be", "at", "one", "have", "this", "from", "or", "had", "by", "not", "word",
    "but", "what", "some", "we", "can", "out", "other", "were", "all", "there", "when", "up",
    "use", "your", "how", "said", "an", "each", "she", "which", "do",
)


class SmokeError(Exception):
    """A request or a scrape failed: the verdict is ERROR, not FAIL."""


class RequestRefused(SmokeError):
    """An HTTP error status on a request. On a counted request through the
    proxy, a 502 (the proxy's own refusal: no usable ticket from the prefill)
    is a FAIL reason; everywhere else it is a SmokeError like any other."""

    def __init__(self, status: int, url: str, detail: str):
        super().__init__(f"HTTP {status} from {url}: {detail}")
        self.status = status
        self.detail = detail


def build_prompts(count: int, words: int, seed: int = SEED) -> List[str]:
    """``count`` unique prompts of at least ``words`` words, byte-stable for a
    seed. The request index leads the text, so no two prompts share a KV
    block past their first few tokens: the decode has to pull each prefix
    from the prefill instead of hitting its own prefix cache, which is the
    transfer the smoke measures."""
    out: List[str] = []
    for i in range(count):
        rng = random.Random(seed + i)
        body = " ".join(rng.choice(VOCAB) for _ in range(words))
        out.append(f"Smoke request {i} of {count}. Reply with one word. {body}")
    return out


def _usage_from_json(raw: bytes) -> Dict[str, Any]:
    doc = json.loads(raw.decode("utf-8"))
    usage = doc.get("usage") if isinstance(doc, dict) else None
    if not isinstance(usage, dict):
        raise SmokeError("the response carried no usage block")
    return usage


def _usage_from_stream(raw: bytes) -> Dict[str, Any]:
    """The last usage block of an SSE stream (sent with stream_options
    include_usage); the chunks before it carry usage null."""
    usage: Optional[Dict[str, Any]] = None
    for line in raw.decode("utf-8").splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        if isinstance(chunk, dict) and isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
    if usage is None:
        raise SmokeError("the streamed response carried no usage chunk")
    return usage


def chat_request(url: str, model: str, prompt: str, *, stream: bool, max_tokens: int,
                 timeout_s: float) -> Dict[str, Any]:
    """One chat completion against ``url``: the status, the usage counts and
    the engine ticket from the response header (None when the header is
    absent, as on a request straight to a role)."""
    body: Dict[str, Any] = {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = int(resp.status)
            header = resp.headers.get(TICKET_HEADER)
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except (OSError, http.client.HTTPException) as read_exc:
            # the refusal body cut short: an ERROR written to --out, never a
            # traceback (review of the amendment, LOW 4)
            raise SmokeError(f"request to {url} failed: HTTP {exc.code} body cut short: {read_exc}") from read_exc
        raise RequestRefused(int(exc.code), url, detail) from exc
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
        # http.client.IncompleteRead (a body cut short) is an HTTPException, not an
        # OSError: it must become an ERROR verdict in --out, never an uncaught
        # traceback that leaves the previous attempt's verdict as the only file
        raise SmokeError(f"request to {url} failed: {exc}") from exc
    try:
        usage = _usage_from_stream(raw) if stream else _usage_from_json(raw)
        ticket = None if header is None else json.loads(header)
    except ValueError as exc:
        raise SmokeError(f"unreadable response from {url}: {exc}") from exc
    details = usage.get("prompt_tokens_details") or {}
    return {
        "status": status,
        "stream": stream,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens") if isinstance(details, dict) else None,
        "completion_tokens": usage.get("completion_tokens"),
        "ticket": ticket,
    }


def ticket_reason(ticket: Any) -> Optional[str]:
    """Why a proxied response's ticket does not prove a pull; None when it does."""
    if not isinstance(ticket, dict):
        return "no engine ticket in the proxy's response header"
    if ticket.get("do_remote_prefill") is not True:
        return "the ticket does not ask the decode to pull (do_remote_prefill is not true)"
    if not ticket.get("remote_block_ids"):
        return "the ticket names no remote blocks"
    return None


def _scrape(url: str, series: Dict[str, Any], role: str) -> Dict[str, Optional[float]]:
    try:
        return scrape_counter_series(url, series)
    except (OSError, ValueError) as exc:
        raise SmokeError(f"scrape of the {role} at {url} failed: {exc}") from exc


def control_record(before: Dict[str, Optional[float]], after: Dict[str, Optional[float]],
                   request: Dict[str, Any]) -> Dict[str, Any]:
    """The negative control: the decode's deltas across one request that
    bypassed the proxy, and the clauses it violated."""
    delta: Dict[str, Optional[float]] = {}
    for key in (*CONTROL_SERIES, "local_compute_tokens"):
        a, b = before.get(key), after.get(key)
        delta[key] = None if a is None or b is None else b - a
    reasons: List[str] = []
    for key in CONTROL_SERIES:
        value = delta[key]
        if value is None:
            reasons.append(f"control: {key} series absent on the decode")
        elif value != 0:
            reasons.append(f"control: a request straight to the decode moved {key} by {value:g}; "
                           "the stack pulls KV without a ticket")
    return {**request, "delta": delta, "no_transfer": not reasons, "reasons": reasons}


def run_smoke(a: argparse.Namespace) -> Dict[str, Any]:
    doc: Dict[str, Any] = {
        "schema": SCHEMA, "adr": ADR, "verdict": None, "error": None,
        "session": a.session, "model": a.model,
        "proxy_url": a.proxy_url, "prefill_url": a.prefill_url, "decode_url": a.decode_url,
        "warmup": None, "requests": [], "transfer": None, "control": None, "reasons": [],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        prompts = build_prompts(a.requests + 2, a.words)
        reasons: List[str] = []
        # one warm-up request through the proxy before the baseline scrape: the
        # decode's NIXL and by-source children then exist whether the engine
        # creates them at logger init or at the first observation ([?] for vLLM
        # 0.19.1 on the Mac); recorded apart under "warmup" and counted nowhere
        warm = chat_request(a.proxy_url, a.model, prompts[a.requests + 1], stream=False,
                            max_tokens=a.max_tokens, timeout_s=a.timeout_s)
        why = ticket_reason(warm["ticket"])
        warm.update({"index": -1, "ticket_ok": why is None})
        doc["warmup"] = warm
        if why is not None:
            reasons.append(f"warmup: {why}")
        scrape_start_ts = time.time()
        decode_0 = _scrape(a.decode_url, PD_TRANSFER_SERIES, "decode")
        prefill_0 = _scrape(a.prefill_url, PD_PREFILL_SERIES, "prefill")
        for i in range(a.requests):
            stream = i % 2 == 1
            try:
                row = chat_request(a.proxy_url, a.model, prompts[i], stream=stream,
                                   max_tokens=a.max_tokens, timeout_s=a.timeout_s)
            except RequestRefused as exc:
                if exc.status != 502:
                    raise
                # the proxy's refusal (pd_proxy.py answers 502 when the prefill hands
                # back no usable ticket): the stack was up and answered, so this is a
                # FAIL reason, recorded as a row with no counts, and the run goes on
                # so the scrapes and the control complete the record
                doc["requests"].append({
                    "index": i, "stream": stream, "status": 502, "error": exc.detail, "ticket": None,
                    "ticket_ok": False, "prompt_tokens": None, "cached_tokens": None, "completion_tokens": None,
                })
                reasons.append(f"request {i}: HTTP 502 from the proxy: {exc.detail}")
                continue
            why = ticket_reason(row["ticket"])
            row.update({"index": i, "ticket_ok": why is None})
            doc["requests"].append(row)
            if why is not None:
                reasons.append(f"request {i}: {why}")
        scrape_end_ts = time.time()
        decode_1 = _scrape(a.decode_url, PD_TRANSFER_SERIES, "decode")
        prefill_1 = _scrape(a.prefill_url, PD_PREFILL_SERIES, "prefill")
        # the record is judged over the rows the decode served (status 200): a
        # refused request moved nothing by definition and is already a reason
        served = [r for r in doc["requests"] if r["status"] == 200]
        counts = [r["prompt_tokens"] for r in served]
        prompt_tokens_sum = sum(counts) if served and all(isinstance(c, int) for c in counts) else None
        record = pd_transfer_record(
            start=decode_0, end=decode_1, prefill_start=prefill_0, prefill_end=prefill_1,
            n_served_rows=len(served), prompt_tokens_sum=prompt_tokens_sum,
            decode_url=a.decode_url, prefill_url=a.prefill_url,
            scrape_start_ts=scrape_start_ts, scrape_end_ts=scrape_end_ts,
        )
        doc["transfer"] = record
        control = chat_request(a.decode_url, a.model, prompts[a.requests], stream=False,
                               max_tokens=a.max_tokens, timeout_s=a.timeout_s)
        control["index"] = a.requests
        decode_2 = _scrape(a.decode_url, PD_TRANSFER_SERIES, "decode")
        doc["control"] = control_record(decode_1, decode_2, control)
        # the per-request reasons (refusals, bad tickets) lead: the planner
        # prints the first three, and a refused stack must not read as a served
        # stack that moved nothing
        doc["reasons"] = reasons + list(record["reasons"]) + list(doc["control"]["reasons"])
        doc["verdict"] = "PASS" if not doc["reasons"] else "FAIL"
    except (SmokeError, OSError, ValueError) as exc:
        doc["verdict"] = "ERROR"
        doc["error"] = str(exc)
    return doc


def _fmt(value: Any) -> str:
    if value is None:
        return "absent"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def summary_line(doc: Dict[str, Any]) -> str:
    if doc["verdict"] == "ERROR":
        return f"{MARKER} ERROR {doc['error']}"
    delta = doc["transfer"]["delta"]
    rows = doc["requests"]
    served = sum(1 for r in rows if r["status"] == 200)
    return (f"{MARKER} {doc['verdict']} requests={len(rows)} served={served} refused={len(rows) - served} "
            f"bytes={_fmt(delta['bytes_sum'])} transfers={_fmt(delta['transfer_count'])} "
            f"failed={_fmt(delta['failed_transfers'])} "
            f"control_no_transfer={doc['control']['no_transfer']} reasons={len(doc['reasons'])}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="S2 block D PD preflight smoke (ADR-0182)")
    p.add_argument("--session", required=True, help="the session the stack serves (recorded)")
    p.add_argument("--model", required=True, help="the served model id, as the stack reports it")
    p.add_argument("--proxy-url", default=DEFAULT_PROXY_URL)
    p.add_argument("--prefill-url", default=DEFAULT_PREFILL_URL)
    p.add_argument("--decode-url", default=DEFAULT_DECODE_URL)
    p.add_argument("--requests", type=int, default=DEFAULT_REQUESTS, help="prompts through the proxy")
    p.add_argument("--words", type=int, default=DEFAULT_WORDS, help="words per prompt")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--timeout-s", type=float, default=300.0)
    p.add_argument("--out", required=True, help="the verdict document (written in every case)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _parser()
    a = parser.parse_args(argv)
    if a.requests < 1:
        parser.error("--requests must be at least 1")
    doc = run_smoke(a)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary_line(doc))
    return 1 if doc["verdict"] == "ERROR" else 0


if __name__ == "__main__":
    raise SystemExit(main())
