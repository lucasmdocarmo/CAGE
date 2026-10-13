"""P-1 (S0F-88): the live-load gate, run before any pod run (owner ruling R17).

S0 attempt 4 lost every calibrated lambda* to a client limit no test ever
exercised: the runner's open-file soft limit of 1,024 failed every request
above about 1,005 in flight with "Too many open files" (B-1). This gate
drives the PRODUCTION client (``VLLMAdapter.async_stream_generate`` under
``OpenLoopDispatcher``, the path every campaign window uses) against an
in-process fake OpenAI-compatible server that holds each stream open, so
about 2,500 requests are in flight at once, and requires every one served.

The soft limit is forced down to 1,024 FIRST and the B-1 helper must raise
it: without that step the gate passes on a Mac whose shell already allows
about one million files and would never have caught B-1.

Process safety: the server runs on the test's own event loop (aiohttp
AppRunner + TCPSite), no subprocess, no thread, no signal. The limit is
restored in ``finally``.

Opt-in (about 15 s and about 5,000 sockets): ``CAGE_LOAD_GATE=1``. Stage 0
of cage_experiment.sh runs it on the Mac and stage 4 runs it on the pod.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytestmark = pytest.mark.skipif(
    os.environ.get("CAGE_LOAD_GATE") != "1",
    reason="opt-in live-load gate (P-1, S0F-88): set CAGE_LOAD_GATE=1",
)

#: Offered rate and arrival count: 3,000 arrivals at 500 rps dispatch over
#: about 6 s; each stream is held HOLD_S, so Little's law puts about
#: 500 x 5 = 2,500 requests in flight at the peak [D].
RATE_RPS = 500.0
N_ARRIVALS = 3000
HOLD_S = 5.0
#: The gate: twice the ~1,005 in-flight ceiling S0 attempt 4 hit.
MIN_PEAK_IN_FLIGHT = 2000


def _sse(payload: Any) -> bytes:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {text}\n\n".encode("utf-8")


async def _handler(request: Any) -> Any:
    from aiohttp import web

    await request.read()
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    await resp.write(_sse({"choices": [{"delta": {"content": "A"}, "finish_reason": None}]}))
    await asyncio.sleep(HOLD_S)
    await resp.write(_sse({"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]}))
    await resp.write(
        _sse({"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}})
    )
    await resp.write(_sse("[DONE]"))
    await resp.write_eof()
    return resp


async def _drive() -> Any:
    from aiohttp import web

    from src.inference.engine import InferenceRequest
    from src.inference.vllm_adapter import VLLMAdapter
    from src.orchestration.load_generator import OpenLoopDispatcher, generate_arrival_schedule

    app = web.Application()
    app.router.add_route("POST", "/{tail:.*}", _handler)
    runner = web.AppRunner(app, handle_signals=False)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0, backlog=4096)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    adapter = VLLMAdapter(model_name="load-gate", api_base=f"http://127.0.0.1:{port}")
    request = InferenceRequest(prompt="p", max_tokens=8, temperature=0.0, top_p=1.0, request_id="load")
    request.messages = [{"role": "user", "content": "p"}]

    async def send(index: int, on_first_token: Any) -> Any:
        return await adapter.async_stream_generate(request, on_first_token=on_first_token)

    try:
        schedule = generate_arrival_schedule(RATE_RPS, seed=88, n_requests=N_ARRIVALS)
        return await OpenLoopDispatcher().run(schedule, send)
    finally:
        await runner.cleanup()


def test_the_client_serves_2000_in_flight_after_raising_a_1024_limit_p1() -> None:
    import resource

    from src.orchestration.load_generator import is_connect_error, raise_open_file_limit

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    forced = 1024 if soft == resource.RLIM_INFINITY else min(1024, soft)
    resource.setrlimit(resource.RLIMIT_NOFILE, (forced, hard))
    try:
        limits = raise_open_file_limit()
        assert limits["soft_after"] == "unlimited" or limits["soft_after"] > 2 * MIN_PEAK_IN_FLIGHT + 512, (
            f"B-1 helper left the soft limit at {limits['soft_after']}: the gate needs "
            f"about {2 * MIN_PEAK_IN_FLIGHT} sockets (client and server share this process)"
        )
        report = asyncio.run(_drive())
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

    connect = [r for r in report.records if is_connect_error(r.result_error) or is_connect_error(r.error)]
    errors = [r.result_error or r.error for r in report.records if not r.completed]
    print(
        f"[load-gate] arrivals={report.n_scheduled} completed={report.n_completed} "
        f"errors={report.n_errors} connect_errors={len(connect)} "
        f"peak_in_flight={report.max_in_flight_observed} limits={limits}"
    )
    assert not connect, f"{len(connect)} connect-phase failure(s), first: {connect[0].result_error or connect[0].error!r}"
    assert report.max_in_flight_observed >= MIN_PEAK_IN_FLIGHT, report.max_in_flight_observed
    assert report.n_completed == N_ARRIVALS, f"{len(errors)} unserved, first: {errors[:1]!r}"
