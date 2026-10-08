"""S0F-27 (ADR-0137), adapter and runner half: a reset the engine confirms.

vLLM v0.19.1 answers 200 to ``POST /reset_prefix_cache`` whatever its block
pool did. With ``?reset_running_requests=true`` a declined reset raises in
the scheduler and the server answers 500, so in that mode 200 means flushed.
The vLLM adapter posts that query, the strict reset turns a 500 into a
refusal, and the window's ``cold_start`` record says whether the engine
itself confirmed the flush (``engine_confirmed``: true for vLLM, null for an
engine whose flush semantics were not read, SGLang).

The proxy half (wake both roles, retry once) is pinned in
tests/test_pd_proxy_s0f27_s0f28.py; the last test here runs the runner's
strict reset through the real proxy against two stub roles.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.errors import EngineCapabilityUnavailableError  # noqa: E402
from src.inference.lmdeploy_adapter import LMDeployAdapter  # noqa: E402
from src.inference.sglang_adapter import SGLangAdapter  # noqa: E402
from src.inference.vllm_adapter import VLLMAdapter  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("run_experiment_s0f27", REPO_ROOT / "scripts" / "3_run" / "run_experiment.py")
pd_proxy = _load("pd_proxy_s0f27_runner", REPO_ROOT / "scripts" / "2_serving" / "pd_proxy.py")

HONEST = "/reset_prefix_cache?reset_running_requests=true"
# S0F-54 (ADR-0149): SGLang's flush is the deferred form (performed once the
# scheduler is fully idle, 400 past the deadline).
SGLANG_FLUSH = "/flush_cache?timeout=20"


class _Engine:
    """A stub engine: /metrics reports zero in-flight requests on the gauge of
    ``backend``; every POST is journaled with its path and answered with
    ``post_status``."""

    def __init__(self, backend: str = "vllm") -> None:
        self.posts: List[str] = []
        self.post_status = 200
        gauge = {"vllm": "vllm:num_requests_running", "sglang": "sglang:num_running_reqs"}[backend]
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802
                body = f"{gauge} 0\n".encode() if self.path == "/metrics" else b"{}"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length:
                    self.rfile.read(length)
                outer.posts.append(self.path)
                body = b'{"error": {"message": "Failed to reset KV cache"}}' if outer.post_status >= 400 else b"{}"
                self.send_response(outer.post_status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def vllm_engine():
    engine = _Engine("vllm")
    try:
        yield engine
    finally:
        engine.close()


@pytest.fixture()
def sglang_engine():
    engine = _Engine("sglang")
    try:
        yield engine
    finally:
        engine.close()


# --- the adapter ----------------------------------------------------------


def test_the_vllm_adapter_posts_the_honest_query(vllm_engine) -> None:
    VLLMAdapter(model_name="m", api_base=vllm_engine.url).flush_cache()
    assert vllm_engine.posts == [HONEST]


def test_the_sglang_adapter_posts_its_native_endpoint_with_the_deferred_query(sglang_engine) -> None:
    SGLangAdapter(model_name="m", api_base=sglang_engine.url).flush_cache()
    assert sglang_engine.posts == [SGLANG_FLUSH]


def test_a_declined_reset_is_a_typed_error_naming_the_status(vllm_engine) -> None:
    vllm_engine.post_status = 500
    with pytest.raises(EngineCapabilityUnavailableError) as exc:
        VLLMAdapter(model_name="m", api_base=vllm_engine.url).flush_cache()
    assert exc.value.capability == "flush_endpoint"
    assert "500" in str(exc.value)


def test_capabilities_say_which_engine_confirms_its_flush() -> None:
    assert VLLMAdapter(model_name="m").capabilities()["flush_confirms_reset"] is True
    # SGLang's 200 means flushed by the source read on 2026-10-08 (ADR-0149);
    # the capability stays absent (None) until a live window records it (S0F-55)
    assert SGLangAdapter(model_name="m").capabilities().get("flush_confirms_reset") is None
    assert LMDeployAdapter(model_name="m").capabilities().get("flush_confirms_reset") is None


def test_the_flush_path_stays_the_bare_endpoint_and_the_query_is_the_proxys() -> None:
    # the proxy matches the bare path and sends the same query to both roles
    assert VLLMAdapter._flush_endpoint == pd_proxy.RESET_PATH == "/reset_prefix_cache"
    assert VLLMAdapter._flush_query == pd_proxy.RESET_QUERY == "reset_running_requests=true"
    # ADR-0149 (2026-10-08): the original pin recorded that SGLang documented no
    # flush query; the installed 0.5.10.post1 accepts ?timeout=<s> (the flush is
    # deferred until the scheduler is fully idle), read from its source on the pod.
    assert SGLangAdapter._flush_query == "timeout=20"


# --- the runner's reset record --------------------------------------------


def test_strict_reset_records_that_the_engine_confirmed(vllm_engine) -> None:
    record = runner._reset_prefix_cache(vllm_engine.url, backend="vllm", model="m", strict=True)
    assert vllm_engine.posts == [HONEST]
    assert record["endpoint"] == "/reset_prefix_cache"
    assert record["mechanism"] == "adapter"
    assert record["verified"] is True  # the quiescence probe read zero
    assert record["engine_confirmed"] is True  # and the engine answered 200 in the honest mode


def test_strict_reset_refuses_a_reset_the_engine_declined(vllm_engine) -> None:
    vllm_engine.post_status = 500
    with pytest.raises(runner.CacheResetError, match="could not reset the vllm cache"):
        runner._reset_prefix_cache(vllm_engine.url, backend="vllm", model="m", strict=True)
    assert vllm_engine.posts == [HONEST]


def test_sglang_reset_is_recorded_as_not_confirmed_by_the_engine(sglang_engine) -> None:
    record = runner._reset_prefix_cache(sglang_engine.url, backend="sglang", model="m", strict=True)
    assert sglang_engine.posts == [SGLANG_FLUSH]
    assert record["verified"] is True
    assert record["engine_confirmed"] is None


def test_the_pilot_path_declined_reset_still_only_warns(vllm_engine, capsys) -> None:
    vllm_engine.post_status = 500
    record = runner._reset_prefix_cache(vllm_engine.url, backend="vllm", model="m")
    assert "WARNING: could not reset prefix cache" in capsys.readouterr().out
    assert record["engine_confirmed"] is None and record["verified"] is False


def test_the_record_always_carries_the_key() -> None:
    src = (REPO_ROOT / "scripts" / "3_run" / "run_experiment.py").read_text(encoding="utf-8")
    block = src[src.index("def _reset_prefix_cache("):src.index("def main():")]
    assert re.search(r'"engine_confirmed":\s*None,', block), "the base record declares the key"
    assert 'record["engine_confirmed"] = caps.get("flush_confirms_reset")' in block


# --- both halves together: the strict reset through the real proxy ---------


def test_strict_reset_through_the_proxy_wakes_and_resets_both_roles() -> None:
    prefill, decode = _Engine("vllm"), _Engine("vllm")
    proxy = pd_proxy.build_server(0, prefill.url, decode.url)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{proxy.server_address[1]}"
        record = runner._reset_prefix_cache(base, backend="vllm", model="m", strict=True)
        for role in (prefill, decode):
            assert role.posts == [pd_proxy.WAKE_PATH, HONEST]
        assert record["verified"] is True and record["engine_confirmed"] is True
        # a role that keeps declining refuses the window
        prefill.post_status = 500
        pd_proxy._RESET_RETRY_PAUSE_S, saved = 0.01, pd_proxy._RESET_RETRY_PAUSE_S
        try:
            with pytest.raises(runner.CacheResetError):
                runner._reset_prefix_cache(base, backend="vllm", model="m", strict=True)
        finally:
            pd_proxy._RESET_RETRY_PAUSE_S = saved
    finally:
        proxy.shutdown()
        proxy.server_close()
        prefill.close()
        decode.close()
