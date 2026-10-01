"""Strict cache reset refuses a backend whose adapter declares no flush endpoint (S0F-10, ADR-0131).

LMDeploy 0.17.0 has no cache-reset route and no engine-level reset (the
``/sleep`` route is a stub TurboMind marks broken; ``/distserve/free_cache``
calls a method TurboMind lacks) [fact sheet 2026-10-01]. The runner's
``_reset_prefix_cache`` used to fall through to the legacy
``POST <api_base>/reset_prefix_cache`` for any adapter without a declared flush
endpoint: against LMDeploy that is a 404, but with a dev-mode vLLM on the
default port 8000 the flush would hit vLLM and be logged as a verified cold
start for the LMDeploy window (reproduced locally by the fact-finding).

Pinned here:
1. strict (campaign) mode: a known adapter without a flush endpoint raises
   ``CacheResetError`` naming the engine, the missing endpoint and the ADR-0131
   decision (a cold window on LMDeploy needs an engine restart), BEFORE the
   quiescence probe and before any HTTP request;
2. the pilot (non-strict) path keeps its historical legacy fall-through
   (``tests/test_integration_wiring.py`` pins it), so nothing else changes;
3. a strict reset on a backend WITH a flush endpoint is untouched.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RUN_EXPERIMENT_PY = REPO_ROOT / "scripts" / "3_run" / "run_experiment.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_experiment_s0f10", RUN_EXPERIMENT_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


class _NoFlushAdapter:
    """An LMDeploy-shaped adapter: capabilities() declares no flush endpoint."""

    instances: List[Any] = []

    def __init__(self, model_name: str, api_base: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.api_base = api_base
        self.flushed = 0
        type(self).instances.append(self)

    def capabilities(self) -> Dict[str, Any]:
        return {"flush_endpoint": None, "streamed_ttft": True}

    def flush_cache(self) -> None:
        self.flushed += 1


class _FlushAdapter(_NoFlushAdapter):
    instances: List[Any] = []

    def capabilities(self) -> Dict[str, Any]:
        return {"flush_endpoint": "/reset_prefix_cache", "streamed_ttft": True}


def _forbid_http(monkeypatch) -> None:
    def _no(*a: Any, **k: Any) -> None:
        raise AssertionError(f"no HTTP request may be made: urlopen{a!r}")

    monkeypatch.setattr("urllib.request.urlopen", _no)


def test_strict_reset_refuses_an_adapter_without_a_flush_endpoint_before_any_http(
        monkeypatch) -> None:
    _NoFlushAdapter.instances = []
    monkeypatch.setattr(runner, "LMDeployAdapter", _NoFlushAdapter)
    monkeypatch.delenv("CAGE_LMDEPLOY_API_BASE", raising=False)
    _forbid_http(monkeypatch)
    with pytest.raises(runner.CacheResetError) as excinfo:
        runner._reset_prefix_cache("http://h:8000", backend="lmdeploy", model="m", strict=True)
    msg = str(excinfo.value)
    assert "lmdeploy" in msg
    assert "declares no flush endpoint" in msg
    assert "restart" in msg.lower() and "ADR-0131" in msg and "S0F-10" in msg
    assert "/reset_prefix_cache" in msg  # the legacy POST it refuses to fall through to
    assert _NoFlushAdapter.instances and _NoFlushAdapter.instances[0].flushed == 0


def test_strict_reset_refuses_the_turbomind_alias_too(monkeypatch) -> None:
    _NoFlushAdapter.instances = []
    monkeypatch.setattr(runner, "LMDeployAdapter", _NoFlushAdapter)
    monkeypatch.delenv("CAGE_LMDEPLOY_API_BASE", raising=False)
    _forbid_http(monkeypatch)
    with pytest.raises(runner.CacheResetError, match="declares no flush endpoint"):
        runner._reset_prefix_cache("http://h:8000", backend="lmdeploy-turbomind", model="m", strict=True)


def test_non_strict_reset_keeps_the_legacy_fall_through(monkeypatch, capsys) -> None:
    # The pilot path is unchanged (test_integration_wiring pins the POST); the
    # record says what happened and never claims verification.
    _NoFlushAdapter.instances = []
    monkeypatch.setattr(runner, "LMDeployAdapter", _NoFlushAdapter)
    monkeypatch.delenv("CAGE_LMDEPLOY_API_BASE", raising=False)
    posts: List[str] = []

    def _fake_urlopen(req: Any, timeout: Any = None) -> None:
        posts.append(req.full_url)

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    record = runner._reset_prefix_cache("http://h:8000", backend="lmdeploy", model="m")
    assert posts == ["http://h:8000/reset_prefix_cache"]
    assert record["mechanism"] == "legacy" and record["verified"] is False


@pytest.mark.parametrize("backend", ["hf-oracle", "gemini", "ollama"])
def test_strict_reset_refuses_a_backend_without_an_adapter_before_any_http(
        monkeypatch, backend) -> None:
    # Review finding (2026-10-01, LOW 1): the adapter map has no entry for the
    # in-process oracle or the legacy backends, so strict mode skipped the
    # ADR-0131 refusal and fell through to the legacy POST against --api-base,
    # the same hazard class (a dev-mode vLLM on 8000 flushed, this window
    # labeled cold). Campaign mode refuses these too, before any request.
    _forbid_http(monkeypatch)
    with pytest.raises(runner.CacheResetError) as excinfo:
        runner._reset_prefix_cache("http://h:8000", backend=backend, model="m", strict=True)
    msg = str(excinfo.value)
    assert backend in msg and "no cache-reset adapter" in msg
    assert "ADR-0131" in msg and "/reset_prefix_cache" in msg


def test_strict_reset_with_a_flush_endpoint_is_unchanged(monkeypatch) -> None:
    _FlushAdapter.instances = []
    monkeypatch.setattr(runner, "VLLMAdapter", _FlushAdapter)
    # the quiescence probe reads a zero gauge, then the adapter flush runs
    monkeypatch.setattr(
        runner, "_probe_running_requests", lambda api_base, backend: (0, "ok")
    )
    _forbid_http(monkeypatch)
    record = runner._reset_prefix_cache("http://h:8000", backend="vllm", model="m", strict=True)
    assert record["mechanism"] == "adapter" and record["verified"] is True
    assert _FlushAdapter.instances[0].flushed == 1
