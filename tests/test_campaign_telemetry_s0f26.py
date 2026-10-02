"""S0F-26 (ADR-0136), runner half: campaign telemetry that is proven.

S0 ran its cells with ``--vllm-telemetry`` and all 40 windows still read
``UNKNOWN_TELEMETRY`` (the clock bug S0F-15), with every gate green. A flag
on the command line proves nothing, so the runner now (1) refuses a campaign
cell on a sampled backend without the flag at run start, (2) asks each
telemetry endpoint for one snapshot after warm-up and refuses before the
measured stage when it carries no numeric KV usage gauge, (3) refuses when
the sampler cannot start, and (4) never samples for the in-process hf oracle
(there the sampler would dial the runner's default port and record whatever
engine listens on it).

The driver half (the flag on every server-engine cell, load_plan pins) lives
in tests/test_pd_launcher.py; the verifier half (check (m)) in
tests/test_verify_results_v2.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RUNNER_PY = REPO_ROOT / "scripts" / "3_run" / "run_experiment.py"

_spec = importlib.util.spec_from_file_location("run_experiment_s0f26", RUNNER_PY)
runner = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = runner
_spec.loader.exec_module(runner)

from src.monitoring import vllm_telemetry as vt  # noqa: E402


# --- the run-start refusal -------------------------------------------------


def test_the_sampled_backends_are_the_engines_with_a_dialect() -> None:
    assert runner.CAMPAIGN_TELEMETRY_BACKENDS == frozenset({"vllm", "sglang"})


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_a_campaign_cell_on_a_sampled_backend_needs_the_flag(backend: str) -> None:
    with pytest.raises(ValueError) as exc:
        runner.require_campaign_telemetry(True, backend, False)
    msg = str(exc.value)
    assert "CAMPAIGN TELEMETRY" in msg and "--vllm-telemetry" in msg and "ADR-0136" in msg
    runner.require_campaign_telemetry(True, backend, True)  # with the flag: no refusal


@pytest.mark.parametrize("backend", ["hf-oracle", "hf_oracle", "lmdeploy", "gemini", "ollama"])
def test_other_backends_carry_no_requirement(backend: str) -> None:
    runner.require_campaign_telemetry(True, backend, False)


def test_the_pilot_path_carries_no_requirement() -> None:
    runner.require_campaign_telemetry(False, "vllm", False)


# --- the dialect resolver ---------------------------------------------------


@pytest.mark.parametrize("backend", ["hf-oracle", "hf_oracle"])
def test_the_in_process_oracle_is_never_sampled(backend: str, capsys) -> None:
    assert runner.resolve_telemetry_dialect(backend) is None
    out = capsys.readouterr().out
    assert "LOUD SKIP" in out and backend in out


def test_the_served_engines_keep_their_dialects(capsys) -> None:
    assert runner.resolve_telemetry_dialect("vllm") == "vllm"
    assert runner.resolve_telemetry_dialect("sglang") == "sglang"
    assert runner.resolve_telemetry_dialect("lmdeploy") is None
    assert "lmdeploy" in capsys.readouterr().out


# --- the probe before the measured stage ------------------------------------


def _capture(result: Any):
    calls: List[Tuple[str, str]] = []

    def _fake(url: str, *, metrics_path: str = "/metrics", api_key: Any = None,
              interval: float = 1.0, dialect: str = "vllm") -> Any:
        calls.append((url, dialect))
        if isinstance(result, BaseException):
            raise result
        return result(url) if callable(result) else result

    return _fake, calls


def test_a_snapshot_with_a_numeric_gauge_passes(monkeypatch) -> None:
    fake, calls = _capture({"kv_usage": 0.0, "running": 0})
    monkeypatch.setattr(vt, "capture_snapshot", fake)
    runner.probe_campaign_telemetry([("single", "http://h:8000")], "vllm")
    assert calls == [("http://h:8000", "vllm")]


def test_every_endpoint_of_a_pd_pair_is_probed(monkeypatch) -> None:
    fake, calls = _capture({"kv_usage": 0.25})
    monkeypatch.setattr(vt, "capture_snapshot", fake)
    runner.probe_campaign_telemetry(
        [("prefill", "http://h:8100"), ("decode", "http://h:8200")], "vllm"
    )
    assert [url for url, _ in calls] == ["http://h:8100", "http://h:8200"]


@pytest.mark.parametrize(
    "snapshot",
    [
        None,                                   # unreachable, or no cage-stats at all
        {"spec_decode": {"x": 1}},              # the dependency-free fallback: no KV gauge
        {"kv_usage": None},                     # cage-stats multi-engine refusal / gauge missing
        {"kv_usage": True},                     # a bool is not a reading
        {"kv_usage": float("nan")},
        {"kv_usage": "0.1"},
    ],
    ids=["no-snapshot", "spec-only", "gauge-null", "bool", "nan", "string"],
)
def test_a_snapshot_without_a_numeric_gauge_refuses(monkeypatch, snapshot: Any) -> None:
    fake, _ = _capture(snapshot)
    monkeypatch.setattr(vt, "capture_snapshot", fake)
    with pytest.raises(RuntimeError) as exc:
        runner.probe_campaign_telemetry([("single", "http://h:8000")], "vllm")
    msg = str(exc.value)
    assert "CAMPAIGN TELEMETRY" in msg and "http://h:8000" in msg and "kv_usage" in msg
    assert "ADR-0136" in msg


def test_a_capture_that_raises_refuses_with_its_cause(monkeypatch) -> None:
    # the sglang dialect probe raises when the installed cage-stats predates it
    fake, _ = _capture(RuntimeError("telemetry dialect 'sglang' requires a cage-stats with SGLang dialect support"))
    monkeypatch.setattr(vt, "capture_snapshot", fake)
    with pytest.raises(RuntimeError, match="SGLang dialect support"):
        runner.probe_campaign_telemetry([("single", "http://h:30000")], "sglang")


def test_one_dead_role_is_named_and_the_healthy_one_is_not(monkeypatch) -> None:
    fake, _ = _capture(lambda url: {"kv_usage": 0.1} if url.endswith("8200") else None)
    monkeypatch.setattr(vt, "capture_snapshot", fake)
    with pytest.raises(RuntimeError) as exc:
        runner.probe_campaign_telemetry(
            [("prefill", "http://h:8100"), ("decode", "http://h:8200")], "vllm"
        )
    assert "prefill=http://h:8100" in str(exc.value)
    assert "decode=http://h:8200" not in str(exc.value)


# --- wiring (source order): where the three checks sit ----------------------


def _body(src: str, start: str, end: str) -> str:
    return src[src.index(start):src.index(end)]


def test_the_checks_sit_before_the_work_they_protect() -> None:
    src = RUNNER_PY.read_text(encoding="utf-8")
    run = _body(src, "def run_experiment(", "class CacheResetError(")
    # (1) the flag refusal: right after the endpoint parse, before any dataset load
    assert run.index("require_campaign_telemetry(") < run.index("loader = get_loader(")
    # (2) the probe: after the warm-up stages, before the samplers start and
    #     before the measured stage opens
    probe = run.index("probe_campaign_telemetry(")
    assert run.index('stage_name="Warmup")') < probe < run.index("stage_t_start = time.time()")
    assert probe < run.index("VllmTelemetrySampler(")
    # (3) a campaign sampler that cannot start refuses instead of printing
    start_block = run[run.index("elif _telemetry_dialect is not None:"):run.index("pd_transfer_start = (")]
    assert "if campaign_session is not None:" in start_block and "raise" in start_block
    # main() refuses before the trial loop (before the first engine contact)
    main = src[src.index("def main():"):]
    assert main.index("require_campaign_telemetry(") < main.index("def _run_with_top_k(")
