"""Wave-1 pins: simulated-transfer provenance, telemetry dialect, multi-GPU energy.

WHAT is pinned and WHY:

1. T3.3 — kv_transfer_params provenance. The ONLY producer of
   kv_transfer_params today is SimulatedKVCacheManager (router-attached,
   asyncio.sleep-faked latency), and run_experiment's distributed gate merely
   checks the metadata EXISTS — it would green-light simulation as
   measurement. Pinned: (a) the simulator and the router payload stamp every
   dict source=="simulated"; (b) in campaign mode enforce_campaign_
   transfer_provenance HARD-FAILS on source=="simulated", on a MISSING source
   (unknown provenance is not evidence — fail-closed), and on unparseable
   payloads, while accepting a real-connector stamp (source=="nixl"); (c) the
   gate is wired behind `campaign_session is not None` so pilot behavior is
   unchanged.

2. T4.3 — per-backend telemetry dialect. VllmTelemetrySampler must forward
   its dialect to capture_snapshot every tick (an SGLang server sampled with
   vllm-named families yields fabricated absence), must abort LOUDLY when the
   dialect-support probe raises (never spin silently collecting nothing), and
   run_experiment's resolve_telemetry_dialect must pick sglang for sglang,
   vllm otherwise, and LOUDLY skip (None) for LMDeploy — no cage-stats
   dialect exists for it and no evidence it serves vLLM-named metrics.

3. T4.4 — NVML energy under TP=N. The old index-0-only read covered 1/N
   GPUs. Pinned: _read_energy_mj sums across ALL nvmlDeviceGetCount()
   devices (the existing energy_mj field becomes that TP-correct SUM), emits
   an index-aligned per-GPU list, survives one bad handle without zeroing the
   rest, and stays (None, None) — never fabricated — when NVML is absent.
   aggregate() pins energy_delta_mj (sum semantics preserved) plus the new
   energy_delta_mj_per_gpu with honest None holes.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "3_run" / "run_experiment.py"


def _missing(name: str) -> bool:
    return name not in sys.modules and importlib.util.find_spec(name) is None


def _install_router_import_stubs() -> None:
    """Stub fastapi/pydantic so router.py imports in the lean venv.

    Import-surface only (decorators pass through, nothing is served) — the
    sys.modules-stub pattern of tests/test_cov_router_logic.py. A venv that
    really has the packages keeps them (find_spec wins).
    """
    if _missing("fastapi"):
        fastapi = types.ModuleType("fastapi")

        class HTTPException(Exception):
            def __init__(self, status_code: int, detail: str = ""):
                super().__init__(f"{status_code}: {detail}")
                self.status_code = status_code
                self.detail = detail

        class FastAPI:
            def __init__(self, *args, **kwargs):
                pass

            def _passthrough(self, *args, **kwargs):
                def decorator(fn):
                    return fn
                return decorator

            get = post = on_event = _passthrough

        fastapi.FastAPI = FastAPI
        fastapi.HTTPException = HTTPException
        responses = types.ModuleType("fastapi.responses")

        class _Response:
            def __init__(self, *args, **kwargs):
                pass

        responses.PlainTextResponse = _Response
        responses.StreamingResponse = _Response
        responses.JSONResponse = _Response
        fastapi.responses = responses
        sys.modules["fastapi"] = fastapi
        sys.modules["fastapi.responses"] = responses
    if _missing("pydantic"):
        pydantic = types.ModuleType("pydantic")

        class BaseModel:
            pass

        pydantic.BaseModel = BaseModel
        pydantic.ConfigDict = dict
        sys.modules["pydantic"] = pydantic


_install_router_import_stubs()

from src.monitoring import vllm_telemetry as vt  # noqa: E402
from src.orchestration.cache_manager import CacheNode, SimulatedKVCacheManager  # noqa: E402
from src.orchestration.router import PrefixAwareRouter, ReplicaConfig  # noqa: E402


def _load_runner():
    # scripts/3_run is not an importable package ("3_run" is not a valid
    # module name), so load by path — same pattern as test_integration_wiring.
    spec = importlib.util.spec_from_file_location("cage_run_experiment_wave1", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _nodes(n: int = 3) -> list:
    return [
        CacheNode(
            node_id=f"replica-{i}",
            host=f"http://localhost:{8000 + i}",
            port=0,
            vram_total=0,
            vram_used=0,
            cache_blocks_capacity=0,
            cache_blocks_used=0,
        )
        for i in range(1, n + 1)
    ]


def _offline_router(policy: str = "replicated") -> PrefixAwareRouter:
    router = PrefixAwareRouter(
        [
            ReplicaConfig(replica_id="replica-1", api_base="http://localhost:8001"),
            ReplicaConfig(replica_id="replica-2", api_base="http://localhost:8002"),
        ]
    )
    # Pre-seed tokenization so route_request_with_simulation never touches the
    # network (initialize_tokenizer would try the replicas' /v1/models).
    router.tokenization_mode = "utf8_fallback"
    router.tokenizer_name = "offline-utf8"
    if policy != "replicated":
        router.cache_manager.policy = policy
    return router


# --------------------------------------------------------------------------
# T3.3 (a): simulator + router payloads are stamped source=="simulated"
# --------------------------------------------------------------------------


def test_simulated_cache_manager_stamps_source_replicated():
    mgr = SimulatedKVCacheManager(_nodes(), policy="replicated")
    params = mgr.resolve_prefix([1, 2, 3, 4])
    assert params["source"] == "simulated"


def test_simulated_cache_manager_stamps_source_sharded_context():
    mgr = SimulatedKVCacheManager(_nodes(), policy="sharded_context")
    params = mgr.resolve_prefix(list(range(64)))
    assert params["source"] == "simulated"
    # The sharded path is the one that fakes a positive transfer cost — the
    # exact payload the campaign gate exists to refuse.
    assert params["transfer_bytes"] > 0


def test_router_payload_stamps_source_replicated():
    router = _offline_router("replicated")
    _, params = asyncio.run(router.route_request_with_simulation("ctx\n\nQ: hi"))
    assert params["source"] == "simulated"


def test_router_payload_stamps_source_sharded_context():
    router = _offline_router("sharded_context")
    _, params = asyncio.run(router.route_request_with_simulation("ctx\n\nQ: hi"))
    assert params["source"] == "simulated"


# --------------------------------------------------------------------------
# T3.3 (b): campaign-mode gate — refusal and acceptance paths
# --------------------------------------------------------------------------


def _row(payload, example_id: str = "q1") -> dict:
    if isinstance(payload, dict):
        payload = json.dumps(payload, sort_keys=True)
    return {"example_id": example_id, "kv_transfer_params": payload}


def test_campaign_gate_refuses_simulated_source():
    rows = [
        _row({"source": "nixl", "transfer_bytes": 10}),
        _row({"source": "simulated", "transfer_bytes": 0}, example_id="q2"),
    ]
    with pytest.raises(RuntimeError, match="simulated"):
        runner.enforce_campaign_transfer_provenance(rows)


def test_campaign_gate_refuses_missing_source():
    rows = [_row({"transfer_bytes": 10, "transfer_latency_ms": 1.5})]
    with pytest.raises(RuntimeError, match="source"):
        runner.enforce_campaign_transfer_provenance(rows)


def test_campaign_gate_refuses_unparseable_payload():
    rows = [_row("not-json{")]
    with pytest.raises(RuntimeError, match="[Uu]nparseable"):
        runner.enforce_campaign_transfer_provenance(rows)


def test_campaign_gate_refuses_non_object_payload():
    rows = [_row(json.dumps(["simulated"]))]
    with pytest.raises(RuntimeError, match="not an"):
        runner.enforce_campaign_transfer_provenance(rows)


@pytest.mark.parametrize("variant", ["SIMULATED", "Simulated", "  simulated  "])
def test_campaign_gate_refuses_case_whitespace_variants(variant):
    """Normalization pin: a case/whitespace variant of "simulated" must never
    pass as real provenance (2026-08-30 verifier minor)."""
    rows = [_row({"source": variant, "transfer_bytes": 0})]
    with pytest.raises(RuntimeError, match="simulated"):
        runner.enforce_campaign_transfer_provenance(rows)


def test_campaign_gate_refuses_unrecognized_source():
    """Allowlist pin: an unknown non-simulated stamp is unknown provenance,
    not evidence — only _REAL_TRANSFER_SOURCES pass."""
    rows = [_row({"source": "mystery-connector", "transfer_bytes": 10})]
    with pytest.raises(RuntimeError, match="unrecognized source"):
        runner.enforce_campaign_transfer_provenance(rows)
    assert runner._REAL_TRANSFER_SOURCES == frozenset({"nixl"}), (
        "allowlist grows ONLY with a real connector integration (Wave-3 T4.5)"
    )


def test_campaign_gate_accepts_real_connector_source():
    rows = [
        _row({"source": "nixl", "transfer_bytes": 4096}),
        # Already-parsed dict rows must be accepted too (synthetic callers).
        {"example_id": "q2", "kv_transfer_params": {"source": "nixl"}},
        # Empty rows are NOT this gate's job: whole-run absence already
        # hard-fails in validate_distributed_artifacts.
        {"example_id": "q3", "kv_transfer_params": ""},
        {"example_id": "q4"},
    ]
    runner.enforce_campaign_transfer_provenance(rows)  # must not raise


# --------------------------------------------------------------------------
# T3.3 (c): gate wired campaign-only, inside the distributed block
# --------------------------------------------------------------------------


def test_campaign_gate_wired_behind_campaign_session():
    src_text = inspect.getsource(runner.run_experiment)
    call_idx = src_text.index("enforce_campaign_transfer_provenance(results)")
    guard_idx = src_text.rindex("if campaign_session is not None:", 0, call_idx)
    # The guard must be the statement immediately governing the call (comment
    # lines between them only), and both must sit inside the distributed block.
    assert call_idx - guard_idx < 500
    dist_idx = src_text.rindex('== "distributed"', 0, guard_idx)
    assert dist_idx < guard_idx < call_idx


# --------------------------------------------------------------------------
# T4.3: sampler dialect pass-through + fail-loud abort
# --------------------------------------------------------------------------


def _run_sampler_one_tick(monkeypatch, sampler, capture_impl):
    monkeypatch.setattr(vt, "capture_snapshot", capture_impl)
    monkeypatch.setattr(
        vt.VllmTelemetrySampler, "_read_energy_mj", lambda self: (None, None)
    )
    sampler._run()  # runs inline (no thread): the fake stops it after 1 tick


def test_sampler_forwards_sglang_dialect(monkeypatch):
    sampler = vt.VllmTelemetrySampler("http://localhost:9", dialect="sglang")
    seen = []

    def fake_capture(url, *, metrics_path="/metrics", api_key=None, interval=1.0,
                     dialect="vllm"):
        seen.append((url, metrics_path, dialect))
        sampler._stop.set()
        return {"connected": True}

    _run_sampler_one_tick(monkeypatch, sampler, fake_capture)
    assert seen == [("http://localhost:9", "/metrics", "sglang")]
    assert len(sampler._samples) == 1


def test_sampler_default_dialect_is_vllm(monkeypatch):
    sampler = vt.VllmTelemetrySampler("http://localhost:9")
    seen = []

    def fake_capture(url, *, metrics_path="/metrics", api_key=None, interval=1.0,
                     dialect="vllm"):
        seen.append(dialect)
        sampler._stop.set()
        return {"connected": True}

    _run_sampler_one_tick(monkeypatch, sampler, fake_capture)
    assert seen == ["vllm"]


def test_sampler_aborts_loudly_when_dialect_probe_raises(monkeypatch, capsys):
    # capture_snapshot handles flaky-network internally (returns None); a
    # RAISE is its fail-loud dialect-support probe. The sampler must surface
    # it and stop — spinning silently would launder "unsupported dialect"
    # into an empty-but-plausible telemetry absence.
    sampler = vt.VllmTelemetrySampler("http://localhost:9", dialect="sglang")

    def fake_capture(url, *, metrics_path="/metrics", api_key=None, interval=1.0,
                     dialect="vllm"):
        raise RuntimeError("telemetry dialect 'sglang' requires a cage-stats with support")

    _run_sampler_one_tick(monkeypatch, sampler, fake_capture)
    assert sampler._samples == []
    out = capsys.readouterr().out
    assert "sampler aborted" in out
    assert "sglang" in out


# --------------------------------------------------------------------------
# T4.3: run_experiment backend -> dialect resolution (incl. LMDeploy skip)
# --------------------------------------------------------------------------


def test_resolve_telemetry_dialect_sglang_and_vllm():
    assert runner.resolve_telemetry_dialect("sglang") == "sglang"
    assert runner.resolve_telemetry_dialect("vllm") == "vllm"
    # Non-lmdeploy backends keep today's vllm-named sampling behavior.
    assert runner.resolve_telemetry_dialect("ollama") == "vllm"


@pytest.mark.parametrize("backend", ["lmdeploy", "lmdeploy-turbomind"])
def test_resolve_telemetry_dialect_lmdeploy_skips_loudly(backend, capsys):
    assert runner.resolve_telemetry_dialect(backend) is None
    out = capsys.readouterr().out
    assert "LOUD SKIP" in out
    assert backend in out
    assert "ABSENT" in out


# --------------------------------------------------------------------------
# T4.4: NVML energy across ALL GPUs
# --------------------------------------------------------------------------


def _install_fake_pynvml(monkeypatch, energies, bad_read_indices=frozenset(),
                         bad_handle_indices=frozenset()):
    mod = types.ModuleType("pynvml")
    mod.nvmlInit = lambda: None
    mod.nvmlDeviceGetCount = lambda: len(energies)

    def get_handle(i):
        if i in bad_handle_indices:
            raise RuntimeError(f"NVML handle {i} unavailable")
        return f"handle-{i}"

    def get_energy(handle):
        idx = int(str(handle).rsplit("-", 1)[1])
        if idx in bad_read_indices:
            raise RuntimeError(f"NVML read failed on GPU {idx}")
        return energies[idx]

    mod.nvmlDeviceGetHandleByIndex = get_handle
    mod.nvmlDeviceGetTotalEnergyConsumption = get_energy
    monkeypatch.setitem(sys.modules, "pynvml", mod)
    return mod


def test_energy_sums_across_all_gpus(monkeypatch):
    _install_fake_pynvml(monkeypatch, [1000.0, 2000.0, 3000.0])
    sampler = vt.VllmTelemetrySampler("http://localhost:9")
    total, per_gpu = sampler._read_energy_mj()
    assert total == 6000.0
    assert per_gpu == [1000.0, 2000.0, 3000.0]


def test_energy_one_bad_read_does_not_zero_the_rest(monkeypatch):
    _install_fake_pynvml(monkeypatch, [1000.0, 2000.0, 3000.0],
                         bad_read_indices={1})
    sampler = vt.VllmTelemetrySampler("http://localhost:9")
    total, per_gpu = sampler._read_energy_mj()
    assert total == 4000.0  # sum of the devices that answered
    assert per_gpu == [1000.0, None, 3000.0]  # honest hole, never a zero
    assert sampler._nvml_failed is False  # one bad read must not kill the probe


def test_energy_one_bad_handle_does_not_zero_the_rest(monkeypatch):
    _install_fake_pynvml(monkeypatch, [1000.0, 2000.0, 3000.0],
                         bad_handle_indices={0})
    sampler = vt.VllmTelemetrySampler("http://localhost:9")
    total, per_gpu = sampler._read_energy_mj()
    assert total == 5000.0
    assert per_gpu == [None, 2000.0, 3000.0]


def test_energy_absent_pynvml_stays_graceful(monkeypatch):
    # A None entry in sys.modules makes `import pynvml` raise ImportError even
    # when the real package is installed — the pre-existing graceful path.
    monkeypatch.setitem(sys.modules, "pynvml", None)
    sampler = vt.VllmTelemetrySampler("http://localhost:9")
    assert sampler._read_energy_mj() == (None, None)
    assert sampler._nvml_failed is True
    # Permanently disabled: subsequent ticks stay absent, never fabricated.
    assert sampler._read_energy_mj() == (None, None)


def test_aggregate_emits_sum_delta_and_per_gpu_deltas():
    sampler = vt.VllmTelemetrySampler("http://localhost:9")
    sampler._samples = [
        {"connected": True, "energy_mj": 6000.0,
         "energy_mj_per_gpu": [1000.0, 2000.0, 3000.0]},
        {"connected": True, "energy_mj": 7500.0,
         "energy_mj_per_gpu": [1500.0, None, 3600.0]},
    ]
    sampler._sample_ts = [1.0, 2.0]
    agg = sampler.aggregate()
    # Existing field name preserved; value is now the TP-correct SUM delta.
    assert agg["energy_delta_mj"] == 1500.0
    # Per-GPU deltas: only where BOTH endpoint ticks answered for that index.
    assert agg["energy_delta_mj_per_gpu"] == [500.0, None, 600.0]
