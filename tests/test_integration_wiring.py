"""Integration-wiring tests for scripts/3_run/run_experiment.py (2026-08-04).

Covers the two integrator features:

1. Backend switch: ``setup_inference_engine`` dispatches the new backend
   tokens (sglang, lmdeploy / lmdeploy-turbomind, hf-oracle / hf_oracle) to
   the right adapter class with the documented constructor kwargs and env-var
   overrides (CAGE_SGLANG_API_BASE / CAGE_LMDEPLOY_API_BASE /
   CAGE_ADAPTER_MAX_RETRIES / CAGE_*_CHAT_TEMPLATE_KWARGS / CAGE_HF_DEVICE /
   CAGE_HF_DTYPE), keeps vllm the default-behavior path, and still fails
   closed on unknown backends.

2. Open-loop workload mode: ``dispatch_open_loop`` drives
   src/orchestration/load_generator.py through a stub async engine (no
   network) and produces per-request rows carrying ``scheduled_ts``,
   ``actual_send_ts`` and ``scheduler_lag_ms``; ``merge_open_loop_row``
   appends those columns after the existing results columns without
   disturbing them; ``run_experiment`` refuses a mis-specified open-loop
   configuration with the typed LoadGeneratorError BEFORE any engine or
   dataset work.

All adapter classes are monkeypatched at the runner-module seam (the pattern
of tests/test_inference.py: mock the transport, never a live server/GPU).

The runner module is loaded via importlib (like tests/test_review_fixes_core
.py) because its directory name "3_run" is not a valid package component.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "3_run" / "run_experiment.py"
PREFLIGHT_PATH = REPO_ROOT / "scripts" / "checks" / "preflight_check.sh"

from src.data.loader import CAGExample  # noqa: E402
from src.inference.engine import InferenceRequest, InferenceResponse  # noqa: E402
from src.orchestration.ir import BM25IRIndex, IRDocument, IRHit  # noqa: E402
from src.orchestration.load_generator import LoadGeneratorError  # noqa: E402


def _load_runner():
    spec = importlib.util.spec_from_file_location("cage_run_experiment_wiring", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


# --------------------------------------------------------------------------
# Fakes (transport-free adapter stand-ins, recording constructor kwargs)
# --------------------------------------------------------------------------


class _RecordingAdapter:
    """Constructor-recording fake for the HTTP adapter classes."""

    loaded_model_override: Optional[str] = "__match__"  # "__match__" -> echo model_name

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.kwargs = kwargs

    def is_ready(self) -> bool:
        return True

    def get_loaded_model(self) -> Optional[str]:
        if self.loaded_model_override == "__match__":
            return self.model_name
        return self.loaded_model_override


class _FakeSGLang(_RecordingAdapter):
    pass


class _FakeLMDeploy(_RecordingAdapter):
    pass


class _FakeVLLM(_RecordingAdapter):
    pass


class _FakeOllama(_RecordingAdapter):
    pass


class _FakeHFOracle:
    """In-process oracle fake (no get_loaded_model -- server check is N/A)."""

    def __init__(self, model_name: str, device: str = "auto", dtype: str = "bfloat16") -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

    def is_ready(self) -> bool:
        return True


@pytest.fixture()
def patched_adapters(monkeypatch):
    monkeypatch.setattr(runner, "SGLangAdapter", _FakeSGLang)
    monkeypatch.setattr(runner, "LMDeployAdapter", _FakeLMDeploy)
    monkeypatch.setattr(runner, "HFOracleAdapter", _FakeHFOracle)
    monkeypatch.setattr(runner, "VLLMAdapter", _FakeVLLM)
    monkeypatch.setattr(runner, "OllamaAdapter", _FakeOllama)
    # Clean env: adapter-affecting variables must not leak between tests.
    for var in (
        "CAGE_SGLANG_API_BASE",
        "CAGE_LMDEPLOY_API_BASE",
        "CAGE_SGLANG_CHAT_TEMPLATE_KWARGS",
        "CAGE_LMDEPLOY_CHAT_TEMPLATE_KWARGS",
        "CAGE_VLLM_CHAT_TEMPLATE_KWARGS",
        "CAGE_ADAPTER_MAX_RETRIES",
        "CAGE_HF_DEVICE",
        "CAGE_HF_DTYPE",
    ):
        monkeypatch.delenv(var, raising=False)
    yield monkeypatch


def _baseline_config(api_base: str = "http://localhost:8000") -> SimpleNamespace:
    return SimpleNamespace(api_base=api_base, enable_prefix_caching=False)


# --------------------------------------------------------------------------
# 1. Backend switch dispatch
# --------------------------------------------------------------------------


def test_backend_sglang_constructs_sglang_adapter(patched_adapters):
    engine = runner.setup_inference_engine(
        "m/one", _baseline_config("http://host:9"), backend="sglang"
    )
    assert isinstance(engine, _FakeSGLang)
    assert engine.model_name == "m/one"
    # Falls back to baseline_config.api_base when the env override is unset.
    assert engine.kwargs["api_base"] == "http://host:9"


def test_backend_sglang_env_api_base_override_wins(patched_adapters):
    patched_adapters.setenv("CAGE_SGLANG_API_BASE", "http://sglang-host:30000")
    engine = runner.setup_inference_engine(
        "m/one", _baseline_config("http://host:9"), backend="sglang"
    )
    assert engine.kwargs["api_base"] == "http://sglang-host:30000"


@pytest.mark.parametrize("token", ["lmdeploy", "lmdeploy-turbomind"])
def test_backend_lmdeploy_tokens_construct_lmdeploy_adapter(patched_adapters, token):
    patched_adapters.setenv("CAGE_LMDEPLOY_API_BASE", "http://lmdeploy-host:23333")
    engine = runner.setup_inference_engine(
        "m/two", _baseline_config(), backend=token
    )
    assert isinstance(engine, _FakeLMDeploy)
    assert engine.kwargs["api_base"] == "http://lmdeploy-host:23333"


def test_backend_adapter_env_extras_forwarded(patched_adapters):
    patched_adapters.setenv("CAGE_ADAPTER_MAX_RETRIES", "2")
    patched_adapters.setenv(
        "CAGE_SGLANG_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": false}'
    )
    engine = runner.setup_inference_engine(
        "m/one", _baseline_config(), backend="sglang"
    )
    assert engine.kwargs["max_retries"] == 2
    assert engine.kwargs["chat_template_kwargs"] == {"enable_thinking": False}


def test_backend_chat_template_kwargs_bad_json_fails_closed(patched_adapters):
    patched_adapters.setenv("CAGE_SGLANG_CHAT_TEMPLATE_KWARGS", "{not json")
    with pytest.raises(ValueError, match="CAGE_SGLANG_CHAT_TEMPLATE_KWARGS"):
        runner.setup_inference_engine("m/one", _baseline_config(), backend="sglang")


def test_backend_chat_template_kwargs_non_object_fails_closed(patched_adapters):
    patched_adapters.setenv("CAGE_LMDEPLOY_CHAT_TEMPLATE_KWARGS", "[1, 2]")
    with pytest.raises(ValueError, match="JSON object"):
        runner.setup_inference_engine("m/one", _baseline_config(), backend="lmdeploy")


def test_backend_sglang_strict_model_mismatch_raises(patched_adapters):
    patched_adapters.setattr(_FakeSGLang, "loaded_model_override", "other/model")
    with pytest.raises(RuntimeError, match="Model mismatch"):
        runner.setup_inference_engine("m/one", _baseline_config(), backend="sglang")


@pytest.mark.parametrize("token", ["hf-oracle", "hf_oracle"])
def test_backend_hf_oracle_constructs_in_process_oracle(patched_adapters, token):
    engine = runner.setup_inference_engine("m/ref", _baseline_config(), backend=token)
    assert isinstance(engine, _FakeHFOracle)
    assert engine.device == "auto"
    assert engine.dtype == "bfloat16"


def test_backend_hf_oracle_env_device_dtype(patched_adapters):
    patched_adapters.setenv("CAGE_HF_DEVICE", "cpu")
    patched_adapters.setenv("CAGE_HF_DTYPE", "float32")
    engine = runner.setup_inference_engine(
        "m/ref", _baseline_config(), backend="hf-oracle"
    )
    assert engine.device == "cpu"
    assert engine.dtype == "float32"


def test_backend_vllm_default_path_preserved(patched_adapters):
    engine = runner.setup_inference_engine(
        "m/base", _baseline_config("http://host:8000"), backend="vllm"
    )
    assert isinstance(engine, _FakeVLLM)
    assert engine.kwargs["api_base"] == "http://host:8000"


def test_backend_vllm_honors_adapter_max_retries_env(patched_adapters):
    """CAGE_ADAPTER_MAX_RETRIES is not backend-scoped: the primary (vllm)
    backend must honor it too, not silently keep the 0-retry default."""
    patched_adapters.setenv("CAGE_ADAPTER_MAX_RETRIES", "3")
    engine = runner.setup_inference_engine(
        "m/base", _baseline_config(), backend="vllm"
    )
    assert isinstance(engine, _FakeVLLM)
    assert engine.kwargs["max_retries"] == 3


def test_backend_vllm_without_env_keeps_default_kwargs(patched_adapters):
    engine = runner.setup_inference_engine(
        "m/base", _baseline_config(), backend="vllm"
    )
    assert "max_retries" not in engine.kwargs  # adapter default (0) preserved


def test_backend_vllm_chat_template_kwargs_env_fails_closed(patched_adapters):
    # VLLMAdapter pins its own verified chat_template_kwargs (ADR-0007);
    # an env override must be refused loudly, never silently dropped.
    patched_adapters.setenv(
        "CAGE_VLLM_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": true}'
    )
    with pytest.raises(ValueError, match="CAGE_VLLM_CHAT_TEMPLATE_KWARGS"):
        runner.setup_inference_engine("m/base", _baseline_config(), backend="vllm")


def test_backend_ollama_legacy_path_untouched(patched_adapters):
    engine = runner.setup_inference_engine(
        "m/oll", _baseline_config("http://host:11434"), backend="ollama", strict=False
    )
    assert isinstance(engine, _FakeOllama)
    assert engine.kwargs["api_base"] == "http://host:11434"


def test_backend_unknown_still_raises(patched_adapters):
    with pytest.raises(ValueError, match="Unsupported backend"):
        runner.setup_inference_engine("m/one", _baseline_config(), backend="tgi")


# --------------------------------------------------------------------------
# 2. Open-loop dispatch (stub async engine, tiny schedule, no network)
# --------------------------------------------------------------------------


def _stub_response(request: InferenceRequest) -> InferenceResponse:
    return InferenceResponse(
        request_id=request.request_id,
        generated_text="stub answer",
        ttft_ms=1.0,
        total_time_ms=2.0,
        num_tokens=2,
        model_name="stub-model",
        finish_reason="stop",
    )


class _StubAsyncEngine:
    """Async-STREAMING-capable stub engine (the tests/test_inference.py
    mocked-transport pattern: the transport is replaced entirely, nothing
    leaves the process). Mirrors ``OpenAIChatAdapter.async_stream_generate``:
    invokes ``on_first_token`` once at the (simulated) first content delta."""

    def __init__(self, fail_request_ids: Optional[set] = None) -> None:
        self.calls: List[str] = []
        self.fail_request_ids = fail_request_ids or set()

    async def async_stream_generate(
        self, request: InferenceRequest, *, on_first_token=None
    ) -> InferenceResponse:
        self.calls.append(request.request_id)
        if request.request_id in self.fail_request_ids:
            raise ValueError(f"boom on {request.request_id}")
        if on_first_token is not None:
            on_first_token()
        return _stub_response(request)


class _NoAsyncEngine:
    """Closed-loop-only engine: must be refused, never silently degraded."""


class _NonStreamingAsyncEngine:
    """Async but NON-streaming engine: its ttft_ms is the full response time
    ('full-response-proxy'), so open-loop dispatch must refuse it rather than
    silently mix TTFT methodologies (charter D6 §6.3)."""

    async def async_generate(self, request: InferenceRequest) -> InferenceResponse:
        return _stub_response(request)


def _requests(n: int) -> List[InferenceRequest]:
    return [
        InferenceRequest(prompt=f"p{i}", max_tokens=4, temperature=0.0, request_id=f"r{i}")
        for i in range(n)
    ]


def test_dispatch_open_loop_rows_carry_the_three_new_fields():
    engine = _StubAsyncEngine()
    kept, report = runner.dispatch_open_loop(
        engine, _requests(3), rate_qps=200.0, seed=7, n_arrivals=3
    )
    assert len(kept) == 3
    assert report.n_scheduled == 3
    assert report.n_completed == 3
    for record in kept:
        row = record.to_row()
        # The three columns the results CSV must gain (D6 section 6.3).
        assert isinstance(row["scheduled_ts"], float)
        assert isinstance(row["actual_send_ts"], float)
        assert isinstance(row["scheduler_lag_ms"], float)
        assert row["scheduler_lag_ms"] >= 0.0
        assert record.result is not None
        assert record.result.generated_text == "stub answer"


def test_dispatch_open_loop_maps_schedule_index_modulo_requests(monkeypatch):
    # 5 arrivals over 2 requests is a deliberate replay: the E4 measured-window
    # replay guard refuses it unless CAGE_ALLOW_REPLAY=1 (labeled
    # non-confirmatory) — this test pins the modulo mapping on that path.
    monkeypatch.setenv("CAGE_ALLOW_REPLAY", "1")
    engine = _StubAsyncEngine()
    kept, _report = runner.dispatch_open_loop(
        engine, _requests(2), rate_qps=200.0, seed=3, n_arrivals=5
    )
    assert len(kept) == 5
    # Index i serves requests[i % 2]: three r0 sends, two r1 sends.
    assert sorted(engine.calls) == ["r0", "r0", "r0", "r1", "r1"]


def test_dispatch_open_loop_warmup_trim_drops_early_arrivals():
    engine = _StubAsyncEngine()
    # Deterministic spacing: arrivals at 0.1, 0.2, 0.3, 0.4, 0.5 s.
    kept, report = runner.dispatch_open_loop(
        engine,
        _requests(5),
        rate_qps=10.0,
        seed=1,
        n_arrivals=5,
        warmup_s=0.25,
        distribution="deterministic",
    )
    assert report.n_scheduled == 5  # the full offered schedule stays in the report
    assert len(kept) == 3  # 0.3, 0.4, 0.5 survive the Jain warmup trim
    assert all(r.scheduled_offset_s >= 0.25 for r in kept)


def test_dispatch_open_loop_send_failures_are_data_not_aborts():
    engine = _StubAsyncEngine(fail_request_ids={"r1"})
    kept, report = runner.dispatch_open_loop(
        engine, _requests(3), rate_qps=200.0, seed=5, n_arrivals=3
    )
    assert report.n_errors == 1
    failed = [r for r in kept if r.error is not None]
    assert len(failed) == 1
    assert "ValueError" in failed[0].error
    assert failed[0].result is None
    # The failed arrival still carries its open-loop accounting row.
    assert failed[0].to_row()["dispatch_error"] == failed[0].error


def test_dispatch_open_loop_requires_async_capable_engine():
    with pytest.raises(LoadGeneratorError, match="async_stream_generate"):
        runner.dispatch_open_loop(
            _NoAsyncEngine(), _requests(1), rate_qps=10.0, seed=1, n_arrivals=1
        )


def test_dispatch_open_loop_refuses_non_streaming_async_engine():
    """An engine exposing only the non-streaming async_generate must be
    REFUSED: its ttft_ms is the full response time, and scoring D6 attainment
    with it against the streamed single-stream baseline would silently mix
    TTFT methodologies (the failure class the batch_generate stream=True
    regression test guards on the closed-loop path)."""
    with pytest.raises(LoadGeneratorError, match="TTFT methodologies"):
        runner.dispatch_open_loop(
            _NonStreamingAsyncEngine(),
            _requests(1),
            rate_qps=10.0,
            seed=1,
            n_arrivals=1,
        )


def test_dispatch_open_loop_rows_carry_streamed_ttft_columns():
    """The streaming stub invokes on_first_token, so every completed row must
    carry first_token_ts plus BOTH TTFT columns (ttft_from_send_ms and the
    §6.3 coordinated-omission-corrected ttft_from_scheduled_ms)."""
    engine = _StubAsyncEngine()
    kept, _report = runner.dispatch_open_loop(
        engine, _requests(2), rate_qps=200.0, seed=9, n_arrivals=2
    )
    assert len(kept) == 2
    for record in kept:
        row = record.to_row()
        assert isinstance(row["first_token_ts"], float)
        assert isinstance(row["ttft_from_send_ms"], float)
        assert isinstance(row["ttft_from_scheduled_ms"], float)
        assert row["ttft_from_send_ms"] >= 0.0
        # From the INTENDED arrival: includes scheduler lag by construction.
        assert row["ttft_from_scheduled_ms"] == pytest.approx(
            row["ttft_from_send_ms"] + row["scheduler_lag_ms"]
        )


def test_dispatch_open_loop_requires_prepared_requests():
    with pytest.raises(LoadGeneratorError, match="at least one prepared request"):
        runner.dispatch_open_loop(
            _StubAsyncEngine(), [], rate_qps=10.0, seed=1, n_arrivals=1
        )


# --------------------------------------------------------------------------
# 3. Row merging + CSV columns (existing columns undisturbed, new appended)
# --------------------------------------------------------------------------


def _one_record() -> Any:
    engine = _StubAsyncEngine()
    kept, _ = runner.dispatch_open_loop(
        engine, _requests(1), rate_qps=200.0, seed=11, n_arrivals=1
    )
    return kept[0]


def test_merge_open_loop_row_appends_after_existing_columns():
    record = _one_record()
    row = {"example_id": "e1", "baseline": "b", "ttft_ms": 12.5, "error": None}
    existing = list(row)
    merged = runner.merge_open_loop_row(row, record)
    assert merged is row
    # Existing columns keep their order and values...
    assert list(merged)[: len(existing)] == existing
    assert merged["ttft_ms"] == 12.5
    # ...and the open-loop columns are appended after them.
    for key in ("scheduled_ts", "actual_send_ts", "scheduler_lag_ms"):
        assert key in merged
        assert list(merged).index(key) >= len(existing)


def test_merge_open_loop_row_collision_fails_closed():
    record = _one_record()
    with pytest.raises(ValueError, match="overwrite existing result columns"):
        runner.merge_open_loop_row({"scheduled_ts": 1.0}, record)


def test_results_csv_writer_gains_open_loop_columns():
    """The union-of-keys DictWriter convention picks up the merged columns."""
    record = _one_record()
    plain_row = {"example_id": "e0", "baseline": "b", "ttft_ms": 3.0}
    merged_row = runner.merge_open_loop_row(
        {"example_id": "e1", "baseline": "b", "ttft_ms": 4.0}, record
    )
    results = [plain_row, merged_row]
    fieldnames = list(dict.fromkeys(k for r in results for k in r))
    # Existing columns first, new columns appended -- nothing disturbed.
    assert fieldnames[:3] == ["example_id", "baseline", "ttft_ms"]
    for key in ("scheduled_ts", "actual_send_ts", "scheduler_lag_ms"):
        assert key in fieldnames
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)  # narrow row + merged row both serialize cleanly
    header = buf.getvalue().splitlines()[0].split(",")
    assert "scheduled_ts" in header
    assert "actual_send_ts" in header
    assert "scheduler_lag_ms" in header


# --------------------------------------------------------------------------
# 4. run_experiment open-loop configuration fails closed BEFORE any work
# --------------------------------------------------------------------------


def _run_experiment_kwargs(**overrides: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = dict(
        baseline="no_cache",
        model="stub-model",
        dataset="squad_v2",
        num_queries=1,
        max_tokens=4,
        api_base="http://localhost:8000",
        use_offline=False,
        output_dir="/nonexistent/never-created",
        seed=1,
        backend="vllm",
        top_k=1,
        embedding_model="stub-embed",
        ir_index_dir="/nonexistent",
        rebuild_ir_index=False,
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        redis_key_prefix="cage",
        redis_ttl_seconds=None,
        flush_redis_namespace=False,
        repeat_queries=1,
        warmup_queries=0,
        workload_mode="open_loop",
        batch_size=1,
        multi_turn_length=1,
        routing_switch_at=None,
        reranker_model=None,
        reranker_device="cpu",
        truncate_prompt_tokens=None,
        max_context_chars=None,
        max_context_docs=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_run_experiment_open_loop_without_rate_fails_closed():
    with pytest.raises(LoadGeneratorError, match="open_loop_rate_qps"):
        runner.run_experiment(**_run_experiment_kwargs(open_loop_duration_s=10.0))


def test_run_experiment_open_loop_duration_xor_count_fails_closed():
    # Neither duration nor count.
    with pytest.raises(LoadGeneratorError, match="exactly one"):
        runner.run_experiment(**_run_experiment_kwargs(open_loop_rate_qps=1.0))
    # Both duration and count.
    with pytest.raises(LoadGeneratorError, match="exactly one"):
        runner.run_experiment(
            **_run_experiment_kwargs(
                open_loop_rate_qps=1.0,
                open_loop_duration_s=10.0,
                open_loop_num_arrivals=5,
            )
        )


def test_run_experiment_closed_loop_defaults_do_not_require_open_loop_args():
    """Default (closed-loop) callers never hit the open-loop validation.

    The call still fails later for unrelated reasons in this offline test
    environment, but it must NOT be the open-loop typed error -- proving the
    new parameters are inert for every existing caller.
    """
    try:
        runner.run_experiment(
            **_run_experiment_kwargs(workload_mode="single", output_dir="/nonexistent/x")
        )
    except LoadGeneratorError as exc:  # pragma: no cover - failure branch
        pytest.fail(f"closed-loop path hit open-loop validation: {exc}")
    except Exception:
        pass  # unrelated offline failure (no dataset/server) is expected


# --------------------------------------------------------------------------
# 5. --retriever wiring (BM25 / hybrid-RRF stage-tagged paths; dense default)
# --------------------------------------------------------------------------


def _ir_docs() -> List[IRDocument]:
    return [
        IRDocument(doc_id="d1", text="alpha beta gamma", metadata={}),
        IRDocument(doc_id="d2", text="delta epsilon zeta", metadata={}),
        IRDocument(doc_id="d3", text="eta theta iota", metadata={}),
    ]


class _FakeDenseIndex:
    """Duck-typed dense index (search + resolve_hits) with a fixed ranking."""

    def __init__(self, docs: List[IRDocument], order: List[str]) -> None:
        self._docs = {d.doc_id: d for d in docs}
        self._order = order

    def search(self, query: str, *, top_k: int = 5) -> List[IRHit]:
        return [
            IRHit(doc_id=doc_id, score=1.0 / (rank + 1))
            for rank, doc_id in enumerate(self._order[:top_k])
        ]

    def resolve_hits(self, hits: Any) -> List[IRDocument]:
        return [self._docs[h.doc_id] for h in hits if h.doc_id in self._docs]


def test_retriever_choices_are_pinned():
    assert runner.RETRIEVER_CHOICES == ("dense", "bm25", "hybrid-rrf")


def test_stage_tagged_retrieve_bm25_serves_stage_tagged_hits():
    docs = _ir_docs()
    idx = BM25IRIndex()
    idx.build(docs)
    hits, stage = runner.stage_tagged_retrieve(
        "delta zeta", retriever="bm25", bm25_index=idx, pool_k=10, served_k=2
    )
    assert [h.doc_id for h in hits] == ["d2"]  # only positive-scoring doc
    assert stage.retriever == "bm25"
    ranks = stage.stage_ranks()
    assert set(ranks) == {"pool", "served"}  # no reranker -> no reranked stage
    assert ranks["served"][0]["doc_id"] == "d2"
    assert ranks["served"][0]["rank"] == 1


def test_stage_tagged_retrieve_hybrid_rrf_fuses_bm25_and_dense():
    docs = _ir_docs()
    bm25 = BM25IRIndex()
    bm25.build(docs)
    dense = _FakeDenseIndex(docs, order=["d3", "d2", "d1"])
    hits, stage = runner.stage_tagged_retrieve(
        "delta epsilon",
        retriever="hybrid-rrf",
        bm25_index=bm25,
        dense_index=dense,
        pool_k=10,
        served_k=2,
    )
    # d2 appears in BOTH rankings (bm25 rank 1, dense rank 2) -> top RRF score.
    assert [h.doc_id for h in hits] == ["d2", "d3"]
    assert stage.retriever == "hybrid-rrf"
    ranks = stage.stage_ranks()
    assert ranks["pool"][0]["doc_id"] == "d2"
    # Served hits resolve to real texts via the shared chunk store.
    resolved = bm25.resolve_hits(hits)
    assert [d.text for d in resolved] == ["delta epsilon zeta", "eta theta iota"]


def test_stage_tagged_retrieve_fails_closed_on_missing_indexes():
    docs = _ir_docs()
    bm25 = BM25IRIndex()
    bm25.build(docs)
    with pytest.raises(RuntimeError, match="BM25 index not initialized"):
        runner.stage_tagged_retrieve("q", retriever="bm25", bm25_index=None)
    with pytest.raises(RuntimeError, match="BOTH"):
        runner.stage_tagged_retrieve(
            "q", retriever="hybrid-rrf", bm25_index=bm25, dense_index=None
        )


def test_run_experiment_unknown_retriever_fails_closed_before_any_work():
    with pytest.raises(ValueError, match="Unknown retriever"):
        runner.run_experiment(
            **_run_experiment_kwargs(workload_mode="single", retriever="sparse")
        )


@pytest.mark.parametrize("baseline", ["redis", "hybrid"])
def test_run_experiment_non_dense_retriever_refuses_cache_baselines(baseline):
    """redis/hybrid cache keys pin the dense pipeline; mixing retrievers would
    silently serve cross-retriever cache hits -- refused up front."""
    with pytest.raises(ValueError, match="dense"):
        runner.run_experiment(
            **_run_experiment_kwargs(
                workload_mode="single", baseline=baseline, retriever="bm25"
            )
        )


# --------------------------------------------------------------------------
# 6. Dataset choices: ruler + scbench wired into the runner
# --------------------------------------------------------------------------


def test_default_dataset_split_for_new_datasets():
    assert runner.default_dataset_split("ruler") == "synthetic"
    assert runner.default_dataset_split("scbench") == "test"
    # Pre-existing conventions untouched.
    assert runner.default_dataset_split("squad_v2") == "validation"
    assert runner.default_dataset_split("humaneval") == "test"


def test_cli_help_lists_new_dataset_and_retriever_choices(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        runner.main()
    assert excinfo.value.code == 0
    # argparse wraps long choice lists arbitrarily; compare whitespace-free.
    compact = "".join(capsys.readouterr().out.split())
    for token in ("ruler", "scbench", "hybrid-rrf", "--retriever",
                  "--ruler-context-tokens", "--ruler-task"):
        assert token in compact, f"missing {token} in --help output"


def test_cli_threads_dataset_retriever_and_ruler_args(monkeypatch):
    calls: List[Dict[str, Any]] = []

    def _recorder(**kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(runner, "run_experiment", _recorder)
    # setenv first so monkeypatch restores the pre-test state even though the
    # code under test writes os.environ directly.
    monkeypatch.setenv("CAGE_RULER_CONTEXT_TOKENS", "sentinel")
    monkeypatch.setenv("CAGE_RULER_TASK", "sentinel")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--baseline", "no_cache",
            "--model", "m/x",
            "--dataset", "ruler",
            "--ruler-context-tokens", "512",
            "--ruler-task", "niah_multikey",
            "--retriever", "bm25",
            "--num-queries", "2",
        ],
    )
    runner.main()
    assert len(calls) == 1
    assert calls[0]["dataset"] == "ruler"
    assert calls[0]["retriever"] == "bm25"
    assert os.environ["CAGE_RULER_CONTEXT_TOKENS"] == "512"
    assert os.environ["CAGE_RULER_TASK"] == "niah_multikey"


def test_get_loader_ruler_env_params_and_per_run_seed(monkeypatch):
    monkeypatch.setenv("CAGE_RULER_CONTEXT_TOKENS", "512")
    monkeypatch.setenv("CAGE_RULER_TASK", "niah_multikey")
    from src.data.loader import get_loader

    loader = get_loader("ruler", split=runner.default_dataset_split("ruler"), seed=7)
    assert type(loader).__name__ == "RulerLoader"
    assert loader.context_length_tokens == 512
    assert loader.task == "niah_multikey"
    assert loader.seed == 7  # the run's --seed feeds the deterministic generator
    items = loader.load(max_examples=2)
    assert len(items) == 2
    assert items[0].metadata["target_context_tokens"] == 512
    assert items[0].metadata["task"] == "niah_multikey"
    assert items[0].metadata["seed"] == 7


def test_get_loader_scbench_constructs_with_the_published_test_split(monkeypatch):
    monkeypatch.delenv("CAGE_SCBENCH_SPLIT", raising=False)
    monkeypatch.delenv("CAGE_SCBENCH_SUBSET", raising=False)
    from src.data.loader import get_loader

    loader = get_loader("scbench", split=runner.default_dataset_split("scbench"), seed=3)
    assert type(loader).__name__ == "SCBenchLoader"
    assert loader.split == "test"  # microsoft/SCBench publishes only "test"
    assert loader.subset == "scbench_kv"  # charter D5#6 default subset


# --------------------------------------------------------------------------
# 7. hf-oracle corpus-prefix preload (adapter seam wiring)
# --------------------------------------------------------------------------


def test_derive_corpus_prompt_prefix_raw_layout():
    from src.utils.prompting import format_qa_prompt

    ctxs = ["SHARED CORPUS BLOCK about xenon lamps."]
    prefix = runner.derive_corpus_prompt_prefix(lambda q: format_qa_prompt(q, ctxs))
    full = format_qa_prompt("What emits light?", ctxs)
    assert full.startswith(prefix)  # the literal-prefix contract the oracle enforces
    assert ctxs[0] in prefix  # the corpus block is inside the cached prefix
    assert full[len(prefix):].startswith("\n\nQuestion:")  # suffix = per-query tail


def test_derive_corpus_prompt_prefix_chat_fallback_layout():
    from src.utils.prompting import format_qa_messages, messages_to_fallback_prompt

    ctxs = ["SHARED CORPUS BLOCK about xenon lamps."]

    def build(q: str) -> str:
        return messages_to_fallback_prompt(format_qa_messages(q, ctxs))

    prefix = runner.derive_corpus_prompt_prefix(build)
    full = build("What emits light?")
    assert full.startswith(prefix)
    assert ctxs[0] in prefix


def test_derive_corpus_prompt_prefix_fails_closed_without_marker():
    with pytest.raises(ValueError, match="corpus-block/question boundary"):
        runner.derive_corpus_prompt_prefix(lambda q: "no marker " + q)


class _FakeOracleEngine:
    """Records preload calls; declares the corpus_prefix_reuse capability."""

    def __init__(self) -> None:
        self.preloads: List[str] = []

    def capabilities(self) -> Dict[str, Any]:
        return {"corpus_prefix_reuse": True}

    def preload_corpus_prefix(self, prefix_text: str, **kwargs: Any) -> float:
        self.preloads.append(prefix_text)
        return 12.5

    def clear_corpus_prefix(self) -> None:
        pass


class _SeamlessOracleEngine:
    """Adapter WITHOUT the preload seam: must be refused honestly."""

    def capabilities(self) -> Dict[str, Any]:
        return {"corpus_prefix_reuse": False}


def _corpus_example(ex_id: str, block: Any, text: str) -> CAGExample:
    return CAGExample(
        id=ex_id,
        question="q?",
        context=[text],
        answer="a",
        metadata={"corpus_prefix": True, "corpus_block": block},
    )


def test_oracle_preloader_requires_the_adapter_seam():
    with pytest.raises(RuntimeError, match="preload_corpus_prefix"):
        runner.OracleCorpusPreloader(_SeamlessOracleEngine(), "raw")


def test_oracle_preloader_preloads_once_per_block_and_on_block_change():
    from src.utils.prompting import format_qa_prompt

    eng = _FakeOracleEngine()
    pre = runner.OracleCorpusPreloader(eng, "raw")
    b0, b1 = "BLOCK ZERO text.", "BLOCK ONE text."
    pre.ensure(_corpus_example("e0a", 0, b0), [b0])
    pre.ensure(_corpus_example("e0b", 0, b0), [b0])
    assert len(eng.preloads) == 1  # same block -> single preload
    pre.ensure(_corpus_example("e1", 1, b1), [b1])
    assert len(eng.preloads) == 2  # block change -> re-preload
    # Every preloaded prefix is a literal prefix of the served prompt.
    assert format_qa_prompt("real question?", [b0]).startswith(eng.preloads[0])
    assert format_qa_prompt("real question?", [b1]).startswith(eng.preloads[1])


def test_oracle_preloader_skips_non_corpus_examples():
    eng = _FakeOracleEngine()
    pre = runner.OracleCorpusPreloader(eng, "raw")
    ex = CAGExample(id="e", question="q?", context=["ctx"], answer="a", metadata={})
    pre.ensure(ex, ["ctx"])
    assert eng.preloads == []  # prefix-free serving stays prefix-free


def test_run_experiment_oracle_corpus_multi_turn_fails_closed(monkeypatch):
    """Corpus-prefix mode requires the literal-prefix prompt layout, which the
    multi-turn history breaks -- refused BEFORE any dataset/engine work."""
    monkeypatch.setenv("CAGE_CORPUS_PREFIX_BUDGET", "1000")
    with pytest.raises(ValueError, match="corpus-prefix"):
        runner.run_experiment(
            **_run_experiment_kwargs(workload_mode="multi_turn", backend="hf-oracle")
        )


# --------------------------------------------------------------------------
# 8. Adapter honesty columns persisted per row (ADR-0007 / charter D2)
# --------------------------------------------------------------------------


def _plain_response() -> InferenceResponse:
    return _stub_response(
        InferenceRequest(prompt="p", max_tokens=1, temperature=0.0, request_id="r")
    )


def test_adapter_honesty_columns_from_a_stamped_response():
    resp = _plain_response()
    resp.engine_id = "vllm"
    resp.usage_telemetry_available = True
    resp.cached_token_telemetry_available = False
    resp.retries = 2
    assert runner.adapter_honesty_columns(resp) == {
        "engine_id": "vllm",
        "usage_telemetry_available": True,
        "cached_token_telemetry_available": False,
        "retries": 2,
        "reference_engine": None,
    }


def test_adapter_honesty_columns_never_fabricated_when_unstamped():
    cols = runner.adapter_honesty_columns(_plain_response())
    assert all(v is None for v in cols.values())  # provenance, never fabricated
    assert list(cols) == [
        "engine_id",
        "usage_telemetry_available",
        "cached_token_telemetry_available",
        "retries",
        "reference_engine",
    ]


def test_adapter_honesty_columns_carry_the_hf_oracle_reference_flag():
    resp = _plain_response()
    resp.engine_id = "hf_reference"
    resp.reference_engine = True
    cols = runner.adapter_honesty_columns(resp)
    assert cols["engine_id"] == "hf_reference"
    assert cols["reference_engine"] is True


def test_adapter_honesty_columns_append_cleanly_via_union_of_keys():
    row = {"example_id": "e1", "baseline": "b", "ttft_ms": 1.0}
    existing = list(row)
    row.update(runner.adapter_honesty_columns(_plain_response()))
    assert list(row)[: len(existing)] == existing  # existing columns undisturbed
    fieldnames = list(dict.fromkeys(row))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(row)
    assert "engine_id" in buf.getvalue().splitlines()[0].split(",")


# --------------------------------------------------------------------------
# 9. _reset_prefix_cache -> adapter.flush_cache() migration (capabilities-gated)
# --------------------------------------------------------------------------


class _FlushRecorderAdapter:
    """Adapter fake exposing the ADR-0007 flush seam."""

    instances: List["_FlushRecorderAdapter"] = []
    flush_endpoint: Optional[str] = "/reset_prefix_cache"

    def __init__(self, model_name: str, api_base: str = "", **kwargs: Any) -> None:
        self.model_name = model_name
        self.api_base = api_base
        self.flushed = 0
        type(self).instances.append(self)

    def capabilities(self) -> Dict[str, Any]:
        return {"flush_endpoint": type(self).flush_endpoint}

    def flush_cache(self) -> None:
        self.flushed += 1


def _forbid_urlopen(monkeypatch, reason: str) -> None:
    def _no_urlopen(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(reason)

    monkeypatch.setattr("urllib.request.urlopen", _no_urlopen)


def _record_urlopen(monkeypatch) -> List[str]:
    posts: List[str] = []

    def _fake_urlopen(req: Any, timeout: Optional[float] = None) -> None:
        posts.append(req.full_url)

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    return posts


def test_reset_prefix_cache_migrates_to_adapter_flush(monkeypatch):
    class _A(_FlushRecorderAdapter):
        instances: List[Any] = []

    monkeypatch.setattr(runner, "VLLMAdapter", _A)
    _forbid_urlopen(monkeypatch, "legacy raw POST must not run when the adapter flush seam is capable")
    runner._reset_prefix_cache("http://h:1", backend="vllm", model="m")
    assert len(_A.instances) == 1
    assert _A.instances[0].flushed == 1


def test_reset_prefix_cache_sglang_resolves_env_api_base(monkeypatch):
    class _A(_FlushRecorderAdapter):
        instances: List[Any] = []
        flush_endpoint = "/flush_cache"

    monkeypatch.setattr(runner, "SGLangAdapter", _A)
    monkeypatch.setenv("CAGE_SGLANG_API_BASE", "http://sg:30000")
    _forbid_urlopen(monkeypatch, "adapter path must serve the flush")
    runner._reset_prefix_cache("http://h:1", backend="sglang", model="m")
    assert _A.instances[0].api_base == "http://sg:30000"
    assert _A.instances[0].flushed == 1


def test_reset_prefix_cache_legacy_fallback_without_flush_capability(monkeypatch):
    class _A(_FlushRecorderAdapter):
        instances: List[Any] = []
        flush_endpoint = None  # LMDeploy documents no flush endpoint

    monkeypatch.setattr(runner, "LMDeployAdapter", _A)
    posts = _record_urlopen(monkeypatch)
    runner._reset_prefix_cache("http://h:1", backend="lmdeploy", model="m")
    assert posts == ["http://h:1/reset_prefix_cache"]  # old path kept as fallback


def test_reset_prefix_cache_unknown_backend_uses_legacy_path(monkeypatch):
    posts = _record_urlopen(monkeypatch)
    runner._reset_prefix_cache("http://h:1", backend="gemini", model="m")
    assert posts == ["http://h:1/reset_prefix_cache"]


def test_reset_prefix_cache_adapter_failure_warns_never_raises(monkeypatch, capsys):
    class _A(_FlushRecorderAdapter):
        instances: List[Any] = []

        def flush_cache(self) -> None:
            raise RuntimeError("dev mode off")

    monkeypatch.setattr(runner, "VLLMAdapter", _A)
    _forbid_urlopen(monkeypatch, "no legacy retry after an adapter flush failure (same endpoint)")
    runner._reset_prefix_cache("http://h:1", backend="vllm", model="m")
    assert "WARNING" in capsys.readouterr().out  # matches the historical helper


# --------------------------------------------------------------------------
# 10. preflight_check.sh telemetry-parity gate (charter D2)
# --------------------------------------------------------------------------


def test_preflight_check_bash_syntax_ok():
    proc = subprocess.run(
        ["bash", "-n", str(PREFLIGHT_PATH)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


def test_preflight_keeps_common_lib_and_declares_the_parity_gate():
    text = PREFLIGHT_PATH.read_text(encoding="utf-8")
    assert 'source "$PROJECT_DIR/scripts/lib/_common.sh"' in text
    assert "CAGE-D2-TELEMETRY-PARITY-GATE" in text
    assert "streamed_ttft" in text
    assert "CAGE_PREFLIGHT_BACKENDS" in text


def _extract_parity_gate_snippet() -> str:
    text = PREFLIGHT_PATH.read_text(encoding="utf-8")
    start = text.index("# CAGE-D2-TELEMETRY-PARITY-GATE")
    end = text.index("\nPY\n", start)
    return text[start:end]


def test_preflight_parity_gate_passes_for_streaming_backends():
    proc = subprocess.run(
        [sys.executable, "-", "vllm,sglang,lmdeploy,lmdeploy-turbomind"],
        input=_extract_parity_gate_snippet(),
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[FAIL]" not in proc.stdout
    assert proc.stdout.count("[caps]") == 4  # full capabilities dict printed per backend
    assert "streamed_ttft" in proc.stdout


def test_preflight_parity_gate_fails_for_backend_without_streamed_ttft():
    proc = subprocess.run(
        [sys.executable, "-", "hf-oracle"],
        input=_extract_parity_gate_snippet(),
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 1
    assert "[FAIL]" in proc.stdout


def test_streaming_adapter_capabilities_offline_contract():
    """The exact contract the shell gate's python asserts, proven offline:
    the HTTP adapter constructors are network-free and declare streamed_ttft."""
    from src.inference.lmdeploy_adapter import LMDeployAdapter
    from src.inference.sglang_adapter import SGLangAdapter
    from src.inference.vllm_adapter import VLLMAdapter

    for cls in (VLLMAdapter, SGLangAdapter, LMDeployAdapter):
        caps = cls(model_name="probe", api_base="http://localhost:1").capabilities()
        assert caps["streamed_ttft"] is True
