"""T4.1 pins: multi-instance (role-tagged) window telemetry — the PD foundation.

WHAT is pinned and WHY:

Before T4.1 a window was 1 api_base -> 1 sampler -> 1 UNLABELED series, so a
prefill+decode pair could not even be represented (2026-08-27 audit). This
file pins the four pieces that fix it, refusal paths first:

1. CAGE_TELEMETRY_ENDPOINTS parsing (run_experiment.parse_telemetry_endpoints
   / resolve_telemetry_endpoints): unset -> None (legacy wiring), valid
   role=url lists parse in order, and EVERY malformed shape — set-but-empty,
   missing '=', bad role token, duplicate role, non-http(s) url, endpoints
   configured while --vllm-telemetry is off — refuses with all problems
   listed, BEFORE any serving work (a config typo must never burn a GPU run).

2. VllmTelemetrySampler role stamping: role="single" default, validated
   non-empty [a-z0-9_-]+ token at construction (fail-closed — a malformed
   role must never reach disk), stamped as `instance` into every series
   record via the shared record shaping, OVERWRITING any rogue same-named
   snapshot key (the sampler polled the endpoint; its role is ground truth).
   The dialect param keeps working alongside role.

3. save_merged_series: two role-tagged samplers, driven through the REAL
   _run tick path (monkeypatched capture_snapshot + time), merge into ONE
   JSONL sorted by ts_s with each record carrying its role; duplicate roles
   refuse; an all-empty merge writes nothing (never a misleading empty
   artifact).

4. campaign_layout: legacy (untagged) files are a DIFFERENTIAL pin — loader
   columns and regime-bridge numbers identical to the pre-T4.1 pinned ZOH
   values; tagged files surface the `instance` column; and the load-bearing
   fail-closed behavior: write_window_regime REFUSES a series with >=2
   distinct instances (pooled single-series regime math over interleaved
   per-role gauges fabricates a fictional pooled instance — per-role budgets
   are T2.3's compute_pd_window_regime_inputs). A single tagged role does
   NOT refuse: the single-sampler path stays green end-to-end.

5. REPAIR (verifier round): a multi-endpoint run whose role samplers ALL
   collected zero samples (role URLs pass grammar validation, reachability
   does not exist until live) must produce an ABSENT vllm_telemetry.json
   plus a LOUD warning — never the pre-T4.1 capture(api_base) fallback,
   which would write a legacy-shaped single-instance artifact in which a
   broken PD sampler set masquerades as healthy. Behavioral pins on
   build_multi_instance_snapshot / speculative_acceptance_from_snapshot,
   plus source pins that BOTH single-instance fallbacks (one-shot capture
   and the spec-decode backstop) are hard-gated on vllm_role_samplers is
   None.
"""

from __future__ import annotations

import importlib.util
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
    """Stub fastapi/pydantic so run_experiment imports in the lean venv.

    Import-surface only — same pattern as tests/test_wave1_runexp_telemetry.py
    (decorators pass through, nothing is served); a venv that really has the
    packages keeps them (find_spec wins).
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
from src.orchestration import campaign_layout as cl  # noqa: E402


def _load_runner():
    # scripts/3_run is not an importable package ("3_run" is not a valid
    # module name), so load by path — same pattern as the wave-1 test.
    spec = importlib.util.spec_from_file_location(
        "cage_run_experiment_t41", RUNNER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


# --------------------------------------------------------------------------
# 1. CAGE_TELEMETRY_ENDPOINTS — accept matrix
# --------------------------------------------------------------------------


def test_parse_unset_is_none_legacy_wiring():
    assert runner.parse_telemetry_endpoints(None) is None


def test_parse_two_roles_in_order():
    pairs = runner.parse_telemetry_endpoints(
        "prefill=http://10.0.0.1:8000,decode=http://10.0.0.2:8001"
    )
    assert pairs == [
        ("prefill", "http://10.0.0.1:8000"),
        ("decode", "http://10.0.0.2:8001"),
    ]


def test_parse_single_https_endpoint():
    assert runner.parse_telemetry_endpoints("single=https://host/path") == [
        ("single", "https://host/path")
    ]


def test_parse_tolerates_surrounding_whitespace():
    pairs = runner.parse_telemetry_endpoints(
        " prefill=http://a:1 , decode=http://b:2 "
    )
    assert pairs == [("prefill", "http://a:1"), ("decode", "http://b:2")]


def test_parse_role_charset_digits_underscore_hyphen():
    pairs = runner.parse_telemetry_endpoints("pd-node_1=http://h:1")
    assert pairs == [("pd-node_1", "http://h:1")]


# --------------------------------------------------------------------------
# 1. CAGE_TELEMETRY_ENDPOINTS — refuse matrix (fail-closed, before serving)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "",  # set-but-empty: the operator tried to configure something
        "   ",
        "prefill",  # no '='
        "=http://a:1",  # empty role
        "Prefill=http://a:1",  # uppercase outside the role grammar
        "pre fill=http://a:1",  # internal whitespace
        "prefill=ftp://a:1",  # non-http(s) scheme
        "prefill=localhost:8000",  # scheme-less url
        "prefill=http://",  # hostless url
        "a=http://x:1,a=http://y:2",  # duplicate role
        "a=http://x:1,",  # trailing comma -> empty entry
    ],
    ids=[
        "empty", "blank", "no-equals", "empty-role", "uppercase-role",
        "space-in-role", "bad-scheme", "schemeless", "hostless",
        "duplicate-role", "trailing-comma",
    ],
)
def test_parse_refuses_malformed(raw):
    with pytest.raises(ValueError, match="CAGE_TELEMETRY_ENDPOINTS"):
        runner.parse_telemetry_endpoints(raw)


def test_parse_lists_every_problem_not_just_the_first():
    with pytest.raises(ValueError) as exc:
        runner.parse_telemetry_endpoints("BAD=http://a:1,alsobad")
    msg = str(exc.value)
    assert "entry 0" in msg and "entry 1" in msg


def test_resolve_refuses_endpoints_without_telemetry_flag(monkeypatch):
    # Configured role telemetry that would be silently unrecorded is a
    # fail-closed mismatch, never a default.
    monkeypatch.setenv("CAGE_TELEMETRY_ENDPOINTS", "prefill=http://a:1")
    with pytest.raises(ValueError, match="vllm-telemetry"):
        runner.resolve_telemetry_endpoints(False)


def test_resolve_returns_pairs_when_enabled(monkeypatch):
    monkeypatch.setenv(
        "CAGE_TELEMETRY_ENDPOINTS", "prefill=http://a:1,decode=http://b:2"
    )
    assert runner.resolve_telemetry_endpoints(True) == [
        ("prefill", "http://a:1"), ("decode", "http://b:2")
    ]


def test_resolve_unset_env_is_none_either_way(monkeypatch):
    monkeypatch.delenv("CAGE_TELEMETRY_ENDPOINTS", raising=False)
    assert runner.resolve_telemetry_endpoints(True) is None
    assert runner.resolve_telemetry_endpoints(False) is None


# --------------------------------------------------------------------------
# 2. Sampler role stamping
# --------------------------------------------------------------------------


def _sampler_with(samples, ts, **kwargs) -> vt.VllmTelemetrySampler:
    sampler = vt.VllmTelemetrySampler("http://localhost:9", **kwargs)
    sampler._samples = list(samples)
    sampler._sample_ts = list(ts)
    return sampler


def test_sampler_default_role_is_single():
    assert vt.VllmTelemetrySampler("http://localhost:9").role == "single"


def test_sampler_role_and_dialect_coexist():
    sampler = vt.VllmTelemetrySampler(
        "http://localhost:9", dialect="sglang", role="prefill"
    )
    assert sampler.role == "prefill" and sampler.dialect == "sglang"


@pytest.mark.parametrize(
    "role", ["", "Prefill", "pre fill", "role!", None, 123],
    ids=["empty", "uppercase", "space", "punct", "none", "int"],
)
def test_sampler_refuses_invalid_role_at_construction(role):
    # Fail-closed at the constructor: a malformed role must never reach disk.
    with pytest.raises(ValueError, match="role"):
        vt.VllmTelemetrySampler("http://localhost:9", role=role)


def test_save_series_stamps_role_on_every_record(tmp_path: Path):
    sampler = _sampler_with(
        [{"kv_usage": 0.5}, {"kv_usage": 0.9}], [2.0, 3.0], role="decode"
    )
    out = tmp_path / "series.jsonl"
    assert sampler.save_series(str(out)) == str(out)
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["instance"] for r in records] == ["decode", "decode"]


def test_single_path_forward_writes_instance_single(tmp_path: Path):
    sampler = _sampler_with([{"kv_usage": 0.5}], [2.0])  # default role
    out = tmp_path / "series.jsonl"
    sampler.save_series(str(out))
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["instance"] == "single"
    # Dual-field shaping is untouched by the stamp.
    assert rec["ts_s"] == rec["ts"] == 2.0
    assert rec["kv_cache_usage"] == rec["kv_usage"] == 0.5


def test_rogue_snapshot_instance_key_is_overwritten(tmp_path: Path):
    # The sampler polled the endpoint: its role is the ground truth of which
    # instance the gauges came from — a rogue key is never trusted.
    sampler = _sampler_with(
        [{"kv_usage": 0.5, "instance": "rogue"}], [2.0], role="prefill"
    )
    out = tmp_path / "series.jsonl"
    sampler.save_series(str(out))
    assert json.loads(out.read_text().splitlines()[0])["instance"] == "prefill"


# --------------------------------------------------------------------------
# 3. Merged two-sampler series (records flow through the REAL tick path)
# --------------------------------------------------------------------------


def _drive_ticks(monkeypatch, sampler, snaps, t0s):
    """Run sampler._run inline for len(snaps) ticks with controlled clocks.

    capture_snapshot is monkeypatched (records flow through the real _run
    tick path); vt's time.time is fed t0 then t0+99 per tick so the recorded
    sample ts is exactly t0 and the inter-tick wait is skipped (dt < 0).
    """
    assert len(snaps) == len(t0s)
    times = []
    for t in t0s:
        times.extend([t, t + 99.0])
    time_iter = iter(times)
    monkeypatch.setattr(vt.time, "time", lambda: next(time_iter))
    snap_iter = iter(snaps)
    remaining = {"n": len(snaps)}

    def fake_capture(url, *, metrics_path="/metrics", api_key=None,
                     interval=1.0, dialect="vllm"):
        snap = next(snap_iter)
        remaining["n"] -= 1
        if remaining["n"] == 0:
            sampler._stop.set()
        return snap

    monkeypatch.setattr(vt, "capture_snapshot", fake_capture)
    monkeypatch.setattr(
        vt.VllmTelemetrySampler, "_read_energy_mj", lambda self: (None, None)
    )
    sampler._run()


def test_merged_two_sampler_series_sorted_and_role_tagged(
    tmp_path: Path, monkeypatch
):
    prefill = vt.VllmTelemetrySampler("http://p:8000", role="prefill")
    decode = vt.VllmTelemetrySampler("http://d:8001", role="decode")
    _drive_ticks(monkeypatch, prefill,
                 [{"kv_usage": 0.1}, {"kv_usage": 0.3}], [100.0, 104.0])
    _drive_ticks(monkeypatch, decode,
                 [{"kv_usage": 0.2}, {"kv_usage": 0.4}], [102.0, 106.0])

    out = tmp_path / "telemetry_series.jsonl"
    assert vt.save_merged_series([prefill, decode], str(out)) == str(out)
    records = [json.loads(line) for line in out.read_text().splitlines()]
    # ONE file, interleaved by ts_s, every record carrying its role.
    assert [r["ts_s"] for r in records] == [100.0, 102.0, 104.0, 106.0]
    assert [r["instance"] for r in records] == [
        "prefill", "decode", "prefill", "decode"
    ]
    # Record shaping is shared with save_series (canonical + legacy fields).
    for rec in records:
        assert rec["ts_s"] == rec["ts"]
        assert rec["kv_cache_usage"] == rec["kv_usage"]


def test_merged_series_refuses_duplicate_roles(tmp_path: Path):
    a = _sampler_with([{"kv_usage": 0.1}], [1.0], role="prefill")
    b = _sampler_with([{"kv_usage": 0.2}], [2.0], role="prefill")
    with pytest.raises(ValueError, match="duplicate"):
        vt.save_merged_series([a, b], str(tmp_path / "out.jsonl"))


def test_merged_series_empty_run_writes_nothing(tmp_path: Path):
    a = _sampler_with([], [], role="prefill")
    b = _sampler_with([], [], role="decode")
    out = tmp_path / "out.jsonl"
    assert vt.save_merged_series([a, b], str(out)) is None
    assert not out.exists()  # never a misleading empty artifact


# --------------------------------------------------------------------------
# 4. Loader: legacy differential + instance column
# --------------------------------------------------------------------------

# The pre-T4.1 pinned ZOH case (tests/test_campaign_layout.py) — the
# differential baseline: legacy input must keep producing EXACTLY these
# loader columns/values and regime numbers.
_LEGACY_RECORDS = [
    {"ts": 2.0, "kv_usage": 0.5, "preemptions_total": 5},
    {"ts": 6.0, "kv_usage": 1.0, "preemptions_total": 5},
    {"ts": 8.0, "kv_usage": 0.8, "preemptions_total": 9},
]


def _write_series(path: Path, records) -> Path:
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return path


def _tagged(records, role):
    return [{**r, "instance": role} for r in records]


def test_loader_legacy_file_keeps_exact_pre_t41_columns(tmp_path: Path):
    frame = cl.load_telemetry_series(
        _write_series(tmp_path / "s.jsonl", _LEGACY_RECORDS)
    )
    # Differential pin: column set AND values identical to pre-T4.1 output.
    assert list(frame.columns) == ["ts_s", "kv_cache_usage", "preemptions_total"]
    assert frame["ts_s"].tolist() == [2.0, 6.0, 8.0]
    assert frame["kv_cache_usage"].tolist() == [0.5, 1.0, 0.8]
    assert frame["preemptions_total"].tolist() == [5, 5, 9]


def test_loader_surfaces_instance_column_when_any_record_tagged(tmp_path: Path):
    records = _tagged(_LEGACY_RECORDS[:2], "prefill") + [_LEGACY_RECORDS[2]]
    frame = cl.load_telemetry_series(_write_series(tmp_path / "s.jsonl", records))
    assert list(frame.columns) == [
        "ts_s", "kv_cache_usage", "preemptions_total", "instance"
    ]
    assert frame["instance"].tolist()[:2] == ["prefill", "prefill"]
    # The untagged record is surfaced as a hole, not hidden or coerced.
    assert frame["instance"].isna().tolist() == [False, False, True]


@pytest.mark.parametrize("bad", [123, "", 1.5], ids=["int", "empty-str", "float"])
def test_loader_refuses_non_string_instance(tmp_path: Path, bad):
    records = [{**_LEGACY_RECORDS[0], "instance": bad}]
    with pytest.raises(cl.CampaignLayoutError, match="instance"):
        cl.load_telemetry_series(_write_series(tmp_path / "s.jsonl", records))


# --------------------------------------------------------------------------
# 4. Regime bridge: legacy differential + the load-bearing >=2-instance refusal
# --------------------------------------------------------------------------


def _regime_doc(tmp_path: Path, records, name: str) -> dict:
    series = _write_series(tmp_path / f"{name}.jsonl", records)
    window = tmp_path / f"window_{name}"
    window.mkdir()
    path = cl.write_window_regime(
        window, t_start=0.0, t_end=10.0, telemetry_path=series
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_regime_bridge_legacy_numbers_unchanged(tmp_path: Path):
    doc = _regime_doc(tmp_path, _LEGACY_RECORDS, "legacy")
    assert doc["telemetry_ok"] is True
    # The pre-T4.1 pinned ZOH numbers, verbatim.
    assert doc["inputs"]["rho_kv_time_avg"] == pytest.approx(0.7)
    assert doc["inputs"]["scarcity_events"] == 4
    assert doc["inputs"]["n_samples"] == 3
    assert doc["inputs"]["coverage"] == pytest.approx(0.8)


def test_regime_bridge_single_role_tag_is_numerically_identical(tmp_path: Path):
    # Differential: the SAME gauges, untagged vs tagged with one role, must
    # certify to byte-equal inputs — the optional column never perturbs math.
    legacy = _regime_doc(tmp_path, _LEGACY_RECORDS, "legacy")
    tagged = _regime_doc(tmp_path, _tagged(_LEGACY_RECORDS, "single"), "tagged")
    assert tagged["telemetry_ok"] is True
    assert tagged["inputs"] == legacy["inputs"]


def test_regime_bridge_refuses_two_distinct_instances(tmp_path: Path):
    # THE load-bearing fail-closed behavior of T4.1: pooled single-series
    # regime math over an interleaved prefill+decode stream must refuse.
    records = (
        _tagged(_LEGACY_RECORDS[:2], "prefill")
        + _tagged(_LEGACY_RECORDS[2:], "decode")
    )
    with pytest.raises(
        cl.CampaignLayoutError,
        match="pooled PD regime math requires per-role budgets",
    ) as exc:
        _regime_doc(tmp_path, records, "pd")
    # The refusal names the right tool so the caller is redirected, not stuck.
    assert "compute_pd_window_regime_inputs" in str(exc.value)
    assert "T2.3" in str(exc.value)


def test_regime_bridge_refuses_tagged_mixed_with_untagged(tmp_path: Path):
    # Untagged gauges inside a tagged file are unattributable — pooling them
    # with "prefill" is the same sin as pooling prefill with decode.
    records = _tagged(_LEGACY_RECORDS[:2], "prefill") + [_LEGACY_RECORDS[2]]
    with pytest.raises(cl.CampaignLayoutError, match="per-role budgets"):
        _regime_doc(tmp_path, records, "mixed")


def test_regime_bridge_accepts_one_distinct_role(tmp_path: Path):
    # A single tagged role is still a single instance: no refusal.
    doc = _regime_doc(tmp_path, _tagged(_LEGACY_RECORDS, "prefill"), "one-role")
    assert doc["telemetry_ok"] is True


# --------------------------------------------------------------------------
# Single-sampler path end-to-end: sampler -> series -> loader -> regime
# --------------------------------------------------------------------------


def test_single_sampler_end_to_end_unchanged(tmp_path: Path):
    sampler = _sampler_with(
        [
            {"kv_usage": 0.5, "preemptions_total": 5},
            {"kv_usage": 1.0, "preemptions_total": 5},
            {"kv_usage": 0.8, "preemptions_total": 9},
        ],
        [2.0, 6.0, 8.0],
    )  # default role="single" — today's one-sampler wiring, forward-written
    series = tmp_path / "telemetry_series.jsonl"
    sampler.save_series(str(series))
    frame = cl.load_telemetry_series(series)
    assert frame["instance"].tolist() == ["single"] * 3
    window = tmp_path / "window_e2e"
    window.mkdir()
    doc = json.loads(
        cl.write_window_regime(
            window, t_start=0.0, t_end=10.0, telemetry_path=series
        ).read_text(encoding="utf-8")
    )
    # Identical certified numbers to the legacy pinned ZOH case.
    assert doc["telemetry_ok"] is True
    assert doc["inputs"]["rho_kv_time_avg"] == pytest.approx(0.7)
    assert doc["inputs"]["scarcity_events"] == 4


# --------------------------------------------------------------------------
# run_experiment wiring pins (source inspection, wave-1 precedent)
# --------------------------------------------------------------------------


def test_endpoint_validation_runs_before_any_serving_work():
    import inspect

    src_text = inspect.getsource(runner.run_experiment)
    parse_idx = src_text.index("resolve_telemetry_endpoints(vllm_telemetry)")
    # The env refusal must precede sampler spawn, the measured stage, and the
    # GPU tracker — i.e. every piece of serving work in run_experiment.
    assert parse_idx < src_text.index("resolve_telemetry_dialect(backend)")
    assert parse_idx < src_text.index("GPUMetricsTracker")
    assert parse_idx < src_text.index("measured_window_t_start")


def test_single_and_multi_wiring_are_mutually_exclusive():
    import inspect

    src_text = inspect.getsource(runner.run_experiment)
    # Legacy single-sampler wiring survives verbatim (env-absent path)...
    assert "vllm_sampler = VllmTelemetrySampler(" in src_text
    assert "vllm_sampler.save_series(_series_path)" in src_text
    # ...and the multi path merges through the shared writer, never per-tick.
    assert "save_merged_series(vllm_role_samplers, _series_path)" in src_text


# --------------------------------------------------------------------------
# 5. REPAIR: a dead multi-endpoint sampler set must never masquerade as a
#    healthy single instance (verifier major + spec-backstop minor)
# --------------------------------------------------------------------------


def test_build_multi_snapshot_healthy_pair_is_per_role_never_pooled():
    per_role = {
        "prefill": {"kv_cache_usage_peak": 0.9},
        "decode": {"kv_cache_usage_peak": 0.4},
    }
    snap, warning = runner.build_multi_instance_snapshot(per_role)
    assert warning is None
    # Exactly the {multi_instance, instances} schema — no pooled top-level
    # gauges fabricated from interleaved prefill+decode samples.
    assert snap == {"multi_instance": True, "instances": per_role}


def test_build_multi_snapshot_keeps_empty_role_as_none():
    # Absence is not zero: a role whose sampler collected nothing stays None
    # inside the artifact (one live role does NOT vouch for the dead one).
    snap, warning = runner.build_multi_instance_snapshot(
        {"prefill": None, "decode": {"kv_cache_usage_peak": 0.4}}
    )
    assert warning is None
    assert snap["multi_instance"] is True
    assert snap["instances"]["prefill"] is None


def test_build_multi_snapshot_all_empty_refuses_with_loud_warning():
    # THE verifier-major lane: grammar-valid but unreachable role URLs leave
    # every sampler empty. The snapshot must be ABSENT (None -> no
    # vllm_telemetry.json is written) and the warning must be loud + specific.
    snap, warning = runner.build_multi_instance_snapshot(
        {"prefill": None, "decode": None}
    )
    assert snap is None
    assert warning is not None
    assert "ZERO samples" in warning
    assert "prefill" in warning and "decode" in warning
    assert "vllm_telemetry.json" in warning  # names the artifact that will be absent
    assert "refuse" in warning  # says regime labeling refuses these windows


def test_speculative_acceptance_reads_top_level_on_single_path():
    # Pre-T4.1 behavior preserved byte-for-byte in meaning: top-level key on
    # the single path, None-safe on absent/empty snapshots.
    assert (
        runner.speculative_acceptance_from_snapshot(
            {"spec_decode_acceptance_rate": 0.61}
        )
        == 0.61
    )
    assert runner.speculative_acceptance_from_snapshot(None) is None
    assert runner.speculative_acceptance_from_snapshot({}) is None


def test_speculative_acceptance_reads_per_role_on_multi_path():
    # A PD pair drafts on ONE role; sampled acceptance inside any
    # instances[role] proves speculation engaged (real data, not fabricated).
    snap = {
        "multi_instance": True,
        "instances": {
            "prefill": None,
            "decode": {"spec_decode_acceptance_rate": 0.7},
        },
    }
    assert runner.speculative_acceptance_from_snapshot(snap) == 0.7


def test_speculative_acceptance_multi_without_role_data_stays_none():
    # No acceptance anywhere -> None keeps the DEGRADED sentinel armed; the
    # multi dict's mere existence must not read as "speculation engaged".
    snap = {
        "multi_instance": True,
        "instances": {"prefill": {"kv_cache_usage_peak": 0.4}, "decode": None},
    }
    assert runner.speculative_acceptance_from_snapshot(snap) is None


def test_capture_fallback_is_hard_gated_to_single_path():
    import inspect

    src_text = inspect.getsource(runner.run_experiment)
    # The ONLY one-shot capture call site sits behind the combined guard: a
    # None snapshot in multi mode (all role samplers empty) must fall
    # through to NO artifact, never to a legacy-shaped single capture.
    call = "vllm_telemetry_snapshot, _ = capture(api_base)"
    assert src_text.count(call) == 1
    guard_idx = src_text.index(
        "if vllm_telemetry_snapshot is None and vllm_role_samplers is None:"
    )
    assert guard_idx < src_text.index(call)
    # The all-empty warning is assembled and printed BEFORE the capture site,
    # so the refusal is loud in the run log, not just an absent file.
    build_idx = src_text.index("build_multi_instance_snapshot(_per_role)")
    warn_idx = src_text.index("print(_all_empty_warning)")
    assert build_idx < warn_idx < guard_idx


def test_spec_backstop_is_hard_gated_to_single_path():
    import inspect

    src_text = inspect.getsource(runner.run_experiment)
    # The multi dict never carries top-level spec_decode_acceptance_rate, so
    # an ungated backstop would ALWAYS scrape api_base in multi mode and
    # bolt unattributed single-endpoint spec fields onto the
    # {multi_instance, instances} schema (verifier minor). Guard precedes it.
    scrape_idx = src_text.index("scrape_spec_decode(api_base)")
    guard_idx = src_text.rindex("if vllm_role_samplers is None and (", 0, scrape_idx)
    assert guard_idx < scrape_idx
    # The DEGRADED sentinel reads acceptance through the schema-aware helper,
    # not a raw top-level .get() that a healthy multi-mode speculative run
    # could never satisfy.
    assert "speculative_acceptance_from_snapshot(vllm_telemetry_snapshot)" in src_text
