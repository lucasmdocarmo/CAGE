"""Tests for src/orchestration/campaign_layout.py — the campaign v2 tree PRODUCER.

Topic-8 #126 (H1/H5): the read side (scripts/4_analysis/organize_results.py)
validated a tree only test fixtures produced. These tests prove the producer
and the reader compose: a run written by the library organizes cleanly
(round-trip), the §3 manifest fail-closes on every required field, writes are
atomic (no .tmp residue), the §5 seal verifies green, the §6.1 regime bridge
reproduces the pinned ZOH case from tests/test_regime_inputs.py on BOTH
telemetry field spellings, the (row_key, dataset, ordinal) uniqueness
invariant refuses duplicates and the H12 zero-padding alias pair, and
VllmTelemetrySampler.save_series emits the canonical field names alongside
the legacy ones.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import organize_results as org  # noqa: E402
from src.analysis.cellspec import CellSpec  # noqa: E402
from src.analysis.goodput import GoodputError, IN_REGIME, UNPRESSURED  # noqa: E402
from src.analysis.regime_inputs import REGIME_UNKNOWN  # noqa: E402
from src.analysis.stats.ledger import LedgerError, verify_ledger  # noqa: E402
from src.monitoring.vllm_telemetry import VllmTelemetrySampler  # noqa: E402
from src.orchestration import campaign_layout as cl  # noqa: E402

RUN_ID = "20260814-1200-a-qwen3-14b"
MODEL = "qwen3-14b"
DATASETS = ("squad_v2", "hotpotqa")
CELL_BASELINES = ("B1", "B3")
WINDOWS_PER_DATASET = 2

#: Pinned ZOH case (tests/test_regime_inputs.py::_canonical): window [0, 10),
#: covered time 8 (first sample at 2), integral 0.5*4 + 1.0*2 + 0.8*2 = 5.6
#: -> mean 0.7, coverage 0.8; counter 5 -> 9 => 4 scarcity events.
_ZOH_LEGACY = [
    {"ts": 2.0, "kv_usage": 0.5, "preemptions_total": 5},
    {"ts": 6.0, "kv_usage": 1.0, "preemptions_total": 5},
    {"ts": 8.0, "kv_usage": 0.8, "preemptions_total": 9},
]
_ZOH_CANONICAL = [
    {"ts_s": 2.0, "kv_cache_usage": 0.5, "preemptions_total": 5},
    {"ts_s": 6.0, "kv_cache_usage": 1.0, "preemptions_total": 5},
    {"ts_s": 8.0, "kv_cache_usage": 0.8, "preemptions_total": 9},
]


def _fake_git(_repo: Path) -> tuple[str, bool]:
    return "deadbeef" * 5, False


def _manifest_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "campaign": "camp1",
        "session": "a",
        "run_id": RUN_ID,
        "model": MODEL,
        "engine": "vllm",
        "engine_version": "0.19.1",
        "seed": 1,
        "provider": "gcp",
        "hardware": "a2-ultragpu-1g x1",
        "dataset_manifests_sha256": "0" * 64,
        "cellspec_schema_version": 1,
        "git_provenance": _fake_git,
    }
    base.update(overrides)
    return base


def _specs() -> list[CellSpec]:
    return [CellSpec.from_baseline(b, model=MODEL) for b in CELL_BASELINES]  # type: ignore[arg-type]


def _add_window(
    cell: cl.CellWriter,
    dataset: str,
    *,
    rep: int = 1,
    cage_stats: list[dict[str, Any]] | None = None,
    **overrides: Any,
) -> cl.WindowHandle:
    kwargs: dict[str, Any] = {
        "seed": 1,
        "rep": rep,
        "t_start": 0.0,
        "t_end": 10.0,
        "requests": [{"example_id": f"{dataset}-e0", "ttft_ms": 100.0}],
        "cage_stats": cage_stats if cage_stats is not None else list(_ZOH_LEGACY),
        "engine_metrics": {"snapshot": "before/after"},
        "qa_evidence": [{"example_id": f"{dataset}-e0", "generated_answer": "x"}],
    }
    kwargs.update(overrides)
    return cell.add_window(dataset, **kwargs)


def _build_run(tmp_path: Path) -> cl.CampaignRun:
    """2 cells x 2 datasets x 2 windows, regime.json per window — the §1 tree."""
    run_root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    run = cl.CampaignRun.create(run_root, **_manifest_kwargs())
    for spec in _specs():
        cell = run.cell(spec)
        for dataset in DATASETS:
            for rep in range(1, WINDOWS_PER_DATASET + 1):
                handle = _add_window(cell, dataset, rep=rep)
                cl.write_window_regime(handle.window_dir, t_start=0.0, t_end=10.0)
    return run


# ---------------------------------------------------------------------------
# Writer constants are pinned to the reader's contract (organize_results is
# THE §1 parser; drift here is exactly the Topic-8 H1 failure mode)
# ---------------------------------------------------------------------------


def test_writer_constants_pin_reader_contract() -> None:
    assert cl.WINDOW_DIR_RE.pattern == org.WINDOW_DIR_RE.pattern
    assert cl.RUN_ID_RE.pattern == org.RUN_ID_RE.pattern
    assert cl.SESSIONS == org.SESSIONS
    assert cl.DATASET_IDS == org.DATASET_IDS
    assert cl.QA_EVIDENCE_EXEMPT_DATASETS == org._QA_EVIDENCE_EXEMPT_DATASETS
    # The writer's required set covers everything the organizer demands, and
    # the model roster matches the coverage grid's.
    assert set(org._MANIFEST_STR_KEYS) <= set(cl.MANIFEST_REQUIRED_FIELDS)
    assert cl._MODELS == set(org.GROUP_OF_MODEL)
    assert cl._BASELINE_OF_CELL == org.BASELINE_OF_CELL


# ---------------------------------------------------------------------------
# Round-trip: library-written tree -> organize_results indexes cleanly
# ---------------------------------------------------------------------------


def test_roundtrip_organize_run_indexes_cleanly(tmp_path: Path) -> None:
    run = _build_run(tmp_path)
    run.seal()
    csv_path, md_path = org.organize_run(run.run_root)  # must not raise LayoutError
    assert csv_path.is_file() and md_path.is_file()
    df = pd.read_csv(csv_path)

    # 2 cells x 2 datasets x 2 windows = 8 window rows.
    assert len(df) == len(CELL_BASELINES) * len(DATASETS) * WINDOWS_PER_DATASET
    assert list(df.columns) == list(org.INDEX_COLUMNS)
    assert set(df["baseline"]) == set(CELL_BASELINES)
    assert set(df["dataset"]) == set(DATASETS)
    # Zero-padded %02d ordinals, exactly as the reader's verbatim window_key.
    assert set(df["window_key"]) == {
        f"{d}-{o:02d}" for d in DATASETS for o in range(1, WINDOWS_PER_DATASET + 1)
    }
    # regime.json rides along as an auxiliary indexed artifact in every window.
    for artifacts in df["artifacts"]:
        assert any(a.endswith("regime.json") for a in artifacts.split(";"))


def test_windows_table_matches_spec_schema(tmp_path: Path) -> None:
    """cell.json windows[]: k -> {dataset, seed, rep, budget_r, rate_frac,
    t_start, t_end} (§1) — and the organizer consumes the cell.json."""
    run = _build_run(tmp_path)
    run.seal()
    csv_path, _ = org.organize_run(run.run_root)
    df = pd.read_csv(csv_path)
    for cell_json_rel in df["cell_json"].unique():
        meta = json.loads((run.run_root / cell_json_rel).read_text(encoding="utf-8"))
        windows = meta["windows"]
        indexed_keys = set(df[df["cell_json"] == cell_json_rel]["window_key"])
        assert indexed_keys == set(windows)
        for entry in windows.values():
            assert set(entry) == {
                "dataset", "seed", "rep", "budget_r", "rate_frac", "t_start", "t_end",
            }
            # F1 cells: pressure coords stay ABSENT (null), never 0.
            assert entry["budget_r"] is None and entry["rate_frac"] is None
            assert entry["t_end"] > entry["t_start"]


def test_pressure_cell_coords_flow_from_spec_to_windows_table(tmp_path: Path) -> None:
    run_root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    run = cl.CampaignRun.create(run_root, **_manifest_kwargs())
    spec = CellSpec(
        "gold-fresh", "none", "none", "single", "vllm", MODEL, "F2",
        budget_r=0.5, rate_frac=0.8,
    )
    handle = _add_window(run.cell(spec), "squad_v2")
    assert "r0.5" in handle.row_key and "lam0.8" in handle.row_key
    meta = json.loads(
        (run_root / "cells" / handle.row_key / "cell.json").read_text(encoding="utf-8")
    )
    entry = meta["windows"][handle.window_key]
    assert entry["budget_r"] == 0.5 and entry["rate_frac"] == 0.8
    run.seal()
    csv_path, _ = org.organize_run(run_root)
    df = pd.read_csv(csv_path)
    assert df["budget_r"].tolist() == [0.5] and df["rate_frac"].tolist() == [0.8]


def test_model_mismatch_cell_refused_at_write_time(tmp_path: Path) -> None:
    run = _build_run(tmp_path)
    other = CellSpec.from_baseline("B1", model="llama-3.3-70b")
    with pytest.raises(cl.CampaignLayoutError, match="one run = one model"):
        run.cell(other)


# ---------------------------------------------------------------------------
# §3 manifest fail-closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"campaign": ""}, "campaign"),
        ({"session": "z"}, "session"),
        ({"model": "gpt-x"}, "roster"),
        ({"engine": "triton"}, "engine"),
        ({"engine_version": ""}, "engine_version"),
        ({"provider": ""}, "provider"),
        ({"hardware": ""}, "hardware"),
        ({"dataset_manifests_sha256": "nothex"}, "sha256"),
        ({"seed": True}, "seed"),
        ({"seed": -1}, "seed"),
        ({"cellspec_schema_version": 0}, "cellspec_schema_version"),
        ({"run_id": "BAD_ID"}, "grammar"),
    ],
)
def test_manifest_refuses_bad_required_fields(
    tmp_path: Path, overrides: dict[str, Any], match: str
) -> None:
    run_root = tmp_path / RUN_ID
    run_root.mkdir()
    with pytest.raises(cl.CampaignLayoutError, match=match):
        cl.write_manifest(run_root, **_manifest_kwargs(**overrides))
    assert not (run_root / "manifest.json").exists()


def test_manifest_run_id_must_match_dirname(tmp_path: Path) -> None:
    run_root = tmp_path / "some-other-dir"
    run_root.mkdir()
    with pytest.raises(cl.CampaignLayoutError, match="directory name"):
        cl.write_manifest(run_root, **_manifest_kwargs())


def test_manifest_refuses_overwrite_amended_never(tmp_path: Path) -> None:
    run_root = tmp_path / RUN_ID
    run_root.mkdir()
    cl.write_manifest(run_root, **_manifest_kwargs())
    with pytest.raises(cl.CampaignLayoutError, match="amended never"):
        cl.write_manifest(run_root, **_manifest_kwargs())


def test_manifest_git_provenance_computed_and_failure_is_loud(tmp_path: Path) -> None:
    run_root = tmp_path / RUN_ID
    run_root.mkdir()
    cl.write_manifest(run_root, **_manifest_kwargs())
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["git_sha"] == "deadbeef" * 5
    assert manifest["git_dirty"] is False
    assert manifest["created_utc"]
    for field in cl.MANIFEST_REQUIRED_FIELDS:
        assert manifest.get(field) not in (None, ""), field

    bad_root = tmp_path / "20260814-1201-a-qwen3-14b"
    bad_root.mkdir()
    with pytest.raises(cl.CampaignLayoutError, match="git_sha"):
        cl.write_manifest(
            bad_root,
            **_manifest_kwargs(
                run_id=bad_root.name, git_provenance=lambda _r: ("", False)
            ),
        )


def test_manifest_extra_cannot_shadow_required_fields(tmp_path: Path) -> None:
    run_root = tmp_path / RUN_ID
    run_root.mkdir()
    with pytest.raises(cl.CampaignLayoutError, match="shadow"):
        cl.write_manifest(
            run_root, **_manifest_kwargs(extra={"git_sha": "spoofed"})
        )
    # Legit extra keys (e.g. the organizer's optional datasets narrowing) pass.
    cl.write_manifest(run_root, **_manifest_kwargs(extra={"datasets": list(DATASETS)}))
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["datasets"] == list(DATASETS)


# ---------------------------------------------------------------------------
# Atomicity: tmp + os.replace, no residue
# ---------------------------------------------------------------------------


def test_no_tmp_residue_after_writes(tmp_path: Path) -> None:
    run = _build_run(tmp_path)
    assert not list(run.run_root.rglob("*.tmp"))


def test_failed_jsonl_write_leaves_neither_tmp_nor_artifact(tmp_path: Path) -> None:
    target = tmp_path / "rows.jsonl"
    with pytest.raises(cl.CampaignLayoutError, match="not JSON-serializable"):
        cl._atomic_write_jsonl(target, [{"ok": 1}, {"bad": object()}])
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_seal_refuses_crash_residue(tmp_path: Path) -> None:
    run = _build_run(tmp_path)
    stray = next(run.run_root.glob("cells/*/window_*")) / "requests.jsonl.tmp"
    stray.write_text("half a row", encoding="utf-8")
    with pytest.raises(cl.CampaignLayoutError, match="crash residue"):
        run.seal()


# ---------------------------------------------------------------------------
# §5 seal
# ---------------------------------------------------------------------------


def test_seal_then_verify_ledger_green(tmp_path: Path) -> None:
    run = _build_run(tmp_path)
    ledger_path = run.seal()
    assert ledger_path == run.run_root / "ledger.json"
    assert verify_ledger(ledger_path, run.run_root) == []
    entries = json.loads(ledger_path.read_text(encoding="utf-8"))["entries"]
    # §5: every artifact under cells/ plus manifest.json, keys run-root-relative.
    assert "manifest.json" in entries
    on_disk = {
        p.relative_to(run.run_root).as_posix()
        for p in run.run_root.glob("cells/**/*")
        if p.is_file()
    }
    assert set(entries) == on_disk | {"manifest.json"}


def test_seal_refuses_reseal_and_post_seal_writes(tmp_path: Path) -> None:
    run = _build_run(tmp_path)
    run.seal()
    with pytest.raises(LedgerError, match="sealed"):
        run.seal()
    with pytest.raises(cl.CampaignLayoutError, match="sealed"):
        run.cell(_specs()[0])


def test_seal_refuses_empty_run(tmp_path: Path) -> None:
    run_root = tmp_path / RUN_ID
    run_root.mkdir()
    cl.write_manifest(run_root, **_manifest_kwargs())
    (run_root / "cells").mkdir()
    with pytest.raises(cl.CampaignLayoutError, match="seals nothing"):
        cl.seal_run(run_root)


# ---------------------------------------------------------------------------
# Uniqueness invariant: (row_key, dataset, ordinal) never emitted twice
# ---------------------------------------------------------------------------


def test_duplicate_window_ordinal_refused(tmp_path: Path) -> None:
    run_root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    run = cl.CampaignRun.create(run_root, **_manifest_kwargs())
    cell = run.cell(_specs()[0])
    handle = _add_window(cell, "squad_v2")
    assert handle.ordinal == 1
    with pytest.raises(cl.CampaignLayoutError, match="already emitted"):
        _add_window(cell, "squad_v2", ordinal=1)
    # Ordinals are per-dataset: another dataset restarts at 01.
    assert _add_window(cell, "hotpotqa").window_key == "hotpotqa-01"


def test_h12_zero_padding_alias_refused(tmp_path: Path) -> None:
    """window_x-1 vs window_x-01 both parse to ordinal 1 (int(group(2))) but
    diverge on window_key — the writer must refuse the alias pair."""
    run_root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    run = cl.CampaignRun.create(run_root, **_manifest_kwargs())
    spec = _specs()[0]
    unpadded = run_root / "cells" / spec.to_row_key() / "window_squad_v2-1"
    unpadded.mkdir(parents=True)
    cell = run.cell(spec)  # scan registers (squad_v2, 1) from the unpadded dir
    with pytest.raises(cl.CampaignLayoutError, match="already emitted"):
        _add_window(cell, "squad_v2", ordinal=1)
    # Auto-minting continues PAST the registered ordinal, canonical %02d.
    assert _add_window(cell, "squad_v2").window_key == "squad_v2-02"


def test_preexisting_alias_pair_on_disk_refused_at_attach(tmp_path: Path) -> None:
    run_root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    run = cl.CampaignRun.create(run_root, **_manifest_kwargs())
    spec = _specs()[0]
    cell_dir = run_root / "cells" / spec.to_row_key()
    (cell_dir / "window_squad_v2-1").mkdir(parents=True)
    (cell_dir / "window_squad_v2-01").mkdir()
    with pytest.raises(cl.CampaignLayoutError, match="alias"):
        run.cell(spec)


# ---------------------------------------------------------------------------
# qa_evidence: required except for the sharegpt load donor (§1)
# ---------------------------------------------------------------------------


def test_qa_evidence_required_except_sharegpt(tmp_path: Path) -> None:
    run_root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    run = cl.CampaignRun.create(run_root, **_manifest_kwargs())
    cell = run.cell(_specs()[0])
    with pytest.raises(cl.CampaignLayoutError, match="qa_evidence"):
        _add_window(cell, "squad_v2", qa_evidence=None)
    handle = _add_window(cell, "sharegpt", qa_evidence=None)
    assert not (handle.window_dir / "qa_evidence.jsonl").exists()
    _add_window(cell, "squad_v2")  # cover the floor dataset, then round-trip
    run.seal()
    csv_path, _ = org.organize_run(run.run_root)
    assert "sharegpt-01" in set(pd.read_csv(csv_path)["window_key"])


# ---------------------------------------------------------------------------
# §6.1 regime bridge (first production caller of compute_window_regime_inputs)
# ---------------------------------------------------------------------------


def _one_window(tmp_path: Path, cage_stats: list[dict[str, Any]]) -> cl.WindowHandle:
    run_root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    run = cl.CampaignRun.create(run_root, **_manifest_kwargs())
    return _add_window(run.cell(_specs()[0]), "squad_v2", cage_stats=cage_stats)


@pytest.mark.parametrize(
    "telemetry", [_ZOH_LEGACY, _ZOH_CANONICAL], ids=["legacy-fields", "canonical-fields"]
)
def test_regime_pinned_zoh_case_both_schemas(
    tmp_path: Path, telemetry: list[dict[str, Any]]
) -> None:
    handle = _one_window(tmp_path, telemetry)
    path = cl.write_window_regime(handle.window_dir, t_start=0.0, t_end=10.0)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["telemetry_ok"] is True
    assert doc["refusal_reason"] is None
    assert doc["inputs"]["rho_kv_time_avg"] == pytest.approx(0.7)
    assert doc["inputs"]["scarcity_events"] == 4
    assert doc["inputs"]["n_samples"] == 3
    assert doc["inputs"]["coverage"] == pytest.approx(0.8)
    # No attainment yet -> §6.1 labeling deferred, never fabricated.
    assert doc["label"] is None and doc["attainment"] is None


def test_regime_label_with_attainment(tmp_path: Path) -> None:
    handle = _one_window(tmp_path, _ZOH_LEGACY)
    path = cl.write_window_regime(
        handle.window_dir, t_start=0.0, t_end=10.0, attainment=0.95
    )
    doc = json.loads(path.read_text(encoding="utf-8"))
    # rho 0.7 < 0.9 with attainment 0.95 -> UNPRESSURED (goodput thresholds).
    assert doc["label"] == UNPRESSURED and doc["attainment"] == 0.95


def test_regime_in_regime_label(tmp_path: Path) -> None:
    telemetry = [
        {"ts_s": 2.0, "kv_cache_usage": 0.95, "preemptions_total": 0},
        {"ts_s": 6.0, "kv_cache_usage": 0.95, "preemptions_total": 1},
        {"ts_s": 8.0, "kv_cache_usage": 0.95, "preemptions_total": 3},
    ]
    handle = _one_window(tmp_path, telemetry)
    path = cl.write_window_regime(
        handle.window_dir, t_start=0.0, t_end=10.0, attainment=0.95
    )
    assert json.loads(path.read_text(encoding="utf-8"))["label"] == IN_REGIME


def test_regime_refusal_absence_stays_absence(tmp_path: Path) -> None:
    telemetry = [
        {"ts_s": 2.0, "kv_cache_usage": 0.5, "preemptions_total": 5},
        {"ts_s": 6.0, "kv_cache_usage": None, "preemptions_total": 5},
        {"ts_s": 8.0, "kv_cache_usage": 0.8, "preemptions_total": 9},
    ]
    handle = _one_window(tmp_path, telemetry)
    path = cl.write_window_regime(handle.window_dir, t_start=0.0, t_end=10.0)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["telemetry_ok"] is False
    assert doc["label"] == REGIME_UNKNOWN
    assert doc["inputs"] is None
    assert "absence is not zero" in doc["refusal_reason"]


def test_regime_empty_series_is_refusal_not_zero(tmp_path: Path) -> None:
    handle = _one_window(tmp_path, [])
    path = cl.write_window_regime(handle.window_dir, t_start=0.0, t_end=10.0)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["telemetry_ok"] is False and doc["label"] == REGIME_UNKNOWN


def test_regime_caller_bugs_raise_not_refuse(tmp_path: Path) -> None:
    handle = _one_window(tmp_path, _ZOH_LEGACY)
    with pytest.raises(cl.CampaignLayoutError, match="t_end"):
        cl.write_window_regime(handle.window_dir, t_start=10.0, t_end=0.0)
    with pytest.raises(GoodputError):
        cl.write_window_regime(
            handle.window_dir, t_start=0.0, t_end=10.0, attainment=1.5
        )
    with pytest.raises(cl.CampaignLayoutError, match="not found"):
        cl.write_window_regime(
            handle.window_dir,
            t_start=0.0,
            t_end=10.0,
            telemetry_path=handle.window_dir / "nope.jsonl",
        )


def test_load_telemetry_series_prefers_canonical_fields(tmp_path: Path) -> None:
    path = tmp_path / "series.jsonl"
    path.write_text(
        json.dumps(
            {
                "ts_s": 2.0,
                "ts": 999.0,
                "kv_cache_usage": 0.5,
                "kv_usage": 0.0,
                "preemptions_total": 5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    frame = cl.load_telemetry_series(path)
    assert frame["ts_s"].tolist() == [2.0]
    assert frame["kv_cache_usage"].tolist() == [0.5]


def test_load_telemetry_series_refuses_timestampless_record(tmp_path: Path) -> None:
    path = tmp_path / "series.jsonl"
    path.write_text(json.dumps({"kv_usage": 0.5}) + "\n", encoding="utf-8")
    with pytest.raises(cl.CampaignLayoutError, match="timestamp"):
        cl.load_telemetry_series(path)


# ---------------------------------------------------------------------------
# Telemetry dual-field emission (save_series canonical + legacy names)
# ---------------------------------------------------------------------------


def _sampler_with(samples: list[dict[str, Any]], ts: list[float]) -> VllmTelemetrySampler:
    sampler = VllmTelemetrySampler("http://localhost:9")
    sampler._samples = list(samples)
    sampler._sample_ts = list(ts)
    return sampler


def test_save_series_emits_canonical_alongside_legacy(tmp_path: Path) -> None:
    sampler = _sampler_with(
        [
            {"kv_usage": 0.5, "preemptions_total": 5},
            {"kv_usage": 1.0, "preemptions_total": 5},
            {"kv_usage": 0.8, "preemptions_total": 9},
        ],
        [2.0, 6.0, 8.0],
    )
    out = tmp_path / "telemetry_series.jsonl"
    assert sampler.save_series(str(out)) == str(out)
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == 3
    for rec in records:
        assert rec["ts_s"] == rec["ts"]
        assert rec["kv_cache_usage"] == rec["kv_usage"]
        assert "preemptions_total" in rec
        # T4.1 (deliberate pin update): the single-sampler path now forward-
        # writes the optional instance role tag; "single" is its reserved
        # default. Legacy files without the field stay readable — see
        # tests/test_multi_instance_telemetry.py for the differential pins.
        assert rec["instance"] == "single"


def test_save_series_absent_gauge_stays_absent(tmp_path: Path) -> None:
    sampler = _sampler_with([{"preemptions_total": 5}], [2.0])
    out = tmp_path / "telemetry_series.jsonl"
    sampler.save_series(str(out))
    rec = json.loads(out.read_text().splitlines()[0])
    assert "kv_cache_usage" not in rec and "kv_usage" not in rec
    assert rec["ts_s"] == rec["ts"] == 2.0


def test_save_series_roundtrips_into_regime_inputs(tmp_path: Path) -> None:
    """End-to-end H1 closure: sampler output -> loader -> pinned ZOH numbers."""
    from src.analysis.regime_inputs import compute_window_regime_inputs

    sampler = _sampler_with(
        [
            {"kv_usage": 0.5, "preemptions_total": 5},
            {"kv_usage": 1.0, "preemptions_total": 5},
            {"kv_usage": 0.8, "preemptions_total": 9},
        ],
        [2.0, 6.0, 8.0],
    )
    out = tmp_path / "telemetry_series.jsonl"
    sampler.save_series(str(out))
    frame = cl.load_telemetry_series(out)
    inputs = compute_window_regime_inputs(frame, 0.0, 10.0)
    assert inputs.rho_kv_time_avg == pytest.approx(0.7)
    assert inputs.scarcity_events == 4


# ---------------------------------------------------------------------------
# W4.2 — gpu_count producer (CellWriter persists it; NAMED write-time
# refusals; end-to-end: a produced cell.json feeds contrast #18 past the
# "no gpu_count" labeled skip)
# ---------------------------------------------------------------------------


def _dist_spec(topology: str) -> CellSpec:
    # The #18 transfer pair rides B3 (corpus-reuse) on the DIST overlay.
    return CellSpec.from_baseline(
        "B3", model=MODEL, family="DIST", topology=topology  # type: ignore[arg-type]
    )


def test_cell_json_persists_gpu_count(tmp_path: Path) -> None:
    run_root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    run = cl.CampaignRun.create(run_root, **_manifest_kwargs())
    cell = run.cell(_dist_spec("tp"), gpu_count=8)
    _add_window(cell, "squad_v2")
    meta = json.loads((cell.cell_dir / "cell.json").read_text(encoding="utf-8"))
    # top-level int under the EXACT key the analysis consumer reads
    # (run_campaign_analysis._GPU_COUNT_CELL_KEY == "gpu_count")
    assert meta["gpu_count"] == 8
    # a single-topology cell WITHOUT a count keeps the key ABSENT (absence
    # stays absence — the consumer's labeled skip is the honest outcome)
    single = run.cell(_specs()[0])
    _add_window(single, "squad_v2")
    meta2 = json.loads((single.cell_dir / "cell.json").read_text(encoding="utf-8"))
    assert "gpu_count" not in meta2


def test_gpu_count_write_time_refusals_are_named(tmp_path: Path) -> None:
    run_root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    run = cl.CampaignRun.create(run_root, **_manifest_kwargs())
    # tp/pd cell with NO count: underivable — refuse, never a 1-GPU guess
    with pytest.raises(cl.CampaignLayoutError, match="NO\\s+gpu_count"):
        run.cell(_dist_spec("tp"))
    # topology/count mismatch: a distributed overlay cell on < 2 GPUs
    with pytest.raises(cl.CampaignLayoutError, match="topology/count contradiction"):
        run.cell(_dist_spec("pd"), gpu_count=1)
    # malformed counts refuse (bool is not an int count; 0 is not a stack)
    with pytest.raises(cl.CampaignLayoutError, match="integer >= 1"):
        run.cell(_specs()[0], gpu_count=0)
    with pytest.raises(cl.CampaignLayoutError, match="integer >= 1"):
        run.cell(_specs()[1], gpu_count=True)


def test_gpu_count_resume_contradiction_refuses_and_adoption_works(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    run = cl.CampaignRun.create(run_root, **_manifest_kwargs())
    spec = _dist_spec("tp")
    cell = run.cell(spec, gpu_count=8)
    _add_window(cell, "squad_v2")
    # a fresh writer with a CONTRADICTING claim refuses (two claims about
    # one cell's hardware cannot both be true)
    with pytest.raises(cl.CampaignLayoutError, match="contradicts"):
        cl.CellWriter(run_root, spec, gpu_count=4)
    # a memoized writer re-requested with a different count refuses too
    with pytest.raises(cl.CampaignLayoutError, match="contradicts"):
        run.cell(spec, gpu_count=4)
    # resume with NO fresh claim adopts the recorded fact and keeps it
    resumed = cl.CellWriter(run_root, spec)
    assert resumed.gpu_count == 8
    _add_window(resumed, "squad_v2", rep=2)
    meta = json.loads((resumed.cell_dir / "cell.json").read_text(encoding="utf-8"))
    assert meta["gpu_count"] == 8


def test_produced_cell_json_feeds_contrast_18_past_the_skip(tmp_path: Path) -> None:
    """W4.2 end-to-end: producer → cell.json → the #18 consumer's read path.

    Before this producer existed, run_campaign_analysis._dist_window_metrics
    found no ``gpu_count`` in any cell.json and every DIST window was the
    labeled '§6.6b basis undefined' skip. This test walks the actual chain
    on fixture windows: CellWriter persists the count, the consumer's exact
    read (top-level int key) recovers it, evaluate_window stamps the per-GPU
    basis, and execute_contrast_18 pairs tp-vs-pd instead of skipping.
    """
    from src.analysis.dist_contrasts import execute_contrast_18
    from src.analysis.goodput import SLOBaseline, evaluate_window

    run_root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    run = cl.CampaignRun.create(run_root, **_manifest_kwargs())
    handles: dict[str, Any] = {}
    for topology in ("tp", "pd"):
        cell = run.cell(_dist_spec(topology), gpu_count=8)
        handles[topology] = _add_window(
            cell,
            "squad_v2",
            requests=[
                {"example_id": "e0", "ok": True, "ttft_ms": 100.0, "tpot_ms": 10.0},
                {"example_id": "e1", "ok": True, "ttft_ms": 120.0, "tpot_ms": 12.0},
            ],
        )

    baseline = SLOBaseline(ttft_s=0.1, tpot_s=0.01)
    windows = []
    for topology, handle in handles.items():
        # the consumer's EXACT read: cell.json top-level "gpu_count" as int
        meta = json.loads(
            (run_root / "cells" / handle.row_key / "cell.json").read_text(
                encoding="utf-8"
            )
        )
        gpu_count = meta.get("gpu_count")
        assert isinstance(gpu_count, int) and not isinstance(gpu_count, bool)
        records = pd.DataFrame(
            [
                {"ok": True, "veridical": True, "ttft_s": 0.1, "tpot_s": 0.01},
                {"ok": True, "veridical": True, "ttft_s": 0.12, "tpot_s": 0.012},
            ]
        )
        metrics = evaluate_window(
            records, baseline, duration_s=10.0, gpu_count=gpu_count
        )
        windows.append(
            {
                "arm": "corpus-reuse",
                "retriever": "none",
                "policy": "none",
                "engine": "vllm",
                "model": MODEL,
                "dataset": "squad_v2",
                "budget_r": None,
                "rate_frac": None,
                "replicate": 1,
                "topology": topology,
                "window": handle.window_key,
                "gpu_count": gpu_count,
                "metrics": metrics,
            }
        )

    section = execute_contrast_18(windows)
    # past the skip: the pair EXECUTES on basis (b), and no skip names the
    # missing gpu_count any more
    assert section["status"] == "EXECUTED"
    assert len(section["pairs"]) == 1
    assert section["pairs"][0]["gpu_count_tp"] == 8
    assert section["pairs"][0]["gpu_count_pd"] == 8
    assert all("gpu_count" not in s["reason"] for s in section["skips"])


# ---------------------------------------------------------------------------
# W4.2/W4.4 — the campaign_session seam: CAGE_GPU_COUNT / the per-task
# window-ordinal base (run_campaign threads both; the session parses and
# applies them)
# ---------------------------------------------------------------------------


def _session_args(root: Path, dataset: str = "ruler") -> Any:
    import types

    return types.SimpleNamespace(
        campaign_root=str(root),
        top_k_sweep=False,
        baseline="no_cache",
        baseline_label="B1_gold-fresh",
        backend="vllm",
        model="Qwen/Qwen3-14B",
        dataset=dataset,
        num_trials=3,
        seed=1,
        kv_cache_dtype=None,
    )


def test_session_parses_gpu_count_and_ordinal_base_from_env(tmp_path: Path) -> None:
    from src.orchestration.campaign_session import CampaignCellSession

    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    session = CampaignCellSession.from_cli(
        _session_args(root),
        env={"CAGE_GPU_COUNT": "8", "CAGE_WINDOW_ORDINAL_BASE": "6"},
    )
    assert session is not None
    assert session.gpu_count == 8
    assert session.window_ordinal_base == 6
    # trial ordinals shift into the step's claimed range: trial 1 -> ruler-07
    assert session.window_key(1) == "ruler-07"
    assert session.window_dir(3).name == "window_ruler-09"
    # unset envs keep the pre-W4.2/W4.4 behavior byte-identical
    legacy = CampaignCellSession.from_cli(_session_args(root), env={})
    assert legacy is not None
    assert legacy.gpu_count is None
    assert legacy.window_ordinal_base == 0
    assert legacy.window_key(1) == "ruler-01"


def test_session_refuses_malformed_seam_env(tmp_path: Path) -> None:
    from src.orchestration.campaign_session import (
        CampaignCellSession,
        CampaignSessionError,
    )

    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    with pytest.raises(CampaignSessionError, match="CAGE_GPU_COUNT"):
        CampaignCellSession.from_cli(
            _session_args(root), env={"CAGE_GPU_COUNT": "eight"}
        )
    with pytest.raises(CampaignSessionError, match="CAGE_GPU_COUNT"):
        CampaignCellSession.from_cli(
            _session_args(root), env={"CAGE_GPU_COUNT": "0"}
        )
    with pytest.raises(CampaignSessionError, match="CAGE_WINDOW_ORDINAL_BASE"):
        CampaignCellSession.from_cli(
            _session_args(root), env={"CAGE_WINDOW_ORDINAL_BASE": "-3"}
        )
