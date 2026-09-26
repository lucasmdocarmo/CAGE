"""Unit pins for src/orchestration/campaign_session.py (the task #116 seam).

Covers the pieces the end-to-end proof (tests/test_run_experiment_campaign_layout.py)
exercises only implicitly: cell-identity derivation (legacy labels, explicit
CAGE_CELL_* axes, fail-closed refusals), model-slug resolution, the write-time
hash journal + seal cross-check (S0-15), per-window resume reset semantics,
and the import-light cell-dir CLI the shell resume gates call.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.cellspec import CellSpec  # noqa: E402
from src.orchestration import campaign_layout as cl  # noqa: E402
from src.orchestration import campaign_session as cs  # noqa: E402

RUN_ID = "20260821-1300-a-qwen3-14b"


def _run_root(tmp_path: Path) -> Path:
    return tmp_path / "results" / "camp1" / "a" / RUN_ID


def _fake_git(_repo: Path) -> tuple[str, bool]:
    return "deadbeef" * 5, False


def _session(tmp_path: Path, **overrides: Any) -> cs.CampaignCellSession:
    kwargs: dict[str, Any] = {
        "run_root": _run_root(tmp_path),
        "dataset": "squad_v2",
        "spec": CellSpec.from_baseline("B1", model="qwen3-14b"),
        "num_trials": 2,
        "run_seed": 7,
    }
    kwargs.update(overrides)
    return cs.CampaignCellSession(**kwargs)


# ---------------------------------------------------------------------------
# Model slug + engine mapping
# ---------------------------------------------------------------------------


def test_model_slug_roster_and_override() -> None:
    assert cs.resolve_model_slug("Qwen/Qwen3-14B", {}) == "qwen3-14b"
    assert cs.resolve_model_slug("qwen3-14b", {}) == "qwen3-14b"  # already a slug
    assert (
        cs.resolve_model_slug("Qwen/Qwen3-8B", {"CAGE_MODEL_SLUG": "qwen3-14b"})
        == "qwen3-14b"
    )  # explicit stand-in labeling (design-input runs)


def test_model_slug_unknown_refuses() -> None:
    with pytest.raises(cs.CampaignSessionError, match="CAGE_MODEL_SLUG"):
        cs.resolve_model_slug("Qwen/Qwen3-8B", {})
    with pytest.raises(cs.CampaignSessionError):
        cs.resolve_model_slug("x", {"CAGE_MODEL_SLUG": "not-a-roster-model"})


def test_derive_from_legacy_labels() -> None:
    spec = cs.derive_cell_spec(
        baseline="redis",
        baseline_label="redis_retrieval_cache_cold",
        backend="vllm",
        model="qwen3-14b",
        env={},
    )
    # redis is §7.5-retired onto the ranked retr-fresh tuple (≈ B6).
    assert spec.to_row_key() == "retr-fresh|rerank|none|single|vllm|qwen3-14b|F1"
    spec2 = cs.derive_cell_spec(
        baseline="compressed_cag",
        baseline_label=None,
        backend="vllm",
        model="qwen3-14b",
        env={},
    )
    assert spec2.policy == "compress-fp8" and spec2.family == "F3"


def test_derive_explicit_axes_override_legacy() -> None:
    env = {
        "CAGE_CELL_ARM": "retr-fresh",
        "CAGE_CELL_RETRIEVER": "bm25",
        "CAGE_CELL_FAMILY": "F2",
        "CAGE_CELL_BUDGET_R": "0.5",
        "CAGE_CELL_RATE_FRAC": "0.8",
    }
    spec = cs.derive_cell_spec(
        baseline="no_cache", baseline_label=None, backend="sglang",
        model="qwen3-14b", env=env,
    )
    assert spec.to_row_key() == "retr-fresh|bm25|none|single|sglang|qwen3-14b|F2|r0.5|lam0.8"


def test_derive_refusals_are_fail_closed() -> None:
    # Arm without its retriever axis.
    with pytest.raises(cs.CampaignSessionError, match="must be set together"):
        cs.derive_cell_spec(
            baseline="no_cache", baseline_label=None, backend="vllm",
            model="qwen3-14b", env={"CAGE_CELL_ARM": "gold-fresh"},
        )
    # Legacy-unmapped label (pilot-only cell).
    with pytest.raises(cs.CampaignSessionError, match="LEGACY_ALIASES"):
        cs.derive_cell_spec(
            baseline="not_a_baseline", baseline_label="distributed_router_replicated",
            backend="vllm", model="qwen3-14b", env={},
        )
    # No charter engine for legacy backends.
    with pytest.raises(cs.CampaignSessionError, match="charter engine"):
        cs.derive_cell_spec(
            baseline="no_cache", baseline_label=None, backend="ollama",
            model="qwen3-14b", env={},
        )
    # Charter-illegal combination surfaces CellSpec's own gate.
    with pytest.raises(cs.CampaignSessionError, match="charter-illegal"):
        cs.derive_cell_spec(
            baseline="no_cache", baseline_label=None, backend="hf-oracle",
            model="qwen3-14b", env={"CAGE_CELL_FAMILY": "F2"},
        )


# ---------------------------------------------------------------------------
# Session construction refusals
# ---------------------------------------------------------------------------


def test_session_refuses_non_roster_dataset(tmp_path: Path) -> None:
    with pytest.raises(cs.CampaignSessionError, match="dataset id"):
        _session(tmp_path, dataset="trivia_qa")


def test_session_refuses_bad_root_shape(tmp_path: Path) -> None:
    with pytest.raises(cs.CampaignSessionError, match="session"):
        cs.CampaignCellSession(
            run_root=tmp_path / "results" / "camp1" / "nope" / RUN_ID,
            dataset="squad_v2",
            spec=CellSpec.from_baseline("B1", model="qwen3-14b"),
            num_trials=1,
            run_seed=7,
        )
    with pytest.raises(cs.CampaignSessionError, match="grammar"):
        cs.CampaignCellSession(
            run_root=tmp_path / "results" / "camp1" / "a" / "Bad_Run_Id",
            dataset="squad_v2",
            spec=CellSpec.from_baseline("B1", model="qwen3-14b"),
            num_trials=1,
            run_seed=7,
        )


def test_from_cli_inactive_returns_none() -> None:
    class _Args:
        campaign_root = None
        baseline = "no_cache"
        baseline_label = None
        backend = "vllm"
        model = "qwen3-14b"
        dataset = "squad_v2"
        num_trials = 1
        seed = 7
        kv_cache_dtype = None
        top_k_sweep = False

    assert cs.CampaignCellSession.from_cli(_Args(), env={}) is None


def test_from_cli_refuses_top_k_sweep(tmp_path: Path) -> None:
    class _Args:
        campaign_root = str(_run_root(tmp_path))
        baseline = "rag"
        baseline_label = None
        backend = "vllm"
        model = "qwen3-14b"
        dataset = "squad_v2"
        num_trials = 1
        seed = 7
        kv_cache_dtype = None
        top_k_sweep = True

    with pytest.raises(cs.CampaignSessionError, match="top-k-sweep"):
        cs.CampaignCellSession.from_cli(_Args(), env={})


def test_from_cli_refuses_stale_index_escape_hatch(tmp_path: Path) -> None:
    # Backlog A6 / F6 (review 2026-09-17 defect 2): the campaign path must never
    # run under CAGE_ALLOW_STALE_INDEX (presence, any value).
    from src.orchestration.ir import STALE_INDEX_OPT_IN_ENV

    assert cs.STALE_INDEX_OPT_IN_ENV == STALE_INDEX_OPT_IN_ENV

    class _Args:
        campaign_root = str(_run_root(tmp_path))
        baseline = "no_cache"
        baseline_label = None
        backend = "vllm"
        model = "qwen3-14b"
        dataset = "squad_v2"
        num_trials = 1
        seed = 7
        kv_cache_dtype = None
        top_k_sweep = False

    for value in ("1", "0", ""):
        with pytest.raises(cs.CampaignSessionError, match="CAGE_ALLOW_STALE_INDEX"):
            cs.CampaignCellSession.from_cli(
                _Args(), env={"CAGE_GPU_COUNT": "1", cs.STALE_INDEX_OPT_IN_ENV: value},
            )


def test_emit_window_refuses_a_stale_index_summary(tmp_path: Path) -> None:
    # The runner persists ir_index.stale_index_opt_in into metrics.json
    # ["experiment"]; a True there means retrieval ran out-of-distribution and
    # the window is refused BEFORE any artifact is written.
    session = _session(tmp_path)
    for summary in (
        {"experiment": {"stale_index_opt_in": True}},
        {"experiment": {"stale_index_opt_in": "1"}},  # non-bool: never coerced
    ):
        with pytest.raises(cs.CampaignSessionError, match="stale_index_opt_in"):
            session.emit_window(
                ordinal=1, trial_seed=7, results_rows=[], staging_dir=tmp_path,
                experiment_summary=summary, backend_metadata={},
                telemetry_snapshot=None, t_start=0.0, t_end=1.0,
            )
    assert not (session.run_root / "manifest.json").exists()
    assert cs.refuse_stale_index_summary({"experiment": {"stale_index_opt_in": False}}) is None
    with pytest.raises(cs.CampaignSessionError, match="stale_index_opt_in"):
        cs.refuse_stale_index_summary({"experiment": {}})  # absent: not a valid window


# ---------------------------------------------------------------------------
# Write-time hash journal + seal cross-check (S0-15)
# ---------------------------------------------------------------------------


def _mini_tree(tmp_path: Path) -> Path:
    """One-cell one-window v2 tree via campaign_layout, fully journaled."""
    run_root = _run_root(tmp_path)
    run = cl.CampaignRun.create(
        run_root,
        campaign="camp1",
        session="a",
        run_id=RUN_ID,
        model="qwen3-14b",
        engine="vllm",
        engine_version="0.0-test",
        seed=1,
        provider="test",
        hardware="test-gpu",
        dataset_manifests_sha256="0" * 64,
        cellspec_schema_version=1,
        git_provenance=_fake_git,
    )
    cell = run.cell(CellSpec.from_baseline("B1", model="qwen3-14b"))
    handle = cell.add_window(
        "squad_v2",
        seed=1,
        rep=1,
        t_start=0.0,
        t_end=10.0,
        requests=[{"example_id": "e0", "ttft_ms": 1.0}],
        cage_stats=[],
        engine_metrics={"snapshot": "x"},
        qa_evidence=[{"example_id": "e0"}],
    )
    paths = [run_root / "manifest.json", handle.window_dir.parent / "cell.json"]
    paths += sorted(p for p in handle.window_dir.iterdir() if p.is_file())
    cs.append_write_time_hashes(run_root, paths)
    return run_root


def test_seal_green_on_journaled_tree(tmp_path: Path) -> None:
    run_root = _mini_tree(tmp_path)
    ledger = cs.seal_campaign_run(run_root)
    assert ledger.is_file()
    journal = cs.read_write_time_journal(run_root)
    sealed = json.loads(ledger.read_text(encoding="utf-8"))["entries"]
    assert set(sealed) <= set(journal)


def test_seal_refuses_post_write_mutation(tmp_path: Path) -> None:
    run_root = _mini_tree(tmp_path)
    victim = next(run_root.glob("cells/*/window_*/requests.jsonl"))
    victim.write_text('{"example_id": "e0", "ttft_ms": 999.0}\n', encoding="utf-8")
    with pytest.raises(cs.CampaignSessionError, match="write-time hash"):
        cs.seal_campaign_run(run_root)
    assert not (run_root / "ledger.json").exists(), "refusal must seal NOTHING"


def test_seal_refuses_unjournaled_file(tmp_path: Path) -> None:
    run_root = _mini_tree(tmp_path)
    stray = next(run_root.glob("cells/*/window_*")) / "extra.json"
    stray.write_text("{}", encoding="utf-8")
    with pytest.raises(cs.CampaignSessionError, match="not in "):
        cs.seal_campaign_run(run_root)


def test_seal_refuses_without_journal(tmp_path: Path) -> None:
    run_root = _mini_tree(tmp_path)
    (run_root / cs.JOURNAL_NAME).unlink()
    with pytest.raises(cs.CampaignSessionError, match=cs.JOURNAL_NAME):
        cs.seal_campaign_run(run_root)


def test_journal_superseded_entries_tolerated(tmp_path: Path) -> None:
    """A rewritten file (e.g. cell.json after every window) seals against its
    LAST journal entry; stale entries for deleted files are ignored."""
    run_root = _mini_tree(tmp_path)
    meta = next(run_root.glob("cells/*/cell.json"))
    doc = json.loads(meta.read_text(encoding="utf-8"))
    doc["baseline"] = doc.get("baseline", "")
    meta.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cs.append_write_time_hashes(run_root, [meta])  # the rewrite is journaled
    assert cs.seal_campaign_run(run_root).is_file()


# ---------------------------------------------------------------------------
# Per-window resume reset
# ---------------------------------------------------------------------------


def test_reset_incomplete_windows_drops_dir_and_declaration(tmp_path: Path) -> None:
    session = _session(tmp_path)
    cell_dir = session.cell_dir
    # Window 01: complete (valid metrics.json). Window 02: half-written (no
    # metrics.json). Window 03 of ANOTHER dataset: must never be touched.
    w1 = session.window_dir(1)
    w1.mkdir(parents=True)
    (w1 / "metrics.json").write_text('{"ok": 1}', encoding="utf-8")
    w2 = session.window_dir(2)
    w2.mkdir(parents=True)
    (w2 / "requests.jsonl").write_text('{"example_id": "e0"}\n', encoding="utf-8")
    other = cell_dir / "window_hotpotqa-01"
    other.mkdir(parents=True)
    (cell_dir / "cell.json").write_text(
        json.dumps(
            {
                "cellspec": session.spec.to_flat_dict(),
                "baseline": "B1",
                "windows": {
                    "squad_v2-01": {"dataset": "squad_v2"},
                    "squad_v2-02": {"dataset": "squad_v2"},
                    "hotpotqa-01": {"dataset": "hotpotqa"},
                },
            }
        ),
        encoding="utf-8",
    )
    assert session.reset_incomplete_windows() == [2]
    assert w1.is_dir() and not w2.exists() and other.is_dir()
    meta = json.loads((cell_dir / "cell.json").read_text(encoding="utf-8"))
    assert set(meta["windows"]) == {"squad_v2-01", "hotpotqa-01"}
    # The cell.json rewrite was journaled at write time (S0-15).
    journal = cs.read_write_time_journal(session.run_root)
    assert f"cells/{session.row_key}/cell.json" in journal


def test_window_complete_is_parse_checked(tmp_path: Path) -> None:
    session = _session(tmp_path)
    w1 = session.window_dir(1)
    w1.mkdir(parents=True)
    assert not session.window_complete(1)  # absent
    (w1 / "metrics.json").write_text('{"trunc', encoding="utf-8")
    assert not session.window_complete(1)  # invalid JSON = incomplete (J2)
    (w1 / "metrics.json").write_text('{"ok": 1}', encoding="utf-8")
    assert session.window_complete(1)


def test_reset_refuses_corrupt_cell_json(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.cell_dir.mkdir(parents=True)
    (session.cell_dir / "cell.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(cs.CampaignSessionError, match="CAGE_FORCE_RERUN"):
        session.reset_incomplete_windows()


# ---------------------------------------------------------------------------
# The cell-dir CLI (what the shell gates call via campaign_cell_dir)
# ---------------------------------------------------------------------------


def test_cell_dir_cli_prints_row_key_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CAGE_CAMPAIGN_ROOT", str(_run_root(tmp_path)))
    rc = cs.main(["cell-dir", "--baseline", "prefix_cache", "--model", "qwen3-14b"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out.endswith("cells/gold-reuse|none|none|single|vllm|qwen3-14b|F1")


def test_cell_dir_cli_refuses_unmapped_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CAGE_CAMPAIGN_ROOT", str(_run_root(tmp_path)))
    rc = cs.main(
        ["cell-dir", "--baseline", "nope", "--baseline-label",
         "distributed_router_replicated", "--model", "qwen3-14b"]
    )
    assert rc == 1
    assert "LEGACY_ALIASES" in capsys.readouterr().err


def test_cell_dir_cli_requires_campaign_root(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CAGE_CAMPAIGN_ROOT", raising=False)
    rc = cs.main(["cell-dir", "--baseline", "no_cache", "--model", "qwen3-14b"])
    assert rc == 2
    assert "CAGE_CAMPAIGN_ROOT" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# ADR-0106: the B12 rung rides the CAGE_CELL_CORPUS_BUDGET env seam
# ---------------------------------------------------------------------------


def _derive(env: dict[str, str]) -> CellSpec:
    return cs.derive_cell_spec(
        baseline="prefix_cache", baseline_label=None, backend="vllm",
        model="qwen3-14b", env=env,
    )


def test_derive_corpus_budget_round_trips_to_the_rung_row_key() -> None:
    spec = _derive({
        "CAGE_CELL_ARM": "corpus-trunc",
        "CAGE_CELL_RETRIEVER": "none",
        "CAGE_CELL_FAMILY": "F3",
        "CAGE_CELL_BUDGET_R": "0.5",
        "CAGE_CELL_RATE_FRAC": "0.8",
        "CAGE_CELL_CORPUS_BUDGET": "700",
    })
    assert spec.corpus_budget_tokens == 700
    assert spec.to_row_key() == (
        "corpus-trunc|none|none|single|vllm|qwen3-14b|F3|r0.5|lam0.8|cb700"
    )
    # Absent stays absent (never a default rung).
    plain = _derive({"CAGE_CELL_ARM": "corpus-trunc", "CAGE_CELL_RETRIEVER": "none"})
    assert plain.corpus_budget_tokens is None


def test_derive_corpus_budget_refusals_are_fail_closed() -> None:
    base = {"CAGE_CELL_ARM": "corpus-trunc", "CAGE_CELL_RETRIEVER": "none"}
    with pytest.raises(cs.CampaignSessionError, match="CAGE_CELL_CORPUS_BUDGET"):
        _derive({**base, "CAGE_CELL_CORPUS_BUDGET": "seven-hundred"})
    with pytest.raises(cs.CampaignSessionError, match="CAGE_CELL_CORPUS_BUDGET"):
        _derive({**base, "CAGE_CELL_CORPUS_BUDGET": "0"})
    # A rung on a non-trunc arm surfaces CellSpec's own gate.
    with pytest.raises(cs.CampaignSessionError, match="charter-illegal"):
        _derive({
            "CAGE_CELL_ARM": "corpus-reuse",
            "CAGE_CELL_RETRIEVER": "none",
            "CAGE_CELL_CORPUS_BUDGET": "1400",
        })


# ---------------------------------------------------------------------------
# ADR-0116 (Batch 2 W3, option a): the emit seam refuses a window whose runner
# summary counted a dropped row. A dropped row is absent from requests.jsonl
# and qa_evidence.jsonl alike, so the count reconciliation cannot see it; the
# consort counters are its only trace and every one of them must read zero.
# ---------------------------------------------------------------------------

ZERO_CONSORT: dict[str, Any] = {
    "n_dropped_prepare": 0,
    "n_dropped_record": 0,
    "n_dropped_turn": 0,
    "evidence_write_failures": 0,
    "evidence_write_first_error": None,
}


def test_refuse_dropped_rows_summary_enforces_every_consort_counter() -> None:
    assert cs.refuse_dropped_rows_summary({"consort": ZERO_CONSORT}) is None
    assert set(cs.CONSORT_COUNTERS) == set(ZERO_CONSORT) - {"evidence_write_first_error"}
    for key in cs.CONSORT_COUNTERS:
        with pytest.raises(cs.CampaignSessionError, match=key):
            cs.refuse_dropped_rows_summary({"consort": {**ZERO_CONSORT, key: 1}})
    # Absence is not zero: a missing block, a missing counter and a
    # non-integer value all refuse (never coerced).
    with pytest.raises(cs.CampaignSessionError, match="consort"):
        cs.refuse_dropped_rows_summary({})
    with pytest.raises(cs.CampaignSessionError, match="n_dropped_turn"):
        cs.refuse_dropped_rows_summary(
            {"consort": {k: v for k, v in ZERO_CONSORT.items() if k != "n_dropped_turn"}}
        )
    with pytest.raises(cs.CampaignSessionError, match="n_dropped_record"):
        cs.refuse_dropped_rows_summary({"consort": {**ZERO_CONSORT, "n_dropped_record": "0"}})
    with pytest.raises(cs.CampaignSessionError, match="n_dropped_prepare"):
        cs.refuse_dropped_rows_summary({"consort": {**ZERO_CONSORT, "n_dropped_prepare": True}})


def test_emit_window_refuses_dropped_rows_before_any_artifact(tmp_path: Path) -> None:
    session = _session(tmp_path)
    summary = {
        "experiment": {"stale_index_opt_in": False},
        "consort": {**ZERO_CONSORT, "n_dropped_record": 1},
    }
    with pytest.raises(cs.CampaignSessionError, match="n_dropped_record"):
        session.emit_window(
            ordinal=1, trial_seed=7, results_rows=[], staging_dir=tmp_path,
            experiment_summary=summary, backend_metadata={},
            telemetry_snapshot=None, t_start=0.0, t_end=1.0,
        )
    assert not (session.run_root / "manifest.json").exists()
    assert not session.cell_dir.exists()


# ---------------------------------------------------------------------------
# Batch 2 W4 (ADR-0117): the two pins run_campaign threads through the cell
# env. CAGE_SLO_FLOORS_JSON lands in manifest.json["slo_floors"] when the
# session creates the manifest and is compared on reopen; CAGE_BUDGET_PLAN_JSON
# lands in cell.json["budget_plan"] through CellWriter. Both parse fail-closed
# and the plan is cross-checked against the cell identity.
# ---------------------------------------------------------------------------

import types  # noqa: E402

FLOORS: dict[str, Any] = {
    "vllm": {"ttft_s": 0.12, "tpot_s": 0.02, "n_requests": 30, "statistic": "median"},
    "sglang": {"ttft_s": 0.15, "tpot_s": 0.025},
}

#: qwen3-14b BF16 KV = 163_840 B/token; 42 tokens' worth of budget lands the
#: canonical own-accounting rows (tests/test_own_accounting.py) on rho = 0.5.
BUDGET_42_TOK = 42 * 163_840


def _budget_plan_doc(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "model": "qwen3-14b",
        "engine": "vllm",
        "r": 0.5,
        "kv_dtype": "bf16",
        "demand_bytes": 2 * BUDGET_42_TOK,
        "budget_bytes_total": BUDGET_42_TOK,
        "budget_tokens_total": 42,
        "tp": 1,
        "topology": "single",
        "per_rank_bytes": BUDGET_42_TOK,
        "per_rank_note": "single rank: pool == total budget",
        "pd_split": None,
        "pools_bytes": None,
        "engine_args": [],
        "verify_live": [],
        "gate_j": {},
        "floor_table_sha256": "0" * 64,
    }
    doc.update(overrides)
    return doc


PRESSURE_ENV: dict[str, str] = {
    "CAGE_CELL_ARM": "gold-fresh",
    "CAGE_CELL_RETRIEVER": "none",
    "CAGE_CELL_FAMILY": "F2",
    "CAGE_CELL_BUDGET_R": "0.5",
    "CAGE_CELL_RATE_FRAC": "0.85",
    "CAGE_GPU_COUNT": "1",
}


def _pressure_spec() -> CellSpec:
    return CellSpec(
        "gold-fresh", "none", "none", "single", "vllm", "qwen3-14b", "F2",
        budget_r=0.5, rate_frac=0.85,
    )


def _pressure_args(root: Path) -> Any:
    return types.SimpleNamespace(
        campaign_root=str(root),
        top_k_sweep=False,
        baseline="no_cache",
        baseline_label="B1_gold-fresh",
        backend="vllm",
        model="Qwen/Qwen3-14B",
        dataset="qasper",
        num_trials=2,
        seed=1,
        kv_cache_dtype=None,
    )


def test_pin_literals_match_the_driver_and_the_consumer() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_campaign_w4_pins", REPO_ROOT / "scripts" / "3_run" / "run_campaign.py"
    )
    rc = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rc
    spec.loader.exec_module(rc)
    analysis_dir = str(REPO_ROOT / "scripts" / "4_analysis")
    if analysis_dir not in sys.path:
        sys.path.insert(0, analysis_dir)
    import run_campaign_analysis as rca

    assert cs.SLO_FLOORS_ENV == rc.SLO_FLOORS_ENV == "CAGE_SLO_FLOORS_JSON"
    assert cs.BUDGET_PLAN_ENV == rc.BUDGET_PLAN_ENV == "CAGE_BUDGET_PLAN_JSON"
    assert cs.SLO_FLOORS_MANIFEST_KEY == rca._SLO_FLOORS_MANIFEST_KEY == "slo_floors"
    assert (
        cs.BUDGET_PLAN_CELL_KEY == cl.BUDGET_PLAN_CELL_KEY
        == rca._BUDGET_PLAN_CELL_KEY == "budget_plan"
    )


def test_parse_slo_floors_refusals() -> None:
    assert cs.parse_slo_floors(json.dumps(FLOORS)) == FLOORS
    bad = [
        ("{not json", "not valid JSON"),
        ("[]", "JSON object"),
        ("{}", "empty"),
        (json.dumps({"hf": {"ttft_s": 0.1, "tpot_s": 0.1}}), "hf"),
        (json.dumps({"triton": {"ttft_s": 0.1, "tpot_s": 0.1}}), "triton"),
        (json.dumps({"vllm": {"ttft_s": 0.1}}), "tpot_s"),
        (json.dumps({"vllm": {"ttft_s": 0.0, "tpot_s": 0.1}}), "ttft_s"),
        (json.dumps({"vllm": {"ttft_s": True, "tpot_s": 0.1}}), "ttft_s"),
        (json.dumps({"vllm": {"ttft_s": "0.1", "tpot_s": 0.1}}), "ttft_s"),
        (json.dumps({"vllm": {"ttft_s": float("nan"), "tpot_s": 0.1}}), "ttft_s"),
        (json.dumps({"vllm": 0.1}), "vllm"),
    ]
    for raw, match in bad:
        with pytest.raises(cs.CampaignSessionError, match=match):
            cs.parse_slo_floors(raw)


def test_parse_budget_plan_refusals() -> None:
    spec = _pressure_spec()
    plan = cs.parse_budget_plan(json.dumps(_budget_plan_doc()), spec)
    assert plan["budget_bytes_total"] == BUDGET_42_TOK and plan["kv_dtype"] == "bf16"
    with pytest.raises(cs.CampaignSessionError, match="not valid JSON"):
        cs.parse_budget_plan("{nope", spec)
    with pytest.raises(cs.CampaignSessionError, match="JSON object"):
        cs.parse_budget_plan("[]", spec)
    bad_fields = [
        ({"budget_bytes_total": None}, "budget_bytes_total"),
        ({"budget_bytes_total": True}, "budget_bytes_total"),
        ({"budget_bytes_total": 0}, "budget_bytes_total"),
        ({"budget_bytes_total": "5"}, "budget_bytes_total"),
        ({"kv_dtype": "int4"}, "kv_dtype"),
        ({"kv_dtype": None}, "kv_dtype"),
        ({"model": "llama-3.3-70b"}, "model"),
        ({"engine": "sglang"}, "engine"),
        ({"r": 1.0}, "budget_r"),
        ({"topology": "pd"}, "topology"),
    ]
    for override, match in bad_fields:
        doc = _budget_plan_doc(**override)
        for key, value in override.items():
            if value is None:
                del doc[key]
        with pytest.raises(cs.CampaignSessionError, match=match):
            cs.parse_budget_plan(json.dumps(doc), spec)
    # A TP-sharded single-instance launch (session b: serving_tp=4) plans
    # topology 'tp' while the cellspec topology is 'single': both are true.
    sharded = _budget_plan_doc(topology="tp", tp=4, per_rank_bytes=BUDGET_42_TOK // 4)
    assert cs.parse_budget_plan(json.dumps(sharded), spec)["tp"] == 4
    # The DIST overlay carries no pressure coordinate: r is the registered
    # dist_budget_r and cannot be cross-checked, the topology can.
    dist_tp = CellSpec.from_baseline("B3", model="qwen3-14b", family="DIST", topology="tp")
    tp_plan = _budget_plan_doc(topology="tp", tp=8, r=1.0, per_rank_bytes=BUDGET_42_TOK // 8)
    assert cs.parse_budget_plan(json.dumps(tp_plan), dist_tp)["r"] == 1.0
    with pytest.raises(cs.CampaignSessionError, match="topology"):
        cs.parse_budget_plan(json.dumps(_budget_plan_doc(r=1.0)), dist_tp)


def test_from_cli_parses_the_pins_and_the_constructor_validates(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    env = {
        **PRESSURE_ENV,
        "CAGE_SLO_FLOORS_JSON": json.dumps(FLOORS),
        "CAGE_BUDGET_PLAN_JSON": json.dumps(_budget_plan_doc()),
    }
    session = cs.CampaignCellSession.from_cli(_pressure_args(root), env=env)
    assert session is not None
    assert session.slo_floors == FLOORS
    assert session.budget_plan == _budget_plan_doc()
    # unset pins keep the pre-W4 behavior: nothing recorded, nothing refused
    legacy = cs.CampaignCellSession.from_cli(_pressure_args(root), env=PRESSURE_ENV)
    assert legacy is not None
    assert legacy.slo_floors is None and legacy.budget_plan is None
    # malformed pins refuse naming the env
    with pytest.raises(cs.CampaignSessionError, match="CAGE_SLO_FLOORS_JSON"):
        cs.CampaignCellSession.from_cli(
            _pressure_args(root), env={**PRESSURE_ENV, "CAGE_SLO_FLOORS_JSON": "{}"},
        )
    with pytest.raises(cs.CampaignSessionError, match="CAGE_BUDGET_PLAN_JSON"):
        cs.CampaignCellSession.from_cli(
            _pressure_args(root),
            env={**PRESSURE_ENV, "CAGE_BUDGET_PLAN_JSON": json.dumps(_budget_plan_doc(engine="sglang"))},
        )
    # the constructor validates direct callers the same way
    with pytest.raises(cs.CampaignSessionError, match="ttft_s"):
        _session(tmp_path, slo_floors={"vllm": {"ttft_s": 0, "tpot_s": 0.1}})
    with pytest.raises(cs.CampaignSessionError, match="engine"):
        _session(tmp_path, spec=_pressure_spec(), budget_plan=_budget_plan_doc(engine="sglang"))
    # the in-process oracle has no budget: a plan on an hf cell is a
    # contradiction whatever its engine field says (review S2/T4: the
    # session refuses at activation, before CellWriter and before serving)
    hf = CellSpec.from_baseline("B3", model="qwen3-14b", engine="hf")
    with pytest.raises(cs.CampaignSessionError, match="oracle"):
        _session(tmp_path, spec=hf, budget_plan=_budget_plan_doc(engine="hf", r=1.0))
    with pytest.raises(cs.CampaignSessionError, match="oracle"):
        _session(tmp_path, spec=hf, budget_plan=_budget_plan_doc())
    # the pinned floors must include THIS cell's engine (review S3)
    sglang = CellSpec(
        "gold-fresh", "none", "none", "single", "sglang", "qwen3-14b", "F2",
        budget_r=0.5, rate_frac=0.85,
    )
    with pytest.raises(cs.CampaignSessionError, match="sglang"):
        _session(tmp_path, spec=sglang, slo_floors={"vllm": FLOORS["vllm"]})
    assert _session(tmp_path, spec=sglang, slo_floors=FLOORS).slo_floors == FLOORS
    assert _session(tmp_path, spec=hf, slo_floors={"vllm": FLOORS["vllm"]}).slo_floors is not None
    # the r cross-check tolerates the :g formatting of CAGE_CELL_BUDGET_R
    # (review S5): a 1/3 grid level arrives as 0.333333 on the cell side
    third = CellSpec(
        "gold-fresh", "none", "none", "single", "vllm", "qwen3-14b", "F2",
        budget_r=float("0.333333"), rate_frac=0.85,
    )
    assert cs.validate_budget_plan(_budget_plan_doc(r=1 / 3), third)["r"] == 1 / 3
    with pytest.raises(cs.CampaignSessionError, match="budget_r"):
        cs.validate_budget_plan(_budget_plan_doc(r=0.334), third)


def _canonical_requests() -> list[dict[str, Any]]:
    """tests/test_own_accounting.py::_canonical, as requests.jsonl rows:
    84 token-seconds over [0, 4) -> 21 tokens average -> rho 0.5 at 42."""
    return [
        {
            "example_id": "r1", "actual_send_ts": 0.0, "first_token_ts": 2.0,
            "completion_ts": 4.0, "prompt_tokens": 10, "num_tokens": 4,
            "group_id": 0, "cached_prompt_tokens": 4, "dropped_by_cap": False,
            "ttft_ms": 2000.0, "tpot_ms": 500.0,
        },
        {
            "example_id": "r2", "actual_send_ts": 1.0, "first_token_ts": None,
            "completion_ts": 3.0, "prompt_tokens": 20, "num_tokens": 0,
            "group_id": 1, "cached_prompt_tokens": None, "dropped_by_cap": False,
            "ttft_ms": 1500.0, "tpot_ms": None,
        },
    ]


def _emit(session: cs.CampaignCellSession, tmp_path: Path, ordinal: int) -> Any:
    staging = tmp_path / f"staging-{session.row_key[:8]}-{ordinal}"
    staging.mkdir(parents=True, exist_ok=True)
    rows = _canonical_requests()
    (staging / "qa_evidence.jsonl").write_text(
        "".join(
            json.dumps({
                "example_id": r["example_id"], "repeat_index": None,
                "record_index": None, "ok": True, "generated_answer": "x",
            }) + "\n"
            for r in rows
        ),
        encoding="utf-8",
    )
    return session.emit_window(
        ordinal=ordinal, trial_seed=7, results_rows=rows, staging_dir=staging,
        experiment_summary={
            "experiment": {"stale_index_opt_in": False},
            "consort": ZERO_CONSORT,
        },
        backend_metadata={"server_version": "0.0-test"},
        telemetry_snapshot=None, t_start=0.0, t_end=4.0,
    )


MANIFEST_ENV: dict[str, str] = {
    "CAGE_PROVIDER": "test",
    "CAGE_HARDWARE": "test-gpu x1",
    "CAGE_DATASET_MANIFESTS_SHA256": "0" * 64,
}


def test_emit_window_lands_the_pins_where_the_analysis_reads_them(tmp_path: Path) -> None:
    """End to end through the seam: the plan-shaped env -> from_cli ->
    emit_window -> manifest.json["slo_floors"] + cell.json["budget_plan"],
    then the analysis consumer's EXACT reads recover both (the #14 floor
    lookup and the rho_own budget), and the journaled tree still seals."""
    from src.analysis.goodput import SLOBaseline, evaluate_window
    import pandas as pd

    analysis_dir = str(REPO_ROOT / "scripts" / "4_analysis")
    if analysis_dir not in sys.path:
        sys.path.insert(0, analysis_dir)
    import run_campaign_analysis as rca

    root = _run_root(tmp_path)
    env = {
        **PRESSURE_ENV, **MANIFEST_ENV,
        "CAGE_SLO_FLOORS_JSON": json.dumps(FLOORS),
        "CAGE_BUDGET_PLAN_JSON": json.dumps(_budget_plan_doc()),
    }
    session = cs.CampaignCellSession.from_cli(_pressure_args(root), env=env)
    assert session is not None
    handle = _emit(session, tmp_path, 1)

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["slo_floors"] == FLOORS
    cell_meta = json.loads((session.cell_dir / "cell.json").read_text(encoding="utf-8"))
    assert cell_meta["budget_plan"] == _budget_plan_doc()

    # The #14 consumer's read: manifest[slo_floors][engine] -> SLOBaseline.
    floors = manifest.get(rca._SLO_FLOORS_MANIFEST_KEY)
    floor = floors[session.spec.engine]
    assert {"ttft_s", "tpot_s"} <= set(floor)
    baseline = SLOBaseline(ttft_s=float(floor["ttft_s"]), tpot_s=float(floor["tpot_s"]))
    records = pd.DataFrame(
        [{"ok": True, "veridical": True, "ttft_s": 2.0, "tpot_s": 0.5},
         {"ok": True, "veridical": True, "ttft_s": 1.5, "tpot_s": 0.1}]
    )
    metrics = evaluate_window(records, baseline, duration_s=4.0)
    assert metrics.n_issued == 2
    # The rho_own consumer's read: cell.json[budget_plan] -> the own-accounting
    # pass computes rho_own = 0.5 on the canonical rows (no labeled skip).
    plan = cell_meta.get(rca._BUDGET_PLAN_CELL_KEY)
    assert isinstance(plan["budget_bytes_total"], int) and isinstance(plan["kv_dtype"], str)
    index = pd.DataFrame([{
        "row_key": session.row_key,
        "dataset": "qasper",
        "window_key": handle.window_key,
        "window_dir": handle.window_dir.relative_to(root).as_posix(),
        "cell_json": f"cells/{session.row_key}/cell.json",
        "model": "qwen3-14b",
    }])
    summary = rca.run_own_accounting_pass(root, index, tmp_path / "analysis")
    assert summary["n_emitted"] == 1
    doc = json.loads(
        (tmp_path / "analysis" / "own_accounting" / index.loc[0, "window_dir"]
         / "own_accounting.json").read_text(encoding="utf-8")
    )
    assert doc["rho_own"] == pytest.approx(0.5)
    assert "skipped" not in doc["occupancy"]  # computed, not the labeled skip
    # write-time journal covers the manifest and cell.json: the tree seals
    assert cs.seal_campaign_run(root).is_file()


def test_reopened_manifest_must_carry_the_same_floors(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    base_env = {**PRESSURE_ENV, **MANIFEST_ENV, "CAGE_BUDGET_PLAN_JSON": json.dumps(_budget_plan_doc())}
    first = cs.CampaignCellSession.from_cli(
        _pressure_args(root), env={**base_env, "CAGE_SLO_FLOORS_JSON": json.dumps(FLOORS)},
    )
    assert first is not None
    _emit(first, tmp_path, 1)
    # a later cell of the same run pinning DIFFERENT floors refuses at
    # ACTIVATION (review S1: before a trial is served, not after it), and
    # emit_window applies the same rule again
    other = {"vllm": {"ttft_s": 0.5, "tpot_s": 0.02}, "sglang": FLOORS["sglang"]}
    with pytest.raises(cs.CampaignSessionError, match="slo_floors"):
        cs.CampaignCellSession.from_cli(
            _pressure_args(root), env={**base_env, "CAGE_SLO_FLOORS_JSON": json.dumps(other)},
        )
    second = cs.CampaignCellSession(
        run_root=root, dataset="qasper", spec=_pressure_spec(), num_trials=2, run_seed=1,
        env={**base_env, "CAGE_SLO_FLOORS_JSON": json.dumps(other)},
    )
    second.slo_floors = cs.validate_slo_floors(other)  # bypass activation on purpose
    with pytest.raises(cs.CampaignSessionError, match="slo_floors"):
        _emit(second, tmp_path, 2)
    assert not second.window_dir(2).exists()
    # the same floors, in any key order, reopen fine
    reordered = {"sglang": FLOORS["sglang"], "vllm": dict(reversed(list(FLOORS["vllm"].items())))}
    third = cs.CampaignCellSession.from_cli(
        _pressure_args(root), env={**base_env, "CAGE_SLO_FLOORS_JSON": json.dumps(reordered)},
    )
    assert third is not None
    _emit(third, tmp_path, 2)
    # a cell with NO floors pin still extends the run (pre-W4 producers),
    # and with no budget pin the recorded budget_plan is adopted, not lost
    # (review T10: the runner's path through emit_window)
    no_pins = {k: v for k, v in base_env.items() if k != "CAGE_BUDGET_PLAN_JSON"}
    fourth = cs.CampaignCellSession.from_cli(_pressure_args(root), env=no_pins)
    assert fourth is not None
    assert fourth.slo_floors is None and fourth.budget_plan is None
    _emit(fourth, tmp_path, 3)
    meta = json.loads((fourth.cell_dir / "cell.json").read_text(encoding="utf-8"))
    assert meta["budget_plan"] == _budget_plan_doc()
    assert set(meta["windows"]) == {"qasper-01", "qasper-02", "qasper-03"}
    # a contradicting budget pin on the populated cell refuses at ACTIVATION
    # (review S1/T10), before any window directory exists
    with pytest.raises(cs.CampaignSessionError, match="contradicts"):
        cs.CampaignCellSession.from_cli(
            _pressure_args(root),
            env={**base_env, "CAGE_SLO_FLOORS_JSON": json.dumps(FLOORS),
                 "CAGE_BUDGET_PLAN_JSON": json.dumps(_budget_plan_doc(budget_bytes_total=BUDGET_42_TOK + 1))},
        )
    assert not (fourth.cell_dir / "window_qasper-04").exists()
    # a manifest created WITHOUT floors cannot be extended by a pinned cell:
    # the floors it lacks can never be added (amended never; new run_id)
    bare_root = tmp_path / "results" / "camp1" / "a" / "20260821-1400-a-qwen3-14b"
    bare = cs.CampaignCellSession.from_cli(_pressure_args(bare_root), env=base_env)
    assert bare is not None
    _emit(bare, tmp_path, 1)
    assert "slo_floors" not in json.loads((bare_root / "manifest.json").read_text(encoding="utf-8"))
    with pytest.raises(cs.CampaignSessionError, match="amended never"):
        cs.CampaignCellSession.from_cli(
            _pressure_args(bare_root), env={**base_env, "CAGE_SLO_FLOORS_JSON": json.dumps(FLOORS)},
        )
    assert not (bare_root / "cells" / bare.row_key / "window_qasper-02").exists()


def test_populated_cell_without_a_budget_record_refuses_a_budget_pin(tmp_path: Path) -> None:
    """Review S4: a cell populated by a no-pin session carries windows but no
    budget_plan; a later pinned session must refuse at activation instead of
    labeling those windows with a budget nobody recorded them under."""
    root = _run_root(tmp_path)
    no_pins = {**PRESSURE_ENV, **MANIFEST_ENV}
    bare = cs.CampaignCellSession.from_cli(_pressure_args(root), env=no_pins)
    assert bare is not None
    _emit(bare, tmp_path, 1)
    meta = json.loads((bare.cell_dir / "cell.json").read_text(encoding="utf-8"))
    assert "budget_plan" not in meta and set(meta["windows"]) == {"qasper-01"}
    with pytest.raises(cs.CampaignSessionError, match="no 'budget_plan' record"):
        cs.CampaignCellSession.from_cli(
            _pressure_args(root),
            env={**no_pins, "CAGE_BUDGET_PLAN_JSON": json.dumps(_budget_plan_doc())},
        )
    # a fresh (unpopulated) cell of the same run takes the pin as usual
    other_env = {**no_pins, "CAGE_CELL_RATE_FRAC": "1.05",
                 "CAGE_BUDGET_PLAN_JSON": json.dumps(_budget_plan_doc())}
    fresh = cs.CampaignCellSession.from_cli(_pressure_args(root), env=other_env)
    assert fresh is not None and fresh.budget_plan == _budget_plan_doc()


# ---------------------------------------------------------------------------
# V8 slice (S0 close-out sheet row 1): the §6.1 regime referee runs at emission.
# Every window gets regime.json from campaign_layout.write_window_regime with
# the measured bounds, the sampled telemetry and completed-over-issued
# attainment, written before the metrics sentinel and inside the hash journal.
# ---------------------------------------------------------------------------

from src.analysis.goodput import IN_REGIME, PAST_CLIFF, UNPRESSURED  # noqa: E402
from src.analysis.regime_inputs import REGIME_UNKNOWN  # noqa: E402


def _regime_series(
    kv: float, *, ts: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0), instance: str | None = None
) -> list[dict[str, Any]]:
    """Canonical sampler records inside the [0, 4) test window: constant
    occupancy ``kv``, a cumulative preemption counter that climbs by one per
    sample (3 scarcity events), full coverage; optionally role-tagged."""
    rows = []
    for i, t in enumerate(ts):
        rec: dict[str, Any] = {"ts_s": t, "kv_cache_usage": kv, "preemptions_total": i}
        if instance is not None:
            rec["instance"] = instance
        rows.append(rec)
    return rows


def _emit_regime(
    session: cs.CampaignCellSession,
    tmp_path: Path,
    ordinal: int,
    *,
    series: list[dict[str, Any]] | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> Any:
    """_emit with an optional staged telemetry series and custom rows."""
    staging = tmp_path / f"staging-regime-{session.row_key[:8]}-{ordinal}"
    staging.mkdir(parents=True, exist_ok=True)
    rows = _canonical_requests() if rows is None else rows
    (staging / "qa_evidence.jsonl").write_text(
        "".join(
            json.dumps({
                "example_id": r["example_id"], "repeat_index": None,
                "record_index": None, "ok": not r.get("error"), "generated_answer": "x",
            }) + "\n"
            for r in rows
        ),
        encoding="utf-8",
    )
    if series is not None:
        (staging / "telemetry_series.jsonl").write_text(
            "".join(json.dumps(rec) + "\n" for rec in series), encoding="utf-8"
        )
    return session.emit_window(
        ordinal=ordinal, trial_seed=7, results_rows=rows, staging_dir=staging,
        experiment_summary={
            "experiment": {"stale_index_opt_in": False},
            "consort": ZERO_CONSORT,
        },
        backend_metadata={"server_version": "0.0-test"},
        telemetry_snapshot=None, t_start=0.0, t_end=4.0,
    )


def _regime_doc(handle: Any) -> dict[str, Any]:
    return json.loads((handle.window_dir / "regime.json").read_text(encoding="utf-8"))


def test_attainment_is_completed_over_issued_and_undefined_on_no_rows() -> None:
    assert cs._attainment([]) is None
    rows = [{"ok": True}, {"ok": False}, {"ok": None}, {}]
    assert cs._attainment(rows) == pytest.approx(0.25)  # only a literal True completes
    assert cs._attainment([{"ok": True}, {"ok": True}]) == 1.0


def test_emit_window_writes_regime_json_into_the_journal_and_unknown_without_telemetry(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path, env=MANIFEST_ENV, num_trials=3)
    handle = _emit_regime(session, tmp_path, 1)  # no staged series at all
    doc = _regime_doc(handle)
    assert doc["label"] == REGIME_UNKNOWN == "UNKNOWN_TELEMETRY"
    assert doc["telemetry_ok"] is False and doc["inputs"] is None
    assert "sample" in doc["refusal_reason"]  # zero in-window samples: the refusal lane
    assert doc["attainment"] == pytest.approx(1.0)  # both canonical rows are ok
    assert (doc["t_start"], doc["t_end"]) == (0.0, 4.0)
    assert doc["telemetry_source"] == "cage_stats.jsonl"
    # S0-15: the referee's file is a write-time-hashed artifact and the tree seals.
    rel = handle.window_dir.relative_to(session.run_root).as_posix() + "/regime.json"
    assert rel in cs.read_write_time_journal(session.run_root)
    assert cs.seal_campaign_run(session.run_root).is_file()


def test_emit_window_regime_labels_follow_the_section_6_1_criterion(tmp_path: Path) -> None:
    session = _session(tmp_path, env=MANIFEST_ENV, num_trials=4)
    # (a) rho 0.95 >= 0.9, (b) 3 scarcity events, (c) attainment 1.0 -> IN_REGIME
    doc = _regime_doc(_emit_regime(session, tmp_path, 1, series=_regime_series(0.95)))
    assert doc["label"] == IN_REGIME and doc["telemetry_ok"] is True
    assert doc["inputs"]["rho_kv_time_avg"] == pytest.approx(0.95)
    assert doc["inputs"]["scarcity_events"] == 3
    assert doc["attainment"] == pytest.approx(1.0)
    # failing (a): occupancy 0.5 -> UNPRESSURED
    doc = _regime_doc(_emit_regime(session, tmp_path, 2, series=_regime_series(0.5)))
    assert doc["label"] == UNPRESSURED
    # failing (c): one of two rows errored -> attainment 0.5 < 0.9 -> PAST_CLIFF,
    # and (c) wins the tie-break even though (a) and (b) hold
    rows = _canonical_requests()
    rows[1] = {**rows[1], "error": "timeout"}
    doc = _regime_doc(_emit_regime(session, tmp_path, 3, series=_regime_series(0.95), rows=rows))
    assert doc["label"] == PAST_CLIFF and doc["attainment"] == pytest.approx(0.5)


def test_emit_window_refuses_a_role_tagged_series_on_a_single_cell_before_the_sentinel(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path, env=MANIFEST_ENV, num_trials=2)
    series = _regime_series(0.95, instance="prefill") + _regime_series(0.9, instance="decode")
    with pytest.raises(cl.CampaignLayoutError, match="distinct instance roles"):
        _emit_regime(session, tmp_path, 1, series=series)
    wdir = session.window_dir(1)
    assert wdir.is_dir() and (wdir / "cage_stats.jsonl").is_file()
    assert not (wdir / "regime.json").exists(), "no quiet pooled label"
    # the referee runs BEFORE the sentinel: the window is incomplete and the
    # resume reset re-emits it instead of treating it as done
    assert not (wdir / "metrics.json").exists()
    assert session.window_complete(1) is False
    assert session.reset_incomplete_windows() == [1]
    assert not wdir.exists()


def test_emit_window_pd_cell_routes_the_recorded_split_to_the_summed_pool_lane(
    tmp_path: Path,
) -> None:
    pd_spec = CellSpec.from_baseline("B3", model="qwen3-14b", family="DIST", topology="pd")
    total = 4 * BUDGET_42_TOK
    plan = _budget_plan_doc(
        topology="pd", r=1.0, tp=2, pd_split=0.75, budget_bytes_total=total,
        pools_bytes=[3 * BUDGET_42_TOK, BUDGET_42_TOK], per_rank_bytes=total // 2,
    )
    session = cs.CampaignCellSession(
        run_root=_run_root(tmp_path), dataset="squad_v2", spec=pd_spec, num_trials=3,
        run_seed=7, gpu_count=2, env=MANIFEST_ENV, budget_plan=plan,
    )
    budgets = {"prefill": 3 * BUDGET_42_TOK, "decode": BUDGET_42_TOK}
    assert session._role_budgets(plan, [{"ts_s": 0.0}]) == budgets
    assert session._role_budgets(plan, []) is None  # empty series: the refusal lane
    series = _regime_series(0.95, instance="prefill") + _regime_series(0.95, instance="decode")
    doc = _regime_doc(_emit_regime(session, tmp_path, 1, series=series))
    assert doc["telemetry_ok"] is True and doc["label"] == IN_REGIME
    assert doc["pd"]["budgets_by_role"] == budgets
    assert set(doc["pd"]["per_role"]) == {"prefill", "decode"}
    assert doc["inputs"]["scarcity_events"] == 6  # summed over the two roles
    # review MEDIUM 1: a budgeted pd cell with NO staged series records
    # UNKNOWN_TELEMETRY like every other cell; the emission succeeds
    doc = _regime_doc(_emit_regime(session, tmp_path, 2))
    assert doc["label"] == REGIME_UNKNOWN and doc["telemetry_ok"] is False
    assert "pd" not in doc
    # a PD pair with one dead sampler (prefill-only series) is NOT certified
    # from half the gauges: the writer names the missing budgeted role
    with pytest.raises(cl.CampaignLayoutError, match="role_budgets carries"):
        _emit_regime(session, tmp_path, 3, series=_regime_series(0.95, instance="prefill"))
    assert not session.window_complete(3)
    # review LOW 3: a resume WITHOUT the env pin adopts the cell record
    # (CellWriter) and still routes the recorded split
    resumed = cs.CampaignCellSession(
        run_root=_run_root(tmp_path), dataset="squad_v2", spec=pd_spec, num_trials=3,
        run_seed=7, gpu_count=2, env=MANIFEST_ENV,
    )
    assert resumed.budget_plan is None
    assert resumed.reset_incomplete_windows() == [3]
    doc = _regime_doc(_emit_regime(resumed, tmp_path, 3, series=series))
    assert doc["label"] == IN_REGIME and doc["pd"]["budgets_by_role"] == budgets
    # a pd record without the split is malformed: refused by name at emission
    # (adopted records) and at activation (validate_budget_plan, LOW 4)
    with pytest.raises(cs.CampaignSessionError, match="pools_bytes"):
        session._role_budgets({**plan, "pools_bytes": None}, [{"ts_s": 0.0}])
    # a pd cell with NO budget pin routes nothing: the writer's T4.1 gate
    # refuses the tagged series instead of pooling it against unrecorded budgets
    bare = cs.CampaignCellSession(
        run_root=tmp_path / "results" / "camp1" / "a" / "20260821-1500-a-qwen3-14b",
        dataset="squad_v2", spec=pd_spec, num_trials=1, run_seed=7, gpu_count=2,
        env=MANIFEST_ENV,
    )
    assert bare._role_budgets(None, [{"ts_s": 0.0}]) is None
    with pytest.raises(cl.CampaignLayoutError, match="per-role budgets"):
        _emit_regime(bare, tmp_path, 1, series=series)


def test_single_cell_never_routes_role_budgets(tmp_path: Path) -> None:
    session = _session(tmp_path, spec=_pressure_spec(), budget_plan=_budget_plan_doc())
    assert session.spec.topology == "single" and session.budget_plan is not None
    assert session._role_budgets(session.budget_plan, [{"ts_s": 0.0}]) is None


def test_validate_budget_plan_checks_the_pd_split_at_activation() -> None:
    # review LOW 4: a pd plan's pools_bytes is validated before serving, not
    # at emission; plan_budget always emits a 2-tuple of ints, so only a
    # hand-built or corrupted pin can reach this refusal
    pd_spec = CellSpec.from_baseline("B3", model="qwen3-14b", family="DIST", topology="pd")
    pd_plan = _budget_plan_doc(topology="pd", r=1.0, tp=2, pd_split=0.75, pools_bytes=[3, 1])
    assert cs.validate_budget_plan(pd_plan, pd_spec)["pools_bytes"] == [3, 1]
    assert cs.validate_budget_plan({**pd_plan, "pools_bytes": (3, 1)}, pd_spec)["pools_bytes"] == [3, 1]
    for bad in (None, [3], [3, 1, 1], [3.0, 1], [3, 0], [True, 1], "3,1", {"prefill": 3}):
        with pytest.raises(cs.CampaignSessionError, match="pools_bytes"):
            cs.validate_budget_plan({**pd_plan, "pools_bytes": bad}, pd_spec)
    # non-pd plans never carry the check (single/tp plans have pools_bytes None)
    assert cs.validate_budget_plan(_budget_plan_doc(), _pressure_spec())["pools_bytes"] is None
