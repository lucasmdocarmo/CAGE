"""ADR-0178 block 1: contrast #12 as E1 (the yield frontier drop), the two
registered confirmatory chains, and the floor suite as a descriptive sidebar.

The fixture is a sealed RESULTS_LAYOUT tree with a K1-shaped pressure block:
control cells (budget_r 1.25) and constrained cells (0.5, 0.25) served at ONE
offered rate (metrics.json workload.open_loop.schedule.rate_qps), each window
graded on the plan's class first-token target through the real predicate
producer (build_predicate_table.py) and the real organizer. The expected
shares are recomputed in the test from the same generator rules, so every
number the executor reports is checked against an independent count.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_predicate_table as bpt  # noqa: E402
import organize_results as org  # noqa: E402
import run_campaign_analysis as rca  # noqa: E402
from src.analysis.cellspec import CellSpec  # noqa: E402
from src.analysis.stats import families  # noqa: E402
from src.analysis.stats.corrections import holm  # noqa: E402
from src.analysis.stats.ledger import hash_artifacts, write_ledger  # noqa: E402

RUN_ID = "20261027-1200-s1-qwen3-14b"
CAMPAIGN = "camp1"
SESSION = "s1"
MODEL = "qwen3-14b"
DATASET = "squad_v2"
SCORING_ID = "s01-fast"
N_EXAMPLES = 16
WINDOWS = 3
CLASS_TOKENS = 4779
TARGET_S = 2.0
RATE = 3.5
#: predicate-FALSE examples (the low indices) in every window of every cell
N_FALSE = 2
#: late completions (first token at 3 s, past the 2 s target) per ordinal
LATE_BY_RUNG: dict[float, dict[int, int]] = {
    1.25: {1: 0, 2: 0, 3: 0},
    0.5: {1: 3, 2: 4, 3: 2},
    0.25: {1: 7, 2: 8, 3: 6},
}


@pytest.fixture(autouse=True)
def _pressure_dataset_and_no_freeze(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The registered pressure dataset is qasper (C1); this fixture serves
    # span-QA rows, so squad_v2 stands in, as tests/test_predicate_chain.py does.
    monkeypatch.setattr(families, "PRESSURE_DATASET", DATASET)
    monkeypatch.setenv(bpt.FREEZE_ENV_VAR, str(tmp_path / "absent_freeze.json"))


def _spec(engine: str, budget_r: float, rate_frac: float, *, arm: str = "gold-fresh",
          family: str = "F2") -> CellSpec:
    return CellSpec(
        arm=arm, retriever="none", policy="none", topology="single",  # type: ignore[arg-type]
        engine=engine, model=MODEL, family=family,  # type: ignore[arg-type]
        budget_r=budget_r, rate_frac=rate_frac,
    )


def _write_window(wdir: Path, *, ordinal: int, late: int, rate: float) -> list[Path]:
    wdir.mkdir(parents=True)
    requests, evidence = [], []
    for i in range(N_EXAMPLES):
        ok = i != N_EXAMPLES - 1  # one serving failure per window
        ttft = 3000.0 if i < late else 400.0 + i
        requests.append(json.dumps({
            "example_id": f"e{i:03d}", "ok": ok, "workload_mode": "open_loop",
            "ttft_ms": ttft if ok else None,
            "ttft_from_scheduled_ms": ttft + 5.0 if ok else None,
            "tpot_ms": 20.0 if ok else None, "num_tokens": 8 if ok else 0,
            "latency_ms": ttft + 100.0,
        }))
        evidence.append(json.dumps({
            "example_id": f"e{i:03d}", "baseline": "blind-token", "repeat_index": 0,
            "record_index": None, "generated_answer": "" if not ok else f"answer {i}",
            "ok": ok, "error": None if ok else "HTTP 500", "empty_generation": False,
        }))
    payloads = {
        "requests.jsonl": "\n".join(requests) + "\n",
        "qa_evidence.jsonl": "\n".join(evidence) + "\n",
        "engine_metrics.json": json.dumps({"snapshot": "x"}),
        "cage_stats.jsonl": json.dumps({"ts_s": 0.0, "kv_cache_usage": 0.9}) + "\n",
        "metrics.json": json.dumps({
            "workload": {"mode": "open_loop", "open_loop": {"schedule": {"rate_qps": rate}}}
        }),
    }
    written = []
    for name, text in payloads.items():
        (wdir / name).write_text(text, encoding="utf-8")
        written.append(wdir / name)
    return written


def _build_run(tmp_path: Path, cells: list[tuple[CellSpec, float]], *, with_f1: bool = False) -> Path:
    """A sealed tree: ``cells`` = (spec, offered rate) pressure cells, plus the
    F1 headline pair when asked; a sealed scoring pass; the real predicate
    table; the real organizer."""
    run_dir = tmp_path / "results" / CAMPAIGN / SESSION / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "campaign": CAMPAIGN, "session": SESSION, "run_id": RUN_ID, "model": MODEL,
        "git_sha": "deadbeef", "git_dirty": False, "engine": "vllm",
        "engine_version": "0.0-test", "seed": 1, "provider": "test", "hardware": "test-gpu",
        "dataset_manifests_sha256": "0" * 64, "cellspec_schema_version": 1,
        "created_utc": "2026-10-27T12:00:00Z", "datasets": [DATASET],
        "slo_floors": {"vllm": {"ttft_s": 0.1, "tpot_s": 0.05}, "sglang": {"ttft_s": 0.1, "tpot_s": 0.05}},
    }), encoding="utf-8")
    all_cells: list[tuple[CellSpec, float | None]] = list(cells)
    if with_f1:
        all_cells += [(CellSpec.from_baseline(b, model=MODEL), None) for b in ("B3", "B6")]  # type: ignore[arg-type]
    sealed: list[Path] = []
    for spec, rate in all_cells:
        baseline = org.BASELINE_OF_CELL.get((spec.arm, spec.retriever), "")
        cell_dir = run_dir / "cells" / spec.to_row_key()
        cell_dir.mkdir(parents=True)
        windows: dict[str, dict[str, Any]] = {}
        for ordinal in range(1, WINDOWS + 1):
            k = f"{DATASET}-{ordinal:02d}"
            windows[k] = {"dataset": DATASET, "seed": 1, "rep": ordinal,
                          "budget_r": spec.budget_r, "rate_frac": spec.rate_frac,
                          "t_start": 0.0, "t_end": 60.0}
            late = LATE_BY_RUNG[spec.budget_r][ordinal] if spec.budget_r is not None else 0
            sealed.extend(_write_window(cell_dir / f"window_{k}", ordinal=ordinal,
                                        late=late, rate=rate or 1.0))
        cell = {"cellspec": spec.to_flat_dict(), "baseline": baseline, "windows": windows,
                "gpu_count": 1}
        if spec.family != "F1":
            cell["budget_plan"] = {"avg_seq_tokens": CLASS_TOKENS, "r": spec.budget_r}
        (cell_dir / "cell.json").write_text(json.dumps(cell), encoding="utf-8")
        sealed.append(cell_dir / "cell.json")
    write_ledger(hash_artifacts(sealed, base_dir=run_dir), run_dir / "ledger.json")
    # the sealed scoring pass: N_FALSE low indices score EM 0 in every window
    raw_sha = json.loads((run_dir / "ledger.json").read_text())["entries_sha256"]
    scoring_dir = run_dir / "scoring" / SCORING_ID
    written: list[Path] = []
    for ev in sorted(run_dir.glob("cells/*/window_*/qa_evidence.jsonl")):
        out = scoring_dir / ev.parent.relative_to(run_dir)
        out.mkdir(parents=True)
        rows = [json.dumps({"example_id": f"e{i:03d}", "repeat_index": "0", "record_index": None,
                            "exact_match": 0.0 if i < N_FALSE else 1.0,
                            "f1_score": 0.0 if i < N_FALSE else 1.0, "grounding_score": None})
                for i in range(N_EXAMPLES)]
        (out / "qa_scores.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
        (out / "quality.json").write_text(json.dumps({"rows": N_EXAMPLES}), encoding="utf-8")
        written += [out / "qa_scores.jsonl", out / "quality.json"]
    manifest = scoring_dir / "scoring_manifest.json"
    manifest.write_text(json.dumps({"scoring_run_id": SCORING_ID, "created_utc": "2026-10-27T13:00:00Z",
                                    "raw_run_ledger_entries_sha256": raw_sha, "mode": "fast"}),
                        encoding="utf-8")
    written.append(manifest)
    write_ledger(hash_artifacts(written, base_dir=scoring_dir), scoring_dir / "ledger.json")
    assert bpt.main([str(run_dir), "--scoring-run-id", SCORING_ID, "--max-null-fraction", "0.5"]) == 0
    assert org.main([str(run_dir)]) == 0
    return run_dir


def _plan(tmp_path: Path, targets: dict[str, float] | None = None) -> Path:
    path = tmp_path / "plan_registered.json"
    path.write_text(json.dumps({"class_first_token_targets_s": targets or {str(CLASS_TOKENS): TARGET_S}}),
                    encoding="utf-8")
    return path


def _k1_cells() -> list[tuple[CellSpec, float]]:
    """vLLM: two control cells at RATE (pooled, D2), constrained r0.5 and
    r0.25 at RATE, one constrained cell at an unmatched rate; SGLang: one
    control and one r0.25 cell at RATE."""
    return [
        (_spec("vllm", 1.25, 0.76), RATE),
        (_spec("vllm", 1.25, 0.761), RATE),
        (_spec("vllm", 0.5, 0.7), RATE),
        (_spec("vllm", 0.25, 0.8), RATE),
        (_spec("vllm", 0.25, 0.85), 2.0),  # no control at 2.0 req/s
        (_spec("sglang", 1.25, 0.77), RATE),
        (_spec("sglang", 0.25, 0.82), RATE),
    ]


def _expected_shares(rung: float) -> dict[str, float]:
    """Independent recount of the generator: ok rows i < 15; accepted = ok and
    i >= N_FALSE; timely = ok and i >= late; yield = both; all over 16."""
    ok = set(range(N_EXAMPLES - 1))
    accepted = {i for i in ok if i >= N_FALSE}
    means = {"timely": 0.0, "accepted": 0.0, "yield": 0.0}
    for ordinal in range(1, WINDOWS + 1):
        late = LATE_BY_RUNG[rung][ordinal]
        timely = {i for i in ok if i >= late}
        means["timely"] += len(timely) / N_EXAMPLES / WINDOWS
        means["accepted"] += len(accepted) / N_EXAMPLES / WINDOWS
        means["yield"] += len(timely & accepted) / N_EXAMPLES / WINDOWS
    return means


def _load_stats(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    analysis_dir = sorted((run_dir / "analysis").iterdir())[-1]
    return analysis_dir, json.loads((analysis_dir / "stats.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------


def test_registered_family_alphas_sum_to_the_total() -> None:
    assert families.PRESSURE_FAMILY_ALPHA + families.CASE_STUDY_FAMILY_ALPHA == pytest.approx(
        families.REGISTERED_ALPHA_TOTAL
    )
    assert rca.REGISTERED_ALPHA == families.REGISTERED_ALPHA_TOTAL == 0.05
    assert families.chain_of(12) == ("pressure", 0.03)
    assert families.chain_of(4) == ("case_study", 0.02)
    assert families.PRIMARY_IDS == {4, 12, 13, 14}
    with pytest.raises(families.FamilyMapError, match="not a registered chain primary"):
        families.chain_of(15)


def test_frontier_point_estimate_decomposition_and_pooled_controls(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path, _k1_cells())
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(_plan(tmp_path))]) == 0
    analysis_dir, stats = _load_stats(run_dir)
    section = stats["frontier"]
    assert section["estimand"] == "frontier_drop" and section["tier"] == "primary"
    legs = {(leg["engine"], leg["leg"]): leg for leg in section["legs"]}
    assert set(legs) == {("vllm", "B1|r0.5"), ("vllm", "B1|r0.25"), ("sglang", "B1|r0.25")}
    control = _expected_shares(1.25)
    for (engine, leg_name), leg in legs.items():
        rung = leg["budget_r"]
        expected = _expected_shares(rung)
        assert leg["yield"]["mean_control"] == pytest.approx(control["yield"])
        assert leg["yield"]["mean_constrained"] == pytest.approx(expected["yield"])
        assert leg["yield"]["delta"] == pytest.approx(expected["yield"] - control["yield"])
        assert leg["timely"]["delta"] == pytest.approx(expected["timely"] - control["timely"])
        # acceptance is identical by construction: the drop is timeliness alone
        assert leg["accepted"]["delta"] == pytest.approx(0.0)
        assert leg["yield"]["delta"] < 0
        assert leg["executed_alternative"] == "less"
        assert leg["target_s"] == TARGET_S and leg["class_tokens"] == CLASS_TOKENS
        assert leg["rate_qps"] == pytest.approx(RATE)
        assert leg["n_windows_constrained"] == WINDOWS
        if engine == "vllm":
            # D2: both control cells at RATE are pooled into every vLLM leg
            assert len(leg["control_row_keys"]) == 2
            assert leg["n_windows_control"] == 2 * WINDOWS
        else:
            assert leg["n_windows_control"] == WINDOWS
        # the bootstrap interval brackets the point estimate and is recorded
        y = leg["yield"]
        assert y["ci95_low_bootstrap"] <= y["delta"] <= y["ci95_high_bootstrap"]
    # the quarter cache drops far more than the half cache
    assert legs[("vllm", "B1|r0.25")]["yield"]["delta"] < legs[("vllm", "B1|r0.5")]["yield"]["delta"]
    assert legs[("vllm", "B1|r0.25")]["yield"]["p_value"] < 0.05
    # the unmatched constrained cell is listed, never refused
    listed = section["not_confirmatory_points"]
    assert len(listed) == 1 and listed[0]["rate_qps"] == pytest.approx(2.0)
    assert "lam0.85" in listed[0]["row_key"]
    summary = (analysis_dir / "summary.md").read_text(encoding="utf-8")
    assert "Yield frontier drop (#12" in summary


def test_frontier_engine_merge_holm_and_intersection_union(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path, _k1_cells())
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(_plan(tmp_path))]) == 0
    _, stats = _load_stats(run_dir)
    section = stats["frontier"]
    (iu,) = section["per_dataset_intersection"]
    assert iu["dataset"] == DATASET and iu["n_legs"] == 2
    by_leg = {leg["leg"]: {} for leg in section["legs"]}
    for leg in section["legs"]:
        by_leg[leg["leg"]][leg["engine"]] = leg["yield"]["p_value"]
    # D11 / ruling A13: engine members merge by the LARGEST p per leg
    merged = {name: max(ps.values()) for name, ps in by_leg.items()}
    for name, record in iu["legs"].items():
        assert record["p_merged"] == pytest.approx(merged[name])
        assert record["engines"] == pytest.approx(by_leg[name])
    adjusted = holm([merged[n] for n in iu["legs"]])
    # ADR-0178 amendment 2: the endpoint is the SMALLEST Holm-adjusted point
    # (disjunctive); the conjunctive maximum rides as a sensitivity line
    expected_endpoint = float(min(adjusted))
    assert iu["p_endpoint"] == pytest.approx(expected_endpoint)
    assert iu["p_intersection_union_sensitivity"] == pytest.approx(float(max(adjusted)))
    assert set(iu["per_arm"]) == {"B1"}
    assert iu["per_arm"]["B1"]["p_holm_any"] == pytest.approx(expected_endpoint)
    assert iu["per_arm"]["B1"]["verdict"].startswith("drop confirmed")
    for leg in section["legs"]:
        assert leg["p_merged_engines"] == pytest.approx(merged[leg["leg"]])
    # the endpoint reached the pressure chain under contrast-12
    gate = stats["gatekeeping"]
    primaries = [p for p in gate["primaries"] if p["endpoint"] == "contrast-12"]
    assert [p["dataset_metric"] for p in primaries] == [f"{DATASET}|frontier_drop"]
    assert primaries[0]["p_value"] == pytest.approx(expected_endpoint)
    assert primaries[0]["alpha"] == 0.03 and primaries[0]["family"] == "pressure"


def test_frontier_bootstrap_is_deterministic_at_the_registered_seed(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path, _k1_cells())
    plan = _plan(tmp_path)
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(plan)]) == 0
    _, first = _load_stats(run_dir)
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(plan)]) == 0
    _, second = _load_stats(run_dir)
    assert first["frontier"]["legs"] == second["frontier"]["legs"]
    assert first["frontier"]["bootstrap"] == {
        "unit": "window", "replicates": 2000, "seed": 42, "interval": "95 percent percentile",
    }


def test_frontier_refuses_without_plan_and_names_the_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = _build_run(tmp_path, _k1_cells())
    assert rca.main([str(run_dir), "--contrasts", "12"]) == 1
    err = capsys.readouterr().err
    assert "MISSING ARTIFACT" in err and "--plan" in err
    assert not (run_dir / "analysis").exists()


def test_frontier_refuses_a_class_without_a_target(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = _build_run(tmp_path, _k1_cells())
    plan = _plan(tmp_path, {"2336": 1.3})
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(plan)]) == 1
    err = capsys.readouterr().err
    assert str(CLASS_TOKENS) in err and "no target" in err


def test_frontier_refuses_without_any_confirmatory_point(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # constrained cells only, every one at a rate no control cell served
    cells = [(_spec("vllm", 0.5, 0.7), 2.0), (_spec("vllm", 0.25, 0.8), 2.5)]
    run_dir = _build_run(tmp_path, cells)
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(_plan(tmp_path))]) == 1
    err = capsys.readouterr().err
    assert "no confirmatory point" in err and "rate_qps" in err
    assert not (run_dir / "analysis").exists()


def test_floor_suite_rides_under_12_as_a_labeled_descriptive_skip(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path, _k1_cells())
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(_plan(tmp_path))]) == 0
    analysis_dir, stats = _load_stats(run_dir)
    fals = stats["falsification"]
    assert fals["contrast_id"] == 12 and fals["tier"] == "descriptive"
    assert "goodput_frac" in fals["skipped"] or "rate_frac" in fals["skipped"]
    summary = (analysis_dir / "summary.md").read_text(encoding="utf-8")
    assert "Floor suite (under #12, descriptive" in summary
    assert "SKIPPED (labeled)" in summary


def test_two_registered_chains_run_at_their_own_alphas(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path, _k1_cells(), with_f1=True)
    assert rca.main([
        str(run_dir), "--contrasts", "4", "12", "--metrics", "ttft_ms", "predicate",
        "--plan", str(_plan(tmp_path)),
    ]) == 0
    _, stats = _load_stats(run_dir)
    gate = stats["gatekeeping"]
    assert gate["alpha"] == 0.05
    assert gate["family_alphas"] == {"case_study": 0.02, "pressure": 0.03}
    fam = gate["families"]
    assert fam["case_study"]["order_registered"] == ["contrast-4"]
    assert fam["case_study"]["order_executed"] == ["contrast-4"]
    assert fam["case_study"]["chain_complete"] is True
    assert fam["pressure"]["order_registered"] == ["contrast-12", "contrast-14", "contrast-13"]
    assert fam["pressure"]["order_executed"] == ["contrast-12"]
    assert fam["pressure"]["chain_complete"] is False
    assert fam["pressure"]["missing_primary_endpoints"] == ["contrast-14", "contrast-13"]
    assert gate["primary_chain_order_executed"] == ["contrast-4", "contrast-12"]
    assert set(gate["missing_primary_endpoints"]) == {"contrast-14", "contrast-13"}
    alphas = {p["endpoint"]: p["alpha"] for p in gate["primaries"]}
    assert alphas == {"contrast-4": 0.02, "contrast-12": 0.03}
    sets = {d["endpoint"]: d for d in gate["set_decisions"]}
    assert sets["contrast-4"]["alpha"] == 0.02 and sets["contrast-12"]["alpha"] == 0.03
    # E1 is the first link of its chain, so it is tested confirmatorily
    assert sets["contrast-12"]["status"] == "confirmatory"


def test_design_input_alpha_is_registration_content(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = _build_run(tmp_path, _k1_cells())
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(_plan(tmp_path)),
                     "--alpha", "0.01"]) == 1
    assert "registered alpha" in capsys.readouterr().err
    assert not (run_dir / "analysis").exists()


def test_compile_family_map_refuses_a_foreign_total_alpha() -> None:
    with pytest.raises(families.FamilyMapError, match="registered total"):
        families.compile_family_map(["squad_v2"], alpha=0.04)


# ---------------------------------------------------------------------------
# Review fixes (fresh Fable 5.1 review of 2026-10-27: H1, M2, M3, M4, L5, L7, L8, L9)
# ---------------------------------------------------------------------------


def _write_unit_window(tmp_path: Path, rows: list[dict[str, Any]], preds: list[dict[str, Any]]) -> tuple[Path, Any, Path]:
    """A bare window for _window_yield_shares: requests + a predicate mirror."""
    run_dir = tmp_path / "run"
    wdir = run_dir / "cells" / "c" / "window_squad_v2-01"
    wdir.mkdir(parents=True)
    (wdir / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    pred_root = run_dir / "predicate" / "p"
    pdir = pred_root / "cells" / "c" / "window_squad_v2-01"
    pdir.mkdir(parents=True)
    (pdir / "predicate.jsonl").write_text("\n".join(json.dumps(p) for p in preds) + "\n", encoding="utf-8")

    class Rec:
        window_dir = "cells/c/window_squad_v2-01"

    return run_dir, Rec(), pred_root


def _row(i: int, *, ok: bool = True, ttft: float = 400.0, scheduled: bool = True) -> dict[str, Any]:
    row: dict[str, Any] = {"example_id": f"e{i:03d}", "ok": ok, "workload_mode": "open_loop",
                           "ttft_ms": ttft if ok else None, "tpot_ms": 20.0, "num_tokens": 8}
    if scheduled and ok:
        row["ttft_from_scheduled_ms"] = ttft + 5.0
    return row


def test_window_shares_count_every_issued_row_and_record_the_clock(tmp_path: Path) -> None:
    rows = [_row(0), _row(1, ttft=3000.0), _row(2, ok=False), _row(3, scheduled=False)]
    # a closed-loop row (no workload_mode) is graded on the send clock
    del rows[3]["workload_mode"]
    preds = [{"example_id": f"e{i:03d}", "predicate": v} for i, v in ((0, True), (1, True), (2, None), (3, False))]
    run_dir, rec, pred_root = _write_unit_window(tmp_path, rows, preds)
    shares = rca._window_yield_shares(run_dir, rec, pred_root, 2.0)
    assert shares["n_issued"] == 4
    assert shares["timely_frac"] == pytest.approx(2 / 4)    # e000 and e003; e001 late, e002 failed
    assert shares["accepted_frac"] == pytest.approx(2 / 4)  # e000, e001
    assert shares["yield_frac"] == pytest.approx(1 / 4)     # e000 only
    assert (shares["n_basis_scheduled"], shares["n_basis_send"]) == (2, 2)


def test_window_shares_refuse_a_null_predicate_on_a_completion_m2(tmp_path: Path) -> None:
    rows = [_row(0), _row(1)]
    preds = [{"example_id": "e000", "predicate": True}, {"example_id": "e001", "predicate": None}]
    run_dir, rec, pred_root = _write_unit_window(tmp_path, rows, preds)
    with pytest.raises(rca.AnalysisError, match="NULL predicate"):
        rca._window_yield_shares(run_dir, rec, pred_root, 2.0)


def test_window_shares_refuse_a_non_boolean_predicate(tmp_path: Path) -> None:
    rows = [_row(0)]
    preds = [{"example_id": "e000", "predicate": 1.0}]
    run_dir, rec, pred_root = _write_unit_window(tmp_path, rows, preds)
    with pytest.raises(rca.AnalysisError, match="not a JSON boolean"):
        rca._window_yield_shares(run_dir, rec, pred_root, 2.0)


def test_window_shares_refuse_rows_without_the_ok_stamp_l7(tmp_path: Path) -> None:
    rows = [{"example_id": "e000", "ttft_ms": 400.0}]
    preds = [{"example_id": "e000", "predicate": True}]
    run_dir, rec, pred_root = _write_unit_window(tmp_path, rows, preds)
    with pytest.raises(rca.AnalysisError, match="no 'ok' column"):
        rca._window_yield_shares(run_dir, rec, pred_root, 2.0)


def test_window_shares_refuse_an_open_loop_row_without_the_scheduled_clock(tmp_path: Path) -> None:
    rows = [_row(0, scheduled=False)]
    preds = [{"example_id": "e000", "predicate": True}]
    run_dir, rec, pred_root = _write_unit_window(tmp_path, rows, preds)
    with pytest.raises(rca.AnalysisError, match="coordinated-omission clock"):
        rca._window_yield_shares(run_dir, rec, pred_root, 2.0)


def test_window_shares_refuse_a_completion_without_a_predicate_row(tmp_path: Path) -> None:
    rows = [_row(0), _row(1)]
    preds = [{"example_id": "e000", "predicate": True}]
    run_dir, rec, pred_root = _write_unit_window(tmp_path, rows, preds)
    with pytest.raises(rca.AnalysisError, match="NO predicate row"):
        rca._window_yield_shares(run_dir, rec, pred_root, 2.0)


def test_rung_above_control_is_listed_not_tested_m4(tmp_path: Path) -> None:
    t_late = dict(LATE_BY_RUNG)
    LATE_BY_RUNG[1.5] = {1: 0, 2: 0, 3: 0}
    try:
        cells = _k1_cells() + [(_spec("vllm", 1.5, 0.9), RATE)]
        run_dir = _build_run(tmp_path, cells)
    finally:
        LATE_BY_RUNG.clear(); LATE_BY_RUNG.update(t_late)
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(_plan(tmp_path))]) == 0
    _, stats = _load_stats(run_dir)
    section = stats["frontier"]
    assert not any(leg["budget_r"] > 1.25 for leg in section["legs"])
    above = [p for p in section["not_confirmatory_points"] if "r1.5" in p["row_key"]]
    assert len(above) == 1 and "above the control rung" in above[0]["reason"]


def test_duplicate_constrained_cell_per_engine_and_rung_refuses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cells = _k1_cells() + [(_spec("vllm", 0.25, 0.81), RATE)]
    run_dir = _build_run(tmp_path, cells)
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(_plan(tmp_path))]) == 1
    assert "supplied twice by engine" in capsys.readouterr().err


def test_class_mismatch_between_point_and_control_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = _build_run(tmp_path, _k1_cells())
    original = rca._cell_class_tokens

    def two_classes(run_root: Path, cell_json_rel: str) -> int:
        return 2336 if "r0.25" in cell_json_rel else original(run_root, cell_json_rel)

    monkeypatch.setattr(rca, "_cell_class_tokens", two_classes)
    plan = _plan(tmp_path, {str(CLASS_TOKENS): TARGET_S, "2336": 1.3})
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(plan)]) == 1
    assert "span classes" in capsys.readouterr().err


def test_non_k1_cell_without_a_plan_class_is_listed_not_refused_l5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _build_run(tmp_path, _k1_cells())
    original = rca._cell_class_tokens

    def foreign_class_on_the_unmatched_cell(run_root: Path, cell_json_rel: str) -> int:
        return 99999 if "lam0.85" in cell_json_rel else original(run_root, cell_json_rel)

    monkeypatch.setattr(rca, "_cell_class_tokens", foreign_class_on_the_unmatched_cell)
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(_plan(tmp_path))]) == 0
    _, stats = _load_stats(run_dir)
    assert len(stats["frontier"]["legs"]) == 3


def test_too_few_windows_is_a_labeled_skip_then_a_named_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys.modules[__name__], "WINDOWS", 1)
    run_dir = _build_run(tmp_path, [(_spec("vllm", 1.25, 0.76), RATE), (_spec("vllm", 0.25, 0.8), RATE)])
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(_plan(tmp_path))]) == 1
    err = capsys.readouterr().err
    assert "no leg could be computed" in err and "needs >= 2 windows per side" in err


def test_floor_suite_sidebar_survives_any_exception_m3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: Any, **k: Any) -> None:
        raise KeyError("rate_frac")

    monkeypatch.setattr(rca, "compute_falsification_suite", boom)
    run_dir = _build_run(tmp_path, _k1_cells())
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(_plan(tmp_path))]) == 0
    _, stats = _load_stats(run_dir)
    assert stats["falsification"]["skipped"].startswith("KeyError")
    assert stats["frontier"]["legs"]


def test_blinded_floor_suite_skip_names_no_arm_h1(tmp_path: Path) -> None:
    from src.analysis.stats.blinding import scramble_labels
    import pandas as pd

    # the F1 pair adds a second and third arm (a seal needs >= 2 labels); a
    # third r0.25 rate lets the suite reach its per-query refusal, the one
    # that names row keys (review H1)
    run_dir = _build_run(tmp_path, _k1_cells() + [(_spec("vllm", 0.25, 0.9), 2.5)], with_f1=True)
    index = pd.read_csv(run_dir / "index" / "cells_index.csv")
    scramble_labels(pd.DataFrame({"arm": sorted(index["arm"].unique())}), "arm", seed=11,
                    sealed_map_path=run_dir / "blinding" / "sealed_arm_map.json")
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(_plan(tmp_path))]) == 0
    analysis_dir, stats = _load_stats(run_dir)
    assert stats["blinding"]["active"] is True
    skip = stats["falsification"]["skipped"]
    assert "gold-fresh" not in skip and "BLINDED:ARM-" in skip
    assert "suppressed" in stats["frontier"]
    summary = (analysis_dir / "summary.md").read_text(encoding="utf-8")
    assert "gold-fresh" not in summary


def test_flat_arm_beside_a_dropping_arm_passes_per_arm(tmp_path: Path) -> None:
    # ADR-0178 amendment 2: a reuse-like arm with no drop beside a fresh arm
    # that drops. The endpoint rejects on the dropping arm; the flat arm reads
    # "no drop"; the old conjunctive rule (the sensitivity line) would fail.
    saved = dict(LATE_BY_RUNG)
    # the same multiset of late counts on both sides of the flat arm (a
    # near-null with variance); the fresh arm keeps its quarter-cache drop
    LATE_BY_RUNG[1.25] = {1: 3, 2: 2, 3: 3}
    LATE_BY_RUNG[0.5] = {1: 2, 2: 3, 3: 3}
    try:
        cells = [
            (_spec("vllm", 1.25, 0.76), RATE), (_spec("vllm", 0.25, 0.8), RATE),
            # the flat arm: its "constrained" rung is the near-null pattern
            (_spec("vllm", 1.25, 0.66, arm="corpus-reuse", family="F3"), 6.6),
            (_spec("vllm", 0.5, 0.6, arm="corpus-reuse", family="F3"), 6.6),
        ]
        run_dir = _build_run(tmp_path, cells)
    finally:
        LATE_BY_RUNG.clear(); LATE_BY_RUNG.update(saved)
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(_plan(tmp_path))]) == 0
    _, stats = _load_stats(run_dir)
    (iu,) = stats["frontier"]["per_dataset_intersection"]
    assert set(iu["per_arm"]) == {"B1", "B3"}
    assert iu["per_arm"]["B1"]["verdict"].startswith("drop confirmed")
    assert iu["per_arm"]["B3"]["verdict"] == "no drop"
    assert iu["p_endpoint"] < 0.03 < iu["p_intersection_union_sensitivity"]
    gate = stats["gatekeeping"]
    (decision,) = [d for d in gate["set_decisions"] if d["endpoint"] == "contrast-12"]
    assert decision["confirmed"] is True
