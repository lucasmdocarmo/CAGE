"""AN-1 (ADR-0210, owner 2026-11-20): the budget comparison at a matched load fraction.

S2 offers each cell a fraction of its OWN rate limit, so a constrained cell and
its control never share an absolute rate and contrast #12 finds no point. AN-1
pairs them at the same load fraction and reports, per leg, the yield share
delta with its Welch interval, the capacity ratio and the yield-rate ratio;
per class, the capacity ratio with the ladder's bracket and its ratio over the
reference class; per engine and load, a pooled row. Descriptive: no alpha, no
gate. Spec: MyDocs/RunPod/S2_PLAN_A_B_QWEN3_235B_2026-11-19.md sections 16 to 16.9.

The fixture is a sealed tree built like tests/test_frontier_e1.py's (real
predicate producer, real organizer); every expected number is recounted here
from the generator's rules.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
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

RUN_ID, CAMPAIGN, SESSION, MODEL, DATASET = "20261120-1200-s1-qwen3-14b", "camp1", "s1", "qwen3-14b", "squad_v2"
SCORING_ID = "s01-fast"
N, N_FALSE, FAIL_FROM, TARGET_S = 20, 2, 14, 2.0
#: baseline -> (arm, retriever, family, class tokens)
ARMS = {"B1": ("gold-fresh", "none", "F2", 4779), "B3": ("corpus-reuse", "none", "F3", 2336),
        "B4": ("corpus-fresh", "none", "F3", 2336), "B6": ("retr-fresh", "rerank", "F2", 1127),
        # AN-2: three more arms, each in the class of an arm above (B2 with B1, B5 with B6, B10 with B3)
        "B2": ("gold-reuse", "none", "F3", 4779), "B5": ("retr-fresh", "dense", "F2", 1127),
        "B10": ("corpus-comp", "none", "F3", 2336)}
#: the rate limit per (baseline, rung), in requests per second: capacity ratios 0.5, 0.75 and
#: 0.4 at r = 0.25; B4 shares B3's token class, so it shares B3's limit. The rates are slow
#: enough that the 20-request pool of every window outlasts the ladder's 10 s warm-up.
LIMIT = {("B1", 1.0): 0.4, ("B1", 0.5): 0.3, ("B1", 0.25): 0.2, ("B1", 1.5): 0.44,
         ("B3", 1.0): 0.6, ("B3", 0.25): 0.45, ("B4", 1.0): 0.6, ("B4", 0.25): 0.45,
         ("B6", 1.0): 1.0, ("B6", 0.25): 0.4, ("B2", 1.0): 0.4, ("B2", 0.25): 0.2,
         ("B5", 1.0): 1.0, ("B5", 0.25): 0.4, ("B10", 1.0): 0.6, ("B10", 0.25): 0.45}
STEP = 1.075  # the ladder's bracket: first unsustainable rate = limit x 1.075
WINDOWS = {0.95: 2, 1.05: 3}
Key = tuple[str, str, float, float]  # baseline, engine, rung, load fraction


@pytest.fixture(autouse=True)
def _pressure_dataset_and_no_freeze(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(families, "PRESSURE_DATASET", DATASET)
    monkeypatch.setenv(bpt.FREEZE_ENV_VAR, str(tmp_path / "absent_freeze.json"))


def _lates(key: Key) -> tuple[int, ...]:
    b, _e, r, f = key
    base = 2 + {1.5: 0, 1.0: 0, 0.5: 2, 0.25: 4}[r] + {0.95: 0, 1.05: 3}[f] + {"B1": 0, "B3": 1, "B6": 2, "B4": 3, "B2": 2, "B5": 1, "B10": 2}[b]
    return tuple(base + (w % 2) for w in range(WINDOWS[f]))


def _grid(engines: tuple[str, ...] = ("vllm",), baselines: tuple[str, ...] = ("B1", "B3", "B6")) -> list[Key]:
    return [(b, e, r, f) for e in engines for b in baselines for r in (1.0, 0.25) for f in (0.95, 1.05)]


def _shares(late: int, sustained: bool) -> tuple[float, float, float]:
    """(timely, accepted, yield) of one window, recounted from the generator."""
    ok = set(range(N if sustained else FAIL_FROM))
    accepted, timely = {i for i in ok if i >= N_FALSE}, {i for i in ok if i >= late}
    return len(timely) / N, len(accepted) / N, len(timely & accepted) / N


def _write_window(wdir: Path, *, late: int, rate: float, sustained: bool, clocks: bool, delay: float = 0.5) -> list[Path]:
    wdir.mkdir(parents=True)
    requests, evidence = [], []
    for i in range(N):
        ok = sustained or i < FAIL_FROM
        ttft = 3000.0 if i < late else 400.0 + i
        row: dict[str, Any] = {
            "example_id": f"e{i:03d}", "ok": ok, "workload_mode": "open_loop",
            "ttft_ms": ttft if ok else None, "ttft_from_scheduled_ms": ttft + 5.0 if ok else None,
            "tpot_ms": 20.0 if ok else None, "num_tokens": 8 if ok else 0, "latency_ms": ttft + 100.0,
            # a real campaign tree: the runner serves with the quality scorers off, so requests.jsonl
            # carries the key with a null and the score lives in the scoring pass alone (review 2026-11-20)
            "faithfulness": None,
        }
        if clocks:  # one arrival every 1 / rate seconds, each completion `delay` seconds after its arrival
            arrival = i / (rate or 1.0)
            row.update(arrival_s=arrival, scheduled_ts=1000.0 + arrival, completion_ts=(1000.0 + arrival + delay) if ok else None)
        requests.append(json.dumps(row))
        evidence.append(json.dumps({
            "example_id": f"e{i:03d}", "baseline": "blind-token", "repeat_index": 0, "record_index": None,
            "generated_answer": f"answer {i}" if ok else "", "ok": ok, "error": None if ok else "HTTP 500",
            "empty_generation": False,
        }))
    payloads = {
        "requests.jsonl": "\n".join(requests) + "\n", "qa_evidence.jsonl": "\n".join(evidence) + "\n",
        "engine_metrics.json": json.dumps({"snapshot": "x"}),
        "cage_stats.jsonl": json.dumps({"ts_s": 0.0, "kv_cache_usage": 0.9}) + "\n",
        "metrics.json": json.dumps({"workload": {"mode": "open_loop", "open_loop": {"schedule": {"rate_qps": rate}}}}),
        "regime.json": json.dumps({"label": "IN_REGIME" if sustained else "OVERLOAD", "stationarity": {"label": "stationary"}}),
    }
    for name, text in payloads.items():
        (wdir / name).write_text(text, encoding="utf-8")
    return [wdir / name for name in payloads]


def _build(tmp_path: Path, keys: list[Key], *, lates: dict[Key, tuple[int, ...]] | None = None,
           sustained: dict[Key, bool] | None = None, clocks: bool = True, rates: dict[Key, float] | None = None,
           with_f1: bool = False, delay: dict[Key, float] | None = None, extra_dataset: str | None = None) -> Path:
    # extra_dataset: every pressure cell also serves that dataset (one row key, two datasets, as S1's
    # 16 shared pressure keys). So that a read that mixes the two datasets cannot pass, the extra
    # dataset differs on everything a comparison reads: its control cells run 2 requests later (its
    # own deltas), it is offered half the rate, and all of its windows are sustainable.
    datasets = [DATASET] + ([extra_dataset] if extra_dataset else [])
    run_dir = tmp_path / "results" / CAMPAIGN / SESSION / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "campaign": CAMPAIGN, "session": SESSION, "run_id": RUN_ID, "model": MODEL, "git_sha": "deadbeef",
        "git_dirty": False, "engine": "vllm", "engine_version": "0.0-test", "seed": 1, "provider": "test",
        "hardware": "test-gpu", "dataset_manifests_sha256": "0" * 64, "cellspec_schema_version": 1,
        "created_utc": "2026-11-20T12:00:00Z", "datasets": datasets,
        "slo_floors": {"vllm": {"ttft_s": 0.1, "tpot_s": 0.05}, "sglang": {"ttft_s": 0.1, "tpot_s": 0.05}},
    }), encoding="utf-8")
    cells: list[tuple[CellSpec, Key | None]] = []
    for key in keys:
        b, e, r, f = key
        arm, retriever, family, _cls = ARMS[b]
        cells.append((CellSpec(arm=arm, retriever=retriever, policy="none", topology="single", engine=e,  # type: ignore[arg-type]
                               model=MODEL, family=family, budget_r=r, rate_frac=f), key))  # type: ignore[arg-type]
    if with_f1:
        cells += [(CellSpec.from_baseline(b, model=MODEL), None) for b in ("B3", "B6")]  # type: ignore[arg-type]
    sealed: list[Path] = []
    for spec, key in cells:
        baseline = org.BASELINE_OF_CELL.get((spec.arm, spec.retriever), "")
        cell_dir = run_dir / "cells" / spec.to_row_key()
        cell_dir.mkdir(parents=True)
        if key is None:
            # the no-load pair differs on its first-token times, so contrast #4 has pairs to rank
            window_lates, ok_window, rate = ((0, 0) if baseline == "B3" else (5, 5)), True, 1.0
        else:
            assert baseline == key[0], (baseline, key)
            window_lates = (lates or {}).get(key, _lates(key))
            ok_window = (sustained or {}).get(key, key[3] < 1.0)
            rate = (rates or {}).get(key, key[3] * LIMIT[(key[0], key[2])])
        windows: dict[str, dict[str, Any]] = {}
        for dataset in (datasets if key is not None else [DATASET]):
            extra = dataset != DATASET and key is not None
            shift = 2 if (extra and key[2] == 1.0) else 0
            for ordinal, late in enumerate(window_lates, start=1):
                k = f"{dataset}-{ordinal:02d}"
                windows[k] = {"dataset": dataset, "seed": 1, "rep": ordinal, "budget_r": spec.budget_r,
                              "rate_frac": spec.rate_frac, "t_start": 0.0, "t_end": 60.0}
                sealed += _write_window(cell_dir / f"window_{k}", late=late + shift, rate=rate * (0.5 if extra else 1.0),
                                        sustained=ok_window or extra, clocks=clocks,
                                        delay=(delay or {}).get(key, 0.5) if key is not None else 0.5)
        cell: dict[str, Any] = {"cellspec": spec.to_flat_dict(), "baseline": baseline, "windows": windows, "gpu_count": 1}
        if key is not None:
            cell["budget_plan"] = {"avg_seq_tokens": ARMS[key[0]][3], "r": spec.budget_r}
        (cell_dir / "cell.json").write_text(json.dumps(cell), encoding="utf-8")
        sealed.append(cell_dir / "cell.json")
    write_ledger(hash_artifacts(sealed, base_dir=run_dir), run_dir / "ledger.json")
    raw_sha = json.loads((run_dir / "ledger.json").read_text())["entries_sha256"]
    scoring_dir = run_dir / "scoring" / SCORING_ID
    written: list[Path] = []
    for ev in sorted(run_dir.glob("cells/*/window_*/qa_evidence.jsonl")):
        out = scoring_dir / ev.parent.relative_to(run_dir)
        out.mkdir(parents=True)
        rows = [json.dumps({"example_id": f"e{i:03d}", "repeat_index": "0", "record_index": None,
                            "exact_match": float(i >= N_FALSE), "f1_score": float(i >= N_FALSE),
                            "grounding_score": None}) for i in range(N)]
        (out / "qa_scores.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
        (out / "quality.json").write_text(json.dumps({"rows": N}), encoding="utf-8")
        written += [out / "qa_scores.jsonl", out / "quality.json"]
    manifest = scoring_dir / "scoring_manifest.json"
    manifest.write_text(json.dumps({"scoring_run_id": SCORING_ID, "created_utc": "2026-11-20T13:00:00Z",
                                    "raw_run_ledger_entries_sha256": raw_sha, "mode": "fast"}), encoding="utf-8")
    write_ledger(hash_artifacts([*written, manifest], base_dir=scoring_dir), scoring_dir / "ledger.json")
    assert bpt.main([str(run_dir), "--scoring-run-id", SCORING_ID, "--max-null-fraction", "0.5"]) == 0
    assert org.main([str(run_dir)]) == 0
    return run_dir


def _rungs(tmp_path: Path, engine: str = "vllm", step: float = STEP, scale: float | dict[float, float] = 1.0) -> Path:
    """One engine's rung calibration file: every class laddered at 1.0 and 0.25, on its first arm;
    ``scale`` multiplies every limit (an engine with a different capacity), per rung when a mapping."""
    by_rung = scale if isinstance(scale, dict) else {1.0: scale, 0.25: scale}
    classes: dict[str, Any] = {}
    for b, (_arm, _ret, _fam, cls) in ARMS.items():
        classes.setdefault(str(cls), {"baseline_id": b, "ladder_prefix_mode": "OFF", "rungs": {
            f"{r:g}": {"label": "ESTIMATED", "lambda_star_qps": LIMIT[(b, r)] * by_rung[r],
                       "sustained_rate_qps": LIMIT[(b, r)] * by_rung[r],
                       "first_unsustainable_qps": LIMIT[(b, r)] * by_rung[r] * step, "descent_steps": 0}
            for r in (1.0, 0.25)}})
    path = tmp_path / f"rungs_T_{engine}.json"
    path.write_text(json.dumps({"schema": "cage-rung-calibration-v2", "engine": engine, "classes": classes}), encoding="utf-8")
    return path


def _plan(tmp_path: Path, rung_files: tuple[Path, ...] = (), *, sha: str | None = None) -> Path:
    artifacts = {json.loads(p.read_text())["engine"]: {"sha256": sha or hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in rung_files}
    path = tmp_path / "plan_registered.json"
    path.write_text(json.dumps({
        "class_first_token_targets_s": {str(cls): TARGET_S for _a, _r, _f, cls in ARMS.values()},
        "calibration": {"budget_fraction": 1.0}, "rung_calibration": {"artifacts": artifacts}}), encoding="utf-8")
    return path


def _run(run_dir: Path, *args: str) -> tuple[Path, dict[str, Any]]:
    # the default contrast (#4) needs no-load cells; #13 is computed on every look and needs none
    request = () if "--contrasts" in args else ("--contrasts", "13")
    assert rca.main([str(run_dir), *request, *args]) == 0
    analysis_dir = sorted((run_dir / "analysis").iterdir())[-1]
    return analysis_dir, json.loads((analysis_dir / "stats.json").read_text(encoding="utf-8"))


def _legs(section: dict[str, Any]) -> dict[tuple[str, str, float, float], dict[str, Any]]:
    return {(leg["baseline"], leg["engine"], leg["budget_r"], leg["rate_frac"]): leg for leg in section["legs"]}


def _mean(key: Key, col: int, sustained: bool) -> float:
    return statistics.fmean(_shares(late, sustained)[col] for late in _lates(key))


# ---------------------------------------------------------------------------


def test_one_leg_is_the_hand_count_and_contrast_12_still_refuses(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = _build(tmp_path, _grid())
    plan = _plan(tmp_path)
    analysis_dir, stats = _run(run_dir, "--plan", str(plan))
    section = stats["matched_load"]
    assert section["tier"] == "descriptive" and section["control_budget_r"] == 1.0
    assert (section["n_pairs_found"], section["n_legs"]) == (6, 6)
    leg = _legs(section)[("B1", "vllm", 0.25, 0.95)]
    con, ctl = ("B1", "vllm", 0.25, 0.95), ("B1", "vllm", 1.0, 0.95)
    for name, col in (("timely", 0), ("accepted", 1), ("yield", 2)):
        assert leg[name]["mean_constrained"] == pytest.approx(_mean(con, col, True))
        assert leg[name]["mean_control"] == pytest.approx(_mean(ctl, col, True))
        assert leg[name]["delta"] == pytest.approx(_mean(con, col, True) - _mean(ctl, col, True))
    assert leg["yield"]["delta"] == pytest.approx(-0.2) and leg["accepted"]["delta"] == pytest.approx(0.0)
    assert (leg["rate_qps_constrained"], leg["rate_qps_control"]) == (pytest.approx(0.19), pytest.approx(0.38))
    assert leg["capacity_ratio"] == pytest.approx(0.5)
    assert leg["yield_rps_constrained"] == pytest.approx(0.19 * 0.675)
    assert leg["yield_rps_ratio"] == pytest.approx(0.5 * 0.675 / 0.875)
    assert (leg["n_windows_constrained"], leg["n_windows_control"]) == (2, 2)
    assert leg["class_tokens"] == 4779 and leg["target_s"] == TARGET_S
    assert leg["regime_labels"] == {"constrained": ["IN_REGIME"] * 2, "control": ["IN_REGIME"] * 2}
    # AN-1f: the Welch interval with the spread and the count, two-sided, and no bootstrap
    y = leg["yield"]
    assert y["executed_alternative"] == "two-sided" and y["df"] == pytest.approx(2.0)
    assert y["sd_constrained"] == pytest.approx(statistics.stdev([0.70, 0.65]))
    assert y["ci95_low_welch"] < y["delta"] < y["ci95_high_welch"]
    assert not any("bootstrap" in k for k in y) and "bootstrap" not in section
    summary = (analysis_dir / "summary.md").read_text(encoding="utf-8")
    block = summary.split("## Budget comparison at a matched load fraction (AN-1", 1)[1].split("\n## ", 1)[0]
    # review 2026-11-20 HIGH: the block carries the three tables, not the section's (empty) skip list
    assert "SKIPPED" not in block and "| squad_v2 | B1|r0.25|lam0.95 | vllm | 2/2 | 0.675 | 0.875 | -0.200 [" in block
    assert "| squad_v2 | vllm | 4779 | B1 | 0.25 | 0.500 n/a |" in block
    assert "- squad_v2 vllm r0.25 load 0.95: 3 arm(s), delta yield median" in block
    # contrast #12 pairs by absolute rate and finds no point on the same tree
    assert rca.main([str(run_dir), "--contrasts", "12", "--plan", str(plan)]) == 1
    assert "no confirmatory point" in capsys.readouterr().err


def test_legs_pair_one_load_fraction_and_an_unpaired_cell_is_listed(tmp_path: Path) -> None:
    keys = [k for k in _grid(("vllm", "sglang")) if k != ("B1", "vllm", 1.0, 1.05)]
    _, stats = _run(_build(tmp_path, keys), "--plan", str(_plan(tmp_path)))
    section = stats["matched_load"]
    legs = _legs(section)
    assert len(legs) == 11 and ("B1", "vllm", 0.25, 1.05) not in legs
    for (_b, _e, _r, f), leg in legs.items():
        assert f"lam{f:g}" in leg["constrained_row_key"] and f"lam{f:g}" in leg["control_row_key"]
    (unpaired,) = section["unpaired"]
    assert "gold-fresh" in unpaired["row_key"] and "lam1.05" in unpaired["row_key"] and "no control" in unpaired["reason"]


def test_each_lower_rung_is_its_own_leg_and_a_rung_above_the_control_is_listed(tmp_path: Path) -> None:
    keys = _grid(baselines=("B1",)) + [("B1", "vllm", 0.5, 0.95), ("B1", "vllm", 1.5, 0.95)]
    _, stats = _run(_build(tmp_path, keys), "--plan", str(_plan(tmp_path)))
    section = stats["matched_load"]
    assert sorted(k[2:] for k in _legs(section)) == [(0.25, 0.95), (0.25, 1.05), (0.5, 0.95)]
    assert _legs(section)[("B1", "vllm", 0.5, 0.95)]["capacity_ratio"] == pytest.approx(0.75)
    (above,) = section["above_control"]
    assert "r1.5" in above["row_key"]


def test_one_window_or_no_variance_reports_the_estimates_without_a_test(tmp_path: Path) -> None:
    one, flat_c, flat_k = ("B1", "vllm", 0.25, 0.95), ("B3", "vllm", 0.25, 0.95), ("B3", "vllm", 1.0, 0.95)
    run_dir = _build(tmp_path, _grid(), lates={one: (6,), flat_c: (7, 7), flat_k: (3, 3)})
    _, stats = _run(run_dir, "--plan", str(_plan(tmp_path)))
    legs = _legs(stats["matched_load"])
    single, flat = legs[one], legs[flat_c]
    assert single["yield"]["delta"] == pytest.approx(0.70 - 0.875) and single["yield"]["p_value"] is None
    assert "2 windows" in single["yield"]["test_skipped"] and single["yield"]["sd_constrained"] is None
    assert flat["yield"]["delta"] == pytest.approx(-0.2) and flat["yield"]["p_value"] is None
    assert "zero variance" in flat["yield"]["test_skipped"]


def test_a_cell_the_helpers_refuse_is_a_labeled_skip_and_the_look_goes_on(tmp_path: Path) -> None:
    bad = ("B3", "vllm", 0.25, 1.05)
    _, stats = _run(_build(tmp_path, _grid(), rates={bad: 0.0}), "--plan", str(_plan(tmp_path)))
    section = stats["matched_load"]
    assert (section["n_pairs_found"], section["n_legs"]) == (6, 5) and bad not in _legs(section)
    (skip,) = section["skipped"]
    assert "corpus-reuse" in skip["pair"] and "positive offered rate" in skip["reason"]
    # the same arm's leg at the other load stays; its limit check cannot read the refused sibling
    sibling = _legs(section)[("B3", "vllm", 0.25, 0.95)]
    assert sibling["yield"]["p_value"] is not None and sibling["limit_check"]["constrained"]["verdict"] == "unavailable"
    assert sibling["limit_check"]["control"]["verdict"] == "confirmed"


def test_without_the_plan_the_section_is_one_labeled_skip_and_nothing_else_moves(tmp_path: Path) -> None:
    run_dir = _build(tmp_path, _grid(baselines=("B3", "B6")), with_f1=True)
    _, bare = _run(run_dir, "--contrasts", "4")
    _, planned = _run(run_dir, "--contrasts", "4", "--plan", str(_plan(tmp_path)))
    assert set(bare["matched_load"]) == {"skipped"} and "--plan" in bare["matched_load"]["skipped"]
    bare_dir = sorted((run_dir / "analysis").iterdir())[0]
    assert "SKIPPED (labeled): no --plan supplied" in (bare_dir / "summary.md").read_text(encoding="utf-8")
    assert planned["matched_load"]["n_legs"] == 4
    varying = {"matched_load", "generated_utc", "cut_roster"}  # the plan also feeds the cut roster note
    assert {k: v for k, v in bare.items() if k not in varying} == {k: v for k, v in planned.items() if k not in varying}
    assert bare["gatekeeping"] == planned["gatekeeping"] and bare["gatekeeping"]["primaries"]


def test_holm_covers_the_named_family_alone(tmp_path: Path) -> None:
    _, stats = _run(_build(tmp_path, _grid(("vllm", "sglang"))), "--plan", str(_plan(tmp_path)))
    section = stats["matched_load"]
    family = [leg for leg in section["legs"] if leg["in_holm_family"]]
    assert sorted((leg["baseline"], leg["engine"], leg["rate_frac"]) for leg in family) == [
        ("B3", "sglang", 1.05), ("B3", "vllm", 1.05), ("B6", "sglang", 1.05), ("B6", "vllm", 1.05)]
    adjusted = holm([leg["yield"]["p_value"] for leg in family])
    assert [leg["yield"]["p_holm_family"] for leg in family] == pytest.approx([float(p) for p in adjusted])
    assert section["holm_family"] == {**section["holm_family"], "baselines": ["B3", "B6"], "rate_frac": 1.05, "n_tests": 4}
    assert all(leg["yield"]["p_holm_family"] is None for leg in section["legs"] if not leg["in_holm_family"])


def test_the_capacity_ratio_carries_the_ladders_bracket_and_its_ratio_over_the_reference_class(tmp_path: Path) -> None:
    run_dir = _build(tmp_path, _grid())
    rungs = _rungs(tmp_path)
    _, stats = _run(run_dir, "--plan", str(_plan(tmp_path, (rungs,))), "--rung-calibration", str(rungs))
    section = stats["matched_load"]
    classes = {row["class_tokens"]: row for row in section["classes"]}
    assert set(classes) == {4779, 2336, 1127} and section["reference_baseline"] == "B6"
    for b, kappa in (("B1", 0.5), ("B3", 0.75), ("B6", 0.4)):
        row = classes[ARMS[b][3]]
        assert row["baselines"] == [b] and row["engine"] == "vllm" and row["budget_r"] == 0.25
        assert row["capacity_ratio"] == pytest.approx(kappa)
        assert row["bracket"] == pytest.approx([kappa / STEP, kappa * STEP])
        assert row["ladder"]["constrained"]["first_unsustainable_qps"] == pytest.approx(LIMIT[(b, 0.25)] * STEP)
        assert row["ladder"]["baseline_id"] == b and row["ladder"]["ladder_prefix_mode"] == "OFF"
    cag, rag = classes[2336], classes[1127]
    assert rag["over_reference"]["is_reference"] is True and rag["over_reference"]["ratio"] == pytest.approx(1.0)
    assert cag["over_reference"]["ratio"] == pytest.approx(0.75 / 0.4)
    assert cag["over_reference"]["bracket"] == pytest.approx([0.75 / 0.4 / STEP**2, 0.75 / 0.4 * STEP**2])
    assert cag["over_reference"]["separable"] is True


def test_a_missing_or_foreign_rung_file_leaves_the_bracket_null_with_the_reason(tmp_path: Path) -> None:
    run_dir = _build(tmp_path, _grid(baselines=("B3", "B6")))
    rungs = _rungs(tmp_path)
    _, bare = _run(run_dir, "--plan", str(_plan(tmp_path)))
    _, foreign = _run(run_dir, "--plan", str(_plan(tmp_path, (rungs,), sha="0" * 64)), "--rung-calibration", str(rungs))
    for stats, word in ((bare, "--rung-calibration"), (foreign, "sha256")):
        for row in stats["matched_load"]["classes"]:
            assert row["capacity_ratio"] is not None and row["bracket"] is None and word in row["bracket_skipped"]
            assert row["over_reference"]["bracket"] is None and row["over_reference"]["separable"] is None


def test_the_pooled_row_is_the_median_and_range_over_the_arms(tmp_path: Path) -> None:
    _, stats = _run(_build(tmp_path, _grid()), "--plan", str(_plan(tmp_path)))
    section = stats["matched_load"]
    pooled = {(row["engine"], row["budget_r"], row["rate_frac"]): row for row in section["pooled"]}
    assert set(pooled) == {("vllm", 0.25, 0.95), ("vllm", 0.25, 1.05)}
    row = pooled[("vllm", 0.25, 0.95)]
    deltas = [leg["yield"]["delta"] for leg in section["legs"] if leg["rate_frac"] == 0.95]
    assert row["n_arms"] == 3 and row["n_negative_delta"] == 3
    assert row["yield_delta"] == pytest.approx({"median": statistics.median(deltas), "min": min(deltas), "max": max(deltas)})
    assert row["capacity_ratio"] == pytest.approx({"median": 0.5, "min": 0.4, "max": 0.75})


def test_the_limit_check_reads_the_ladders_rule_on_the_served_windows(tmp_path: Path) -> None:
    late_limit = ("B3", "vllm", 0.25, 1.05)   # still sustainable above the limit: the limit sits too low
    early_limit = ("B6", "vllm", 1.0, 0.95)   # unsustainable below the limit: the limit sits too high
    slow = ("B1", "vllm", 0.25, 0.95)         # every request completes, 30 s late: served too few in the window
    run_dir = _build(tmp_path, _grid(), sustained={late_limit: True, early_limit: False}, delay={slow: 30.0})
    _, stats = _run(run_dir, "--plan", str(_plan(tmp_path)))
    legs = _legs(stats["matched_load"])
    assert legs[("B3", "vllm", 0.25, 0.95)]["limit_check"]["control"] == {"verdict": "confirmed", "detail": None}
    # review 2026-11-20 (test gap): attainment 1.0 and a served ratio below 0.9 is unsustainable too
    b1 = legs[slow]
    assert b1["ladder_rule"]["constrained"][0]["attainment"] == pytest.approx(1.0)
    assert b1["ladder_rule"]["constrained"][0]["served_ratio"] < 0.9 and b1["ladder_rule"]["constrained"][0]["sustainable"] is False
    assert b1["limit_check"]["constrained"]["verdict"] == "not confirmed" and b1["limit_check"]["control"]["verdict"] == "confirmed"
    b3, b6 = legs[late_limit]["limit_check"], legs[("B6", "vllm", 0.25, 0.95)]["limit_check"]
    assert b3["constrained"]["verdict"] == "not confirmed" and "3 of 3 windows above 1.0 sustainable" in b3["constrained"]["detail"]
    assert b3["control"]["verdict"] == "confirmed"
    assert b6["control"]["verdict"] == "not confirmed" and "2 of 2 windows below 1.0 unsustainable" in b6["control"]["detail"]
    window = legs[late_limit]["ladder_rule"]["constrained"][0]
    # 20 requests at 1.05 x 0.45 req/s: the window ends at 20 / 0.4725 s; 15 arrivals after the warm-up, all served in it
    assert window == {"attainment": pytest.approx(1.0), "served_ratio": pytest.approx(1.0), "sustainable": True,
                      "window_end_s": pytest.approx(20 / 0.4725)}


def test_rows_without_clocks_make_the_limit_check_unavailable_and_keep_the_leg(tmp_path: Path) -> None:
    _, stats = _run(_build(tmp_path, _grid(baselines=("B1",)), clocks=False), "--plan", str(_plan(tmp_path)))
    leg = _legs(stats["matched_load"])[("B1", "vllm", 0.25, 0.95)]
    assert leg["yield"]["delta"] == pytest.approx(-0.2)
    assert leg["limit_check"]["constrained"]["verdict"] == "unavailable" and "arrival_s" in leg["limit_check"]["constrained"]["detail"]


def test_an_active_blinding_seal_suppresses_the_section(tmp_path: Path) -> None:
    from src.analysis.stats.blinding import scramble_labels

    run_dir = _build(tmp_path, _grid())
    index = pd.read_csv(run_dir / "index" / "cells_index.csv")
    scramble_labels(pd.DataFrame({"arm": sorted(index["arm"].unique())}), "arm", seed=11,
                    sealed_map_path=run_dir / "blinding" / "sealed_arm_map.json")
    analysis_dir, stats = _run(run_dir, "--plan", str(_plan(tmp_path)))
    assert stats["blinding"]["active"] is True and set(stats["matched_load"]) == {"suppressed"}
    assert "AN-1" not in (analysis_dir / "summary.md").read_text(encoding="utf-8")


def _rule(tmp_path: Path, rows: list[dict[str, Any]], rate: float) -> dict[str, Any]:
    wdir = tmp_path / "w"
    wdir.mkdir(exist_ok=True)
    (wdir / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return rca._window_ladder_rule(tmp_path, SimpleNamespace(window_dir="w"), rate)


def test_the_served_window_ends_at_the_pools_design_duration(tmp_path: Path) -> None:
    # review 2026-11-20 MEDIUM: the window is [10 s, pool / rate), the analog of the ladder's fixed 75 s;
    # the last Poisson arrival moved the end by seconds on real windows (79.98 s to 87.0 s for 85.03 s)
    def row(arrival: float, done: float | None) -> dict[str, Any]:
        return {"ok": done is not None, "arrival_s": arrival, "scheduled_ts": 500.0 + arrival,
                "completion_ts": None if done is None else 500.0 + done}

    rows = [row(2.0, 3.0), row(11.0, 12.0), row(20.0, 21.0), row(30.0, 39.5), row(38.0, 41.0), row(45.0, 46.0)]
    rule = _rule(tmp_path, rows, rate=0.15)   # 6 requests at 0.15 req/s: the window ends at 40 s
    # scheduled: the arrivals at 11, 20, 30 and 38 s (2 s is warm-up, 45 s is past the end), all completed;
    # served inside [10, 40): the completions at 12, 21 and 39.5 s (41 s and 46 s are past the end)
    assert rule == {"attainment": pytest.approx(1.0), "served_ratio": pytest.approx(0.75), "sustainable": False,
                    "window_end_s": pytest.approx(40.0)}
    late_only = _rule(tmp_path, [row(1.0, 2.0), row(3.0, 4.0)], rate=1.0)   # the pool ends before the warm-up does
    assert late_only["sustainable"] is None and "warm-up" in late_only["reason"]


def test_flat_shares_with_a_rounding_variance_are_not_tested(tmp_path: Path) -> None:
    # review 2026-11-20 LOW: three identical shares of 14/20 have a computed SD of 1e-17, not 0;
    # Welch on two such sides returned p = 2e-46
    con, ctl = ("B3", "vllm", 0.25, 1.05), ("B3", "vllm", 1.0, 1.05)
    run_dir = _build(tmp_path, _grid(baselines=("B3", "B6")), lates={con: (6, 6, 6), ctl: (4, 4, 4)},
                     sustained={con: True, ctl: True})
    _, stats = _run(run_dir, "--plan", str(_plan(tmp_path)))
    y = _legs(stats["matched_load"])[con]["yield"]
    assert y["delta"] == pytest.approx(0.70 - 0.80) and y["p_value"] is None and "zero variance" in y["test_skipped"]
    assert (y["sd_constrained"], y["sd_control"]) == (0.0, 0.0)


def test_a_holm_refusal_keeps_every_leg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = rca._holm_or_refuse

    def refuse(p_values: Any, what: str) -> Any:
        if what.startswith("AN-1"):
            raise rca.AnalysisError(f"{what}: p-values must be finite")
        return real(p_values, what)

    monkeypatch.setattr(rca, "_holm_or_refuse", refuse)
    _, stats = _run(_build(tmp_path, _grid()), "--plan", str(_plan(tmp_path)))
    section = stats["matched_load"]
    assert section["n_legs"] == 6 and len(section["classes"]) == 3
    assert "p-values must be finite" in section["holm_family"]["skipped"][0]
    assert all(leg["yield"]["p_holm_family"] is None for leg in section["legs"])


def test_arms_of_one_class_share_one_capacity_ratio_and_a_wide_bracket_is_not_separable(tmp_path: Path) -> None:
    run_dir = _build(tmp_path, _grid(("vllm", "sglang"), ("B1", "B3", "B4", "B6")))
    wide = _rungs(tmp_path, "vllm", step=1.5)   # a ladder that brackets each limit within 50 percent
    _, stats = _run(run_dir, "--plan", str(_plan(tmp_path, (wide,))), "--rung-calibration", str(wide))
    section = stats["matched_load"]
    rows = {(row["engine"], row["class_tokens"]): row for row in section["classes"]}
    assert len(rows) == 6   # three classes on two engines: B3 and B4 are one class
    cag = rows[("vllm", 2336)]
    assert cag["baselines"] == ["B3", "B4"] and cag["capacity_ratio"] == pytest.approx(0.75) and cag["notes"] == []
    assert set(cag["limit_check_by_baseline"]) == {"B3", "B4"}
    assert cag["bracket"] == pytest.approx([0.75 / 1.5, 0.75 * 1.5])
    # 0.75 / 0.4 = 1.875, bracket 1.875 / 2.25 to 1.875 x 2.25: it holds 1.0, so the two classes do not separate
    assert cag["over_reference"]["bracket"][0] < 1.0 < cag["over_reference"]["bracket"][1]
    assert cag["over_reference"]["separable"] is False
    # one rung file of two: the engine without a file reports its ratio with no bracket, and says why
    other = rows[("sglang", 2336)]
    assert other["capacity_ratio"] == pytest.approx(0.75) and other["bracket"] is None
    assert "no --rung-calibration file for engine 'sglang'" in other["bracket_skipped"]
    assert section["holm_family"]["n_family_legs"] == 4 and section["holm_family"]["n_tests"] == 4


def test_a_row_key_served_on_two_datasets_gives_each_dataset_its_own_leg(tmp_path: Path) -> None:
    # review 2026-11-20 MEDIUM: one row key can hold windows of two datasets (16 of S1's 108 pressure keys);
    # a leg reads the windows, the rate and the limit check of its own dataset alone
    run_dir = _build(tmp_path, _grid(baselines=("B1",)), extra_dataset="musique")
    _, stats = _run(run_dir, "--plan", str(_plan(tmp_path)))
    section = stats["matched_load"]
    legs = {(leg["dataset"], leg["budget_r"], leg["rate_frac"]): leg for leg in section["legs"]}
    assert sorted(legs) == [(d, 0.25, f) for d in ("musique", DATASET) for f in (0.95, 1.05)]
    first, second = legs[(DATASET, 0.25, 0.95)], legs[("musique", 0.25, 0.95)]
    for leg in (first, second):
        assert (leg["n_windows_constrained"], leg["n_windows_control"]) == (2, 2)
    assert first["yield"]["delta"] == pytest.approx(-0.2) and first["yield"]["mean_control"] == pytest.approx(0.875)
    # the second dataset: its control cell runs 2 requests later, at half the rate, never unsustainable
    assert second["yield"]["mean_control"] == pytest.approx(0.775) and second["yield"]["delta"] == pytest.approx(-0.1)
    assert (first["rate_qps_constrained"], second["rate_qps_constrained"]) == (pytest.approx(0.19), pytest.approx(0.095))
    assert first["limit_check"]["control"]["verdict"] == "confirmed"
    assert second["limit_check"]["control"]["verdict"] == "not confirmed"
    assert "3 of 3 windows above 1.0 sustainable" in second["limit_check"]["control"]["detail"]
    # every derived row and every listed entry names its dataset
    assert sorted((row["dataset"], row["class_tokens"]) for row in section["classes"]) == [("musique", 4779), (DATASET, 4779)]
    assert sorted((row["dataset"], row["rate_frac"]) for row in section["pooled"]) == [
        (d, f) for d in ("musique", DATASET) for f in (0.95, 1.05)]


def test_listed_cells_name_their_dataset(tmp_path: Path) -> None:
    # review 2026-11-20 LOW: an unpaired cell and a rung above the control are listed once per dataset, named
    keys = [k for k in _grid(baselines=("B1",)) if k != ("B1", "vllm", 1.0, 1.05)] + [("B1", "vllm", 1.5, 0.95)]
    _, stats = _run(_build(tmp_path, keys, extra_dataset="musique"), "--plan", str(_plan(tmp_path)))
    section = stats["matched_load"]
    assert sorted(entry["dataset"] for entry in section["unpaired"]) == ["musique", DATASET]
    assert sorted(entry["dataset"] for entry in section["above_control"]) == ["musique", DATASET]
