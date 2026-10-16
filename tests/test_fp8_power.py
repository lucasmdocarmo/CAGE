"""Tests for scripts/4_analysis/fp8_power.py (Spec Block A3, 2026-10-16).

Synthetic run roots only; no GPU, no network. Each test pins one clause: the
identity-triple join and the loader validity rule, mismatched triples as a named
skip, the minimum pair count and the whole-pass refusal, the paired t formula
against the textbook value, the fourfold rule when the variance doubles, the
registered guard refusing a tie-heavy pool while the tie-preserving method still
answers, the tie-preserving injection's two invariants, the Wilcoxon simulation
bracketing the t answer on a continuous pool, the all-zero pool, the ladder
verdict and the CLI outputs.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "4_analysis" / "fp8_power.py"


def _load():
    spec = importlib.util.spec_from_file_location("fp8_power_a3", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


fp = _load()

SCORING = "full-cuda-1"


def _key(arm: str, engine: str = "vllm") -> str:
    return f"{arm}|none|none|single|{engine}|qwen3-14b|F1"


def _cell(run_root: Path, arm: str, engine: str = "vllm") -> None:
    key = _key(arm, engine)
    d = run_root / "cells" / key
    d.mkdir(parents=True, exist_ok=True)
    spec = {"arm": arm, "retriever": "none", "policy": "none", "topology": "single", "engine": engine,
            "model": "qwen3-14b", "family": "F1"}
    (d / "cell.json").write_text(json.dumps({"cellspec": spec, "windows": {"squad_v2-01": {"dataset": "squad_v2", "rep": 1}}}), encoding="utf-8")


def _scores(run_root: Path, arm: str, f1s, *, engine: str = "vllm", window: str = "window_squad_v2-01",
            invalid_at=(), ids=None) -> None:
    key = _key(arm, engine)
    d = run_root / "scoring" / SCORING / "cells" / key / window
    d.mkdir(parents=True, exist_ok=True)
    ids = ids or [f"ex{i}" for i in range(len(f1s))]
    with open(d / "qa_scores.jsonl", "w", encoding="utf-8") as fh:
        for i, (ex, f1) in enumerate(zip(ids, f1s)):
            row = {"example_id": ex, "repeat_index": "0", "record_index": None, "f1_score": f1,
                   "error": None, "empty_generation": False, "not_ok": False}
            if i in invalid_at:
                row["empty_generation"] = True
            fh.write(json.dumps(row) + "\n")


def _run_root(tmp_path: Path, cell_f1, ref_f1, **kw) -> Path:
    root = tmp_path / "run"
    _cell(root, "corpus-comp")
    _cell(root, "corpus-reuse")
    _scores(root, "corpus-comp", cell_f1, **{k: v for k, v in kw.items() if k != "ref_invalid"})
    _scores(root, "corpus-reuse", ref_f1, invalid_at=kw.get("ref_invalid", ()))
    return root


def test_pools_join_on_identity_and_drop_invalid_pairs(tmp_path: Path) -> None:
    n = 60
    cell = [0.5] * n
    ref = [0.5] * n
    cell[3] = 0.0  # one discordant pair
    root = _run_root(tmp_path, cell, ref, invalid_at=(7,), ref_invalid=(9,))
    (pool,) = fp.collect_pools(root, scoring_run_id=SCORING, cell_arm="corpus-comp", ref_arm="corpus-reuse")
    assert pool.skip_reason == ""
    assert pool.engine == "vllm" and pool.dataset == "squad_v2" and pool.family == "F1"
    assert pool.diffs.size == n - 2 and pool.n_dropped_invalid == 2
    assert np.isclose(pool.diffs.sum(), -0.5)


def test_mismatched_identity_triples_is_a_named_skip(tmp_path: Path) -> None:
    root = _run_root(tmp_path, [0.5] * 60, [0.5] * 59)
    (pool,) = fp.collect_pools(root, scoring_run_id=SCORING, cell_arm="corpus-comp", ref_arm="corpus-reuse")
    assert "mismatched identity triples" in pool.skip_reason and "refusing to intersect" in pool.skip_reason


def test_fewer_than_min_pairs_is_a_named_skip_and_the_pass_refuses(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    root = _run_root(tmp_path, [0.5] * 40, [0.4] * 40)
    (pool,) = fp.collect_pools(root, scoring_run_id=SCORING, cell_arm="corpus-comp", ref_arm="corpus-reuse")
    assert f"(< {fp.MIN_PAIRS})" in pool.skip_reason
    rc = fp.main([str(root), "--scoring-run-id", SCORING, "--out", str(tmp_path / "out"), "--n-sims", "5"])
    assert rc == fp.REFUSED_RC == 3
    assert "every pool is a named skip" in capsys.readouterr().err


def test_missing_reference_window_is_a_named_skip(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _cell(root, "corpus-comp")
    _scores(root, "corpus-comp", [0.5] * 60)
    (pool,) = fp.collect_pools(root, scoring_run_id=SCORING, cell_arm="corpus-comp", ref_arm="corpus-reuse")
    assert "reference arm 'corpus-reuse' has no scored window" in pool.skip_reason


def test_paired_t_matches_the_textbook_value() -> None:
    # sd 1, delta 0.5, alpha 0.05 two-sided, power 0.8: (1.96 + 0.8416)^2 * 4 = 31.4 -> 32
    n = fp.t_required_n(1.0, 0.5, alpha=0.05, power=0.8)
    assert n == 32 and abs(n - 31.4) / 31.4 < 0.10
    assert fp.t_required_n(0.0, 0.5, alpha=0.05, power=0.8) is None


def test_doubling_the_sd_quadruples_the_t_requirement() -> None:
    n1 = fp.t_required_n(0.12, 0.01, alpha=0.05, power=0.8)
    n2 = fp.t_required_n(0.24, 0.01, alpha=0.05, power=0.8)
    assert 3.9 < n2 / n1 < 4.1


def test_tie_preserving_injection_keeps_zeros_and_hits_the_target_mean() -> None:
    rng = np.random.default_rng(0)
    null = np.array([0.0] * 80 + [0.3, -0.3, 0.5, -0.5] * 5)
    out = fp.tie_preserving_shift(rng, null, 0.02)
    assert np.count_nonzero(out == 0.0) == 80          # every tie is still a tie
    assert np.isclose(out.mean() - null.mean(), 0.02)  # the pool mean rose by exactly the target
    assert np.array_equal(fp.tie_preserving_shift(rng, np.zeros(10), 0.02), np.zeros(10))  # no carrier: unchanged
    assert np.array_equal(fp.tie_preserving_shift(rng, null, 0.0), null)


def test_tie_heavy_pool_refuses_the_registered_shift_and_the_tie_preserving_method_answers(tmp_path: Path) -> None:
    # 90 percent zeros: collision probability above 0.81 -> the P0 guard refuses the shift.
    rng = np.random.default_rng(1)
    n = 100
    cell = [0.5] * n
    ref = [0.5] * n
    for i in rng.choice(n, size=10, replace=False):
        cell[int(i)] = 0.5 + float(rng.choice([-0.4, -0.2, 0.2, 0.4]))
    root = _run_root(tmp_path, cell, ref)
    (pool,) = fp.collect_pools(root, scoring_run_id=SCORING, cell_arm="corpus-comp", ref_arm="corpus-reuse")
    rows, tables = fp.analyze_pool(pool, targets=(0.02,), n_grid=(50, 200, 800), n_sims=20, seed=42, alpha=0.05)
    by_method = {r["method"]: r for r in rows}
    assert by_method[fp.METHOD_REGISTERED]["status"] == fp.STATUS_REFUSED
    assert "P0 2026-08-02" in by_method[fp.METHOD_REGISTERED]["note"]
    assert fp.METHOD_REGISTERED not in tables
    tp = by_method[fp.METHOD_TIE_PRESERVING]
    assert tp["status"] == fp.STATUS_SIMULATED and fp.METHOD_TIE_PRESERVING in tables
    assert isinstance(tp["required_n"], int) or tp["required_n"] == fp.NOT_REACHED
    assert tp["per_discordant_shift"] == pytest.approx(0.02 / 0.1)       # target / discordant rate
    assert 0.0 <= tp["realized_alpha_max"] <= 1.0 and "step function" in tp["note"]
    assert any(r["effect"] == 0.0 for r in tables[fp.METHOD_TIE_PRESERVING])  # the size row
    assert by_method[fp.METHOD_T]["status"] == fp.STATUS_CLOSED_FORM
    assert by_method[fp.METHOD_T]["tie_mass"] == pytest.approx(0.9)


def test_continuous_pool_runs_the_registered_shift_and_brackets_the_t_answer(tmp_path: Path) -> None:
    # F1-like continuous differences (no ties, inside the F1 range): the t approximation
    # on the SAMPLE sd gives the textbook n at delta 0.06 (about 66 to 88 for an sd near
    # 0.2); the Wilcoxon's efficiency against the t is about 0.955 under normality, so
    # power must be below 0.8 at n = 30 and above 0.8 at n = 150 for both simulated methods.
    rng = np.random.default_rng(7)
    diffs = np.clip(rng.normal(0.0, 0.2, size=200), -0.6, 0.6)
    root = _run_root(tmp_path, list(0.5 + diffs), [0.5] * 200)
    (pool,) = fp.collect_pools(root, scoring_run_id=SCORING, cell_arm="corpus-comp", ref_arm="corpus-reuse")
    sd = pool.diagnostics()["sd_diff"]
    assert pool.diagnostics()["collision_probability"] < fp.MAX_SHIFT_COLLISION
    expected = int(np.ceil(((1.959964 + 0.841621) * sd / 0.06) ** 2))
    assert fp.t_required_n(sd, 0.06, alpha=0.05, power=0.8) == expected and 30 < expected < 150
    rows, tables = fp.analyze_pool(pool, targets=(0.06,), n_grid=(30, 150), n_sims=300, seed=42, alpha=0.05)
    by_method = {r["method"]: r for r in rows}
    assert by_method[fp.METHOD_REGISTERED]["status"] == fp.STATUS_SIMULATED
    assert by_method[fp.METHOD_TIE_PRESERVING]["per_discordant_shift"] == pytest.approx(0.06)  # discordant rate 1
    for method in (fp.METHOD_REGISTERED, fp.METHOD_TIE_PRESERVING):
        power = {r["n"]: r["power"] for r in tables[method] if r["effect"] == 0.06}
        size = {r["n"]: r["power"] for r in tables[method] if r["effect"] == 0.0}
        assert power[30] < 0.8 < power[150], (method, power)
        assert all(v < 0.12 for v in size.values()), (method, size)  # the realized alpha row stays near 0.05
        assert by_method[method]["required_n"] == 150


def test_all_zero_pool_reports_no_variance(tmp_path: Path) -> None:
    root = _run_root(tmp_path, [0.5] * 60, [0.5] * 60)
    (pool,) = fp.collect_pools(root, scoring_run_id=SCORING, cell_arm="corpus-comp", ref_arm="corpus-reuse")
    rows, _ = fp.analyze_pool(pool, targets=(0.01,), n_grid=(50,), n_sims=5, seed=42, alpha=0.05)
    status = {r["method"]: r["status"] for r in rows}
    assert status[fp.METHOD_REGISTERED] == fp.STATUS_REFUSED      # collision probability 1.0
    assert status[fp.METHOD_TIE_PRESERVING] == fp.STATUS_NO_DISCORDANT
    assert status[fp.METHOD_T] == fp.STATUS_NO_VARIANCE
    assert all(r["required_n"] == "" for r in rows)


def test_ladder_verdict() -> None:
    assert fp.ladder_verdict(900) == ("1200", "within ladder")
    assert fp.ladder_verdict(1600) == ("1600", "within ladder")
    assert fp.ladder_verdict(2001) == ("", "exceeds ladder (max rung 2000)")
    assert fp.ladder_verdict(None)[0] == ""


def test_cli_writes_csv_json_markdown_and_recommendation(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    rng = np.random.default_rng(3)
    n = 100
    cell = [0.5] * n
    for i in rng.choice(n, size=20, replace=False):
        cell[int(i)] = 0.5 + float(rng.choice([-0.4, -0.2, 0.2, 0.4]))
    root = _run_root(tmp_path, cell, [0.5] * n)
    out = tmp_path / "out"
    rc = fp.main([str(root), "--scoring-run-id", SCORING, "--out", str(out), "--n-sims", "10", "--n-grid", "50", "200", "2000", "--targets", "0.02"])
    assert rc == 0
    rows = list(csv.DictReader(open(out / "fp8_power.csv", encoding="utf-8")))
    assert len(rows) == 3 and set(rows[0].keys()) == set(fp.CSV_COLUMNS)
    assert {r["method"] for r in rows} == {fp.METHOD_REGISTERED, fp.METHOD_TIE_PRESERVING, fp.METHOD_T}
    doc = json.loads((out / "fp8_power.json").read_text(encoding="utf-8"))
    assert doc["config"]["ladder"] == [2000, 1600, 1200] and doc["config"]["seed"] == 42
    md = (out / "fp8_power.md").read_text(encoding="utf-8")
    assert "## Recommendation for the S1 registration" in md and "squad_v2: delta 0.02:" in md
    assert "## Methods and limitations" in md
    assert "squad_v2: delta 0.02:" in capsys.readouterr().out


def test_cli_refuses_a_results_tree_and_same_arms(tmp_path: Path) -> None:
    root = _run_root(tmp_path, [0.5] * 60, [0.5] * 60)
    assert fp.main([str(root), "--scoring-run-id", SCORING, "--out", str(tmp_path / "results" / "x")]) == 3
    with pytest.raises(SystemExit):
        fp.main([str(root), "--scoring-run-id", SCORING, "--out", str(tmp_path / "o"), "--cell-arm", "a", "--ref-arm", "a"])


def test_shift_that_leaves_the_f1_range_is_refused_per_target(tmp_path: Path) -> None:
    # Review 2026-10-16 HIGH 2: 2 discordant pairs in 50 (rate 0.04) with |diff| 0.5: at
    # target 0.05 the per-pair shift is 1.25 and 0.5 + 1.25 > 1, impossible for an F1
    # difference; at target 0.01 the shift is 0.25 and 0.75 stays in range.
    cell = [0.5] * 50
    cell[0] = 1.0
    cell[1] = 0.0
    root = _run_root(tmp_path, cell, [0.5] * 50)
    (pool,) = fp.collect_pools(root, scoring_run_id=SCORING, cell_arm="corpus-comp", ref_arm="corpus-reuse")
    rows, tables = fp.analyze_pool(pool, targets=(0.01, 0.05), n_grid=(50, 200), n_sims=10, seed=42, alpha=0.05)
    tp = {r["target"]: r for r in rows if r["method"] == fp.METHOD_TIE_PRESERVING}
    assert tp[0.05]["status"] == fp.STATUS_OUT_OF_RANGE and tp[0.05]["required_n"] == ""
    assert tp[0.05]["per_discordant_shift"] == pytest.approx(1.25)
    assert tp[0.01]["status"] == fp.STATUS_SIMULATED and tp[0.01]["per_discordant_shift"] == pytest.approx(0.25)
    assert sorted({r["effect"] for r in tables[fp.METHOD_TIE_PRESERVING]}) == [0.0, 0.01]  # 0.05 never simulated


def test_seeds_do_not_shift_when_an_earlier_pool_is_skipped(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    # Review 2026-10-16 MEDIUM 3: the seed of a pool is seed + 100 * its index over ALL pools.
    root = _run_root(tmp_path, [0.5] * 50 + [], [0.5] * 50)
    # a second cell pair on sglang whose cell window mismatches (skipped), sorted BEFORE vllm's
    _cell(root, "corpus-comp", engine="sglang")
    _cell(root, "corpus-reuse", engine="sglang")
    _scores(root, "corpus-comp", [0.5] * 60, engine="sglang")
    _scores(root, "corpus-reuse", [0.5] * 59, engine="sglang")
    cell_vllm = [0.5] * 50
    for i in range(5):
        cell_vllm[i] = 0.6  # 5 discordant pairs of 0.1: shift 0.02 / 0.1 = 0.2 stays inside the F1 range
    _scores(root, "corpus-comp", cell_vllm)
    rc = fp.main([str(root), "--scoring-run-id", SCORING, "--out", str(tmp_path / "out"), "--n-sims", "5", "--n-grid", "50", "--targets", "0.02"])
    assert rc == 0
    doc = json.loads((tmp_path / "out" / "fp8_power.json").read_text(encoding="utf-8"))
    (skip,) = doc["skips"]
    assert skip["engine"] == "sglang"
    tp = [r for r in doc["rows"] if r["method"] == fp.METHOD_TIE_PRESERVING]
    assert len(tp) == 1 and "seed 143" in tp[0]["note"]  # 42 + 100 * 1 + 1: the vllm pool keeps index 1


def test_validity_rule_uses_the_loader_predicates(tmp_path: Path) -> None:
    assert fp._row_valid({"error": None, "empty_generation": False, "not_ok": False})
    assert fp._row_valid({"error": "none", "empty_generation": "false"})       # the loader's falsy strings
    assert not fp._row_valid({"error": "timeout", "empty_generation": False})
    assert not fp._row_valid({"error": None, "empty_generation": True})
    assert not fp._row_valid({"error": None, "empty_generation": False, "not_ok": True})
