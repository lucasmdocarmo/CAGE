"""S1 audit of 2026-10-17, batch 3 (owner: "keep going lets close all open
problems").

G22: the predicate builder's null-fraction sanity bound judges the completed
(ok) rows; a not-ok row is nulled by design and no longer deletes the table
when a past-cliff window's error share exceeds the bound. G31 with E12 (a):
the stage 13 regime gate, now scripts/6_experiments/regime_gate.py, requires
per (engine, arm, dataset, budget rung, rate rung >= 1.0) at least
MIN_IN_REGIME_WINDOWS windows labeled IN_REGIME and not transient. G34 is
pinned in tests/test_write_analysis.py (the summary reads mode_stamp). Each
test fails on the code before the change it names.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "scripts" / "6_experiments", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import regime_gate  # noqa: E402
from src.analysis.predicate import PredicateError, compute_window_predicate, join_window_rows  # noqa: E402
from test_predicate_chain import DATASET, _config, _evidence_row, _score_row  # noqa: E402
from test_run_experiment_campaign_layout import runner  # noqa: E402


# --------------------------------------------------------------------------- #
# G28: the adapter timeout covers the window span plus the drain
# --------------------------------------------------------------------------- #


def test_the_adapter_timeout_is_the_drain_plus_the_window_span() -> None:
    f = runner.adapter_request_timeout_s
    assert f(None, rate_qps=10.0, num_arrivals=1000, duration_s=None) is None  # no drain: adapter default
    # count mode: 1,060 arrivals at 10 rps span 106 s (the Qasper D_class)
    assert f(300.0, rate_qps=10.0, num_arrivals=1060, duration_s=None) == pytest.approx(406.0)
    # duration mode (dry window): the duration itself
    assert f(300.0, rate_qps=10.0, num_arrivals=None, duration_s=75.0) == pytest.approx(375.0)
    # neither known: the drain alone (pre-G28 value), never less
    assert f(300.0, rate_qps=None, num_arrivals=None, duration_s=None) == pytest.approx(300.0)
    assert runner.adapter_timeout_kwargs(f(300.0, rate_qps=10.0, num_arrivals=1060, duration_s=None)) == {"timeout": 406}


# --------------------------------------------------------------------------- #
# G22: the bound judges the completed rows
# --------------------------------------------------------------------------- #


def _window(n_ok: int, n_not_ok: int, n_missing_verdict: int) -> list[dict[str, Any]]:
    ev, sc = [], []
    i = 0
    for _ in range(n_ok):
        ev.append(_evidence_row(i)); sc.append(_score_row(i, exact_match=1.0)); i += 1
    for _ in range(n_missing_verdict):
        ev.append(_evidence_row(i)); sc.append(_score_row(i, exact_match=None)); i += 1
    for _ in range(n_not_ok):
        ev.append(_evidence_row(i, ok=False)); sc.append(_score_row(i, exact_match=1.0)); i += 1
    return join_window_rows(ev, sc, window="w")


def test_not_ok_rows_do_not_count_against_the_null_bound() -> None:
    # 10 rows: 4 completed and scored, 6 cut by the drain (not ok): every-row
    # null share 0.6, completed-row null share 0.0
    rows, summary = compute_window_predicate(_window(4, 6, 0), DATASET, _config(max_null_fraction=0.2), window="w")
    assert len(rows) == 10
    assert summary["n_null"] == 6 and summary["n_not_ok_nulled"] == 6
    assert summary["null_fraction"] == pytest.approx(0.6)  # still reported over every row
    assert summary["n_ok_rows"] == 4 and summary["null_fraction_ok_rows"] == 0.0


def test_missing_verdicts_among_completed_rows_still_breach_the_bound() -> None:
    # 4 completed rows, 2 without a verdict (0.5 of the completed rows), 6 cut
    with pytest.raises(PredicateError, match=r"among the completed rows 0\.500 \(2/4"):
        compute_window_predicate(_window(2, 6, 2), DATASET, _config(max_null_fraction=0.2), window="w")
    rows, summary = compute_window_predicate(_window(2, 6, 2), DATASET, _config(max_null_fraction=0.5), window="w")
    assert summary["null_fraction_ok_rows"] == pytest.approx(0.5) and summary["null_fraction"] == pytest.approx(0.8)


def test_a_window_of_only_cut_rows_has_no_completed_share() -> None:
    rows, summary = compute_window_predicate(_window(0, 5, 0), DATASET, _config(max_null_fraction=0.0), window="w")
    assert summary["n_ok_rows"] == 0 and summary["null_fraction_ok_rows"] == 0.0 and summary["null_fraction"] == 1.0


# --------------------------------------------------------------------------- #
# G31 + E12 (a): the regime gate
# --------------------------------------------------------------------------- #


def _row_key(engine: str, arm: str = "retr-rerank", family: str = "F2", r: float | None = 1.0,
             lam: float | None = 1.0) -> str:
    parts = [arm, "rerank", "none", "single", engine, "qwen3-14b", family]
    if r is not None:
        parts.append(f"r{r:g}")
    if lam is not None:
        parts.append(f"lam{lam:g}")
    return "|".join(parts)


def _write_regime(root: Path, row_key: str, dataset: str, k: int, label: str, stationarity: str | None) -> None:
    d = root / "cells" / row_key / f"window_{dataset}-{k:02d}"
    d.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {"label": label}
    if stationarity is not None:
        doc["stationarity"] = {"label": stationarity}
    (d / "regime.json").write_text(json.dumps(doc), encoding="utf-8")


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    # vllm, qasper, rung 1.0: three replicates, two qualify (one in regime but transient)
    key = _row_key("vllm")
    _write_regime(root, key, "qasper", 1, "IN_REGIME", "stationary")
    _write_regime(root, key, "qasper", 2, "IN_REGIME", "transient")
    _write_regime(root, key, "qasper", 3, "IN_REGIME", "unknown")
    # vllm, qasper, rung 0.9: below the floor, never judged
    key09 = _row_key("vllm", lam=0.9)
    _write_regime(root, key09, "qasper", 1, "UNPRESSURED", "stationary")
    # sglang, qasper, rung 1.0: one of three qualifies
    skey = _row_key("sglang")
    _write_regime(root, skey, "qasper", 1, "IN_REGIME", "stationary")
    _write_regime(root, skey, "qasper", 2, "UNPRESSURED", "stationary")
    _write_regime(root, skey, "qasper", 3, "UNKNOWN_TELEMETRY", None)
    # an F1 cell: never part of the criterion
    _write_regime(root, _row_key("vllm", arm="no-cache", family="F1", r=None, lam=None), "squad_v2", 1, "UNPRESSURED", "stationary")
    return root


def test_the_gate_counts_in_regime_and_not_transient_per_group(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    res = regime_gate.evaluate(root, 2)
    assert res["n_windows"] == 8
    groups = res["groups"]
    v = groups["vllm|retr-rerank|qasper|1.0|1.0"]
    assert v == {"n_windows": 3, "n_qualifying": 2, "n_in_regime_transient": 1, "n_in_regime_stationarity_unknown": 1}
    assert "vllm|retr-rerank|qasper|1.0|0.9" not in groups  # below the rate rung floor
    s = groups["sglang|retr-rerank|qasper|1.0|1.0"]
    assert s["n_qualifying"] == 1 and s["n_windows"] == 3
    assert res["short"] == {"sglang|retr-rerank|qasper|1.0|1.0": 1} and res["ok"] is False
    assert res["by_pair"]["vllm F2"] == {"IN_REGIME": 3, "UNPRESSURED": 1}
    # the old per-engine count (3 IN_REGIME on vllm) would have passed need=2; E12 judges the group
    assert regime_gate.evaluate(root, 1)["ok"] is True
    assert regime_gate.evaluate(root, None)["ok"] is True and regime_gate.evaluate(root, None)["short"] == {}


def test_an_engine_without_a_rung_at_the_floor_fails_when_the_criterion_is_set(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_regime(root, _row_key("vllm", lam=0.9), "qasper", 1, "IN_REGIME", "stationary")
    res = regime_gate.evaluate(root, 2)
    assert res["groups"] == {} and res["engines_without_a_rate_rung_at_or_above_floor"] == ["vllm"] and res["ok"] is False


def test_the_cli_exit_codes_and_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _tree(tmp_path)
    assert regime_gate.main(["regime_gate.py", str(root), ""]) == 0
    out = capsys.readouterr().out
    assert "regime labels per (engine, family): sglang F2: IN_REGIME=1, UNKNOWN_TELEMETRY=1, UNPRESSURED=1" in out
    assert "report only" in out
    assert regime_gate.main(["regime_gate.py", str(root), "2"]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] MIN_IN_REGIME_WINDOWS=2: group(s) below it: {'sglang|retr-rerank|qasper|1.0|1.0': 1}" in out
    assert "qualifying=2/3 (in-regime but transient=1)" in out
    assert regime_gate.main(["regime_gate.py", str(root), "1"]) == 0
    assert regime_gate.main(["regime_gate.py", str(root), "x"]) == 2
    assert regime_gate.main(["regime_gate.py", str(root), "0"]) == 2
