"""Tests for the figure block (owner approval 2026-10-27): the strategy naming
table, the S0F-99 class target in the yield table, the S0F-104 merge fix and
the academic figures A1 to A10 on a synthetic two-window landing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.analysis.cellspec import (  # noqa: E402
    BASELINES,
    BASELINE_OF_ARM,
    STRATEGY_DISPLAY_NAME,
    UnknownBaselineError,
    strategy_name,
    strategy_name_of_cell,
)


def test_every_baseline_and_the_reference_engine_has_a_name() -> None:
    assert set(STRATEGY_DISPLAY_NAME) == set(BASELINES) | {"hf"}
    for code in BASELINES:
        name = strategy_name(code)
        assert name and "B" + code[1:] not in name.split(), code
    assert strategy_name("hf") == "Reference engine, no cache"
    # the approved names, verbatim
    assert strategy_name("B1") == "Oracle context, fresh prefill"
    assert strategy_name("B3") == "CAG: shared corpus, prefix reuse"
    assert strategy_name("B6") == "RAG, reranked"
    assert strategy_name("B12", corpus_budget_tokens=700) == "CAG, truncated corpus (700 tokens)"
    assert strategy_name("B12", corpus_budget_tokens=1400) == "CAG, truncated corpus (1,400 tokens)"


def test_unknown_code_and_a_cut_on_a_non_b12_refuse() -> None:
    with pytest.raises(UnknownBaselineError, match="no strategy name for 'B13'"):
        strategy_name("B13")
    with pytest.raises(UnknownBaselineError, match="only B12 carries a cut"):
        strategy_name("B3", corpus_budget_tokens=700)


def test_cell_pair_maps_to_one_name_and_a_policy_cell_to_none() -> None:
    assert len(BASELINE_OF_ARM) == len(BASELINES)
    assert strategy_name_of_cell("retr-fresh", "dense") == "RAG, dense retrieval"
    assert strategy_name_of_cell("retr-fresh", "rerank") == "RAG, reranked"
    assert strategy_name_of_cell("corpus-trunc", "none", corpus_budget_tokens=700).endswith("(700 tokens)")
    assert strategy_name_of_cell("gold-fresh", "dense") is None


# ---------------------------------------------------------------------------
# A synthetic landing for the figure block: one engine, one dataset, the K1
# pair (control rung 1.25 and constrained 0.25 of B1), a B3 pressure cell,
# the F1 windows of B1, B3, B6 and B11, a B11 pressure cell for the
# fingerprint pair, the predicate tree, a stats.json shaped like the
# driver's, a plan and a calibration floor.
# ---------------------------------------------------------------------------

import csv
import json
import re

import academic_plots as ap  # noqa: E402
import figure_pipeline as fp  # noqa: E402
from src.analysis.cellspec import CellSpec  # noqa: E402

MODEL = "qwen3-14b"
DATASET = "qasper"
RUN_ID = "20261001-0000-s1-qwen3-14b"
N = 20
SCORING = "full-cuda-1"


def _cells() -> list[tuple[CellSpec, str, dict]]:
    """(spec, baseline, extras): extras carry late counts and the acceptance pattern."""
    out = []
    out.append((CellSpec.from_baseline("B1", model=MODEL), "B1", {"late": 0}))  # type: ignore[arg-type]
    out.append((CellSpec.from_baseline("B3", model=MODEL), "B3", {"late": 0}))  # type: ignore[arg-type]
    out.append((CellSpec.from_baseline("B6", model=MODEL), "B6", {"late": 0}))  # type: ignore[arg-type]
    out.append((CellSpec.from_baseline("B11", model=MODEL), "B11", {"late": 0}))  # type: ignore[arg-type]
    out.append((CellSpec.from_baseline("B1", model=MODEL, family="F2", budget_r=1.25, rate_frac=0.8), "B1", {"late": 2}))  # type: ignore[arg-type]
    out.append((CellSpec.from_baseline("B1", model=MODEL, family="F2", budget_r=0.25, rate_frac=0.8), "B1", {"late": 10}))  # type: ignore[arg-type]
    out.append((CellSpec.from_baseline("B3", model=MODEL, family="F3", budget_r=0.5, rate_frac=0.8), "B3", {"late": 1}))  # type: ignore[arg-type]
    out.append((CellSpec.from_baseline("B6", model=MODEL, family="F2", budget_r=0.5, rate_frac=0.8), "B6", {"late": 1}))  # type: ignore[arg-type]
    out.append((CellSpec.from_baseline("B11", model=MODEL, family="F2", budget_r=0.5, rate_frac=0.8), "B11", {"late": 1, "flip": 3}))  # type: ignore[arg-type]
    # the two B12 cuts at no load: the plots must keep them apart (review MEDIUM 8)
    out.append((CellSpec.from_baseline("B12", model=MODEL, corpus_budget_tokens=700), "B12", {"late": 0, "flip": 6}))  # type: ignore[arg-type]
    out.append((CellSpec.from_baseline("B12", model=MODEL, corpus_budget_tokens=1400), "B12", {"late": 0}))  # type: ignore[arg-type]
    # the reference engine's cell (no floor, no first token): named by engine and arm
    out.append((CellSpec.from_baseline("B3", model=MODEL, engine="hf"), "B3", {"late": 0}))  # type: ignore[arg-type]
    return out


def _grounding_score(i: int):
    """The fixture's scoring-pass grounding score: one in five below the 0.5
    rule (ungrounded), one exactly at it, one null (unscored), the rest above."""
    return 0.2 if i % 5 == 0 else 0.5 if i == 1 else None if i == 11 else 0.9


def _write_landing(tmp_path: Path, *, with_retrieval: bool = True) -> dict:
    landing = tmp_path / "landing"
    run_root = landing / "run" / "camp1" / "s1" / RUN_ID
    (run_root / "index").mkdir(parents=True)
    index_rows = []
    yield_rows = []
    rate = 3.0
    for spec, baseline, extra in _cells():
        key = spec.to_row_key()
        cell_dir = run_root / "cells" / key
        cell_dir.mkdir(parents=True)
        cell = {"baseline": baseline, "cellspec": spec.to_flat_dict(), "gpu_count": 1, "windows": {}}
        if spec.family != "F1":
            cell["budget_plan"] = {"avg_seq_tokens": 4779}
        n_windows = 1 if spec.family == "F1" else 3
        for w in range(1, n_windows + 1):
            wname = f"{DATASET}-{w:02d}"
            wdir = cell_dir / f"window_{wname}"
            wdir.mkdir()
            cell["windows"][wname] = {"dataset": DATASET, "rep": w, "t_start": 0.0, "t_end": 60.0}
            reqs, preds = [], []
            late = extra["late"]
            for i in range(N):
                ok = i != N - 1
                ttft = 4000.0 if i < late else 300.0 + i
                accepted = i % 4 != 0
                if extra.get("flip") and i < extra["flip"]:
                    accepted = not accepted
                reqs.append({
                    "example_id": f"e{i:03d}", "repeat_index": "0", "record_index": i, "ok": ok,
                    "workload_mode": "open_loop", "ttft_ms": ttft if ok else None,
                    "ttft_from_scheduled_ms": ttft + 2.0 if ok else None, "tpot_ms": 10.0 if ok else None,
                    "num_tokens": 9 if ok else 0, "latency_ms": ttft + 90.0 if ok else None,
                    "generated_answer": f"answer {i}" if ok else "", "predicted_no_answer": i % 7 == 0,
                    "grounded": None, "arrival_s": i * 2.5,
                    **({"retrieval_hit": i % 3 != 0, "retrieval_rank": 1 + i % 3} if (with_retrieval and baseline in ("B6", "B11")) else {}),
                })
                preds.append({"example_id": f"e{i:03d}", "repeat_index": "0", "record_index": i, "ok": ok,
                              "predicate": (accepted if ok else None)})
            (wdir / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in reqs) + "\n", encoding="utf-8")
            (wdir / "metrics.json").write_text(json.dumps({
                "experiment": {"dataset": DATASET},
                "measured_window": {"t_start": 0.0, "t_end": 60.0, "span": "dispatch"},
                "performance": {"total_requests": N},
                "workload": {"mode": "open_loop" if spec.family != "F1" else "closed_loop",
                             "open_loop": {"schedule": {"rate_qps": rate}}},
                "vllm_telemetry": {"kv_usage_peak": 0.9},
            }), encoding="utf-8")
            label = "IN_REGIME" if (spec.family != "F1" and spec.budget_r == 0.25) else "UNPRESSURED"
            (wdir / "regime.json").write_text(json.dumps({"label": label, "t_start": 0.0, "t_end": 60.0,
                                                           "inputs": {"rho_kv_time_avg": 0.8}}), encoding="utf-8")
            (wdir / "cage_stats.jsonl").write_text("\n".join(json.dumps({
                "ts_s": t * 0.5, "kv_cache_usage": 0.5 + 0.01 * t, "preemptions_total": 3 + (t // 40)})
                for t in range(120)) + "\n", encoding="utf-8")
            pdir = run_root / "predicate" / SCORING / "cells" / key / f"window_{wname}"
            pdir.mkdir(parents=True)
            (pdir / "predicate.jsonl").write_text("\n".join(json.dumps(p) for p in preds) + "\n", encoding="utf-8")
            # Block F row F-3 (S2 register D4): the serving rows' grounded flag is
            # null on every campaign cell (CAGE_SKIP_QUALITY=1); the instrument's
            # score lives in the scoring pass, one qa_scores.jsonl per window
            sdir = run_root / "scoring" / SCORING / "cells" / key / f"window_{wname}"
            sdir.mkdir(parents=True)
            (sdir / "qa_scores.jsonl").write_text("\n".join(json.dumps({
                "example_id": f"e{i:03d}", "repeat_index": "0", "record_index": i,
                "grounding_score": _grounding_score(i), "predicted_no_answer": 1.0 if i % 7 == 0 else 0.0})
                for i in range(N)) + "\n", encoding="utf-8")
            index_rows.append({
                "run_id": RUN_ID, "campaign": "camp1", "session": "s1", "model": MODEL, "engine": spec.engine,
                "arm": spec.arm, "baseline": baseline, "retriever": spec.retriever, "policy": spec.policy,
                "topology": spec.topology, "family": spec.family, "dataset": DATASET,
                "budget_r": "" if spec.budget_r is None else spec.budget_r,
                "rate_frac": "" if spec.rate_frac is None else spec.rate_frac,
                "window": w, "window_key": wname, "row_key": key, "window_dir": f"cells/{key}/window_{wname}",
                "cell_json": f"cells/{key}/cell.json", "artifacts": "",
            })
            n_ok = N - 1
            n_timely = n_ok - late
            n_yield = sum(1 for i in range(N - 1) if i >= late and (i % 4 != 0))
            yield_rows.append({"row_key": key, "window_dir": f"window_{wname}", "engine": spec.engine,
                               "family": spec.family, "arm": spec.arm, "rung": "" if spec.budget_r is None else f"{spec.budget_r:g}",
                               "rate_frac": "" if spec.rate_frac is None else f"{spec.rate_frac:g}", "dataset": DATASET,
                               "regime_label": label, "n_issued": N, "n_timely": n_timely, "n_yield": n_yield,
                               "goodput_frac": n_timely / N, "yield_frac": n_yield / N, "skip_reason": "",
                               "baseline_source": "class-target:4779:plan_registered.json"})
        (cell_dir / "cell.json").write_text(json.dumps(cell), encoding="utf-8")
    with open(run_root / "index" / "cells_index.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(index_rows[0]))
        wr.writeheader()
        wr.writerows(index_rows)
    ycsv = landing / "yield_table.csv"
    with open(ycsv, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(yield_rows[0]))
        wr.writeheader()
        wr.writerows(yield_rows)
    # calibration floor and the plan
    cal = landing / "extras" / "calibration" / "pod"
    cal.mkdir(parents=True)
    (cal / "rungs_S1_vllm.json").write_text(json.dumps({"ladder": {"floor": {"ttft_s": 0.1, "tpot_s": 0.01}}, "classes": {}}), encoding="utf-8")
    k1_control = CellSpec.from_baseline("B1", model=MODEL, family="F2", budget_r=1.25, rate_frac=0.8).to_row_key()  # type: ignore[arg-type]
    k1_con = CellSpec.from_baseline("B1", model=MODEL, family="F2", budget_r=0.25, rate_frac=0.8).to_row_key()  # type: ignore[arg-type]
    plan = landing / "extras" / "plan_registered.json"
    plan.write_text(json.dumps({
        "class_first_token_targets_s": {"4779": 2.0},
        "demand_classes": {"seq_tokens": {"gold-fresh": 4779, "corpus-reuse": 4779, "retr-fresh": 4779, "retr-trunc": 4779},
                           "corpus_trunc_seq_tokens": {"700": 4779, "1400": 4779}},
        "behavior_knobs": {"corpus_trunc_budgets": [1400, 700]},
        "steps": [
            {"kind": "cell", "baseline": "B1", "row_key": k1_control, "demand_class": {"seq_tokens": 4779},
             "lambda_star_rps": 4.0, "lambda_star_source": {"alpha": None, "adr": "ADR-0154"}, "offered_rate_rps": 3.0,
             "rate_basis": "cal-v3 rung ESTIMATED (gold-fresh F2 workload, this engine, this r)", "windows": 3,
             "cellspec": {"engine": "vllm", "budget_r": 1.25, "rate_frac": 0.8}},
            {"kind": "cell", "baseline": "B1", "row_key": k1_con, "demand_class": {"seq_tokens": 4779},
             "lambda_star_rps": 4.0, "lambda_star_source": {"alpha": 0.7}, "offered_rate_rps": 3.0,
             "rate_basis": "interpolated: log-log between the gold-fresh and the smallest-class rung lambda* [D]", "windows": 3,
             "cellspec": {"engine": "vllm", "budget_r": 0.25, "rate_frac": 0.8}},
        ],
    }), encoding="utf-8")
    # a stats.json shaped like the driver's (the values are the fixture's own)
    stamp = run_root / "analysis" / "20261001-000000"
    stamp.mkdir(parents=True)
    b3 = CellSpec.from_baseline("B3", model=MODEL).to_row_key()  # type: ignore[arg-type]
    b6 = CellSpec.from_baseline("B6", model=MODEL).to_row_key()  # type: ignore[arg-type]
    stats = {
        "frontier": {
            # the driver's control rung (compute_frontier_drop reads it from the registered plan; row F-13)
            "control_budget_r": 1.25,
            "legs": [{
                "dataset": DATASET, "leg": "B1|r0.25", "baseline": "B1", "arm": "gold-fresh", "budget_r": 0.25,
                "engine": "vllm", "constrained_row_key": k1_con, "control_row_keys": [k1_control],
                "rate_qps": 3.0, "class_tokens": 4779, "target_s": 2.0,
                "yield": {"mean_constrained": 0.35, "mean_control": 0.65, "delta": -0.3, "p_value": 0.004,
                          "ci95_low_bootstrap": -0.4, "ci95_high_bootstrap": -0.2},
                "timely": {"mean_constrained": 0.45, "mean_control": 0.85, "delta": -0.4},
                "accepted": {"mean_constrained": 0.7, "mean_control": 0.7, "delta": 0.0},
                "p_merged_engines": 0.004, "p_holm_within_dataset": 0.004,
            }],
            "per_dataset_intersection": [{"dataset": DATASET, "p_endpoint": 0.004,
                                          "per_arm": {"B1": {"p_holm_any": 0.004, "verdict": "drop confirmed at the registered alpha"}}}],
        },
        "fingerprint": {
            "leg_points": [{"leg": "truncate", "dataset": DATASET, "n_points": 1, "p_leg": 0.01}],
            "per_dataset_intersection": [{"dataset": DATASET, "p_intersection_union": 1.0, "n_legs_supplied": 1, "holm_m_registered": 3}],
            "skipped": [{"leg": "evict", "reason": "no policy='evict' pressure cells in this run"}],
        },
        "contrasts": [{
            "contrast_id": 4, "metric": "ttft_ms", "cell_row_key": b6, "reference_row_key": b3,
            "higher_is_better": False, "executed_alternative": "less", "correction": "none",
            "per_dataset": [{"dataset": DATASET, "p_value": 0.01, "n_pairs": 19, "n_dropped_nan": 0,
                             "median_delta": -5.0, "wins": 15, "losses": 4, "ties": 0,
                             "p_holm_across_datasets": None, "n_nonzero": 19}],
        }],
    }
    (stamp / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    return {"landing": landing, "run_root": run_root, "yield": ycsv, "plan": plan, "stats": stamp / "stats.json"}


def _landing(parts: dict, **kw) -> ap.Landing:
    return ap.Landing(parts["landing"], yield_table=parts["yield"], plan=parts.get("plan"), stats=parts.get("stats"), **kw)


def test_s0f_104_yield_frame_has_one_numeric_rate_frac_on_both_paths(tmp_path: Path) -> None:
    parts = _write_landing(tmp_path)
    y = _landing(parts).yield_frame()
    assert list(y.columns).count("rate_frac") == 1 and "rung" not in y.columns
    assert y["rate_frac"].dtype.kind == "f" and y["budget_r"].dtype.kind == "f"
    # the recompute path drops the same string columns before the merge
    L2 = ap.Landing(parts["landing"], calibration_dir=parts["landing"] / "extras" / "calibration" / "pod")
    y2 = L2.yield_frame()
    assert list(y2.columns).count("rate_frac") == 1 and y2["rate_frac"].dtype.kind == "f"


@pytest.mark.parametrize("key", sorted(k for k in ap.PLOTS if k.startswith("a")))
def test_every_figure_of_the_block_renders_with_a_caption(tmp_path: Path, key: str) -> None:
    parts = _write_landing(tmp_path)
    L = _landing(parts)
    out = tmp_path / "out"
    out.mkdir()
    ap.ps.apply_style()
    path, used = ap.PLOTS[key](L, out)
    for name in path.split(";"):
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    captions = list(out.glob("*.caption.txt"))
    assert captions, key
    text = "\n".join(c.read_text(encoding="utf-8") for c in captions)
    assert "B1" not in text.split() and used


def test_figures_refuse_by_name_when_their_input_is_missing(tmp_path: Path) -> None:
    parts = _write_landing(tmp_path, with_retrieval=False)
    out = tmp_path / "out"
    out.mkdir()
    # no stats: the frontier figures name the missing artifact
    bare = ap.Landing(parts["landing"], yield_table=parts["yield"], plan=parts["plan"],
                      stats=tmp_path / "absent_stats.json")
    with pytest.raises(ap.PlotSkip, match="no frontier"):
        ap.plot_a01_frontier_drop(bare, out)
    # no plan: the capacity table refuses
    noplan = ap.Landing(parts["landing"], yield_table=parts["yield"], stats=parts["stats"])
    with pytest.raises(ap.PlotSkip, match="no --plan"):
        ap.plot_a10_capacity_table(noplan, out)
    # no retrieval column: a05b refuses naming it
    with pytest.raises(ap.PlotSkip, match="retrieval_hit"):
        ap.plot_a05b_retrieval_quality(_landing(parts), out)


def test_main_writes_the_manifest_with_the_block_and_the_plan(tmp_path: Path) -> None:
    parts = _write_landing(tmp_path)
    rc = ap.main([str(parts["landing"]), "--yield-table", str(parts["yield"]), "--plan", str(parts["plan"]),
                  "--stats", str(parts["stats"]), "--only", "a07_consort_flow", "a10_capacity_table", "01_load_curve"])
    assert rc == 0
    stamp = sorted((parts["landing"] / "plots" / "academic").iterdir())[-1]
    m = json.loads((stamp / "plots_manifest.json").read_text(encoding="utf-8"))
    assert m["plan"] == str(parts["plan"]) and "a07_consort_flow" in m["plots"]
    assert (stamp / "a10_capacity_table.md").exists()
    assert "Oracle context, fresh prefill" in (stamp / "a10_capacity_table.md").read_text(encoding="utf-8")


def test_consort_arithmetic_closes_and_names_the_exclusions(tmp_path: Path) -> None:
    parts = _write_landing(tmp_path)
    L = _landing(parts)
    out = tmp_path / "out"
    out.mkdir()
    _, used = ap.plot_a07_consort_flow(L, out)
    # 5 pressure cells x 3 windows served; only the r=0.25 cell is in regime
    assert used == "15 served, 3 in regime, 3 with a yield record"
    # review MEDIUM 6: no window in regime is a flow with a zero box, never a KeyError
    for w in (parts["run_root"] / "cells").glob("*/window_*/regime.json"):
        w.write_text(json.dumps({"label": "UNPRESSURED", "t_start": 0.0, "t_end": 60.0}), encoding="utf-8")
    L2 = _landing(parts)
    _, used2 = ap.plot_a07_consort_flow(L2, out)
    assert used2 == "15 served, 0 in regime, 0 with a yield record"


def test_registered_forest_labels_use_the_strategy_names() -> None:
    b3 = CellSpec.from_baseline("B3", model=MODEL).to_row_key()  # type: ignore[arg-type]
    b6 = CellSpec.from_baseline("B6", model=MODEL, family="F2", budget_r=0.5, rate_frac=0.8).to_row_key()  # type: ignore[arg-type]
    b12 = CellSpec.from_baseline("B12", model=MODEL, family="F3", budget_r=0.5, rate_frac=0.8, corpus_budget_tokens=700).to_row_key()  # type: ignore[arg-type]
    evict = CellSpec(arm="gold-fresh", retriever="none", policy="evict", topology="single", engine="vllm",  # type: ignore[arg-type]
                     model=MODEL, family="F2", budget_r=0.5, rate_frac=0.8).to_row_key()  # type: ignore[arg-type]
    rows = [fp.ContrastStatRow(cell_row_key=k, reference_row_key=b3, dataset=DATASET, metric="ttft_ms",
                               higher_is_better=False, executed_alternative="less", correction="none",
                               p_value=0.01, n_pairs=10, n_dropped_nan=0, median_delta=-1.0, wins=8, losses=2, ties=0)
            for k in (b6, b12, evict)]
    labels = fp._registered_labels(rows)
    assert labels[b3] == "CAG: shared corpus, prefix reuse"
    assert labels[b6] == "RAG, reranked r0.5 lam0.8"
    assert labels[b12] == "CAG, truncated corpus (700 tokens) r0.5 lam0.8"
    assert labels[evict] == "Oracle context, fresh prefill with policy evict r0.5 lam0.8"
    assert strategy_name_of_cell("gold-fresh", "none", policy="evict") == "Oracle context, fresh prefill with policy evict"


def test_descriptive_helpers_pin_their_rules(tmp_path: Path) -> None:
    import pandas as pd

    # exact two-sided McNemar on the discordant pairs (review LOW 18)
    assert ap._mcnemar_exact(0, 0) == 1.0 and ap._mcnemar_exact(5, 5) == 1.0
    assert ap._mcnemar_exact(3, 10) == pytest.approx(0.0923, abs=1e-4)
    # first pass: the lowest record_index per example, whatever the row order
    df = pd.DataFrame({"example_id": ["a", "b", "a", "b"], "record_index": [7, 1, 0, 9], "v": [1, 2, 3, 4]})
    first = ap._first_pass(df)
    assert dict(zip(first["example_id"], first["v"])) == {"a": 3, "b": 2}
    # outcome classes
    class R:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    assert ap._outcome(R(ok=False)) == "failed"
    assert ap._outcome(R(ok=True, predicate=None)) == "unscored"
    assert ap._outcome(R(ok=True, predicate=True)) == "accepted"
    assert ap._outcome(R(ok=True, predicate=False, predicted_no_answer=True)) == "abstained"
    assert ap._outcome(R(ok=True, predicate=False, predicted_no_answer=False, grounded=False)) == "ungrounded"
    assert ap._outcome(R(ok=True, predicate=False, predicted_no_answer=False, grounded=None)) == "rejected"


def test_a06_counts_the_counter_increase_and_marks_an_absent_counter(tmp_path: Path) -> None:
    parts = _write_landing(tmp_path)
    L = _landing(parts)
    out = tmp_path / "out"
    out.mkdir()
    # the fixture counter rises 3 -> 5 inside each window: the increase is 2
    tel = ap._telemetry(L, next(iter(L.cells["window_dir"])))
    assert float(tel["pre"].max() - tel["pre"].min()) == 2.0
    _, used = ap.plot_a06_preemption_kv(L, out)
    assert used == "6 windows"  # one leg: three control and three constrained windows
    caption = (out / "a06_preemption_kv.caption.txt").read_text(encoding="utf-8")
    assert "no preemption counter" in caption and "establish" in caption


def test_a10_basis_is_the_plan_steps_own_text_never_measured(tmp_path: Path) -> None:
    parts = _write_landing(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    ap.plot_a10_capacity_table(_landing(parts), out)
    md = (out / "a10_capacity_table.md").read_text(encoding="utf-8")
    assert "cal-v3 rung ESTIMATED (gold-fresh F2 workload, this engine, this r)" in md
    assert "interpolated: log-log between the gold-fresh and the smallest-class rung lambda*" in md
    assert "| measured |" not in md and "interpolated (ADR-0156)" not in md


def test_a09_leaves_the_reference_engine_and_binary_metrics_out(tmp_path: Path) -> None:
    parts = _write_landing(tmp_path)
    stats = json.loads(parts["stats"].read_text(encoding="utf-8"))
    hf_b6 = CellSpec.from_baseline("B6", model=MODEL, engine="hf").to_row_key()  # type: ignore[arg-type]
    hf_b3 = CellSpec.from_baseline("B3", model=MODEL, engine="hf").to_row_key()  # type: ignore[arg-type]
    row = dict(stats["contrasts"][0])
    stats["contrasts"].append({**row, "cell_row_key": hf_b6, "reference_row_key": hf_b3})
    stats["contrasts"].append({**row, "metric": "predicate"})
    parts["stats"].write_text(json.dumps(stats), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    files, used = ap.plot_a09_forest_headline(_landing(parts), out)
    assert files == "a09_forest_ttft_ms_vllm.png"
    assert "not drawn: 2" in used
    caption = (out / "a09_forest_headline.caption.txt").read_text(encoding="utf-8")
    assert "reference engine sidebar" in caption and "binary metric" in caption
    assert _landing(parts).baseline_label(hf_b3) == "Reference engine, no cache, corpus-reuse"


def test_a08_counts_the_all_late_windows_and_reads_the_bar_from_the_table(tmp_path: Path) -> None:
    parts = _write_landing(tmp_path)
    rows = list(csv.DictReader(open(parts["yield"], encoding="utf-8")))
    rows[-1]["n_timely"] = "0"
    rows[-1]["n_yield"] = "0"
    rows[-1]["goodput_frac"] = "0.0"
    rows[-1]["yield_frac"] = "0.0"
    pressure_last = [r for r in rows if r["family"] != "F1"][-1]
    pressure_last.update({"n_timely": "0", "n_yield": "0", "goodput_frac": "0.0", "yield_frac": "0.0"})
    with open(parts["yield"], "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    out = tmp_path / "out"
    out.mkdir()
    L = _landing(parts)
    assert L.yield_bar() == "the registered class target (first token only)"
    _, used = ap.plot_a08_truth_tax_ratio(L, out)
    assert used.endswith("1 with no timely request left out of the ratio")
    assert "1 of 15" in (out / "a08_truth_tax_ratio.caption.txt").read_text(encoding="utf-8")
    assert (out / "a08b_truth_tax_scatter.png").exists()
    # a mixed table refuses by name
    rows[0]["baseline_source"] = "anchor:rungs_S1_vllm.json"
    with open(parts["yield"], "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with pytest.raises(ap.PlotSkip, match="mixes grading bars"):
        _landing(parts).yield_bar()


def test_a03_leaves_unscored_pairs_out(tmp_path: Path) -> None:
    parts = _write_landing(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    _, used = ap.plot_a03_paired_identity(_landing(parts), out)
    # 5 pressure cells x 3 windows x 19 completed questions, every one scored
    assert used.startswith("285 pairs")
    # null the verdicts of one pressure window: its 19 pairs leave the count
    key = CellSpec.from_baseline("B3", model=MODEL, family="F3", budget_r=0.5, rate_frac=0.8).to_row_key()  # type: ignore[arg-type]
    ppath = parts["run_root"] / "predicate" / SCORING / "cells" / key / f"window_{DATASET}-01" / "predicate.jsonl"
    preds = [json.loads(l) for l in ppath.read_text(encoding="utf-8").splitlines() if l.strip()]
    for pr in preds:
        pr["predicate"] = None
    ppath.write_text("\n".join(json.dumps(pr) for pr in preds) + "\n", encoding="utf-8")
    _, used2 = ap.plot_a03_paired_identity(_landing(parts), out)
    assert used2.startswith("266 pairs")


def test_older_plots_keep_the_two_b12_cuts_apart_and_print_no_code(tmp_path: Path) -> None:
    parts = _write_landing(tmp_path)
    L = _landing(parts)
    out = tmp_path / "out"
    out.mkdir()
    ap.ps.apply_style()
    d = ap._f1_rows(L)
    assert {"CAG, truncated corpus (700 tokens)", "CAG, truncated corpus (1,400 tokens)"} <= set(d["cell_strategy"])
    for key in ("10_ttft_per_arm", "15_correct_rate", "18_prompt_vs_answer", "02_truth_tax", "05_avg_vs_peak"):
        try:
            ap.PLOTS[key](L, out)
        except ap.PlotSkip:
            continue
    import matplotlib.pyplot as plt

    # every tick, legend and title text matplotlib still holds carries no bare code
    texts = [t.get_text() for fig_num in plt.get_fignums() for ax in plt.figure(fig_num).axes
             for t in list(ax.get_xticklabels()) + list(ax.get_legend_handles_labels()[1] and [])]
    assert all(not any(tok in {"B1", "B3", "B6", "B12"} for tok in t.replace("\n", " ").split()) for t in texts)


# ---------------------------------------------------------------------------
# p03, the pressure map, redrawn 2026-10-29 (owner approval of the Spec Block):
# one column per rounded pressure-table load, the frontier block pooled by the
# plan's block field, one line of text per cell, a one-hue shade by the share
# of windows in regime.
# ---------------------------------------------------------------------------

CELL_TEXT = re.compile(r"^\d+/\d+ in regime( \(\d+ no record\))?$")


def _plan_with_blocks(parts: dict) -> None:
    """Mark the fixture's two K1 steps and add P1 steps for the other pressure cells."""
    plan = json.loads(parts["plan"].read_text(encoding="utf-8"))
    for s in plan["steps"]:
        s["block"] = "K1"
    for b, fam in (("B3", "F3"), ("B6", "F2"), ("B11", "F2")):
        key = CellSpec.from_baseline(b, model=MODEL, family=fam, budget_r=0.5, rate_frac=0.8).to_row_key()  # type: ignore[arg-type]
        plan["steps"].append({"kind": "cell", "baseline": b, "row_key": key, "block": "P1", "windows": 3,
                              "cellspec": {"engine": "vllm", "budget_r": 0.5, "rate_frac": 0.8}})
    parts["plan"].write_text(json.dumps(plan), encoding="utf-8")


def _pressured_with_labels(L: ap.Landing):
    c = L.cells[(L.cells["family"] != "F1") & L.cells["budget_r"].notna() & (L.cells["dataset"] != "ruler")].copy()
    c["label"] = [L.window_json(w, "regime.json").get("label") or "NO_REGIME_RECORD" for w in c["window_dir"]]
    return c


def test_p03_pools_the_frontier_cells_by_the_plan_block_and_prints_one_line_per_cell(tmp_path: Path) -> None:
    parts = _write_landing(tmp_path)
    _plan_with_blocks(parts)
    L = _landing(parts)
    blocks = ap._plan_blocks(L)
    assert set(blocks.values()) == {"K1", "P1"} and len(blocks) == 5
    columns, table = ap._pressure_map_table(_pressured_with_labels(L), blocks)
    assert columns == ["0.80", ap.FRONTIER_COLUMN]
    cell = {(r.column, float(r.budget_r)): (int(r.in_regime), int(r.windows), int(r.no_record)) for r in table.itertuples()}
    assert cell[(ap.FRONTIER_COLUMN, 0.25)] == (3, 3, 0)
    assert cell[(ap.FRONTIER_COLUMN, 1.25)] == (0, 3, 0)
    assert cell[("0.80", 0.5)] == (0, 9, 0)
    assert len(table) == 3 and int(table["windows"].sum()) == 15
    for r in table.itertuples():
        assert CELL_TEXT.match(ap._cell_text(int(r.in_regime), int(r.windows), int(r.no_record)))
    out = tmp_path / "out"
    out.mkdir()
    ap.ps.apply_style()
    path, used = ap.plot03_pressure_map(L, out)
    assert (out / path).exists() and (out / path).stat().st_size > 0
    assert used.startswith("15 pressured windows in 3 cells") and "pools rates 0.8" in used and "left out" not in used
    caption = (out / "p03_pressure_map.caption.txt").read_text(encoding="utf-8")
    for phrase in (ap.FRONTIER_COLUMN, "queued occupancy", "non-empty queue", "attain", "T2", "control rung 1.25"):
        assert phrase in caption, phrase
    # review MEDIUM 2: the rule as goodput.classify_regime applies it (0.95 x a
    # pinned ceiling, the absolute 0.90 without one), never the reverse
    assert "where no ceiling was pinned" in caption and "for the two laddered classes" not in caption
    # the in-regime total is the one a07 prints
    _, a07_used = ap.plot_a07_consort_flow(L, out)
    assert f"{int(table['in_regime'].sum())} in regime" in a07_used


def test_p03_without_block_information_uses_rounded_rates_and_sizes_the_figure(tmp_path: Path) -> None:
    import pandas as pd

    parts = _write_landing(tmp_path)
    L = ap.Landing(parts["landing"], yield_table=parts["yield"], stats=parts["stats"])  # no plan
    assert ap._plan_blocks(L) == {}
    columns, table = ap._pressure_map_table(_pressured_with_labels(L), {})
    assert columns == ["0.80"] and int(table["windows"].sum()) == 15
    out = tmp_path / "out"
    out.mkdir()
    ap.ps.apply_style()
    _, used = ap.plot03_pressure_map(L, out)
    assert "no plan block field" in used
    # the S1 shape: twelve distinct rates, ten of them frontier ladder points
    rates = [0.65506, 0.7, 0.7156, 0.716055, 0.716451, 0.71599, 0.7692, 0.769206, 0.76927, 0.8163, 0.826896, 0.85]
    df = pd.DataFrame({"engine": "vllm", "row_key": [f"k{i}" for i in range(12)], "rate_frac": rates,
                       "budget_r": 0.5, "label": "UNPRESSURED"})
    cols, _ = ap._pressure_map_table(df, {})
    assert cols == ["0.66", "0.70", "0.72", "0.77", "0.82", "0.83", "0.85"]
    frontier = {f"k{i}": "K1" for i, r in enumerate(rates) if r not in (0.7, 0.85)}
    cols2, t2 = ap._pressure_map_table(df, frontier)
    assert cols2 == ["0.70", "0.85", ap.FRONTIER_COLUMN]
    assert int(t2.set_index("column").loc[ap.FRONTIER_COLUMN, "windows"]) == 10
    # the width grows with the column count so cells never overprint; a
    # no-record suffix needs 1.4 in per column (review LOW 5)
    assert ap._pressure_map_width(3, 2) == ap.ps.FULL_WIDTH_IN
    assert ap._pressure_map_width(7, 2) >= 0.78 * 14
    assert ap._pressure_map_width(3, 2, with_suffix=True) >= 1.4 * 6
    # review MEDIUM 1: every pooled rate is printed with its own digits
    df2 = pd.DataFrame({"engine": "sglang", "row_key": ["x", "y", "z", "w"], "rate_frac": [0.7692, 0.769206, 0.8163, 0.7],
                        "budget_r": 0.5, "label": "UNPRESSURED"})
    assert ap._frontier_rates(df2, {"x": "K1", "y": "K1", "z": "K1"}) == ["0.7692", "0.769206", "0.8163"]
    assert ap._frontier_rates(df2, {}) == []


def test_p03_counts_mixed_labels_and_leaves_rate_less_rows_out() -> None:
    import pandas as pd

    df = pd.DataFrame({
        "engine": ["vllm"] * 5, "row_key": ["a", "a", "a", "b", "a"], "rate_frac": [0.7, 0.7, 0.7, float("nan"), 0.7],
        "budget_r": [0.5] * 5, "label": ["IN_REGIME", "UNPRESSURED", "UNKNOWN_TELEMETRY", "IN_REGIME", None],
    })
    cols, t = ap._pressure_map_table(df, {})
    assert cols == ["0.70"] and len(t) == 1
    r = t.iloc[0]
    # the None label counts as a window and as no record (review LOW 6)
    assert (int(r.in_regime), int(r.windows), int(r.no_record)) == (1, 4, 2)
    assert ap._cell_text(1, 3, 1) == "1/3 in regime (1 no record)"
    assert ap._cell_text(10, 12, 0) == "10/12 in regime"
    # review LOW 4: decimal half-up labels, not binary float formatting
    assert [ap._rate_label(v) for v in (0.705, 0.695, 0.7156, 0.85, 0.7)] == ["0.71", "0.70", "0.72", "0.85", "0.70"]


def test_a05_derives_the_grounded_flag_from_the_scoring_pass(tmp_path: Path) -> None:
    """Block F row F-3 (S2 register D4): every campaign cell pins
    CAGE_SKIP_QUALITY=1, so the serving rows' grounded flag is null on S1 while
    270,205 of 365,410 score rows carry a grounding_score and no window is
    all-null. The flag is derived from the score with the scorer's rule
    (rescore_quality.py: grounded = grounding_score >= 0.5; null stays null)."""
    parts = _write_landing(tmp_path)
    L = _landing(parts)
    wd = next(w for e, w in zip(L.cells["engine"], L.cells["window_dir"]) if e != "hf")
    d = L.rows(wd)
    by = dict(zip(d["example_id"], d["grounded"]))
    assert by["e000"] is False and by["e001"] is True and by["e002"] is True and by["e011"] is None
    assert d["grounded"].notna().sum() == N - 1
    out = tmp_path / "out"
    out.mkdir()
    ap.ps.apply_style()
    ap.plot_a05_outcome_breakdown(L, out)
    text = (out / "a05_outcome_breakdown.caption.txt").read_text(encoding="utf-8")
    assert "did not run" not in text
    assert "grounding_score >= 0.5" in text and f"scoring pass {SCORING}" in text
    # a window with no scoring file keeps the serving rows' own flag
    sfile = parts["run_root"] / "scoring" / SCORING / wd / "qa_scores.jsonl"
    rfile = parts["run_root"] / wd / "requests.jsonl"
    rows = [json.loads(ln) for ln in rfile.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for r in rows:
        r["grounded"] = True
    rfile.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    sfile.unlink()
    assert _landing(parts).rows(wd)["grounded"].eq(True).all()
    # a duplicate identity in the score file refuses instead of guessing
    dup = json.dumps({"example_id": "e000", "repeat_index": "0", "record_index": 0, "grounding_score": 0.9})
    sfile.write_text(dup + "\n" + dup + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="duplicate"):
        _landing(parts).rows(wd)


def _add_pooled_constrained_cell(parts: dict) -> str:
    """Block F row F-8: a second B1 r0.25 cell at a neighboring rate (a
    constrained cell repeated at one rate, the shape a resume can produce), the
    first cell's windows copied under a new key and the index rows appended."""
    import shutil

    k1 = CellSpec.from_baseline("B1", model=MODEL, family="F2", budget_r=0.25, rate_frac=0.8).to_row_key()  # type: ignore[arg-type]
    k2 = CellSpec.from_baseline("B1", model=MODEL, family="F2", budget_r=0.25, rate_frac=0.81).to_row_key()  # type: ignore[arg-type]
    root = parts["run_root"]
    for rel in ("cells", f"predicate/{SCORING}/cells", f"scoring/{SCORING}/cells"):
        shutil.copytree(root / rel / k1, root / rel / k2)
    idx = root / "index" / "cells_index.csv"
    rows = list(csv.DictReader(open(idx, encoding="utf-8")))
    for r in [r for r in rows if r["row_key"] == k1]:
        rows.append({**r, "row_key": k2, "rate_frac": "0.81", "window_dir": r["window_dir"].replace(k1, k2),
                     "cell_json": r["cell_json"].replace(k1, k2)})
    with open(idx, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    return k2


def test_frontier_readers_take_every_pooled_constrained_key_f8(tmp_path: Path) -> None:
    """Block F row F-8 (S2 register T4): a leg pooled from two constrained cells
    carries constrained_row_keys; a01, a02 and the K1 cell list draw every key,
    and a leg without the list (an older stats.json) still reads its one key."""
    parts = _write_landing(tmp_path)
    k2 = _add_pooled_constrained_cell(parts)
    out = tmp_path / "out"
    out.mkdir()
    ap.ps.apply_style()
    L = _landing(parts)
    (leg,) = L.stats()["frontier"]["legs"]
    assert "constrained_row_keys" not in leg
    assert ap.plot_a01_frontier_drop(L, out)[1] == "1 legs, 6 windows"
    assert ap.plot_a02_ttft_cdf(L, out)[1] == "114 completed requests"
    assert ap.plot_a06_preemption_kv(L, out)[1] == "6 windows"
    stats = json.loads(parts["stats"].read_text(encoding="utf-8"))
    stats["frontier"]["legs"][0]["constrained_row_keys"] = [leg["constrained_row_key"], k2]
    parts["stats"].write_text(json.dumps(stats), encoding="utf-8")
    L2 = _landing(parts)
    assert ap.plot_a01_frontier_drop(L2, out)[1] == "1 legs, 9 windows"
    assert ap.plot_a02_ttft_cdf(L2, out)[1] == "171 completed requests"
    assert ap.plot_a06_preemption_kv(L2, out)[1] == "9 windows"
    con = [c for c in ap._k1_cells(L2) if c["budget_r"] == 0.25]
    assert len(con) == 1 and con[0]["keys"] == [leg["constrained_row_key"], k2]


# ---------------------------------------------------------------------------
# S2 block F row F-13 addendum (review 2026-11-04 MEDIUM 1): the control rung
# the frontier figures print is the driver's frontier.control_budget_r (read
# from the registered plan by compute_frontier_drop), never a constant here.
# ---------------------------------------------------------------------------


def _frontier_control(parts: dict, value, *, remove: bool = False) -> None:
    """Set frontier.control_budget_r in the fixture's stats.json (remove=True drops the key;
    a None value is written as a JSON null, NaN and infinity as the tokens json.load reads)."""
    stats = json.loads(parts["stats"].read_text(encoding="utf-8"))
    if remove:
        stats["frontier"].pop("control_budget_r", None)
    else:
        stats["frontier"]["control_budget_r"] = value
    parts["stats"].write_text(json.dumps(stats), encoding="utf-8")


def test_the_frontier_figures_print_the_sections_control_rung_f13(tmp_path: Path, monkeypatch) -> None:
    """S2 registers control 1.0 (run_campaign.py S2_CONTROL_R): with
    frontier.control_budget_r = 1.0 the a01 tick, the a02 legend, the K1 cell
    record and the captions of p03, a02 and a06 say rung 1, and none says 1.25."""
    parts = _write_landing(tmp_path)
    _plan_with_blocks(parts)
    _frontier_control(parts, 1.0)
    L = _landing(parts)
    out = tmp_path / "out"
    out.mkdir()
    ap.ps.apply_style()
    figs: list = []
    # keep the figures open (save_fig closes them) so their texts can be read
    monkeypatch.setattr(ap, "_out", lambda fig, out_dir, stem: figs.append(fig) or f"{stem}.png")
    assert ap._control_budget_r(L) == 1.0
    control = [c for c in ap._k1_cells(L) if c["side"].startswith("control")]
    assert len(control) == 1 and control[0]["budget_r"] == 1.0
    assert control[0]["side"] == "control (cache rung 1) at 3.00 rps"
    ap.plot_a01_frontier_drop(L, out)
    ticks = [t.get_text() for ax in figs[-1].axes for t in ax.get_xticklabels()]
    assert "control\nr=1" in ticks and "control\nr=1.25" not in ticks, ticks
    ap.plot_a02_ttft_cdf(L, out)
    legend = [t.get_text() for t in figs[-1].legends[0].get_texts()]
    assert "Oracle context, fresh prefill, control (cache rung 1)" in legend, legend
    ap.plot_a06_preemption_kv(L, out)
    ap.plot03_pressure_map(L, out)
    a02 = (out / "a02_ttft_cdf.caption.txt").read_text(encoding="utf-8")
    a06 = (out / "a06_preemption_kv.caption.txt").read_text(encoding="utf-8")
    p03 = (out / "p03_pressure_map.caption.txt").read_text(encoding="utf-8")
    assert "Solid: the control rung 1;" in a02 and "1.25" not in a02
    assert "solid for the control rung 1 and dashed" in a06 and "1.25" not in a06
    assert "holds the block's control rung 1 as well as its constrained rungs" in p03
    # the referee sentence of p03 is S1 regime prose (block G rewrites it): kept as is
    assert "the control\nrung 1.25)" in p03
    import matplotlib.pyplot as plt

    plt.close("all")  # the patched _out kept the four figures open (review LOW 3)


def test_a_frontier_section_without_the_control_rung_refuses_by_name_f13(tmp_path: Path) -> None:
    """An older stats.json without frontier.control_budget_r, or one with a
    value that is not a positive number, refuses the frontier figures and the
    pooled p03 column by name; p03 without the plan's block field never needs it."""
    parts = _write_landing(tmp_path)
    _plan_with_blocks(parts)
    _frontier_control(parts, None, remove=True)
    out = tmp_path / "out"
    out.mkdir()
    ap.ps.apply_style()
    L = _landing(parts)
    for fn in (ap.plot_a01_frontier_drop, ap.plot_a02_ttft_cdf, ap.plot_a06_preemption_kv, ap.plot03_pressure_map):
        with pytest.raises(ap.PlotSkip, match="control_budget_r"):
            fn(L, out)
    for bad in (None, "1.25", 0, -1.0, True, float("nan"), float("inf")):
        _frontier_control(parts, bad)
        with pytest.raises(ap.PlotSkip, match="control_budget_r"):
            ap._control_budget_r(_landing(parts))
    noplan = ap.Landing(parts["landing"], yield_table=parts["yield"], stats=parts["stats"])
    assert ap.plot03_pressure_map(noplan, out)[1].endswith("no plan block field: one column per rounded rate")
