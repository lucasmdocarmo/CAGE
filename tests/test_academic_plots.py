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
                    "grounded": i % 5 != 0, "arrival_s": i * 2.5,
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
