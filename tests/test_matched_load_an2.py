"""AN-2 (ADR-0211, owner 2026-11-20): two more descriptive blocks of the matched_load section.

``arm_pairs``: the registry's arm pairs whose two arms share a token class are offered
the same absolute rate under pressure (they share one measured rate limit), so their
yield shares compare at identical traffic. ``engine_pairs``: the same arm, rung and
load on another engine against the anchor engine (vLLM), with the ratio of the two
engines' rate limits and the bracket of both ladders. Neither spends alpha or enters a
gate. Spec: MyDocs/RunPod/S2_PLAN_A_B_QWEN3_235B_2026-11-19.md sections 17 to 17.8.

The fixture stack is tests/test_matched_load_an1.py's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import test_matched_load_an1 as an1
from test_matched_load_an1 import LIMIT, STEP, _build, _grid, _plan, _rungs, _run

_pressure_dataset_and_no_freeze = an1._pressure_dataset_and_no_freeze  # the autouse fixture of the stack


def _arm_rows(section: dict[str, Any]) -> dict[tuple[int, str, float, float], dict[str, Any]]:
    return {(r["contrast_id"], r["engine"], r["budget_r"], r["rate_frac"]): r for r in section["arm_pairs"]["rows"]}


def _engine_rows(section: dict[str, Any]) -> dict[tuple[str, str, float, float], dict[str, Any]]:
    return {(r["baseline"], r["engine"], r["budget_r"], r["rate_frac"]): r for r in section["engine_pairs"]["rows"]}


def test_two_arms_of_one_class_compare_at_one_offered_rate(tmp_path: Path) -> None:
    analysis_dir, stats = _run(_build(tmp_path, _grid(baselines=("B3", "B4", "B6"))), "--plan", str(_plan(tmp_path)))
    section = stats["matched_load"]
    rows = _arm_rows(section)
    # contrast #2 is "B4 against B3" in the registry: B4 is the cell, B3 the reference; every rung and load
    assert sorted(rows) == [(2, "vllm", r, f) for r in (0.25, 1.0) for f in (0.95, 1.05)]
    row = rows[(2, "vllm", 0.25, 0.95)]
    assert (row["pair"], row["cell_baseline"], row["reference_baseline"]) == ("B4 vs B3", "B4", "B3")
    assert "corpus-fresh" in row["cell_row_key"] and "corpus-reuse" in row["reference_row_key"]
    # B4 serves 11 and 10 of 20 in time, B3 serves 13 and 12: the hand count
    y = row["yield"]
    assert (y["mean_cell"], y["mean_reference"]) == (pytest.approx(0.525), pytest.approx(0.625))
    assert y["delta"] == pytest.approx(-0.1) and row["timely"]["delta"] == pytest.approx(-0.1)
    assert row["accepted"]["delta"] == pytest.approx(0.0)
    assert y["executed_alternative"] == "two-sided" and y["df"] == pytest.approx(2.0) and y["p_value"] is not None
    assert (row["n_windows_cell"], row["n_windows_reference"]) == (2, 2)
    # identical traffic: one class, one rate limit, one load fraction
    assert row["rate_qps_cell"] == row["rate_qps_reference"] == pytest.approx(0.95 * 0.45)
    assert row["class_tokens"] == 2336
    assert row["limit_check"] == {"cell": {"verdict": "confirmed", "detail": None},
                                  "reference": {"verdict": "confirmed", "detail": None}}
    summary = (analysis_dir / "summary.md").read_text(encoding="utf-8")
    assert "Two arms at one offered rate" in summary and "| squad_v2 | #2 B4 vs B3 | vllm | 0.25 | 0.95 | 2/2 | 0.525 | 0.625 | -0.100 [" in summary
    assert "4 rows, each p unadjusted" in summary


def test_three_arms_of_one_class_give_the_registered_pairs_alone(tmp_path: Path) -> None:
    _, stats = _run(_build(tmp_path, _grid(baselines=("B3", "B4", "B10"))), "--plan", str(_plan(tmp_path)))
    block = stats["matched_load"]["arm_pairs"]
    pairs = sorted({(r["contrast_id"], r["pair"]) for r in block["rows"]})
    assert pairs == [(2, "B4 vs B3"), (8, "B10 vs B3")] and len(block["rows"]) == 8   # never B10 against B4
    pooled = {p["pair"]: p for p in block["pooled"]}
    # B4 is 2 late requests behind B3 in every window: -0.1 on each of the 4 rows
    assert pooled["B4 vs B3"]["n_rows"] == 4 and pooled["B4 vs B3"]["n_negative_delta"] == 4
    assert pooled["B4 vs B3"]["yield_delta"] == pytest.approx({"median": -0.1, "min": -0.1, "max": -0.1})


def test_a_pair_forms_across_the_two_pressure_families(tmp_path: Path) -> None:
    # B1 is served in family F2 and B2 in F3; they share the 4,779-token class and its rate limit
    _, stats = _run(_build(tmp_path, _grid(baselines=("B1", "B2"))), "--plan", str(_plan(tmp_path)))
    row = _arm_rows(stats["matched_load"])[(1, "vllm", 0.25, 0.95)]
    assert row["pair"] == "B1 vs B2" and "|F2|" in row["cell_row_key"] and "|F3|" in row["reference_row_key"]
    assert row["yield"]["delta"] == pytest.approx(0.675 - 0.575)


def test_pairs_in_different_classes_or_with_an_absent_arm_are_listed_not_formed(tmp_path: Path) -> None:
    _, stats = _run(_build(tmp_path, _grid()), "--plan", str(_plan(tmp_path)))   # B1, B3, B6: three classes
    block = stats["matched_load"]["arm_pairs"]
    assert block["rows"] == []
    listed = {entry["contrast_id"]: entry for entry in block["not_formed"]}
    assert len(listed) == len(block["not_formed"])   # one line per pair, never two reasons for one
    assert all(entry["dataset"] == "squad_v2" for entry in block["not_formed"])
    assert listed[3]["pair"] == "B1 vs B6" and "different classes (4779 and 1127)" in listed[3]["reason"]
    assert listed[4]["pair"] == "B6 vs B3" and "different classes (1127 and 2336)" in listed[4]["reason"]
    assert "B2 is not served under pressure" in listed[1]["reason"]
    assert "B8 is not served under pressure" in listed[9]["reason"]


def test_two_arms_at_rates_apart_are_a_labeled_skip(tmp_path: Path) -> None:
    off = ("B4", "vllm", 0.25, 0.95)
    _, stats = _run(_build(tmp_path, _grid(baselines=("B3", "B4")), rates={off: 0.30}), "--plan", str(_plan(tmp_path)))
    block = stats["matched_load"]["arm_pairs"]
    assert len(block["rows"]) == 3 and (2, "vllm", 0.25, 0.95) not in _arm_rows(stats["matched_load"])
    (skip,) = block["skipped"]
    assert "B4 vs B3" in skip["pair"] and "0.3" in skip["reason"] and "0.4275" in skip["reason"]


def test_one_window_on_a_side_gives_the_arm_pair_estimate_without_a_test(tmp_path: Path) -> None:
    one = ("B4", "vllm", 0.25, 0.95)
    _, stats = _run(_build(tmp_path, _grid(baselines=("B3", "B4")), lates={one: (9,)}), "--plan", str(_plan(tmp_path)))
    y = _arm_rows(stats["matched_load"])[(2, "vllm", 0.25, 0.95)]["yield"]
    assert y["delta"] == pytest.approx(0.55 - 0.625) and y["p_value"] is None and "2 windows" in y["test_skipped"]


#: SGLang's rate limit over vLLM's, per rung (different at each rung, so a bracket read from the
#: wrong rung cannot pass: review 2026-11-20 LOW 9)
SGLANG_SCALE = {1.0: 0.5, 0.25: 0.4}


def _two_engine_tree(tmp_path: Path) -> tuple[Path, Path, tuple[Path, Path]]:
    """B3 and B6 on vLLM and on an SGLang with a smaller rate limit; one SGLang cell 2 requests later."""
    keys = _grid(("vllm", "sglang"), ("B3", "B6"))
    rates = {k: SGLANG_SCALE[k[2]] * k[3] * LIMIT[(k[0], k[2])] for k in keys if k[1] == "sglang"}
    run_dir = _build(tmp_path, keys, rates=rates, lates={("B3", "sglang", 0.25, 0.95): (9, 10)})
    files = (_rungs(tmp_path, "vllm"), _rungs(tmp_path, "sglang", scale=SGLANG_SCALE))
    return run_dir, _plan(tmp_path, files), files


def test_an_engine_against_the_anchor_engine_at_one_load_fraction(tmp_path: Path) -> None:
    run_dir, plan, files = _two_engine_tree(tmp_path)
    analysis_dir, stats = _run(run_dir, "--plan", str(plan), "--rung-calibration", *(str(f) for f in files))
    block = stats["matched_load"]["engine_pairs"]
    assert block["anchor_engine"] == "vllm" and len(block["rows"]) == 8   # 2 arms, 2 rungs, 2 loads
    row = _engine_rows(stats["matched_load"])[("B3", "sglang", 0.25, 0.95)]
    assert (row["engine"], row["anchor_engine"]) == ("sglang", "vllm")
    assert "|sglang|" in row["cell_row_key"] and "|vllm|" in row["anchor_row_key"]
    y = row["yield"]
    assert (y["mean_cell"], y["mean_anchor"], y["delta"]) == (pytest.approx(0.525), pytest.approx(0.625), pytest.approx(-0.1))
    assert (row["rate_qps_cell"], row["rate_qps_anchor"]) == (pytest.approx(0.4 * 0.4275), pytest.approx(0.4275))
    assert row["capacity_ratio"] == pytest.approx(0.4)
    assert row["yield_rps_ratio"] == pytest.approx(0.4 * 0.525 / 0.625)
    assert set(row["limit_check"]) == {"cell", "anchor"}
    # one capacity ratio per class and rung, with the bracket of the two ladders
    capacity = {(c["class_tokens"], c["budget_r"]): c for c in block["capacity"]}
    assert sorted(capacity) == [(1127, 0.25), (1127, 1.0), (2336, 0.25), (2336, 1.0)]
    cag = capacity[(2336, 0.25)]
    assert (cag["engine"], cag["anchor_engine"], cag["baselines"]) == ("sglang", "vllm", ["B3"])
    assert cag["capacity_ratio"] == pytest.approx(0.4) and cag["bracket"] == pytest.approx([0.4 / STEP, 0.4 * STEP])
    assert capacity[(2336, 1.0)]["bracket"] == pytest.approx([0.5 / STEP, 0.5 * STEP])
    (pooled,) = block["pooled"]
    assert (pooled["engine"], pooled["anchor_engine"], pooled["n_rows"], pooled["n_negative_delta"]) == ("sglang", "vllm", 8, 1)
    assert pooled["capacity_ratio"] == pytest.approx({"median": 0.45, "min": 0.4, "max": 0.5})
    summary = (analysis_dir / "summary.md").read_text(encoding="utf-8")
    assert "sglang against vllm" in summary and "| squad_v2 | sglang/vllm | 2336 | B3 | 0.25 | 0.400 [0.372, 0.43" in summary
    # review 2026-11-20 LOW 6: the reader is told what the p-values are, under each AN-2 table
    assert "8 rows, each p unadjusted" in summary and "one limit ratio per class and rung" in summary
    assert "| squad_v2 | B3 | sglang | 0.25 | 0.95 | 2/2 | 0.525 | 0.625 | -0.100 [" in summary


def test_the_engine_capacity_ratio_has_no_bracket_without_both_rung_files(tmp_path: Path) -> None:
    run_dir, _plan_path, files = _two_engine_tree(tmp_path)
    _, stats = _run(run_dir, "--plan", str(_plan(tmp_path, files[:1])), "--rung-calibration", str(files[0]))
    for entry in stats["matched_load"]["engine_pairs"]["capacity"]:
        assert entry["capacity_ratio"] == pytest.approx(SGLANG_SCALE[entry["budget_r"]]) and entry["bracket"] is None
        assert "no --rung-calibration file for engine 'sglang'" in entry["bracket_skipped"]


def test_without_the_anchor_engine_the_engine_block_holds_no_row(tmp_path: Path) -> None:
    _, stats = _run(_build(tmp_path, _grid(("sglang",), ("B3",))), "--plan", str(_plan(tmp_path)))
    block = stats["matched_load"]["engine_pairs"]
    assert block["rows"] == [] and block["capacity"] == [] and "vllm" in block["note"]
    assert stats["matched_load"]["n_legs"] == 2   # the AN-1 legs do not need the anchor


def _s2a_metrics() -> list[str]:
    text = (an1.REPO_ROOT / "scripts" / "6_experiments" / "profiles" / "S2A.env").read_text(encoding="utf-8")
    (value,) = [line.split("=", 1)[1].strip('"') for line in text.splitlines() if line.startswith("ANALYSIS_METRICS=")]
    return value.split()


def test_the_s2a_metric_list_runs_on_rows_shaped_like_a_real_tree(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # AN-2b: the profile's own list, on rows whose faithfulness is the null a campaign tree carries.
    # Review 2026-11-20 HIGH: with faithfulness in the list the look refused on every real tree.
    run_dir = _build(tmp_path, _grid(baselines=("B3", "B6")), with_f1=True)
    metrics = _s2a_metrics()
    _, stats = _run(run_dir, "--contrasts", "4", "--metrics", *metrics, "--plan", str(_plan(tmp_path)))
    assert stats["metrics"] == metrics == ["ttft_ms", "predicate"]
    by_metric = {c["metric"]: c for c in stats["contrasts"]}
    assert (by_metric["ttft_ms"]["test"], by_metric["predicate"]["test"]) == ("paired_wilcoxon", "mcnemar_binary")
    legs = {p["dataset_metric"] for p in stats["gatekeeping"]["primaries"] if p["endpoint"] == "contrast-4"}
    assert legs == {"squad_v2|ttft_ms", "squad_v2|predicate"}   # the registered co-primary set of #4, complete
    assert stats["matched_load"]["n_legs"] == 4
    # why the faithfulness score is not in the list: the loader joins no column for it, and the look refuses
    assert an1.rca.main([str(run_dir), "--contrasts", "4", "--metrics", "faithfulness"]) == 1
    assert "metric 'faithfulness' appears in no requests.jsonl/qa_evidence.jsonl record" in capsys.readouterr().err


def test_a_crash_in_one_an2_block_keeps_the_an1_legs_and_the_other_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # review 2026-11-20 LOW 5: the blocks are isolated from each other and from the legs
    def boom(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(an1.rca, "_arm_pairs_block", boom)
    run_dir, plan, files = _two_engine_tree(tmp_path)
    _, stats = _run(run_dir, "--plan", str(plan), "--rung-calibration", *(str(f) for f in files))
    section = stats["matched_load"]
    assert section["n_legs"] == 8 and len(section["engine_pairs"]["rows"]) == 8
    assert section["arm_pairs"]["rows"] == [] and section["arm_pairs"]["skipped"] == [
        {"pair": "arm_pairs", "reason": "RuntimeError: boom"}]


def test_two_arms_that_never_share_an_engine_rung_and_load_are_listed_not_formed(tmp_path: Path) -> None:
    # review 2026-11-20 LOW 7: both arms are served under pressure, but never side by side
    keys = [k for k in _grid(("vllm", "sglang"), ("B3", "B4")) if (k[0], k[1]) in {("B3", "vllm"), ("B4", "sglang")}]
    _, stats = _run(_build(tmp_path, keys), "--plan", str(_plan(tmp_path)))
    block = stats["matched_load"]["arm_pairs"]
    listed = {entry["contrast_id"]: entry["reason"] for entry in block["not_formed"]}
    assert len(listed) == len(block["not_formed"])
    assert block["rows"] == [] and "never served side by side" in listed[2]


def test_the_arm_pairs_of_two_datasets_stay_apart(tmp_path: Path) -> None:
    # review 2026-11-20 LOW: the AN-2 blocks on a row key that serves two datasets
    analysis_dir, stats = _run(_build(tmp_path, _grid(baselines=("B3", "B4")), extra_dataset="musique"),
                               "--plan", str(_plan(tmp_path)))
    rows = stats["matched_load"]["arm_pairs"]["rows"]
    assert len(rows) == 8 and sorted({row["dataset"] for row in rows}) == ["musique", "squad_v2"]
    extra = next(r for r in rows if (r["dataset"], r["budget_r"], r["rate_frac"]) == ("musique", 0.25, 0.95))
    assert (extra["n_windows_cell"], extra["n_windows_reference"]) == (2, 2)
    assert extra["rate_qps_cell"] == extra["rate_qps_reference"] == pytest.approx(0.5 * 0.95 * 0.45)
    assert sorted(p["dataset"] for p in stats["matched_load"]["arm_pairs"]["pooled"]) == ["musique", "squad_v2"]


def test_a_block_without_rows_prints_no_table_header(tmp_path: Path) -> None:
    # review 2026-11-20 LOW: three classes, so no arm pair forms; the summary lists why and shows no empty table
    analysis_dir, _stats = _run(_build(tmp_path, _grid()), "--plan", str(_plan(tmp_path)))
    summary = (analysis_dir / "summary.md").read_text(encoding="utf-8")
    block = summary.split("### Two arms at one offered rate", 1)[1].split("\n### ", 1)[0]
    assert "not formed: squad_v2 #3 B1 vs B6" in block and "| dataset | pair |" not in block and "rows, each p" not in block
