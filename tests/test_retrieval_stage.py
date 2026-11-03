"""Tests for src/analysis/retrieval_stage.py and scripts/4_analysis/retrieval_stage.py
(S2 block F row F-7, 2026-11-03).

Synthetic run roots in the real layout (cells/<row_key>/cell.json and
window_<dataset>-<k>/requests.jsonl). Every case pins one clause of the Spec
Block: the row rule (ok rows, first pass per question, a None position is a
miss that stays in the denominator), the Wilson bounds and the exact McNemar on
hand-computed values, the F1-and-retrieval-arm selection, the named refusal
(exit 3), the unreadable file (exit 2), the skipped cells, and the two outputs
carrying the same numbers.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "4_analysis" / "retrieval_stage.py"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.analysis import retrieval_stage as rs  # noqa: E402

B5 = "retr-fresh|dense|none|single|{eng}|qwen3-14b|F1"
B6 = "retr-fresh|rerank|none|single|{eng}|qwen3-14b|F1"
B8 = "retr-store|rerank|none|single|{eng}|qwen3-14b|F1"
GOLD = "gold-fresh|none|none|single|{eng}|qwen3-14b|F1"
B5_F2 = "retr-fresh|dense|none|single|{eng}|qwen3-14b|F2|r0.5|lam1.05"


def _row(ex: str, gip, *, ok: bool = True, rep="0", hit=None) -> dict:
    rec = {"example_id": ex, "ok": ok, "gold_position_in_prompt": gip, "retrieval_hit": hit}
    if rep is not None:
        rec["repeat_index"] = rep
    return rec


def _cell(root: Path, row_key: str, *, baseline, windows: dict) -> Path:
    cdir = root / "cells" / row_key
    cdir.mkdir(parents=True, exist_ok=True)
    if baseline is not None:
        (cdir / "cell.json").write_text(json.dumps({"baseline": baseline, "windows": {}}), encoding="utf-8")
    for wname, rows in windows.items():
        wdir = cdir / wname
        wdir.mkdir(exist_ok=True)
        if rows is not None:
            (wdir / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return cdir


def _run(root: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), str(root), "--out", str(out)],
                          capture_output=True, text=True, cwd=str(REPO))


def test_rates_table_counts_first_pass_ok_rows_and_wilson_bounds(tmp_path: Path) -> None:
    rows = [
        _row("q1", 0, hit=1.0), _row("q2", 1, hit=1.0), _row("q3", 2, hit=1.0),
        _row("q4", 0, hit=1.0), _row("q5", 3, hit=1.0),
        _row("q6", -1, hit=1.0),        # retrieved, then cut from the prompt: a miss, a lost hit
        _row("q7", None, hit=0.0),      # no position: a miss that stays in the denominator
        _row("q8", -1, hit=0.0),
        _row("q9", 0, ok=False),        # a failed request never counts
        _row("q1", -1, hit=0.0),        # the second pass of q1: the first pass wins
    ]
    _cell(tmp_path, B5.format(eng="vllm"), baseline="B5", windows={"window_squad_v2-01": rows})
    table = rs.rates_table(rs.collect(tmp_path))
    assert len(table) == 1
    t = table[0]
    assert (t["dataset"], t["baseline"], t["engine"]) == ("squad_v2", "B5", "vllm")
    assert (t["n"], t["n_none"], t["hits"]) == (8, 1, 5)
    assert t["rate"] == pytest.approx(0.625)
    assert t["wilson_low"] == pytest.approx(0.305742, abs=1e-6)
    assert t["wilson_high"] == pytest.approx(0.863156, abs=1e-6)
    assert (t["retrieval_hits"], t["lost"]) == (6, 1)


def test_repeat_index_absent_is_pass_zero_and_another_repeat_is_its_own_question(tmp_path: Path) -> None:
    rows = [_row("q1", 0, rep=None), _row("q1", -1, rep="0"), _row("q1", -1, rep="1")]
    _cell(tmp_path, B5.format(eng="vllm"), baseline="B5", windows={"window_qasper-01": rows})
    (t,) = rs.rates_table(rs.collect(tmp_path))
    assert (t["n"], t["hits"]) == (2, 1)


def test_paired_table_on_a_known_pair_and_on_zero_discordant_pairs(tmp_path: Path) -> None:
    b5 = {"q1": 0, "q2": 0, "q3": 0, "q4": 0, "q5": -1, "q6": -1}
    b6 = {"q1": -1, "q2": -1, "q3": -1, "q4": 0, "q5": 0, "q6": -1, "q7": 0}  # q7 is not in B5
    _cell(tmp_path, B5.format(eng="vllm"), baseline="B5",
          windows={"window_qasper-01": [_row(q, g) for q, g in b5.items()]})
    _cell(tmp_path, B6.format(eng="vllm"), baseline="B6",
          windows={"window_qasper-01": [_row(q, g) for q, g in b6.items()]})
    same = [_row(q, g) for q, g in b5.items()]
    _cell(tmp_path, B5.format(eng="sglang"), baseline="B5", windows={"window_qasper-01": same})
    _cell(tmp_path, B6.format(eng="sglang"), baseline="B6", windows={"window_qasper-01": same})
    paired = {(p["dataset"], p["engine"]): p for p in rs.paired_table(rs.collect(tmp_path))}
    v = paired[("qasper", "vllm")]
    assert (v["n_pairs"], v["b5_only"], v["b6_only"], v["both"], v["neither"]) == (6, 3, 1, 1, 1)
    assert v["delta"] == pytest.approx(-2 / 6)
    assert v["mcnemar_p"] == pytest.approx(0.625)
    s = paired[("qasper", "sglang")]
    assert (s["b5_only"], s["b6_only"], s["delta"], s["mcnemar_p"]) == (0, 0, 0.0, 1.0)


def test_only_f1_windows_of_retrieval_arms_count_and_b8_is_one(tmp_path: Path) -> None:
    hit = [_row("q1", 0)]
    _cell(tmp_path, B5.format(eng="vllm"), baseline="B5", windows={"window_musique-01": hit})
    _cell(tmp_path, B8.format(eng="vllm"), baseline="B8", windows={"window_musique-01": hit})
    _cell(tmp_path, GOLD.format(eng="vllm"), baseline="B1", windows={"window_musique-01": hit})
    _cell(tmp_path, B5_F2.format(eng="vllm"), baseline="B5", windows={"window_musique-01": hit, "window_musique-02": hit})
    keys = sorted((t["dataset"], t["baseline"], t["engine"], t["n"]) for t in rs.rates_table(rs.collect(tmp_path)))
    assert keys == [("musique", "B5", "vllm", 1), ("musique", "B8", "vllm", 1)]


def test_no_f1_retrieval_window_is_a_named_refusal_with_exit_3(tmp_path: Path) -> None:
    _cell(tmp_path, GOLD.format(eng="vllm"), baseline="B1", windows={"window_musique-01": [_row("q1", 0)]})
    proc = _run(tmp_path, tmp_path / "out")
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "[retrieval_stage] REFUSED" in proc.stderr and "no F1 window of a retrieval arm" in proc.stderr
    assert not (tmp_path / "out").exists()


def test_an_unreadable_or_missing_requests_file_exits_2_naming_it(tmp_path: Path) -> None:
    cdir = _cell(tmp_path, B5.format(eng="vllm"), baseline="B5", windows={"window_musique-01": [_row("q1", 0)]})
    bad = cdir / "window_musique-01" / "requests.jsonl"
    bad.write_text('{"example_id": "q1", "ok": true\n', encoding="utf-8")
    proc = _run(tmp_path, tmp_path / "out")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert str(bad) in proc.stderr
    bad.unlink()
    proc = _run(tmp_path, tmp_path / "out")
    assert proc.returncode == 2 and str(bad) in proc.stderr


def test_short_row_keys_and_cells_without_a_record_are_skipped_and_listed(tmp_path: Path) -> None:
    _cell(tmp_path, B5.format(eng="vllm"), baseline="B5", windows={"window_musique-01": [_row("q1", 0)]})
    _cell(tmp_path, "retr-fresh|dense", baseline="B5", windows={"window_musique-01": [_row("q1", 0)]})
    _cell(tmp_path, B6.format(eng="vllm"), baseline=None, windows={"window_musique-01": [_row("q1", 0)]})
    proc = _run(tmp_path, tmp_path / "out")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = json.loads((tmp_path / "out" / "retrieval_stage.json").read_text(encoding="utf-8"))
    skipped = {s["cell"]: s["reason"] for s in doc["skipped_cells"]}
    assert "fewer than seven parts" in skipped["retr-fresh|dense"]
    assert "cell.json" in skipped[B6.format(eng="vllm")]
    assert [t["baseline"] for t in doc["rates"]] == ["B5"]


def test_markdown_and_json_carry_the_same_numbers_for_every_group(tmp_path: Path) -> None:
    for eng in ("vllm", "sglang"):
        _cell(tmp_path, B5.format(eng=eng), baseline="B5", windows={
            "window_qasper-01": [_row("q1", 0), _row("q2", -1), _row("q3", 0)],
            "window_squad_v2-01": [_row("q1", 0), _row("q2", 0)],
        })
        _cell(tmp_path, B6.format(eng=eng), baseline="B6", windows={
            "window_qasper-01": [_row("q1", 0), _row("q2", 0), _row("q3", -1)],
        })
    out = tmp_path / "out"
    proc = _run(tmp_path, out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[retrieval_stage] groups=6 pairs=2" in proc.stdout
    doc = json.loads((out / "retrieval_stage.json").read_text(encoding="utf-8"))
    md = (out / "retrieval_stage.md").read_text(encoding="utf-8")
    assert len(doc["rates"]) == 6 and len(doc["paired"]) == 2
    for t in doc["rates"]:
        line = f"| {t['dataset']} | {t['baseline']} | {t['engine']} | {t['n']} | {t['n_none']} | {t['hits']} | {t['rate']:.3f} |"
        assert line in md, line
    for p in doc["paired"]:
        assert f"| {p['dataset']} | {p['engine']} | {p['n_pairs']} | {p['b5_only']} | {p['b6_only']} |" in md
    assert "\u2014" not in md   # no em dash in the rendered sidebar (owner rule)


def test_wilson_bounds_are_clamped_and_mcnemar_is_exact() -> None:
    lo, hi = rs.wilson(4, 4)
    assert lo == pytest.approx(0.510109, abs=1e-6) and hi == pytest.approx(1.0) and hi <= 1.0
    lo, hi = rs.wilson(0, 3)
    assert lo == 0.0 and hi == pytest.approx(0.561497, abs=1e-6)
    assert rs.mcnemar_exact(3, 1) == pytest.approx(0.625)
    assert rs.mcnemar_exact(0, 0) == 1.0
    assert rs.mcnemar_exact(147, 23) == pytest.approx(2.5674365e-23, rel=1e-6)


def test_window_name_pattern_equals_the_layout_module_pattern() -> None:
    """The module keeps its own copy of the window-dir regex (the layout module
    imports pandas); the copy must never drift from the layout's."""
    from src.orchestration.campaign_layout import WINDOW_DIR_RE
    assert rs.WINDOW_DIR_RE.pattern == WINDOW_DIR_RE.pattern


def test_a_row_that_is_not_an_object_exits_2_and_a_non_object_cell_record_is_skipped(tmp_path: Path) -> None:
    """Review L3 (2026-11-03): a JSON line that is a list crashed the pass with an
    AttributeError; so did a cell.json that is a list."""
    cdir = _cell(tmp_path, B5.format(eng="vllm"), baseline="B5", windows={"window_musique-01": [_row("q1", 0)]})
    bad = cdir / "window_musique-01" / "requests.jsonl"
    bad.write_text('[1, 2]\n', encoding="utf-8")
    proc = _run(tmp_path, tmp_path / "out")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert f"{bad}: line 1 is not a JSON object" in proc.stderr
    bad.write_text(json.dumps(_row("q1", 0)) + "\n", encoding="utf-8")
    other = _cell(tmp_path, B6.format(eng="vllm"), baseline="B6", windows={"window_musique-01": [_row("q1", 0)]})
    (other / "cell.json").write_text("[]", encoding="utf-8")
    proc = _run(tmp_path, tmp_path / "out")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = json.loads((tmp_path / "out" / "retrieval_stage.json").read_text(encoding="utf-8"))
    assert {s["cell"] for s in doc["skipped_cells"]} == {B6.format(eng="vllm")}
    assert [t["baseline"] for t in doc["rates"]] == ["B5"]


def test_rows_without_an_example_id_are_skipped_and_counted(tmp_path: Path) -> None:
    """Review L6 (2026-11-03): rows without example_id merged into one question
    named "None"; they are skipped and counted instead."""
    rows = [_row("q1", 0), _row("q2", None), {"ok": True, "gold_position_in_prompt": 0}, {"ok": True, "gold_position_in_prompt": 1, "example_id": None}]
    _cell(tmp_path, B5.format(eng="vllm"), baseline="B5", windows={"window_musique-01": rows})
    proc = _run(tmp_path, tmp_path / "out")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = json.loads((tmp_path / "out" / "retrieval_stage.json").read_text(encoding="utf-8"))
    assert doc["rates"][0]["n"] == 2 and doc["rates"][0]["hits"] == 1
    assert doc["n_rows_without_example_id"] == 2
    assert "Rows without example_id (skipped): 2" in (tmp_path / "out" / "retrieval_stage.md").read_text(encoding="utf-8")
