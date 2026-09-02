"""Campaign engine-pair mode of the sec. 8.9 T=0 divergence instrument (T6.4).

WHAT: pins scripts/4_analysis/token_divergence.py --campaign-root — the walker
over a RESULTS_LAYOUT-v2 tree (cells/<row_key>/window_<dataset>-NN/) that
groups windows by (model, dataset, arm, grid-point, replicate) and computes
the pilot-proven sec. 8.9 statistics for EVERY engine pair in each group, with
hand-computed agreement/first-divergence/answer-changing numbers on a
synthetic three-engine mini campaign.

WHY: the charter targets the instrument at model x ENGINE-PAIR (the pilot
compared arms on ONE engine), and its labels are registration text — HF pairs
are the ORACLE, engine-engine pairs without HF are DeepSeek-V3's recorded D4
substitute and must be labeled substitute/weaker-than-oracle. The fail-closed
lanes are behavior, not comments: no eligible groups = loud error naming the
search; mismatched query sets = labeled skip (never a silent intersection);
windows without generations = labeled skip; existing --out artifacts refuse
without --force (build_floor_table mirror).

No GPU/network: the tree is synthetic, the whitespace tokenizer is pure
Python, and the answer-changing classification imports the local quality
module only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "4_analysis"))
sys.path.insert(0, str(_REPO_ROOT))
import token_divergence as td  # noqa: E402
from src.analysis.cellspec import CellSpec  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic RESULTS_LAYOUT-v2 mini campaign
# --------------------------------------------------------------------------- #


def _spec(engine: str) -> CellSpec:
    """One F1 anchor cell that is legal for vllm, sglang AND hf (hf is the
    sub-pressure single-device oracle, so the shared tuple must be F1/single)."""
    return CellSpec(
        arm="gold-fresh",
        retriever="none",
        policy="none",
        topology="single",
        engine=engine,  # type: ignore[arg-type]
        model="qwen3-14b",
        family="F1",
    )


def _ev(ex: str, answer: str, gold: str, *, rep: int = 0, err: Optional[str] = None) -> Dict[str, Any]:
    """One qa_evidence.jsonl row carrying the #127 identity triple + validity."""
    empty = (not err) and not answer.strip()
    return {
        "example_id": ex,
        "repeat_index": rep,
        "record_index": None,
        "generated_answer": answer,
        "reference_answer": gold,
        "ok": (not err) and not empty,
        "error": err,
        "empty_generation": empty,
    }


def _mk_cell(
    run_root: Path,
    spec: CellSpec,
    windows: Dict[str, Optional[List[Dict[str, Any]]]],
    *,
    declare: Optional[List[str]] = None,
) -> Path:
    """Write cells/<row_key>/ with cell.json + window dirs.

    ``windows``: window_key -> qa_evidence rows (None = window dir WITHOUT
    qa_evidence.jsonl). ``declare`` limits which window keys enter cell.json's
    windows[] table (default: all) — an undeclared dir models resume residue.
    """
    cell_dir = run_root / "cells" / spec.to_row_key()
    windows_meta: Dict[str, Any] = {}
    for window_key, rows in windows.items():
        dataset, ordinal = window_key.rsplit("-", 1)
        wdir = cell_dir / f"window_{window_key}"
        wdir.mkdir(parents=True)
        if rows is not None:
            (wdir / "qa_evidence.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
            )
        if declare is None or window_key in declare:
            windows_meta[window_key] = {
                "dataset": dataset,
                "seed": 7,
                "rep": int(ordinal),
                "budget_r": spec.budget_r,
                "rate_frac": spec.rate_frac,
                "t_start": 0.0,
                "t_end": 1.0,
            }
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "cell.json").write_text(
        json.dumps(
            {"cellspec": spec.to_flat_dict(), "baseline": "B1", "windows": windows_meta}
        ),
        encoding="utf-8",
    )
    return cell_dir


#: Shared four-query manifest slice with KNOWN divergences (hand-computed below).
_HF_ROWS = [
    _ev("q1", "Paris", "Paris"),
    _ev("q2", "London", "London"),
    _ev("q3", "Rome", "Rome"),
    _ev("q4", "Madrid", "Madrid"),
]
_VLLM_ROWS = [
    _ev("q1", "Paris", "Paris"),
    _ev("q2", "London", "London"),
    _ev("q3", "Rome", "Rome"),
    _ev("q4", "Barcelona", "Madrid"),  # EM flip vs gold -> answer-CHANGING
]
_SGLANG_ROWS = [
    _ev("q1", "Paris", "Paris"),
    _ev("q2", "London.", "London"),  # punctuation reword -> answer-PRESERVING
    _ev("q3", "Rome", "Rome"),
    _ev("q4", "Barcelona", "Madrid"),
]


def _mk_three_engine_campaign(root: Path) -> Path:
    for engine, rows in (("hf", _HF_ROWS), ("vllm", _VLLM_ROWS), ("sglang", _SGLANG_ROWS)):
        _mk_cell(root / f"run-{engine}", _spec(engine), {"squad_v2-01": rows})
    return root


def _pairs_by_engines(group: Dict[str, Any]) -> Dict[tuple, Dict[str, Any]]:
    return {(p["engine_a"], p["engine_b"]): p for p in group["pairs"]}


# --------------------------------------------------------------------------- #
# Hand-computed engine-pair statistics + labeling
# --------------------------------------------------------------------------- #


def test_campaign_pairs_hand_computed(tmp_path: Path) -> None:
    _mk_three_engine_campaign(tmp_path)
    result = td.compute_campaign_divergence(str(tmp_path))

    assert result["mode"] == "campaign-engine-pair"
    assert result["counts"]["run_roots"] == 3
    assert result["counts"]["cells"] == 3
    assert result["counts"]["windows_scanned"] == 3
    assert result["counts"]["groups_compared"] == 1
    assert result["counts"]["groups_skipped"] == 0

    (report,) = result["reports"]
    assert (report["model"], report["dataset"]) == ("qwen3-14b", "squad_v2")
    assert report["counts"] == {"groups_compared": 1, "pairs": 3, "skips": 0}
    assert "temperature=0.0" in report["t0_contract"]  # T=0 producer contract travels with the numbers

    (group,) = report["groups"]
    assert group["arm"] == "gold-fresh" and group["rep"] == 1
    assert group["n_queries"] == 4
    assert set(group["engines"]) == {"hf", "sglang", "vllm"}
    assert group["engines"]["vllm"]["n_answers"] == 4
    pairs = _pairs_by_engines(group)
    assert set(pairs) == {("sglang", "hf"), ("vllm", "hf"), ("sglang", "vllm")}

    # (sglang, hf) oracle: q2 (punct) + q4 diverge -> agreement 2/4; q2 survives
    # tokenization (pos 0), classification vs gold: q2 preserving, q4 changing.
    p = pairs[("sglang", "hf")]
    assert p["n_compared"] == 4
    assert p["raw_divergent"] == 2
    assert p["agreement_rate"] == 0.5
    assert p["normalized_divergent"] == 1  # q2's 'London.' normalizes equal
    assert p["token_agreement_rate"] == 0.5
    assert p["first_divergence"]["n_token_divergent"] == 2
    assert p["first_divergence"]["median_position"] == 0
    assert p["first_divergence"]["max_position"] == 0
    ad = p["answer_divergence"]
    assert ad["answer_changing"] == 1 and ad["answer_preserving"] == 1
    assert ad["answer_changing_rate"] == 0.25
    assert ad["answer_changing_share_of_divergent"] == 0.5
    assert ad["classification_basis"] == ["gold"]

    # (vllm, hf) oracle: only q4 diverges (EM flip -> changing).
    p = pairs[("vllm", "hf")]
    assert p["raw_divergent"] == 1
    assert p["agreement_rate"] == 0.75
    assert p["answer_divergence"]["answer_changing"] == 1
    assert p["answer_divergence"]["answer_changing_share_of_divergent"] == 1.0

    # (sglang, vllm) substitute: only q2 diverges; both said Barcelona on q4
    # so the substitute CANNOT see the shared wrong answer — exactly why it is
    # weaker than the oracle.
    p = pairs[("sglang", "vllm")]
    assert p["raw_divergent"] == 1
    assert p["agreement_rate"] == 0.75
    assert p["normalized_divergent"] == 0
    assert p["answer_divergence"]["answer_changing"] == 0
    assert p["answer_divergence"]["answer_preserving"] == 1


def test_pair_labeling_oracle_vs_substitute(tmp_path: Path) -> None:
    _mk_three_engine_campaign(tmp_path)
    result = td.compute_campaign_divergence(str(tmp_path))
    pairs = _pairs_by_engines(result["reports"][0]["groups"][0])

    for key in (("sglang", "hf"), ("vllm", "hf")):
        assert pairs[key]["label"] == td.PAIR_LABEL_ORACLE == "oracle"
        assert pairs[key]["engine_b"] == "hf"  # HF is always the reference side
    sub = pairs[("sglang", "vllm")]
    # Registered D4 label verbatim: must say SUBSTITUTE, and the strength must
    # say it is weaker than the oracle — never readable as an oracle number.
    assert sub["label"] == td.PAIR_LABEL_SUBSTITUTE
    assert sub["label"] == "cross-engine agreement — oracle-exempt substitute (D4)"
    assert "substitute" in sub["strength"] and "weaker than oracle" in sub["strength"]


def test_window_dir_re_matches_layout_producer() -> None:
    # The local regex + exemption set MUST stay byte-identical to the layout
    # producer's (importing campaign_layout at runtime would drag pandas into
    # a stdlib-only CLI; this pin replaces the import).
    from src.orchestration.campaign_layout import (
        QA_EVIDENCE_EXEMPT_DATASETS,
        WINDOW_DIR_RE,
    )

    assert td._CAMPAIGN_WINDOW_DIR_RE.pattern == WINDOW_DIR_RE.pattern
    assert td._QA_EVIDENCE_EXEMPT_DATASETS == QA_EVIDENCE_EXEMPT_DATASETS


# --------------------------------------------------------------------------- #
# Fail-closed lanes
# --------------------------------------------------------------------------- #


def test_mismatched_query_sets_are_labeled_skip_never_intersected(tmp_path: Path) -> None:
    _mk_cell(tmp_path / "run-vllm", _spec("vllm"), {"squad_v2-01": _VLLM_ROWS})
    _mk_cell(tmp_path / "run-sglang", _spec("sglang"), {"squad_v2-01": _SGLANG_ROWS[:3]})  # q4 missing

    result = td.compute_campaign_divergence(str(tmp_path))
    assert result["counts"]["groups_multi_engine"] == 1
    assert result["counts"]["groups_compared"] == 0
    assert result["counts"]["groups_skipped"] == 1
    (report,) = result["reports"]
    assert report["groups"] == []  # never a silently intersected comparison
    (skip,) = report["skips"]
    assert "mismatched query sets" in skip["reason"]
    assert "refusing to silently intersect" in skip["reason"]
    assert "sglang: 3 key(s), missing 1" in skip["reason"]
    assert "q4" in skip["reason"]  # the offending key is named


def test_window_without_generations_is_labeled_skip(tmp_path: Path) -> None:
    _mk_cell(tmp_path / "run-hf", _spec("hf"), {"squad_v2-01": None})  # no qa_evidence.jsonl
    _mk_cell(tmp_path / "run-vllm", _spec("vllm"), {"squad_v2-01": _VLLM_ROWS})
    _mk_cell(tmp_path / "run-sglang", _spec("sglang"), {"squad_v2-01": _SGLANG_ROWS})

    result = td.compute_campaign_divergence(str(tmp_path))
    assert result["counts"]["windows_without_generations"] == 1
    (report,) = result["reports"]
    (group,) = report["groups"]
    # hf dropped out -> the ONLY pair is the substitute; no oracle fabricated.
    (pair,) = group["pairs"]
    assert (pair["engine_a"], pair["engine_b"]) == ("sglang", "vllm")
    assert pair["label"] == td.PAIR_LABEL_SUBSTITUTE
    (skip,) = report["skips"]
    assert "without T=0 generations" in skip["reason"]
    assert "run-hf" in skip["where"]


def test_undeclared_window_dir_is_labeled_skip(tmp_path: Path) -> None:
    # window_squad_v2-02 exists on disk but is NOT in cell.json windows[]:
    # resume residue — its replicate id is unknown and must not be guessed.
    _mk_cell(
        tmp_path / "run-vllm",
        _spec("vllm"),
        {"squad_v2-01": _VLLM_ROWS, "squad_v2-02": _VLLM_ROWS},
        declare=["squad_v2-01"],
    )
    _mk_cell(tmp_path / "run-sglang", _spec("sglang"), {"squad_v2-01": _SGLANG_ROWS})

    result = td.compute_campaign_divergence(str(tmp_path))
    assert result["counts"]["groups_compared"] == 1
    (report,) = result["reports"]
    (skip,) = report["skips"]
    assert "not declared in cell.json windows[]" in skip["reason"]
    assert "refusing to guess" in skip["reason"]
    assert "window_squad_v2-02" in skip["where"]


def test_ambiguous_replicate_same_engine_twice_is_labeled_skip(tmp_path: Path) -> None:
    # Two vllm run trees serve the SAME (model, dataset, arm, grid, rep):
    # choosing one silently would hide a duplicated dispatch.
    _mk_cell(tmp_path / "run-vllm-a", _spec("vllm"), {"squad_v2-01": _VLLM_ROWS})
    _mk_cell(tmp_path / "run-vllm-b", _spec("vllm"), {"squad_v2-01": _VLLM_ROWS})
    _mk_cell(tmp_path / "run-sglang", _spec("sglang"), {"squad_v2-01": _SGLANG_ROWS})

    result = td.compute_campaign_divergence(str(tmp_path))
    assert result["counts"]["groups_compared"] == 0
    assert result["counts"]["groups_skipped"] == 1
    (skip,) = result["reports"][0]["skips"]
    assert "ambiguous replicate" in skip["reason"]
    assert "refusing to choose" in skip["reason"]
    assert "vllm: 2 windows" in skip["reason"]


def test_no_eligible_groups_is_loud_error_naming_the_search(tmp_path: Path) -> None:
    # A single engine can form no pair: loud error, never an empty artifact.
    _mk_cell(tmp_path / "run-vllm", _spec("vllm"), {"squad_v2-01": _VLLM_ROWS})

    with pytest.raises(td.CampaignDivergenceError) as exc_info:
        td.compute_campaign_divergence(str(tmp_path))
    msg = str(exc_info.value)
    assert "no eligible engine-pair groups" in msg
    assert str(tmp_path) in msg  # names WHAT was searched
    assert "1 run tree(s)" in msg and "1 cell(s)" in msg and "1 window(s)" in msg
    assert "single engine" in msg
    assert ">=2 engines" in msg


def test_empty_root_is_loud_error(tmp_path: Path) -> None:
    with pytest.raises(td.CampaignDivergenceError) as exc_info:
        td.compute_campaign_divergence(str(tmp_path))
    assert "no RESULTS_LAYOUT-v2 run tree" in str(exc_info.value)


def test_row_key_roundtrip_mismatch_refuses(tmp_path: Path) -> None:
    # cell.json's cellspec must round-trip to the dirname (§2 identity):
    # a renamed cell dir is layout drift, refused with the problem named.
    cell_dir = _mk_cell(tmp_path / "run-vllm", _spec("vllm"), {"squad_v2-01": _VLLM_ROWS})
    cell_dir.rename(cell_dir.with_name("hand-built-name"))
    _mk_cell(tmp_path / "run-sglang", _spec("sglang"), {"squad_v2-01": _SGLANG_ROWS})

    with pytest.raises(td.CampaignDivergenceError) as exc_info:
        td.compute_campaign_divergence(str(tmp_path))
    msg = str(exc_info.value)
    assert "RESULTS_LAYOUT contract violation" in msg
    assert "round-trips to row key" in msg


# --------------------------------------------------------------------------- #
# CLI: per-(model, dataset) artifacts, overwrite refusal, stdout summary
# --------------------------------------------------------------------------- #


def test_cli_writes_reports_and_refuses_overwrite_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    _mk_three_engine_campaign(tmp_path / "campaign")
    out_dir = tmp_path / "reports"

    rc = td.main(["--campaign-root", str(tmp_path / "campaign"), "--out", str(out_dir)])
    assert rc == 0
    artifact = out_dir / "token_divergence__qwen3-14b__squad_v2.json"
    assert artifact.exists()
    report = json.loads(artifact.read_text(encoding="utf-8"))
    assert report["mode"] == "campaign-engine-pair"
    assert report["counts"]["pairs"] == 3
    capsys.readouterr()

    # Second run WITHOUT --force: refused (build_floor_table mirror), artifact intact.
    before = artifact.read_text(encoding="utf-8")
    rc = td.main(["--campaign-root", str(tmp_path / "campaign"), "--out", str(out_dir)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err and str(artifact) in err and "--force" in err
    assert artifact.read_text(encoding="utf-8") == before

    # WITH --force: deliberate overwrite succeeds.
    rc = td.main(
        ["--campaign-root", str(tmp_path / "campaign"), "--out", str(out_dir), "--force"]
    )
    assert rc == 0


def test_cli_no_eligible_groups_exits_2_with_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    _mk_cell(tmp_path / "run-vllm", _spec("vllm"), {"squad_v2-01": _VLLM_ROWS})
    out_dir = tmp_path / "reports"

    rc = td.main(["--campaign-root", str(tmp_path), "--out", str(out_dir)])
    assert rc == 2  # loud, unlike the pilot lane's benign SKIP (exit 0)
    err = capsys.readouterr().err
    assert "REFUSED" in err and "no eligible engine-pair groups" in err
    assert not out_dir.exists()  # nothing minted on refusal


def test_cli_stdout_summary_shape(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _mk_three_engine_campaign(tmp_path / "campaign")
    rc = td.main(
        ["--campaign-root", str(tmp_path / "campaign"), "--out", str(tmp_path / "reports")]
    )
    assert rc == 0
    out = capsys.readouterr().out
    lines = out.splitlines()

    (header,) = [l for l in lines if l.startswith("model") and "pair" in l]
    for column in ("model", "dataset", "pair", "kind", "groups", "n", "agree %", "ans-chg %"):
        assert column in header
    table_rows = [l for l in lines if l.startswith("qwen3-14b")]
    assert len(table_rows) == 3  # one row per engine pair
    assert any("sglang<->vllm" in l and "substitute" in l for l in table_rows)
    assert sum("oracle" in l for l in table_rows) == 2
    # Pooled display numbers match the hand-computed rates.
    (sub_row,) = [l for l in table_rows if "substitute" in l]
    assert "75.00%" in sub_row and "0.00%" in sub_row
    assert "squad_v2" in sub_row
    # The searched-scope line and the artifact pointer are both present.
    assert any("searched: 3 run tree(s)" in l for l in lines)
    assert any("token_divergence__qwen3-14b__squad_v2.json" in l for l in lines)


def test_cli_mode_exclusivity(tmp_path: Path) -> None:
    # Exactly one lane: neither, or both, is a usage error (argparse exit 2).
    with pytest.raises(SystemExit):
        td.main([])
    with pytest.raises(SystemExit):
        td.main(
            ["--results-dir", str(tmp_path), "--campaign-root", str(tmp_path)]
        )
    with pytest.raises(SystemExit):  # campaign lane without --out
        td.main(["--campaign-root", str(tmp_path)])
    with pytest.raises(SystemExit):  # pilot lane must not take campaign flags
        td.main(["--results-dir", str(tmp_path), "--force"])
