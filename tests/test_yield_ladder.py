"""T6.2 — the charter-S1 yield ladder + independence null made VISIBLE.

WHAT is pinned and WHY:

- S1(a) ladder mandate, STRUCTURAL: ``yield_ladder`` is the only ladder-table
  producer and it refuses any row lacking throughput_rps/goodput_rps/
  yield_rps — so Y can never be printed without raw throughput and G beside
  it. Pinned because S1 is BINDING on every figure/table and a "just this
  once" Y-only column is exactly the regression the mandate forbids.
- S1(b) independence null: G·E[v] and the covariance gap render beside every
  Y from the SAME WindowMetrics fields lane 1a carries — never recomputed,
  never fabricated. The covariance gap IS the finding; a table that dropped
  the null would present Y as if timeliness ⊥ veridicality were established.
- §6.6 never-mixed-bases guard at table generation: the pooled multi-window
  table calls lane-1a's ``assert_single_basis`` and propagates its §6.6
  refusals verbatim (mixed/unlabeled/empty pools). Pinned so the guard
  cannot be quietly bypassed by a future "fast path".
- Missing-field honesty: pre-ladder records (BOTH null fields absent) render
  the EXPLICIT ``n/a (pre-ladder data)`` cell while Y still shows with its
  S1(a) companions; a PARTIAL null group or a present-but-NaN value refuses
  — the carve-out covers absent fields only, never broken ones. Pinned
  because 'or 0.0'-style back-fill on absent metrics is this repo's named
  bug class.
- Wiring: a real end-to-end campaign analysis (the tests/test_predicate_chain
  fixture tree, #14 requested) lands ``yield_ladder.{json,md}`` beside
  stats.json with per-window values matching goodput.evaluate_window; the
  pass skips LOUDLY with no artifact when there are no ladder inputs and is
  suppressed under an active §9.8 seal (window dirs carry arm-bearing axes).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yield_ladder as yl  # noqa: E402
from src.analysis.goodput import (  # noqa: E402
    BASIS_AGGREGATE,
    BASIS_PER_GPU,
    GoodputError,
    SLOBaseline,
    evaluate_window,
)

_BASELINE = SLOBaseline(ttft_s=0.1, tpot_s=0.05)


def _metrics(*, gpu_count: int = 1, n_slow: int = 2, n_unveridical: int = 3):
    """A real WindowMetrics with a NONZERO covariance gap: the slow (untimely)
    rows are drawn from the veridical block, so timely and veridical are
    correlated and Y != G·E[v]."""
    n = 10
    records = pd.DataFrame(
        {
            # SLO gate: ttft <= 1.0s. The first n_slow rows blow it.
            "ttft_s": [5.0 if i < n_slow else 0.2 for i in range(n)],
            "tpot_s": [0.02] * n,
            "ok": [True] * n,
            # the LAST n_unveridical rows fail the §8.5 predicate.
            "veridical": [i < n - n_unveridical for i in range(n)],
        }
    )
    return evaluate_window(
        records, _BASELINE, duration_s=20.0, gpu_count=gpu_count
    )


def _items(**kw):
    return [("cells/a/window_w1", _metrics(**kw))]


# ---------------------------------------------------------------------------
# Ladder rows complete (S1 a+b): every column, values = the metrics fields
# ---------------------------------------------------------------------------


def test_rows_carry_the_full_ladder_in_input_order() -> None:
    m1 = _metrics()
    m2 = _metrics(gpu_count=2)
    rows = yl.build_ladder_rows([("w1", m1), ("w2", m2)])
    assert [r["window"] for r in rows] == ["w1", "w2"]
    for row, m in zip(rows, (m1, m2)):
        assert tuple(row) == yl.MACHINE_COLUMNS
        assert row["throughput_rps"] == pytest.approx(m.throughput_rps)
        assert row["goodput_rps"] == pytest.approx(m.goodput_rps)
        assert row["yield_rps"] == pytest.approx(m.yield_rps)
        assert row["independence_null_rps"] == pytest.approx(
            m.independence_null_rps
        )
        assert row["covariance_gap_rps"] == pytest.approx(m.covariance_gap_rps)
        assert row["gpu_count"] == m.gpu_count
        assert row["basis"] == BASIS_AGGREGATE
        assert row["note"] is None
    # the fixture really exercises clause (b): the gap is nonzero.
    assert rows[0]["covariance_gap_rps"] != pytest.approx(0.0)


def test_json_dict_rows_are_first_class_inputs() -> None:
    # The machine table must build from JSON-roundtripped to_flat_dict rows
    # (bases tuples become lists — lane 1a's accepted mapping form).
    m = _metrics(gpu_count=2)
    item = json.loads(json.dumps(m.to_flat_dict()))
    rows = yl.build_ladder_rows([("w1", item)])
    assert rows[0]["yield_rps"] == pytest.approx(m.yield_rps)
    assert rows[0]["gpu_count"] == 2


def test_markdown_renders_ladder_and_stamp() -> None:
    md = yl.render_ladder_markdown(
        [("w1", _metrics())], stamp="DESIGN-INPUT-ONLY"
    )
    assert "DESIGN-INPUT-ONLY" in md
    for header in ("G (req/s)", "Y (req/s)", "G·E[v] null (req/s)", "cov gap"):
        assert header in md
    assert "| w1 |" in md
    m = _metrics()
    assert format(m.yield_rps, ".6g") in md
    assert format(m.independence_null_rps, ".6g") in md


# ---------------------------------------------------------------------------
# S1(a) refusals: Y can never be printed without its companions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing", ["throughput_rps", "goodput_rps", "yield_rps"]
)
def test_row_missing_a_mandate_field_refuses(missing: str) -> None:
    item = _metrics().to_flat_dict()
    del item[missing]
    with pytest.raises(yl.LadderError, match=f"S1 clause \\(a\\)|{missing}"):
        yl.build_ladder_rows([("w1", item)])


def test_y_alone_cannot_render() -> None:
    # The structural core of the mandate: a bare Y row has no path to a table.
    with pytest.raises(yl.LadderError, match="S1"):
        yl.build_ladder_rows([("w1", {"yield_rps": 0.5})])


def test_nonfinite_mandate_field_refuses() -> None:
    item = _metrics().to_flat_dict()
    item["throughput_rps"] = float("nan")
    with pytest.raises(yl.LadderError, match="finite"):
        yl.build_ladder_rows([("w1", item)])


def test_string_mandate_field_refuses() -> None:
    item = _metrics().to_flat_dict()
    item["goodput_rps"] = "0.4"
    with pytest.raises(yl.LadderError, match="must be a number"):
        yl.build_ladder_rows([("w1", item)])


# ---------------------------------------------------------------------------
# §6.6 pool refusals (delegated to lane 1a — citation must survive)
# ---------------------------------------------------------------------------


def test_mixed_basis_pool_refuses_with_6_6_citation() -> None:
    good = _metrics().to_flat_dict()
    mixed = _metrics(gpu_count=2).to_flat_dict()
    mixed["goodput_per_gpu"] = mixed["goodput_rps"]  # forgot the division
    with pytest.raises(GoodputError, match="§6.6"):
        yl.build_ladder_rows([("w1", good), ("w2", mixed)])


def test_unlabeled_record_refuses_with_6_6_citation() -> None:
    item = _metrics().to_flat_dict()
    del item["bases"]
    with pytest.raises(GoodputError, match="§6.6"):
        yl.build_ladder_rows([("w1", item)])


def test_empty_pool_refuses() -> None:
    with pytest.raises(GoodputError, match="zero windows"):
        yl.build_ladder_rows([])


def test_duplicate_window_id_refuses() -> None:
    with pytest.raises(yl.LadderError, match="duplicate"):
        yl.build_ladder_rows([("w1", _metrics()), ("w1", _metrics())])


@pytest.mark.parametrize("basis", [BASIS_PER_GPU, "bogus"])
def test_non_aggregate_basis_refuses(basis: str) -> None:
    # A per-GPU ladder would need throughput/null fields lane 1a deliberately
    # does not ship — refusing beats fabricating derived columns.
    with pytest.raises(yl.LadderError, match="aggregate"):
        yl.build_ladder_rows(_items(), basis=basis)


# ---------------------------------------------------------------------------
# Missing-field honesty: the n/a carve-out (and its edges) — S1(b)
# ---------------------------------------------------------------------------


def _pre_ladder_item() -> dict:
    item = _metrics().to_flat_dict()
    del item["independence_null_rps"]
    del item["covariance_gap_rps"]
    return item


def test_pre_ladder_rows_render_explicit_na_with_companions() -> None:
    item = _pre_ladder_item()
    rows = yl.build_ladder_rows([("old_pilot_w", item)])
    row = rows[0]
    assert row["independence_null_rps"] is None
    assert row["covariance_gap_rps"] is None
    assert row["note"] == yl.PRE_LADDER_NA
    # Y still appears WITH its available companions (never dropped, never 0.0)
    assert row["yield_rps"] == pytest.approx(item["yield_rps"])
    assert row["throughput_rps"] == pytest.approx(item["throughput_rps"])
    assert row["goodput_rps"] == pytest.approx(item["goodput_rps"])
    md = yl.render_ladder_markdown([("old_pilot_w", item)], stamp="X")
    assert yl.PRE_LADDER_NA in md
    assert format(item["yield_rps"], ".6g") in md


def test_partial_null_group_refuses_as_malformed() -> None:
    item = _metrics().to_flat_dict()
    del item["covariance_gap_rps"]  # null present, gap missing: not "old"
    with pytest.raises(yl.LadderError, match="partial independence-null"):
        yl.build_ladder_rows([("w1", item)])


def test_present_but_nan_null_field_refuses() -> None:
    # The carve-out covers ABSENT fields; a NaN is corrupt, not pre-ladder.
    item = _metrics().to_flat_dict()
    item["independence_null_rps"] = float("nan")
    with pytest.raises(yl.LadderError, match="finite"):
        yl.build_ladder_rows([("w1", item)])


# ---------------------------------------------------------------------------
# write_yield_ladder: files land, refusals leave nothing behind
# ---------------------------------------------------------------------------


def test_write_yield_ladder_lands_both_files(tmp_path: Path) -> None:
    result = yl.write_yield_ladder(
        tmp_path, [("w1", _metrics()), ("w2", _pre_ladder_item())],
        stamp="DESIGN-INPUT-ONLY",
    )
    assert result["n_rows"] == 2 and result["n_pre_ladder"] == 1
    doc = json.loads((tmp_path / yl.LADDER_JSON_NAME).read_text())
    assert doc["mode_stamp"] == "DESIGN-INPUT-ONLY"
    assert doc["basis"] == BASIS_AGGREGATE
    assert doc["columns"] == list(yl.MACHINE_COLUMNS)
    assert len(doc["rows"]) == 2
    assert doc["rows"][1]["note"] == yl.PRE_LADDER_NA
    md = (tmp_path / yl.LADDER_MD_NAME).read_text()
    assert "DESIGN-INPUT-ONLY" in md and yl.PRE_LADDER_NA in md


def test_unstamped_write_refuses(tmp_path: Path) -> None:
    with pytest.raises(yl.LadderError, match="stamp"):
        yl.write_yield_ladder(tmp_path, _items(), stamp="")
    assert not (tmp_path / yl.LADDER_JSON_NAME).exists()


def test_refused_ladder_leaves_no_partial_artifact(tmp_path: Path) -> None:
    bad = _metrics().to_flat_dict()
    del bad["throughput_rps"]
    with pytest.raises(yl.LadderError):
        yl.write_yield_ladder(
            tmp_path, [("w1", _metrics()), ("w2", bad)], stamp="X"
        )
    assert not (tmp_path / yl.LADDER_JSON_NAME).exists()
    assert not (tmp_path / yl.LADDER_MD_NAME).exists()


# ---------------------------------------------------------------------------
# Wiring: the campaign analysis emits the artifact beside its outputs
# ---------------------------------------------------------------------------

import build_predicate_table as bpt  # noqa: E402
import organize_results as org  # noqa: E402
import run_campaign_analysis as rca  # noqa: E402
import test_predicate_chain as tpc  # noqa: E402  (the #119 tree builders)


@pytest.fixture(autouse=True)
def _no_machine_freeze_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Hermetic on machines that carry the τ freeze artifact (same rationale
    # as the identical fixture in tests/test_predicate_chain.py).
    monkeypatch.setenv(bpt.FREEZE_ENV_VAR, str(tmp_path / "absent_freeze.json"))


@pytest.fixture()
def predicate_run(tmp_path: Path) -> Path:
    run_dir = tpc._build_sealed_run(tmp_path)
    tpc._write_scoring_pass(run_dir)
    assert bpt.main([
        str(run_dir), "--scoring-run-id", tpc.SCORING_ID,
        "--max-null-fraction", "0.5",
    ]) == 0
    assert org.main([str(run_dir)]) == 0
    return run_dir


def test_analysis_run_lands_ladder_in_output_tree(predicate_run: Path) -> None:
    rc = rca.main([
        str(predicate_run), "--contrasts", "4", "14",
        "--metrics", "ttft_ms", "predicate",
    ])
    assert rc == 0
    analysis_dir = sorted((predicate_run / "analysis").iterdir())[-1]
    json_path = analysis_dir / yl.LADDER_JSON_NAME
    md_path = analysis_dir / yl.LADDER_MD_NAME
    assert json_path.is_file() and md_path.is_file()
    doc = json.loads(json_path.read_text())
    assert doc["mode_stamp"] == rca.DESIGN_STAMP
    # the #14 executor evaluated the 4 F2 windows (2 engines × 2 windows)
    rows = doc["rows"]
    assert len(rows) == 4
    assert all(r["note"] is None for r in rows)  # fresh data: full ladder
    assert all(r["basis"] == BASIS_AGGREGATE for r in rows)
    # values for the vllm ordinal-1 window, derived from the fixture design:
    # 16 issued, 1 serving failure (all completions timely under the floors),
    # 1 predicate-false completion, 60 s window.
    (row,) = [
        r for r in rows if "vllm" in r["window"] and r["window"].endswith("-01")
    ]
    assert row["throughput_rps"] == pytest.approx(15 / 60)
    assert row["goodput_rps"] == pytest.approx(15 / 60)
    assert row["yield_rps"] == pytest.approx(14 / 60)
    assert row["independence_null_rps"] == pytest.approx((15 / 60) * (14 / 16))
    assert row["covariance_gap_rps"] == pytest.approx(
        14 / 60 - (15 / 60) * (14 / 16)
    )
    # ladder mandate visible in the human table too
    md = md_path.read_text()
    assert "Y (req/s)" in md and "G·E[v] null (req/s)" in md


def test_pass_skips_loudly_without_ladder_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A look without #14 computes no goodput windows: loud skip, NO artifact
    # (an empty table would itself violate the §6.6 empty-pool refusal).
    assert rca.run_yield_ladder_pass(
        tmp_path, {}, rca.DESIGN_STAMP, blinding_active=False
    ) is None
    assert "SKIP" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_pass_suppressed_under_active_blinding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Window dirs embed row keys (arm-bearing axes): an active §9.8 seal
    # withholds the artifact — same rule as the stats truth_tax section.
    assert rca.run_yield_ladder_pass(
        tmp_path,
        {"cells/armed_key/window_w1": _metrics()},
        rca.DESIGN_STAMP,
        blinding_active=True,
    ) is None
    assert "SUPPRESSED" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []
