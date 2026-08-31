"""Offline tests for scripts/4_analysis/build_floor_table.py (P6 floor table).

WHAT is pinned and WHY:

- EXACT anchor arithmetic (qwen3-14b, c=9, s=32768: D = 48_318_382_080 B;
  r=0.5 → 147_456 tokens, concurrency ceiling 4) — the ±15% floor-first
  falsification (PUBLICATION.md P6, D6 §6.1) compares measured collapse onset
  against THESE numbers; a drifted byte or floored token silently moves the
  goalposts of a pre-registered test.
- lambda_compute is NULL with an explicit ``requires_calibration`` marker and
  lambda*_pred is labeled kv-bound-only — the fail-closed doctrine forbids
  fabricating the compute bound before the §9.2 roofline calibration exists.
- Registered-grid drift guards: budget levels {1.5,1.0,0.75,0.5,0.25} (full,
  §6.1) / {1.0,0.5,0.25} (reduced, §6.8) and rate fractions taken from
  load_generator's registered constants — the artifact and the dispatcher can
  never disagree about the grid.
- Overwrite REFUSAL without --force: a silently rewritten prediction is a
  postdiction; that refusal IS the point of persisting the table.
- Fail-closed matrix: unknown model/engine/grid/kv_dtype, the HF oracle
  (P2 excludes it from pressure sweeps), and non-positive/non-finite numerics
  all raise FloorTableError — never a silent default.
- Schema round-trip: the built dict survives json.dumps/loads unchanged, so
  the persisted artifact IS the in-memory prediction (no tuple/float drift).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.orchestration.load_generator import (  # noqa: E402
    D6_RATE_FRACTIONS,
    D6_REDUCED_RATE_FRACTIONS,
)

_SCRIPT_PATH = REPO_ROOT / "scripts" / "4_analysis" / "build_floor_table.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_floor_table", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # register BEFORE exec (dataclass-safe)
    spec.loader.exec_module(module)
    return module


ft = _load_module()

# The registered anchor shape used throughout (qwen3-14b, 160 KiB/token).
_ANCHOR = dict(
    model="qwen3-14b",
    engine="vllm",
    concurrency_target=9,
    avg_seq_tokens=32768,
    kv_dtype="bf16",
    grid="full",
    service_time_s=8.0,  # W chosen so lambda ceilings land on exact binary floats
)

_D_ANCHOR = 48_318_382_080  # 9 × 32768 × 163_840 B
_PER_SEQ_ANCHOR = 5_368_709_120  # 32768 × 163_840 B


def _rows_by_r(table):
    return {row["r"]: row for row in table["rows"]}


# --------------------------------------------------------------------------- #
# Exact anchor arithmetic (the registered prediction numbers)
# --------------------------------------------------------------------------- #


def test_anchor_full_grid_exact_arithmetic():
    table = ft.build_floor_table(**_ANCHOR)
    assert [row["r"] for row in table["rows"]] == [1.5, 1.0, 0.75, 0.5, 0.25]
    rows = _rows_by_r(table)

    # Demand D is the SAME per-family number on every budget row (§6.1: D is
    # defined at target concurrency, independent of r).
    assert all(row["demand_bytes"] == _D_ANCHOR for row in table["rows"])
    assert all(row["per_seq_bytes"] == _PER_SEQ_ANCHOR for row in table["rows"])

    # Hand-derived pins: budget B = floor(r×D); tokens = B // 163_840;
    # ceiling = B // per_seq; lambda_KV = ceiling / 8.0 s (Little's law).
    expected = {
        1.5: (72_477_573_120, 442_368, 13, 1.625),
        1.0: (48_318_382_080, 294_912, 9, 1.125),
        0.75: (36_238_786_560, 221_184, 6, 0.75),
        0.5: (24_159_191_040, 147_456, 4, 0.5),
        0.25: (12_079_595_520, 73_728, 2, 0.25),
    }
    for r, (budget, tokens, ceiling, lam) in expected.items():
        row = rows[r]
        assert row["budget_bytes"] == budget
        assert row["kv_token_capacity"] == tokens
        assert row["concurrency_ceiling"] == ceiling
        assert row["lambda_kv_rps"] == lam
        assert row["lambda_star_pred_rps"] == lam
        assert row["predicted_collapse_onset_rps"] == lam


def test_offered_rates_are_fraction_times_lambda_star():
    table = ft.build_floor_table(**_ANCHOR)
    for row in table["rows"]:
        assert row["rate_fractions"] == list(D6_RATE_FRACTIONS)
        assert row["offered_rates_rps_pred"] == [
            f * row["lambda_star_pred_rps"] for f in D6_RATE_FRACTIONS
        ]


def test_fp8_dtype_halves_bytes_not_tokens():
    # fp8 halves bytes/token, so D halves but token capacity at equal r is
    # unchanged — the dtype factor must cancel in the token arithmetic.
    table = ft.build_floor_table(**{**_ANCHOR, "kv_dtype": "fp8"})
    rows = _rows_by_r(table)
    assert rows[1.0]["demand_bytes"] == _D_ANCHOR // 2
    assert rows[1.0]["kv_token_capacity"] == 294_912
    assert rows[1.0]["concurrency_ceiling"] == 9


def test_hybrid_fixed_state_is_charged_per_sequence():
    # qwen3-next-80b: 24_576 B/token KV + a 38 MiB recurrent slot per seq
    # (NOT dtype-scaled). The slot must depress the concurrency ceiling.
    table = ft.build_floor_table(
        model="qwen3-next-80b",
        engine="sglang",
        concurrency_target=4,
        avg_seq_tokens=8192,
        kv_dtype="bf16",
        grid="reduced",
        service_time_s=2.0,
    )
    rows = _rows_by_r(table)
    per_seq = 8192 * 24_576 + 38 * 1024 * 1024  # 241_172_480
    assert all(row["per_seq_bytes"] == per_seq for row in table["rows"])
    assert rows[1.0]["demand_bytes"] == 964_689_920
    assert rows[1.0]["concurrency_ceiling"] == 4
    assert rows[1.0]["kv_token_capacity"] == 39_253
    assert rows[0.5]["budget_bytes"] == 482_344_960
    assert rows[0.5]["concurrency_ceiling"] == 2
    assert rows[0.5]["lambda_kv_rps"] == 1.0


def test_reduced_grid_levels_and_fractions():
    table = ft.build_floor_table(**{**_ANCHOR, "grid": "reduced"})
    assert [row["r"] for row in table["rows"]] == [1.0, 0.5, 0.25]
    for row in table["rows"]:
        assert row["rate_fractions"] == list(D6_REDUCED_RATE_FRACTIONS)
    assert table["provenance"]["rate_fractions_source"] == (
        "load_generator.D6_REDUCED_RATE_FRACTIONS"
    )


def test_registered_grids_are_the_charter_values():
    # Drift guards on the module constants themselves (§6.1 / §6.8).
    assert ft.FULL_BUDGET_LEVELS == (1.5, 1.0, 0.75, 0.5, 0.25)
    assert ft.REDUCED_BUDGET_LEVELS == (1.0, 0.5, 0.25)
    assert ft.GRID_RATE_FRACTIONS["full"] == D6_RATE_FRACTIONS
    assert ft.GRID_RATE_FRACTIONS["reduced"] == D6_REDUCED_RATE_FRACTIONS


# --------------------------------------------------------------------------- #
# lambda_compute: null, marked, never fabricated
# --------------------------------------------------------------------------- #


def test_lambda_compute_is_null_and_marked_never_fabricated():
    table = ft.build_floor_table(**_ANCHOR)
    for row in table["rows"]:
        assert row["lambda_compute_rps"] is None
        assert row["lambda_compute_status"] == "requires_calibration"
        assert row["lambda_star_basis"] == "kv-bound-only [pending calibration]"
        # With lambda_compute null, min(lambda_KV, lambda_compute) must
        # degenerate to EXACTLY the KV bound — never some other number.
        assert row["lambda_star_pred_rps"] == row["lambda_kv_rps"]


def test_zero_ceiling_budget_is_honest_not_refused():
    # c=1 at r=0.25 cannot hold even one full sequence: lambda_KV = 0 is the
    # honest prediction (collapse at ANY load), labeled by a note — refusing
    # would hide a legitimate grid point; inventing capacity would fabricate.
    table = ft.build_floor_table(
        **{**_ANCHOR, "concurrency_target": 1, "grid": "reduced"}
    )
    rows = _rows_by_r(table)
    assert rows[0.25]["concurrency_ceiling"] == 0
    assert rows[0.25]["lambda_kv_rps"] == 0.0
    assert rows[0.25]["lambda_star_pred_rps"] == 0.0
    assert rows[0.25]["offered_rates_rps_pred"] == [0.0, 0.0, 0.0]
    assert "predicted collapse at ANY offered load" in rows[0.25]["note"]
    assert rows[1.0]["concurrency_ceiling"] == 1
    assert "note" not in rows[1.0]


# --------------------------------------------------------------------------- #
# Fail-closed matrix
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "override, match",
    [
        ({"model": "gpt-oss-120b"}, "unknown model"),
        ({"engine": "tgi"}, "unknown engine"),
        ({"engine": "hf"}, "correctness oracle"),
        ({"grid": "medium"}, "unknown grid"),
        ({"kv_dtype": "int4"}, "unknown kv_dtype"),
        ({"concurrency_target": 0}, "concurrency_target"),
        ({"concurrency_target": True}, "concurrency_target"),
        ({"concurrency_target": 4.5}, "concurrency_target"),
        ({"avg_seq_tokens": 0}, "avg_seq_tokens"),
        ({"avg_seq_tokens": -1}, "avg_seq_tokens"),
        ({"service_time_s": 0.0}, "service_time_s"),
        ({"service_time_s": -3.0}, "service_time_s"),
        ({"service_time_s": float("inf")}, "service_time_s"),
        ({"service_time_s": float("nan")}, "service_time_s"),
        ({"service_time_s": "8"}, "service_time_s"),
    ],
)
def test_fail_closed_matrix(override, match):
    with pytest.raises(ft.FloorTableError, match=match):
        ft.build_floor_table(**{**_ANCHOR, **override})


# --------------------------------------------------------------------------- #
# CLI: artifact write, overwrite refusal, --force
# --------------------------------------------------------------------------- #


def _cli_argv(out: Path, service_time_s: str = "8.0") -> list:
    return [
        "--model", "qwen3-14b",
        "--engine", "vllm",
        "--concurrency-target", "9",
        "--avg-seq-tokens", "32768",
        "--kv-dtype", "bf16",
        "--grid", "full",
        "--service-time-s", service_time_s,
        "--out", str(out),
    ]


def test_cli_writes_artifact_and_echoes_inputs(tmp_path, capsys):
    out = tmp_path / "floor_table.json"
    assert ft._main(_cli_argv(out)) == 0
    assert "PENDING CALIBRATION" in capsys.readouterr().out
    table = json.loads(out.read_text())
    assert table["schema"] == "floor-table-v1"
    assert table["generated_inputs"] == {
        "model": "qwen3-14b",
        "engine": "vllm",
        "concurrency_target": 9,
        "avg_seq_tokens": 32768,
        "kv_dtype": "bf16",
        "grid": "full",
        "service_time_s": 8.0,
        "out": str(out),
    }
    assert table["provenance"]["kv_formula"] == "2*40*8*128*2"
    assert table["provenance"]["charter_refs"] == ["P6", "6.1", "6.8"]
    assert len(table["rows"]) == 5
    assert table["rows"][3]["kv_token_capacity"] == 147_456


def test_cli_refuses_overwrite_without_force(tmp_path, capsys):
    out = tmp_path / "floor_table.json"
    assert ft._main(_cli_argv(out)) == 0
    original = out.read_bytes()
    capsys.readouterr()

    # Same inputs, no --force: refuse AND leave the artifact byte-identical.
    assert ft._main(_cli_argv(out)) == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err
    assert "silently rewritten" in err
    assert out.read_bytes() == original

    # --force with changed inputs: the rewrite happens and is visible.
    assert ft._main(_cli_argv(out, service_time_s="4.0") + ["--force"]) == 0
    rewritten = json.loads(out.read_text())
    assert rewritten["generated_inputs"]["service_time_s"] == 4.0
    assert rewritten["rows"][3]["lambda_kv_rps"] == 1.0  # ceiling 4 / 4.0 s


def test_cli_refuses_unknown_model(tmp_path, capsys):
    out = tmp_path / "floor_table.json"
    argv = _cli_argv(out)
    argv[argv.index("qwen3-14b")] = "mystery-model"
    assert ft._main(argv) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not out.exists()


# --------------------------------------------------------------------------- #
# Schema round-trip
# --------------------------------------------------------------------------- #


def test_schema_round_trips_through_json():
    # The persisted artifact must BE the in-memory prediction: no tuples that
    # turn into lists, no ints that turn into floats, nothing unserializable.
    table = ft.build_floor_table(**_ANCHOR)
    assert json.loads(json.dumps(table)) == table
