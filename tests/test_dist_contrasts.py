"""Tests for src/analysis/dist_contrasts.py + the driver's DIST wiring (T1.1/T7.2).

WHAT: pins the Wave-3 contrast-execution layer for the registered DIST
contrasts — #18 (distribution's buy-back at transfer price, tp-vs-pd matched
per-GPU deltas on §6.6 basis b), #19 (does dedup survive the wire — raw
per-instance prefix-hit accounting from the T4.1 role-tagged / T4.2
raw-counter telemetry series), and the T7.2 #14 normalized-pressure alignment
on the CAGE-OWN rho axis (§8.8) — plus the additive
run_campaign_analysis.py wiring (run_dist_contrasts_pass /
run_pressure_alignment_pass, dispatch, artifacts).

WHY: the fail-closed discipline is the deliverable — hand-computed pairings
and deltas pin the arithmetic; the pairing matcher must REFUSE ambiguous
duplicates; per-GPU basis discipline must refuse mixed-basis pools via
goodput.assert_single_basis; EVERY labeled-skip path (missing gpu_count,
missing instance column, missing raw counter fields, zero-delta denominators,
missing own-accounting artifacts citing §8.8) must name its exact missing
input; #14 bucketing edges (inside/outside tol, boundary, in-band ties) are
pinned; and NO NOT-IMPLEMENTED label may remain for #18/#19 — when inputs
are complete they execute, and when incomplete the label is an auditable
skip listing what is absent (the #13 gate reports PENDING, never PASS).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_campaign_analysis as rca  # noqa: E402
import src.analysis.dist_contrasts as dc  # noqa: E402
from src.analysis.goodput import (  # noqa: E402
    BASIS_AGGREGATE,
    BASIS_PER_GPU,
    GoodputError,
    WINDOW_BASES,
    WindowMetrics,
)
from src.analysis.stats.families import (  # noqa: E402
    UNGATED,
    WINDOW_SECONDARY_UPSTREAM,
)

MODEL = "qwen3-14b"


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _wm(
    goodput_rps: float = 8.0,
    yield_rps: float = 6.0,
    gpu_count: int = 2,
    *,
    yield_frac: float = 0.6,
) -> WindowMetrics:
    """A consistent hand-built WindowMetrics (per-GPU == rps / gpu_count, the
    §6.6 internal invariant assert_single_basis audits)."""
    return WindowMetrics(
        n_issued=100,
        n_completed=95,
        n_timely=80,
        n_veridical=70,
        n_yield=60,
        duration_s=10.0,
        attainment=0.95,
        throughput_rps=9.5,
        goodput_rps=goodput_rps,
        yield_rps=yield_rps,
        goodput_frac=0.8,
        yield_frac=yield_frac,
        veridical_frac=0.7,
        independence_null_rps=goodput_rps * 0.7,
        independence_null_frac=0.56,
        covariance_gap=yield_frac - 0.56,
        covariance_gap_rps=yield_rps - goodput_rps * 0.7,
        truth_tax_rps=goodput_rps - yield_rps,
        truth_tax_frac=0.8 - yield_frac,
        gpu_count=gpu_count,
        goodput_per_gpu=goodput_rps / gpu_count,
        yield_per_gpu=yield_rps / gpu_count,
        bases=WINDOW_BASES,
    )


def _w18(
    window: str,
    topology: str,
    *,
    gpu_count: Any = 2,
    goodput_rps: float = 8.0,
    yield_rps: float = 6.0,
    metrics: Any = "auto",
    **axes: Any,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "arm": "gold-fresh",
        "retriever": "none",
        "policy": "none",
        "engine": "vllm",
        "model": MODEL,
        "dataset": "squad_v2",
        "budget_r": 0.5,
        "rate_frac": 0.9,
        "replicate": 1,
    }
    base.update(axes)
    if metrics == "auto":
        metrics = _wm(goodput_rps, yield_rps, gpu_count)
    return {
        **base,
        "topology": topology,
        "window": window,
        "gpu_count": gpu_count,
        "metrics": metrics,
    }


def _samples(instance: str, q0: float, h0: float, q1: float, h1: float) -> list[dict]:
    return [
        {
            "instance": instance,
            "prefix_cache_queries_total": q0,
            "prefix_cache_hits_total": h0,
        },
        {
            "instance": instance,
            "prefix_cache_queries_total": q1,
            "prefix_cache_hits_total": h1,
        },
    ]


def _w19(
    window: str,
    topology: str,
    series: list[dict] | None,
    *,
    baseline: str = "B3",
    **axes: Any,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "arm": "corpus-reuse",
        "retriever": "none",
        "policy": "none",
        "engine": "vllm",
        "model": MODEL,
        "dataset": "squad_v2",
        "budget_r": 0.5,
        "rate_frac": 0.9,
        "replicate": 1,
    }
    base.update(axes)
    return {
        **base,
        "topology": topology,
        "baseline": baseline,
        "window": window,
        "series": series,
    }


def _aw(
    window: str,
    engine: str,
    rho_own: float | None,
    *,
    dataset: str = "squad_v2",
    yield_frac: float = 0.6,
    metrics: Any = "auto",
) -> dict[str, Any]:
    if metrics == "auto":
        metrics = _wm(yield_frac=yield_frac)
    return {
        "window": window,
        "engine": engine,
        "dataset": dataset,
        "rho_own": rho_own,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# #18 — pairing, deltas, basis discipline, gate, labeled skips
# ---------------------------------------------------------------------------


class TestContrast18:
    def test_hand_computed_pair_deltas_on_basis_b(self) -> None:
        # tp: 1 GPU, G=8 rps -> 8/GPU, Y=6 -> 6/GPU.
        # pd: 2 GPUs, G=12 -> 6/GPU, Y=10 -> 5/GPU.
        # delta (pd - tp): goodput_per_gpu = -2, yield_per_gpu = -1.
        section = dc.execute_contrast_18(
            [
                _w18("w-tp", "tp", gpu_count=1, goodput_rps=8.0, yield_rps=6.0),
                _w18("w-pd", "pd", gpu_count=2, goodput_rps=12.0, yield_rps=10.0),
            ]
        )
        assert section["status"] == "EXECUTED"
        assert section["basis"] == BASIS_PER_GPU
        (pair,) = section["pairs"]
        assert pair["basis"] == BASIS_PER_GPU
        assert pair["delta_goodput_per_gpu"] == pytest.approx(-2.0)
        assert pair["delta_yield_per_gpu"] == pytest.approx(-1.0)
        assert pair["goodput_per_gpu_tp"] == pytest.approx(8.0)
        assert pair["goodput_per_gpu_pd"] == pytest.approx(6.0)
        (entry,) = section["per_dataset"]
        assert entry["n_pairs"] == 1
        assert entry["mean_delta_goodput_per_gpu"] == pytest.approx(-2.0)
        assert entry["paired_deltas_yield_per_gpu"] == [pytest.approx(-1.0)]
        assert entry["basis"] == BASIS_PER_GPU

    def test_replicates_pair_ordinal_to_ordinal_and_mean_over_pairs(self) -> None:
        windows = [
            _w18("tp-1", "tp", gpu_count=1, goodput_rps=8.0, replicate=1),
            _w18("pd-1", "pd", gpu_count=2, goodput_rps=12.0, replicate=1),
            _w18("tp-2", "tp", gpu_count=1, goodput_rps=10.0, replicate=2),
            _w18("pd-2", "pd", gpu_count=2, goodput_rps=24.0, replicate=2),
        ]
        section = dc.execute_contrast_18(windows)
        assert len(section["pairs"]) == 2
        (entry,) = section["per_dataset"]
        # deltas: rep1 = 6-8 = -2; rep2 = 12-10 = +2 -> mean 0.
        assert entry["n_pairs"] == 2
        assert entry["mean_delta_goodput_per_gpu"] == pytest.approx(0.0)
        assert sorted(entry["paired_deltas_goodput_per_gpu"]) == [
            pytest.approx(-2.0),
            pytest.approx(2.0),
        ]

    def test_ci_is_the_pending_marker_never_an_interval(self) -> None:
        section = dc.execute_contrast_18(
            [_w18("t", "tp", gpu_count=1), _w18("p", "pd", gpu_count=2)]
        )
        (entry,) = section["per_dataset"]
        assert entry["ci"] == dc.PENDING_CI_MARKER
        assert "pending" in entry["ci"]
        assert "ci95_low" not in entry  # no fabricated interval fields

    def test_gate_upstream_is_families_registered_topology(self) -> None:
        section = dc.execute_contrast_18([])
        assert section["gate"]["upstream"] == WINDOW_SECONDARY_UPSTREAM["DIST"]
        assert section["gate"]["upstream"] == "contrast-13"

    def test_gate_absent_reports_pending_never_pass(self) -> None:
        section = dc.execute_contrast_18(
            [_w18("t", "tp", gpu_count=1), _w18("p", "pd", gpu_count=2)],
            gate_13=None,
        )
        assert section["gate"]["status"] == "PENDING"
        assert section["gate"]["outcome"] is None
        assert "PASS" not in json.dumps(section["gate"])

    @pytest.mark.parametrize(
        "passed,status", [(True, "OPEN"), (False, "CLOSED")]
    )
    def test_gate_outcome_is_consumed_not_recomputed(
        self, passed: bool, status: str
    ) -> None:
        section = dc.execute_contrast_18(
            [], gate_13={"endpoint": "contrast-13", "passed": passed}
        )
        assert section["gate"]["status"] == status
        assert section["gate"]["outcome"]["passed"] is passed

    def test_malformed_gate_refuses(self) -> None:
        with pytest.raises(dc.DistContrastError, match="passed"):
            dc.execute_contrast_18([], gate_13={"verdict": "yes"})

    def test_missing_gpu_count_is_labeled_skip_naming_window(self) -> None:
        win = _w18("no-gpus", "pd", gpu_count=2)
        win["gpu_count"] = None  # caller failed to supply the count
        section = dc.execute_contrast_18([win])
        assert section["status"] == "SKIPPED-INPUTS-INCOMPLETE"
        (skip,) = section["skips"]
        assert skip["window"] == "no-gpus"
        assert "gpu_count" in skip["reason"]

    def test_missing_metrics_is_labeled_skip(self) -> None:
        section = dc.execute_contrast_18([_w18("no-m", "tp", metrics=None)])
        (skip,) = section["skips"]
        assert skip["window"] == "no-m"
        assert "WindowMetrics" in skip["reason"]

    def test_single_topology_is_labeled_out_not_paired(self) -> None:
        section = dc.execute_contrast_18([_w18("s", "single")])
        (skip,) = section["skips"]
        assert "not a #18 side" in skip["reason"]

    def test_unmatched_window_is_labeled_skip(self) -> None:
        section = dc.execute_contrast_18([_w18("lonely-tp", "tp", gpu_count=1)])
        (skip,) = section["skips"]
        assert "no pd partner" in skip["reason"]
        assert section["pairs"] == []

    def test_duplicate_side_on_one_key_refuses_ambiguous(self) -> None:
        with pytest.raises(dc.DistContrastError, match="ambiguous"):
            dc.execute_contrast_18(
                [
                    _w18("pd-a", "pd", gpu_count=2),
                    _w18("pd-b", "pd", gpu_count=2),
                ]
            )

    def test_differing_axes_never_pair(self) -> None:
        # Identical except rate_frac -> two unmatched keys, no pair.
        section = dc.execute_contrast_18(
            [
                _w18("t", "tp", gpu_count=1, rate_frac=0.8),
                _w18("p", "pd", gpu_count=2, rate_frac=0.9),
            ]
        )
        assert section["pairs"] == []
        assert len(section["skips"]) == 2

    def test_unset_coord_matches_unset_never_a_number(self) -> None:
        section = dc.execute_contrast_18(
            [
                _w18("t", "tp", gpu_count=1, budget_r=None),
                _w18("p", "pd", gpu_count=2, budget_r=None),
            ]
        )
        assert len(section["pairs"]) == 1
        assert section["pairs"][0]["axes"]["budget_r"] == "<unset>"

    def test_mixed_basis_pool_refuses_via_assert_single_basis(self) -> None:
        broken = _wm(12.0, 10.0, 2).to_flat_dict()
        broken["goodput_per_gpu"] = 12.0  # forgot the ÷gpu_count — mixed basis
        with pytest.raises(GoodputError, match="mixes bases"):
            dc.execute_contrast_18(
                [
                    _w18("t", "tp", gpu_count=1),
                    _w18("p", "pd", gpu_count=2, metrics=broken),
                ]
            )

    def test_gpu_count_contradicting_metrics_refuses(self) -> None:
        with pytest.raises(dc.DistContrastError, match="contradicts"):
            dc.execute_contrast_18(
                [
                    _w18("t", "tp", gpu_count=1),
                    _w18("p", "pd", gpu_count=4, metrics=_wm(12.0, 10.0, 2)),
                ]
            )

    def test_missing_identity_axis_refuses_loud(self) -> None:
        win = _w18("w", "tp", gpu_count=1)
        del win["policy"]
        with pytest.raises(dc.DistContrastError, match="policy"):
            dc.execute_contrast_18([win])


# ---------------------------------------------------------------------------
# #19 — raw per-instance hit accounting across the wire
# ---------------------------------------------------------------------------


class TestContrast19:
    def _happy_windows(self) -> list[dict[str, Any]]:
        pd_series = _samples("prefill-0", 100, 40, 200, 90) + _samples(
            "decode-0", 10, 5, 60, 15
        )
        comp_series = _samples("node-0", 0, 0, 100, 80)
        return [
            _w19("w-pd", "pd", pd_series),
            _w19("w-tp", "tp", comp_series),
        ]

    def test_hand_computed_role_rates_and_deltas(self) -> None:
        section = dc.execute_contrast_19(self._happy_windows())
        assert section["status"] == "EXECUTED"
        (row,) = section["rows"]
        # prefill: Δq=100 Δh=50 -> 0.5; decode: Δq=50 Δh=10 -> 0.2;
        # comparator: Δq=100 Δh=80 -> 0.8.
        assert row["prefill"]["hit_rate"] == pytest.approx(0.5)
        assert row["decode"]["hit_rate"] == pytest.approx(0.2)
        assert row["comparator"]["hit_rate"] == pytest.approx(0.8)
        assert row["delta_prefill_vs_comparator"] == pytest.approx(-0.3)
        assert row["delta_decode_vs_comparator"] == pytest.approx(-0.6)
        assert row["comparator_topology"] == "tp"

    def test_exploratory_and_ungated_by_registration(self) -> None:
        section = dc.execute_contrast_19(self._happy_windows())
        assert section["tier"] == "exploratory"
        assert section["gate"] == UNGATED

    def test_missing_instance_column_is_labeled_skip_naming_t41(self) -> None:
        series = [{"prefix_cache_queries_total": 1, "prefix_cache_hits_total": 0}]
        section = dc.execute_contrast_19([_w19("w-pd", "pd", series)])
        (skip,) = section["skips"]
        assert "instance" in skip["reason"]
        assert "T4.1" in skip["reason"]

    def test_missing_raw_fields_is_labeled_skip_naming_t42(self) -> None:
        series = [
            {"instance": "prefill-0", "gpu_prefix_cache_hit_rate": 0.5},
            {"instance": "prefill-0", "gpu_prefix_cache_hit_rate": 0.6},
        ]
        section = dc.execute_contrast_19([_w19("w-pd", "pd", series)])
        (skip,) = section["skips"]
        assert "prefix_cache_queries_total" in skip["reason"]
        assert "T4.2" in skip["reason"]

    def test_zero_delta_denominator_is_labeled_skip(self) -> None:
        series = _samples("prefill-0", 100, 40, 100, 40) + _samples(
            "decode-0", 10, 5, 60, 15
        )
        section = dc.execute_contrast_19([_w19("w-pd", "pd", series)])
        (skip,) = section["skips"]
        assert "zero-delta denominator" in skip["reason"]

    def test_counter_regression_refuses_loud(self) -> None:
        series = _samples("prefill-0", 200, 90, 100, 40)
        with pytest.raises(dc.DistContrastError, match="BACKWARDS"):
            dc.execute_contrast_19([_w19("w-pd", "pd", series)])

    def test_hits_exceeding_queries_refuses(self) -> None:
        series = _samples("prefill-0", 0, 0, 10, 20)
        with pytest.raises(dc.DistContrastError, match="unphysical"):
            dc.execute_contrast_19([_w19("w-pd", "pd", series)])

    def test_underivable_role_is_labeled_skip(self) -> None:
        series = _samples("node-1", 0, 0, 10, 5)
        section = dc.execute_contrast_19([_w19("w-pd", "pd", series)])
        (skip,) = section["skips"]
        assert "node-1" in skip["reason"]
        assert "role" in skip["reason"]

    def test_explicit_roles_mapping_overrides_derivation(self) -> None:
        pd_series = _samples("node-1", 100, 40, 200, 90) + _samples(
            "node-2", 10, 5, 60, 15
        )
        windows = [
            _w19("w-pd", "pd", pd_series),
            _w19("w-tp", "tp", _samples("node-0", 0, 0, 100, 80)),
        ]
        section = dc.execute_contrast_19(
            windows, roles={"node-1": "prefill", "node-2": "decode"}
        )
        (row,) = section["rows"]
        assert row["prefill"]["hit_rate"] == pytest.approx(0.5)

    def test_missing_role_side_is_labeled_skip(self) -> None:
        series = _samples("prefill-0", 100, 40, 200, 90)  # no decode instance
        section = dc.execute_contrast_19([_w19("w-pd", "pd", series)])
        (skip,) = section["skips"]
        assert "decode" in skip["reason"]

    def test_no_comparator_is_labeled_skip(self) -> None:
        pd_series = _samples("prefill-0", 100, 40, 200, 90) + _samples(
            "decode-0", 10, 5, 60, 15
        )
        section = dc.execute_contrast_19([_w19("w-pd", "pd", pd_series)])
        (skip,) = section["skips"]
        assert "comparator" in skip["reason"]

    def test_non_b3_baseline_is_labeled_out(self) -> None:
        section = dc.execute_contrast_19(
            [_w19("w-pd", "pd", None, baseline="B6", arm="gold-fresh")]
        )
        (skip,) = section["skips"]
        assert "B3" in skip["reason"]

    def test_missing_series_skip_names_the_source(self) -> None:
        win = _w19("w-pd", "pd", None)
        win["series_source"] = "/tree/w/cage_stats.jsonl missing"
        section = dc.execute_contrast_19([win])
        (skip,) = section["skips"]
        assert "cage_stats.jsonl missing" in skip["reason"]

    def test_duplicate_pd_on_one_key_refuses_ambiguous(self) -> None:
        s = _samples("prefill-0", 0, 0, 10, 5)
        with pytest.raises(dc.DistContrastError, match="ambiguous"):
            dc.execute_contrast_19(
                [_w19("a", "pd", s), _w19("b", "pd", s)]
            )

    def test_both_comparator_topologies_yield_two_rows(self) -> None:
        windows = self._happy_windows() + [
            _w19("w-single", "single", _samples("node-0", 0, 0, 100, 60))
        ]
        section = dc.execute_contrast_19(windows)
        assert {r["comparator_topology"] for r in section["rows"]} == {
            "tp",
            "single",
        }


# ---------------------------------------------------------------------------
# T7.2 — #14 normalized-pressure alignment on the rho_own axis
# ---------------------------------------------------------------------------


class TestPressureAlignment:
    def test_hand_computed_cross_engine_row(self) -> None:
        section = dc.align_pressure_bundles(
            [
                _aw("v1", "vllm", 0.52, yield_frac=0.6),
                _aw("v2", "vllm", 0.48, yield_frac=0.7),
                _aw("s1", "sglang", 0.55, yield_frac=0.5),
            ],
            r_levels=[0.5, 0.9],
        )
        assert section["status"] == "EXECUTED"
        (row,) = section["rows"]
        assert row["r_level"] == pytest.approx(0.5)
        assert row["engine"] == "sglang"
        assert row["anchor_engine"] == "vllm"
        assert row["mean_y_anchor"] == pytest.approx(0.65)
        assert row["mean_y_engine"] == pytest.approx(0.5)
        assert row["delta_y"] == pytest.approx(-0.15)
        assert row["basis"] == BASIS_AGGREGATE
        assert row["y_field"] == "yield_frac"
        assert row["tol"] == pytest.approx(0.10)
        assert row["n_windows_anchor"] == 2

    def test_tol_recorded_and_boundary_is_inside(self) -> None:
        # |0.6 - 0.5| == tol exactly -> INSIDE the band.
        section = dc.align_pressure_bundles(
            [
                _aw("v", "vllm", 0.6),
                _aw("s", "sglang", 0.5),
            ],
            r_levels=[0.5, 2.0],
            tol=0.10,
        )
        assert section["tol"] == pytest.approx(0.10)
        assert section["labeled_out"] == []
        assert len(section["rows"]) == 1

    def test_outside_tol_is_labeled_out_with_distances(self) -> None:
        section = dc.align_pressure_bundles(
            [_aw("far", "vllm", 0.72)], r_levels=[0.5, 0.9], tol=0.10
        )
        (out,) = section["labeled_out"]
        assert out["window"] == "far"
        assert out["nearest_r"] == pytest.approx(0.9)
        assert "tol" in out["reason"]
        assert section["rows"] == []

    def test_in_band_equidistant_tie_refuses(self) -> None:
        with pytest.raises(dc.DistContrastError, match="equidistant"):
            dc.align_pressure_bundles(
                [_aw("tie", "vllm", 0.5)], r_levels=[0.4, 0.6], tol=0.10
            )

    def test_out_of_band_equidistant_labels_out_instead(self) -> None:
        section = dc.align_pressure_bundles(
            [_aw("tie-far", "vllm", 0.7)], r_levels=[0.5, 0.9], tol=0.10
        )
        (out,) = section["labeled_out"]
        assert out["window"] == "tie-far"

    def test_missing_rho_own_is_labeled_skip_citing_8_8(self) -> None:
        section = dc.align_pressure_bundles(
            [_aw("no-own", "vllm", None)], r_levels=[0.5]
        )
        (skip,) = section["skips"]
        assert "§8.8" in skip["reason"]
        assert "own_accounting" in skip["reason"]
        assert "corroboration" in skip["reason"]

    def test_missing_metrics_is_labeled_skip(self) -> None:
        section = dc.align_pressure_bundles(
            [_aw("no-m", "vllm", 0.5, metrics=None)], r_levels=[0.5]
        )
        (skip,) = section["skips"]
        assert "WindowMetrics" in skip["reason"]

    def test_missing_anchor_is_bucket_skip(self) -> None:
        section = dc.align_pressure_bundles(
            [_aw("s", "sglang", 0.5), _aw("l", "lmdeploy", 0.5)],
            r_levels=[0.5],
        )
        (skip,) = section["bucket_skips"]
        assert "vllm" in skip["reason"]
        assert section["rows"] == []

    def test_per_gpu_basis_reads_yield_per_gpu(self) -> None:
        section = dc.align_pressure_bundles(
            [
                _aw("v", "vllm", 0.5, metrics=_wm(8.0, 6.0, 2)),
                _aw("s", "sglang", 0.5, metrics=_wm(8.0, 4.0, 2)),
            ],
            r_levels=[0.5],
            basis=BASIS_PER_GPU,
        )
        (row,) = section["rows"]
        assert row["y_field"] == "yield_per_gpu"
        assert row["mean_y_anchor"] == pytest.approx(3.0)
        assert row["mean_y_engine"] == pytest.approx(2.0)

    def test_alignment_axis_is_own_accounting_not_engine_gauge(self) -> None:
        section = dc.align_pressure_bundles([], r_levels=[0.5])
        assert "rho_own_time_avg" in section["alignment_axis"]
        assert "§8.8" in section["alignment_axis"]
        # Structural: no engine-gauge parameter exists on the function.
        import inspect

        params = inspect.signature(dc.align_pressure_bundles).parameters
        assert "rho_engine" not in params

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"r_levels": []}, "empty"),
            ({"r_levels": [0.5, 0.5]}, "duplicate"),
            ({"r_levels": [0.0]}, "> 0"),
            ({"r_levels": [0.5], "tol": 0.0}, "tol"),
            ({"r_levels": [0.5], "basis": "bogus"}, "basis"),
        ],
    )
    def test_bad_grid_inputs_refuse(self, kwargs: dict, match: str) -> None:
        with pytest.raises(dc.DistContrastError, match=match):
            dc.align_pressure_bundles([], **kwargs)


# ---------------------------------------------------------------------------
# Driver wiring — dispatch, artifacts, no NOT-IMPLEMENTED for #18/#19
# ---------------------------------------------------------------------------

_INDEX_COLUMNS = list(rca._INDEX_REQUIRED_COLUMNS)


def _index_row(**over: Any) -> dict[str, Any]:
    row = {
        "run_id": "r1",
        "campaign": "c",
        "session": "s",
        "model": MODEL,
        "engine": "vllm",
        "arm": "corpus-reuse",
        "baseline": "B3",
        "retriever": "none",
        "policy": "none",
        "topology": "single",
        "family": "F1",
        "dataset": "squad_v2",
        "budget_r": None,
        "rate_frac": None,
        "window": 1,
        "window_key": "squad_v2-01",
        "row_key": "rk",
        "window_dir": "cells/rk/window_squad_v2-01",
        "cell_json": "cells/rk/cell.json",
        "artifacts": "",
    }
    row.update(over)
    return row


def _index_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_INDEX_COLUMNS)


class TestDriverDispatch:
    def test_18_19_are_executor_backed_never_not_implemented(self) -> None:
        computable, skipped = rca.resolve_contrasts([18, 19])
        assert computable == []
        assert [s.label for s in skipped] == ["EXECUTOR-BACKED"] * 2
        for s in skipped:
            assert rca.NOT_IMPLEMENTED_LABEL not in s.label
            assert "dist_contrasts" in s.reason

    def test_classify_reason_names_the_executor(self) -> None:
        for cid in (18, 19):
            reason = rca.classify_contrast(rca.CONTRAST_BY_ID[cid])
            assert reason is not None
            assert "executor-backed" in reason
            assert rca.NOT_IMPLEMENTED_LABEL not in reason

    def test_pass_not_requested_returns_none_no_artifact(
        self, tmp_path: Path
    ) -> None:
        result = rca.run_dist_contrasts_pass(
            tmp_path,
            _index_frame([_index_row()]),
            tmp_path,
            rca.DESIGN_STAMP,
            requested_ids=[4],
            gate_13=None,
            predicate_root=None,
            blinding_active=False,
        )
        assert result is None
        assert not (tmp_path / rca.DIST_CONTRASTS_NAME).exists()

    def test_blinding_suppresses_loudly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        result = rca.run_dist_contrasts_pass(
            tmp_path,
            _index_frame([_index_row()]),
            tmp_path,
            rca.DESIGN_STAMP,
            requested_ids=[18],
            gate_13=None,
            predicate_root=None,
            blinding_active=True,
        )
        assert result is None
        assert "SUPPRESSED" in capsys.readouterr().out
        assert not (tmp_path / rca.DIST_CONTRASTS_NAME).exists()

    def test_no_dist_rows_writes_auditable_skip_artifact(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        result = rca.run_dist_contrasts_pass(
            run_dir,
            _index_frame([_index_row()]),  # F1 only — no DIST family
            analysis_dir,
            rca.DESIGN_STAMP,
            requested_ids=[18, 19],
            gate_13=None,
            predicate_root=None,
            blinding_active=False,
        )
        assert result is not None
        doc = json.loads(
            (analysis_dir / rca.DIST_CONTRASTS_NAME).read_text(encoding="utf-8")
        )
        assert doc["mode_stamp"] == rca.DESIGN_STAMP
        assert doc["requested"] == [18, 19]
        reasons = json.dumps(doc["input_skips"])
        assert "no DIST-family rows" in reasons
        assert doc["contrast_18"]["status"] == "SKIPPED-INPUTS-INCOMPLETE"
        assert doc["contrast_18"]["gate"]["status"] == "PENDING"
        assert doc["contrast_19"]["status"] == "SKIPPED-INPUTS-INCOMPLETE"
        assert "NOT-IMPLEMENTED" not in json.dumps(doc)
        assert "SKIP" in capsys.readouterr().out

    def test_18_missing_gpu_count_skip_names_verify_live_gap(
        self, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        cell = run_dir / "cells" / "rk"
        cell.mkdir(parents=True)
        (cell / "cell.json").write_text("{}", encoding="utf-8")
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        rows = [
            _index_row(
                family="DIST",
                topology=t,
                arm="gold-fresh",
                budget_r=0.5,
                rate_frac=0.9,
                window_dir=f"cells/rk/window_{t}",
            )
            for t in ("tp", "pd")
        ]
        result = rca.run_dist_contrasts_pass(
            run_dir,
            _index_frame(rows),
            analysis_dir,
            rca.DESIGN_STAMP,
            requested_ids=[18],
            gate_13=None,
            predicate_root=None,
            blinding_active=False,
        )
        assert result is not None
        skips = json.dumps(result["input_skips"])
        assert "gpu_count" in skips
        assert "VERIFY-LIVE" in skips
        assert result["contrast_18"]["status"] == "SKIPPED-INPUTS-INCOMPLETE"

    def test_19_end_to_end_from_telemetry_files(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        pd_dir = run_dir / "cells" / "pd-cell" / "window_squad_v2-01"
        tp_dir = run_dir / "cells" / "tp-cell" / "window_squad_v2-01"
        pd_dir.mkdir(parents=True)
        tp_dir.mkdir(parents=True)
        pd_series = _samples("prefill-0", 100, 40, 200, 90) + _samples(
            "decode-0", 10, 5, 60, 15
        )
        (pd_dir / "cage_stats.jsonl").write_text(
            "\n".join(json.dumps(s) for s in pd_series) + "\n", encoding="utf-8"
        )
        (tp_dir / "cage_stats.jsonl").write_text(
            "\n".join(json.dumps(s) for s in _samples("node-0", 0, 0, 100, 80))
            + "\n",
            encoding="utf-8",
        )
        rows = [
            _index_row(
                family="DIST",
                topology="pd",
                budget_r=0.5,
                rate_frac=0.9,
                row_key="pd-cell",
                window_dir="cells/pd-cell/window_squad_v2-01",
                cell_json="cells/pd-cell/cell.json",
            ),
            _index_row(
                family="F3",
                topology="tp",
                budget_r=0.5,
                rate_frac=0.9,
                row_key="tp-cell",
                window_dir="cells/tp-cell/window_squad_v2-01",
                cell_json="cells/tp-cell/cell.json",
            ),
        ]
        result = rca.run_dist_contrasts_pass(
            run_dir,
            _index_frame(rows),
            analysis_dir,
            rca.DESIGN_STAMP,
            requested_ids=[19],
            gate_13=None,
            predicate_root=None,
            blinding_active=False,
        )
        assert result is not None
        section = result["contrast_19"]
        assert section["status"] == "EXECUTED"
        (row,) = section["rows"]
        assert row["prefill"]["hit_rate"] == pytest.approx(0.5)
        assert row["decode"]["hit_rate"] == pytest.approx(0.2)
        assert row["comparator"]["hit_rate"] == pytest.approx(0.8)
        doc = json.loads(
            (analysis_dir / rca.DIST_CONTRASTS_NAME).read_text(encoding="utf-8")
        )
        assert doc["contrast_19"]["rows"]

    def test_18_end_to_end_with_complete_inputs_executes(
        self, tmp_path: Path
    ) -> None:
        """No NOT-IMPLEMENTED and no skip when EVERY #18 input exists: the
        pair computes with hand-checkable per-GPU deltas."""
        run_dir = tmp_path / "run"
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        predicate_root = tmp_path / "predicate" / "score-1"
        (run_dir / "manifest.json").parent.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {"slo_floors": {"vllm": {"ttft_s": 0.1, "tpot_s": 0.05}}}
            ),
            encoding="utf-8",
        )
        rows = []
        for topology, gpu_count in (("tp", 1), ("pd", 2)):
            cell_dir = run_dir / "cells" / f"{topology}-cell"
            wdir = cell_dir / "window_squad_v2-01"
            wdir.mkdir(parents=True)
            (cell_dir / "cell.json").write_text(
                json.dumps(
                    {
                        "gpu_count": gpu_count,
                        "windows": {
                            "squad_v2-01": {"t_start": 0.0, "t_end": 10.0}
                        },
                    }
                ),
                encoding="utf-8",
            )
            reqs = [
                {
                    "example_id": f"e{i}",
                    "ok": True,
                    "ttft_ms": 500.0,
                    "tpot_ms": 100.0,
                }
                for i in range(4)
            ]
            (wdir / "requests.jsonl").write_text(
                "\n".join(json.dumps(r) for r in reqs) + "\n", encoding="utf-8"
            )
            pdir = predicate_root / f"cells/{topology}-cell/window_squad_v2-01"
            pdir.mkdir(parents=True)
            (pdir / "predicate.jsonl").write_text(
                "\n".join(
                    json.dumps({"example_id": f"e{i}", "predicate": True})
                    for i in range(4)
                )
                + "\n",
                encoding="utf-8",
            )
            rows.append(
                _index_row(
                    family="DIST",
                    topology=topology,
                    arm="gold-fresh",
                    budget_r=0.5,
                    rate_frac=0.9,
                    row_key=f"{topology}-cell",
                    window_dir=f"cells/{topology}-cell/window_squad_v2-01",
                    cell_json=f"cells/{topology}-cell/cell.json",
                )
            )
        result = rca.run_dist_contrasts_pass(
            run_dir,
            _index_frame(rows),
            analysis_dir,
            rca.DESIGN_STAMP,
            requested_ids=[18],
            gate_13={"endpoint": "contrast-13", "passed": True},
            predicate_root=predicate_root,
            blinding_active=False,
        )
        assert result is not None
        section = result["contrast_18"]
        assert section["status"] == "EXECUTED"
        assert result["input_skips"] == []
        (pair,) = section["pairs"]
        # 4 timely+veridical requests over 10 s -> G = Y = 0.4 rps;
        # tp/1 GPU -> 0.4 per GPU, pd/2 GPUs -> 0.2 -> delta = -0.2.
        assert pair["goodput_per_gpu_tp"] == pytest.approx(0.4)
        assert pair["goodput_per_gpu_pd"] == pytest.approx(0.2)
        assert pair["delta_goodput_per_gpu"] == pytest.approx(-0.2)
        assert pair["delta_yield_per_gpu"] == pytest.approx(-0.2)
        assert section["gate"]["status"] == "OPEN"
        assert "NOT-IMPLEMENTED" not in json.dumps(result)


class TestPressureAlignmentPass:
    def _index(self) -> pd.DataFrame:
        return _index_frame(
            [
                _index_row(
                    family="F2",
                    arm="gold-fresh",
                    engine=engine,
                    budget_r=0.5,
                    rate_frac=0.9,
                    row_key=f"{engine}-cell",
                    window_dir=f"cells/{engine}-cell/window_squad_v2-01",
                )
                for engine in ("vllm", "sglang")
            ]
        )

    def _write_own(
        self, analysis_dir: Path, window: str, rho_own: float
    ) -> None:
        out = (
            analysis_dir
            / rca.OWN_ACCOUNTING_DIRNAME
            / window
            / rca.OWN_ACCOUNTING_NAME
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rho_own": rho_own}), encoding="utf-8")

    def test_rows_computed_from_own_accounting_artifacts(
        self, tmp_path: Path
    ) -> None:
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        metrics = {
            "cells/vllm-cell/window_squad_v2-01": _wm(yield_frac=0.7),
            "cells/sglang-cell/window_squad_v2-01": _wm(yield_frac=0.5),
        }
        for window in metrics:
            self._write_own(analysis_dir, window, 0.52)
        result = rca.run_pressure_alignment_pass(
            analysis_dir,
            self._index(),
            metrics,
            rca.DESIGN_STAMP,
            blinding_active=False,
        )
        assert result is not None
        assert result["mode_stamp"] == rca.DESIGN_STAMP
        assert result["tol"] == pytest.approx(rca.PRESSURE_ALIGNMENT_TOL)
        (row,) = result["rows"]
        assert row["engine"] == "sglang"
        assert row["delta_y"] == pytest.approx(0.5 - 0.7)
        doc = json.loads(
            (analysis_dir / rca.PRESSURE_ALIGNMENT_NAME).read_text(
                encoding="utf-8"
            )
        )
        assert doc["rows"]

    def test_missing_own_accounting_is_labeled_skip_citing_8_8(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        metrics = {"cells/vllm-cell/window_squad_v2-01": _wm(yield_frac=0.7)}
        index = self._index()
        result = rca.run_pressure_alignment_pass(
            analysis_dir,
            index,
            metrics,
            rca.DESIGN_STAMP,
            blinding_active=False,
        )
        assert result is not None
        (skip,) = result["skips"]
        assert "§8.8" in skip["reason"]
        assert "SKIP" in capsys.readouterr().out

    def test_no_ladder_metrics_skips_loudly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        result = rca.run_pressure_alignment_pass(
            tmp_path,
            self._index(),
            {},
            rca.DESIGN_STAMP,
            blinding_active=False,
        )
        assert result is None
        assert "SKIP" in capsys.readouterr().out
        assert not (tmp_path / rca.PRESSURE_ALIGNMENT_NAME).exists()

    def test_blinding_suppresses(self, tmp_path: Path) -> None:
        result = rca.run_pressure_alignment_pass(
            tmp_path,
            self._index(),
            {"w": _wm()},
            rca.DESIGN_STAMP,
            blinding_active=True,
        )
        assert result is None
        assert not (tmp_path / rca.PRESSURE_ALIGNMENT_NAME).exists()
