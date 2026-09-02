"""Pins the §6.6 iso-basis machinery in src.analysis.goodput (charter T7.1).

WHAT is pinned and WHY:
- gpu_count=1 default byte-identity: every pre-§6.6 WindowMetrics field keeps
  the exact name and value it had before gpu_count existed, and the per-GPU
  fields coincide with the aggregates exactly (IEEE /1) — so every existing
  caller is untouched by this feature landing.
- Basis-(b) arithmetic: gpu_count=8 divides BOTH G and Y on the rate scale and
  divides NOTHING else — aggregate currencies must stay aggregate, because
  §6.6a (iso-aggregate-bytes) figures read them across differing GPU counts.
- gpu_count validation refusals: floats (even integral), bools (np.bool_
  included — the pandas-sourced twin of True==1), strings, and counts < 1 are
  refused loudly — a fabricated or coerced GPU count would silently re-scale
  the deployment-basis numbers in contrasts #18/#19.
- assert_single_basis accept + refuse paths: the ONE seam figure code (T6.2)
  calls before pooling. Unlabeled records, foreign/tampered basis labelings,
  and records whose per-GPU and aggregate values disagree (the forgotten
  division on a gpu_count>1 window) must all raise, never be defaulted —
  the charter forbids mixing the two §6.6 bases in one figure.
- EVERY field of the requested basis is audited, *_frac included: the seam
  returns column names the caller will read verbatim, so a record missing
  goodput_frac — or carrying goodput_frac=NaN — must be refused on the
  aggregate basis even though frac fields have no per-GPU invariant to trip
  (the fail-closed hole caught in the T7.1 verification round).
- Serialization roundtrip: the new fields ride to_flat_dict/JSON and come back
  poolable, because downstream figures consume CSV/JSON rows, not live
  dataclasses.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from src.analysis.goodput import (
    BASIS_AGGREGATE,
    BASIS_PER_GPU,
    BasisRecord,
    GoodputError,
    SLOBaseline,
    WINDOW_BASES,
    WindowMetrics,
    assert_single_basis,
    evaluate_window,
)

# Same shape as the canonical test_goodput fixture: thresholds ttft <= 1.0 s,
# tpot <= 0.1 s under this baseline.
BASELINE = SLOBaseline(ttft_s=0.1, tpot_s=0.02)

# Every field WindowMetrics had before the §6.6 machinery landed; the
# byte-identity contract is defined over exactly this set.
PRE_ISO_BASIS_FIELDS: tuple[str, ...] = (
    "n_issued",
    "n_completed",
    "n_timely",
    "n_veridical",
    "n_yield",
    "duration_s",
    "attainment",
    "throughput_rps",
    "goodput_rps",
    "yield_rps",
    "goodput_frac",
    "yield_frac",
    "veridical_frac",
    "independence_null_rps",
    "independence_null_frac",
    "covariance_gap",
    "covariance_gap_rps",
    "truth_tax_rps",
    "truth_tax_frac",
)


def _window() -> pd.DataFrame:
    """10 issued: 6 timely (4 veridical), 2 completed-but-slow (1 veridical),
    2 failed (NaN latencies, non-veridical). With duration_s=10: G=0.6 rps,
    Y=0.4 rps."""
    return pd.DataFrame(
        {
            "ttft_s": [0.5] * 6 + [2.0, 2.0] + [np.nan, np.nan],
            "tpot_s": [0.05] * 6 + [0.05, 0.05] + [np.nan, np.nan],
            "ok": [True] * 8 + [False, False],
            "veridical": [True] * 4 + [False] * 2 + [True, False] + [False, False],
            "arrival_s": list(np.linspace(100.0, 109.0, 10)),
        }
    )


class TestGpuCountDefaultByteIdentity:
    def test_default_equals_explicit_gpu_count_1(self) -> None:
        default = evaluate_window(_window(), BASELINE, duration_s=10.0)
        explicit = evaluate_window(_window(), BASELINE, duration_s=10.0, gpu_count=1)
        assert default == explicit

    def test_pre_iso_basis_fields_keep_exact_values(self) -> None:
        # The exact currencies the pre-§6.6 module produced on this window;
        # equality is ==, not approx — byte-identity, not closeness.
        m = evaluate_window(_window(), BASELINE, duration_s=10.0)
        assert m.n_issued == 10
        assert m.n_completed == 8
        assert m.n_timely == 6
        assert m.n_yield == 4
        assert m.goodput_rps == 6 / 10.0
        assert m.yield_rps == 4 / 10.0
        assert m.goodput_frac == 6 / 10
        assert m.yield_frac == 4 / 10
        assert m.truth_tax_rps == 6 / 10.0 - 4 / 10.0
        assert m.truth_tax_frac == 6 / 10 - 4 / 10

    def test_per_gpu_coincides_with_aggregate_at_one_gpu(self) -> None:
        m = evaluate_window(_window(), BASELINE, duration_s=10.0)
        assert m.gpu_count == 1
        # IEEE division by 1 is exact: no approx needed or wanted.
        assert m.goodput_per_gpu == m.goodput_rps
        assert m.yield_per_gpu == m.yield_rps
        assert m.bases == WINDOW_BASES


class TestPerGpuArithmetic:
    def test_eight_gpus_divide_both_g_and_y(self) -> None:
        m = evaluate_window(_window(), BASELINE, duration_s=10.0, gpu_count=8)
        assert m.gpu_count == 8
        assert m.goodput_per_gpu == m.goodput_rps / 8
        assert m.yield_per_gpu == m.yield_rps / 8
        assert m.goodput_per_gpu == pytest.approx(0.075)
        assert m.yield_per_gpu == pytest.approx(0.05)

    def test_aggregates_are_never_divided(self) -> None:
        # §6.6a reads raw aggregate G/Y across differing GPU counts; gpu_count
        # must touch ONLY the per-GPU fields.
        one = evaluate_window(_window(), BASELINE, duration_s=10.0, gpu_count=1)
        eight = evaluate_window(_window(), BASELINE, duration_s=10.0, gpu_count=8)
        for name in PRE_ISO_BASIS_FIELDS:
            assert getattr(eight, name) == getattr(one, name), name

    def test_numpy_integer_gpu_count_accepted_and_coerced(self) -> None:
        # Cell configs come through pandas; np.int64 counts must not refuse.
        m = evaluate_window(
            _window(), BASELINE, duration_s=10.0, gpu_count=np.int64(8)
        )
        assert m.gpu_count == 8
        assert type(m.gpu_count) is int


class TestGpuCountValidation:
    @pytest.mark.parametrize(
        "gpu_count",
        [0, -3, 2.5, 8.0, True, False, "8", None, math.nan, np.bool_(True)],
        ids=[
            "zero",
            "negative",
            "fractional",
            "integral-float",
            "true",
            "false",
            "string",
            "none",
            "nan",
            "np-bool-true",
        ],
    )
    def test_invalid_gpu_count_refused(self, gpu_count: object) -> None:
        # 8.0 is refused despite being integral: a float GPU count is always
        # an upstream bookkeeping bug, and True==1 must not pass as 1 GPU —
        # np.bool_(True) included: it is not a bool subclass yet implements
        # __index__, so operator.index alone would accept it as a 1-GPU cell.
        with pytest.raises(GoodputError, match="gpu_count"):
            evaluate_window(
                _window(), BASELINE, duration_s=10.0, gpu_count=gpu_count  # type: ignore[arg-type]
            )


def _metrics(gpu_count: int) -> WindowMetrics:
    return evaluate_window(
        _window(), BASELINE, duration_s=10.0, gpu_count=gpu_count
    )


class TestAssertSingleBasis:
    def test_accepts_windowmetrics_and_returns_basis_fields(self) -> None:
        pool = [_metrics(1), _metrics(8)]
        assert assert_single_basis(pool, BASIS_AGGREGATE) == (
            "goodput_rps",
            "yield_rps",
            "goodput_frac",
            "yield_frac",
        )
        assert assert_single_basis(pool, BASIS_PER_GPU) == (
            "goodput_per_gpu",
            "yield_per_gpu",
        )

    def test_returned_fields_exist_on_windowmetrics(self) -> None:
        # Guards drift between the basis record and the dataclass: the seam's
        # answer must name real columns, or T6.2 reads nothing.
        m = _metrics(8)
        for basis in (BASIS_AGGREGATE, BASIS_PER_GPU):
            for name in assert_single_basis([m], basis):
                assert hasattr(m, name), name

    def test_accepts_flat_dict_form(self) -> None:
        pool = [_metrics(1).to_flat_dict(), _metrics(8).to_flat_dict()]
        assert assert_single_basis(pool, BASIS_PER_GPU) == (
            "goodput_per_gpu",
            "yield_per_gpu",
        )

    def test_unknown_basis_label_refused(self) -> None:
        # "per_gpu" (identifier spelling) is NOT the machine label; refusing
        # it here is what keeps figure code on the exported constants.
        for bad in ("per_gpu", "PER-GPU", "aggregate-bytes", ""):
            with pytest.raises(GoodputError, match="basis"):
                assert_single_basis([_metrics(1)], bad)

    def test_empty_pool_refused(self) -> None:
        with pytest.raises(GoodputError, match="empty"):
            assert_single_basis([], BASIS_PER_GPU)

    def test_unlabeled_record_refused(self) -> None:
        # A bare numbers dict has no bases declaration: never poolable, never
        # defaulted to a basis.
        flat = _metrics(8).to_flat_dict()
        del flat["bases"]
        with pytest.raises(GoodputError, match="bases"):
            assert_single_basis([flat], BASIS_PER_GPU)

    def test_missing_gpu_count_refused(self) -> None:
        flat = _metrics(8).to_flat_dict()
        del flat["gpu_count"]
        with pytest.raises(GoodputError, match="gpu_count"):
            assert_single_basis([flat], BASIS_PER_GPU)

    def test_legacy_record_without_per_gpu_fields_refused(self) -> None:
        # Pre-§6.6 CSV rows must be rebuilt, not silently promoted: the seam
        # never derives a missing per-GPU value on the caller's behalf.
        flat = _metrics(8).to_flat_dict()
        del flat["goodput_per_gpu"]
        with pytest.raises(GoodputError, match="goodput_per_gpu"):
            assert_single_basis([flat], BASIS_PER_GPU)

    def test_tampered_basis_labeling_refused(self) -> None:
        flat = _metrics(8).to_flat_dict()
        flat["bases"] = {
            "aggregate": ("goodput_per_gpu", "yield_per_gpu"),
            "per_gpu": ("goodput_rps", "yield_rps", "goodput_frac", "yield_frac"),
        }
        with pytest.raises(GoodputError, match="different basis labeling"):
            assert_single_basis([flat], BASIS_PER_GPU)

    def test_non_record_bases_refused(self) -> None:
        flat = _metrics(8).to_flat_dict()
        flat["bases"] = "per-gpu"
        with pytest.raises(GoodputError, match="BasisRecord"):
            assert_single_basis([flat], BASIS_PER_GPU)

    def test_forgotten_division_refused(self) -> None:
        # THE mixed-basis signature: gpu_count>1 but per-GPU == aggregate
        # (someone pooled an undivided value into a basis-b column). Only
        # detectable at gpu_count>1, where the two bases genuinely differ.
        flat = _metrics(8).to_flat_dict()
        flat["goodput_per_gpu"] = flat["goodput_rps"]
        with pytest.raises(GoodputError, match="mixes bases"):
            assert_single_basis([flat], BASIS_PER_GPU)

    def test_one_bad_item_poisons_the_pool(self) -> None:
        bad = _metrics(8).to_flat_dict()
        bad["yield_per_gpu"] = bad["yield_rps"]
        with pytest.raises(GoodputError, match="mixes bases"):
            assert_single_basis([_metrics(1), _metrics(8), bad], BASIS_AGGREGATE)

    def test_non_numeric_and_nonfinite_values_refused(self) -> None:
        nan_flat = _metrics(8).to_flat_dict()
        nan_flat["goodput_rps"] = math.nan
        with pytest.raises(GoodputError, match="finite"):
            assert_single_basis([nan_flat], BASIS_AGGREGATE)
        str_flat = _metrics(8).to_flat_dict()
        str_flat["yield_per_gpu"] = "0.05"
        with pytest.raises(GoodputError, match="number"):
            assert_single_basis([str_flat], BASIS_AGGREGATE)

    def test_bool_gpu_count_in_record_refused(self) -> None:
        flat = _metrics(1).to_flat_dict()
        flat["gpu_count"] = True
        with pytest.raises(GoodputError, match="gpu_count"):
            assert_single_basis([flat], BASIS_AGGREGATE)

    def test_np_bool_gpu_count_in_record_refused(self) -> None:
        # The numpy twin of the True==1 hazard: pandas .item()-less reads hand
        # back np.bool_, which operator.index would happily treat as 1 GPU.
        flat = _metrics(1).to_flat_dict()
        flat["gpu_count"] = np.bool_(True)
        with pytest.raises(GoodputError, match="gpu_count"):
            assert_single_basis([flat], BASIS_AGGREGATE)

    def test_missing_frac_field_refused_on_aggregate_basis(self) -> None:
        # The verification-round hole: goodput_frac is half of the returned
        # aggregate column tuple, yet it has no per-GPU invariant to trip.
        # Deleting it must refuse — otherwise the caller is handed a column
        # name that does not exist in the pooled records.
        flat = _metrics(8).to_flat_dict()
        del flat["goodput_frac"]
        with pytest.raises(GoodputError, match="goodput_frac"):
            assert_single_basis([flat], BASIS_AGGREGATE)

    def test_nan_frac_field_refused_on_aggregate_basis(self) -> None:
        flat = _metrics(8).to_flat_dict()
        flat["goodput_frac"] = math.nan
        with pytest.raises(GoodputError, match="finite"):
            assert_single_basis([flat], BASIS_AGGREGATE)

    def test_string_frac_field_refused_on_aggregate_basis(self) -> None:
        flat = _metrics(8).to_flat_dict()
        flat["yield_frac"] = "0.4"
        with pytest.raises(GoodputError, match="number"):
            assert_single_basis([flat], BASIS_AGGREGATE)

    def test_broken_frac_field_ignored_on_per_gpu_basis(self) -> None:
        # Contract boundary pin: the audit covers the REQUESTED basis tuple
        # (plus the rate-pair invariant). frac fields are not part of basis b,
        # so a per-gpu pool does not read them — refusing here would demand
        # columns the figure never touches.
        flat = _metrics(8).to_flat_dict()
        del flat["goodput_frac"]
        assert assert_single_basis([flat], BASIS_PER_GPU) == (
            "goodput_per_gpu",
            "yield_per_gpu",
        )

    def test_non_record_item_refused(self) -> None:
        with pytest.raises(GoodputError, match="bases"):
            assert_single_basis([0.6], BASIS_AGGREGATE)


class TestSerialization:
    def test_flat_dict_carries_the_new_fields(self) -> None:
        m = _metrics(8)
        flat = m.to_flat_dict()
        assert flat["gpu_count"] == 8
        assert flat["goodput_per_gpu"] == m.goodput_per_gpu
        assert flat["yield_per_gpu"] == m.yield_per_gpu
        assert flat["bases"] == asdict(WINDOW_BASES)
        # One key per dataclass field, new fields included (the CSV contract).
        assert set(flat) == {
            f.name for f in WindowMetrics.__dataclass_fields__.values()
        }

    def test_flat_dict_reconstruction_roundtrip(self) -> None:
        m = _metrics(8)
        flat = m.to_flat_dict()
        rebuilt = WindowMetrics(**{**flat, "bases": BasisRecord(**flat["bases"])})
        assert rebuilt == m

    def test_json_roundtrip_stays_poolable(self) -> None:
        # Downstream figures consume JSON/CSV rows: tuples come back as lists,
        # and the seam must still accept the record on either basis.
        loaded = json.loads(json.dumps(_metrics(8).to_flat_dict()))
        assert loaded["gpu_count"] == 8
        assert loaded["goodput_per_gpu"] == pytest.approx(0.075)
        assert assert_single_basis([loaded], BASIS_PER_GPU) == (
            "goodput_per_gpu",
            "yield_per_gpu",
        )
        assert assert_single_basis([loaded], BASIS_AGGREGATE) == (
            "goodput_rps",
            "yield_rps",
            "goodput_frac",
            "yield_frac",
        )
