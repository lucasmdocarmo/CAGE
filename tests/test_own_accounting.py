"""Tests for src.analysis.own_accounting (T2.5/T8.2, audit §2.6/§8.8) + its
run_campaign_analysis own-accounting pass wiring.

WHAT is pinned and WHY:

- ρ_own byte integrals against EXACT hand-computed values (qwen3-14b's
  163_840 B/tok for round numbers): the pressure referee's Layer-1 quantity
  must come from CAGE's OWN accounting of the recorded request intervals —
  the piecewise-linear closed form (prompt held from send, linear decode
  growth between first-token and completion) is the registered
  approximation, so its arithmetic is pinned to the token-second, including
  window clipping, the per-segment series, the hybrid fixed-state term
  (qwen3-next-80b) and the fp8 dtype factor.
- Fail-closed refusal paths: absent/None required fields NAME the field and
  the request (skip-and-underestimate is the exact bug class this module
  replaces); empty request sets refuse (absence of rows is not zero
  occupancy); unknown model/kv_dtype propagate cache_budget's
  CacheBudgetError (arithmetic imported, never copied). The ONE legal
  exclusion — dropped_by_cap rows, provably zero engine bytes — is counted,
  never silent.
- divergence_report None-honesty: an absent engine gauge yields gap=None,
  NEVER 0 (a zero gap would certify agreement that was never measured), and
  NaN refuses (an upstream None-honesty bug must surface, not launder).
- ρ_reuse arithmetic incl. the zero-prefix and full-prefix bounds, mapping
  and scalar modes, token-weighted window aggregation, and the engine
  cached_prompt_tokens CORROBORATION column (None-honest when unreported —
  the engine self-report is demoted, never the primary).
- Wiring: run_own_accounting_pass writes own_accounting.json per window in a
  tmp campaign tree (MIRRORED under analysis/own_accounting/ — the sealed
  raw tree is never written into), computes rho_own/rho_engine/gaps/
  rho_reuse_own/cached corroboration when the inputs exist, and emits LOUD
  per-leg skip lines (naming the [WAVE-3] budget-plan gap) instead of
  fabricating anything when they do not.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.own_accounting import (
    OwnAccountingError,
    OwnOccupancyWindow,
    OwnReuseWindow,
    compute_own_occupancy,
    compute_own_reuse,
    divergence_report,
    window_bounds_from_requests,
)
from src.orchestration.cache_budget import MODEL_KV, CacheBudgetError

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_campaign_analysis as rca  # noqa: E402

# qwen3-14b: 163_840 B/tok (BF16), no fixed per-seq state — round numbers.
KV = MODEL_KV["qwen3-14b"].kv_bytes_per_token
assert KV == 163_840
#: Budget = 42 tokens' worth, so the canonical fixture lands on rho = 0.5.
BUDGET_42_TOK = 42 * KV  # 6_881_280


def _row(
    example_id: str,
    *,
    send: float | None,
    ft: float | None,
    end: float | None,
    prompt: int | None,
    num: int | None,
    group: int | None = None,
    cached: int | None = None,
    dropped: bool = False,
) -> dict:
    return {
        "example_id": example_id,
        "record_index": None,
        "actual_send_ts": send,
        "first_token_ts": ft,
        "completion_ts": end,
        "prompt_tokens": prompt,
        "num_tokens": num,
        "group_id": group,
        "cached_prompt_tokens": cached,
        "dropped_by_cap": dropped,
    }


def _canonical() -> list[dict]:
    """Window [0, 4), hand-computed to 84 token-seconds:

    r1: prompt=10 held over [0,4) -> 40; decode 4 tokens linear over [2,4)
        -> +4  (N/(end-ft) * (b-ft)^2/2 = 4/2 * 4/2)          => 44 tok*s
    r2: prompt=20 held over [1,3), no decode                   => 40 tok*s
    total 84 tok*s / 4 s = 21 tokens avg; budget 42 tokens -> rho 0.5.
    Peak: just before t=3, 30 + 2*(3-2) + ... = 32 tokens.
    """
    return [
        _row("r1", send=0.0, ft=2.0, end=4.0, prompt=10, num=4, group=0, cached=4),
        _row("r2", send=1.0, ft=None, end=3.0, prompt=20, num=0, group=1, cached=None),
    ]


# ---------------------------------------------------------------------------
# T2.5 compute_own_occupancy — exact byte integrals
# ---------------------------------------------------------------------------


class TestOwnOccupancy:
    def test_canonical_exact_integral(self):
        occ = compute_own_occupancy(
            _canonical(),
            model="qwen3-14b",
            budget_bytes=BUDGET_42_TOK,
            window_start_s=0.0,
            window_end_s=4.0,
        )
        assert isinstance(occ, OwnOccupancyWindow)
        assert occ.byte_seconds == pytest.approx(84 * KV)  # 13_762_560
        assert occ.bytes_time_avg == pytest.approx(21 * KV)  # 3_440_640
        assert occ.rho_own_time_avg == pytest.approx(0.5)
        assert occ.peak_bytes == pytest.approx(32 * KV)  # 5_242_880
        assert occ.peak_rho == pytest.approx(32 / 42)
        assert occ.n_requests == 2
        assert occ.n_in_flight == 2
        assert occ.n_dropped_excluded == 0
        assert occ.kv_bytes_per_token_effective == KV
        assert occ.fixed_state_bytes_per_seq == 0

    def test_canonical_series_segments(self):
        """The per-segment series is exact: breakpoints {0,1,2,3,4}, segment
        token means 10/30/31/13, and the segment integrals re-sum to the
        window integral (regime tooling can consume it without residue)."""
        occ = compute_own_occupancy(
            _canonical(),
            model="qwen3-14b",
            budget_bytes=BUDGET_42_TOK,
            window_start_s=0.0,
            window_end_s=4.0,
        )
        assert [(e["t0_s"], e["t1_s"]) for e in occ.series] == [
            (0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0),
        ]
        means = [e["bytes_time_avg"] for e in occ.series]
        assert means == pytest.approx([10 * KV, 30 * KV, 31 * KV, 13 * KV])
        # Piecewise-linear endpoint values: [2,3) runs 30 -> 32 tokens.
        seg = occ.series[2]
        assert seg["bytes_t0"] == pytest.approx(30 * KV)
        assert seg["bytes_t1"] == pytest.approx(32 * KV)
        assert seg["rho_time_avg"] == pytest.approx(31 / 42)
        resum = sum(e["bytes_time_avg"] * (e["t1_s"] - e["t0_s"]) for e in occ.series)
        assert resum == pytest.approx(occ.byte_seconds)

    def test_window_clipping_partial(self):
        """Clip to [1, 3): r1 contributes 10 over [1,2) + 11 over [2,3)
        (10*1 + 4/2 * 1^2/2), r2 contributes 40 -> 61 tok*s over 2 s."""
        occ = compute_own_occupancy(
            _canonical(),
            model="qwen3-14b",
            budget_bytes=BUDGET_42_TOK,
            window_start_s=1.0,
            window_end_s=3.0,
        )
        assert occ.byte_seconds == pytest.approx(61 * KV)
        assert occ.bytes_time_avg == pytest.approx(30.5 * KV)
        assert occ.rho_own_time_avg == pytest.approx(61 / 84)

    def test_out_of_window_request_is_affirmatively_zero(self):
        """A fully-recorded interval OUTSIDE the window is affirmative data
        (held nothing in-window) — excluded from the integral, still counted
        in n_requests; the honest 0 contribution is not a fabricated one."""
        rows = _canonical() + [
            _row("r3", send=10.0, ft=None, end=12.0, prompt=1000, num=0)
        ]
        occ = compute_own_occupancy(
            rows,
            model="qwen3-14b",
            budget_bytes=BUDGET_42_TOK,
            window_start_s=0.0,
            window_end_s=4.0,
        )
        assert occ.rho_own_time_avg == pytest.approx(0.5)
        assert occ.n_requests == 3
        assert occ.n_in_flight == 2

    def test_fp8_dtype_halves_bytes(self):
        occ = compute_own_occupancy(
            _canonical(),
            model="qwen3-14b",
            budget_bytes=BUDGET_42_TOK,
            window_start_s=0.0,
            window_end_s=4.0,
            kv_dtype="fp8",
        )
        assert occ.kv_bytes_per_token_effective == KV // 2
        assert occ.rho_own_time_avg == pytest.approx(0.25)

    def test_hybrid_fixed_state_term(self):
        """qwen3-next-80b holds a 38 MiB recurrent slot PER SEQUENCE on top
        of its 24_576 B/tok KV — the slot must ride the full in-flight
        interval: 100 tok * 2 s * 24576 + 2 s * 38 MiB = 84_606_976 B*s."""
        mk = MODEL_KV["qwen3-next-80b"]
        rows = [_row("h1", send=0.0, ft=None, end=2.0, prompt=100, num=0)]
        occ = compute_own_occupancy(
            rows,
            model="qwen3-next-80b",
            budget_bytes=84_606_976,
            window_start_s=0.0,
            window_end_s=2.0,
        )
        expected = 200 * mk.kv_bytes_per_token + 2 * mk.fixed_state_bytes_per_seq
        assert occ.byte_seconds == pytest.approx(expected)
        assert occ.rho_own_time_avg == pytest.approx(0.5)
        assert occ.fixed_state_bytes_per_seq == 38 * 1024 * 1024

    def test_dropped_by_cap_excluded_and_counted(self):
        """dropped_by_cap is the ONE legal exclusion (never sent -> zero
        engine bytes by construction) — and it is COUNTED, never silent."""
        rows = _canonical() + [
            _row("dropped", send=None, ft=None, end=None, prompt=None,
                 num=None, dropped=True)
        ]
        occ = compute_own_occupancy(
            rows,
            model="qwen3-14b",
            budget_bytes=BUDGET_42_TOK,
            window_start_s=0.0,
            window_end_s=4.0,
        )
        assert occ.rho_own_time_avg == pytest.approx(0.5)
        assert occ.n_dropped_excluded == 1


class TestOwnOccupancyRefusals:
    def _run(self, rows, **overrides):
        kwargs = dict(
            model="qwen3-14b",
            budget_bytes=BUDGET_42_TOK,
            window_start_s=0.0,
            window_end_s=4.0,
        )
        kwargs.update(overrides)
        return compute_own_occupancy(rows, **kwargs)

    def test_empty_requests_refuse(self):
        with pytest.raises(OwnAccountingError, match="empty"):
            self._run([])

    def test_missing_completion_names_field_and_request(self):
        rows = [_row("broken", send=0.0, ft=1.0, end=None, prompt=10, num=2)]
        with pytest.raises(OwnAccountingError) as exc:
            self._run(rows)
        assert "completion_ts" in str(exc.value)
        assert "broken" in str(exc.value)

    def test_missing_first_token_with_decode_names_field(self):
        rows = [_row("nostream", send=0.0, ft=None, end=2.0, prompt=10, num=5)]
        with pytest.raises(OwnAccountingError, match="first_token_ts"):
            self._run(rows)

    def test_missing_prompt_tokens_names_field(self):
        rows = [_row("notok", send=0.0, ft=None, end=2.0, prompt=None, num=0)]
        with pytest.raises(OwnAccountingError, match="prompt_tokens"):
            self._run(rows)

    def test_out_of_window_row_still_validated(self):
        """Field validation precedes window membership: a row that LOOKS
        out-of-window cannot buy itself a pass on absent fields (membership
        of an unreconstructable interval is undecidable)."""
        rows = _canonical() + [
            _row("outside", send=10.0, ft=None, end=12.0, prompt=None, num=0)
        ]
        with pytest.raises(OwnAccountingError, match="prompt_tokens"):
            self._run(rows)

    def test_completion_before_send_refuses(self):
        rows = [_row("rev", send=5.0, ft=None, end=1.0, prompt=10, num=0)]
        with pytest.raises(OwnAccountingError, match="unreconstructable"):
            self._run(rows)

    def test_first_token_outside_interval_refuses(self):
        rows = [_row("ft-out", send=1.0, ft=0.5, end=2.0, prompt=10, num=2)]
        with pytest.raises(OwnAccountingError, match="first_token_ts"):
            self._run(rows)

    def test_unknown_model_propagates_cache_budget_error(self):
        with pytest.raises(CacheBudgetError, match="unknown model"):
            self._run(_canonical(), model="gpt-oss-9000")

    def test_unknown_kv_dtype_propagates_cache_budget_error(self):
        with pytest.raises(CacheBudgetError, match="unknown kv_dtype"):
            self._run(_canonical(), kv_dtype="int4")

    def test_nonpositive_budget_refuses(self):
        with pytest.raises(OwnAccountingError, match="budget_bytes"):
            self._run(_canonical(), budget_bytes=0)

    def test_budget_must_be_int(self):
        with pytest.raises(OwnAccountingError, match="budget_bytes"):
            self._run(_canonical(), budget_bytes=1e9)

    def test_bad_window_bounds_refuse(self):
        with pytest.raises(OwnAccountingError, match="window_end_s"):
            self._run(_canonical(), window_start_s=4.0, window_end_s=4.0)


class TestWindowBounds:
    def test_bounds_from_canonical(self):
        assert window_bounds_from_requests(_canonical()) == (0.0, 4.0)

    def test_all_dropped_refuses(self):
        rows = [_row("d", send=None, ft=None, end=None, prompt=None, num=None,
                     dropped=True)]
        with pytest.raises(OwnAccountingError, match="dropped_by_cap"):
            window_bounds_from_requests(rows)

    def test_zero_width_refuses(self):
        rows = [_row("z", send=1.0, ft=None, end=1.0, prompt=5, num=0)]
        with pytest.raises(OwnAccountingError, match="zero-width"):
            window_bounds_from_requests(rows)


# ---------------------------------------------------------------------------
# divergence_report — None-honesty
# ---------------------------------------------------------------------------


class TestDivergenceReport:
    def test_gap_computed(self):
        report = divergence_report(0.5, 0.4)
        assert report["rho_own"] == pytest.approx(0.5)
        assert report["rho_engine"] == pytest.approx(0.4)
        assert report["abs_gap"] == pytest.approx(0.1)
        assert report["rel_gap"] == pytest.approx(0.2)

    def test_absent_engine_gauge_is_none_never_zero(self):
        report = divergence_report(0.5, None)
        assert report == {
            "rho_own": 0.5,
            "rho_engine": None,
            "abs_gap": None,
            "rel_gap": None,
        }

    def test_zero_primary_rel_gap_is_none_not_inf(self):
        report = divergence_report(0.0, 0.3)
        assert report["abs_gap"] == pytest.approx(0.3)
        assert report["rel_gap"] is None

    def test_nan_engine_gauge_refuses(self):
        with pytest.raises(OwnAccountingError, match="rho_engine"):
            divergence_report(0.5, math.nan)

    def test_non_finite_primary_refuses(self):
        with pytest.raises(OwnAccountingError, match="rho_own"):
            divergence_report(math.nan, 0.5)
        with pytest.raises(OwnAccountingError, match="rho_own"):
            divergence_report(True, 0.5)  # bool is not a rho


# ---------------------------------------------------------------------------
# T8.2 compute_own_reuse — manifest primary, engine corroboration
# ---------------------------------------------------------------------------


def _reuse_rows() -> list[dict]:
    return [
        _row("a", send=0.0, ft=None, end=1.0, prompt=100, num=0, group=0, cached=30),
        _row("b", send=0.0, ft=None, end=1.0, prompt=50, num=0, group=0, cached=None),
        _row("c", send=0.0, ft=None, end=1.0, prompt=80, num=0, group=1, cached=0),
    ]


class TestOwnReuse:
    def test_mapping_mode_token_weighted(self):
        reuse = compute_own_reuse(_reuse_rows(), {0: 50, 1: 0})
        assert isinstance(reuse, OwnReuseWindow)
        fracs = [r["reuse_frac"] for r in reuse.per_request]
        # per-request incl. the FULL-prefix bound (b: 50/50) and the
        # ZERO-prefix bound (c: 0/80).
        assert fracs == pytest.approx([0.5, 1.0, 0.0])
        assert reuse.total_prompt_tokens == 230
        assert reuse.total_shared_prefix_tokens == 100
        assert reuse.rho_reuse_own == pytest.approx(100 / 230)

    def test_cached_corroboration_none_honest(self):
        """The engine cached_prompt_tokens column is CORROBORATION: weighted
        over reporting rows only (a missing report never drags toward 0),
        and None when nothing reported."""
        reuse = compute_own_reuse(_reuse_rows(), {0: 50, 1: 0})
        corr = reuse.cached_tokens_corroboration
        assert corr["n_reported"] == 2
        assert corr["n_missing"] == 1
        assert corr["token_weighted_mean"] == pytest.approx(30 / 180)

        rows = [_row("x", send=0.0, ft=None, end=1.0, prompt=10, num=0,
                     group=0, cached=None)]
        reuse2 = compute_own_reuse(rows, {0: 10})
        assert reuse2.cached_tokens_corroboration["token_weighted_mean"] is None
        assert reuse2.cached_tokens_corroboration["n_reported"] == 0

    def test_scalar_mode(self):
        rows = _reuse_rows()[:2]  # prompts 100 and 50
        reuse = compute_own_reuse(rows, 50)
        assert reuse.rho_reuse_own == pytest.approx(100 / 150)
        assert reuse.per_request[1]["reuse_frac"] == pytest.approx(1.0)

    def test_zero_scalar_is_honest_zero(self):
        reuse = compute_own_reuse(_reuse_rows()[:1], 0)
        assert reuse.rho_reuse_own == 0.0


class TestOwnReuseRefusals:
    def test_empty_requests_refuse(self):
        with pytest.raises(OwnAccountingError, match="empty"):
            compute_own_reuse([], {0: 10})

    def test_missing_prompt_tokens_names_field(self):
        rows = [_row("np", send=0.0, ft=None, end=1.0, prompt=None, num=0, group=0)]
        with pytest.raises(OwnAccountingError, match="prompt_tokens"):
            compute_own_reuse(rows, {0: 5})

    def test_zero_prompt_refuses(self):
        rows = [_row("zp", send=0.0, ft=None, end=1.0, prompt=0, num=0, group=0)]
        with pytest.raises(OwnAccountingError, match=">= 1"):
            compute_own_reuse(rows, {0: 0})

    def test_missing_group_names_field(self):
        rows = [_row("ng", send=0.0, ft=None, end=1.0, prompt=10, num=0, group=None)]
        with pytest.raises(OwnAccountingError, match="group_id"):
            compute_own_reuse(rows, {0: 5})

    def test_unknown_group_refuses_never_skips(self):
        rows = [_row("ug", send=0.0, ft=None, end=1.0, prompt=10, num=0, group=7)]
        with pytest.raises(
            OwnAccountingError, match="no manifest shared-prefix entry"
        ):
            compute_own_reuse(rows, {0: 5})

    def test_prefix_exceeding_prompt_refuses(self):
        """A reuse fraction > 1 is unphysical — it signals manifest/engine
        tokenizer divergence and must SURFACE, never clamp."""
        rows = [_row("ovr", send=0.0, ft=None, end=1.0, prompt=50, num=0, group=0)]
        with pytest.raises(OwnAccountingError, match="unphysical"):
            compute_own_reuse(rows, {0: 60})

    def test_bad_cached_tokens_refuse(self):
        rows = [_row("bc", send=0.0, ft=None, end=1.0, prompt=10, num=0,
                     group=0, cached=-3)]
        with pytest.raises(OwnAccountingError, match="cached_prompt_tokens"):
            compute_own_reuse(rows, {0: 5})

    def test_negative_scalar_refuses(self):
        with pytest.raises(OwnAccountingError, match=">= 0"):
            compute_own_reuse(_reuse_rows(), -1)

    def test_wrong_type_prefix_input_refuses(self):
        with pytest.raises(OwnAccountingError, match="mapping"):
            compute_own_reuse(_reuse_rows(), "50")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Wiring: run_campaign_analysis.run_own_accounting_pass in a tmp tree
# ---------------------------------------------------------------------------


def _write_window(
    run_dir: Path,
    *,
    with_budget_plan: bool,
    with_regime: bool,
) -> pd.DataFrame:
    """Minimal v2 tree: one cell, one window, canonical request rows."""
    row_key = "rk1"
    window_rel = f"cells/{row_key}/window_squad_v2-01"
    window_dir = run_dir / window_rel
    window_dir.mkdir(parents=True)
    with (window_dir / "requests.jsonl").open("w", encoding="utf-8") as fh:
        for row in _canonical():
            fh.write(json.dumps(row) + "\n")
    cell_meta: dict = {"cellspec": {}, "windows": {}}
    if with_budget_plan:
        cell_meta["budget_plan"] = {
            "budget_bytes_total": BUDGET_42_TOK,
            "kv_dtype": "bf16",
        }
    cell_json = run_dir / f"cells/{row_key}/cell.json"
    cell_json.write_text(json.dumps(cell_meta), encoding="utf-8")
    if with_regime:
        (window_dir / "regime.json").write_text(
            json.dumps(
                {
                    "telemetry_ok": True,
                    "inputs": {"rho_kv_time_avg": 0.4},
                    "label": None,
                    "refusal_reason": None,
                }
            ),
            encoding="utf-8",
        )
    return pd.DataFrame(
        [
            {
                "row_key": row_key,
                "dataset": "squad_v2",
                "window_key": "squad_v2-01",
                "window_dir": window_rel,
                "cell_json": f"cells/{row_key}/cell.json",
                "model": "qwen3-14b",
            }
        ]
    )


def _write_manifest(path: Path, *, dataset: str = "squad_v2") -> Path:
    # Block prefix lengths chosen against the canonical prompts (10 and 20):
    # r1 (group 0) -> 5/10, r2 (group 1) -> 0/20 => rho_reuse = 5/30.
    path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "blocks": [
                    {"block_id": 0, "token_count": 5},
                    {"block_id": 1, "token_count": 0},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


class TestWiring:
    def test_full_artifact_in_tmp_tree(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        index = _write_window(run_dir, with_budget_plan=True, with_regime=True)
        analysis_dir = tmp_path / "run" / "analysis" / "t0"
        manifest = _write_manifest(tmp_path / "manifest.json")

        summary = rca.run_own_accounting_pass(
            run_dir, index, analysis_dir, workload_manifest=manifest
        )
        assert summary["n_emitted"] == 1
        assert summary["n_skipped_entirely"] == 0

        out = (
            analysis_dir
            / "own_accounting"
            / "cells/rk1/window_squad_v2-01"
            / "own_accounting.json"
        )
        assert out.is_file(), "artifact mirrors the window path"
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["rho_own"] == pytest.approx(0.5)
        assert doc["rho_engine"] == pytest.approx(0.4)
        assert doc["gaps"]["abs_gap"] == pytest.approx(0.1)
        assert doc["rho_reuse_own"] == pytest.approx(5 / 30)
        corr = doc["cached_tokens_corroboration"]
        assert corr["n_reported"] == 1  # only r1 reported cached tokens
        assert corr["n_missing"] == 1
        assert corr["token_weighted_mean"] == pytest.approx(4 / 10)
        # The raw tree stays untouched: nothing new under the window dir.
        assert not (run_dir / "cells/rk1/window_squad_v2-01" / "own_accounting.json").exists()
        assert (analysis_dir / "own_accounting" / "summary.json").is_file()

    def test_missing_inputs_loud_skip_no_artifact(self, tmp_path: Path, capsys):
        """No budget plan ([WAVE-3] gap) and no manifest: both legs skip
        LOUDLY and no artifact is fabricated."""
        run_dir = tmp_path / "run"
        index = _write_window(run_dir, with_budget_plan=False, with_regime=False)
        analysis_dir = tmp_path / "run" / "analysis" / "t0"

        summary = rca.run_own_accounting_pass(run_dir, index, analysis_dir)
        assert summary["n_emitted"] == 0
        assert summary["n_skipped_entirely"] == 1
        assert not (analysis_dir / "own_accounting" / "cells").exists()
        out = capsys.readouterr().out
        assert "[own-accounting] SKIP" in out
        assert "WAVE-3" in out  # the budget-plan gap is NAMED
        assert "--workload-manifest" in out

    def test_reuse_only_artifact_with_occupancy_skip_recorded(
        self, tmp_path: Path
    ):
        """Manifest present but no budget plan: the reuse leg emits, the
        occupancy leg's skip reason is recorded IN the artifact and rho_own
        stays None (absence explicit, never 0), with None-honest gaps."""
        run_dir = tmp_path / "run"
        index = _write_window(run_dir, with_budget_plan=False, with_regime=True)
        analysis_dir = tmp_path / "run" / "analysis" / "t0"
        manifest = _write_manifest(tmp_path / "manifest.json")

        summary = rca.run_own_accounting_pass(
            run_dir, index, analysis_dir, workload_manifest=manifest
        )
        assert summary["n_emitted"] == 1
        doc = json.loads(
            (
                analysis_dir
                / "own_accounting"
                / "cells/rk1/window_squad_v2-01"
                / "own_accounting.json"
            ).read_text(encoding="utf-8")
        )
        assert doc["rho_own"] is None
        assert doc["gaps"] is None
        assert "budget_plan" in doc["occupancy"]["skipped"]
        assert doc["rho_reuse_own"] == pytest.approx(5 / 30)
        # The engine gauge rides along even without the occupancy leg.
        assert doc["rho_engine"] == pytest.approx(0.4)

    def test_dataset_mismatch_manifest_skips_reuse(self, tmp_path: Path, capsys):
        run_dir = tmp_path / "run"
        index = _write_window(run_dir, with_budget_plan=True, with_regime=False)
        analysis_dir = tmp_path / "run" / "analysis" / "t0"
        manifest = _write_manifest(tmp_path / "manifest.json", dataset="nq_open")

        rca.run_own_accounting_pass(
            run_dir, index, analysis_dir, workload_manifest=manifest
        )
        doc = json.loads(
            (
                analysis_dir
                / "own_accounting"
                / "cells/rk1/window_squad_v2-01"
                / "own_accounting.json"
            ).read_text(encoding="utf-8")
        )
        assert doc["rho_reuse_own"] is None
        assert "cross-dataset" in doc["reuse"]["skipped"]
        # occupancy still computed; absent regime.json -> None-honest gaps.
        assert doc["rho_own"] == pytest.approx(0.5)
        assert doc["rho_engine"] is None
        assert doc["gaps"]["abs_gap"] is None
        assert "[own-accounting] SKIP" in capsys.readouterr().out

    def test_malformed_manifest_fails_loud(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        index = _write_window(run_dir, with_budget_plan=True, with_regime=False)
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"dataset": "squad_v2"}), encoding="utf-8")
        with pytest.raises(rca.AnalysisError, match="blocks"):
            rca.run_own_accounting_pass(
                run_dir, index, tmp_path / "a", workload_manifest=bad
            )

    def test_cli_exposes_workload_manifest_flag(self):
        args = rca._build_parser().parse_args(
            ["/tmp/run", "--workload-manifest", "m.json"]
        )
        assert args.workload_manifest == Path("m.json")
