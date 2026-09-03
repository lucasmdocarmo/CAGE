#!/usr/bin/env python3
"""Order:     preflight — BEFORE calibrate_cell and any campaign dispatch
Objective: Produce the P6 floor table: per-cell-family PRE-MEASUREMENT predictions
           (demand D, KV-side saturation bound, predicted collapse onset)
Cloud:     local

The charter's floor-first falsification (PUBLICATION.md P6, D6 §6.1) tests
measured collapse onset against a onset PREDICTED from our own KV/token
arithmetic, within a ×/÷1.15 band (§9.2). That test is only honest if the
prediction is persisted BEFORE the campaign — this script is the producer of
that artifact. The audit found every ingredient present (MODEL_KV,
demand_bytes, the registered grids) but no producer and no artifact.

Prediction recipe (charter §6.1, pinned 2026-08-02 = §9.2):

    lambda* = min(lambda_KV, lambda_compute)

- lambda_KV is derivable NOW from pure arithmetic: budget B = floor(r × D)
  holds ``concurrency_ceiling`` full sequences at the registered shape, and
  Little's law (L = lambda × W) converts that ceiling into a rate via the
  operator-estimated single-request service time W.
- lambda_compute needs a CALIBRATED roofline service-time model
  (src/orchestration/calibration.py runs that probe per cell, later). It is
  therefore emitted as null with an explicit ``requires_calibration`` marker
  — never a fabricated number — and lambda*_pred is labeled
  "kv-bound-only [pending calibration]" until calibration fills the min().

Registered grids (§6.1 budgets; §6.8 pruning rule; rate fractions imported
from load_generator so this artifact can never drift from the dispatcher):

- full        (Group A anchor): r ∈ {1.5, 1.0, 0.75, 0.5, 0.25} × 6 rates
- reduced     (Groups B/C/D):   r ∈ {1.0, 0.5, 0.25} × 3 rates
- anchor-fine (Group A + §6.4): the full factorial's 5 levels PLUS the §6.4
  fine additions {1.25, 0.375} — 7 budget rows. Per-row rate fractions: the
  full 6-rate factorial on the 5 factorial levels, the two §6.4 chassis
  rates {0.85, 1.05} on the fine-only levels (a fine-only row never serves
  the other 4 rates, so predicting them would over-claim). This is the grid
  a session-'a' campaign plan requires (run_campaign pre-resolves every
  registered r, fine levels included).

Fail-closed doctrine: unknown model/engine/grid/kv_dtype and non-positive
numerics REFUSE with the knowns named; an existing --out REFUSES without
--force, because a silently rewritten prediction is a postdiction — defeating
the entire point of P6.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.orchestration.cache_budget import (  # noqa: E402
    KV_DTYPE_FACTOR,
    MODEL_KV,
    demand_bytes,
)
from src.orchestration.load_generator import (  # noqa: E402
    D6_RATE_FRACTIONS,
    D6_REDUCED_RATE_FRACTIONS,
)

__all__ = [
    "FloorTableError",
    "SCHEMA",
    "FULL_BUDGET_LEVELS",
    "REDUCED_BUDGET_LEVELS",
    "ANCHOR_FINE_BUDGET_LEVELS",
    "ANCHOR_FINE_RATE_FRACTIONS",
    "GRID_BUDGET_LEVELS",
    "GRID_RATE_FRACTIONS",
    "PRESSURE_ENGINES",
    "build_floor_table",
]

SCHEMA: Final[str] = "floor-table-v1"

# §6.1: pre-registered budget ratios r = B/D for the Group-A anchor factorial
# (r=1.5 is the comfortable control rung, never in-regime by design).
FULL_BUDGET_LEVELS: Final[Tuple[float, ...]] = (1.5, 1.0, 0.75, 0.5, 0.25)
# §6.8 pruning rule (2026-08-02): Groups B, C, D run the reduced 3×3 grid.
REDUCED_BUDGET_LEVELS: Final[Tuple[float, ...]] = (1.0, 0.5, 0.25)
# §6.4 Graft C (W4.3): the anchor fine 7-level r-grid — the full factorial's
# 5 levels plus {1.25, 0.375}, run at the two chassis-validated rates below.
# Mirrors run_campaign.ANCHOR_FINE_* (each module registers the charter
# literal, matching the FULL/REDUCED convention above).
ANCHOR_FINE_BUDGET_LEVELS: Final[Tuple[float, ...]] = (
    1.5, 1.25, 1.0, 0.75, 0.5, 0.375, 0.25,
)
ANCHOR_FINE_RATE_FRACTIONS: Final[Tuple[float, ...]] = (0.85, 1.05)
assert set(ANCHOR_FINE_RATE_FRACTIONS) <= set(D6_RATE_FRACTIONS), (
    "§6.4 fine rates drifted outside load_generator.D6_RATE_FRACTIONS"
)
assert set(FULL_BUDGET_LEVELS) <= set(ANCHOR_FINE_BUDGET_LEVELS), (
    "§6.4 fine levels no longer cover the §6.1 factorial — the anchor-fine "
    "grid must serve a session-'a' plan's every registered r"
)

GRID_BUDGET_LEVELS: Final[Dict[str, Tuple[float, ...]]] = {
    "full": FULL_BUDGET_LEVELS,
    "reduced": REDUCED_BUDGET_LEVELS,
    "anchor-fine": ANCHOR_FINE_BUDGET_LEVELS,
}
# Rate fractions come from the dispatcher's registered constants — the floor
# table and the load generator can never disagree about the grid. The
# anchor-fine grid maps to the FULL factorial here (its factorial rows serve
# all 6 rates); the fine-ONLY rows override per row to the two §6.4 rates —
# see the per-row selection inside build_floor_table.
GRID_RATE_FRACTIONS: Final[Dict[str, Tuple[float, ...]]] = {
    "full": D6_RATE_FRACTIONS,
    "reduced": D6_REDUCED_RATE_FRACTIONS,
    "anchor-fine": D6_RATE_FRACTIONS,
}

# HF Transformers is the correctness oracle, excluded from pressure sweeps by
# protocol P2 ("HF none") — a floor table for it would predict nothing.
PRESSURE_ENGINES: Final[Tuple[str, ...]] = ("vllm", "sglang", "lmdeploy")

LAMBDA_COMPUTE_STATUS: Final[str] = "requires_calibration"
LAMBDA_STAR_BASIS: Final[str] = "kv-bound-only [pending calibration]"


class FloorTableError(ValueError):
    """A floor-table request that cannot be predicted honestly (fail closed)."""


def _require_int_ge1(name: str, value: Any) -> int:
    # bool is an int subclass; True silently meaning 1 would be a fabricated input.
    if not isinstance(value, int) or isinstance(value, bool):
        raise FloorTableError(f"{name} must be an integer >= 1 (got {value!r})")
    if value < 1:
        raise FloorTableError(f"{name} must be an integer >= 1 (got {value})")
    return value


def _require_positive_finite(name: str, value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise FloorTableError(f"{name} must be a real number > 0 (got {value!r})")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise FloorTableError(f"{name} must be finite and > 0 (got {value!r})")
    return out


def build_floor_table(
    *,
    model: str,
    engine: str,
    concurrency_target: int,
    avg_seq_tokens: int,
    kv_dtype: str,
    grid: str,
    service_time_s: float,
    out: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the floor-table-v1 dict for ONE model×engine cell family.

    ``service_time_s`` is the operator-ESTIMATED single-request E2E time W in
    seconds — lambda_compute needs measurement, so W is the only way to turn
    the concurrency ceiling into a rate before calibration exists. The
    calibrated roofline (§9.2) later replaces this estimate; the estimate is
    recorded here so the replacement is a visible diff, not a silent one.
    """
    if model not in MODEL_KV:
        raise FloorTableError(
            f"unknown model {model!r} — known: {sorted(MODEL_KV)} (fail closed; "
            "add the charter-derived KV arithmetic before predicting)"
        )
    if engine == "hf":
        raise FloorTableError(
            "HF Transformers is the correctness oracle, excluded from pressure "
            "sweeps by protocol P2 ('HF none') — there is no floor to predict"
        )
    if engine not in PRESSURE_ENGINES:
        raise FloorTableError(
            f"unknown engine {engine!r} — known pressure engines: {PRESSURE_ENGINES}"
        )
    if grid not in GRID_BUDGET_LEVELS:
        raise FloorTableError(
            f"unknown grid {grid!r} — known: {sorted(GRID_BUDGET_LEVELS)} "
            "(full = Group-A anchor §6.1; reduced = Groups B/C/D §6.8; "
            "anchor-fine = full + the §6.4 fine levels — what a session-'a' "
            "campaign plan requires)"
        )
    if kv_dtype not in KV_DTYPE_FACTOR:
        raise FloorTableError(
            f"unknown kv_dtype {kv_dtype!r} — known: {sorted(KV_DTYPE_FACTOR)}"
        )
    concurrency = _require_int_ge1("concurrency_target", concurrency_target)
    seq_tokens = _require_int_ge1("avg_seq_tokens", avg_seq_tokens)
    service_s = _require_positive_finite("service_time_s", service_time_s)

    mk = MODEL_KV[model]
    dtype_factor = KV_DTYPE_FACTOR[kv_dtype]
    # Same flooring as cache_budget.plan_budget — the floor table and the
    # budget planner must agree byte-for-byte or gate (j) chases ghosts.
    eff_kv_per_token = math.floor(mk.kv_bytes_per_token * dtype_factor)
    # Hybrid recurrent-state slots are per-sequence and NOT dtype-scaled
    # (model-internal state, not KV) — mirrors demand_bytes exactly.
    per_seq_bytes = seq_tokens * eff_kv_per_token + mk.fixed_state_bytes_per_seq

    demand = demand_bytes(
        model,
        concurrency=concurrency,
        avg_seq_tokens=seq_tokens,
        kv_dtype=kv_dtype,
    )

    rows: List[Dict[str, Any]] = []
    for r in GRID_BUDGET_LEVELS[grid]:
        # Per-row rate fractions: uniform per grid, EXCEPT the anchor-fine
        # grid's fine-ONLY levels (§6.4: those coordinates run at exactly the
        # two chassis rates — predicting the other 4 would over-claim).
        if grid == "anchor-fine" and r not in FULL_BUDGET_LEVELS:
            fractions = list(ANCHOR_FINE_RATE_FRACTIONS)
        else:
            fractions = list(GRID_RATE_FRACTIONS[grid])
        budget = math.floor(r * demand)
        kv_token_capacity = budget // eff_kv_per_token
        concurrency_ceiling = budget // per_seq_bytes
        # Little's law L = lambda × W: the KV-side saturation bound is the
        # highest arrival rate the budget can hold in steady state.
        lambda_kv = concurrency_ceiling / service_s
        row: Dict[str, Any] = {
            "r": float(r),
            "demand_bytes": demand,
            "budget_bytes": budget,
            "kv_token_capacity": kv_token_capacity,
            "per_seq_bytes": per_seq_bytes,
            "concurrency_ceiling": concurrency_ceiling,
            "lambda_kv_rps": lambda_kv,
            # lambda_compute needs the calibrated roofline (§9.2) — a number
            # here would be fabricated, so it is null with a loud marker.
            "lambda_compute_rps": None,
            "lambda_compute_status": LAMBDA_COMPUTE_STATUS,
            # lambda* = min(lambda_KV, lambda_compute); with lambda_compute
            # null the min degenerates to the KV bound — labeled, not hidden.
            "lambda_star_pred_rps": lambda_kv,
            "lambda_star_basis": LAMBDA_STAR_BASIS,
            "predicted_collapse_onset_rps": lambda_kv,
            "rate_fractions": fractions,
            "offered_rates_rps_pred": [f * lambda_kv for f in fractions],
        }
        if concurrency_ceiling == 0:
            row["note"] = (
                "budget below one full sequence at the registered shape — "
                "predicted collapse at ANY offered load (lambda_KV = 0)"
            )
        rows.append(row)

    return {
        "schema": SCHEMA,
        "generated_inputs": {
            "model": model,
            # Demand is engine-independent BY DESIGN (P2: bytes are the same
            # physical quantity on every engine); engine is the cell-family
            # key that binds this prediction to one calibration later.
            "engine": engine,
            "concurrency_target": concurrency,
            "avg_seq_tokens": seq_tokens,
            "kv_dtype": kv_dtype,
            "grid": grid,
            "service_time_s": service_s,
            "out": out,
        },
        "provenance": {
            "kv_formula": mk.formula,
            "kv_bytes_per_token": mk.kv_bytes_per_token,
            "fixed_state_bytes_per_seq": mk.fixed_state_bytes_per_seq,
            "kv_dtype_factor": dtype_factor,
            "service_time_basis": "operator-estimate [pre-calibration]",
            "rate_fractions_source": (
                "load_generator.D6_RATE_FRACTIONS"
                if grid == "full"
                else "load_generator.D6_RATE_FRACTIONS + "
                "ANCHOR_FINE_RATE_FRACTIONS (§6.4 fine-only rows)"
                if grid == "anchor-fine"
                else "load_generator.D6_REDUCED_RATE_FRACTIONS"
            ),
            "charter_refs": ["P6", "6.1", "6.4", "6.8"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
        },
        "rows": rows,
    }


def _main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="build_floor_table",
        description=(
            "Produce the P6 floor table: pre-measurement demand/saturation/"
            "collapse predictions per cell family. lambda_compute is emitted "
            "null (requires calibration) — this table is the honest-prediction "
            "half of the ±15% falsification test."
        ),
    )
    p.add_argument("--model", required=True, help=f"one of {sorted(MODEL_KV)}")
    p.add_argument("--engine", required=True, help=f"one of {PRESSURE_ENGINES}")
    p.add_argument("--concurrency-target", required=True, type=int)
    p.add_argument("--avg-seq-tokens", required=True, type=int)
    p.add_argument(
        "--kv-dtype", default="bf16", help=f"one of {sorted(KV_DTYPE_FACTOR)} (P5: BF16 first)"
    )
    p.add_argument(
        "--grid",
        required=True,
        help="full (Group-A anchor §6.1) | reduced (Groups B/C/D §6.8) | "
        "anchor-fine (full + the §6.4 fine levels; required by a session-'a' "
        "campaign plan)",
    )
    p.add_argument(
        "--service-time-s",
        required=True,
        type=float,
        help="operator-estimated single-request E2E seconds (Little's law W)",
    )
    p.add_argument("--out", required=True, help="JSON artifact path")
    p.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing artifact (a rewritten prediction is a "
        "postdiction — use only with a recorded amendment)",
    )
    a = p.parse_args(argv)

    out_path = Path(a.out)
    if out_path.exists() and not a.force:
        print(
            f"REFUSED: {out_path} already exists — predictions must not be "
            "silently rewritten (P6: the ±15% test is only honest if the "
            "prediction predates the measurement). Pass --force only with a "
            "recorded amendment.",
            file=sys.stderr,
        )
        return 2

    try:
        table = build_floor_table(
            model=a.model,
            engine=a.engine,
            concurrency_target=a.concurrency_target,
            avg_seq_tokens=a.avg_seq_tokens,
            kv_dtype=a.kv_dtype,
            grid=a.grid,
            service_time_s=a.service_time_s,
            out=str(out_path),
        )
    except FloorTableError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    try:
        out_path.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    except OSError as exc:
        print(f"REFUSED: cannot write {out_path}: {exc}", file=sys.stderr)
        return 2

    print(
        f"wrote {out_path} ({len(table['rows'])} budget rows; lambda* is "
        "kv-bound-only PENDING CALIBRATION)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI shim
    raise SystemExit(_main())
