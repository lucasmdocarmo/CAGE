#!/usr/bin/env python3
"""§9.6 simulation-based power CLI — sets the registered N the run-shape leaves open.

DESIGN-INPUT ONLY. Like the §9.7 calibration CLI, every number here is an
operating characteristic of the measurement machinery measured against
pilot-calibrated noise; nothing is a scientific finding (THE-WORK framing,
2026-07-27). This tool never runs the campaign driver, never takes the §9.11
one-look, and never writes an ``analysis_lock.json``.

What it does (thin composition over ``src.analysis.stats.power_sim`` + the
§9.4 registered test paths in ``src.analysis.stats.tests_by_unit`` + the §9.5
registered ``conditional_tost``):

1. **Per-query stage** (chain primary #4 headline + every per_query secondary):
   per dataset × registered metric family × REAL PILOT ARM-PAIR regime. The
   null model resamples the SYMMETRIZED real paired per-example differences of
   two named pilot arms over the same queries — this preserves within-query
   correlation and, for the binary predicate, the REAL paired tie/discordant
   structure (the 2026-08-07 verification finding: a cross-pair independent
   null inflates discordant mass to ~1−Σp² ≈ 0.47-0.50 where real same-query
   pairs show 0.003-0.24, and McNemar power is first-order in that rate).
   Two regimes are simulated and labeled:
   - ``cross_mechanism`` (rag vs prefix_cache): the headline-#4 analog —
     THE BINDING REGIME for the registered N.
   - ``same_family`` (prefix_cache vs no_cache): the low-discordance floor,
     reported as sensitivity.
   Continuous families use ``shift_injection`` on centered symmetrized diffs;
   the binary family uses the HONEST ``tie_flip_injection`` (audit §2.5) on
   the real paired tie mass. Tests route through ``tests_by_unit`` verbatim.
2. **Window stage** (chain primaries #13/#14 unit, §9.4 loaded cells): Welch
   ``batch_means_contrast`` on simulated window means over a queries-per-window
   (W) × windows-per-arm grid (validated ≤ the §9.4 ``DEFAULT_MAX_WINDOWS``
   guard — a refused grid fails LOUD, it is never counted as degenerate).
   LABELED CAVEAT: pilot data is sub-pressure/closed-loop, so simulated
   window-mean variance excludes queueing autocorrelation — required window
   counts are a LOWER bound and the registration adds margin on top.
3. **Conditional-TOST stage** (the three NONE legs of chain primary #13,
   §9.3/§9.5): probability that the registered two-layer ``conditional_tost``
   declares equivalence when equivalence is TRUE, as a function of total
   queries × policy-event fraction — the conditional population is the known
   §9.5 hazard (pilot: 15/289 discordant), and this stage makes its sample-
   size demand explicit instead of leaving three of six #13 sub-hypotheses
   unpowered.
4. Emits the §9.6 deliverables: full power tables (CSV), ``required_n`` at the
   0.8 target for every grid effect (NOT-REACHED recorded, never silently
   dropped), a recommendation block at the declared candidate MDEs, power-curve
   figures, and labeled provenance including the git WORKING-TREE state (a
   dirty/untracked tree marks the artifact PROVISIONAL — §9.6 registers code +
   seed, so the final artifact must be regenerated at a committed SHA).

Alpha sensitivity — three roles in BOTH per-query and window stages: full
α=0.05 (gatekept primaries spend full α per dataset), α/k where k is the
registered count of #13 Holm superiority legs (derived from
``families.FINGERPRINT_SUB_HYPOTHESES``, G15 — k=3 as registered), and α/m
where m is the LARGEST Holm-corrected secondary family in the COMPILED §9.3
family map over the four charter datasets (computed at import, not
hard-coded; m=9 as of 2026-08-16 — the unit-split family map no longer pools
per-query and window rows in one family, which the pre-split m=12 did).

Pooling across datasets is PROHIBITED for the headline (§9.1) — required N is
per dataset and the binding recommendation is the max. Qasper has zero pilot
data (§9.6's stated reason this is simulation-based): it inherits the binding
N with a labeled caveat, it is never invented.

Effect-unit doctrine (2026-08-07 verification finding): the binary family's
``effect`` is the FRACTION OF PAIRED-DIFF TIES flipped to +1 wins
(``power_sim.tie_flip_injection`` semantics). This is NOT the §9.7 calibration
CLI's unit (fraction of one arm's 0-OUTCOMES flipped); every binary row also
carries ``implied_marginal_pp = effect × real tie mass`` so the registered MDE
denotes a stable, reconcilable quantity.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _results_loader as loader  # noqa: E402
import run_calibration as cal  # noqa: E402  (guards + pilot-run map + helpers)
from src.analysis.stats.equivalence import (  # noqa: E402
    DEFAULT_MIN_EVENTS,
    conditional_tost,
)
from src.analysis.stats.families import (  # noqa: E402
    FINGERPRINT_SUB_HYPOTHESES,
    KNOWN_DATASETS,
    compile_family_map,
)
from src.analysis.stats.power_sim import (  # noqa: E402
    PowerSimError,
    required_n,
    shift_injection,
    simulate_campaign,
    tie_flip_injection,
)
from src.analysis.stats.tests_by_unit import (  # noqa: E402
    DEFAULT_MAX_WINDOWS,
    batch_means_contrast,
    mcnemar_binary,
    paired_wilcoxon,
)

STAMP = (
    "POWER SIMULATION / DESIGN-INPUT ONLY (§9.6): simulated operating "
    "characteristics of the CAGE stats machinery under pilot-calibrated noise. "
    "Properties of the measurement machinery — NEVER scientific findings of "
    "the study; no number here may be cited as a result (THE-WORK framing "
    "2026-07-27). Output = the registered run-shape numbers (queries per "
    "window, window counts)."
)

DEFAULT_SEED = 20260807
DEFAULT_N_SIMS = 400  # matches the §9.7 injection split count; exact binomial CI
TARGET_POWER = 0.8  # §9.6: "0.8 power at pre-declared MDEs"
_CI_CONFIDENCE = 0.95

# Per-query stage: candidate per-dataset paired-query counts (the sub-pressure
# unit); the pilot ran ~289-300 pooled per-example units per dataset. 1600 and
# 2400 densify the binding boundary (2026-08-07 adversary amendment A7 — the
# old grid's 1200->2000 gap made "exactly 2000" partly a grid artifact).
DEFAULT_N_GRID: tuple[int, ...] = (50, 100, 200, 300, 500, 800, 1200, 1600, 2000, 2400)
# Window stage: candidate queries-per-window × windows-per-arm-per-cell.
DEFAULT_W_GRID: tuple[int, ...] = (25, 50, 100, 200)
DEFAULT_NW_GRID: tuple[int, ...] = (3, 5, 8, 12, 20)

# --smoke: wiring-sanity mode (seconds, not minutes) through the IDENTICAL
# stage code — tiny grids and sim counts for every stage INCLUDING the TOST
# stage (whose full-run sizes are otherwise module constants the CLI cannot
# shrink). The output is stamped SMOKE in config/report and must NEVER feed
# the registration; the §9.6 artifact is the full default-grid run.
SMOKE_N_SIMS = 8
SMOKE_N_GRID: tuple[int, ...] = (50, 100)
SMOKE_W_GRID: tuple[int, ...] = (25,)
SMOKE_NW_GRID: tuple[int, ...] = (3, 5)
SMOKE_TOST_N_GRID: tuple[int, ...] = (100, 300)
SMOKE_TOST_N_SIMS = 4
# equivalence.conditional_tost refuses bootstrap_iters < 100 (its registered
# floor) — the smoke uses exactly that floor, never below it.
SMOKE_TOST_BOOTSTRAP_ITERS = 100
SMOKE_STAMP = (
    "SMOKE ARTIFACT: tiny grids/sim counts for wiring sanity ONLY — every "
    "number here is statistically meaningless and this output must NEVER be "
    "embedded in the registration (the §9.6 artifact is the full "
    "default-grid run at the final pre-freeze SHA)."
)

# Real pilot arm-pair regimes for the per-query null (2026-08-07 fix: the null
# must reproduce the REAL paired tie/discordant structure, not a cross-pair
# independence fiction). The binding regime is the headline-#4 analog.
PAIR_REGIMES: dict[str, tuple[str, str]] = {
    "cross_mechanism": ("rag", "prefix_cache"),
    "same_family": ("prefix_cache", "no_cache"),
}
BINDING_REGIME = "cross_mechanism"

BINARY_EFFECT_UNIT = (
    "fraction of zero paired-diffs (ties) flipped to +1 wins — "
    "power_sim.tie_flip_injection semantics; NOT the §9.7 calibration CLI's "
    "0-outcome-flip unit (see provenance effect_unit_reconciliation)"
)

# G15 (2026-08-16): the pilot archive predates the §8.5 predicate column, so
# the per-query binary family simulates on exact_match — a LABELED PROXY for
# the registered predicate, stamped on every binary per-query artifact row.
EXACT_MATCH_PROXY_NOTE = (
    "exact_match is a labeled PROXY for the registered §8.5 per-dataset Y "
    "predicate (the pilot archive predates the predicate column); the "
    "registered campaign test runs on 'predicate' (G15)"
)

EFFECT_UNIT_RECONCILIATION = (
    "The §9.7 calibration CLI (calibration.inject_effect kind='flip') flips a "
    "fraction of one arm's 0-OUTCOMES (denominator: marginal zero mass 1-p); "
    "this CLI's tie_flip_injection flips a fraction of zero PAIRED-DIFFS "
    "(denominator: paired tie mass). The two nominal scales coincide only at "
    "p=0.5 and are NOT interchangeable; each binary row carries "
    "implied_marginal_pp = effect x real tie mass for a stable pp-scale "
    "reading of the MDE."
)


def _compiled_holm_worst() -> tuple[int, str]:
    """Largest Holm-corrected family in the compiled §9.3 map (charter datasets).

    Computed from ``families.compile_family_map`` — never hard-coded (the
    2026-08-07 verification found a hard-coded 10 vs the compiled 12).
    """
    fam_map = compile_family_map(sorted(KNOWN_DATASETS))
    holm = fam_map[fam_map["correction"] == "holm"]
    sizes = holm.groupby("family_id").size()
    if sizes.empty:
        raise RuntimeError("compiled family map has no Holm-corrected families")
    m = int(sizes.max())
    return m, str(sizes.idxmax())


SECONDARY_HOLM_M, SECONDARY_HOLM_WORST_FAMILY = _compiled_holm_worst()

# G15 (2026-08-16): the fingerprint worst-case divisor is DERIVED from the
# registered §9.3 decomposition — the count of Holm-corrected superiority legs
# in families.FINGERPRINT_SUB_HYPOTHESES — never a literal /3 that could
# silently drift from the registry (the TOST NONE legs spend no Holm alpha).
FINGERPRINT_HOLM_LEGS: int = sum(
    1
    for _policy, _correction, _sidedness, _predicted in FINGERPRINT_SUB_HYPOTHESES
    if _correction == "holm"
)
if FINGERPRINT_HOLM_LEGS < 1:
    raise RuntimeError(
        "families.FINGERPRINT_SUB_HYPOTHESES registers no Holm legs — the "
        "fingerprint alpha role cannot be derived"
    )

# α sensitivity roles — identical role set in BOTH stages (the 2026-08-07
# verification found the window stage missing the secondary worst case while
# window-unit secondaries #15/#17/#18/#20 are Holm-corrected). The role KEY
# 'fingerprint_holm3_worst' stays stable across artifacts; its value is
# derived above.
ALPHA_ROLES: dict[str, float] = {
    "primary_full_alpha": 0.05,
    "fingerprint_holm3_worst": 0.05 / FINGERPRINT_HOLM_LEGS,
    "secondary_holm_worst": 0.05 / SECONDARY_HOLM_M,
}
WINDOW_ALPHA_ROLES: dict[str, float] = dict(ALPHA_ROLES)

# Candidate MDEs the recommendation block reads out (the values the
# registration will declare; the FULL grid is still simulated and reported).
DECLARED_MDE: dict[str, float] = {
    "serving_continuous": 25.0,  # ms TTFT shift
    "quality_continuous": 0.05,  # faithfulness score shift
    "binary_predicate": 0.05,  # fraction of paired-diff ties flipped
    "window_ttft_ms": 25.0,  # ms window-mean shift
    "window_pass_rate": 0.05,  # pp window pass-rate shift (Y proxy)
    "window_truth_tax": 0.05,  # pp shift on window G−Y (#14 registered variable)
    "window_faithfulness_mean": 0.05,  # score shift on window-mean faithfulness (#13 compress leg)
}

WINDOW_METRICS: tuple[tuple[str, str, tuple[float, ...], str], ...] = (
    (
        "window_ttft_ms",
        "ttft_ms",
        (10.0, 25.0, 50.0),
        "ms shift on the window mean (serving side of #13/#14)",
    ),
    (
        "window_pass_rate",
        "exact_match",
        (0.02, 0.05, 0.10),
        "pp shift on the window pass-rate (labeled proxy for the §9.2 "
        "Y/truth-tax estimand; a mean shift IS the window-level estimand)",
    ),
    (
        "window_faithfulness_mean",
        "faithfulness",
        (0.02, 0.05, 0.10),
        "score-unit shift on the window-mean dual-reference faithfulness — "
        "the #13 compress-leg 'evidence destruction' variable (2026-08-07 "
        "amendment A4). Window aggregation de-ties the metric, so an additive "
        "window-mean shift is honest here (the per-query guard refusal does "
        "not apply to batch means).",
    ),
    (
        "window_truth_tax",
        "__truth_tax__",
        (0.02, 0.05, 0.10),
        "pp shift on the window truth-tax G−Y — the #14 registered variable "
        "simulated DIRECTLY (2026-08-07 amendment A4): per-query g_i−y_i with "
        "G from the labeled 10x-median relative-SLO proxy. Pilot is "
        "sub-pressure so G≈1 and the variance is Y-dominated — recorded in "
        "g_pass_rate; LOWER-bound caveat applies.",
    ),
)

# Relative-SLO proxy for the truth-tax G component (labeled; the campaign's
# real G uses the §6.1 registered SLOs — the pilot has no loaded windows).
TRUTH_TAX_SLO_MULTIPLIER = 10.0


def load_truth_tax_pool(run_root: Path) -> tuple[np.ndarray, dict[str, float]]:
    """Per-query truth-tax contributions g_i − y_i, joined on example_id.

    g_i = 1[ttft_i <= 10 x median ttft] (relative-SLO proxy, labeled);
    y_i = exact_match. Window means of W joint draws are exactly G_w − Y_w,
    so ``simulate_window_power`` on this pool simulates the #14 registered
    variable directly.
    """
    frames: dict[str, pd.DataFrame] = {}
    cell_dir = Path(run_root) / cal.AA_TREE / cal.AA_ARM
    if not cell_dir.is_dir():
        raise PowerSimCLIError(f"A/A arm directory not found: {cell_dir}")
    df = loader.load_cell(cell_dir, cal.AA_ARM)
    for metric in ("ttft_ms", "exact_match"):
        frames[metric] = loader.per_example(df, metric)[["example_id", "value"]]
    joined = frames["ttft_ms"].merge(
        frames["exact_match"], on="example_id", suffixes=("_ttft", "_em")
    )
    if len(joined) < cal.MIN_OBSERVATIONS:
        raise PowerSimCLIError(
            f"truth-tax pool joins only {len(joined)} examples under {run_root}"
        )
    ttft = joined["value_ttft"].to_numpy(dtype=float)
    em = joined["value_em"].to_numpy(dtype=float)
    thresh = TRUTH_TAX_SLO_MULTIPLIER * float(np.median(ttft))
    g = (ttft <= thresh).astype(float)
    diagnostics = {
        "n_pairs": int(len(joined)),
        "slo_threshold_ms": thresh,
        "g_pass_rate": float(g.mean()),
        "y_pass_rate": float(em.mean()),
    }
    return g - em, diagnostics

# Conditional-TOST stage (the #13 NONE legs, §9.3/§9.5).
TOST_METRIC = "faithfulness"
TOST_MARGIN = DECLARED_MDE["quality_continuous"]  # domain margin, metric units
TOST_EVENT_FRACTIONS: tuple[float, ...] = (0.05, 0.25, 1.0)
TOST_N_GRID: tuple[int, ...] = (100, 300, 500, 800, 1200, 1600, 2000, 2400)
TOST_N_SIMS = 150
TOST_BOOTSTRAP_ITERS = 500  # inner dominance bootstrap (reduced + labeled)

# Amendment A2 (2026-08-07): besides the #13 NONE legs, the HEADLINE #4
# claim-ladder row-2 equivalence machinery is simulated on its OWN co-primary
# metrics (unconditional population, f=1.0) at declared §9.5 domain margins =
# the family MDEs: predicate 0.05 risk difference (pp), ttft 25 ms. Without
# this, the expected mixed-outcome claim would rest on a different stage's sim.
# (family, metric, margin, event_fractions, binary_pool, note)
TOST_SPECS: tuple[tuple[str, str, float, tuple[float, ...], bool, str], ...] = (
    ("fingerprint_none_legs", TOST_METRIC, TOST_MARGIN, TOST_EVENT_FRACTIONS,
     False, "#13 NONE legs; §9.5 CONDITIONAL policy-event population"),
    ("headline_predicate_equiv", "exact_match", 0.05, (1.0,), True,
     "#4 claim-ladder row-2 equivalence on the predicate (risk-difference "
     "margin in pp; unconditional population)"),
    ("headline_ttft_equiv", "ttft_ms", 25.0, (1.0,), False,
     "#4 claim-ladder row-2 equivalence on TTFT (ms margin; unconditional "
     "population)"),
)

NOT_REACHED = "NOT_REACHED_ON_GRID"

MAX_SHIFT_COLLISION = cal.MAX_SHIFT_COLLISION


class PowerSimCLIError(RuntimeError):
    """Invalid input or unusable pilot data (fail closed)."""


# --------------------------------------------------------------------------
# Registered test paths as power_sim ``PairedTestFn`` (diff-vector) adapters.
# --------------------------------------------------------------------------


def wilcoxon_diff_p(diffs: np.ndarray) -> float:
    """Continuous paired path: ``tests_by_unit.paired_wilcoxon`` on the diffs.

    Passing (diffs, 0) reproduces the campaign computation exactly —
    ``paired_wilcoxon`` differences its inputs then runs Wilcoxon with the
    REGISTERED ``zero_method="pratt"`` (owner decision 2026-08-16 b,
    ADR-0088: zeros kept in the ranking; pinned normal approximation with
    continuity correction) — the exact path ``run_campaign_analysis``
    executes, never the back-compat ``wilcox`` default the pre-restamp
    simulation used.
    """
    return float(
        paired_wilcoxon(
            diffs,
            np.zeros_like(diffs),
            alternative="two-sided",
            zero_method="pratt",
        ).p_value
    )


def mcnemar_diff_p(diffs: np.ndarray) -> float:
    """Binary paired path: reconstruct arrays and run the registered McNemar.

    A paired-binary difference is in {-1, 0, +1}. (+1) -> (1,0), (-1) -> (0,1),
    (0) -> (0,0): McNemar's exact-binomial p depends only on the discordant
    counts, so the concordant tie type is immaterial — this routes the
    simulation through ``tests_by_unit.mcnemar_binary`` verbatim.
    """
    arr = np.asarray(diffs, dtype=float)
    if not np.isin(arr, (-1.0, 0.0, 1.0)).all():
        raise PowerSimError(
            "mcnemar_diff_p requires paired-binary differences in {-1, 0, 1}; "
            "the variance model must resample a strictly binary paired metric"
        )
    a = (arr > 0).astype(float)
    b = (arr < 0).astype(float)
    return float(mcnemar_binary(a, b, alternative="two-sided").p_value)


# --------------------------------------------------------------------------
# Real-pair null models (per-query stage).
# --------------------------------------------------------------------------


def load_pair_diffs(
    run_root: Path, arm_a: str, arm_b: str, metric: str
) -> np.ndarray:
    """REAL paired per-example differences of two pilot arms on shared queries.

    READ-ONLY on the pilot archive via the canonical loader; joins the pooled
    per-example estimand on example_id (the exact §9.4 pairing unit). Fails
    closed on missing cells or too few joined pairs.
    """
    frames: dict[str, pd.DataFrame] = {}
    for arm in (arm_a, arm_b):
        cell_dir = Path(run_root) / cal.AA_TREE / arm
        if not cell_dir.is_dir():
            raise PowerSimCLIError(f"pair arm directory not found: {cell_dir}")
        df = loader.load_cell(cell_dir, arm)
        frames[arm] = loader.per_example(df, metric)[["example_id", "value"]]
    joined = frames[arm_a].merge(
        frames[arm_b], on="example_id", suffixes=("_a", "_b")
    )
    if len(joined) < cal.MIN_OBSERVATIONS:
        raise PowerSimCLIError(
            f"pair ({arm_a}, {arm_b}) metric {metric!r} under {run_root} joins "
            f"only {len(joined)} examples (< {cal.MIN_OBSERVATIONS})"
        )
    return (
        joined["value_a"].to_numpy(dtype=float)
        - joined["value_b"].to_numpy(dtype=float)
    )


def pair_diagnostics(diffs: np.ndarray) -> dict[str, float]:
    """Tie/discordance structure of a real paired-diff pool."""
    n = diffs.size
    ties = float(np.mean(diffs == 0.0))
    return {
        "n_pairs": int(n),
        "tie_mass": ties,
        "discordant_rate": 1.0 - ties,
        "diff_collision_probability": cal.collision_probability(diffs),
    }


def guard_pair_kind(
    kind: str,
    diffs: np.ndarray,
    *,
    metric: str,
    pair: tuple[str, str],
    max_collision: float = MAX_SHIFT_COLLISION,
) -> dict[str, float]:
    """Honest-injection doctrine on PAIRED DIFFS (P0 2026-08-02 decision).

    - ``shift`` only on effectively continuous diff pools (collision ≤
      threshold): an additive shift on a tie-heavy diff pool breaks every tie
      and inflates power (the P0 bug).
    - ``flip`` only on strictly {-1, 0, 1} paired-binary diffs.
    """
    diag = pair_diagnostics(diffs)
    if kind == "shift":
        if diag["diff_collision_probability"] > max_collision:
            raise PowerSimCLIError(
                f"REFUSED (P0 2026-08-02 doctrine): additive-shift injection on "
                f"tie-heavy paired diffs for metric {metric!r} pair {pair} "
                f"(diff collision probability "
                f"{diag['diff_collision_probability']:.3f} > {max_collision:g})"
            )
    elif kind == "flip":
        if not np.isin(diffs, (-1.0, 0.0, 1.0)).all():
            raise PowerSimCLIError(
                f"REFUSED: tie-flip injection requires strictly binary paired "
                f"diffs in {{-1, 0, 1}}; metric {metric!r} pair {pair} is not."
            )
    else:
        raise PowerSimCLIError(f"unknown injection kind {kind!r} (shift|flip)")
    return diag


def make_paired_null_model(
    diffs: np.ndarray, kind: str
) -> Callable[[np.random.Generator, int], np.ndarray]:
    """H0 null from SYMMETRIZED real paired diffs (2026-08-07 critical fix).

    - binary (``flip``): pool = concat(d, -d) — enforces P(+1)=P(-1)=
      (observed discordant)/2 and keeps the REAL paired tie mass, the quantity
      McNemar power is first-order in (audit §2.5).
    - continuous (``shift``): pool = concat(c, -c) with c = d - mean(d) —
      removes the pair's true location effect (H0) while keeping the real
      paired dispersion INCLUDING treatment-effect heterogeneity, which makes
      the continuous required_n an UPPER bound (conservative; labeled).
    """
    d = np.asarray(diffs, dtype=float)
    if d.size < cal.MIN_OBSERVATIONS:
        raise PowerSimCLIError(
            f"paired null needs >= {cal.MIN_OBSERVATIONS} pairs, got {d.size}"
        )
    if kind == "flip":
        pool = np.concatenate([d, -d])
    else:
        c = d - d.mean()
        pool = np.concatenate([c, -c])

    def model(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.choice(pool, size=n, replace=True)

    return model


# --------------------------------------------------------------------------
# Window stage: Welch batch-means power (own seeded loop — the unpaired
# two-sample contrast cannot be expressed through simulate_campaign's
# paired-diff contract).
# --------------------------------------------------------------------------


def simulate_window_power(
    values: np.ndarray,
    effect_grid: Sequence[float],
    w_grid: Sequence[int],
    nw_grid: Sequence[int],
    n_sims: int,
    seed: int,
    *,
    alpha: float,
) -> pd.DataFrame:
    """Power table for ``batch_means_contrast`` over (effect × W × n_windows).

    Each simulated window mean averages W values resampled from the pilot
    per-query pool; arm A receives the additive window-mean effect. The
    n_windows grid is validated against the §9.4 ``DEFAULT_MAX_WINDOWS`` guard
    UPFRONT (a refused grid fails loud — the 2026-08-07 verification found the
    guard's ValueError silently counted as 'degenerate'). Genuinely degenerate
    draws (both arms zero-variance) are counted as non-rejections and reported
    in ``n_degenerate``.
    """
    effects = [float(e) for e in effect_grid]
    ws = [int(w) for w in w_grid]
    nws = [int(n) for n in nw_grid]
    if not effects or not ws or not nws:
        raise PowerSimCLIError("effect/W/n_windows grids must be non-empty")
    if any(w < 1 for w in ws):
        raise PowerSimCLIError(f"W grid must be >= 1, got {ws}")
    if any(n < 2 for n in nws):
        raise PowerSimCLIError(f"n_windows grid must be >= 2 (Welch), got {nws}")
    if any(n > DEFAULT_MAX_WINDOWS for n in nws):
        raise PowerSimCLIError(
            f"n_windows grid exceeds the §9.4 batch-means guard "
            f"(max_windows={DEFAULT_MAX_WINDOWS}); got {nws} — the registered "
            f"test would refuse such input, so simulating it is meaningless"
        )
    if n_sims < 1:
        raise PowerSimCLIError(f"n_sims must be >= 1, got {n_sims}")
    pool = np.asarray(values, dtype=float)
    if pool.size < cal.MIN_OBSERVATIONS:
        raise PowerSimCLIError(
            f"window model needs >= {cal.MIN_OBSERVATIONS} pilot observations, "
            f"got {pool.size}"
        )

    grid = [(e, w, nw) for e in effects for w in ws for nw in nws]
    children = np.random.SeedSequence(seed).spawn(len(grid))
    rows: list[dict[str, float | int]] = []
    for point, (effect, w, nw) in enumerate(grid):
        rng = np.random.default_rng(children[point])
        rejections = 0
        degenerate = 0
        for _ in range(n_sims):
            means_a = rng.choice(pool, size=(nw, w), replace=True).mean(axis=1) + effect
            means_b = rng.choice(pool, size=(nw, w), replace=True).mean(axis=1)
            try:
                res = batch_means_contrast(
                    means_a, means_b, alternative="two-sided"
                )
            except ValueError as exc:
                if "zero variance" not in str(exc):
                    raise  # §9.4 guard or shape errors must fail LOUD
                degenerate += 1
                continue
            if res.p_value < alpha:
                rejections += 1
        ci = binomtest(rejections, n_sims).proportion_ci(
            confidence_level=_CI_CONFIDENCE, method="exact"
        )
        rows.append(
            {
                "effect": effect,
                "queries_per_window": w,
                "n_windows": nw,
                "n_sims": n_sims,
                "alpha": alpha,
                "power": rejections / n_sims,
                "ci_low": float(ci.low),
                "ci_high": float(ci.high),
                "n_degenerate": degenerate,
            }
        )
    return pd.DataFrame(rows)


def required_windows(
    table: pd.DataFrame,
    effect: float,
    queries_per_window: int,
    *,
    target_power: float = TARGET_POWER,
) -> int | str:
    """Smallest n_windows reaching the target at (effect, W); NOT_REACHED label."""
    at = table[
        (table["effect"] == float(effect))
        & (table["queries_per_window"] == int(queries_per_window))
    ]
    if at.empty:
        raise PowerSimCLIError(
            f"(effect={effect}, W={queries_per_window}) not on the simulated grid"
        )
    reaching = at[at["power"] >= target_power]
    if reaching.empty:
        return NOT_REACHED
    return int(reaching["n_windows"].min())


# --------------------------------------------------------------------------
# Conditional-TOST stage (the #13 NONE legs, §9.3/§9.5).
# --------------------------------------------------------------------------


def simulate_tost_power(
    diffs: np.ndarray,
    n_grid: Sequence[int],
    event_fractions: Sequence[float],
    n_sims: int,
    seed: int,
    *,
    margin: float = TOST_MARGIN,
    alpha: float = 0.05,
    bootstrap_iters: int = TOST_BOOTSTRAP_ITERS,
    binary: bool = False,
) -> pd.DataFrame:
    """P(two-layer equivalence declared | equivalence TRUE) over n × event frac.

    H0-equivalence null: symmetrized real pair diffs — centered for continuous
    metrics; for ``binary=True`` (paired-binary diffs in {-1, 0, 1}) the pool
    is concat(d, -d) WITHOUT centering, which preserves the ternary support
    and yields exact mean 0 by construction (amendment A2). Each sim draws n
    paired diffs, marks round(f·n) queries as policy-touched, and runs the
    REGISTERED ``conditional_tost`` (both layers: domain t-TOST at ``margin``
    + Cliff's-δ dominance). ``insufficient-n`` outcomes (below the registered
    ``DEFAULT_MIN_EVENTS``) count as failures and are reported separately —
    that is exactly the §9.5 conditional-population hazard this stage exists
    to expose.
    """
    ns = [int(n) for n in n_grid]
    fracs = [float(f) for f in event_fractions]
    if not ns or not fracs:
        raise PowerSimCLIError("n_grid and event_fractions must be non-empty")
    if any(n < 2 for n in ns):
        raise PowerSimCLIError(f"n_grid must be >= 2, got {ns}")
    if any(not 0.0 < f <= 1.0 for f in fracs):
        raise PowerSimCLIError(f"event fractions must be in (0, 1], got {fracs}")
    d = np.asarray(diffs, dtype=float)
    if d.size < cal.MIN_OBSERVATIONS:
        raise PowerSimCLIError(
            f"TOST null needs >= {cal.MIN_OBSERVATIONS} pairs, got {d.size}"
        )
    if binary:
        if not np.isin(d, (-1.0, 0.0, 1.0)).all():
            raise PowerSimCLIError(
                "binary TOST pool requires paired-binary diffs in {-1, 0, 1}"
            )
        pool = np.concatenate([d, -d])
    else:
        c = d - d.mean()
        pool = np.concatenate([c, -c])

    grid = [(f, n) for f in fracs for n in ns]
    children = np.random.SeedSequence(seed).spawn(len(grid))
    rows: list[dict[str, float | int]] = []
    for point, (frac, n) in enumerate(grid):
        rng = np.random.default_rng(children[point])
        n_events = int(round(frac * n))
        equivalent = 0
        insufficient = 0
        for _ in range(n_sims):
            sim_d = rng.choice(pool, size=n, replace=True)
            mask = np.zeros(n, dtype=bool)
            if n_events > 0:
                mask[rng.choice(n, size=n_events, replace=False)] = True
            res = conditional_tost(
                sim_d,
                np.zeros(n),
                mask,
                margin=margin,
                alpha=alpha,
                bootstrap_iters=bootstrap_iters,
                seed=int(rng.integers(2**31)),
            )
            if res.domain_verdict == "insufficient-n":
                insufficient += 1
            elif res.equivalent:
                equivalent += 1
        ci = binomtest(equivalent, n_sims).proportion_ci(
            confidence_level=_CI_CONFIDENCE, method="exact"
        )
        rows.append(
            {
                "event_fraction": frac,
                "n": n,
                "n_events_target": n_events,
                "n_sims": n_sims,
                "alpha": alpha,
                "margin": margin,
                "prob_equivalent": equivalent / n_sims,
                "ci_low": float(ci.low),
                "ci_high": float(ci.high),
                "n_insufficient": insufficient,
            }
        )
    return pd.DataFrame(rows)


def tost_required(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Smallest n reaching 0.8 P(equivalence) per spec × group × event fraction."""
    out: list[dict[str, Any]] = []
    keys = ["family", "metric", "dataset", "pair_regime", "alpha_role"]
    for (family, metric, dataset, regime, role), sub in table.groupby(
        keys, sort=True
    ):
        for frac in sorted(sub["event_fraction"].unique()):
            at = sub[sub["event_fraction"] == frac]
            reaching = at[at["prob_equivalent"] >= TARGET_POWER]
            out.append(
                {
                    "stage": "tost_conditional",
                    "family": family,
                    "metric": metric,
                    "dataset": dataset,
                    "pair_regime": regime,
                    "alpha_role": role,
                    "event_fraction": float(frac),
                    "margin": float(at["margin"].iloc[0]),
                    "target_power": TARGET_POWER,
                    "required_n": (
                        NOT_REACHED if reaching.empty else int(reaching["n"].min())
                    ),
                }
            )
    return out


# --------------------------------------------------------------------------
# Stage runners.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PerQueryFamily:
    """One per-query-stage family bound to its power_sim adapters."""

    name: str
    metric: str
    kind: str
    effect_sizes: tuple[float, ...]
    effect_unit: str
    test_fn: Callable[[np.ndarray], float]
    injection: Callable[..., np.ndarray]


def _per_query_families() -> tuple[PerQueryFamily, ...]:
    """The three §9.7-registered families, re-bound to diff-vector adapters.

    The binary family's effect unit is THIS CLI's tie-flip unit, never the
    §9.7 CLI's 0-outcome-flip string (2026-08-07 verification finding).
    """
    adapters: dict[str, tuple[Callable[[np.ndarray], float], Callable[..., np.ndarray]]] = {
        "shift": (wilcoxon_diff_p, shift_injection),
        "flip": (mcnemar_diff_p, tie_flip_injection),
    }
    fams = []
    for f in cal.FAMILIES:
        test_fn, injection = adapters[f.kind]
        fams.append(
            PerQueryFamily(
                name=f.name,
                metric=f.metric,
                kind=f.kind,
                effect_sizes=tuple(float(e) for e in f.effect_sizes),
                effect_unit=(
                    BINARY_EFFECT_UNIT if f.kind == "flip" else f.effect_unit
                ),
                test_fn=test_fn,
                injection=injection,
            )
        )
    return tuple(fams)


def run_per_query_stage(
    dataset_runs: Mapping[str, Path],
    *,
    seed: int,
    n_grid: Sequence[int],
    n_sims: int,
    alpha_roles: Mapping[str, float] = ALPHA_ROLES,
    pair_regimes: Mapping[str, tuple[str, str]] = PAIR_REGIMES,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Per-query power tables per dataset × pair regime × family × alpha role.

    Returns (table, refusals): a (regime × family) whose real diff pool fails
    the injection guard (e.g. a near-identical same_family pair makes a
    continuous shift dishonest) is RECORDED as a refusal and skipped — never
    silently simulated, never a crash.
    """
    frames: list[pd.DataFrame] = []
    refusals: list[dict[str, Any]] = []
    for i_ds, (dataset, run_root) in enumerate(sorted(dataset_runs.items())):
        for i_reg, (regime, (arm_a, arm_b)) in enumerate(sorted(pair_regimes.items())):
            for i_fam, fam in enumerate(_per_query_families()):
                diffs = load_pair_diffs(Path(run_root), arm_a, arm_b, fam.metric)
                try:
                    diag = guard_pair_kind(
                        fam.kind, diffs, metric=fam.metric, pair=(arm_a, arm_b)
                    )
                except PowerSimCLIError as exc:
                    refusals.append(
                        {
                            "dataset": dataset,
                            "pair_regime": regime,
                            "family": fam.name,
                            "metric": fam.metric,
                            "reason": str(exc),
                        }
                    )
                    print(
                        f"[run_power_sim] REFUSED per_query {dataset}/{regime}/"
                        f"{fam.name}: {exc}",
                        flush=True,
                    )
                    continue
                model = make_paired_null_model(diffs, fam.kind)
                for i_role, (role, alpha) in enumerate(sorted(alpha_roles.items())):
                    point_seed = (
                        seed
                        + 100_000 * i_ds
                        + 10_000 * i_reg
                        + 1_000 * i_fam
                        + 10 * i_role
                    )
                    table = simulate_campaign(
                        fam.effect_sizes,
                        n_grid,
                        model,
                        n_sims,
                        point_seed,
                        alpha=alpha,
                        test_fn=fam.test_fn,
                        injection=fam.injection,
                    )
                    table.insert(0, "stage", "per_query")
                    table.insert(1, "dataset", dataset)
                    table.insert(2, "pair_regime", regime)
                    table.insert(3, "family", fam.name)
                    table.insert(4, "metric", fam.metric)
                    table.insert(5, "alpha_role", role)
                    table["effect_unit"] = fam.effect_unit
                    table["kind"] = fam.kind
                    table["seed"] = point_seed
                    table["pair"] = f"{arm_a}-vs-{arm_b}"
                    table["n_pairs"] = diag["n_pairs"]
                    table["tie_mass"] = diag["tie_mass"]
                    table["discordant_rate"] = diag["discordant_rate"]
                    table["implied_marginal_pp"] = (
                        table["effect"] * diag["tie_mass"]
                        if fam.kind == "flip"
                        else np.nan
                    )
                    # G15: the proxy status rides the artifact row itself, not
                    # just prose — a table consumer must see it.
                    table["metric_note"] = (
                        EXACT_MATCH_PROXY_NOTE
                        if fam.metric == "exact_match"
                        else ""
                    )
                    frames.append(table)
                    print(
                        f"[run_power_sim] per_query {dataset}/{regime}/"
                        f"{fam.name}/{role}: {len(table)} grid points done",
                        flush=True,
                    )
    if not frames:
        raise PowerSimCLIError("every per-query (regime × family) was refused")
    return pd.concat(frames, ignore_index=True), refusals


def run_window_stage(
    dataset_runs: Mapping[str, Path],
    *,
    seed: int,
    w_grid: Sequence[int],
    nw_grid: Sequence[int],
    n_sims: int,
    alpha_roles: Mapping[str, float] = WINDOW_ALPHA_ROLES,
) -> pd.DataFrame:
    """Window-unit power tables (batch-means Welch) per dataset × metric × role."""
    frames: list[pd.DataFrame] = []
    for i_ds, (dataset, run_root) in enumerate(sorted(dataset_runs.items())):
        for i_met, (name, metric, effects, unit) in enumerate(WINDOW_METRICS):
            tt_diag: dict[str, float] | None = None
            if metric == "__truth_tax__":
                values, tt_diag = load_truth_tax_pool(Path(run_root))
            else:
                values = cal.load_arm_metric(Path(run_root), metric)
            for i_role, (role, alpha) in enumerate(sorted(alpha_roles.items())):
                point_seed = (
                    seed + 500_000 + 100_000 * i_ds + 1_000 * i_met + 10 * i_role
                )
                table = simulate_window_power(
                    values, effects, w_grid, nw_grid, n_sims, point_seed,
                    alpha=alpha,
                )
                table.insert(0, "stage", "window")
                table.insert(1, "dataset", dataset)
                table.insert(2, "family", name)
                table.insert(3, "metric", metric)
                table.insert(4, "alpha_role", role)
                table["effect_unit"] = unit
                table["kind"] = "window_mean_shift"
                table["seed"] = point_seed
                table["n_pilot_observations"] = int(values.size)
                if tt_diag is not None:
                    table["g_pass_rate"] = tt_diag["g_pass_rate"]
                    table["y_pass_rate"] = tt_diag["y_pass_rate"]
                    table["slo_threshold_ms"] = tt_diag["slo_threshold_ms"]
                frames.append(table)
                print(
                    f"[run_power_sim] window {dataset}/{name}/{role}: "
                    f"{len(table)} grid points done",
                    flush=True,
                )
    return pd.concat(frames, ignore_index=True)


def run_tost_stage(
    dataset_runs: Mapping[str, Path],
    *,
    seed: int,
    n_grid: Sequence[int] = TOST_N_GRID,
    n_sims: int = TOST_N_SIMS,
    bootstrap_iters: int = TOST_BOOTSTRAP_ITERS,
    pair_regimes: Mapping[str, tuple[str, str]] = PAIR_REGIMES,
    specs: Sequence[tuple[str, str, float, tuple[float, ...], bool, str]] = TOST_SPECS,
) -> pd.DataFrame:
    """Equivalence power per TOST spec × dataset × pair regime (α=0.05).

    Specs (amendment A2): the #13 NONE legs (conditional population) plus the
    headline #4 equivalence machinery on its own co-primary metrics
    (unconditional, f=1.0) at the declared §9.5 margins.
    """
    frames: list[pd.DataFrame] = []
    for i_ds, (dataset, run_root) in enumerate(sorted(dataset_runs.items())):
        for i_reg, (regime, (arm_a, arm_b)) in enumerate(sorted(pair_regimes.items())):
            for i_spec, (family, metric, margin, fracs, binary, note) in enumerate(
                specs
            ):
                diffs = load_pair_diffs(Path(run_root), arm_a, arm_b, metric)
                point_seed = (
                    seed + 900_000 + 100_000 * i_ds + 10_000 * i_reg
                    + 1_000 * i_spec
                )
                table = simulate_tost_power(
                    diffs, n_grid, fracs, n_sims, point_seed,
                    margin=margin, binary=binary,
                    bootstrap_iters=bootstrap_iters,
                )
                table.insert(0, "stage", "tost_conditional")
                table.insert(1, "dataset", dataset)
                table.insert(2, "pair_regime", regime)
                table.insert(3, "family", family)
                table.insert(4, "metric", metric)
                table.insert(5, "alpha_role", "primary_full_alpha")
                table["effect_unit"] = (
                    f"H0-true equivalence at domain margin {margin:g} "
                    f"(+ Cliff's-delta dominance layer); min_events="
                    f"{DEFAULT_MIN_EVENTS}; {note}"
                )
                table["kind"] = "conditional_tost"
                table["seed"] = point_seed
                table["pair"] = f"{arm_a}-vs-{arm_b}"
                frames.append(table)
                print(
                    f"[run_power_sim] tost {dataset}/{regime}/{family}: "
                    f"{len(table)} grid points done",
                    flush=True,
                )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Required-N extraction + recommendation.
# --------------------------------------------------------------------------


def per_query_required(table: pd.DataFrame) -> list[dict[str, Any]]:
    """``required_n`` (§9.6, target 0.8) per per-query group × effect."""
    out: list[dict[str, Any]] = []
    keys = ["dataset", "pair_regime", "family", "metric", "alpha_role"]
    for (dataset, regime, family, metric, role), sub in table.groupby(
        keys, sort=True
    ):
        for effect in sorted(sub["effect"].unique()):
            try:
                n_req: int | str = required_n(
                    sub, float(effect), target_power=TARGET_POWER
                )
            except PowerSimError:
                n_req = NOT_REACHED
            out.append(
                {
                    "stage": "per_query",
                    "dataset": dataset,
                    "pair_regime": regime,
                    "family": family,
                    "metric": metric,
                    "alpha_role": role,
                    "effect": float(effect),
                    "target_power": TARGET_POWER,
                    "required_n": n_req,
                }
            )
    return out


def window_required(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Smallest window count per (group × effect × W) at the 0.8 target."""
    out: list[dict[str, Any]] = []
    keys = ["dataset", "family", "metric", "alpha_role"]
    for (dataset, family, metric, role), sub in table.groupby(keys, sort=True):
        for effect in sorted(sub["effect"].unique()):
            for w in sorted(sub["queries_per_window"].unique()):
                out.append(
                    {
                        "stage": "window",
                        "dataset": dataset,
                        "family": family,
                        "metric": metric,
                        "alpha_role": role,
                        "effect": float(effect),
                        "queries_per_window": int(w),
                        "target_power": TARGET_POWER,
                        "required_n_windows": required_windows(
                            sub, float(effect), int(w)
                        ),
                    }
                )
    return out


def _binding(rows: list[dict[str, Any]], value_key: str) -> dict[str, Any]:
    """Max requirement across datasets; NOT_REACHED dominates (fail closed)."""
    if not rows:
        raise PowerSimCLIError("no rows to take a binding requirement over")
    if any(r[value_key] == NOT_REACHED for r in rows):
        return {"binding": NOT_REACHED, "per_dataset": rows}
    return {
        "binding": max(int(r[value_key]) for r in rows),
        "per_dataset": rows,
    }


def build_recommendation(
    pq_required: list[dict[str, Any]],
    win_required: list[dict[str, Any]],
    tost_req: list[dict[str, Any]],
    refusals: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """The §9.6 read-out at the declared candidate MDEs, per alpha role.

    Per-dataset numbers are kept (headline pooling is prohibited, §9.1); the
    binding number is the max across measured datasets IN THE BINDING REGIME
    (cross_mechanism — the headline-#4 analog); same_family appears as a
    labeled sensitivity block. Qasper inherits the binding number by labeled
    caveat — it has zero pilot data (§9.6).
    """
    rec: dict[str, Any] = {
        "target_power": TARGET_POWER,
        "declared_candidate_mdes": dict(DECLARED_MDE),
        "binding_pair_regime": BINDING_REGIME,
        "qasper_policy": (
            "zero pilot data (§9.6): qasper inherits the binding (max) "
            "requirement across measured datasets; labeled caveat, not a "
            "simulated number"
        ),
        "per_query": {},
        "per_query_sensitivity_same_family": {},
        "window": {},
        "tost_conditional": {},
    }
    regimes = sorted({r["pair_regime"] for r in pq_required})
    for fam in sorted({r["family"] for r in pq_required}):
        mde = DECLARED_MDE.get(fam)
        if mde is None:
            continue
        for regime in regimes:
            key = (
                "per_query"
                if regime == BINDING_REGIME
                else "per_query_sensitivity_same_family"
            )
            rec[key][fam] = rec[key].get(fam, {})
            for role in sorted({r["alpha_role"] for r in pq_required}):
                rows = [
                    r
                    for r in pq_required
                    if r["family"] == fam
                    and r["pair_regime"] == regime
                    and r["alpha_role"] == role
                    and r["effect"] == mde
                ]
                if not rows:
                    continue  # regime × family refused by the guard — labeled
                rec[key][fam][role] = {"mde": mde, **_binding(rows, "required_n")}
    for fam in sorted({r["family"] for r in win_required}):
        mde = DECLARED_MDE.get(fam)
        if mde is None:
            continue
        rec["window"][fam] = {}
        for role in sorted({r["alpha_role"] for r in win_required}):
            per_w: dict[str, Any] = {}
            for w in sorted(
                {
                    r["queries_per_window"]
                    for r in win_required
                    if r["family"] == fam
                }
            ):
                rows = [
                    r
                    for r in win_required
                    if r["family"] == fam
                    and r["alpha_role"] == role
                    and r["effect"] == mde
                    and r["queries_per_window"] == w
                ]
                per_w[str(w)] = _binding(rows, "required_n_windows")
            rec["window"][fam][role] = {"mde": mde, "by_queries_per_window": per_w}
    for family in sorted({r["family"] for r in tost_req}):
        for frac in sorted(
            {r["event_fraction"] for r in tost_req if r["family"] == family}
        ):
            for regime in sorted({r["pair_regime"] for r in tost_req}):
                rows = [
                    r
                    for r in tost_req
                    if r["family"] == family
                    and r["event_fraction"] == frac
                    and r["pair_regime"] == regime
                ]
                if not rows:
                    continue
                rec["tost_conditional"][
                    f"{family}|{regime}@event_fraction={frac:g}"
                ] = _binding(rows, "required_n")
    # Guard refusals surface EXPLICITLY in the read-out (2026-08-07 residual
    # finding: a family refused everywhere must not silently vanish while its
    # declared MDE still advertises a number — 'never silently dropped').
    refused_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in refusals:
        refused_by_key.setdefault((r["family"], r["pair_regime"]), []).append(
            {"dataset": r["dataset"], "reason": r["reason"]}
        )
    for (fam, regime), items in sorted(refused_by_key.items()):
        key = (
            "per_query"
            if regime == BINDING_REGIME
            else "per_query_sensitivity_same_family"
        )
        block = rec[key].setdefault(fam, {})
        block["status"] = "REFUSED_BY_INJECTION_GUARD"
        if DECLARED_MDE.get(fam) is not None:
            block["mde"] = DECLARED_MDE[fam]
        block["guard_refusals"] = items
        block["resolution_pending"] = (
            "no honest additive-shift model exists for this family on real "
            "paired diffs (tie-heavy — the P0 doctrine refuses inflation); the "
            "§9.6 registered N is UNDELIVERED for this family. Owner decision "
            "required: recommended = tie-flip-at-magnitude model (audit §2.5) "
            "with a re-declared MDE unit, then re-simulate."
        )
        # Poison any role bindings computed from partial (some-dataset) survival.
        for role_block in block.values():
            if isinstance(role_block, dict) and "binding" in role_block:
                role_block["binding_caveat"] = (
                    "INCOMPLETE: guard refusals present for this family/regime"
                )
    return rec


# --------------------------------------------------------------------------
# Provenance: git working-tree state (§9.6 registers code + seed).
# --------------------------------------------------------------------------


def _git_state() -> dict[str, Any]:
    """HEAD + working-tree cleanliness. A dirty tree marks the artifact
    PROVISIONAL: the recorded SHA does not contain the code that ran."""
    head = cal._git_head()
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    except OSError:
        lines = ["<git status unavailable>"]
    return {
        "head": head,
        "dirty": bool(lines),
        "changed_or_untracked": lines[:40],
        "provisional": bool(lines),
        "note": (
            "PROVISIONAL if dirty: §9.6 registers code + seed, so the final "
            "registration artifact must be regenerated at a committed SHA "
            "that contains this driver."
        ),
    }


# --------------------------------------------------------------------------
# Figures.
# --------------------------------------------------------------------------


def write_figures(
    pq_table: pd.DataFrame, win_table: pd.DataFrame, out_dir: Path
) -> list[Path]:
    """Power curves at the primary alpha role; per-query curves show the
    BINDING regime (cross_mechanism)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "plots"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    pq = pq_table[
        (pq_table["alpha_role"] == "primary_full_alpha")
        & (pq_table["pair_regime"] == BINDING_REGIME)
    ]
    for fam in sorted(pq["family"].unique()):
        sub = pq[pq["family"] == fam]
        datasets = sorted(sub["dataset"].unique())
        fig, axes = plt.subplots(
            1, len(datasets), figsize=(4.2 * len(datasets), 3.4), sharey=True
        )
        axes = np.atleast_1d(axes)
        for ax, ds in zip(axes, datasets):
            d = sub[sub["dataset"] == ds]
            for effect in sorted(d["effect"].unique()):
                e = d[d["effect"] == effect].sort_values("n")
                ax.plot(e["n"], e["power"], marker="o", label=f"effect={effect:g}")
                ax.fill_between(e["n"], e["ci_low"], e["ci_high"], alpha=0.15)
            ax.axhline(TARGET_POWER, color="black", ls="--", lw=0.8)
            ax.set_xscale("log")
            ax.set_title(ds)
            ax.set_xlabel("queries per dataset (paired n)")
        axes[0].set_ylabel("simulated power")
        axes[0].legend(fontsize=8)
        fig.suptitle(
            f"per-query power — {fam} ({BINDING_REGIME} null, α=0.05, "
            f"target {TARGET_POWER})"
        )
        fig.tight_layout()
        p = fig_dir / f"per_query_{fam}.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        written.append(p)

    win = win_table[win_table["alpha_role"] == "primary_full_alpha"]
    for fam in sorted(win["family"].unique()):
        sub = win[win["family"] == fam]
        ws = sorted(sub["queries_per_window"].unique())
        fig, axes = plt.subplots(
            1, len(ws), figsize=(3.6 * len(ws), 3.4), sharey=True
        )
        axes = np.atleast_1d(axes)
        for ax, w in zip(axes, ws):
            d = sub[sub["queries_per_window"] == w]
            for effect in sorted(d["effect"].unique()):
                e = (
                    d[d["effect"] == effect]
                    .groupby("n_windows", as_index=False)["power"]
                    .min()  # worst dataset per point — binding view, hides nothing
                    .sort_values("n_windows")
                )
                ax.plot(
                    e["n_windows"], e["power"], marker="o", label=f"effect={effect:g}"
                )
            ax.axhline(TARGET_POWER, color="black", ls="--", lw=0.8)
            ax.set_title(f"W={w} queries/window")
            ax.set_xlabel("windows per arm")
        axes[0].set_ylabel("simulated power (min over datasets)")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"window power — {fam} (α=0.05, target {TARGET_POWER})")
        fig.tight_layout()
        p = fig_dir / f"window_{fam}.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        written.append(p)
    return written


# --------------------------------------------------------------------------
# Report assembly.
# --------------------------------------------------------------------------


def _fmt_req(v: int | str) -> str:
    return str(v) if v == NOT_REACHED else f"{int(v)}"


def _markdown_report(
    config: dict[str, Any],
    pq_required: list[dict[str, Any]],
    win_required: list[dict[str, Any]],
    tost_req: list[dict[str, Any]],
    recommendation: dict[str, Any],
    refusals: list[dict[str, Any]],
    translation: list[dict[str, Any]],
) -> str:
    repo = config["repo_state"]
    tost_cfg = config.get("tost") or {}
    tost_n_sims = tost_cfg.get("n_sims", TOST_N_SIMS)
    tost_bootstrap_iters = tost_cfg.get("bootstrap_iters", TOST_BOOTSTRAP_ITERS)
    lines: list[str] = [
        "# §9.6 simulation-based power — design-input report",
        "",
        f"> **{STAMP}**",
        "",
    ]
    if config.get("smoke"):
        lines += [f"> **{SMOKE_STAMP}**", ""]
    if repo["dirty"]:
        lines += [
            "> **PROVISIONAL ARTIFACT:** generated on a DIRTY working tree "
            f"(HEAD `{repo['head']}` does not contain all the code that ran). "
            "Regenerate at a committed SHA before the registration embeds "
            "these numbers (§9.6 registers code + seed).",
            "",
        ]
    lines += [
        f"Generated {config['generated_utc']} at repo HEAD `{repo['head']}` "
        f"(dirty={repo['dirty']}); seed `{config['seed']}`; "
        f"n_sims/grid-point={config['n_sims']} (per-query & window stages; "
        f"TOST stage: {tost_n_sims}); target power {TARGET_POWER} "
        f"(§9.6). Secondary Holm worst case = α/{config['secondary_holm_m']} "
        f"(largest compiled Holm family: `{config['secondary_holm_worst_family']}`).",
        "",
        "## Per-query stage — required paired queries per dataset",
        "",
        "Null models resample SYMMETRIZED REAL pilot arm-pair differences "
        f"(binding regime `{BINDING_REGIME}` = headline-#4 analog; "
        "`same_family` = low-discordance sensitivity floor). Tests are the "
        "registered §9.4 paths (`paired_wilcoxon` with the registered "
        "`zero_method='pratt'`, ADR-0088 / `mcnemar_binary`); the "
        "binary family uses the honest tie-flip injection on the REAL paired "
        "tie mass (audit §2.5). Continuous required_n is an UPPER bound "
        "(symmetrized dispersion keeps treatment-effect heterogeneity); the "
        "binary effect unit is the tie-flip unit (see reconciliation note). "
        f"PROXY LABEL (G15): {EXACT_MATCH_PROXY_NOTE}.",
        "",
        "| dataset | regime | family | α role | effect | required n |",
        "|---|---|---|---|---|---|",
    ]
    for r in pq_required:
        lines.append(
            f"| {r['dataset']} | {r['pair_regime']} | {r['family']} | "
            f"{r['alpha_role']} | {r['effect']:g} | {_fmt_req(r['required_n'])} |"
        )
    if refusals:
        lines += [
            "",
            "### Guard refusals (recorded, not simulated)",
            "",
        ]
        for r in refusals:
            lines.append(
                f"- {r['dataset']}/{r['pair_regime']}/{r['family']}: {r['reason']}"
            )
    lines += [
        "",
        "## Window stage — required windows per arm (batch-means Welch, §9.4)",
        "",
        "Chain primaries #13/#14 operate at window level. LABELED CAVEAT: the "
        "pilot is sub-pressure/closed-loop, so simulated window-mean variance "
        "excludes queueing autocorrelation — these window counts are a LOWER "
        "bound; the registration adds margin on top.",
        "",
        "| dataset | family | α role | effect | W (q/window) | required windows |",
        "|---|---|---|---|---|---|",
    ]
    for r in win_required:
        lines.append(
            f"| {r['dataset']} | {r['family']} | {r['alpha_role']} | "
            f"{r['effect']:g} | {r['queries_per_window']} | "
            f"{_fmt_req(r['required_n_windows'])} |"
        )
    lines += [
        "",
        "## Equivalence (TOST) stage — #13 NONE legs + headline #4 machinery",
        "",
        f"P(two-layer equivalence declared | equivalence TRUE) via the "
        f"REGISTERED `conditional_tost` (min_events={DEFAULT_MIN_EVENTS}; "
        f"dominance bootstrap {tost_bootstrap_iters} iters, reduced + "
        "labeled). The NONE-leg family models the CONDITIONAL policy-touched "
        "population (§9.5 hazard); the headline families (amendment A2) run "
        "unconditional (f=1.0) at their declared margins = the family MDEs.",
        "",
        "| family | metric | dataset | regime | margin | event fraction | required n |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in tost_req:
        lines.append(
            f"| {r['family']} | {r['metric']} | {r['dataset']} | "
            f"{r['pair_regime']} | {r['margin']:g} | {r['event_fraction']:g} | "
            f"{_fmt_req(r['required_n'])} |"
        )
    lines += [
        "",
        "## Binary MDE translation (amendment A6 — per-dataset pp effects)",
        "",
        "The binary MDE is uniform in the tie-flip unit but NOT in real "
        "percentage points: tie mass differs per dataset, so the co-primary "
        "set is powered at per-dataset pp-effects. Stated plainly:",
        "",
    ]
    # G15 (2026-08-16): effect columns are DERIVED from the simulated grid —
    # the old hard-coded 0.02/0.05/0.1 keys raised KeyError AFTER a long sim
    # whenever the binary effect grid changed.
    effect_keys: list[str] = sorted(
        {key for t in translation for key in t["implied_pp"]}, key=float
    )
    if not effect_keys:
        lines.append("(no binary flip rows were simulated — nothing to translate)")
    else:
        lines += [
            "| dataset | regime | tie mass | "
            + " | ".join(f"effect {key} -> pp" for key in effect_keys)
            + " |",
            "|---|---|---|" + "---|" * len(effect_keys),
        ]
        for t in translation:
            cells = " | ".join(
                f"{t['implied_pp'][key]:.4f}" if key in t["implied_pp"] else "n/a"
                for key in effect_keys
            )
            lines.append(
                f"| {t['dataset']} | {t['pair_regime']} | "
                f"{t['tie_mass']:.3f} | {cells} |"
            )
    lines += [
        "",
        "## Recommendation at the declared candidate MDEs",
        "",
        "```json",
        json.dumps(recommendation, indent=2, sort_keys=True),
        "```",
        "",
        "## Provenance summary",
        "",
        "- source runs: "
        + ", ".join(f"`{p}`" for p in config["source_runs"].values()),
        f"- pair regimes: {config['pair_regimes']} (binding: `{BINDING_REGIME}`)",
        f"- null model: {config['null_model']}",
        f"- effect-unit reconciliation: {config['effect_unit_reconciliation']}",
        f"- loader validity rule: {config['loader_validity_rule']}",
        f"- alpha roles: {config['alpha_roles']} (both stages)",
        f"- direction labels: per-query continuous = UPPER bound; window "
        f"counts = LOWER bound; binary = real-tie-mass tie-flip",
        f"- qasper: {recommendation['qasper_policy']}",
        f"- repo state: {repo}",
        "",
    ]
    return "\n".join(lines)


def write_outputs(
    pq_table: pd.DataFrame,
    win_table: pd.DataFrame,
    tost_table: pd.DataFrame,
    config: dict[str, Any],
    refusals: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Path]:
    out_dir = cal._check_out_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pq_required = per_query_required(pq_table)
    win_required = window_required(win_table)
    tost_req = tost_required(tost_table)
    recommendation = build_recommendation(
        pq_required, win_required, tost_req, refusals
    )
    # Amendment A6: per-dataset pp translation of the tie-flip binary MDEs.
    flip = pq_table[pq_table["kind"] == "flip"]
    translation: list[dict[str, Any]] = []
    for (dataset, regime), sub in flip.groupby(["dataset", "pair_regime"], sort=True):
        tie_mass = float(sub["tie_mass"].iloc[0])
        translation.append(
            {
                "dataset": dataset,
                "pair_regime": regime,
                "tie_mass": tie_mass,
                "implied_pp": {
                    f"{e:g}": float(e) * tie_mass
                    for e in sorted(sub["effect"].unique())
                },
            }
        )

    tables_path = out_dir / "power_tables.csv"
    pd.concat([pq_table, win_table, tost_table], ignore_index=True).to_csv(
        tables_path, index=False
    )

    report = {
        "stamp": STAMP,
        "config": config,
        "target_power": TARGET_POWER,
        "per_query_required_n": pq_required,
        "window_required_n": win_required,
        "tost_required_n": tost_req,
        "binary_effect_translation": translation,
        "guard_refusals": refusals,
        "recommendation": recommendation,
    }
    report_path = out_dir / "power_sim_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    md_path = out_dir / "POWER_SIM_REPORT.md"
    md_path.write_text(
        _markdown_report(
            config, pq_required, win_required, tost_req, recommendation,
            refusals, translation,
        ),
        encoding="utf-8",
    )

    figures = write_figures(pq_table, win_table, out_dir)

    prov_path = out_dir / "provenance.json"
    prov_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths = {
        "tables": tables_path,
        "report": report_path,
        "markdown": md_path,
        "provenance": prov_path,
    }
    for i, f in enumerate(figures):
        paths[f"figure_{i}"] = f
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "§9.6 simulation-based power on pilot-calibrated noise "
            "(design-input only; pilot data is read-only)."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=_REPO_ROOT / "results" / "phase2",
        help="Pilot archive root holding the three full 100x3 runs (READ-ONLY).",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMS)
    parser.add_argument(
        "--n-grid", type=int, nargs="+", default=list(DEFAULT_N_GRID)
    )
    parser.add_argument(
        "--w-grid", type=int, nargs="+", default=list(DEFAULT_W_GRID)
    )
    parser.add_argument(
        "--nw-grid", type=int, nargs="+", default=list(DEFAULT_NW_GRID)
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(cal.DEFAULT_DATASETS),
        choices=sorted(cal.DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Wiring-sanity mode (seconds): run every stage — including the "
            "TOST stage, whose full-run sizes are module constants — on tiny "
            "grids/sim counts through the identical code. Output is stamped "
            "SMOKE and is NEVER a registration input. Explicit --n-sims/"
            "--n-grid/--w-grid/--nw-grid are overridden by the smoke sizes."
        ),
    )
    args = parser.parse_args(argv)

    if args.smoke:
        args.n_sims = SMOKE_N_SIMS
        args.n_grid = list(SMOKE_N_GRID)
        args.w_grid = list(SMOKE_W_GRID)
        args.nw_grid = list(SMOKE_NW_GRID)
    tost_n_grid: tuple[int, ...] = SMOKE_TOST_N_GRID if args.smoke else TOST_N_GRID
    tost_n_sims: int = SMOKE_TOST_N_SIMS if args.smoke else TOST_N_SIMS
    tost_bootstrap_iters: int = (
        SMOKE_TOST_BOOTSTRAP_ITERS if args.smoke else TOST_BOOTSTRAP_ITERS
    )

    out_dir = cal._check_out_dir(args.out_dir)
    dataset_runs: dict[str, Path] = {}
    for ds in args.datasets:
        run_root = Path(args.results_root) / cal.RUN_OF_DATASET[ds]
        if not run_root.is_dir():
            raise PowerSimCLIError(f"pilot run not found: {run_root}")
        dataset_runs[ds] = run_root

    print(f"[run_power_sim] {STAMP}", flush=True)
    if args.smoke:
        print(f"[run_power_sim] {SMOKE_STAMP}", flush=True)
    repo_state = _git_state()
    if repo_state["dirty"]:
        print(
            "[run_power_sim] WARNING: dirty working tree — artifact is "
            "PROVISIONAL; regenerate at a committed SHA before registration.",
            flush=True,
        )
    config: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_state": repo_state,
        "smoke": bool(args.smoke),
        "seed": args.seed,
        "n_sims": args.n_sims,
        "n_grid": [int(n) for n in args.n_grid],
        "w_grid": [int(w) for w in args.w_grid],
        "nw_grid": [int(n) for n in args.nw_grid],
        "alpha_roles": dict(ALPHA_ROLES),
        "secondary_holm_m": SECONDARY_HOLM_M,
        "secondary_holm_worst_family": SECONDARY_HOLM_WORST_FAMILY,
        "declared_candidate_mdes": dict(DECLARED_MDE),
        "target_power": TARGET_POWER,
        "source_runs": {ds: str(p) for ds, p in sorted(dataset_runs.items())},
        "pair_regimes": {k: list(v) for k, v in sorted(PAIR_REGIMES.items())},
        "binding_pair_regime": BINDING_REGIME,
        "null_model": (
            "per-query: symmetrized REAL pilot arm-pair per-example "
            "differences (binary: concat(d, -d) preserving the real paired "
            "tie mass; continuous: centered concat(c, -c) — UPPER bound, "
            "keeps treatment-effect heterogeneity). window: marginal "
            "per-query pool of arm baselines/no_cache, window means of W "
            "iid resamples — LOWER bound (no queueing autocorrelation). "
            "tost: centered symmetrized real pair diffs (H0-true equivalence)."
        ),
        "effect_unit_reconciliation": EFFECT_UNIT_RECONCILIATION,
        "loader_validity_rule": cal.LOADER_VALIDITY_RULE,
        "registered_test_paths": {
            "per_query_continuous": (
                "tests_by_unit.paired_wilcoxon (two-sided, "
                "zero_method='pratt' — the ADR-0088 registered execution: "
                "zeros kept in the ranking, pinned normal approximation "
                "with continuity correction)"
            ),
            "per_query_binary": (
                "tests_by_unit.mcnemar_binary (two-sided, exact) — "
                + EXACT_MATCH_PROXY_NOTE
            ),
            "window": "tests_by_unit.batch_means_contrast (Welch, two-sided)",
            "tost": "equivalence.conditional_tost (two-layer, §9.5)",
        },
        "tost": {
            "specs": [
                {
                    "family": fam,
                    "metric": metric,
                    "margin": margin,
                    "event_fractions": list(fracs),
                    "binary_pool": binary,
                    "note": note,
                }
                for fam, metric, margin, fracs, binary, note in TOST_SPECS
            ],
            "n_grid": [int(n) for n in tost_n_grid],
            "n_sims": tost_n_sims,
            "bootstrap_iters": tost_bootstrap_iters,
            "min_events": DEFAULT_MIN_EVENTS,
        },
        "window_caveat": (
            "pilot is sub-pressure/closed-loop: window-mean variance excludes "
            "queueing autocorrelation -> window counts are a LOWER bound"
        ),
    }

    pq_table, refusals = run_per_query_stage(
        dataset_runs, seed=args.seed, n_grid=args.n_grid, n_sims=args.n_sims
    )
    win_table = run_window_stage(
        dataset_runs,
        seed=args.seed,
        w_grid=args.w_grid,
        nw_grid=args.nw_grid,
        n_sims=args.n_sims,
    )
    tost_table = run_tost_stage(
        dataset_runs,
        seed=args.seed,
        n_grid=tost_n_grid,
        n_sims=tost_n_sims,
        bootstrap_iters=tost_bootstrap_iters,
    )
    paths = write_outputs(
        pq_table, win_table, tost_table, config, refusals, out_dir
    )

    rec = json.loads((out_dir / "power_sim_report.json").read_text())[
        "recommendation"
    ]
    print("[run_power_sim] recommendation at declared candidate MDEs "
          f"(binding regime {BINDING_REGIME}):", flush=True)
    for fam, roles in rec["per_query"].items():
        if roles.get("status") == "REFUSED_BY_INJECTION_GUARD":
            print(
                f"[run_power_sim]   per_query {fam}: REFUSED_BY_INJECTION_GUARD "
                f"— registered N UNDELIVERED (owner decision pending)",
                flush=True,
            )
        for role, block in roles.items():
            if isinstance(block, dict) and "binding" in block:
                print(
                    f"[run_power_sim]   per_query {fam} @ {role}: "
                    f"MDE={block['mde']:g} -> binding n={block['binding']}",
                    flush=True,
                )
    for name, p in paths.items():
        print(f"[run_power_sim] wrote {name}: {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
