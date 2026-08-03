"""§9.6 — simulation-based power: the seeded campaign-power skeleton.

Pilot variance is unusable for analytic power (zero in-regime cells, zero
Qasper, tie saturation — audit §2.5), so power is SIMULATION-based and its
output IS the per-window query count / window count the charter leaves open:
the smallest n on the grid reaching 0.8 power at the pre-declared MDE.

``variance_model`` is the null-noise generator ``(rng, n) -> ndarray`` of n
per-unit differences under H0. IT IS CALIBRATED ON PILOT DATA BY THE CALLER
(e.g. resampling the pilot per-query archives inside the §8.13 dry-run; for
tie-heavy metrics it must emit the observed tie mass). How the effect enters
is the ``injection`` model:

- ``shift_injection`` (default): additive location shift — continuous
  metrics (TTFT deltas) only.
- ``tie_flip_injection``: converts an ``effect`` fraction of the null TIES
  into discordant wins — MANDATORY for tie-heavy metrics (predicate /
  grounding), because an additive shift turns every tie into signed evidence
  and makes power track N instead of the discordant-pair process, violating
  audit §2.5 / §9.6 (the 2026-08-02 finding: 95%-tie pilot shape at effect
  0.02 → additive power ≈ 1.0 vs honest flip power ≈ 0.27).

This module is the deterministic skeleton only: code + seed (+ the named
injection model) are what get registered.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

VarianceModel = Callable[[np.random.Generator, int], np.ndarray]
PairedTestFn = Callable[[np.ndarray], float]
# (rng, null_diffs, effect) -> diffs under the alternative.
InjectionModel = Callable[[np.random.Generator, np.ndarray, float], np.ndarray]

_CI_CONFIDENCE = 0.95


class PowerSimError(ValueError):
    """Invalid simulation input or an unusable model/test result (fail closed)."""


def wilcoxon_signed_p(diffs: np.ndarray) -> float:
    """Default paired test: one-sample Wilcoxon signed-rank on the differences.

    Matches the §9.4 unloaded-cell unit of analysis. All-zero differences carry
    no evidence -> p = 1.0 (scipy would raise instead of deciding).
    """
    if not np.any(diffs != 0.0):
        return 1.0
    return float(wilcoxon(diffs, zero_method="wilcox").pvalue)


def shift_injection(
    rng: np.random.Generator, null_diffs: np.ndarray, effect: float
) -> np.ndarray:
    """Additive location shift (continuous metrics). ``rng`` unused by design —
    the signature is the shared ``InjectionModel`` contract."""
    return null_diffs + effect


def tie_flip_injection(
    rng: np.random.Generator,
    null_diffs: np.ndarray,
    effect: float,
    *,
    magnitude: float = 1.0,
) -> np.ndarray:
    """Discordant-pair injection for tie-heavy metrics (audit §2.5 / §9.6).

    Flips ``round(effect · n_ties)`` of the exact-zero null differences to
    ``+magnitude`` discordant wins; ``effect`` is the flipped FRACTION OF TIES
    in [0, 1]. A tie-free draw under a positive effect fails loud — it means
    the variance model does not reproduce the tie-heavy premise this
    injection exists for.
    """
    if not np.isfinite(effect) or not 0.0 <= effect <= 1.0:
        raise PowerSimError(
            f"tie_flip_injection effect is a tie fraction in [0, 1], got {effect!r}"
        )
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        raise PowerSimError(f"magnitude must be finite and > 0, got {magnitude!r}")
    if effect == 0.0:
        return null_diffs.copy()
    tie_idx = np.flatnonzero(null_diffs == 0.0)
    if tie_idx.size == 0:
        raise PowerSimError(
            "tie_flip_injection found no ties in the null draw; a tie-heavy "
            "variance model must emit the observed tie mass (audit §2.5)"
        )
    n_flip = int(round(effect * tie_idx.size))
    flipped = null_diffs.copy()
    if n_flip > 0:
        flipped[rng.choice(tie_idx, size=n_flip, replace=False)] = magnitude
    return flipped


def simulate_campaign(
    effect_grid: Sequence[float],
    n_grid: Sequence[int],
    variance_model: VarianceModel,
    n_sims: int,
    seed: int,
    *,
    alpha: float = 0.05,
    test_fn: PairedTestFn = wilcoxon_signed_p,
    injection: InjectionModel = shift_injection,
) -> pd.DataFrame:
    """Power table over an effect × n grid; pure and deterministic given seed.

    For each (effect, n) grid point runs ``n_sims`` simulations: draw n null
    differences from ``variance_model``, apply ``injection`` (default =
    additive shift; tie-heavy metrics MUST pass ``tie_flip_injection`` so the
    simulated power tracks the discordant-pair process — module docstring),
    apply ``test_fn``; power = rejection fraction with an exact binomial CI.
    Each grid point gets an independent child seed from one
    ``SeedSequence(seed)`` spawn (positional: the same seed + the same grids
    reproduce exactly; reordering a grid is a different registered
    simulation).

    Returns columns: effect, n, n_sims, alpha, power, ci_low, ci_high.
    """
    effects = [float(e) for e in effect_grid]
    ns = [int(n) for n in n_grid]
    if not effects or not ns:
        raise PowerSimError("effect_grid and n_grid must be non-empty")
    if any(not np.isfinite(e) for e in effects):
        raise PowerSimError(f"effect_grid contains non-finite values: {effects}")
    if any(n < 2 for n in ns):
        raise PowerSimError(f"n_grid values must be >= 2, got {ns}")
    if n_sims < 1:
        raise PowerSimError(f"n_sims must be >= 1, got {n_sims}")
    if not 0.0 < alpha < 1.0:
        raise PowerSimError(f"alpha must be in (0, 1), got {alpha}")

    children = np.random.SeedSequence(seed).spawn(len(effects) * len(ns))
    rows: list[dict[str, float | int]] = []
    for point, (effect, n) in enumerate((e, n) for e in effects for n in ns):
        rng = np.random.default_rng(children[point])
        rejections = 0
        for _ in range(n_sims):
            noise = np.asarray(variance_model(rng, n), dtype=float)
            if noise.shape != (n,):
                raise PowerSimError(
                    f"variance_model returned shape {noise.shape}, expected ({n},)"
                )
            if not np.all(np.isfinite(noise)):
                raise PowerSimError("variance_model returned non-finite values")
            injected = np.asarray(injection(rng, noise, effect), dtype=float)
            if injected.shape != (n,):
                raise PowerSimError(
                    f"injection returned shape {injected.shape}, expected ({n},)"
                )
            if not np.all(np.isfinite(injected)):
                raise PowerSimError("injection returned non-finite values")
            p = float(test_fn(injected))
            if not (0.0 <= p <= 1.0) or not np.isfinite(p):
                raise PowerSimError(f"test_fn returned invalid p-value {p!r}")
            if p < alpha:
                rejections += 1
        ci = binomtest(rejections, n_sims).proportion_ci(
            confidence_level=_CI_CONFIDENCE, method="exact"
        )
        rows.append(
            {
                "effect": effect,
                "n": n,
                "n_sims": n_sims,
                "alpha": alpha,
                "power": rejections / n_sims,
                "ci_low": float(ci.low),
                "ci_high": float(ci.high),
            }
        )
    return pd.DataFrame(rows)


def required_n(
    power_table: pd.DataFrame, effect: float, *, target_power: float = 0.8
) -> int:
    """Smallest grid n whose simulated power reaches the target at ``effect``.

    Raises (fail closed) when the effect is off the grid or no grid n reaches
    the target — the caller must extend the grid, not extrapolate.
    """
    if not 0.0 < target_power <= 1.0:
        raise PowerSimError(f"target_power must be in (0, 1], got {target_power!r}")
    at_effect = power_table[power_table["effect"] == float(effect)]
    if at_effect.empty:
        raise PowerSimError(
            f"effect {effect!r} not on the simulated grid: "
            f"{sorted(power_table['effect'].unique())}"
        )
    reaching = at_effect[at_effect["power"] >= target_power]
    if reaching.empty:
        raise PowerSimError(
            f"no simulated n reaches power {target_power} at effect {effect!r} "
            f"(max power {at_effect['power'].max():.3f}); extend n_grid"
        )
    return int(reaching["n"].min())
