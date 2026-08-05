"""Conditional two-layer TOST + Bayesian ROPE sensitivity (PUBLICATION.md
§9.5; audit §2.4/§2.8).

Equivalence claims (the §8.11 policy-NONE fingerprints) are tested on the
CONDITIONAL policy-event population — only the queries the policy actually
touched (caller passes the S2-ledger event mask). Rationale: at T=0
saturation the tie mass makes an unconditional TOST pass trivially (the pilot
recorded 15/289 discordant grounding pairs) while conditional harm stays
invisible; the discordant-pair count is therefore surfaced on every result.

Two layers, BOTH required for an equivalence claim:
- Layer 1 (domain): paired-t TOST against a domain-justified margin in metric
  units (Lakens).
- Layer 2 (standardized): the tie-aware paired dominance statistic
  (#(d>0) − #(d<0)) / n — ties in the denominator, never the between-groups
  Cliff's delta — with a seeded bootstrap CI; equivalent iff the whole CI
  lies inside ±0.147 (Romano 2006 "negligible").

``rope_sensitivity`` is the §9.5 companion instrument ("Bayesian ROPE
reported as a sensitivity line beside every TOST conclusion"): the Benavoli
et al. 2017 Bayesian signed-rank (Dirichlet-process posterior with a prior
pseudo-observation at 0 — the tie-robust choice the audit §2.4 names),
reporting posterior probabilities of the left / ROPE / right regions. It is
a sensitivity LINE, never the confirmatory gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats as _stats

from src.analysis.stats.wlt import _as_float_1d

Verdict = Literal["equivalent", "not-equivalent", "insufficient-n"]

# Romano et al. 2006: |delta| < 0.147 is "negligible".
DOMINANCE_MARGIN_NEGLIGIBLE: float = 0.147
# Declared minimum-n rule for the conditional population (audit §2.4).
DEFAULT_MIN_EVENTS: int = 10


@dataclass(frozen=True)
class ConditionalTostResult:
    n_total: int
    n_events: int
    n_discordant: int
    mean_diff: float
    margin: float
    p_tost: float
    domain_verdict: Verdict
    dominance: float
    dominance_margin: float
    dominance_ci_low: float
    dominance_ci_high: float
    dominance_verdict: Verdict

    @property
    def equivalent(self) -> bool:
        """The §9.5 claim: BOTH layers must hold."""
        return (
            self.domain_verdict == "equivalent"
            and self.dominance_verdict == "equivalent"
        )


def _as_bool_mask(name: str, values: Any, n: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.dtype != bool:
        as_float = np.asarray(arr, dtype=float)
        if not np.all(np.isfinite(as_float)) or not np.all(
            np.isin(as_float, (0.0, 1.0))
        ):
            raise ValueError(f"{name} must be boolean (the S2 policy-event mask)")
        arr = as_float.astype(bool)
    if arr.size != n:
        raise ValueError(f"{name} length {arr.size} != data length {n}")
    return arr


def conditional_tost(
    a: Any,
    b: Any,
    event_mask: Any,
    *,
    margin: float,
    alpha: float = 0.05,
    dominance_margin: float = DOMINANCE_MARGIN_NEGLIGIBLE,
    min_events: int = DEFAULT_MIN_EVENTS,
    bootstrap_iters: int = 10_000,
    seed: int = 42,
) -> ConditionalTostResult:
    """Two-layer TOST on the conditional policy-event population.

    ``a``/``b`` are paired per-query metric values for the two cells;
    ``event_mask`` selects the queries with ≥1 policy event (S2 ledger).
    ``margin`` is the pre-registered domain margin in metric units (> 0).
    Below ``min_events`` both verdicts are ``insufficient-n`` — a labeled
    outcome, not an exception, because sparse event populations are an
    expected data state under mild pressure.
    """
    if margin <= 0.0 or not np.isfinite(margin):
        raise ValueError(f"margin={margin} must be finite and > 0")
    if dominance_margin <= 0.0 or dominance_margin >= 1.0:
        raise ValueError(f"dominance_margin={dominance_margin} must be in (0, 1)")
    if not 0.0 < alpha < 0.5:
        raise ValueError(
            f"alpha={alpha} must be in (0, 0.5) — alpha is a one-sided TOST "
            "significance level (Westlake 1976/Schuirmann 1987), and the "
            "dominance-layer CI below is built as a (1 - 2*alpha)*100% "
            "bootstrap interval, which inverts (ci_low > ci_high) for "
            "alpha >= 0.5"
        )
    if min_events < 2:
        raise ValueError(f"min_events={min_events} must be ≥ 2")
    if bootstrap_iters < 100:
        raise ValueError(f"bootstrap_iters={bootstrap_iters} must be ≥ 100")
    arr_a = _as_float_1d("a", a)
    arr_b = _as_float_1d("b", b)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"paired arrays differ in length: {arr_a.size} vs {arr_b.size}")
    mask = _as_bool_mask("event_mask", event_mask, arr_a.size)

    d = (arr_a - arr_b)[mask]
    n_events = int(d.size)
    n_discordant = int(np.count_nonzero(d != 0.0))
    if n_events < min_events:
        return ConditionalTostResult(
            n_total=arr_a.size, n_events=n_events, n_discordant=n_discordant,
            mean_diff=float(d.mean()) if n_events else float("nan"),
            margin=margin, p_tost=float("nan"),
            domain_verdict="insufficient-n",
            dominance=float("nan"), dominance_margin=dominance_margin,
            dominance_ci_low=float("nan"), dominance_ci_high=float("nan"),
            dominance_verdict="insufficient-n",
        )

    mean_d = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        # Constant difference: equivalence is exact, not estimated.
        p_tost = 0.0 if abs(mean_d) < margin else 1.0
    else:
        se = sd / np.sqrt(n_events)
        df = n_events - 1
        t_lower = (mean_d + margin) / se
        t_upper = (mean_d - margin) / se
        p_lower = float(_stats.t.sf(t_lower, df))
        p_upper = float(_stats.t.cdf(t_upper, df))
        p_tost = max(p_lower, p_upper)
    domain_verdict: Verdict = "equivalent" if p_tost < alpha else "not-equivalent"

    signs = np.sign(d)
    dominance = float(signs.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_events, size=(bootstrap_iters, n_events))
    boot = signs[idx].mean(axis=1)
    # (1 - 2*alpha)*100% CI — matches two one-sided tests each at level alpha
    # (the same alpha the domain/Layer-1 t-test above uses), not a fixed 95%.
    ci_low = float(np.percentile(boot, 100 * alpha))
    ci_high = float(np.percentile(boot, 100 * (1 - alpha)))
    dominance_verdict: Verdict = (
        "equivalent"
        if max(abs(ci_low), abs(ci_high)) < dominance_margin
        else "not-equivalent"
    )
    return ConditionalTostResult(
        n_total=arr_a.size,
        n_events=n_events,
        n_discordant=n_discordant,
        mean_diff=mean_d,
        margin=margin,
        p_tost=p_tost,
        domain_verdict=domain_verdict,
        dominance=dominance,
        dominance_margin=dominance_margin,
        dominance_ci_low=ci_low,
        dominance_ci_high=ci_high,
        dominance_verdict=dominance_verdict,
    )


@dataclass(frozen=True)
class RopeResult:
    """Posterior region probabilities from the Bayesian signed-rank ROPE.

    ``p_left/p_rope/p_right`` = fraction of posterior draws in which that
    region carries the largest probability mass; ``mean_theta_*`` = posterior
    mean mass per region. ``verdict`` is a labeled READING (equivalent iff
    p_rope ≥ posterior_threshold), reported beside — never instead of — the
    TOST conclusion (§9.5).
    """

    n_total: int
    n_events: int
    rope: float
    prior_pseudocount: float
    n_samples: int
    posterior_threshold: float
    p_left: float
    p_rope: float
    p_right: float
    mean_theta_left: float
    mean_theta_rope: float
    mean_theta_right: float
    verdict: Verdict


def rope_sensitivity(
    a: Any,
    b: Any,
    event_mask: Any,
    *,
    rope: float,
    prior_pseudocount: float = 0.5,
    n_samples: int = 2_000,
    seed: int = 42,
    min_events: int = DEFAULT_MIN_EVENTS,
    posterior_threshold: float = 0.95,
) -> RopeResult:
    """Benavoli et al. 2017 Bayesian signed-rank with a ROPE of ±``rope``.

    Same pairing and conditional-population semantics as ``conditional_tost``
    (``rope`` is naturally the same domain margin). Dirichlet-process
    posterior: weights w ~ Dirichlet(s, 1, …, 1) over the prior
    pseudo-observation z₀=0 plus the n conditional differences; per draw the
    region masses are θ_left = Σ wᵢwⱼ 1[(zᵢ+zⱼ)/2 < −rope], θ_right the
    mirror, θ_rope the remainder. O(n²) memory in the event count — fine for
    conditional populations. Deterministic given ``seed``. Below
    ``min_events`` the outcome is the labeled ``insufficient-n``, matching
    ``conditional_tost``.
    """
    if not np.isfinite(rope) or rope <= 0.0:
        raise ValueError(f"rope={rope} must be finite and > 0")
    if not np.isfinite(prior_pseudocount) or prior_pseudocount <= 0.0:
        raise ValueError(f"prior_pseudocount={prior_pseudocount} must be > 0")
    if n_samples < 100:
        raise ValueError(f"n_samples={n_samples} must be ≥ 100")
    if not 0.0 < posterior_threshold < 1.0:
        raise ValueError(f"posterior_threshold={posterior_threshold} must be in (0, 1)")
    if min_events < 2:
        raise ValueError(f"min_events={min_events} must be ≥ 2")
    arr_a = _as_float_1d("a", a)
    arr_b = _as_float_1d("b", b)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"paired arrays differ in length: {arr_a.size} vs {arr_b.size}")
    mask = _as_bool_mask("event_mask", event_mask, arr_a.size)

    d = (arr_a - arr_b)[mask]
    n_events = int(d.size)
    if n_events < min_events:
        nan = float("nan")
        return RopeResult(
            n_total=arr_a.size, n_events=n_events, rope=rope,
            prior_pseudocount=prior_pseudocount, n_samples=n_samples,
            posterior_threshold=posterior_threshold,
            p_left=nan, p_rope=nan, p_right=nan,
            mean_theta_left=nan, mean_theta_rope=nan, mean_theta_right=nan,
            verdict="insufficient-n",
        )

    z = np.concatenate(([0.0], d))
    pair_mean = (z[:, None] + z[None, :]) / 2.0
    left_mask = pair_mean < -rope
    right_mask = pair_mean > rope
    alpha_dir = np.concatenate(([prior_pseudocount], np.ones(n_events)))
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(alpha_dir, size=n_samples)
    theta_left = ((weights @ left_mask) * weights).sum(axis=1)
    theta_right = ((weights @ right_mask) * weights).sum(axis=1)
    theta_rope = 1.0 - theta_left - theta_right

    stacked = np.stack([theta_left, theta_rope, theta_right])
    winners = np.argmax(stacked, axis=0)
    p_left = float(np.mean(winners == 0))
    p_rope = float(np.mean(winners == 1))
    p_right = float(np.mean(winners == 2))
    verdict: Verdict = (
        "equivalent" if p_rope >= posterior_threshold else "not-equivalent"
    )
    return RopeResult(
        n_total=arr_a.size,
        n_events=n_events,
        rope=rope,
        prior_pseudocount=prior_pseudocount,
        n_samples=n_samples,
        posterior_threshold=posterior_threshold,
        p_left=p_left,
        p_rope=p_rope,
        p_right=p_right,
        mean_theta_left=float(theta_left.mean()),
        mean_theta_rope=float(theta_rope.mean()),
        mean_theta_right=float(theta_right.mean()),
        verdict=verdict,
    )
