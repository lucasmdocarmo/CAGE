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

Window clustering (owner decision 2026-08-16 c, findings G18/G13): the §9.5
TOST legs on PRESSURE (F2/F3) cells keep the per-example estimand via an
explicit registered carve-out from the §9.4 per-query-under-load
prohibition — but their UNCERTAINTY must respect the queueing dependence
inside a batch-means window. Both entry points therefore accept an optional
``window_ids`` array; when supplied, every resampling step treats the
WINDOW as the resampling unit (block bootstrap: resample windows with
replacement, keep all examples of a sampled window; for the Dirichlet-
process ROPE, the exact Bayesian analogue — window-clustered Dirichlet
weights split evenly inside each window). Point estimates are untouched
(per-example estimand preserved); only CIs/p-values/posteriors widen, so
within-window dependence can never make equivalence artificially EASY.
When ``window_ids`` is absent, behavior is bit-identical to the historical
per-example resampling (sub-pressure F1 cells).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats as _stats

from src.analysis.stats.wlt import _as_float_1d

Verdict = Literal["equivalent", "not-equivalent", "insufficient-n"]
Resampling = Literal["per-example", "window-block"]

# Romano et al. 2006: |delta| < 0.147 is "negligible".
DOMINANCE_MARGIN_NEGLIGIBLE: float = 0.147
# Declared minimum-n rule for the conditional population (audit §2.4).
DEFAULT_MIN_EVENTS: int = 10
# REGISTERED floor (owner decision 2026-08-16 c — surface in the prereg
# text): a block bootstrap over windows needs enough distinct blocks for
# the resampling distribution to carry any information; below 5 unique
# windows in the event population the call is REFUSED (fail-loud), never
# silently degraded to per-example resampling.
MIN_UNIQUE_WINDOWS: int = 5


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
    # Decision 2026-08-16 c: how uncertainty was resampled. "window-block"
    # ⇒ n_windows = unique windows in the EVENT population; "per-example"
    # (the historical path) ⇒ n_windows is None.
    resampling: Resampling = "per-example"
    n_windows: int | None = None

    @property
    def equivalent(self) -> bool:
        """The §9.5 claim: BOTH layers must hold."""
        return (
            self.domain_verdict == "equivalent"
            and self.dominance_verdict == "equivalent"
        )


def _as_window_ids(name: str, values: Any, n: int) -> np.ndarray:
    """Validate a window-id label vector (any hashable dtype) against n."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size != n:
        raise ValueError(f"{name} length {arr.size} != data length {n}")
    return arr


def _event_window_inverse(
    window_ids: Any, mask: np.ndarray, n_total: int
) -> tuple[np.ndarray, int]:
    """Window labels → (inverse indices over the EVENT population, n_windows).

    The unique-window floor is checked by the callers AFTER their
    ``min_events`` gate, so a sparse event population stays the labeled
    ``insufficient-n`` outcome it always was.
    """
    w = _as_window_ids("window_ids", window_ids, n_total)[mask]
    uniq, inv = np.unique(w, return_inverse=True)
    return np.asarray(inv), int(uniq.size)


def _check_window_floor(n_windows: int) -> None:
    if n_windows < MIN_UNIQUE_WINDOWS:
        raise ValueError(
            f"only {n_windows} unique windows in the event population — a "
            f"block bootstrap over windows needs ≥ {MIN_UNIQUE_WINDOWS} "
            f"(registered floor, decision 2026-08-16 c / §9.5); refusing "
            f"rather than silently degrading to per-example resampling"
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
    window_ids: Any | None = None,
) -> ConditionalTostResult:
    """Two-layer TOST on the conditional policy-event population.

    ``a``/``b`` are paired per-query metric values for the two cells;
    ``event_mask`` selects the queries with ≥1 policy event (S2 ledger).
    ``margin`` is the pre-registered domain margin in metric units (> 0).
    Below ``min_events`` both verdicts are ``insufficient-n`` — a labeled
    outcome, not an exception, because sparse event populations are an
    expected data state under mild pressure.

    ``window_ids`` (decision 2026-08-16 c; F2/F3 pressure cells): per-query
    batch-means-window labels aligned with ``a``/``b``. When supplied, BOTH
    layers compute their uncertainty by block bootstrap with the WINDOW as
    the resampling unit (windows drawn with replacement; a drawn window
    contributes all its events):

    - Layer 1 (domain): the analytic paired-t TOST is replaced by its
      percentile-bootstrap dual — p_lower = P̂(mean* ≤ −margin), p_upper =
      P̂(mean* ≥ +margin), ``p_tost`` = max of the two; ``equivalent`` iff
      p_tost < alpha, which is exactly the (1−2α) block-bootstrap-CI-in-
      (−margin, margin) rule. Leaving Layer 1 analytic would let within-
      window dependence shrink its SE and make equivalence artificially
      easy — the precise failure the decision closes.
    - Layer 2 (dominance): the seeded sign bootstrap resamples windows
      instead of examples; same (1−2α) percentile CI and verdict rule.

    Point estimates (``mean_diff``, ``dominance``) stay per-example over
    the full event population — the registered estimand is unchanged.
    Fewer than ``MIN_UNIQUE_WINDOWS`` (= 5, registered floor) unique
    windows in the event population is REFUSED. Without ``window_ids`` the
    historical per-example path runs bit-identically (same rng stream).
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
    if window_ids is not None:
        inv, n_windows = _event_window_inverse(window_ids, mask, arr_a.size)
        resampling: Resampling = "window-block"
    else:
        inv, n_windows = None, None
        resampling = "per-example"
    if n_events < min_events:
        return ConditionalTostResult(
            n_total=arr_a.size, n_events=n_events, n_discordant=n_discordant,
            mean_diff=float(d.mean()) if n_events else float("nan"),
            margin=margin, p_tost=float("nan"),
            domain_verdict="insufficient-n",
            dominance=float("nan"), dominance_margin=dominance_margin,
            dominance_ci_low=float("nan"), dominance_ci_high=float("nan"),
            dominance_verdict="insufficient-n",
            resampling=resampling, n_windows=n_windows,
        )

    mean_d = float(d.mean())
    signs = np.sign(d)
    dominance = float(signs.mean())
    rng = np.random.default_rng(seed)

    if inv is None:
        # Historical per-example path — bit-identical (same rng stream).
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
        idx = rng.integers(0, n_events, size=(bootstrap_iters, n_events))
        boot_dom = signs[idx].mean(axis=1)
    else:
        _check_window_floor(n_windows)
        # Block bootstrap: draw n_windows windows with replacement; each
        # drawn window contributes ALL its events (window sums/counts), so
        # the replicate statistic is the pooled per-example mean — the
        # estimand is preserved while uncertainty is window-clustered.
        counts = np.bincount(inv, minlength=n_windows).astype(float)
        sum_d = np.bincount(inv, weights=d, minlength=n_windows)
        sum_sign = np.bincount(inv, weights=signs, minlength=n_windows)
        idx = rng.integers(0, n_windows, size=(bootstrap_iters, n_windows))
        tot = counts[idx].sum(axis=1)
        boot_mean = sum_d[idx].sum(axis=1) / tot
        boot_dom = sum_sign[idx].sum(axis=1) / tot
        # Percentile-bootstrap dual of TOST (boundary draws count toward
        # non-equivalence — conservative): p < alpha ⟺ the (1−2α)
        # percentile CI of the block-bootstrap mean lies inside ±margin.
        p_lower = float(np.mean(boot_mean <= -margin))
        p_upper = float(np.mean(boot_mean >= margin))
        p_tost = max(p_lower, p_upper)
    domain_verdict: Verdict = "equivalent" if p_tost < alpha else "not-equivalent"

    # (1 - 2*alpha)*100% CI — matches two one-sided tests each at level alpha
    # (the same alpha the domain/Layer-1 test above uses), not a fixed 95%.
    ci_low = float(np.percentile(boot_dom, 100 * alpha))
    ci_high = float(np.percentile(boot_dom, 100 * (1 - alpha)))
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
        resampling=resampling,
        n_windows=n_windows,
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
    # Decision 2026-08-16 c: posterior-weight clustering. "window-block" ⇒
    # n_windows = unique windows in the EVENT population; "per-example"
    # (the historical path) ⇒ n_windows is None.
    resampling: Resampling = "per-example"
    n_windows: int | None = None


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
    window_ids: Any | None = None,
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

    ``window_ids`` (decision 2026-08-16 c): the ROPE has no literal
    bootstrap, so the block-bootstrap decision maps to its Bayesian
    analogue — the cluster Bayesian bootstrap. Per draw,
    u ~ Dirichlet(s, 1, …, 1) over the prior pseudo-observation plus the W
    unique event-population windows (ONE unit of concentration per WINDOW —
    the window is the exchangeable resampling unit, mirroring "draw windows
    with replacement"); every event of window w carries raw weight u_w and
    the full atom-weight vector is renormalized. A window's posterior mass
    is then n_w·u_w / (u₀ + Σ n_w·u_w) — the same ratio-estimator form as
    the frequentist block bootstrap, so the per-example estimand is
    preserved (posterior-mean event weight ≈ 1/(s + n)) while all events
    of a window co-move with between-window dispersion matched to cluster
    resampling. (Splitting Dirichlet(s, n₁, …, n_W) mass evenly inside
    windows would be WRONG: by Dirichlet aggregation it is distributionally
    identical to the unclustered posterior on window-constant data —
    no clustering at all.) Fewer than ``MIN_UNIQUE_WINDOWS`` (= 5,
    registered floor) unique windows is REFUSED. Without ``window_ids``
    the historical per-example weighting runs bit-identically (same rng
    stream).
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
    if window_ids is not None:
        inv, n_windows = _event_window_inverse(window_ids, mask, arr_a.size)
        resampling: Resampling = "window-block"
    else:
        inv, n_windows = None, None
        resampling = "per-example"
    if n_events < min_events:
        nan = float("nan")
        return RopeResult(
            n_total=arr_a.size, n_events=n_events, rope=rope,
            prior_pseudocount=prior_pseudocount, n_samples=n_samples,
            posterior_threshold=posterior_threshold,
            p_left=nan, p_rope=nan, p_right=nan,
            mean_theta_left=nan, mean_theta_rope=nan, mean_theta_right=nan,
            verdict="insufficient-n",
            resampling=resampling, n_windows=n_windows,
        )

    z = np.concatenate(([0.0], d))
    pair_mean = (z[:, None] + z[None, :]) / 2.0
    left_mask = pair_mean < -rope
    right_mask = pair_mean > rope
    rng = np.random.default_rng(seed)
    if inv is None:
        # Historical per-example path — bit-identical (same rng stream).
        alpha_dir = np.concatenate(([prior_pseudocount], np.ones(n_events)))
        weights = rng.dirichlet(alpha_dir, size=n_samples)
    else:
        _check_window_floor(n_windows)
        # Cluster Bayesian bootstrap (see docstring): ONE unit of Dirichlet
        # concentration per WINDOW; every event of a window carries its
        # window's raw weight, then the whole atom vector (prior + events)
        # is renormalized. Events of one window co-move with the
        # between-window dispersion of cluster resampling, while a window's
        # mass keeps the ratio-estimator form n_w·u_w / Σ (estimand
        # preserved).
        alpha_dir = np.concatenate(([prior_pseudocount], np.ones(n_windows)))
        u = rng.dirichlet(alpha_dir, size=n_samples)
        raw = np.empty((n_samples, n_events + 1))
        raw[:, 0] = u[:, 0]
        raw[:, 1:] = u[:, 1:][:, inv]
        weights = raw / raw.sum(axis=1, keepdims=True)
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
        resampling=resampling,
        n_windows=n_windows,
    )
