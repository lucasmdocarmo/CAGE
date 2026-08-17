"""Tests by unit of analysis (PUBLICATION.md §9.4; audit §2.3).

- Unloaded paired cells → ``paired_wilcoxon`` (per-query), with the PAIRED
  Cliff's delta: the tie-aware dominance over within-pair differences,
  (#(d>0) − #(d<0)) / n_pairs. The pilot's ``statistical_tests.py`` computed
  Cliff's delta UNPAIRED on paired arrays (all a×b cross pairs), which
  vanishes under between-query spread — the audited defect this module fixes.
- Binary outcomes (the §8.5 Y predicate) → ``mcnemar_binary`` on discordant
  pairs, exact binomial.
- Loaded/pressure cells → ``batch_means_contrast`` on window means (Welch t).
  Per-query pairing under load is PROHIBITED (queueing autocorrelation
  destroys exchangeability) — the function refuses per-query-sized input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats as _stats

from src.analysis.stats.wlt import _as_float_1d

Alternative = Literal["two-sided", "greater", "less"]
ZeroMethod = Literal["wilcox", "pratt"]

_ALTERNATIVES: frozenset[str] = frozenset({"two-sided", "greater", "less"})
_ZERO_METHODS: frozenset[str] = frozenset({"wilcox", "pratt"})
# §9.4 guard: batch-means inputs are per-window aggregates (≥3 replications,
# at most a few dozen windows); a per-query vector under load is 100s long.
DEFAULT_MAX_WINDOWS: int = 50


def _check_alternative(alternative: str) -> None:
    if alternative not in _ALTERNATIVES:
        raise ValueError(
            f"alternative={alternative!r} not in {sorted(_ALTERNATIVES)}"
        )


def _check_zero_method(zero_method: str) -> None:
    if zero_method not in _ZERO_METHODS:
        raise ValueError(
            f"zero_method={zero_method!r} not in {sorted(_ZERO_METHODS)}"
        )


def _paired(a: Any, b: Any) -> tuple[np.ndarray, np.ndarray]:
    arr_a = _as_float_1d("a", a)
    arr_b = _as_float_1d("b", b)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"paired arrays differ in length: {arr_a.size} vs {arr_b.size}")
    return arr_a, arr_b


@dataclass(frozen=True)
class PairedWilcoxonResult:
    n_pairs: int
    n_pos: int
    n_neg: int
    n_zero: int
    statistic: float
    p_value: float
    cliffs_delta_paired: float
    alternative: Alternative
    # Effective n (decision 2026-08-16 b): the count of non-zero pairs, the
    # sample size that actually feeds the signed-rank statistic — reported
    # beside the §8.13 W/L/T triple because under pilot-level tie saturation
    # (~95% zeros) n_pairs wildly overstates the evidence.
    n_nonzero: int
    zero_method: ZeroMethod


def paired_wilcoxon(
    a: Any,
    b: Any,
    *,
    alternative: Alternative = "two-sided",
    zero_method: ZeroMethod = "wilcox",
) -> PairedWilcoxonResult:
    """Wilcoxon signed-rank on per-query paired values (sub-pressure only).

    ``cliffs_delta_paired`` is the tie-aware within-pair dominance statistic
    (ties counted in the denominator), NOT the between-groups delta. The
    n_pos/n_neg/n_zero triple is the §8.13-mandatory win/loss/tie raw form
    (direction-neutral: n_pos counts a > b). ``n_nonzero`` = n_pos + n_neg is
    the effective n.

    Sidedness (owner decision 2026-08-16 a): ``alternative`` passes through
    to scipy — ``"greater"`` tests H1: a > b (positive differences dominate),
    ``"less"`` tests H1: a < b. The REGISTERED per-row sidedness is supplied
    explicitly by the driver from the §9.3 family map; the ``"two-sided"``
    default here is back-compat only, never the registration.

    Tie handling (owner decision 2026-08-16 b) — the pinned
    exact-vs-approx rule:

    - ``zero_method="wilcox"`` (back-compat default): zero differences are
      discarded before ranking; the scipy call keeps scipy's own defaults
      (``method="auto"``, ``correction=False``). scipy's auto rule (pinned
      scipy 1.18): exact distribution iff there are no zeros AND no tied
      |d| AND n ≤ 50; exhaustive permutation iff ties/zeros with n ≤ 13;
      otherwise the normal approximation WITHOUT continuity correction.
    - ``zero_method="pratt"`` (the REGISTERED value, passed explicitly by
      the driver per §9.4): zeros are kept in the ranking (Pratt 1959), so
      the tie mass penalizes the statistic instead of silently shrinking n.
      The execution is pinned to the normal approximation WITH continuity
      correction (``method="approx"``, ``correction=True``) UNCONDITIONALLY:
      scipy's auto rule would otherwise switch between permutation and
      asymptotic paths depending on the realized tie pattern, making the
      executed test data-dependent — unacceptable for a registered
      procedure under ~95% tie saturation (audit S11 / finding G8).

    The all-zero vector short-circuits to the T=0 identity outcome
    (p = 1.0) under BOTH zero methods — scipy raises on it for either.
    """
    _check_alternative(alternative)
    _check_zero_method(zero_method)
    arr_a, arr_b = _paired(a, b)
    diffs = arr_a - arr_b
    n_pos = int(np.count_nonzero(diffs > 0))
    n_neg = int(np.count_nonzero(diffs < 0))
    n_zero = diffs.size - n_pos - n_neg
    n_nonzero = n_pos + n_neg
    delta = (n_pos - n_neg) / diffs.size
    if n_nonzero == 0:
        # Legitimate T=0 identity outcome (e.g. B1 vs B2), not an error.
        return PairedWilcoxonResult(
            n_pairs=diffs.size, n_pos=0, n_neg=0, n_zero=n_zero,
            statistic=0.0, p_value=1.0, cliffs_delta_paired=0.0,
            alternative=alternative, n_nonzero=0, zero_method=zero_method,
        )
    if zero_method == "pratt":
        # Registered execution: pinned normal approximation + continuity
        # correction (see docstring). "approx" is scipy's asymptotic path.
        res = _stats.wilcoxon(
            diffs, zero_method="pratt", alternative=alternative,
            method="approx", correction=True,
        )
    else:
        # Back-compat path: bit-identical to the pre-decision behavior
        # (scipy defaults: method="auto", correction=False).
        res = _stats.wilcoxon(diffs, zero_method="wilcox", alternative=alternative)
    return PairedWilcoxonResult(
        n_pairs=diffs.size,
        n_pos=n_pos,
        n_neg=n_neg,
        n_zero=n_zero,
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        cliffs_delta_paired=float(delta),
        alternative=alternative,
        n_nonzero=n_nonzero,
        zero_method=zero_method,
    )


@dataclass(frozen=True)
class McNemarResult:
    n_pairs: int
    n_11: int
    n_00: int
    n_10: int
    n_01: int
    p_value: float
    proportion_diff: float
    alternative: Alternative

    @property
    def n_discordant(self) -> int:
        return self.n_10 + self.n_01


def _as_binary(name: str, values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if arr.dtype != bool:
        as_float = np.asarray(arr, dtype=float)
        if not np.all(np.isfinite(as_float)) or not np.all(
            np.isin(as_float, (0.0, 1.0))
        ):
            raise ValueError(
                f"{name} must be boolean or strictly 0/1 (binary outcome, "
                f"e.g. the §8.5 Y predicate); got values outside {{0, 1}}"
            )
        return as_float.astype(bool)
    return arr


def mcnemar_binary(
    a: Any, b: Any, *, alternative: Alternative = "two-sided"
) -> McNemarResult:
    """McNemar's test on paired binary outcomes: exact binomial on the
    discordant pairs (n_10 successes out of n_10 + n_01 at p=0.5).

    Direction mapping (owner decision 2026-08-16 a) — under
    H0, n_10 ~ Binomial(n_discordant, ½); with b := n_10:

    - ``alternative="greater"`` tests H1: arm ``a`` succeeds where ``b``
      fails MORE often than the reverse (P(a=1,b=0) > P(a=0,b=1), i.e.
      ``a`` is the better arm); p = P[X ≥ b] (exact binomial upper tail,
      ``scipy.stats.binomtest`` with ``alternative="greater"``).
    - ``alternative="less"`` tests H1: arm ``a`` is WORSE than arm ``b``;
      p = P[X ≤ b] (lower tail).
    - ``alternative="two-sided"`` is scipy's exact two-sided binomial test.

    The registered per-row sidedness is supplied explicitly by the driver
    from the §9.3 family map; ``"two-sided"`` is the back-compat default.
    Zero discordant pairs is the identity outcome (p = 1.0) under every
    alternative.
    """
    _check_alternative(alternative)
    arr_a = _as_binary("a", a)
    arr_b = _as_binary("b", b)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"paired arrays differ in length: {arr_a.size} vs {arr_b.size}")
    n_11 = int(np.count_nonzero(arr_a & arr_b))
    n_00 = int(np.count_nonzero(~arr_a & ~arr_b))
    n_10 = int(np.count_nonzero(arr_a & ~arr_b))
    n_01 = int(np.count_nonzero(~arr_a & arr_b))
    n_disc = n_10 + n_01
    if n_disc == 0:
        p_value = 1.0
    else:
        p_value = float(
            _stats.binomtest(n_10, n_disc, 0.5, alternative=alternative).pvalue
        )
    return McNemarResult(
        n_pairs=arr_a.size,
        n_11=n_11,
        n_00=n_00,
        n_10=n_10,
        n_01=n_01,
        p_value=p_value,
        proportion_diff=float(arr_a.mean() - arr_b.mean()),
        alternative=alternative,
    )


@dataclass(frozen=True)
class BatchMeansResult:
    n_windows_a: int
    n_windows_b: int
    mean_a: float
    mean_b: float
    mean_diff: float
    statistic: float
    df: float
    p_value: float
    ci95_low: float
    ci95_high: float
    alternative: Alternative


def batch_means_contrast(
    means_a: Any,
    means_b: Any,
    *,
    alternative: Alternative = "two-sided",
    max_windows: int = DEFAULT_MAX_WINDOWS,
) -> BatchMeansResult:
    """Welch t contrast on WINDOW-LEVEL batch means (loaded cells, §9.4).

    Inputs are per-window aggregates (Jain-warmup-removed batch means across
    the ≥3 replications), never raw per-query values: any input longer than
    ``max_windows`` is refused because per-query pairing under load is
    prohibited by the registration.

    Direction mapping (owner decision 2026-08-16 a): ``alternative`` passes
    through to the one-sided Welch t — ``"greater"`` tests
    H1: mean(means_a) > mean(means_b), ``"less"`` the reverse. The
    registered per-row sidedness is supplied explicitly by the driver from
    the §9.3 family map; ``"two-sided"`` is the back-compat default. The
    ``ci95_*`` bounds stay a descriptive two-sided 95% CI regardless of
    ``alternative``.
    """
    _check_alternative(alternative)
    arr_a = _as_float_1d("means_a", means_a)
    arr_b = _as_float_1d("means_b", means_b)
    for name, arr in (("means_a", arr_a), ("means_b", arr_b)):
        if arr.size > max_windows:
            raise ValueError(
                f"{name} has {arr.size} values > max_windows={max_windows}: "
                f"this looks like PER-QUERY data — per-query pairing under "
                f"load is prohibited (§9.4); pass window-level batch means"
            )
        if arr.size < 2:
            raise ValueError(
                f"{name} needs ≥2 windows for a variance estimate, got {arr.size}"
            )
    mean_a = float(arr_a.mean())
    mean_b = float(arr_b.mean())
    var_a = float(arr_a.var(ddof=1))
    var_b = float(arr_b.var(ddof=1))
    n_a, n_b = arr_a.size, arr_b.size
    if var_a == 0.0 and var_b == 0.0:
        # Degenerate but well-defined: identical constants tie, distinct
        # constants separate with certainty. Raise rather than fabricate a t.
        raise ValueError(
            "both window-mean samples have zero variance — a Welch contrast "
            "is undefined; inspect the windows (replication collapse?)"
        )
    res = _stats.ttest_ind(arr_a, arr_b, equal_var=False, alternative=alternative)
    se = float(np.sqrt(var_a / n_a + var_b / n_b))
    df = (var_a / n_a + var_b / n_b) ** 2 / (
        (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    )
    half_width = float(_stats.t.ppf(0.975, df) * se)
    diff = mean_a - mean_b
    return BatchMeansResult(
        n_windows_a=n_a,
        n_windows_b=n_b,
        mean_a=mean_a,
        mean_b=mean_b,
        mean_diff=diff,
        statistic=float(res.statistic),
        df=float(df),
        p_value=float(res.pvalue),
        ci95_low=diff - half_width,
        ci95_high=diff + half_width,
        alternative=alternative,
    )
