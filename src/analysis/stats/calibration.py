"""§9.7 UPGRADE 1 — pipeline calibration: A/A split-half + effect-injection tests.

Before its SHA is registered, the stats machinery must pass (a) A/A split-half
tests — the empirical false-positive rate on same-arm splits of pilot data must
approximate the nominal α — and (b) effect-injection tests — synthetic effects
of known size must be recovered at the simulated power. The resulting
``CalibrationReport`` ships as a paper artifact: measured operating
characteristics of our exact code, not assumed ones.

``test_fn`` is the caller's own test (the one that will run in the campaign):
``test_fn(a, b) -> p_value`` on two 1-D float arrays. Calibration exercises the
literal campaign code path — substituting a stand-in test here would defeat the
point of §9.7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np
from scipy.stats import binomtest

TestFn = Callable[[np.ndarray, np.ndarray], float]
InjectionKind = Literal["shift", "flip"]

_CI_CONFIDENCE = 0.95


class CalibrationError(ValueError):
    """Invalid calibration input or an unusable test_fn result (fail closed)."""


@dataclass(frozen=True)
class AAResult:
    """A/A split-half outcome: empirical false-positive rate with exact CI."""

    n_splits: int
    alpha: float
    n_rejections: int
    fp_rate: float
    ci_low: float
    ci_high: float

    @property
    def approximates_nominal(self) -> bool:
        """True iff the exact CI for the FP rate covers the nominal α."""
        return self.ci_low <= self.alpha <= self.ci_high


@dataclass(frozen=True)
class InjectionResult:
    """Effect-injection outcome: empirical power at one injected effect size."""

    effect_size: float
    kind: InjectionKind
    n_splits: int
    alpha: float
    n_rejections: int
    power: float
    ci_low: float
    ci_high: float
    target_power: float | None = None

    @property
    def meets_target(self) -> bool | None:
        """True iff the MEASURED recovery power reaches the simulated target
        (§9.7: "recovered at the simulated power"); None when no target set.

        Fail-closed by design: the criterion is the point estimate, never the
        CI upper bound — ``ci_high >= target`` answered "can we rule the
        target out?" and passed easiest exactly when n_splits was smallest
        (the 2026-08-02 finding: 2/5 rejections, power 0.40, CI up to 0.853,
        'PASS' at target 0.8). The CI is reported context, not the gate.
        """
        if self.target_power is None:
            return None
        return self.power >= self.target_power


@dataclass(frozen=True)
class CalibrationReport:
    """§9.7 artifact: A/A + injection operating characteristics of our code."""

    seed: int
    n_observations: int
    aa: AAResult
    injections: tuple[InjectionResult, ...] = field(default_factory=tuple)

    def to_markdown(self) -> str:
        lines = [
            "## Pipeline calibration report (§9.7)",
            "",
            f"- seed: `{self.seed}`  ·  observations: {self.n_observations}",
            "",
            "### A/A split-half (false-positive rate vs nominal α)",
            "",
            "| n_splits | α | rejections | FP rate | 95% CI | approximates α |",
            "|---|---|---|---|---|---|",
            (
                f"| {self.aa.n_splits} | {self.aa.alpha:g} | {self.aa.n_rejections} "
                f"| {self.aa.fp_rate:.4f} | [{self.aa.ci_low:.4f}, {self.aa.ci_high:.4f}] "
                f"| {'PASS' if self.aa.approximates_nominal else 'FAIL'} |"
            ),
            "",
            "### Effect injection (empirical power at known effect sizes)",
            "",
            "| effect | kind | n_splits | α | power | 95% CI | target | verdict |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for inj in self.injections:
            target = f"{inj.target_power:g}" if inj.target_power is not None else "—"
            verdict = {True: "PASS", False: "FAIL", None: "—"}[inj.meets_target]
            lines.append(
                f"| {inj.effect_size:g} | {inj.kind} | {inj.n_splits} | {inj.alpha:g} "
                f"| {inj.power:.4f} | [{inj.ci_low:.4f}, {inj.ci_high:.4f}] "
                f"| {target} | {verdict} |"
            )
        lines.append("")
        return "\n".join(lines)


def _as_1d_float(data: Sequence[float] | np.ndarray, *, min_n: int) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    if values.ndim != 1:
        raise CalibrationError(f"data must be 1-D, got shape {values.shape}")
    if values.size < min_n:
        raise CalibrationError(f"need at least {min_n} observations, got {values.size}")
    if not np.all(np.isfinite(values)):
        raise CalibrationError("data contains non-finite values")
    return values


def _check_p(p: float, context: str) -> float:
    p = float(p)
    if not (0.0 <= p <= 1.0) or not np.isfinite(p):
        raise CalibrationError(f"test_fn returned invalid p-value {p!r} ({context})")
    return p


def _rejection_ci(k: int, n: int) -> tuple[float, float]:
    ci = binomtest(k, n).proportion_ci(confidence_level=_CI_CONFIDENCE, method="exact")
    return float(ci.low), float(ci.high)


def _split_halves(
    values: np.ndarray, n_splits: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    # One permuted index matrix for all splits; odd n drops the last element per split.
    idx = np.argsort(rng.random((n_splits, values.size)), axis=1)
    half = values.size // 2
    permuted = values[idx]
    return permuted[:, :half], permuted[:, half : 2 * half]


def aa_split_half(
    data: Sequence[float] | np.ndarray,
    test_fn: TestFn,
    n_splits: int,
    seed: int,
    *,
    alpha: float = 0.05,
) -> AAResult:
    """A/A test: random same-arm split-halves must reject at ≈ the nominal α.

    Runs ``test_fn`` on ``n_splits`` seeded random half/half partitions of
    ``data`` (all drawn from ONE arm — no real effect exists) and reports the
    empirical false-positive rate with an exact binomial CI.
    """
    if n_splits < 1:
        raise CalibrationError(f"n_splits must be >= 1, got {n_splits}")
    if not 0.0 < alpha < 1.0:
        raise CalibrationError(f"alpha must be in (0, 1), got {alpha}")
    values = _as_1d_float(data, min_n=4)
    rng = np.random.default_rng(seed)
    left, right = _split_halves(values, n_splits, rng)
    rejections = sum(
        _check_p(test_fn(left[i], right[i]), f"A/A split {i}") < alpha
        for i in range(n_splits)
    )
    ci_low, ci_high = _rejection_ci(rejections, n_splits)
    return AAResult(
        n_splits=n_splits,
        alpha=alpha,
        n_rejections=rejections,
        fp_rate=rejections / n_splits,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def inject_effect(
    data: Sequence[float] | np.ndarray,
    effect_size: float,
    kind: InjectionKind = "shift",
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Return a copy of ``data`` carrying a synthetic effect of known size.

    - ``shift``: additive location shift by ``effect_size`` (metric units).
    - ``flip``: for binary 0/1 outcomes, flips a ``effect_size`` fraction of the
      0-rows to 1 (the discordant-pair process the audit §2.5 demands for
      tie-heavy metrics); requires ``seed`` to pick rows deterministically.
    """
    values = _as_1d_float(data, min_n=1)
    if kind == "shift":
        if not np.isfinite(effect_size):
            raise CalibrationError(f"effect_size must be finite, got {effect_size!r}")
        return values + float(effect_size)
    if kind == "flip":
        if not 0.0 < effect_size <= 1.0:
            raise CalibrationError(
                f"flip effect_size is a fraction in (0, 1], got {effect_size!r}"
            )
        if seed is None:
            raise CalibrationError("kind='flip' requires a seed")
        if not np.isin(values, (0.0, 1.0)).all():
            raise CalibrationError("kind='flip' requires binary 0/1 data")
        zero_idx = np.flatnonzero(values == 0.0)
        if zero_idx.size == 0:
            raise CalibrationError("kind='flip' requires at least one 0 outcome")
        n_flip = max(1, int(round(effect_size * zero_idx.size)))
        rng = np.random.default_rng(seed)
        flipped = values.copy()
        flipped[rng.choice(zero_idx, size=n_flip, replace=False)] = 1.0
        return flipped
    raise CalibrationError(f"unknown injection kind {kind!r}; allowed: shift, flip")


def recover_power(
    data: Sequence[float] | np.ndarray,
    test_fn: TestFn,
    effect_size: float,
    n_splits: int,
    seed: int,
    *,
    kind: InjectionKind = "shift",
    alpha: float = 0.05,
    target_power: float | None = None,
) -> InjectionResult:
    """Empirical power: inject a known effect into one A/A half and re-test.

    Same seeded split-halves as ``aa_split_half``; the right half receives the
    injected effect, so every rejection is a TRUE positive. ``target_power`` is
    the §9.6 simulated power for this (effect, n) point, letting the report
    state recovered-vs-simulated directly.
    """
    if n_splits < 1:
        raise CalibrationError(f"n_splits must be >= 1, got {n_splits}")
    if not 0.0 < alpha < 1.0:
        raise CalibrationError(f"alpha must be in (0, 1), got {alpha}")
    if target_power is not None and not 0.0 < target_power <= 1.0:
        raise CalibrationError(f"target_power must be in (0, 1], got {target_power!r}")
    values = _as_1d_float(data, min_n=4)
    rng = np.random.default_rng(seed)
    left, right = _split_halves(values, n_splits, rng)
    rejections = 0
    for i in range(n_splits):
        injected = inject_effect(right[i], effect_size, kind, seed=seed + i + 1)
        if _check_p(test_fn(left[i], injected), f"injection split {i}") < alpha:
            rejections += 1
    ci_low, ci_high = _rejection_ci(rejections, n_splits)
    return InjectionResult(
        effect_size=float(effect_size),
        kind=kind,
        n_splits=n_splits,
        alpha=alpha,
        n_rejections=rejections,
        power=rejections / n_splits,
        ci_low=ci_low,
        ci_high=ci_high,
        target_power=target_power,
    )


def build_report(
    data: Sequence[float] | np.ndarray,
    test_fn: TestFn,
    n_splits: int,
    seed: int,
    *,
    alpha: float = 0.05,
    effect_sizes: Sequence[float] = (),
    kind: InjectionKind = "shift",
    target_power: float | None = None,
) -> CalibrationReport:
    """Run the full §9.7 suite (one A/A pass + one injection pass per effect)."""
    values = _as_1d_float(data, min_n=4)
    aa = aa_split_half(values, test_fn, n_splits, seed, alpha=alpha)
    injections = tuple(
        recover_power(
            values,
            test_fn,
            effect,
            n_splits,
            seed,
            kind=kind,
            alpha=alpha,
            target_power=target_power,
        )
        for effect in effect_sizes
    )
    return CalibrationReport(
        seed=seed, n_observations=values.size, aa=aa, injections=injections
    )
