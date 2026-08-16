"""Registered lambda*/SLO-floor calibration procedure (D6 §6.1; finding E3).

Charter authority: MyDocs/PUBLICATION.md D6 §6.1 pre-registers the offered
rates of the pressure campaign as FRACTIONS {0.5, 0.7, 0.85, 0.95, 1.05, 1.2}
of a PREDICTED saturation rate lambda* per model×engine×budget cell
(``load_generator.D6_RATE_FRACTIONS``), and the primary SLO pair as RELATIVE
thresholds — TTFT ≤ 10× and TPOT ≤ 5× the same cell family's MEASURED
single-stream floor (``goodput.TTFT_SLO_MULTIPLIER`` /
``goodput.TPOT_SLO_MULTIPLIER``). Neither lambda* nor the floor is a free
parameter at analysis time: BOTH come from this module's registered
procedure, run once per cell BEFORE the confirmatory campaign.

THE REGISTERED PROCEDURE (this docstring is the text embedded in
PRE_REGISTRATION.md; the constants below ARE the registered values):

1. Single-stream floor — ``FLOOR_N_REQUESTS`` (30) sequential streamed
   requests at concurrency 1; the floor is the ``FLOOR_STATISTIC`` (median)
   of the per-request streamed TTFT and TPOT, reported in seconds
   (``summarize_floor``). The floor pair feeds ``goodput.SLOBaseline``
   unchanged. A floor is computed over COMPLETE telemetry only: any missing
   or non-finite per-request value fails the calibration closed — a floor is
   never computed over partial telemetry.

2. lambda* probe — a geometric rate ladder (factor ``PROBE_LADDER_FACTOR``
   = 1.3, at most ``PROBE_MAX_STEPS`` = 12 steps) of open-loop Poisson
   windows, each ``PROBE_WINDOW_S`` = 75 s long (inside the charter's
   60–90 s band) with the first ``PROBE_WARMUP_S`` = 10 s trimmed (Jain
   warmup removal, D6 §6.3). A ladder step is SUSTAINABLE iff its completed
   fraction (attainment) is ≥ ``PROBE_ATTAINMENT_MIN`` = 0.9 AND its
   completed throughput is not retrograde versus the previous sustainable
   step. lambda* is the HIGHEST sustainable rate immediately followed by an
   unsustainable step (``decide_lambda_star``).

3. Honest labels, never guesses (mirroring ``goodput.OnsetEstimate``): a
   probe that never brackets saturation returns ``LADDER_EXHAUSTED`` (extend
   the ladder — NEVER extrapolate); a probe whose lowest rate is already
   unsustainable returns ``NONE_SUSTAINABLE``. Only ``ESTIMATED`` carries a
   lambda* value.

CALIBRATION DATA NEVER ENTERS CONFIRMATORY ANALYSIS. The floor requests and
probe windows exist solely to place the registered rate grid and SLO
thresholds; their latency/throughput rows are provenance for the run
manifest (``CellCalibration.to_manifest``) and MUST NOT appear in any
confirmatory window, figure, or test statistic. Confirmatory windows are
fresh, separately dispatched runs.

Fail-closed doctrine: invalid inputs raise the typed :class:`CalibrationError`
(mirroring ``LoadGeneratorError`` in ``src/orchestration/load_generator.py``)
— never a silent default, because a silently altered calibration voids the
pre-registered D6 rate grid.

This module is PURE decision logic: dataclasses + the registered decision
rules, no network and no GPU. The live driving (adapter calls, open-loop
dispatch) lives in ``scripts/3_run/calibrate_cell.py``.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Final, Literal, Optional, Sequence, Tuple

__all__ = [
    "CalibrationError",
    "CellCalibration",
    "FLOOR_N_REQUESTS",
    "FLOOR_STATISTIC",
    "FloorMeasurement",
    "LambdaStarEstimate",
    "LambdaStarLabel",
    "PROBE_ATTAINMENT_MIN",
    "PROBE_LADDER_FACTOR",
    "PROBE_MAX_STEPS",
    "PROBE_WARMUP_S",
    "PROBE_WINDOW_S",
    "ProbeStep",
    "decide_lambda_star",
    "geometric_rate_ladder",
    "summarize_floor",
]

# --------------------------------------------------------------------------- #
# Registered procedure constants.
#
# These ARE the registered values (D6 §6.1; PRE_REGISTRATION.md embeds this
# module's docstring). Changing any of them after the D9 freeze is a protocol
# deviation and must be reported as one — they are constants, not knobs.
# --------------------------------------------------------------------------- #

# Floor stage: sequential single-stream requests (concurrency 1) and the
# summary statistic over their per-request streamed TTFT/TPOT.
FLOOR_N_REQUESTS: Final[int] = 30
FLOOR_STATISTIC: Final[str] = "median"

# Probe stage: geometric ladder of open-loop Poisson windows.
PROBE_LADDER_FACTOR: Final[float] = 1.3   # geometric step between probe rates
PROBE_WINDOW_S: Final[float] = 75.0       # per-rate window (charter band 60-90 s)
PROBE_WARMUP_S: Final[float] = 10.0       # Jain warmup trim per window (§6.3)
PROBE_ATTAINMENT_MIN: Final[float] = 0.9  # completed fraction for "sustainable"
PROBE_MAX_STEPS: Final[int] = 12          # ladder length cap

PROCEDURE_VERSION: Final[str] = "cal-v1 (2026-08-12)"

LambdaStarLabel = Literal["ESTIMATED", "NONE_SUSTAINABLE", "LADDER_EXHAUSTED"]

_PROCEDURE_MANIFEST: Final[Dict[str, Any]] = {
    "floor_n_requests": FLOOR_N_REQUESTS,
    "floor_statistic": FLOOR_STATISTIC,
    "probe_ladder_factor": PROBE_LADDER_FACTOR,
    "probe_window_s": PROBE_WINDOW_S,
    "probe_warmup_s": PROBE_WARMUP_S,
    "probe_attainment_min": PROBE_ATTAINMENT_MIN,
    "probe_max_steps": PROBE_MAX_STEPS,
}


class CalibrationError(RuntimeError):
    """Typed fail-closed error for invalid calibration inputs.

    Mirrors ``LoadGeneratorError`` (src/orchestration/load_generator.py): a
    bad parameter raises immediately with the offending parameter named — it
    is never silently coerced or defaulted, because a silently altered
    calibration voids the pre-registered D6 rate grid and SLO thresholds.
    """

    def __init__(self, parameter: str, value: Any, reason: str) -> None:
        self.parameter = parameter
        self.value = value
        self.reason = reason
        super().__init__(
            f"calibration parameter '{parameter}'={value!r} invalid: {reason}"
        )


def _require_real(parameter: str, value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CalibrationError(parameter, value, "must be a real number")
    return float(value)


def _require_positive_finite(parameter: str, value: Any) -> float:
    out = _require_real(parameter, value)
    if not math.isfinite(out) or out <= 0.0:
        raise CalibrationError(parameter, value, "must be finite and > 0")
    return out


def _require_int(parameter: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CalibrationError(parameter, value, "must be an int")
    return value


# --------------------------------------------------------------------------- #
# Probe rate ladder
# --------------------------------------------------------------------------- #


def geometric_rate_ladder(
    start_qps: float,
    *,
    factor: float = PROBE_LADDER_FACTOR,
    max_steps: int = PROBE_MAX_STEPS,
) -> Tuple[float, ...]:
    """The registered geometric probe ladder: start_qps · factor^k, k = 0..max_steps-1.

    Geometric (not linear) spacing keeps the relative rate resolution constant
    across cells whose absolute lambda* differs by orders of magnitude — the
    same reasoning as the §9.2 multiplicative ×/÷ resolution band.
    """
    start = _require_positive_finite("start_qps", start_qps)
    fac = _require_real("factor", factor)
    if not math.isfinite(fac) or fac <= 1.0:
        raise CalibrationError("factor", factor, "must be finite and > 1 (geometric ladder)")
    steps = _require_int("max_steps", max_steps)
    if steps < 2:
        raise CalibrationError(
            "max_steps", max_steps, "must be >= 2 (a single rate can never bracket lambda*)"
        )
    return tuple(start * fac**k for k in range(steps))


# --------------------------------------------------------------------------- #
# Probe step accounting
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProbeStep:
    """One probe-ladder window's outcome (post-warmup-trim counts).

    ``n_scheduled`` / ``n_completed`` count the INTENDED open-loop arrivals
    inside the trimmed measurement window and how many of them completed
    without error; ``throughput_rps`` is completed requests per measured
    second. Attainment (completed fraction) is the §6.1 sustainability
    currency, mirroring ``goodput.ATTAINMENT_MIN``.
    """

    rate_qps: float
    n_scheduled: int
    n_completed: int
    throughput_rps: float

    def __post_init__(self) -> None:
        _require_positive_finite("rate_qps", self.rate_qps)
        n_sched = _require_int("n_scheduled", self.n_scheduled)
        if n_sched <= 0:
            raise CalibrationError("n_scheduled", self.n_scheduled, "must be > 0")
        n_comp = _require_int("n_completed", self.n_completed)
        if n_comp < 0:
            raise CalibrationError("n_completed", self.n_completed, "must be >= 0")
        if n_comp > n_sched:
            raise CalibrationError(
                "n_completed",
                self.n_completed,
                f"cannot exceed n_scheduled={n_sched} (completions are a subset of arrivals)",
            )
        tput = _require_real("throughput_rps", self.throughput_rps)
        if not math.isfinite(tput) or tput < 0.0:
            raise CalibrationError(
                "throughput_rps", self.throughput_rps, "must be finite and >= 0"
            )

    @property
    def attainment(self) -> float:
        """Completed fraction of intended arrivals (the §6.1 attainment)."""
        if self.n_scheduled <= 0:
            raise CalibrationError(
                "n_scheduled", self.n_scheduled, "attainment undefined for n_scheduled <= 0"
            )
        return self.n_completed / self.n_scheduled

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "rate_qps": self.rate_qps,
            "n_scheduled": self.n_scheduled,
            "n_completed": self.n_completed,
            "throughput_rps": self.throughput_rps,
            "attainment": self.attainment,
        }


# --------------------------------------------------------------------------- #
# The registered lambda* decision rule
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LambdaStarEstimate:
    """Outcome of the lambda* probe — a labeled decision, never a guess.

    Mirrors ``goodput.OnsetEstimate``'s philosophy: ``lambda_star_qps`` is set
    ONLY when label == "ESTIMATED" (the ladder bracketed saturation).
    ``sustained_rate_qps`` is the highest sustainable rate observed;
    ``first_unsustainable_qps`` is the lowest rate shown unsustainable. For
    LADDER_EXHAUSTED the caller must EXTEND the ladder and re-probe — the
    registered rule never extrapolates beyond the last measured rate.
    """

    label: LambdaStarLabel
    lambda_star_qps: Optional[float]
    sustained_rate_qps: Optional[float]
    first_unsustainable_qps: Optional[float]
    steps: Tuple[ProbeStep, ...]

    def __post_init__(self) -> None:
        if self.label not in ("ESTIMATED", "NONE_SUSTAINABLE", "LADDER_EXHAUSTED"):
            raise CalibrationError("label", self.label, "unknown lambda* label")
        if (self.label == "ESTIMATED") != (self.lambda_star_qps is not None):
            raise CalibrationError(
                "lambda_star_qps",
                self.lambda_star_qps,
                "set if and only if label == 'ESTIMATED' (labels are honest, not guesses)",
            )

    def to_manifest(self) -> Dict[str, Any]:
        """Provenance record for the calibration JSON / run manifest."""
        return {
            "label": self.label,
            "lambda_star_qps": self.lambda_star_qps,
            "sustained_rate_qps": self.sustained_rate_qps,
            "first_unsustainable_qps": self.first_unsustainable_qps,
            "n_steps": len(self.steps),
            "steps": [step.to_manifest() for step in self.steps],
        }


def decide_lambda_star(steps: Sequence[ProbeStep]) -> LambdaStarEstimate:
    """THE registered lambda* decision rule over a probed rate ladder.

    A step is SUSTAINABLE iff attainment ≥ ``PROBE_ATTAINMENT_MIN`` AND its
    throughput is not retrograde (throughput ≥ the previous SUSTAINABLE
    step's throughput; the first step has no retrograde test — retrograde
    throughput at rising offered rate is the §6.1 cliff signature, so a step
    that completes plenty of requests while total throughput falls is still
    past saturation). Then:

    - ESTIMATED: lambda* = the HIGHEST sustainable rate immediately followed
      by an unsustainable step; ``first_unsustainable_qps`` = that next rate.
    - NONE_SUSTAINABLE: no step sustainable — the ladder started past
      saturation; lower the start rate and re-probe.
    - LADDER_EXHAUSTED: the ladder ended on a sustainable step without ever
      bracketing saturation — extend the ladder and re-probe; the rule NEVER
      extrapolates a lambda* it did not bracket.

    Steps must be strictly increasing in rate (a ladder, not a bag).
    """
    seq = tuple(steps)
    if not seq:
        raise CalibrationError("steps", steps, "at least one probe step is required")
    for i, step in enumerate(seq):
        if not isinstance(step, ProbeStep):
            raise CalibrationError(f"steps[{i}]", step, "must be a ProbeStep")
    for prev, curr in zip(seq, seq[1:]):
        if curr.rate_qps <= prev.rate_qps:
            raise CalibrationError(
                "steps",
                (prev.rate_qps, curr.rate_qps),
                "probe rates must be strictly increasing (a ladder, not a bag)",
            )

    sustainable: list[bool] = []
    last_sustained_throughput: Optional[float] = None
    for step in seq:
        ok = step.attainment >= PROBE_ATTAINMENT_MIN
        if (
            ok
            and last_sustained_throughput is not None
            and step.throughput_rps < last_sustained_throughput
        ):
            ok = False  # retrograde throughput: past saturation despite attainment
        sustainable.append(ok)
        if ok:
            last_sustained_throughput = step.throughput_rps

    # Highest sustainable step immediately followed by an unsustainable one.
    bracket_idx: Optional[int] = None
    for i in range(len(seq) - 1):
        if sustainable[i] and not sustainable[i + 1]:
            bracket_idx = i
    if bracket_idx is not None:
        return LambdaStarEstimate(
            label="ESTIMATED",
            lambda_star_qps=seq[bracket_idx].rate_qps,
            sustained_rate_qps=seq[bracket_idx].rate_qps,
            first_unsustainable_qps=seq[bracket_idx + 1].rate_qps,
            steps=seq,
        )
    if not any(sustainable):
        return LambdaStarEstimate(
            label="NONE_SUSTAINABLE",
            lambda_star_qps=None,
            sustained_rate_qps=None,
            first_unsustainable_qps=seq[0].rate_qps,
            steps=seq,
        )
    # Some (possibly all) steps sustainable and no sustainable->unsustainable
    # transition: the LAST step is sustainable, so the ladder never bracketed
    # saturation. Extend the ladder; never extrapolate.
    highest_sustained = max(
        step.rate_qps for step, ok in zip(seq, sustainable) if ok
    )
    return LambdaStarEstimate(
        label="LADDER_EXHAUSTED",
        lambda_star_qps=None,
        sustained_rate_qps=highest_sustained,
        first_unsustainable_qps=None,
        steps=seq,
    )


# --------------------------------------------------------------------------- #
# Single-stream SLO floor
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FloorMeasurement:
    """The measured single-stream latency floor for one cell family.

    ``ttft_s`` / ``tpot_s`` are in SECONDS and feed
    ``goodput.SLOBaseline(ttft_s=..., tpot_s=...)`` unchanged; the §6.1
    primary SLO pair is then TTFT ≤ 10× and TPOT ≤ 5× these floors.
    """

    ttft_s: float
    tpot_s: float
    n_requests: int
    statistic: str = FLOOR_STATISTIC

    def __post_init__(self) -> None:
        _require_positive_finite("ttft_s", self.ttft_s)
        _require_positive_finite("tpot_s", self.tpot_s)
        n = _require_int("n_requests", self.n_requests)
        if n <= 0:
            raise CalibrationError("n_requests", self.n_requests, "must be > 0")
        if not isinstance(self.statistic, str) or not self.statistic.strip():
            raise CalibrationError("statistic", self.statistic, "must be a non-empty string")

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "ttft_s": self.ttft_s,
            "tpot_s": self.tpot_s,
            "n_requests": self.n_requests,
            "statistic": self.statistic,
        }


def summarize_floor(
    ttft_ms_values: Sequence[float],
    tpot_ms_values: Sequence[float],
    *,
    n_min: int = FLOOR_N_REQUESTS,
) -> FloorMeasurement:
    """The registered floor summary: median over per-request streamed TTFT/TPOT.

    Inputs are per-request MILLISECOND values (the adapters' streamed
    ``ttft_ms`` and per-token decode time); the returned floors are in
    SECONDS to match ``goodput.SLOBaseline``. Drop-nothing: the two sequences
    must be paired (equal length, one entry per floor request) and every
    value must be a finite positive number — a ``None`` or non-finite entry
    raises, because a floor computed over partial telemetry silently loosens
    every SLO derived from it.
    """
    n_floor = _require_int("n_min", n_min)
    if n_floor <= 0:
        raise CalibrationError("n_min", n_min, "must be > 0")
    ttft = list(ttft_ms_values)
    tpot = list(tpot_ms_values)
    if len(ttft) != len(tpot):
        raise CalibrationError(
            "tpot_ms_values",
            len(tpot),
            f"must pair 1:1 with ttft_ms_values (got {len(ttft)} TTFT vs {len(tpot)} TPOT)",
        )
    if len(ttft) < n_floor:
        raise CalibrationError(
            "ttft_ms_values",
            len(ttft),
            f"floor requires >= {n_floor} complete single-stream requests "
            f"(FLOOR_N_REQUESTS); got {len(ttft)}",
        )
    for name, values in (("ttft_ms_values", ttft), ("tpot_ms_values", tpot)):
        for i, value in enumerate(values):
            v = _require_real(f"{name}[{i}]", value)
            if not math.isfinite(v) or v <= 0.0:
                raise CalibrationError(
                    f"{name}[{i}]",
                    value,
                    "must be finite and > 0 (a floor is never computed over "
                    "partial or degenerate telemetry)",
                )
    return FloorMeasurement(
        ttft_s=statistics.median(float(v) for v in ttft) / 1000.0,
        tpot_s=statistics.median(float(v) for v in tpot) / 1000.0,
        n_requests=len(ttft),
        statistic=FLOOR_STATISTIC,
    )


# --------------------------------------------------------------------------- #
# Per-cell calibration record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CellCalibration:
    """One model×engine×budget cell's calibration: the floor + lambda* pair.

    ``to_manifest()`` is the calibration JSON the campaign driver consumes to
    build the D6 rate grid (``build_rate_grid_schedules(lambda_star_qps=...)``)
    and the SLO thresholds (``goodput.SLOBaseline``), and which the run
    manifest embeds for provenance. Calibration rows themselves NEVER enter
    confirmatory analysis (module docstring).
    """

    model: str
    engine: str
    budget_fraction: float
    floor: FloorMeasurement
    lambda_star: LambdaStarEstimate
    procedure_version: str = PROCEDURE_VERSION

    def __post_init__(self) -> None:
        for name, value in (("model", self.model), ("engine", self.engine)):
            if not isinstance(value, str) or not value.strip():
                raise CalibrationError(name, value, "must be a non-empty string")
        # budget_fraction is the charter's KV budget RATIO r (allocated KV /
        # demanded KV), NOT a ≤1 fraction: §6.1 measures the SLO floor at
        # r=1.5 (goodput.SLOBaseline docstring), so values above 1 are
        # legitimate charter points. Positive and finite is the only bound.
        _require_positive_finite("budget_fraction", self.budget_fraction)
        if not isinstance(self.floor, FloorMeasurement):
            raise CalibrationError("floor", self.floor, "must be a FloorMeasurement")
        if not isinstance(self.lambda_star, LambdaStarEstimate):
            raise CalibrationError(
                "lambda_star", self.lambda_star, "must be a LambdaStarEstimate"
            )
        if not isinstance(self.procedure_version, str) or not self.procedure_version.strip():
            raise CalibrationError(
                "procedure_version", self.procedure_version, "must be a non-empty string"
            )

    def to_manifest(self) -> Dict[str, Any]:
        """The calibration JSON schema (campaign driver + run manifest input)."""
        return {
            "procedure_version": self.procedure_version,
            "model": self.model,
            "engine": self.engine,
            "budget_fraction": self.budget_fraction,
            "procedure": dict(_PROCEDURE_MANIFEST),
            "confirmatory": False,  # calibration data NEVER enters confirmatory analysis
            "floor": self.floor.to_manifest(),
            "lambda_star": self.lambda_star.to_manifest(),
        }
