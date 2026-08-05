"""L4 instrument-validation harness (charter D8 §8.6; feeds the D9 §9.13 gate).

The meta-layer of the D8 quality module: BEFORE the campaign, every grounding
instrument (LettuceDetect, the TRUE-selected claim checker, ...) must be
calibrated against a public anchor set (RAGTruth / TRUE subsets, D8 §8.6(a))
and pass a per-length-bin discrimination gate (D8 §8.6(b)); AFTER the
campaign, a drift audit re-scores a stored anchor sample and checks
calibration transfer (D8 §8.6(e)). The resulting
:class:`InstrumentCalibrationReport` is a registration artifact: the D9 chain
refuses to register on a failed calibration (mirroring
``src.analysis.stats.prereg.assemble_preregistration``'s treatment of the
§9.7 stats ``CalibrationReport``) via :func:`assert_registrable`.

Charter rules implemented here (PUBLICATION.md, NORMATIVE):

- **τ selection (D8 §8.6(c), AMENDED 2026-08-02 — the DEFINITE rule):** "the
  smallest threshold achieving ≥90% precision on the RAGTruth/TRUE
  calibration anchor". Implemented verbatim in :func:`select_tau`; the anchor
  identity (dataset, split, content fingerprint) is recorded in the output so
  the frozen PRE_REGISTRATION.md can name exactly what τ was calibrated on.
- **Per-length-bin discrimination gate (D8 §8.6(b)):** anchor items are
  binned by context length (default charter grid 1k/4k/8k/16k/32k); the
  instrument must discriminate grounded from ungrounded within EVERY bin
  (per-bin AUC against a pre-registered floor). A detector that fails only on
  long contexts would silently bias exactly the pressure axis — hence the
  per-bin, not pooled, requirement.
- **Post-campaign drift audit (D8 §8.6(e)):** re-scored anchor sample vs the
  calibration-time scores; drift metrics + thresholds; a τ-flip rate because
  what matters downstream is whether the FROZEN τ still separates.

Engineering doctrine: fail-closed — missing files, missing columns,
one-class bins, unreachable precision floors and mismatched drift samples all
raise :class:`InstrumentCalibrationError` (typed, mirroring
``InstrumentUnavailableError`` in ``src/evaluation/quality.py``); nothing
silently degrades. Pure numpy/pandas — no model loading, no network; anchor
data arrives as a DataFrame or a JSONL path staged by the datasets builder.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

# JSON artifact schema version (bump on any breaking key change).
SCHEMA_VERSION: int = 1
# Artifact kind tag, so the pre-registration assembler / loaders can dispatch.
ARTIFACT_KIND: str = "instrument_calibration_report"

# Charter default: D8 §8.6(c) — "≥90% precision" is fixed by the charter, so
# 0.90 is a legitimate default (it is the registered rule, not a guess).
DEFAULT_PRECISION_FLOOR: float = 0.90

# Charter default length grid: D8 §8.6(b) names 1k/4k/8k/16k/32k. Bins are
# right-open [lo, hi); the trailing +inf bin catches beyond-32k contexts so no
# anchor item can silently fall off the grid.
DEFAULT_BIN_EDGES: tuple[float, ...] = (
    0.0,
    1024.0,
    4096.0,
    8192.0,
    16384.0,
    32768.0,
    math.inf,
)

# Absolute tolerance for floor comparisons: k/n vs a decimal floor can differ
# by 1 ulp; 1e-12 is far below any attainable 1/n granularity, so this can
# never promote a genuinely-below-floor value.
_FLOOR_ATOL: float = 1e-12

_REQUIRED_REPORT_KEYS: tuple[str, ...] = (
    "schema_version",
    "artifact",
    "instrument",
    "anchor",
    "tau_selection",
    "length_bin_gate",
    "drift_audit",
    "passed",
)


class InstrumentCalibrationError(RuntimeError):
    """L4 calibration input/gate failure (fail closed, D8 §8.6).

    Mirrors ``InstrumentUnavailableError`` in ``src/evaluation/quality.py``:
    a calibration that cannot be computed, or an anchor that cannot support
    the registered rule, must raise — never return a degraded number under
    the same name.
    """


# --------------------------------------------------------------------------- #
# Anchor data access + identity
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AnchorIdentity:
    """Identity of the calibration anchor set (D8 §8.6(a)/(c)).

    Recorded inside every τ selection and report so PRE_REGISTRATION.md can
    freeze WHAT the instrument was calibrated on: dataset name, split, item
    count and an order-invariant content fingerprint.
    """

    dataset: str
    split: str
    n_items: int
    fingerprint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _as_finite_1d(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise InstrumentCalibrationError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise InstrumentCalibrationError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise InstrumentCalibrationError(f"{name} contains non-finite values")
    return arr


def _as_binary_labels(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise InstrumentCalibrationError(f"labels must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise InstrumentCalibrationError("labels are empty")
    if arr.dtype == bool:
        arr = arr.astype(float)
    arr = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(arr)) or not np.isin(arr, (0.0, 1.0)).all():
        raise InstrumentCalibrationError(
            "labels must be binary grounded indicators in {0, 1} "
            "(1 = grounded, 0 = ungrounded/hallucinated)"
        )
    return arr


def anchor_fingerprint(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Order-invariant SHA-256 content fingerprint of the anchor columns.

    Rows are canonically sorted before hashing, so the fingerprint identifies
    the anchor CONTENT (which items, which scores, which labels) independent
    of row order or storage format (DataFrame vs JSONL).
    """
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise InstrumentCalibrationError(
            f"anchor data missing required columns {missing}; has {list(frame.columns)}"
        )
    canon = frame.loc[:, list(columns)].copy()
    canon = canon.astype(str)
    canon = canon.sort_values(by=list(columns), kind="mergesort").reset_index(drop=True)
    payload = canon.to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_anchor(
    data: pd.DataFrame | str | Path,
    *,
    dataset: str,
    split: str,
    score_col: str = "score",
    label_col: str = "label",
    length_col: str | None = "context_length",
    id_col: str | None = None,
) -> tuple[pd.DataFrame, AnchorIdentity]:
    """Load + validate an anchor set from a DataFrame or a JSONL path.

    No downloading happens here (D8 §8.6 is local/$0): the datasets builder
    stages RAGTruth/TRUE-style anchors; this function only consumes a path or
    an in-memory frame. Fail-closed: a missing file, unreadable JSONL or
    missing column raises :class:`InstrumentCalibrationError`.

    Returns the validated frame plus its :class:`AnchorIdentity` (fingerprint
    over exactly the consumed columns, order-invariant).
    """
    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.is_file():
            raise InstrumentCalibrationError(f"anchor file not found: {path}")
        # Parse line-by-line with the stdlib parser: ``pd.read_json`` defaults
        # to fast-but-imprecise float parsing, which would make the content
        # fingerprint of a JSONL anchor differ from the identical in-memory
        # DataFrame by 1 ulp — a silent identity break (fail closed instead).
        records: list[dict[str, Any]] = []
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InstrumentCalibrationError(
                    f"anchor file {path} is not valid JSONL (line {lineno}): {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise InstrumentCalibrationError(
                    f"anchor file {path} line {lineno} is not a JSON object"
                )
            records.append(record)
        if not records:
            raise InstrumentCalibrationError(f"anchor file {path} is empty")
        frame = pd.DataFrame.from_records(records)
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        raise InstrumentCalibrationError(
            f"anchor data must be a DataFrame or a JSONL path, got {type(data)!r}"
        )
    if not dataset or not split:
        raise InstrumentCalibrationError(
            "anchor identity requires non-empty dataset and split names (D8 §8.6(c))"
        )

    columns = [score_col, label_col]
    if length_col is not None:
        columns.append(length_col)
    if id_col is not None:
        columns.append(id_col)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise InstrumentCalibrationError(
            f"anchor data missing required columns {missing}; has {list(frame.columns)}"
        )
    if frame.empty:
        raise InstrumentCalibrationError("anchor data is empty")

    # Validate content eagerly so downstream math never sees bad rows.
    _as_finite_1d(frame[score_col].to_numpy(), f"anchor column {score_col!r}")
    _as_binary_labels(frame[label_col].to_numpy())
    if length_col is not None:
        lengths = _as_finite_1d(
            frame[length_col].to_numpy(), f"anchor column {length_col!r}"
        )
        if (lengths < 0).any():
            raise InstrumentCalibrationError(
                f"anchor column {length_col!r} contains negative context lengths"
            )

    identity = AnchorIdentity(
        dataset=dataset,
        split=split,
        n_items=int(len(frame)),
        fingerprint_sha256=anchor_fingerprint(frame, columns),
    )
    return frame, identity


# --------------------------------------------------------------------------- #
# (1) τ selection — the D8 §8.6(c) definite rule
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TauSelection:
    """Result of the D8 §8.6(c) τ rule, with the anchor identity frozen in.

    ``sensitivity``/``specificity`` at τ are included because §8.6(c) also
    mandates Rogan-Gladen misclassification correction of Y
    (Rogan & Gladen, 1978, "Estimating prevalence from the results of a
    screening test", Am. J. Epidemiology 107(1)) — those two numbers are its
    inputs, so the calibration artifact carries them.
    """

    tau: float
    precision_floor: float
    precision_at_tau: float
    recall_at_tau: float  # = sensitivity
    specificity_at_tau: float
    n_predicted_grounded: int
    n_grounded: int
    n_ungrounded: int
    n_candidate_thresholds: int
    anchor: AnchorIdentity

    @property
    def sensitivity_at_tau(self) -> float:
        return self.recall_at_tau

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["anchor"] = self.anchor.to_dict()
        return payload


def select_tau(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    anchor: AnchorIdentity,
    *,
    precision_floor: float = DEFAULT_PRECISION_FLOOR,
) -> TauSelection:
    """Select τ by the charter's DEFINITE rule (PUBLICATION.md D8 §8.6(c)).

    Rule, verbatim from the charter (AMENDED 2026-08-02, "no more e.g."):
    "τ chosen by DEFINITE pre-registered rule: the smallest threshold
    achieving ≥90% precision on the RAGTruth/TRUE calibration anchor".

    Operationalization: an item is predicted GROUNDED iff ``score >= τ``;
    precision(τ) = truly-grounded / predicted-grounded. Precision as a
    function of τ is a step function that changes only at observed score
    values, so the candidate grid is exactly the unique observed anchor
    scores (ascending) and "smallest threshold" = the smallest candidate
    whose precision reaches the floor. Deterministic given the anchor.

    Fail-closed: an anchor with only one class cannot calibrate precision;
    an anchor where NO threshold reaches the floor means the instrument
    fails §8.6 calibration — both raise :class:`InstrumentCalibrationError`.
    """
    if not 0.0 < precision_floor <= 1.0:
        raise InstrumentCalibrationError(
            f"precision_floor must be in (0, 1], got {precision_floor!r}"
        )
    s = _as_finite_1d(scores, "scores")
    y = _as_binary_labels(labels)
    if s.size != y.size:
        raise InstrumentCalibrationError(
            f"scores ({s.size}) and labels ({y.size}) differ in length"
        )
    n_grounded = int(y.sum())
    n_ungrounded = int(y.size - n_grounded)
    if n_grounded == 0 or n_ungrounded == 0:
        raise InstrumentCalibrationError(
            "anchor must contain BOTH grounded and ungrounded items to "
            f"calibrate τ (got {n_grounded} grounded / {n_ungrounded} ungrounded)"
        )

    # Vectorized sweep over the candidate grid (unique observed scores,
    # ascending). Sort scores descending once; cumulative sums give
    # TP(τ)/FP(τ) for τ at each distinct score.
    order = np.argsort(-s, kind="mergesort")
    s_desc = s[order]
    y_desc = y[order]
    tp_cum = np.cumsum(y_desc)
    n_pred_cum = np.arange(1, s_desc.size + 1, dtype=float)
    # Last index of each distinct score in the descending array = the full
    # predicted set for τ == that score (ties included).
    is_last_of_value = np.r_[s_desc[1:] != s_desc[:-1], True]
    cand_scores = s_desc[is_last_of_value]  # descending unique
    cand_tp = tp_cum[is_last_of_value].astype(float)
    cand_n_pred = n_pred_cum[is_last_of_value]
    cand_precision = cand_tp / cand_n_pred

    achieving = cand_precision >= (precision_floor - _FLOOR_ATOL)
    if not achieving.any():
        raise InstrumentCalibrationError(
            f"no threshold achieves precision >= {precision_floor:g} on anchor "
            f"{anchor.dataset}/{anchor.split} (best = {cand_precision.max():.4f}) "
            "— instrument FAILS D8 §8.6(c) calibration"
        )
    # cand_scores is descending, so the LAST achieving index is the smallest τ.
    idx = int(np.flatnonzero(achieving)[-1])
    tau = float(cand_scores[idx])
    tp = float(cand_tp[idx])
    n_pred = float(cand_n_pred[idx])
    fp = n_pred - tp
    tn = float(n_ungrounded) - fp
    return TauSelection(
        tau=tau,
        precision_floor=float(precision_floor),
        precision_at_tau=tp / n_pred,
        recall_at_tau=tp / n_grounded,
        specificity_at_tau=tn / n_ungrounded,
        n_predicted_grounded=int(n_pred),
        n_grounded=n_grounded,
        n_ungrounded=n_ungrounded,
        n_candidate_thresholds=int(cand_scores.size),
        anchor=anchor,
    )


# --------------------------------------------------------------------------- #
# (2) Per-length-bin discrimination gate — D8 §8.6(b)
# --------------------------------------------------------------------------- #


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the rank-sum (Mann-Whitney U) identity, midrank ties.

    Implements AUC = (R⁺ − n⁺(n⁺+1)/2) / (n⁺ · n⁻) where R⁺ is the sum of
    midranks of the positive-class scores — the equivalence of AUC and the
    Wilcoxon-Mann-Whitney statistic per Hanley & McNeil (1982), "The meaning
    and use of the area under a receiver operating characteristic (ROC)
    curve", Radiology 143(1); midrank tie handling per Fawcett (2006), "An
    introduction to ROC analysis", Pattern Recognition Letters 27(8).
    Pure numpy/pandas (no scipy), matching this module's no-model contract.
    """
    s = _as_finite_1d(scores, "scores")
    y = _as_binary_labels(labels)
    if s.size != y.size:
        raise InstrumentCalibrationError(
            f"scores ({s.size}) and labels ({y.size}) differ in length"
        )
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise InstrumentCalibrationError(
            "AUC undefined without both classes present "
            f"(got {n_pos} positive / {n_neg} negative)"
        )
    ranks = pd.Series(s).rank(method="average").to_numpy()
    rank_sum_pos = float(ranks[y == 1.0].sum())
    u_pos = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u_pos / (n_pos * n_neg)


@dataclass(frozen=True)
class LengthBinResult:
    """Discrimination outcome for one context-length bin (D8 §8.6(b))."""

    bin_label: str
    lo: float
    hi: float
    n_grounded: int
    n_ungrounded: int
    auc: float
    auc_floor: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class LengthBinGate:
    """The full §8.6(b) gate: EVERY bin must discriminate above the floor."""

    auc_floor: float
    bin_edges: tuple[float, ...]
    bins: tuple[LengthBinResult, ...]

    @property
    def passed(self) -> bool:
        return all(b.passed for b in self.bins)

    @property
    def failed_bins(self) -> tuple[str, ...]:
        return tuple(b.bin_label for b in self.bins if not b.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "auc_floor": self.auc_floor,
            "bin_edges": list(self.bin_edges),
            "bins": [b.to_dict() for b in self.bins],
            "passed": self.passed,
            "failed_bins": list(self.failed_bins),
        }


def _bin_label(lo: float, hi: float) -> str:
    def fmt(v: float) -> str:
        if math.isinf(v):
            return "inf"
        if v >= 1024 and v % 1024 == 0:
            return f"{int(v // 1024)}k"
        return f"{int(v)}" if float(v).is_integer() else f"{v:g}"

    return f"[{fmt(lo)}, {fmt(hi)})"


def length_bin_gate(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    context_lengths: Sequence[float] | np.ndarray,
    *,
    auc_floor: float,
    bin_edges: Sequence[float] = DEFAULT_BIN_EDGES,
) -> LengthBinGate:
    """Per-length-bin discrimination gate (PUBLICATION.md D8 §8.6(b)).

    Charter: "Per-length-bin discrimination gate: [...] pre-registered
    separation threshold; an instrument failing a bin loses that bin". A
    detector that fails only on long contexts would silently bias exactly
    the memory-pressure axis, so discrimination is required WITHIN every
    bin, never pooled. Bins are right-open ``[lo, hi)`` over ``bin_edges``
    (default = the charter's 1k/4k/8k/16k/32k grid plus a +inf tail).

    ``auc_floor`` is the pre-registered separation threshold (§8.6(b)); it
    is deliberately REQUIRED — a silent default here would become the
    registered value without anyone deciding it (fail-closed doctrine).

    Fail-closed: an empty bin or a one-class bin is an anchor-coverage
    failure (the gate cannot be evaluated there) and raises
    :class:`InstrumentCalibrationError`; a passing report must have measured
    discrimination in EVERY bin. Items outside the edge range also raise.
    """
    if not 0.0 < auc_floor <= 1.0:
        raise InstrumentCalibrationError(
            f"auc_floor must be in (0, 1], got {auc_floor!r}"
        )
    edges = tuple(float(e) for e in bin_edges)
    if len(edges) < 2:
        raise InstrumentCalibrationError(
            f"bin_edges needs at least 2 edges, got {len(edges)}"
        )
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise InstrumentCalibrationError(
            f"bin_edges must be strictly increasing, got {edges}"
        )
    s = _as_finite_1d(scores, "scores")
    y = _as_binary_labels(labels)
    lengths = _as_finite_1d(context_lengths, "context_lengths")
    if not (s.size == y.size == lengths.size):
        raise InstrumentCalibrationError(
            f"scores ({s.size}), labels ({y.size}) and context_lengths "
            f"({lengths.size}) differ in length"
        )
    if (lengths < edges[0]).any() or (lengths >= edges[-1]).any():
        raise InstrumentCalibrationError(
            f"context_lengths fall outside the bin range [{edges[0]}, {edges[-1]})"
        )

    results: list[LengthBinResult] = []
    for lo, hi in zip(edges, edges[1:]):
        mask = (lengths >= lo) & (lengths < hi)
        label = _bin_label(lo, hi)
        if not mask.any():
            raise InstrumentCalibrationError(
                f"length bin {label} is EMPTY — the §8.6(b) gate requires "
                "anchor coverage in every bin (fail closed); adjust bin_edges "
                "or extend the anchor"
            )
        bin_y = y[mask]
        n_g = int(bin_y.sum())
        n_u = int(bin_y.size - n_g)
        if n_g == 0 or n_u == 0:
            raise InstrumentCalibrationError(
                f"length bin {label} has only one class "
                f"({n_g} grounded / {n_u} ungrounded) — discrimination is "
                "unmeasurable there (fail closed, §8.6(b))"
            )
        auc = roc_auc(s[mask], bin_y)
        results.append(
            LengthBinResult(
                bin_label=label,
                lo=lo,
                hi=hi,
                n_grounded=n_g,
                n_ungrounded=n_u,
                auc=auc,
                auc_floor=float(auc_floor),
                passed=bool(auc >= auc_floor - _FLOOR_ATOL),
            )
        )
    return LengthBinGate(
        auc_floor=float(auc_floor), bin_edges=edges, bins=tuple(results)
    )


# --------------------------------------------------------------------------- #
# (3) Post-campaign drift audit — D8 §8.6(e)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DriftAudit:
    """Calibration-transfer check (PUBLICATION.md D8 §8.6(e)).

    Compares a re-scored anchor sample against the stored calibration-time
    scores for the SAME items. ``flip_rate_at_tau`` is the operative
    metric: the fraction of items whose grounded/ungrounded verdict at the
    frozen τ changed — if verdicts flip, the registered τ no longer means
    what it meant at calibration time.
    """

    n_items: int
    tau: float
    mean_abs_delta: float
    max_abs_delta: float
    flip_rate_at_tau: float
    max_mean_abs_delta: float | None
    max_flip_rate: float | None
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _as_score_series(
    obj: Mapping[str, float] | pd.Series, name: str
) -> pd.Series:
    """id -> score series with unique ids (fail closed on duplicates —
    ``dict()`` would silently keep the last duplicate, hiding a provenance
    bug in the stored anchor sample)."""
    if isinstance(obj, pd.Series):
        series = obj.astype(float)
    elif isinstance(obj, Mapping):
        series = pd.Series(dict(obj), dtype=float)
    else:
        raise InstrumentCalibrationError(
            f"{name} must be a mapping or pandas Series of id -> score, "
            f"got {type(obj)!r}"
        )
    if series.empty:
        raise InstrumentCalibrationError(f"{name} is empty")
    if series.index.has_duplicates:
        raise InstrumentCalibrationError(f"{name} has duplicate item ids")
    return series


def drift_audit(
    calibration_scores: Mapping[str, float] | pd.Series,
    recheck_scores: Mapping[str, float] | pd.Series,
    tau: float,
    *,
    max_mean_abs_delta: float | None = None,
    max_flip_rate: float | None = None,
) -> DriftAudit:
    """Post-campaign drift audit hook (PUBLICATION.md D8 §8.6(e)).

    ``calibration_scores`` and ``recheck_scores`` map item id -> instrument
    score (calibration-time vs re-scored). Items are joined by id and must
    match EXACTLY — a missing or extra item means the stored anchor sample
    was not what was re-scored, which is a provenance failure (fail closed).

    At least one threshold (``max_mean_abs_delta`` or ``max_flip_rate``)
    must be provided: a drift audit with no acceptance criterion cannot
    fail, which would make the §8.6(e) check decorative.
    """
    if max_mean_abs_delta is None and max_flip_rate is None:
        raise InstrumentCalibrationError(
            "drift audit requires at least one threshold "
            "(max_mean_abs_delta and/or max_flip_rate) — an audit that "
            "cannot fail is not a check (§8.6(e))"
        )
    for name, value in (
        ("max_mean_abs_delta", max_mean_abs_delta),
        ("max_flip_rate", max_flip_rate),
    ):
        if value is not None and (not np.isfinite(value) or value < 0.0):
            raise InstrumentCalibrationError(
                f"{name} must be a finite non-negative float, got {value!r}"
            )
    if not np.isfinite(tau):
        raise InstrumentCalibrationError(f"tau must be finite, got {tau!r}")

    cal = _as_score_series(calibration_scores, "calibration_scores")
    new = _as_score_series(recheck_scores, "recheck_scores")
    missing = sorted(set(cal.index) - set(new.index))
    extra = sorted(set(new.index) - set(cal.index))
    if missing or extra:
        raise InstrumentCalibrationError(
            "drift audit item sets differ — re-scored sample must be exactly "
            f"the stored anchor sample (missing from recheck: {missing[:5]}, "
            f"unexpected in recheck: {extra[:5]})"
        )
    new = new.reindex(cal.index)
    if not np.all(np.isfinite(cal.to_numpy())) or not np.all(
        np.isfinite(new.to_numpy())
    ):
        raise InstrumentCalibrationError("drift audit scores contain non-finite values")

    deltas = (new - cal).abs()
    mean_abs = float(deltas.mean())
    max_abs = float(deltas.max())
    flips = (cal.to_numpy() >= tau) != (new.to_numpy() >= tau)
    flip_rate = float(flips.mean())

    passed = True
    if max_mean_abs_delta is not None and mean_abs > max_mean_abs_delta + _FLOOR_ATOL:
        passed = False
    if max_flip_rate is not None and flip_rate > max_flip_rate + _FLOOR_ATOL:
        passed = False
    return DriftAudit(
        n_items=int(cal.size),
        tau=float(tau),
        mean_abs_delta=mean_abs,
        max_abs_delta=max_abs,
        flip_rate_at_tau=flip_rate,
        max_mean_abs_delta=max_mean_abs_delta,
        max_flip_rate=max_flip_rate,
        passed=passed,
    )


# --------------------------------------------------------------------------- #
# (4) The calibration report artifact — feeds the D9 §9.13 gate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InstrumentCalibrationReport:
    """The L4 calibration artifact for ONE instrument (D8 §8.6 → D9 §9.13).

    ``passed`` is the registration gate: τ selection succeeded (its very
    existence proves the precision floor was reached), the per-length-bin
    gate passed, and — when present — the drift audit passed. The D9 chain
    calls :func:`assert_registrable` on this report before assembling
    PRE_REGISTRATION.md, mirroring how
    ``src.analysis.stats.prereg.assemble_preregistration`` hard-fails on a
    failing §9.7 stats ``CalibrationReport``.
    """

    instrument_name: str
    instrument_version: str
    anchor: AnchorIdentity
    tau_selection: TauSelection
    length_bin_gate: LengthBinGate
    drift: DriftAudit | None = field(default=None)

    @property
    def passed(self) -> bool:
        gate_ok = self.length_bin_gate.passed
        drift_ok = self.drift is None or self.drift.passed
        return gate_ok and drift_ok

    def to_dict(self) -> dict[str, Any]:
        """JSON-artifact payload (schema ``SCHEMA_VERSION``).

        A superset of what the pre-registration assembler needs: identity,
        the frozen τ + its anchor, per-bin gate outcomes and the overall
        ``passed`` verdict the D9 chain gates on.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": ARTIFACT_KIND,
            "instrument": {
                "name": self.instrument_name,
                "version": self.instrument_version,
            },
            "anchor": self.anchor.to_dict(),
            "tau_selection": self.tau_selection.to_dict(),
            "length_bin_gate": self.length_bin_gate.to_dict(),
            "drift_audit": self.drift.to_dict() if self.drift is not None else None,
            "passed": self.passed,
        }

    def to_markdown(self) -> str:
        """Registration-embeddable section, mirroring the §9.7 stats
        ``CalibrationReport.to_markdown`` so the assembler can concatenate
        both under its calibration heading."""
        ts = self.tau_selection
        lines = [
            f"## Instrument calibration report (D8 §8.6) — "
            f"{self.instrument_name}@{self.instrument_version}",
            "",
            f"- anchor: `{self.anchor.dataset}/{self.anchor.split}` "
            f"(n={self.anchor.n_items}, "
            f"sha256=`{self.anchor.fingerprint_sha256[:16]}…`)",
            f"- overall: {'PASS' if self.passed else 'FAIL'}",
            "",
            "### τ selection (§8.6(c): smallest threshold with precision ≥ "
            f"{ts.precision_floor:g})",
            "",
            "| τ | precision | recall (sens.) | specificity | predicted grounded |",
            "|---|---|---|---|---|",
            (
                f"| {ts.tau:g} | {ts.precision_at_tau:.4f} | {ts.recall_at_tau:.4f} "
                f"| {ts.specificity_at_tau:.4f} | {ts.n_predicted_grounded} |"
            ),
            "",
            f"### Per-length-bin discrimination gate (§8.6(b), AUC floor "
            f"{self.length_bin_gate.auc_floor:g})",
            "",
            "| bin | n grounded | n ungrounded | AUC | verdict |",
            "|---|---|---|---|---|",
        ]
        for b in self.length_bin_gate.bins:
            lines.append(
                f"| {b.bin_label} | {b.n_grounded} | {b.n_ungrounded} "
                f"| {b.auc:.4f} | {'PASS' if b.passed else 'FAIL'} |"
            )
        lines.append("")
        if self.drift is not None:
            d = self.drift
            lines += [
                "### Post-campaign drift audit (§8.6(e))",
                "",
                "| n | mean |Δ| | max |Δ| | flip rate @ τ | verdict |",
                "|---|---|---|---|---|",
                (
                    f"| {d.n_items} | {d.mean_abs_delta:.4f} | {d.max_abs_delta:.4f} "
                    f"| {d.flip_rate_at_tau:.4f} | {'PASS' if d.passed else 'FAIL'} |"
                ),
                "",
            ]
        return "\n".join(lines)


def calibrate_instrument(
    anchor_data: pd.DataFrame | str | Path,
    *,
    instrument_name: str,
    instrument_version: str,
    dataset: str,
    split: str,
    auc_floor: float,
    precision_floor: float = DEFAULT_PRECISION_FLOOR,
    bin_edges: Sequence[float] = DEFAULT_BIN_EDGES,
    score_col: str = "score",
    label_col: str = "label",
    length_col: str = "context_length",
) -> InstrumentCalibrationReport:
    """Run the pre-campaign L4 suite (τ rule + length-bin gate) in one call.

    Consumes staged anchor data (DataFrame or JSONL path; nothing is
    downloaded) and returns the :class:`InstrumentCalibrationReport`
    artifact. The post-campaign drift audit is attached later via
    :func:`attach_drift_audit`, because it can only exist after the
    campaign (D8 §8.6(e)).
    """
    frame, identity = load_anchor(
        anchor_data,
        dataset=dataset,
        split=split,
        score_col=score_col,
        label_col=label_col,
        length_col=length_col,
    )
    scores = frame[score_col].to_numpy(dtype=float)
    labels = frame[label_col].to_numpy()
    lengths = frame[length_col].to_numpy(dtype=float)
    tau_sel = select_tau(scores, labels, identity, precision_floor=precision_floor)
    gate = length_bin_gate(
        scores, labels, lengths, auc_floor=auc_floor, bin_edges=bin_edges
    )
    return InstrumentCalibrationReport(
        instrument_name=instrument_name,
        instrument_version=instrument_version,
        anchor=identity,
        tau_selection=tau_sel,
        length_bin_gate=gate,
    )


def attach_drift_audit(
    report: InstrumentCalibrationReport, drift: DriftAudit
) -> InstrumentCalibrationReport:
    """Return a new report carrying the post-campaign drift audit (§8.6(e)).

    The drift τ must equal the report's frozen τ — auditing transfer at a
    different threshold than the registered one would be a category error.
    """
    if not math.isclose(drift.tau, report.tau_selection.tau, rel_tol=0, abs_tol=1e-12):
        raise InstrumentCalibrationError(
            f"drift audit τ ({drift.tau!r}) differs from the calibrated τ "
            f"({report.tau_selection.tau!r}) — the audit must use the frozen τ"
        )
    return dataclasses.replace(report, drift=drift)


def write_report(report: InstrumentCalibrationReport, path: str | Path) -> Path:
    """Write the JSON calibration artifact (always — failure is documented,
    registration is where a failed report becomes fatal via
    :func:`assert_registrable`)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return out


def load_report(path: str | Path) -> InstrumentCalibrationReport:
    """Load + validate a JSON calibration artifact (fail closed on schema)."""
    p = Path(path)
    if not p.is_file():
        raise InstrumentCalibrationError(f"calibration report not found: {p}")
    try:
        payload = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise InstrumentCalibrationError(
            f"calibration report {p} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise InstrumentCalibrationError(
            f"calibration report {p} must be a JSON object"
        )
    missing = [k for k in _REQUIRED_REPORT_KEYS if k not in payload]
    if missing:
        raise InstrumentCalibrationError(
            f"calibration report {p} missing required keys {missing}"
        )
    if payload["artifact"] != ARTIFACT_KIND:
        raise InstrumentCalibrationError(
            f"calibration report {p} has artifact kind {payload['artifact']!r}, "
            f"expected {ARTIFACT_KIND!r}"
        )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise InstrumentCalibrationError(
            f"calibration report {p} has schema_version "
            f"{payload['schema_version']!r}, expected {SCHEMA_VERSION}"
        )
    try:
        anchor = AnchorIdentity(**payload["anchor"])
        tau_payload = dict(payload["tau_selection"])
        tau_payload["anchor"] = AnchorIdentity(**tau_payload["anchor"])
        tau_sel = TauSelection(**tau_payload)
        gate_payload = payload["length_bin_gate"]
        gate = LengthBinGate(
            auc_floor=gate_payload["auc_floor"],
            bin_edges=tuple(gate_payload["bin_edges"]),
            bins=tuple(LengthBinResult(**b) for b in gate_payload["bins"]),
        )
        drift_payload = payload["drift_audit"]
        drift = DriftAudit(**drift_payload) if drift_payload is not None else None
    except (KeyError, TypeError) as exc:
        raise InstrumentCalibrationError(
            f"calibration report {p} has a malformed section: {exc}"
        ) from exc
    report = InstrumentCalibrationReport(
        instrument_name=payload["instrument"]["name"],
        instrument_version=payload["instrument"]["version"],
        anchor=anchor,
        tau_selection=tau_sel,
        length_bin_gate=gate,
        drift=drift,
    )
    if bool(payload["passed"]) != report.passed:
        raise InstrumentCalibrationError(
            f"calibration report {p} stored passed={payload['passed']!r} but "
            f"recomputation gives {report.passed} — artifact is inconsistent"
        )
    return report


def assert_registrable(report: InstrumentCalibrationReport) -> None:
    """The D9 refuse-to-register hook (PUBLICATION.md §9.13, D8 §8.6).

    Raises :class:`InstrumentCalibrationError` when the instrument failed
    L4 calibration — mirroring ``assemble_preregistration``'s hard failure
    on a failing §9.7 stats calibration, so the registration chain refuses
    to register on failed instrument calibration exactly as designed.
    """
    if report.passed:
        return
    reasons: list[str] = []
    if not report.length_bin_gate.passed:
        reasons.append(
            "length-bin gate FAILED in bins "
            f"{list(report.length_bin_gate.failed_bins)} (§8.6(b))"
        )
    if report.drift is not None and not report.drift.passed:
        reasons.append("drift audit FAILED (§8.6(e))")
    raise InstrumentCalibrationError(
        f"instrument {report.instrument_name}@{report.instrument_version} failed "
        f"L4 calibration — registration is BLOCKED (§9.13): " + "; ".join(reasons)
    )
