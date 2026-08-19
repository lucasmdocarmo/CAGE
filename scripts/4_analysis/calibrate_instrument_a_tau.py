#!/usr/bin/env python3
"""Instrument-A (LettuceDetect) τ calibration for the Qasper Y predicate.

Task #146 (owner decision #120/F7, DECIDED 2026-08-19; ADR-0091): charter
§8.5 defines the Qasper veridicality predicate as "groundedness-based
(Instrument A at calibrated τ)", but the only registered τ (0.817024,
ADR-0038) belongs to Instrument B. This CLI produces the REGISTERED
Instrument-A τ by calibrating the PINNED LettuceDetect detector on the same
public RAGTruth/TRUE anchor pool the 2026-08-05 Instrument-B selection used
(MyDocs/registration/instrument_selection_2026-08-05/anchors/, 4,724 items).
The resulting τ is consumed by scripts/4_analysis/build_predicate_table.py
--qasper-tau (src/analysis/predicate.py refuses Qasper windows without it)
and embedded into PRE_REGISTRATION.md at the #112 freeze.

STAMP: instrument calibration on PUBLIC anchors — design input, NOT study
findings (same stamp as the Instrument-B selection calibration).

REGISTERED SELECTION RULE (registration-bound; coded and documented here
BEFORE any anchor was ever scored by Instrument A — prep task 2026-08-19):

- Scoring: each anchor (question?, context, claim) triple is scored by
  ``QualityEvaluator.evaluate_hallucination`` (src/evaluation/quality.py, the
  campaign's own Instrument-A path: pinned model via CAGE_LETTUCEDETECT_MODEL,
  optional fail-closed revision pin via CAGE_LETTUCEDETECT_REVISION, strict
  mode). The anchor score is ``grounding_score`` in [0, 1] (1 − hallucinated
  char ratio). An unscoreable anchor REFUSES the whole calibration.
- Direction (matches src/analysis/predicate.py QASPER_RULE
  ``qasper_grounding_at_tau``): an item is predicted GROUNDED iff
  ``grounding_score >= τ``; predicted HALLUCINATED iff ``grounding_score < τ``.
  Operating characteristics are reported for HALLUCINATION DETECTION
  (positive class = hallucinated = anchor ``label == 0``).
- Candidate grid: the unique observed anchor scores (ascending) — the
  operating characteristics are step functions changing only at observed
  scores, so this grid realizes every operating point that can be OPTIMAL
  for τ ∈ [0, 1]. (The one point it omits — τ above the max observed
  score, i.e. predict-everything-hallucinated — has balanced accuracy 0.5,
  which the smallest candidate always matches, so it can never be the
  optimum nor extend the sensitivity band.)
- PRIMARY RULE: τ* = the candidate τ maximizing BALANCED ACCURACY
  ( (recall_hallucinated + specificity_hallucinated) / 2 ) on the POOLED
  4,724-item anchor (RAGTruth-test + FRANK-test + QAGS-CNNDM + QAGS-XSum).
  Ties broken deterministically by the SMALLEST such τ.
- SENSITIVITY BAND: all candidate τ whose balanced accuracy is within 1
  percentage point of the optimum (``BAND_TOLERANCE = 0.01``); reported as
  [τ_low, τ_high] plus the qualifying-candidate count.
- SUPPLEMENTARY (reported, NOT registered): the same rule on the
  RAGTruth-test-only subset (n=2675) — sensitivity information for the D9
  anchor-scope discussion (ADR-0038 chose RAGTruth-only for Instrument B).

MANIFEST SCHEMA (registration-bound; ``calibration_manifest.json`` keys):

    schema_version        int   (= SCHEMA_VERSION)
    artifact              str   (= ARTIFACT_KIND)
    created_utc           str   ISO-8601 UTC
    task                  str   "#146 Instrument-A (LettuceDetect) tau"
    tau                   float THE registered Instrument-A τ
    rule                  dict  the pre-declared selection rule (SELECTION_RULE)
    band                  dict  {tolerance, balanced_accuracy_floor, tau_low,
                                 tau_high, n_candidates_in_band}
    operating_point       dict  full operating row at τ*
    operating_table       list  per-candidate {tau, tp, fp, fn, tn, precision,
                                 recall, f1, specificity, balanced_accuracy}
    supplementary_ragtruth_only  dict  same-rule τ on RAGTruth-test only
    anchor_inventory      dict  per-component {path, sha256, n_rows,
                                 n_grounded, n_hallucinated} + totals
    expected_anchors      int   the --expected-anchors the pool was validated
                                 against
    instrument            dict  {instrument_id, provenance
                                 (QualityEvaluator.instrument_provenance()),
                                 env {CAGE_LETTUCEDETECT_MODEL,
                                      CAGE_LETTUCEDETECT_REVISION}}
    seed                  int   recorded seed (scoring is deterministic; any
                                 future subsampling must derive from it)
    scores_file           str   sibling per-item score JSONL (drift-audit
                                 stash, §8.6(e) convention)
    scores_sha256         str   sha256 of that file

Fail-closed doctrine (mirrors src/evaluation/instrument_calibration.py):
missing/malformed anchors, count mismatch against --expected-anchors,
one-class components, unscoreable items, and manifest overwrite without
--force all REFUSE. ``--plan`` validates the anchors and prints the execution
plan + runtime estimate WITHOUT importing any model stack (prep-verification
mode; the scoring path lives behind the ``build_scorer`` seam so tests stub
the detector).

Run with the project venv: ``.venv/bin/python``. GO time only (multi-hour
CPU run + one-time model download) — see
MyDocs/registration/instrument_a_tau_calibration/PREP_2026-08-19.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION: int = 1
ARTIFACT_KIND: str = "instrument_a_tau_calibration"
MANIFEST_NAME: str = "calibration_manifest.json"
SCORES_NAME: str = "scores_instrument_a.jsonl"
TASK_LABEL: str = "#146 Instrument-A (LettuceDetect) tau"

#: Registration-bound band width: candidates within 1 percentage point of the
#: optimum balanced accuracy form the sensitivity band.
BAND_TOLERANCE: float = 0.01

#: Recorded seed. Scoring is deterministic; the seed exists so any FUTURE
#: subsampling (none today) has a registered origin.
DEFAULT_SEED: int = 20260819

#: Default LettuceDetect model id. MUST equal the QualityEvaluator in-code
#: default (drift-guarded by tests/test_calibrate_instrument_a_tau.py) —
#: duplicated here so --plan mode never imports the model stack.
DEFAULT_LETTUCEDETECT_MODEL: str = "KRLabsOrg/lettucedect-base-modernbert-en-v1"

#: The anchor pool: the 2026-08-05 Instrument-B selection anchors, REUSED
#: verbatim (owner decision #120/F7: "the SAME RAGTruth/TRUE anchors").
DEFAULT_ANCHORS_DIR: Path = (
    REPO_ROOT / "MyDocs/registration/instrument_selection_2026-08-05/anchors"
)
DEFAULT_OUT_ROOT: Path = (
    REPO_ROOT / "MyDocs/registration/instrument_a_tau_calibration"
)

#: Fixed component files (stems) of the anchor pool; a missing file refuses.
ANCHOR_COMPONENTS: tuple[str, ...] = (
    "frank_test",
    "qags_cnndm",
    "qags_xsum",
    "ragtruth_test",
)
#: The component whose subset feeds the supplementary (non-registered) τ.
SUPPLEMENTARY_COMPONENT: str = "ragtruth_test"

#: Anchor row contract (seeded 2026-08-05 schema). ``label``: 1 = grounded,
#: 0 = hallucinated. ``question`` may be null (FRANK/QAGS; RAGTruth QA rows
#: carry text).
REQUIRED_ANCHOR_FIELDS: tuple[str, ...] = (
    "id",
    "component",
    "source_dataset",
    "question",
    "context",
    "claim",
    "label",
)

#: The pre-declared selection rule, embedded verbatim into the manifest.
SELECTION_RULE: dict[str, str] = {
    "id": "max_balanced_accuracy_pooled_v1",
    "positive_class": "hallucinated (anchor label == 0)",
    "prediction": (
        "predicted hallucinated iff grounding_score < tau; predicted grounded "
        "iff grounding_score >= tau (matches src/analysis/predicate.py "
        "qasper_grounding_at_tau)"
    ),
    "candidate_grid": "unique observed anchor grounding scores (ascending)",
    "primary": (
        "tau* = candidate tau maximizing balanced accuracy "
        "((recall_hallucinated + specificity_hallucinated) / 2) on the POOLED "
        "RAGTruth/TRUE anchor; ties -> smallest tau"
    ),
    "band": (
        "all candidate tau with balanced accuracy >= optimum - 0.01 "
        "(1 percentage point); reported as [tau_low, tau_high] + count"
    ),
    "declared": (
        "coded and documented at prep time 2026-08-19, BEFORE any anchor was "
        "scored by Instrument A (task #146 prep; owner decision #120/F7, "
        "ADR-0091)"
    ),
}

#: Runtime-estimate basis (plan mode + prep doc). CPU seconds per anchor for
#: a ~150M-param ModernBERT token-classifier on 110-2295-token contexts
#: (Apple-Silicon CPU, single process; contexts all fit L_max ~8k natively).
SECONDS_PER_ITEM_RANGE: tuple[float, float] = (1.5, 3.0)

_MANIFEST_REQUIRED_KEYS: tuple[str, ...] = (
    "schema_version",
    "artifact",
    "created_utc",
    "task",
    "tau",
    "rule",
    "band",
    "operating_point",
    "operating_table",
    "supplementary_ragtruth_only",
    "anchor_inventory",
    "expected_anchors",
    "instrument",
    "seed",
    "scores_file",
    "scores_sha256",
)


class CalibrationError(RuntimeError):
    """Any #146 calibration input/contract violation (fail loud, message first)."""


# --------------------------------------------------------------------------- #
# (1) Anchor loading + validation (fail-closed)
# --------------------------------------------------------------------------- #


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CalibrationError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise CalibrationError(
                f"{path}:{lineno}: record must be a JSON object, "
                f"got {type(obj).__name__}"
            )
        rows.append(obj)
    return rows


def _validate_anchor_row(row: dict[str, Any], component: str, where: str) -> None:
    missing = [k for k in REQUIRED_ANCHOR_FIELDS if k not in row]
    if missing:
        raise CalibrationError(f"{where}: missing required fields {missing}")
    if not isinstance(row["id"], str) or not row["id"].strip():
        raise CalibrationError(f"{where}: 'id' must be a non-empty string")
    if row["component"] != component:
        raise CalibrationError(
            f"{where}: component {row['component']!r} != file component "
            f"{component!r} (anchor files are per-component)"
        )
    if not isinstance(row["source_dataset"], str) or not row["source_dataset"].strip():
        raise CalibrationError(f"{where}: 'source_dataset' must be a non-empty string")
    if row["question"] is not None and not isinstance(row["question"], str):
        raise CalibrationError(f"{where}: 'question' must be null or a string")
    for field in ("context", "claim"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise CalibrationError(f"{where}: {field!r} must be a non-empty string")
    label = row["label"]
    if isinstance(label, bool) or label not in (0, 1):
        raise CalibrationError(
            f"{where}: 'label' must be 0 (hallucinated) or 1 (grounded), "
            f"got {label!r}"
        )


def load_anchors(
    anchors_dir: Path, expected_anchors: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load + validate the pooled anchor set (fail-closed).

    Returns ``(rows, inventory)``: the pooled anchor rows (in component-file
    order) and the manifest ``anchor_inventory`` payload (per-component path,
    sha256, counts, label breakdown + pooled totals).
    """
    if not isinstance(expected_anchors, int) or isinstance(expected_anchors, bool):
        raise CalibrationError(
            f"expected_anchors must be an int, got {expected_anchors!r}"
        )
    if expected_anchors <= 0:
        raise CalibrationError(
            f"expected_anchors must be positive, got {expected_anchors}"
        )
    anchors_dir = Path(anchors_dir)
    if not anchors_dir.is_dir():
        raise CalibrationError(f"anchors dir not found: {anchors_dir}")

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    per_component: dict[str, Any] = {}
    for component in ANCHOR_COMPONENTS:
        path = anchors_dir / f"{component}.jsonl"
        if not path.is_file():
            raise CalibrationError(
                f"anchor component file missing: {path} — the #146 pool is "
                f"the FIXED 2026-08-05 seeded set {list(ANCHOR_COMPONENTS)}"
            )
        component_rows = _read_jsonl(path)
        if not component_rows:
            raise CalibrationError(f"anchor component file is empty: {path}")
        n_grounded = 0
        for i, row in enumerate(component_rows, 1):
            _validate_anchor_row(row, component, f"{path}:row {i}")
            if row["id"] in seen_ids:
                raise CalibrationError(
                    f"{path}:row {i}: duplicate anchor id {row['id']!r} in pool"
                )
            seen_ids.add(row["id"])
            n_grounded += int(row["label"])
        n_hall = len(component_rows) - n_grounded
        if n_grounded == 0 or n_hall == 0:
            raise CalibrationError(
                f"anchor component {component!r} has only one class "
                f"({n_grounded} grounded / {n_hall} hallucinated) — "
                "operating characteristics are unmeasurable (fail closed)"
            )
        try:
            rel = str(path.resolve().relative_to(REPO_ROOT))
        except ValueError:
            rel = str(path)
        per_component[component] = {
            "path": rel,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_rows": len(component_rows),
            "n_grounded": n_grounded,
            "n_hallucinated": n_hall,
        }
        rows.extend(component_rows)

    if len(rows) != expected_anchors:
        raise CalibrationError(
            f"anchor pool has {len(rows)} rows but --expected-anchors "
            f"{expected_anchors} — refusing to calibrate on an unexpected "
            "pool (fail closed)"
        )
    inventory = {
        "components": per_component,
        "n_total": len(rows),
        "n_grounded": sum(c["n_grounded"] for c in per_component.values()),
        "n_hallucinated": sum(c["n_hallucinated"] for c in per_component.values()),
    }
    return rows, inventory


# --------------------------------------------------------------------------- #
# (2) τ sweep + pre-declared selection (pure math; offline-testable)
# --------------------------------------------------------------------------- #


def sweep_tau(
    scores: Sequence[float], hallucinated: Sequence[int]
) -> list[dict[str, Any]]:
    """Operating table over the candidate grid (unique observed scores, asc).

    ``hallucinated`` is the POSITIVE-class indicator (1 = hallucinated =
    anchor label 0). Per candidate τ: predicted hallucinated iff
    ``score < τ``. Returns one row per candidate with tp/fp/fn/tn, precision
    (None when nothing is predicted hallucinated), recall, f1, specificity
    and balanced accuracy — all for hallucination detection.
    """
    s = np.asarray(scores, dtype=float)
    h = np.asarray(hallucinated, dtype=float)
    if s.ndim != 1 or h.ndim != 1 or s.size == 0 or s.size != h.size:
        raise CalibrationError(
            f"scores ({s.shape}) and hallucinated labels ({h.shape}) must be "
            "equal-length non-empty 1-D arrays"
        )
    if not np.all(np.isfinite(s)) or s.min() < 0.0 or s.max() > 1.0:
        raise CalibrationError(
            "scores must be finite grounding scores within [0, 1]"
        )
    if not np.isin(h, (0.0, 1.0)).all():
        raise CalibrationError("hallucinated labels must be binary in {0, 1}")
    n_hall = int(h.sum())
    n_grounded = int(h.size - n_hall)
    if n_hall == 0 or n_grounded == 0:
        raise CalibrationError(
            "anchor pool must contain BOTH classes to sweep tau "
            f"(got {n_grounded} grounded / {n_hall} hallucinated)"
        )

    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    h_sorted = h[order]
    is_first = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    first_idx = np.flatnonzero(is_first)
    candidates = s_sorted[first_idx]
    # Items with score < candidates[k] are exactly the first_idx[k] sorted
    # items; hallucinated among them via a 0-prepended cumulative sum.
    cum_h = np.r_[0.0, np.cumsum(h_sorted)]
    tp = cum_h[first_idx]
    n_below = first_idx.astype(float)
    fp = n_below - tp
    fn = float(n_hall) - tp
    tn = float(n_grounded) - fp

    table: list[dict[str, Any]] = []
    for k in range(candidates.size):
        tp_k, fp_k, fn_k, tn_k = tp[k], fp[k], fn[k], tn[k]
        n_pred = tp_k + fp_k
        precision = (tp_k / n_pred) if n_pred > 0 else None
        recall = tp_k / n_hall
        specificity = tn_k / n_grounded
        f1 = (2.0 * tp_k) / (2.0 * tp_k + fp_k + fn_k)
        table.append(
            {
                "tau": float(candidates[k]),
                "tp": int(tp_k),
                "fp": int(fp_k),
                "fn": int(fn_k),
                "tn": int(tn_k),
                "precision": None if precision is None else float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "specificity": float(specificity),
                "balanced_accuracy": float((recall + specificity) / 2.0),
            }
        )
    return table


def select_tau(
    table: list[dict[str, Any]], *, band_tolerance: float = BAND_TOLERANCE
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the PRE-DECLARED rule (SELECTION_RULE) to an operating table.

    Returns ``(operating_point, band)``: the selected row (max balanced
    accuracy, ties -> smallest τ; the table is candidate-ascending by
    construction) and the sensitivity band (candidates within
    ``band_tolerance`` of the optimum).
    """
    if not table:
        raise CalibrationError("operating table is empty — nothing to select")
    ba = np.asarray([row["balanced_accuracy"] for row in table], dtype=float)
    best = float(ba.max())
    idx = int(np.flatnonzero(ba >= best)[0])  # smallest tau among exact ties
    floor = best - band_tolerance
    in_band = [row for row, v in zip(table, ba) if v >= floor]
    band = {
        "tolerance": float(band_tolerance),
        "balanced_accuracy_floor": float(floor),
        "tau_low": float(min(r["tau"] for r in in_band)),
        "tau_high": float(max(r["tau"] for r in in_band)),
        "n_candidates_in_band": len(in_band),
    }
    return dict(table[idx]), band


# --------------------------------------------------------------------------- #
# (3) Scoring seam — the ONLY place the model stack is touched
# --------------------------------------------------------------------------- #

# A scorer maps (question, context_docs, answer) -> grounding score in [0, 1].
Scorer = Callable[[str, list[str], str], Any]


def build_scorer(device: str = "cpu") -> tuple[Scorer, Callable[[], dict[str, Any]]]:
    """Construct the REAL Instrument-A scorer (imports the model stack).

    Deliberately the single seam between this CLI and
    ``src/evaluation/quality.py``: ``--plan`` mode never calls it, and tests
    stub either this function or the ``lettucedetect`` module underneath it.
    Returns ``(scorer, provenance)`` where ``provenance()`` yields the
    manifest ``instrument`` payload (must be called AFTER scoring so the
    lazily-captured HF revision is populated).
    """
    from src.evaluation.quality import QualityEvaluator  # heavy import: GO only

    evaluator = QualityEvaluator(
        use_nli=False,
        use_embeddings=False,
        use_bertscore=False,
        use_rouge=False,
        use_lettucedetect=True,
        device=device,
        strict=True,
        claim_checker="nli",
    )

    def scorer(question: str, context_docs: list[str], answer: str) -> Any:
        result = evaluator.evaluate_hallucination(question, context_docs, answer)
        return result["grounding_score"]

    def provenance() -> dict[str, Any]:
        return {
            "instrument_id": evaluator._grounding_instrument_id(),
            "provenance": evaluator.instrument_provenance(),
            "env": {
                "CAGE_LETTUCEDETECT_MODEL": os.environ.get(
                    "CAGE_LETTUCEDETECT_MODEL"
                ),
                "CAGE_LETTUCEDETECT_REVISION": os.environ.get(
                    "CAGE_LETTUCEDETECT_REVISION"
                ),
            },
        }

    return scorer, provenance


def score_anchors(
    rows: list[dict[str, Any]],
    scorer: Scorer,
    *,
    progress_every: int = 250,
) -> list[dict[str, Any]]:
    """Score every anchor; an unscoreable item refuses the calibration.

    Returns per-item score rows ``{id, component, label, score}`` in pool
    order (the drift-audit stash payload, §8.6(e) convention).
    """
    scored: list[dict[str, Any]] = []
    for i, row in enumerate(rows, 1):
        value = scorer(row["question"] or "", [row["context"]], row["claim"])
        if value is None:
            raise CalibrationError(
                f"anchor {row['id']!r} returned no grounding score — a "
                "partially-scored pool cannot register a τ (fail closed)"
            )
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise CalibrationError(
                f"anchor {row['id']!r} returned invalid grounding score "
                f"{value!r} (must be finite in [0, 1])"
            )
        scored.append(
            {
                "id": row["id"],
                "component": row["component"],
                "label": int(row["label"]),
                "score": value,
            }
        )
        if progress_every and i % progress_every == 0:
            print(f"[calibrate_instrument_a_tau] scored {i}/{len(rows)}")
    return scored


# --------------------------------------------------------------------------- #
# (4) Outputs
# --------------------------------------------------------------------------- #


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def refuse_overwrite(out_dir: Path, force: bool) -> None:
    """Refuse to clobber an existing calibration (registration artifact)."""
    for name in (MANIFEST_NAME, SCORES_NAME):
        path = out_dir / name
        if path.exists() and not force:
            raise CalibrationError(
                f"{path} already exists — a registered calibration is never "
                "silently overwritten; re-run deliberately with --force"
            )


def runtime_estimate(n_anchors: int) -> dict[str, Any]:
    lo_s, hi_s = SECONDS_PER_ITEM_RANGE
    return {
        "n_anchors": n_anchors,
        "seconds_per_item_range": [lo_s, hi_s],
        "estimated_hours_range": [
            round(n_anchors * lo_s / 3600.0, 1),
            round(n_anchors * hi_s / 3600.0, 1),
        ],
        "basis": (
            f"{n_anchors} anchors x {lo_s}-{hi_s} s/anchor for a ~150M-param "
            "ModernBERT token-classifier on 110-2295-token contexts "
            "(Apple-Silicon CPU, single process; every anchor fits L_max "
            "natively, no windowed pass) + one-time model download"
        ),
    }


def run_calibration(
    *,
    anchors_dir: Path,
    expected_anchors: int,
    out_dir: Path,
    seed: int,
    force: bool,
    scorer_factory: Callable[[], tuple[Scorer, Callable[[], dict[str, Any]]]],
) -> Path:
    """Full GO-time pipeline; returns the manifest path. Pure orchestration —
    the model stack enters only through ``scorer_factory``."""
    rows, inventory = load_anchors(anchors_dir, expected_anchors)
    refuse_overwrite(out_dir, force)  # BEFORE hours of scoring

    scorer, provenance = scorer_factory()
    scored = score_anchors(rows, scorer)

    scores = [r["score"] for r in scored]
    hallucinated = [1 - r["label"] for r in scored]
    table = sweep_tau(scores, hallucinated)
    operating_point, band = select_tau(table)

    supp_rows = [r for r in scored if r["component"] == SUPPLEMENTARY_COMPONENT]
    supp_table = sweep_tau(
        [r["score"] for r in supp_rows], [1 - r["label"] for r in supp_rows]
    )
    supp_point, supp_band = select_tau(supp_table)
    supplementary = {
        "component": SUPPLEMENTARY_COMPONENT,
        "n_rows": len(supp_rows),
        "tau": supp_point["tau"],
        "operating_point": supp_point,
        "band": supp_band,
        "note": (
            "same rule on RAGTruth-test only — sensitivity information for "
            "the D9 anchor-scope discussion (ADR-0038); NOT the registered τ"
        ),
    }

    scores_path = out_dir / SCORES_NAME
    _atomic_write_text(
        scores_path, "".join(json.dumps(r) + "\n" for r in scored)
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": TASK_LABEL,
        "tau": operating_point["tau"],
        "rule": SELECTION_RULE,
        "band": band,
        "operating_point": operating_point,
        "operating_table": table,
        "supplementary_ragtruth_only": supplementary,
        "anchor_inventory": inventory,
        "expected_anchors": expected_anchors,
        "instrument": provenance(),
        "seed": seed,
        "scores_file": SCORES_NAME,
        "scores_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
    }
    manifest_path = out_dir / MANIFEST_NAME
    _atomic_write_text(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest_path


def print_plan(
    anchors_dir: Path, expected_anchors: int, out_dir: Path, force: bool
) -> None:
    """--plan: validate anchors + print the GO plan. NO model import."""
    rows, inventory = load_anchors(anchors_dir, expected_anchors)
    estimate = runtime_estimate(len(rows))
    model = os.environ.get("CAGE_LETTUCEDETECT_MODEL", DEFAULT_LETTUCEDETECT_MODEL)
    revision = os.environ.get("CAGE_LETTUCEDETECT_REVISION") or (
        "(unset — record-only; set for a fail-closed pin)"
    )
    manifest_path = out_dir / MANIFEST_NAME
    collision = manifest_path.exists()
    print("[calibrate_instrument_a_tau] PLAN (no model loaded, nothing written)")
    print(f"  anchors        : {anchors_dir}")
    for name, comp in inventory["components"].items():
        print(
            f"    {name:<14}: n={comp['n_rows']:<5} grounded={comp['n_grounded']:<5}"
            f" hallucinated={comp['n_hallucinated']:<5} sha256={comp['sha256'][:16]}…"
        )
    print(
        f"  pool           : n={inventory['n_total']} "
        f"(grounded {inventory['n_grounded']} / hallucinated "
        f"{inventory['n_hallucinated']}) == --expected-anchors {expected_anchors} OK"
    )
    print(f"  instrument     : {model}")
    print(f"  revision pin   : {revision}")
    print(f"  rule           : {SELECTION_RULE['id']} — {SELECTION_RULE['primary']}")
    print(f"  band           : {SELECTION_RULE['band']}")
    print(f"  manifest       : {manifest_path}"
          + (" [EXISTS — GO will refuse without --force]" if collision and not force
             else ""))
    print(f"  scores stash   : {out_dir / SCORES_NAME}")
    print(
        f"  estimated time : {estimate['estimated_hours_range'][0]}-"
        f"{estimate['estimated_hours_range'][1]} h ({estimate['basis']})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate the Instrument-A (LettuceDetect) τ for the "
                    "Qasper Y predicate on the public RAGTruth/TRUE anchor "
                    "pool (task #146; owner decision #120/F7)."
    )
    parser.add_argument("--anchors-dir", type=Path, default=DEFAULT_ANCHORS_DIR,
                        help="anchor pool dir (default: the 2026-08-05 "
                             "Instrument-B selection anchors)")
    parser.add_argument("--expected-anchors", type=int, required=True,
                        help="REQUIRED pooled row count the anchors must "
                             "match exactly (fail closed); the seeded "
                             "2026-08-05 pool has 4724")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                        help="registration root; output goes to "
                             "<out-root>/<date>/")
    parser.add_argument("--date", type=str, default=None,
                        help="output subdir date (YYYY-MM-DD; default: "
                             "today UTC)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="recorded seed (scoring is deterministic)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="torch-style device for the detector "
                             "(default: cpu)")
    parser.add_argument("--plan", action="store_true",
                        help="validate anchors + print the execution plan and "
                             "runtime estimate WITHOUT loading any model "
                             "(prep-verification mode)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing calibration manifest "
                             "(never silent)")
    args = parser.parse_args(argv)

    date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        print(f"ERROR: --date {date!r} is not YYYY-MM-DD", file=sys.stderr)
        return 1
    out_dir = Path(args.out_root) / date

    try:
        if args.plan:
            print_plan(args.anchors_dir, args.expected_anchors, out_dir, args.force)
            return 0
        manifest_path = run_calibration(
            anchors_dir=args.anchors_dir,
            expected_anchors=args.expected_anchors,
            out_dir=out_dir,
            seed=args.seed,
            force=args.force,
            scorer_factory=lambda: build_scorer(device=args.device),
        )
    except CalibrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"[calibrate_instrument_a_tau] manifest: {manifest_path}")
    print(
        f"[calibrate_instrument_a_tau] tau = {manifest['tau']:.6g} "
        f"(balanced accuracy {manifest['operating_point']['balanced_accuracy']:.4f}; "
        f"band [{manifest['band']['tau_low']:.6g}, "
        f"{manifest['band']['tau_high']:.6g}])"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
