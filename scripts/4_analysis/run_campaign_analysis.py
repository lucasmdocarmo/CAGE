#!/usr/bin/env python3
"""Campaign analysis driver — the §9.11 ONE-LOOK policy in code (PUBLICATION.md D9).

Input: ONE organized run directory (results/<campaign>/<session>/<run_id>) that
already carries ``index/cells_index.csv`` from ``organize_results.py``. This
driver never organizes a tree itself — if the index is absent it tells you to
run the organizer first and stops.

Two modes, mutually exclusive:

- ``--design-input`` (DEFAULT): every output is stamped ``DESIGN-INPUT-ONLY``.
  Run it as often as you like; its numbers may inform design but are barred
  from the paper's confirmatory claims.
- ``--confirmatory``: requires BOTH ``--i-understand-one-look`` AND a
  ``--registered-sha`` string (the SHA of the frozen pre-registration this
  look executes). Outputs are stamped ``CONFIRMATORY`` and the run writes
  ``<run>/analysis_lock.json``; a second confirmatory invocation on the same
  run REFUSES — the campaign data is analyzed once (§9.11).

Registration binding (G1, 2026-08-16): the confirmatory look is BOUND to the
frozen registration, not to the CLI —

- metrics come from ``families.DEFAULT_METRICS`` (the registered §9.1
  co-primary pair); ``--metrics`` is design-input-only and a confirmatory
  override differing from the registered pair REFUSES naming both lists;
- ``--registered-sha`` must be 7-64 lowercase hex, must match the EXECUTING
  code's git HEAD, the worktree must be clean, and
  ``MyDocs/registration/PRE_REGISTRATION.md`` must exist with a matching
  embedded Machinery SHA (absent pre-freeze -> refusal naming task #112);
- ``--alpha`` must equal the registered ``REGISTERED_ALPHA``;
- §9.5 margins come from the registered-margins artifact
  (``registered_margins.json`` beside the prereg); a CLI ``--tost-margin``
  must match it or refuse;
- ADR-0086 realized-n: every primary per-query row records its realized n
  and confirmatory refuses below the registered floor unless a pre-declared
  step-down rung (``ADR0086_REALIZED_N_LADDER``) is explicitly accepted.

Pipeline per run:

1. Load ``index/cells_index.csv`` (the organize_results handoff table).
2. Resolve the requested §7.8 contrast ids against the registered
   ``src.analysis.stats.families.CONTRASTS`` registry (default: #4, the
   B6-vs-B3 headline, per dataset). Unknown ids FAIL LOUD.
3. Select the concrete cell/reference row-key pairs (family F1, one tuple
   slot differing) and join per-query metrics from each selected window's
   ``requests.jsonl`` + ``qa_evidence.jsonl`` (the v2-layout loader lives in
   this file; the pilot ``_results_loader.py`` stays pilot-only).
4. Per contrast × metric × dataset: ``tests_by_unit.paired_wilcoxon`` for
   continuous metrics, ``tests_by_unit.mcnemar_binary`` for the binary §8.5 Y
   predicate (§9.4), plus ``wlt.win_loss_tie`` (§8.13 mandatory triple) either
   way; window-unit baseline-pair contrasts take
   ``tests_by_unit.batch_means_contrast`` on window-level means (§9.4 — the
   only legal test under load). Multiplicity correction is tier-conditional
   (§9.1/§9.3): primary-tier contrasts carry NO cross-dataset correction
   (full α per dataset, pooling prohibited); every other tier gets a
   DIAGNOSTIC ``corrections.holm`` ACROSS DATASETS within each contrast ×
   metric — while the REGISTERED §9.3 Holm-WITHIN-FAMILY correction (sibling
   contrasts pooled per group×metric×dataset×family×unit via
   ``families.compile_family_map``) is executed by
   ``gatekeeping.evaluate_chain`` and reported in ``stats['gatekeeping']``
   with the full auditable trace (Dmitrienko serial order =
   ``families.PRIMARY_CHAIN_ORDER``).
5. §9.5: the family map's declared conditional-TOST equivalence legs
   (fingerprint NONE predictions) are computed via
   ``equivalence.conditional_tost`` (+ a ROPE sensitivity line) when the
   registered margin, the policy/none cell pair, and the S2 ``policy_event``
   mask exist — and listed as labeled skips otherwise.
6. Emit ``<run>/analysis/<timestamp>/{stats.json, summary.md,
   forest_<metric>.png, wlt_<metric>.png,
   wlt_<metric>_pooled_supplementary.png}``. Figures CONSUME the registered
   statistics (audit I1): ``render_figures`` feeds
   ``figure_pipeline.plot_forest_registered`` /
   ``plot_win_loss_tie_registered`` from the same contrast dicts written into
   stats.json — nothing is recomputed, per-dataset panels are the default
   W/L/T view (§9.1, audit I2), the pooled view is a disclosed supplementary
   file, and unrenderable metrics become counted skip entries in
   ``stats['figures']`` (audit I11). Rendering happens AFTER the one-time
   unblinding is logged (audit I10). Every output carries the mode stamp.

Confirmatory preconditions (all checked BEFORE the §9.11 lock is touched):

- ``--calibration-report``: a PASSING §9.7 calibration artifact
  (CalibrationReport JSON; refusal mirrors ``prereg.py``).
- §9.10 ledger verification: the raw tree must verify against its sealed
  content-hash ledger (``stats.ledger.verify_ledger``).
- §9.8 blinding: with a sealed arm map present, design-input outputs stay
  scrambled (labels masked, figures suppressed); the confirmatory look
  performs and records the ONE-TIME logged unblinding.

Guards (§9.4 unit-of-analysis rules):

- Rows whose family is F2/F3 (pressure) are NEVER per-query paired — they are
  batch-means-only territory. Pressure rows not consumed by a computed
  batch-means contrast are listed in a labeled skip block instead of
  producing wrong statistics.
- Registered contrasts whose selector is not a single baseline pair (e.g. the
  gated #6, the engine-slot #10) are reported as labeled skips, never
  silently dropped. The estimand-carrying primaries have executors (G4):
  #13's fingerprint superiority legs run per
  ``families.FINGERPRINT_SUB_HYPOTHESES`` (3 one-sided legs, Holm at the
  registered m=3, intersection-union p as the chain endpoint contribution);
  #14 (truth_tax) runs ``compute_truth_tax`` (task #119): per in-regime F2
  window, G − Y from ``goodput.evaluate_window`` over the requests ⋈
  §8.5-predicate join (predicate/<scoring_run_id>/ table, ms→s conversion at
  this one seam), batch-means contrast per cross-engine leg vs the vLLM
  anchor — with the SAME loud refusal when the predicate table, SLO floors,
  or regime artifacts are missing (naming exactly what is absent); #12
  (lambda_star_onset) computes the interpolated Chiu-Jain argmax from the
  F2 rate grid or fails loud naming the missing artifact.
- Sidedness (owner decision a, 2026-08-16): each row EXECUTES its REGISTERED
  sidedness from the §9.3 family-map row — one-sided rows derive their scipy
  ``alternative`` from the registered contrast direction
  (``REGISTERED_CELL_DIRECTION``) x the metric's direction; the executed
  alternative is recorded per row. Tie handling (decision b): the driver
  passes ``zero_method='pratt'`` (the registered value) into every paired
  Wilcoxon and surfaces the effective n (``n_nonzero``) per row.
- Tier routing (G2): a row's tier comes from the family-map ROW, never from
  ``Contrast.tier`` alone — ADR-0087-demoted exploratory rows (faithfulness)
  can never enter the primary chain or a Holm family; they receive BH-FDR in
  the separated ``stats['exploratory']`` section (G3), clearly
  non-confirmatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field, replace as dc_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import figure_pipeline as fp  # noqa: E402
from organize_results import GROUP_OF_MODEL  # noqa: E402
from src.analysis.stats import blinding as blinding_mod  # noqa: E402
from src.analysis.stats.blinding import (  # noqa: E402
    AlreadyUnblindedError,
    BlindingError,
)
from src.analysis.stats.calibration import (  # noqa: E402
    AAResult,
    CalibrationReport,
    InjectionResult,
)
from src.analysis.stats.corrections import benjamini_hochberg, holm  # noqa: E402
from src.analysis.stats.equivalence import (  # noqa: E402
    conditional_tost,
    rope_sensitivity,
)
from src.analysis.stats.families import (  # noqa: E402
    CONTRASTS,
    Contrast,
    DEFAULT_METRICS as REGISTERED_DEFAULT_METRICS,
    FINGERPRINT_CONTRAST_ID,
    FINGERPRINT_SUB_HYPOTHESES,
    FLOOR_SUITE_CONTRAST_ID,
    HEADLINE_CONTRAST_ID,
    KNOWN_DATASETS,
    PREDICATE_METRIC,
    PRIMARY_CHAIN_ORDER,
    UNGATED,
    FamilyMapError,
    chain_endpoint,
    compile_family_map,
)
from src.analysis.stats.gatekeeping import (  # noqa: E402
    GatekeepingError,
    GatekeepingTrace,
    PrimaryOutcome,
    SecondaryOutcome,
    evaluate_chain,
)
from src.analysis.stats.ledger import LedgerError, read_ledger, verify_ledger  # noqa: E402
from src.analysis.stats.tests_by_unit import (  # noqa: E402
    batch_means_contrast,
    mcnemar_binary,
    paired_wilcoxon,
)
from src.analysis.stats.wlt import win_loss_tie  # noqa: E402
from src.analysis.goodput import (  # noqa: E402
    GoodputError,
    IN_REGIME,
    SLOBaseline,
    evaluate_window,
)
from src.analysis.predicate import PREDICATE_DATASETS  # noqa: E402
from src.observability.provenance import (  # noqa: E402
    git_dirty as _prov_git_dirty,
    git_sha as _prov_git_sha,
)

Mode = Literal["design-input", "confirmatory"]

DESIGN_STAMP = "DESIGN-INPUT-ONLY"
CONFIRMATORY_STAMP = "CONFIRMATORY"
NOT_IMPLEMENTED_LABEL = "NOT-IMPLEMENTED-YET"
LOCK_NAME = "analysis_lock.json"
ANALYSIS_DIRNAME = "analysis"
STATS_JSON_NAME = "stats.json"
SUMMARY_MD_NAME = "summary.md"
SCHEMA_VERSION = 2
LEDGER_NAME = "ledger.json"

#: §9.8 blinding artifacts (run-root relative). The sealed map is produced at
#: scoring time by ``src.analysis.stats.blinding.scramble_labels``; this driver
#: only CONSUMES it.
BLINDING_DIRNAME = "blinding"
SEALED_MAP_NAME = "sealed_arm_map.json"
UNBLIND_LOG_NAME = "unblind_log.jsonl"

#: Registered gating topology (decision d, 2026-08-16 — G10): each secondary
#: row's upstream comes from the family map's ``upstream`` column, never from
#: a driver hard-code. The headline (#4) gates per (dataset x the secondary's
#: OWN metric); the estimand endpoints gate per (dataset x their registered
#: estimand variable) — this table maps a chain endpoint to the metric leg its
#: outcomes are keyed under (the co-primary keys are ``<dataset>|<metric>``).
_UPSTREAM_LEG_METRIC: dict[str, str] = {
    chain_endpoint(FINGERPRINT_CONTRAST_ID): "fingerprint",
    chain_endpoint(14): "truth_tax",
}

#: G1 (2026-08-16): confirmatory registration binding.
REGISTERED_ALPHA: float = 0.05
_REGISTERED_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
#: The SHA line ``prereg.assemble_preregistration`` embeds ("Machinery SHA:
#: `<sha>`") — the confirmatory look must execute exactly that machinery.
_PREREG_EMBEDDED_SHA_RE = re.compile(r"Machinery SHA: `([0-9a-f]{7,64})`")
#: The ONE tracked registration document (freeze task #112 lands it).
PREREG_PATH: Path = _REPO_ROOT / "MyDocs" / "registration" / "PRE_REGISTRATION.md"
#: Machine-readable §9.5 registered margins (metric -> margin, metric units),
#: frozen beside the prereg at #112. prereg.py's ``margins`` argument is
#: free-text registration prose; this sidecar is its numeric dual so the
#: driver can CHECK a margin instead of trusting the CLI (G1d/G9).
REGISTERED_MARGINS_PATH: Path = (
    _REPO_ROOT / "MyDocs" / "registration" / "registered_margins.json"
)

#: ADR-0086 (G16): registered per-dataset primary realized-n floor with the
#: pre-declared step-down ladder. The first rung is the registered floor; a
#: confirmatory look below it must explicitly accept a later rung
#: (``--accept-step-down``) and the acceptance is recorded in stats.json.
ADR0086_REALIZED_N_LADDER: tuple[int, ...] = (2000, 1600, 1200)

#: G14: the registered resampling seeds, passed EXPLICITLY to the §9.5
#: primitives (registration lives in the driver, not in primitive defaults)
#: and echoed into stats['provenance'].
BOOTSTRAP_SEED: int = 42
ROPE_SEED: int = 42

#: Registered direction of each ONE-SIDED baseline-pair contrast (decision a,
#: 2026-08-16): whether the registered claim is that the CELL (baseline_a)
#: comes out better or worse than the reference on the tested metric.
#: 'cell-worse' encodes the price/ablation contrasts (#5's note pins the
#: convention: "BERGEN monotone chain: B5 < B6"); #3 is the mirrored oracle
#: claim (retrieval PAYS vs gold context, so the B1 cell is better); #15 is
#: metric-dependent by its own registered wording ("latency saved vs truth
#: lost"): serving metrics -> cell-better, quality metrics -> cell-worse.
#: A one-sided row whose contrast has no entry here FAILS LOUD — running an
#: undeclared tail would be an unregistered test.
REGISTERED_CELL_DIRECTION: dict[int, str] = {
    1: "cell-worse",
    2: "cell-worse",
    3: "cell-better",
    5: "cell-worse",
    7: "cell-worse",
    15: "serving-better-quality-worse",
}

#: Direction registry for the registry-internal estimand variables (#12/#14
#: pinned metrics + the #13 pseudo-metric). Kept SEPARATE from
#: HIGHER_IS_BETTER because families.REGISTERED_METRICS is pinned equal to
#: the HIGHER_IS_BETTER keys (the G7 caller roster) and estimand variables
#: are never caller-suppliable.
ESTIMAND_HIGHER_IS_BETTER: dict[str, bool] = {
    # truth_tax = G - Y: a TAX, lower is better (§9.2).
    "truth_tax": False,
    # onset later = the system copes longer before the knee (§9.2).
    "lambda_star_onset": True,
    # fingerprint legs test harm on a quality instrument (higher better).
    "fingerprint": True,
}

#: §6.1/§9.2 floor suite (#12): rates are FRACTIONS of the predicted λ*, so
#: the prediction under test is onset at rate_frac = 1.0 with the registered
#: multiplicative ×/÷1.15 band; the knee estimator is the interpolated
#: Chiu-Jain power-metric argmax over the F2 rate grid.
LAMBDA_STAR_BAND: float = 1.15
LAMBDA_STAR_MIN_GRID_POINTS: int = 3
#: Per-window inputs of the Chiu-Jain power metric (goodput-weighted offered
#: rate over response time).
_LAMBDA_STAR_REQUIRED_COLUMNS: tuple[str, ...] = ("goodput_frac", "latency_ms")

#: Task #119 (§8.5): the joined predicate table — a post-seal sibling tree
#: produced by scripts/4_analysis/build_predicate_table.py, mirroring
#: cells/<row_key>/window_<k>/ so the per-query loader can join
#: ``predicate.jsonl`` beside requests.jsonl/qa_evidence.jsonl.
PREDICATE_DIRNAME = "predicate"
PREDICATE_ROWS_NAME = "predicate.jsonl"
PREDICATE_MANIFEST_NAME = "predicate_manifest.json"
_PREDICATE_MANIFEST_REQUIRED_KEYS: tuple[str, ...] = (
    "schema_version",
    "scoring_run_id",
    "raw_run_ledger_entries_sha256",
    "config",
    "counts",
)
#: The one honest producer-naming hint for a missing predicate column/table.
_PREDICATE_FIX_HINT = (
    "no §8.5 predicate table is joined for this run — build one from a "
    "sealed scoring pass: scripts/4_analysis/build_predicate_table.py "
    "<run> --scoring-run-id <id> --max-null-fraction <bound>"
)

#: #14 (truth_tax) executor inputs (G4c, task #119). Window pairs contrast
#: the ENGINE slot at identical pressure coordinates; the anchor engine is
#: vLLM (§7.3: the within-vLLM frontier #13 and the cross-engine bundles #14
#: share the anchor). Per-request serving columns required per window; the
#: ms→s conversion into ``goodput.evaluate_window`` happens HERE, at the one
#: registered seam (Topic-6 F2: no conversion existed anywhere).
_TRUTH_TAX_ANCHOR_ENGINE = "vllm"
_TRUTH_TAX_GROUP_AXES: tuple[str, ...] = (
    "model", "arm", "retriever", "policy", "topology", "budget_r", "rate_frac",
)
_TRUTH_TAX_REQUEST_COLUMNS: tuple[str, ...] = ("ok", "ttft_ms", "tpot_ms")
#: §3-extra manifest key carrying the §6.1 single-stream floors per engine:
#: {"slo_floors": {"<engine>": {"ttft_s": ..., "tpot_s": ...}}} — produced by
#: the E3 floor calibration (src/orchestration/calibration.summarize_floor).
_SLO_FLOORS_MANIFEST_KEY = "slo_floors"

#: #13 fingerprint superiority legs (G4a): the policy-axis pairings for the
#: 3 registered Holm legs. 'truncate' is NOT a policy value (§7.3) — the
#: truncation fingerprint rides the B11-vs-B6 ARM pair under pressure.
_FINGERPRINT_POLICY_OF_LEG: dict[str, str] = {
    "evict": "evict",
    "compress": "compress-fp8",
}
_FINGERPRINT_TRUNCATE_PAIR: tuple[str, str] = ("B11", "B6")
#: Registered Holm family size of the fingerprint superiority legs (§9.3:
#: 3 superiority predictions). Missing legs are padded at p=1.0 (conservative)
#: and force the intersection-union p to 1.0 — an incomplete fingerprint can
#: never pass its chain step.
FINGERPRINT_HOLM_M: int = sum(
    1 for _, corr, _, _ in FINGERPRINT_SUB_HYPOTHESES if corr == "holm"
)

#: §9.5: the three pre-registered NONE fingerprints tested by conditional TOST.
#: 'distribute' is a topology-slot leg (no policy-axis pairing exists for it in
#: a single-topology run) — it is DECLARED and reported as a labeled skip until
#: DIST-topology data exists.
_TOST_POLICY_AXIS_LEGS: frozenset[str] = frozenset({"recompute", "offload"})
#: Per-query S2 policy-event mask column (conditional population, §9.5).
POLICY_EVENT_COLUMN = "policy_event"

#: §9.4: per-query pairing is sub-pressure (F1) only; F2/F3 are batch-means.
PRESSURE_FAMILIES: frozenset[str] = frozenset({"F2", "F3"})

#: One tuple-slot-differs discipline: a cell/reference pair must agree on every
#: axis the baseline ids do not determine (baseline fixes arm + retriever).
_PAIR_MATCH_AXES: tuple[str, ...] = ("engine", "model", "topology", "policy")

#: The FULL organize_results.INDEX_COLUMNS contract (RESULTS_LAYOUT.md, 20
#: columns): the window-pair selector and the §9.5 equivalence matcher group
#: on budget_r/rate_frac, and cell_json/artifacts carry the provenance handoff
#: — a truncated index must refuse, not silently degrade the pairing.
_INDEX_REQUIRED_COLUMNS: tuple[str, ...] = (
    "run_id", "campaign", "session", "model", "engine", "arm", "baseline",
    "retriever", "policy", "topology", "family", "dataset", "budget_r",
    "rate_frac", "window", "window_key", "row_key", "window_dir", "cell_json",
    "artifacts",
)

#: Metric direction registry — FAIL CLOSED on unknown metrics rather than
#: guessing a direction (a flipped W/L/T is worse than a refusal).
HIGHER_IS_BETTER: dict[str, bool] = {
    # serving (lower is better)
    "ttft_ms": False,
    "latency_ms": False,
    "tpot_ms": False,
    "e2e_ms": False,
    # quality (higher is better)
    "f1_score": True,
    "exact_match": True,
    "f1_answerable": True,
    "exact_match_answerable": True,
    "grounding_score": True,
    "faithfulness": True,
    "context_relevance": True,
    "completeness_bertscore": True,
    "completeness_rouge_l": True,
    "predicate": True,
    "goodput_frac": True,
    "yield_frac": True,
}

DEFAULT_METRICS: tuple[str, ...] = ("ttft_ms",)

CONTRAST_BY_ID: dict[int, Contrast] = {c.id: c for c in CONTRASTS}

#: The two per-query artifacts the v2 loader joins (RESULTS_LAYOUT §1). The
#: dataset-scoped exemption (sharegpt has no qa_evidence.jsonl) is handled by
#: presence, not by name.
_PER_QUERY_ARTIFACTS: tuple[str, ...] = ("requests.jsonl", "qa_evidence.jsonl")


class AnalysisError(RuntimeError):
    """Any refusal or data violation in the driver (fail loud, message first)."""


class OneLookError(AnalysisError):
    """§9.11 one-look policy refusal (lock present / flags missing)."""


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------


def load_index(run_dir: Path) -> pd.DataFrame:
    """Load ``index/cells_index.csv`` — refuse (with the fix) when absent."""
    index_path = run_dir / "index" / "cells_index.csv"
    if not index_path.is_file():
        raise AnalysisError(
            f"no index at {index_path} — this driver consumes an ORGANIZED run. "
            f"Run: python scripts/4_analysis/organize_results.py {run_dir}  "
            "(it validates the tree and emits index/cells_index.csv), then re-run "
            "this analysis. The organizer is not auto-run on purpose: its "
            "fail-loud layout report deserves its own look."
        )
    index = pd.read_csv(index_path)
    missing = [c for c in _INDEX_REQUIRED_COLUMNS if c not in index.columns]
    if missing:
        raise AnalysisError(
            f"{index_path} is missing required columns {missing} — re-run "
            "organize_results.py (the index schema moved under you)"
        )
    if index.empty:
        raise AnalysisError(f"{index_path} has no rows — nothing to analyze")
    return index


# ---------------------------------------------------------------------------
# Contrast resolution + cell selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkippedContrast:
    """A requested-but-not-computed contrast, always carried into the outputs."""

    contrast_id: int
    name: str
    label: str
    reason: str


@dataclass(frozen=True)
class ResolvedPair:
    """One concrete (cell, reference) row-key pair for a per-query contrast."""

    contrast: Contrast
    cell_row_key: str
    reference_row_key: str
    datasets: tuple[str, ...]


def classify_contrast(contrast: Contrast) -> str | None:
    """Return a skip reason when this driver cannot compute the contrast.

    Computable paths: (a) per-query baseline-pair contrasts on sub-pressure F1
    rows (paired Wilcoxon / McNemar); (b) window-unit baseline-pair contrasts
    on loaded rows via window-level batch means
    (``tests_by_unit.batch_means_contrast``, §9.4). Selector contrasts (no
    single baseline pair — e.g. the gated #6, the engine-slot #10) remain
    labeled skips. The estimand primaries never reach this classifier: #13,
    #12 and #14 have executors (G4).
    """
    if contrast.baseline_a is None or contrast.baseline_b is None:
        return (
            f"selector (slot={contrast.slot!r}) is not a single baseline pair; "
            f"driver support is {NOT_IMPLEMENTED_LABEL}"
        )
    if contrast.unit == "per_query" and contrast.family not in PRESSURE_FAMILIES:
        return None
    if contrast.unit == "window":
        return None  # batch-means path (§9.4)
    return (
        f"unit={contrast.unit!r}, family={contrast.family}: no registered "
        f"driver path; support is {NOT_IMPLEMENTED_LABEL}"
    )


def resolve_contrasts(
    contrast_ids: Sequence[int],
) -> tuple[list[Contrast], list[SkippedContrast]]:
    """Map requested §7.8 ids to registry entries; unknown ids FAIL LOUD."""
    unknown = [i for i in contrast_ids if i not in CONTRAST_BY_ID]
    if unknown:
        raise AnalysisError(
            f"unknown §7.8 contrast id(s) {unknown}; the registry "
            f"(src.analysis.stats.families) holds ids {sorted(CONTRAST_BY_ID)}"
        )
    computable: list[Contrast] = []
    skipped: list[SkippedContrast] = []
    for cid in dict.fromkeys(contrast_ids):  # dedupe, keep order
        contrast = CONTRAST_BY_ID[cid]
        reason = classify_contrast(contrast)
        if reason is None:
            computable.append(contrast)
        else:
            skipped.append(
                SkippedContrast(
                    contrast_id=cid,
                    name=contrast.name,
                    label=NOT_IMPLEMENTED_LABEL,
                    reason=reason,
                )
            )
    return computable, skipped


def select_contrast_pairs(index: pd.DataFrame, contrast: Contrast) -> list[ResolvedPair]:
    """Concrete row-key pairs for one baseline-pair contrast (F1 rows only).

    Pairs cells with references agreeing on every non-baseline axis
    (engine/model/topology/policy) — the one-tuple-slot-differs discipline.
    """
    assert contrast.baseline_a is not None and contrast.baseline_b is not None
    f1 = index[index["family"] == "F1"]
    cells = f1[f1["baseline"] == contrast.baseline_a]
    refs = f1[f1["baseline"] == contrast.baseline_b]
    if cells.empty or refs.empty:
        present = sorted(f1["baseline"].dropna().unique())
        raise AnalysisError(
            f"contrast #{contrast.id} ({contrast.name}) needs F1 cells for both "
            f"{contrast.baseline_a} and {contrast.baseline_b}; this run's F1 "
            f"baselines: {present}"
        )
    pairs: list[ResolvedPair] = []
    cell_groups = cells.groupby(list(_PAIR_MATCH_AXES), dropna=False)
    ref_groups = {
        key: grp for key, grp in refs.groupby(list(_PAIR_MATCH_AXES), dropna=False)
    }
    for key, cell_grp in cell_groups:
        ref_grp = ref_groups.get(key)
        if ref_grp is None:
            continue
        cell_keys = sorted(cell_grp["row_key"].unique())
        ref_keys = sorted(ref_grp["row_key"].unique())
        if len(cell_keys) != 1 or len(ref_keys) != 1:
            raise AnalysisError(
                f"contrast #{contrast.id}: axes {dict(zip(_PAIR_MATCH_AXES, key))} "
                f"select multiple cells per side (cells={cell_keys}, "
                f"refs={ref_keys}) — the pair is ambiguous; refusing to guess"
            )
        datasets = tuple(
            sorted(set(cell_grp["dataset"].unique()) & set(ref_grp["dataset"].unique()))
        )
        if not datasets:
            raise AnalysisError(
                f"contrast #{contrast.id}: {cell_keys[0]!r} and {ref_keys[0]!r} "
                "share no dataset — nothing to pair"
            )
        pairs.append(
            ResolvedPair(
                contrast=contrast,
                cell_row_key=cell_keys[0],
                reference_row_key=ref_keys[0],
                datasets=datasets,
            )
        )
    if not pairs:
        raise AnalysisError(
            f"contrast #{contrast.id} ({contrast.name}): {contrast.baseline_a} and "
            f"{contrast.baseline_b} cells never share the same "
            f"{'/'.join(_PAIR_MATCH_AXES)} axes in this run — no legal pair"
        )
    return pairs


def pressure_row_skip(
    index: pd.DataFrame, consumed_row_keys: frozenset[str] = frozenset()
) -> dict[str, Any] | None:
    """The §9.4 guard block: F2/F3 rows NOT consumed by a computed batch-means
    contrast, listed so they are never silently absent (never per-query paired).
    """
    pressure = index[
        index["family"].isin(sorted(PRESSURE_FAMILIES))
        & ~index["row_key"].isin(consumed_row_keys)
    ]
    if pressure.empty:
        return None
    return {
        "label": "PRESSURE-ROWS-NOT-IN-A-COMPUTED-CONTRAST",
        "reason": (
            "family F2/F3 rows are PRESSURE cells: per-query pairing under load "
            "is prohibited (§9.4) — they take window-level batch means "
            "(tests_by_unit.batch_means_contrast, wired for registered "
            "baseline-pair window contrasts). These rows matched no computed "
            "batch-means contrast in this invocation. Listed so they are never "
            "silently absent."
        ),
        "row_keys": sorted(pressure["row_key"].unique()),
        "n_windows": int(len(pressure)),
    }


@dataclass(frozen=True)
class WindowPair:
    """One concrete (cell, reference) pair for a window-unit (§9.4) contrast."""

    contrast: Contrast
    family: str
    cell_row_key: str
    reference_row_key: str
    datasets: tuple[str, ...]


#: Window pairs must additionally agree on the §6.1 pressure coordinates —
#: contrasting different (r, λ) grid points would confound the slot.
_WINDOW_PAIR_MATCH_AXES: tuple[str, ...] = (*_PAIR_MATCH_AXES, "budget_r", "rate_frac")

#: The §6.1 pressure coordinates are OPTIONAL on legal F2/F3 cells: CellSpec
#: declares budget_r/rate_frac as ``float | None`` and only forbids SETTING
#: them on F1 (src/analysis/cellspec.py) — so index rows may carry them as NaN.
_PRESSURE_COORD_AXES: tuple[str, ...] = ("budget_r", "rate_frac")
#: Group-KEYING sentinel for an unset pressure coordinate. Can never collide
#: with a set coordinate: those key as ``repr(float(value))``.
_UNSET_COORD_KEY = "<unset>"


def _coord_keyed(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``frame`` with budget_r/rate_frac normalized for GROUP KEYING.

    ``groupby(dropna=False)`` KEEPS NaN group keys, but a tuple key holding
    NaN can never be FOUND in a dict built from another groupby: NaN != NaN,
    and dict lookup falls back to equality after the identity check, so
    probing one side's group dict with the other side's keys silently never
    matches. Legal F2/F3 cells with unset coordinates (see
    ``_PRESSURE_COORD_AXES``) index as NaN and MUST still pair, so the coords
    are string-keyed: unset-vs-unset matches, unset-vs-0.5 stays distinct —
    absence is matched as absence, never coerced to a number. KEYING ONLY:
    callers group the returned copy but read only columns this rewrite does
    not touch (row_key/dataset); ``dropna=False`` semantics elsewhere are
    unchanged.
    """
    keyed = frame.copy()
    for axis in _PRESSURE_COORD_AXES:
        normalized: list[str] = []
        for value in keyed[axis]:
            if value is None or (isinstance(value, float) and math.isnan(value)) or value == "":
                # Absent per cellspec.CellSpec.from_flat_dict (None/""/NaN).
                normalized.append(_UNSET_COORD_KEY)
                continue
            try:
                normalized.append(repr(float(value)))
            except (TypeError, ValueError):
                raise AnalysisError(
                    f"index column {axis!r} holds non-numeric value {value!r} "
                    "— pressure coordinates must be floats or absent "
                    "(cellspec.CellSpec)"
                ) from None
        keyed[axis] = normalized
    return keyed


def select_window_pairs(
    index: pd.DataFrame, contrast: Contrast
) -> tuple[list[WindowPair], list[str]]:
    """Concrete pairs for a window-unit baseline-pair contrast (all legs).

    Unlike the per-query selector, an absent leg is NOT an error: pressure
    cells are run-scoped, so a run without them yields a labeled skip reason
    instead. Ambiguous pairs (multiple cells per axes-slot) still fail loud.
    """
    assert contrast.baseline_a is not None and contrast.baseline_b is not None
    legs: list[tuple[str, str, str]] = [
        (contrast.baseline_a, contrast.baseline_b, contrast.family)
    ]
    legs.extend(
        (leg.baseline_a, leg.baseline_b, leg.family) for leg in contrast.extra_legs
    )
    pairs: list[WindowPair] = []
    skip_reasons: list[str] = []
    for baseline_a, baseline_b, family in legs:
        fam_rows = index[index["family"] == family]
        cells = fam_rows[fam_rows["baseline"] == baseline_a]
        refs = fam_rows[fam_rows["baseline"] == baseline_b]
        if cells.empty or refs.empty:
            skip_reasons.append(
                f"contrast #{contrast.id} leg {baseline_a}-vs-{baseline_b} "
                f"(family {family}): no loaded-window cells for "
                f"{'both sides' if cells.empty and refs.empty else (baseline_a if cells.empty else baseline_b)} "
                "in this run"
            )
            continue
        # Group the coord-keyed copies: NaN pressure coords would make the
        # cross-groupby dict probe below never match (see _coord_keyed).
        ref_groups = {
            key: grp
            for key, grp in _coord_keyed(refs).groupby(
                list(_WINDOW_PAIR_MATCH_AXES), dropna=False
            )
        }
        matched = False
        for key, cell_grp in _coord_keyed(cells).groupby(
            list(_WINDOW_PAIR_MATCH_AXES), dropna=False
        ):
            ref_grp = ref_groups.get(key)
            if ref_grp is None:
                continue
            cell_keys = sorted(cell_grp["row_key"].unique())
            ref_keys = sorted(ref_grp["row_key"].unique())
            if len(cell_keys) != 1 or len(ref_keys) != 1:
                raise AnalysisError(
                    f"contrast #{contrast.id} leg {baseline_a}-vs-{baseline_b}: "
                    f"axes {dict(zip(_WINDOW_PAIR_MATCH_AXES, key))} select "
                    f"multiple cells per side (cells={cell_keys}, "
                    f"refs={ref_keys}) — the pair is ambiguous; refusing to guess"
                )
            datasets = tuple(
                sorted(
                    set(cell_grp["dataset"].unique())
                    & set(ref_grp["dataset"].unique())
                )
            )
            if not datasets:
                continue
            matched = True
            pairs.append(
                WindowPair(
                    contrast=contrast,
                    family=family,
                    cell_row_key=cell_keys[0],
                    reference_row_key=ref_keys[0],
                    datasets=datasets,
                )
            )
        if not matched:
            skip_reasons.append(
                f"contrast #{contrast.id} leg {baseline_a}-vs-{baseline_b} "
                f"(family {family}): {baseline_a} and {baseline_b} cells never "
                f"share the same {'/'.join(_WINDOW_PAIR_MATCH_AXES)} axes+coords "
                "and a dataset in this run — no legal pair"
            )
    return pairs, skip_reasons


# ---------------------------------------------------------------------------
# v2-layout per-query loader (this file's own; _results_loader.py is pilot-only)
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise AnalysisError(
                    f"{path}:{lineno}: record must be a JSON object, "
                    f"got {type(obj).__name__}"
                )
            records.append(obj)
    return records


def load_per_query(
    run_dir: Path,
    index: pd.DataFrame,
    row_keys: Iterable[str],
    *,
    predicate_root: Path | None = None,
) -> pd.DataFrame:
    """Long per-query table for the given cells across ALL their index windows.

    Joins ``requests.jsonl`` + ``qa_evidence.jsonl`` per window on
    ``example_id`` (numeric fields; duplicate example_id lines within a
    window are multiple trials — disambiguated by ``record_index`` — and are
    averaged, matching the figure-pipeline convention). With a
    ``predicate_root`` (task #119: predicate/<scoring_run_id>/, resolved by
    ``resolve_predicate_root``), each window's ``predicate.jsonl`` joins as a
    third per-query artifact from the mirrored path — its JSON-null
    predicates stay ABSENT for that example (None-propagation: counted
    downstream as dropped pairs, never fabricated). Output columns:
    row_key, dataset, window_key, example_id, plus every numeric field seen.
    A window with zero joinable records fails loud.

    G11 (2026-08-16): JSON booleans are COERCED to 1.0/0.0 instead of being
    silently dropped (the #119 predicate producer emits true/false);
    the coerced field names are surfaced on the returned frame
    as ``frame.attrs['bool_coerced_fields']``. Two records in ONE artifact
    file sharing the same ``(example_id, record_index)`` key (including both
    missing ``record_index``) are a duplicate-row hazard (H3: replay rows
    are duplicated BY DESIGN and must carry the disambiguating key) — the
    loader REFUSES naming the window.
    """
    wanted = set(row_keys)
    selection = index[index["row_key"].isin(wanted)]
    if selection.empty:
        raise AnalysisError(f"no index rows for row keys {sorted(wanted)}")
    rows: list[dict[str, Any]] = []
    bool_coerced: set[str] = set()
    for rec in selection.itertuples(index=False):
        window_dir = run_dir / str(rec.window_dir)
        merged: dict[str, dict[str, list[float]]] = {}
        n_unjoined = 0
        artifact_paths = [window_dir / name for name in _PER_QUERY_ARTIFACTS]
        if predicate_root is not None:
            artifact_paths.append(
                predicate_root / str(rec.window_dir) / PREDICATE_ROWS_NAME
            )
        for path in artifact_paths:
            if not path.is_file():
                continue  # qa_evidence is dataset-exempt for load donors (§1)
            seen_keys: set[tuple[str, Any]] = set()
            for obj in _read_jsonl(path):
                example_id = obj.get("example_id")
                if not isinstance(example_id, str) or not example_id:
                    n_unjoined += 1
                    continue
                dup_key = (example_id, obj.get("record_index"))
                if dup_key in seen_keys:
                    raise AnalysisError(
                        f"{path}: duplicate (example_id, record_index) = "
                        f"{dup_key!r} — replayed rows must carry a distinct "
                        "record_index (H3/#127); refusing to average "
                        "indistinguishable rows (G11)"
                    )
                seen_keys.add(dup_key)
                bucket = merged.setdefault(example_id, {})
                for field_name, value in obj.items():
                    if isinstance(value, bool):
                        # true/false -> 1.0/0.0, noted — never dropped (G11).
                        bool_coerced.add(field_name)
                        bucket.setdefault(field_name, []).append(float(value))
                        continue
                    if not isinstance(value, (int, float)):
                        continue
                    bucket.setdefault(field_name, []).append(float(value))
        if not merged:
            raise AnalysisError(
                f"{window_dir}: no per-query records with an example_id in "
                f"{'/'.join(_PER_QUERY_ARTIFACTS)} — cannot pair anything "
                f"({n_unjoined} record(s) lacked example_id)"
            )
        for example_id, fields in merged.items():
            rows.append(
                {
                    "row_key": rec.row_key,
                    "dataset": rec.dataset,
                    "window_key": rec.window_key,
                    "example_id": example_id,
                    **{k: float(np.mean(v)) for k, v in fields.items()},
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs["bool_coerced_fields"] = sorted(bool_coerced)
    return frame


def _verify_predicate_tree(run_dir: Path, pred_dir: Path) -> dict[str, Any]:
    """Fail-loud schema/seal guard on one predicate/<scoring_run_id>/ tree.

    Task #119 wiring: the joined table is CONSUMED only when its manifest
    carries the required keys, it names THIS run's raw seal, and it verifies
    against its OWN ledger — a predicate table derived from a different tree
    (or tampered after the build) must never feed Y.
    """
    manifest_path = pred_dir / PREDICATE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise AnalysisError(
            f"{manifest_path} missing — not a predicate table "
            "(scripts/4_analysis/build_predicate_table.py writes it)"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"{manifest_path}: invalid JSON: {exc}") from exc
    missing = [k for k in _PREDICATE_MANIFEST_REQUIRED_KEYS if k not in manifest]
    if missing:
        raise AnalysisError(
            f"{manifest_path}: missing required key(s) {missing} — refuse "
            "to consume a predicate table without its provenance"
        )
    raw_ledger = run_dir / LEDGER_NAME
    if raw_ledger.is_file():
        try:
            read_ledger(raw_ledger)
            raw_sha = json.loads(raw_ledger.read_text(encoding="utf-8"))[
                "entries_sha256"
            ]
        except (LedgerError, json.JSONDecodeError, KeyError) as exc:
            raise AnalysisError(
                f"raw ledger unusable for predicate-table verification: {exc}"
            ) from exc
        if manifest["raw_run_ledger_entries_sha256"] != raw_sha:
            raise AnalysisError(
                f"predicate table {pred_dir.name!r} was built against a "
                "DIFFERENT raw seal "
                f"({manifest['raw_run_ledger_entries_sha256']!r} != "
                f"{raw_sha!r}) — its rows do not describe this tree"
            )
    own_ledger = pred_dir / LEDGER_NAME
    if not own_ledger.is_file():
        raise AnalysisError(
            f"{own_ledger} missing — a predicate table is sealed at build "
            "time (build_predicate_table.py); an unsealed table proves nothing"
        )
    try:
        mismatches = verify_ledger(own_ledger, pred_dir)
    except LedgerError as exc:
        raise AnalysisError(f"{own_ledger}: {exc}") from exc
    if mismatches:
        raise AnalysisError(
            f"predicate table {pred_dir.name!r} fails its own seal: "
            + "; ".join(mismatches)
        )
    return manifest


def resolve_predicate_root(
    run_dir: Path, predicate_run_id: str | None
) -> tuple[Path | None, dict[str, Any] | None]:
    """Locate + verify the §8.5 predicate table this analysis joins (#119).

    Explicit ``predicate_run_id`` names predicate/<id>/ and refuses when
    absent. Without it: zero tables -> (None, None) — the registered
    predicate legs then surface through the existing missing-column
    refusals; exactly one -> auto-selected; more than one -> refusal naming
    --predicate-run-id (never guess which verdicts feed Y).
    """
    root = run_dir / PREDICATE_DIRNAME
    if predicate_run_id is not None:
        pred_dir = root / predicate_run_id
        if not pred_dir.is_dir():
            raise AnalysisError(
                f"--predicate-run-id {predicate_run_id!r}: no predicate table "
                f"at {pred_dir} — {_PREDICATE_FIX_HINT}"
            )
        return pred_dir, _verify_predicate_tree(run_dir, pred_dir)
    if not root.is_dir():
        return None, None
    candidates = sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if not candidates:
        return None, None
    if len(candidates) > 1:
        raise AnalysisError(
            f"multiple predicate tables under {root} "
            f"({[p.name for p in candidates]}) — name the one this analysis "
            "consumes via --predicate-run-id (refusing to guess which "
            "verdicts feed Y)"
        )
    return candidates[0], _verify_predicate_tree(run_dir, candidates[0])


# ---------------------------------------------------------------------------
# Statistics per contrast
# ---------------------------------------------------------------------------


def _metric_direction(metric: str) -> bool:
    if metric in HIGHER_IS_BETTER:
        return HIGHER_IS_BETTER[metric]
    if metric in ESTIMAND_HIGHER_IS_BETTER:
        return ESTIMAND_HIGHER_IS_BETTER[metric]
    raise AnalysisError(
        f"metric {metric!r} has no registered direction — add it to "
        f"HIGHER_IS_BETTER (caller roster) or ESTIMAND_HIGHER_IS_BETTER "
        f"(estimand variables) in run_campaign_analysis.py (known: "
        f"{sorted(HIGHER_IS_BETTER) + sorted(ESTIMAND_HIGHER_IS_BETTER)}). "
        "Guessing a direction flips W/L/T."
    )


def _derive_alternative(
    contrast_id: int, sidedness: str, metric: str, higher_is_better: bool
) -> str:
    """The scipy ``alternative`` a registered row EXECUTES (decision a).

    Two-sided rows pass through. One-sided rows resolve the registered
    contrast direction (``REGISTERED_CELL_DIRECTION``) against the metric's
    direction: with a = cell and b = reference on every §9.4 primitive,
    "cell better" means H1: a > b on a higher-is-better metric and H1: a < b
    on a lower-is-better one — i.e. ``alternative = "greater" iff
    claim_cell_better == higher_is_better``. The mapping holds verbatim for
    the McNemar binary route (``"greater"`` = arm a succeeds more, which is
    "better" exactly when higher_is_better). A one-sided contrast with no
    registered direction FAILS LOUD: an undeclared tail is an unregistered
    test.
    """
    if sidedness == "two-sided":
        return "two-sided"
    if sidedness != "one-sided":
        raise AnalysisError(
            f"contrast #{contrast_id}: sidedness {sidedness!r} has no direct "
            "test execution (TOST rows run through the §9.5 equivalence "
            "machinery, never this router)"
        )
    direction = REGISTERED_CELL_DIRECTION.get(contrast_id)
    if direction is None:
        raise AnalysisError(
            f"contrast #{contrast_id} is registered one-sided but has no "
            "entry in REGISTERED_CELL_DIRECTION — refusing to guess a tail "
            "(decision a, 2026-08-16)"
        )
    if direction == "serving-better-quality-worse":
        # #15 "latency saved vs truth lost": the cell is predicted BETTER on
        # serving metrics (lower is better) and WORSE on quality metrics.
        claim_cell_better = not higher_is_better
    else:
        claim_cell_better = direction == "cell-better"
    return "greater" if claim_cell_better == higher_is_better else "less"


# ---------------------------------------------------------------------------
# G1: confirmatory registration binding (SHA, worktree, prereg, alpha, margins)
# ---------------------------------------------------------------------------


def _git_head_state(repo_dir: Path = _REPO_ROOT) -> tuple[str | None, bool | None]:
    """(HEAD sha, dirty flag) of the EXECUTING code — subprocess git, like
    campaign_layout (src.observability.provenance owns the fallbacks)."""
    return _prov_git_sha(str(repo_dir)), _prov_git_dirty(str(repo_dir))


def _sha_matches(short_or_full: str, full: str) -> bool:
    return full == short_or_full or full.startswith(short_or_full)


def check_registration_binding(
    registered_sha: str, *, alpha: float, metrics: Sequence[str],
    metrics_overridden: bool,
) -> dict[str, Any]:
    """G1 (2026-08-16): bind the confirmatory look to the frozen registration.

    Refuses (AnalysisError, BEFORE the §9.11 lock) unless: the SHA is 7-64
    lowercase hex; it names the EXECUTING code's git HEAD; the worktree is
    clean; ``PREREG_PATH`` exists and embeds a matching Machinery SHA; alpha
    equals ``REGISTERED_ALPHA``; and any ``--metrics`` override equals the
    registered pair (checked by the caller before this runs — recorded here).
    Returns the summary recorded into stats['preconditions']['registration'].
    """
    if not _REGISTERED_SHA_RE.fullmatch(registered_sha):
        raise AnalysisError(
            f"--registered-sha {registered_sha!r} is not a 7-64 char "
            "lowercase hex git SHA — the confirmatory look executes a frozen "
            "registration, not a label (G1)"
        )
    head_sha, dirty = _git_head_state()
    if head_sha is None or dirty is None:
        raise AnalysisError(
            "cannot resolve the EXECUTING code's git HEAD/dirty state — the "
            "confirmatory look must prove it runs the registered machinery "
            "(G1); run from the registered checkout"
        )
    if not _sha_matches(registered_sha, head_sha):
        raise AnalysisError(
            f"--registered-sha {registered_sha!r} does not name the EXECUTING "
            f"code (git HEAD {head_sha!r}) — the registered look must run "
            "exactly the frozen machinery (G1); check out the registered SHA"
        )
    if dirty:
        raise AnalysisError(
            f"the worktree at HEAD {head_sha!r} is DIRTY — uncommitted edits "
            "mean the executing code is NOT the registered code; commit or "
            "stash before the confirmatory look (G1)"
        )
    if not PREREG_PATH.is_file():
        raise AnalysisError(
            f"no frozen registration document at {PREREG_PATH} — the "
            "confirmatory look executes a registration that does not exist "
            "yet; the §9.13 freeze (task #112) must land PRE_REGISTRATION.md "
            "before any confirmatory analysis (G1)"
        )
    prereg_text = PREREG_PATH.read_text(encoding="utf-8")
    match = _PREREG_EMBEDDED_SHA_RE.search(prereg_text)
    if match is None:
        raise AnalysisError(
            f"{PREREG_PATH} carries no embedded 'Machinery SHA: `<sha>`' "
            "line — a registration without its machinery SHA cannot be "
            "executed against (G1); re-assemble via prereg.py"
        )
    embedded = match.group(1)
    if not (_sha_matches(registered_sha, embedded) or _sha_matches(embedded, registered_sha)):
        raise AnalysisError(
            f"--registered-sha {registered_sha!r} does not match the SHA "
            f"embedded in {PREREG_PATH.name} ({embedded!r}) — the look must "
            "execute the registration it names (G1)"
        )
    if alpha != REGISTERED_ALPHA:
        raise AnalysisError(
            f"--alpha {alpha!r} differs from the registered alpha "
            f"{REGISTERED_ALPHA!r} — alpha is registration content, not a "
            "CLI knob (G1)"
        )
    return {
        "registered_sha": registered_sha,
        "executing_git_sha": head_sha,
        "executing_git_dirty": dirty,
        "prereg_path": str(PREREG_PATH),
        "prereg_embedded_sha": embedded,
        "alpha": alpha,
        "metrics": list(metrics),
        "metrics_source": (
            "CLI override (== registered pair)"
            if metrics_overridden
            else "families.DEFAULT_METRICS (registered §9.1 co-primary pair)"
        ),
        "verdict": "BOUND",
    }


def resolve_registered_margin(
    tost_margin: float | None, equivalence_metric: str | None
) -> tuple[float | None, dict[str, Any]]:
    """G1d/G9: confirmatory §9.5 margins come from the registered artifact.

    Returns (margin_to_use, record). With a registered-margins artifact
    present and an equivalence metric named, the registered margin is
    CONSUMED (a CLI ``--tost-margin`` must match it or refuse). Without the
    artifact, a CLI margin REFUSES — margins are registration content and
    cannot be minted at the one look; no margin at all leaves the §9.5 legs
    as labeled skips (the honest pre-freeze state).
    """
    record: dict[str, Any] = {
        "artifact": str(REGISTERED_MARGINS_PATH),
        "artifact_present": REGISTERED_MARGINS_PATH.is_file(),
        "equivalence_metric": equivalence_metric,
        "cli_margin": tost_margin,
        "margin_used": None,
    }
    if equivalence_metric is None:
        if tost_margin is not None:
            raise AnalysisError(
                "--tost-margin without --equivalence-metric cannot be "
                "cross-checked against the registered margins (G1d); name "
                "the metric the registered margin belongs to"
            )
        return None, record
    if not REGISTERED_MARGINS_PATH.is_file():
        if tost_margin is not None:
            raise AnalysisError(
                f"confirmatory --tost-margin={tost_margin} refused: no "
                f"registered-margins artifact at {REGISTERED_MARGINS_PATH} — "
                "§9.5 margins are registration content (G1d/G9); the §9.13 "
                "freeze (task #112) writes registered_margins.json beside "
                "PRE_REGISTRATION.md"
            )
        return None, record
    try:
        margins = json.loads(REGISTERED_MARGINS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisError(
            f"registered margins artifact {REGISTERED_MARGINS_PATH} is not "
            f"valid JSON: {exc}"
        ) from exc
    if not isinstance(margins, dict):
        raise AnalysisError(
            f"registered margins artifact {REGISTERED_MARGINS_PATH} must be "
            "a JSON object mapping metric -> margin"
        )
    if equivalence_metric not in margins:
        if tost_margin is not None:
            raise AnalysisError(
                f"metric {equivalence_metric!r} has no registered §9.5 "
                f"margin in {REGISTERED_MARGINS_PATH.name} (registered: "
                f"{sorted(margins)}) — an unregistered margin cannot run at "
                "the confirmatory look (G1d)"
            )
        return None, record
    raw_margin = margins[equivalence_metric]
    # K-COV2 (task #140): the registered value must BE a margin — a typed
    # refusal, never a float() crash (string) or a silent NaN/<=0 consumption.
    if isinstance(raw_margin, bool) or not isinstance(raw_margin, (int, float)):
        raise AnalysisError(
            f"registered §9.5 margin for {equivalence_metric!r} in "
            f"{REGISTERED_MARGINS_PATH.name} is not a number "
            f"({raw_margin!r}) — a TOST margin is a finite positive number "
            "(G1d); fix the registered artifact at the freeze, not at the look"
        )
    registered = float(raw_margin)
    if not math.isfinite(registered) or registered <= 0:
        raise AnalysisError(
            f"registered §9.5 margin for {equivalence_metric!r} in "
            f"{REGISTERED_MARGINS_PATH.name} is {registered!r} — a TOST "
            "margin must be finite and > 0 (G1d); fix the registered "
            "artifact at the freeze, not at the look"
        )
    if tost_margin is not None and tost_margin != registered:
        raise AnalysisError(
            f"--tost-margin={tost_margin} contradicts the REGISTERED margin "
            f"{registered} for {equivalence_metric!r} "
            f"({REGISTERED_MARGINS_PATH.name}) — the registered margin is "
            "what runs (G1d/G9)"
        )
    record["margin_used"] = registered
    record["margin_source"] = REGISTERED_MARGINS_PATH.name
    return registered, record


#: §9.1: primaries are NEVER pooled/corrected across datasets (full alpha per
#: dataset; the registered pass/fail rule is gatekeeping.evaluate_chain's
#: intra-set rule, not a Holm correction — see stats['gatekeeping'] for the
#: executed chain). The Holm shown for every other tier here is
#: across-DATASET only — a labeled diagnostic; the registered §9.3
#: Holm-WITHIN-FAMILY correction (sibling contrasts sharing
#: group×metric×dataset×family×unit per families.compile_family_map) is
#: executed by gatekeeping.evaluate_chain and reported in
#: stats['gatekeeping'].
_PRIMARY_CORRECTION_LABEL = (
    "none (primary tier, full alpha per dataset, §9.1 co-primary set — "
    "cross-dataset pooling/correction PROHIBITED; the registered pass/fail "
    "rule is gatekeeping.evaluate_chain's intra-set rule, executed in "
    "stats['gatekeeping'])"
)
_DIAGNOSTIC_HOLM_LABEL = (
    "holm across datasets within contrast × metric (DIAGNOSTIC ONLY — the "
    "registered §9.3 Holm-within-family correction pools sibling contrasts "
    "sharing group×metric×dataset×family×unit via families.compile_family_map "
    "and is executed by gatekeeping.evaluate_chain in stats['gatekeeping'])"
)
_EXPLORATORY_CORRECTION_LABEL = (
    "bh-fdr in stats['exploratory'] (registered §9.3 exploratory tier — "
    "NON-CONFIRMATORY, ungated, no diagnostic Holm here)"
)


@dataclass(frozen=True)
class MapRow:
    """One §9.3 family-map row's registered execution attributes (G2/G10:
    the ROW — not ``Contrast`` — is the routing authority)."""

    tier: str
    family_id: str
    upstream: str
    sidedness: str
    unit: str
    correction: str


def _tier_correction(
    tier: str, per_dataset: Sequence[dict[str, Any]]
) -> str:
    """Attach the tier-conditional across-dataset correction in place."""
    computed = [row for row in per_dataset if "p_value" in row]
    if tier == "primary":
        for row in computed:
            row["p_holm_across_datasets"] = None
        return _PRIMARY_CORRECTION_LABEL
    if tier == "exploratory":
        # G2/G3: exploratory rows are BH-FDR territory (stats['exploratory']);
        # a diagnostic Holm here would dress them as near-confirmatory.
        for row in computed:
            row["p_holm_across_datasets"] = None
        return _EXPLORATORY_CORRECTION_LABEL
    adjusted = holm([row["p_value"] for row in computed]) if computed else []
    for row, p_adj in zip(computed, adjusted):
        row["p_holm_across_datasets"] = float(p_adj)
    return _DIAGNOSTIC_HOLM_LABEL


def compute_pair_stats(
    per_query: pd.DataFrame,
    pair: ResolvedPair,
    metric: str,
    *,
    map_row: MapRow | None = None,
) -> dict[str, Any]:
    """Per-dataset test + W/L/T for one pair.

    ``map_row`` is the §9.3 family-map row for (contrast, metric, dataset,
    family) — when present it is the REGISTERED routing authority (G2): the
    row's tier drives correction routing, the row's unit drives the
    McNemar-vs-Wilcoxon route (G8: by UNIT, not by the metric's name), and
    the row's sidedness is EXECUTED (decision a) via the derived scipy
    alternative. Without a map (non-charter data, design-input), the
    ``Contrast`` registry attributes apply. Tie handling is the registered
    ``zero_method='pratt'`` (decision b) with the effective n surfaced.
    """
    higher_is_better = _metric_direction(metric)
    if metric not in per_query.columns:
        raise AnalysisError(
            f"metric {metric!r} appears in no requests.jsonl/qa_evidence.jsonl "
            f"record for contrast #{pair.contrast.id} "
            f"({pair.cell_row_key} vs {pair.reference_row_key})"
        )
    tier = map_row.tier if map_row is not None else pair.contrast.tier
    sidedness = map_row.sidedness if map_row is not None else pair.contrast.sidedness
    if map_row is not None:
        unit = map_row.unit
    else:
        unit = (
            "binary"
            if pair.contrast.unit == "per_query" and metric == PREDICATE_METRIC
            else pair.contrast.unit
        )
    alternative = _derive_alternative(
        pair.contrast.id, sidedness, metric, higher_is_better
    )
    per_dataset: list[dict[str, Any]] = []
    for dataset in pair.datasets:
        sub = per_query[
            (per_query["dataset"] == dataset)
            & (per_query["row_key"].isin([pair.cell_row_key, pair.reference_row_key]))
        ]
        wide = (
            sub.groupby(["example_id", "row_key"], observed=True)[metric]
            .mean()
            .unstack("row_key")
        )
        for key in (pair.cell_row_key, pair.reference_row_key):
            if key not in wide.columns:
                raise AnalysisError(
                    f"contrast #{pair.contrast.id} × {dataset}: no {metric!r} "
                    f"values for {key!r}"
                )
        n_candidates = len(wide)
        wide = wide.dropna(subset=[pair.cell_row_key, pair.reference_row_key])
        n_dropped = n_candidates - len(wide)
        if wide.empty:
            raise AnalysisError(
                f"contrast #{pair.contrast.id} × {dataset}: no overlapping "
                f"example_id with finite {metric!r} on both sides"
            )
        a = wide[pair.cell_row_key].to_numpy(dtype=float)
        b = wide[pair.reference_row_key].to_numpy(dtype=float)
        triple = win_loss_tie(a, b, higher_is_better=higher_is_better)
        if unit == "binary":
            mcnemar = mcnemar_binary(a, b, alternative=alternative)
            row: dict[str, Any] = {
                "dataset": dataset,
                "n_pairs": mcnemar.n_pairs,
                "realized_n": mcnemar.n_pairs,
                "n_dropped_nan": int(n_dropped),
                "median_delta": float(np.median(a - b)),
                "p_value": mcnemar.p_value,
                "n_11": mcnemar.n_11,
                "n_00": mcnemar.n_00,
                "n_10": mcnemar.n_10,
                "n_01": mcnemar.n_01,
                "n_discordant": mcnemar.n_discordant,
                "proportion_diff": mcnemar.proportion_diff,
                "wins": triple.wins,
                "losses": triple.losses,
                "ties": triple.ties,
            }
        else:
            # Decision b (2026-08-16): the REGISTERED tie handling — Pratt
            # zeros with the pinned unconditional normal-approx execution —
            # passed explicitly by the driver, never a primitive default.
            wilcoxon = paired_wilcoxon(
                a, b, alternative=alternative, zero_method="pratt"
            )
            row = {
                "dataset": dataset,
                "n_pairs": wilcoxon.n_pairs,
                "realized_n": wilcoxon.n_pairs,
                "n_dropped_nan": int(n_dropped),
                "median_delta": float(np.median(a - b)),
                "statistic": wilcoxon.statistic,
                "p_value": wilcoxon.p_value,
                "cliffs_delta_paired": wilcoxon.cliffs_delta_paired,
                "n_nonzero": wilcoxon.n_nonzero,
                "zero_method": wilcoxon.zero_method,
                "wins": triple.wins,
                "losses": triple.losses,
                "ties": triple.ties,
            }
        per_dataset.append(row)

    correction_label = _tier_correction(tier, per_dataset)

    return {
        "contrast_id": pair.contrast.id,
        "name": pair.contrast.name,
        "tier": tier,
        "tier_source": (
            "family-map row (§9.3)" if map_row is not None
            else "contrast registry (no family map for this run)"
        ),
        "family": pair.contrast.family,
        "family_id": map_row.family_id if map_row is not None else None,
        "upstream": map_row.upstream if map_row is not None else None,
        "cell_baseline": pair.contrast.baseline_a,
        "reference_baseline": pair.contrast.baseline_b,
        "cell_row_key": pair.cell_row_key,
        "reference_row_key": pair.reference_row_key,
        "metric": metric,
        "unit": unit,
        "test": "mcnemar_binary" if unit == "binary" else "paired_wilcoxon",
        "higher_is_better": higher_is_better,
        "registered_sidedness": sidedness,
        "executed_alternative": alternative,
        "correction": correction_label,
        "per_dataset": per_dataset,
    }


def compute_window_pair_stats(
    per_query: pd.DataFrame,
    pair: WindowPair,
    metric: str,
    *,
    map_row: MapRow | None = None,
) -> dict[str, Any]:
    """Batch-means Welch contrast for one loaded-window pair (§9.4).

    Per dataset: per-window means of ``metric`` on each side feed
    ``tests_by_unit.batch_means_contrast`` (Welch t on window-level batch
    means — per-query pairing under load is PROHIBITED). Datasets with < 2
    windows on either side are labeled skips inside ``per_dataset``, never
    silently dropped. No W/L/T triple here: §8.13's triple is a per-query
    mandate and window means are unpaired across cells. ``map_row`` supplies
    the REGISTERED tier/sidedness (G2 / decision a) exactly as in
    ``compute_pair_stats``.
    """
    higher_is_better = _metric_direction(metric)
    if metric not in per_query.columns:
        raise AnalysisError(
            f"metric {metric!r} appears in no requests.jsonl/qa_evidence.jsonl "
            f"record for window contrast #{pair.contrast.id} "
            f"({pair.cell_row_key} vs {pair.reference_row_key})"
        )
    tier = map_row.tier if map_row is not None else pair.contrast.tier
    sidedness = map_row.sidedness if map_row is not None else pair.contrast.sidedness
    alternative = _derive_alternative(
        pair.contrast.id, sidedness, metric, higher_is_better
    )
    per_dataset: list[dict[str, Any]] = []
    for dataset in pair.datasets:
        sub = per_query[per_query["dataset"] == dataset]
        means_a = (
            sub[sub["row_key"] == pair.cell_row_key]
            .groupby("window_key", observed=True)[metric]
            .mean()
            .dropna()
        )
        means_b = (
            sub[sub["row_key"] == pair.reference_row_key]
            .groupby("window_key", observed=True)[metric]
            .mean()
            .dropna()
        )
        if len(means_a) < 2 or len(means_b) < 2:
            per_dataset.append(
                {
                    "dataset": dataset,
                    "skipped": (
                        f"needs >= 2 windows per side for a Welch variance "
                        f"estimate; got cell={len(means_a)}, "
                        f"reference={len(means_b)}"
                    ),
                }
            )
            continue
        result = batch_means_contrast(
            means_a.to_numpy(dtype=float),
            means_b.to_numpy(dtype=float),
            alternative=alternative,  # registered sidedness (decision a)
        )
        per_dataset.append(
            {
                "dataset": dataset,
                "n_windows_cell": result.n_windows_a,
                "n_windows_reference": result.n_windows_b,
                "mean_cell": result.mean_a,
                "mean_reference": result.mean_b,
                "mean_diff": result.mean_diff,
                "statistic": result.statistic,
                "df": result.df,
                "p_value": result.p_value,
                "ci95_low": result.ci95_low,
                "ci95_high": result.ci95_high,
            }
        )

    correction_label = _tier_correction(tier, per_dataset)

    return {
        "contrast_id": pair.contrast.id,
        "name": pair.contrast.name,
        "tier": tier,
        "tier_source": (
            "family-map row (§9.3)" if map_row is not None
            else "contrast registry (no family map for this run)"
        ),
        "family": pair.family,
        "family_id": map_row.family_id if map_row is not None else None,
        "upstream": map_row.upstream if map_row is not None else None,
        "cell_baseline": pair.contrast.baseline_a,
        "reference_baseline": pair.contrast.baseline_b,
        "cell_row_key": pair.cell_row_key,
        "reference_row_key": pair.reference_row_key,
        "metric": metric,
        "unit": "window",
        "test": "batch_means_welch_t (tests_by_unit.batch_means_contrast, §9.4)",
        "higher_is_better": higher_is_better,
        "registered_sidedness": sidedness,
        "executed_alternative": alternative,
        "correction": correction_label,
        "per_dataset": per_dataset,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _baseline_of_key(index: pd.DataFrame, row_key: str) -> str:
    values = index.loc[index["row_key"] == row_key, "baseline"].dropna().unique()
    return str(values[0]) if len(values) else row_key


def _figure_rows_for_metric(
    contrast_entries: Sequence[Mapping[str, Any]], metric: str
) -> list[fp.ContrastStatRow]:
    """ContrastStatRow inputs built from the SAME dicts written to stats.json.

    Audit I1: the figures consume these rows and nothing else — the values a
    figure renders are bit-for-bit the values stats.json carries, so a
    published figure is structurally unable to contradict the registered
    statistics. Window-unit entries carry no §8.13 per-query triple and are
    excluded here (they have no forest/W-L-T rendering).
    """
    rows: list[fp.ContrastStatRow] = []
    for entry in contrast_entries:
        if entry["metric"] != metric or entry["unit"] == "window":
            continue
        label = f"{entry['cell_baseline']} vs {entry['reference_baseline']}"
        for row in entry["per_dataset"]:
            if "p_value" not in row:  # labeled skip rows carry no statistics
                continue
            rows.append(
                fp.ContrastStatRow(
                    cell_row_key=str(entry["cell_row_key"]),
                    reference_row_key=str(entry["reference_row_key"]),
                    dataset=str(row["dataset"]),
                    metric=metric,
                    higher_is_better=bool(entry["higher_is_better"]),
                    executed_alternative=str(entry["executed_alternative"]),
                    correction=str(entry["correction"]),
                    p_value=float(row["p_value"]),
                    n_pairs=int(row["n_pairs"]),
                    n_dropped_nan=int(row["n_dropped_nan"]),
                    median_delta=float(row["median_delta"]),
                    wins=int(row["wins"]),
                    losses=int(row["losses"]),
                    ties=int(row["ties"]),
                    p_corrected=row.get("p_holm_across_datasets"),
                    n_nonzero=row.get("n_nonzero"),
                    ci_low=row.get("ci_low"),
                    ci_high=row.get("ci_high"),
                    contrast_label=label,
                )
            )
    return rows


def _figure_record(
    path: Path, kind: str, metric: str, rows: Sequence[fp.ContrastStatRow]
) -> dict[str, Any]:
    """stats.json figure metadata: source + counted drops + consumed values.

    ``consumed`` records exactly the statistics the renderer was fed (the
    regression seam for the figures-agree-with-stats test); ``n_dropped_nan``
    totals are the I11 counted disclosure.
    """
    return {
        "file": path.name,
        "kind": kind,
        "metric": metric,
        "source": "stats.json contrast per_dataset rows "
        "(registered statistics — never recomputed, audit I1)",
        "n_dropped_nan_total": int(sum(r.n_dropped_nan for r in rows)),
        "consumed": [
            {
                "cell_row_key": r.cell_row_key,
                "reference_row_key": r.reference_row_key,
                "dataset": r.dataset,
                "p_value": r.p_value,
                "p_corrected": r.p_corrected,
                "median_delta": r.median_delta,
                "n_pairs": r.n_pairs,
                "n_dropped_nan": r.n_dropped_nan,
                "wins": r.wins,
                "losses": r.losses,
                "ties": r.ties,
            }
            for r in rows
        ],
    }


def render_figures(
    contrast_entries: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    per_query_columns: Sequence[str],
    out_dir: Path,
    stamp: str,
    index: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Publication figures FROM THE REGISTERED STATISTICS (audit I1/I2/I11).

    Consumes the computed contrast entries (the dicts serialized into
    stats.json) — never per-query data: wlt_<metric>.png renders the
    per-dataset §8.13 triples as small multiples (the §9.1 co-primary view),
    wlt_<metric>_pooled_supplementary.png is the disclosed pooled extra, and
    forest_<metric>[__vs_<ref>].png renders one forest per reference. Metrics
    with nothing renderable become COUNTED skip entries in the returned list
    instead of silent omissions (audit I11). Every returned record embeds the
    consumed statistics for the figures-agree-with-stats regression seam.
    """
    figures: list[dict[str, Any]] = []
    for metric in metrics:
        if metric not in per_query_columns:
            figures.append(
                {
                    "skipped_metric": metric,
                    "reason": "metric appears in no per-query artifact — no "
                    "figure rendered (counted disclosure, audit I11; the "
                    "registered-set consequences are in "
                    "skipped.confirmatory_exclusions)",
                }
            )
            continue
        rows = _figure_rows_for_metric(contrast_entries, metric)
        if not rows:
            figures.append(
                {
                    "skipped_metric": metric,
                    "reason": "no per-query contrast entry carries this "
                    "metric (window-unit contrasts have no §8.13 W/L/T "
                    "triple) — no figure rendered (counted disclosure, "
                    "audit I11)",
                }
            )
            continue

        wlt_path = out_dir / f"wlt_{metric}.png"
        fp.plot_win_loss_tie_registered(
            rows,
            wlt_path,
            title=f"[{stamp}] W/L/T per contrast × dataset — {metric} "
            "(registered per-dataset triples, stats.json)",
        )
        figures.append(_figure_record(wlt_path, "wlt_per_dataset", metric, rows))

        pooled_path = out_dir / f"wlt_{metric}_pooled_supplementary.png"
        fp.plot_win_loss_tie_registered(
            rows,
            pooled_path,
            title=f"[{stamp}] W/L/T per contrast — {metric}",
            pooled=True,
        )
        figures.append(
            _figure_record(pooled_path, "wlt_pooled_supplementary", metric, rows)
        )

        by_reference: dict[str, list[fp.ContrastStatRow]] = {}
        for row in rows:
            by_reference.setdefault(row.reference_row_key, []).append(row)
        multi_reference = len(by_reference) > 1
        for reference_key, ref_rows in by_reference.items():
            ref_label = _baseline_of_key(index, reference_key)
            if multi_reference:
                name = f"forest_{metric}__vs_{ref_label}.png"
            else:
                name = f"forest_{metric}.png"
            forest_path = out_dir / name
            fp.plot_forest_registered(
                ref_rows,
                forest_path,
                title=f"[{stamp}] paired Δ{metric} vs {ref_label} "
                "(registered statistics)",
            )
            figures.append(
                _figure_record(forest_path, "forest", metric, ref_rows)
            )
    return figures


# ---------------------------------------------------------------------------
# Confirmatory preconditions: §9.7 calibration artifact + §9.10 ledger seal
# ---------------------------------------------------------------------------


class CalibrationGateError(AnalysisError):
    """The §9.7 calibration precondition failed (absent/invalid/failing report)."""


def load_calibration_report(path: Path) -> CalibrationReport:
    """Load a §9.7 calibration-report JSON artifact (fail closed).

    Expected schema = the ``src.analysis.stats.calibration.CalibrationReport``
    dataclass tree serialized as JSON: top-level ``seed`` / ``n_observations``
    / ``aa`` (AAResult fields) / optional ``injections`` (InjectionResult
    fields each). This is the same report object ``prereg.py`` gates
    registration on — the confirmatory look reuses its schema and refusal
    pattern (§9.13: the registered SHA must be the calibrated one).
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CalibrationGateError(
            f"calibration report not found: {path} — confirmatory mode "
            "requires the §9.7 calibration artifact (A/A + effect-injection "
            "operating characteristics of the exact registered code)"
        ) from None
    except json.JSONDecodeError as exc:
        raise CalibrationGateError(
            f"calibration report is not valid JSON: {path} ({exc})"
        ) from exc
    if not isinstance(raw, dict):
        raise CalibrationGateError(
            f"calibration report root must be an object: {path}"
        )
    try:
        aa = AAResult(
            n_splits=int(raw["aa"]["n_splits"]),
            alpha=float(raw["aa"]["alpha"]),
            n_rejections=int(raw["aa"]["n_rejections"]),
            fp_rate=float(raw["aa"]["fp_rate"]),
            ci_low=float(raw["aa"]["ci_low"]),
            ci_high=float(raw["aa"]["ci_high"]),
        )
        injections = tuple(
            InjectionResult(
                effect_size=float(inj["effect_size"]),
                kind=inj["kind"],
                n_splits=int(inj["n_splits"]),
                alpha=float(inj["alpha"]),
                n_rejections=int(inj["n_rejections"]),
                power=float(inj["power"]),
                ci_low=float(inj["ci_low"]),
                ci_high=float(inj["ci_high"]),
                target_power=(
                    float(inj["target_power"])
                    if inj.get("target_power") is not None
                    else None
                ),
            )
            for inj in raw.get("injections", ())
        )
        report = CalibrationReport(
            seed=int(raw["seed"]),
            n_observations=int(raw["n_observations"]),
            aa=aa,
            injections=injections,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationGateError(
            f"calibration report {path} does not match the "
            "CalibrationReport schema (seed / n_observations / aa{n_splits, "
            "alpha, n_rejections, fp_rate, ci_low, ci_high} / injections[...]): "
            f"{exc!r}"
        ) from exc
    return report


def check_calibration(report: CalibrationReport, source: Path) -> dict[str, Any]:
    """§9.7 gate, mirroring ``prereg.assemble_preregistration``'s refusals.

    Returns the summary recorded into stats.json when the gate passes.
    """
    if not report.aa.approximates_nominal:
        raise CalibrationGateError(
            f"calibration A/A FAILED (FP rate CI [{report.aa.ci_low:.4f}, "
            f"{report.aa.ci_high:.4f}] excludes nominal α={report.aa.alpha:g}) "
            f"in {source} — the confirmatory look is BLOCKED until the "
            "machinery passes §9.7 (same rule prereg.py applies to "
            "registration)"
        )
    failed = [i for i in report.injections if i.meets_target is False]
    if failed:
        raise CalibrationGateError(
            f"calibration injection targets missed at effects "
            f"{[i.effect_size for i in failed]} in {source} — the "
            "confirmatory look is BLOCKED (§9.7/§9.13)"
        )
    return {
        "path": str(source),
        "seed": report.seed,
        "n_observations": report.n_observations,
        "aa_fp_rate": report.aa.fp_rate,
        "aa_approximates_nominal": report.aa.approximates_nominal,
        "n_injections": len(report.injections),
        "verdict": "PASS",
    }


def verify_run_ledger(run_dir: Path) -> dict[str, Any]:
    """§9.10 hard precondition of confirmatory mode: the raw tree must verify
    against its sealed content-hash ledger — analyzed data is provably the
    sealed data. Any mismatch (or a missing/tampered ledger) refuses."""
    ledger_path = run_dir / LEDGER_NAME
    try:
        entries = read_ledger(ledger_path)
        mismatches = verify_ledger(ledger_path, run_dir)
    except LedgerError as exc:
        raise AnalysisError(
            f"LEDGER PRECONDITION FAILED (§9.10): {exc} — confirmatory "
            "analysis runs only on sealed, verifiable data"
        ) from exc
    if mismatches:
        shown = "\n".join(f"  {line}" for line in mismatches[:20])
        raise AnalysisError(
            f"LEDGER PRECONDITION FAILED (§9.10): {len(mismatches)} artifact "
            f"mismatch(es) against {ledger_path}:\n{shown}\n"
            "— the raw tree is not the sealed tree; refusing the "
            "confirmatory look"
        )
    return {
        "path": LEDGER_NAME,
        "entries": len(entries),
        "mismatches": 0,
        "verified": True,
    }


# ---------------------------------------------------------------------------
# §9.8 blinding integration (sealed map consumer)
# ---------------------------------------------------------------------------


def sealed_map_path(run_dir: Path) -> Path:
    return run_dir / BLINDING_DIRNAME / SEALED_MAP_NAME


def load_blinding_state(run_dir: Path) -> dict[str, Any] | None:
    """Load + tamper-verify the run's sealed arm map, if one exists.

    Returns None when no sealed map exists. Uses the blinding module's own
    seal loader so tamper detection stays in ONE place; a tampered seal
    raises (never analyzed around).
    """
    path = sealed_map_path(run_dir)
    if not path.is_file():
        return None
    # Same-package seal verification (map_sha256 self-hash) — the loader is
    # module-internal by convention but IS the single verification authority.
    try:
        sealed = blinding_mod._load_sealed(path)  # noqa: SLF001
    except BlindingError as exc:
        raise AnalysisError(
            f"blinding refusal (§9.8): {exc} — a tampered/unreadable seal "
            "cannot certify a blinded analysis"
        ) from exc
    return dict(sealed)


def _blind_value(mapping: Mapping[str, str], value: str) -> str:
    """Real->blind code for one label; unknown labels fail loud (a label the
    seal never covered would leak through a blinded output)."""
    try:
        return mapping[value]
    except KeyError:
        raise AnalysisError(
            f"blinding: label {value!r} is not in the sealed map — the seal "
            "does not cover this run's labels; re-seal or remove the map"
        ) from None


def apply_blinding_to_entry(
    entry: dict[str, Any], mapping: Mapping[str, str], index: pd.DataFrame
) -> dict[str, Any]:
    """Mask arm-revealing labels in one contrast entry (§9.8 blinded output).

    The pipeline resolves contrasts on real labels internally (the registered
    selectors are baseline ids); the OUTPUT is what stays scrambled until the
    logged unblinding. Row keys and baseline ids both reveal the arm, so both
    are replaced by the blind code of the cell's arm.
    """
    masked = dict(entry)
    for key_field, baseline_field in (
        ("cell_row_key", "cell_baseline"),
        ("reference_row_key", "reference_baseline"),
    ):
        code = _blind_row_key(entry[key_field], mapping, index)
        masked[key_field] = code
        masked[baseline_field] = code
    return masked


def _blind_row_key(
    row_key: str, mapping: Mapping[str, str], index: pd.DataFrame
) -> str:
    """Blind code for one row key (its arm is the leading tuple slot)."""
    arms = index.loc[index["row_key"] == row_key, "arm"].dropna().unique()
    arm = str(arms[0]) if len(arms) else row_key.split("|", 1)[0]
    return f"BLINDED:{_blind_value(mapping, arm)}"


def apply_blinding_to_sections(
    stats: dict[str, Any], mapping: Mapping[str, str], index: pd.DataFrame
) -> None:
    """§9.8 masking for EVERY arm-revealing section (G12), in place.

    The contrast entries are masked by ``apply_blinding_to_entry``; this
    covers the sections the audit found leaking raw row keys: the §9.5
    equivalence results, the #13 fingerprint legs, and the pressure-skip
    block (row keys carry the arm as their leading tuple slot).
    """
    for result in stats.get("equivalence", {}).get("results", ()):
        for key_field in ("cell_row_key", "reference_row_key"):
            result[key_field] = _blind_row_key(result[key_field], mapping, index)
    for leg in stats.get("fingerprint", {}).get("legs", ()):
        for key_field in ("cell_row_key", "reference_row_key"):
            leg[key_field] = _blind_row_key(leg[key_field], mapping, index)
    # Skip-reason strings may embed raw row keys ("cell vs ref" phrasing) —
    # rewrite every known row key inside them to its blind code.
    key_codes = {
        str(k): _blind_row_key(str(k), mapping, index)
        for k in index["row_key"].dropna().unique()
    }
    for section_name in ("equivalence", "fingerprint"):
        for skip_entry in stats.get(section_name, {}).get("skipped", ()):
            reason = str(skip_entry.get("reason", ""))
            for raw, code in key_codes.items():
                reason = reason.replace(raw, code)
            skip_entry["reason"] = reason
    falsification = stats.get("falsification")
    if falsification:
        for result in falsification.get("results", ()):
            axes = result.get("cell_axes", {})
            if "arm" in axes:
                axes["arm"] = f"BLINDED:{_blind_value(mapping, axes['arm'])}"
    pressure_block = stats.get("skipped", {}).get("pressure_rows")
    if pressure_block:
        pressure_block["row_keys"] = sorted(
            {
                _blind_row_key(k, mapping, index)
                for k in pressure_block["row_keys"]
            }
        )


# ---------------------------------------------------------------------------
# §9.3 family map + Dmitrienko serial gatekeeping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyContext:
    """The compiled §9.3 registered test table, scoped to this run."""

    group: str
    datasets: tuple[str, ...]
    table: pd.DataFrame
    #: membership set: (contrast_id, metric, dataset) rows for this run's group.
    keys: frozenset[tuple[int, str, str]]
    #: (contrast_id, metric, dataset, family) -> the registered row (G2: the
    #: ROW is the routing authority; on ADR-0087 duplicate keys the
    #: exploratory row wins — demotion is the stronger registration fact).
    rows: Mapping[tuple[int, str, str, str], MapRow] = field(
        default_factory=dict
    )

    def map_row(
        self, contrast_id: int, metric: str, dataset: str, family: str
    ) -> MapRow | None:
        return self.rows.get((contrast_id, metric, dataset, family))

    def registered_family_sizes(self) -> dict[str, int]:
        """family_id -> registered Holm m (the map's holm-corrected rows)."""
        holm_rows = self.table[self.table["correction"] == "holm"]
        return {
            str(fid): int(n)
            for fid, n in holm_rows.groupby("family_id").size().items()
        }

    def upstream_by_family(self) -> dict[str, str]:
        """family_id -> registered upstream endpoint (gated families only)."""
        gated = self.table[self.table["upstream"] != UNGATED]
        return {
            str(fid): str(ups.iloc[0])
            for fid, ups in gated.groupby("family_id")["upstream"]
        }

    def registered_set_legs(self, contrast_id: int) -> tuple[str, ...]:
        """The registered co-primary legs of a chain endpoint (§9.1/G5) —
        the ``<dataset>|<metric>`` keys the driver keys outcomes under.
        #13's six sub-hypothesis rows collapse to one ``|fingerprint`` leg
        per dataset (the intersection-union endpoint contribution); other
        endpoints read the ADR-0087-preferred routing rows, so a demoted
        (exploratory) row never re-enters the registered set expectation."""
        if contrast_id == FINGERPRINT_CONTRAST_ID:
            rows = self.table[
                (self.table["contrast_id"] == contrast_id)
                & (self.table["tier"] == "primary")
            ]
            return tuple(
                sorted({f"{r.dataset}|fingerprint" for r in rows.itertuples(index=False)})
            )
        return tuple(
            sorted(
                {
                    f"{dataset}|{metric}"
                    for (cid, metric, dataset, _family), row in self.rows.items()
                    if cid == contrast_id and row.tier == "primary"
                }
            )
        )


def build_family_context(
    index: pd.DataFrame, metrics: Sequence[str], alpha: float
) -> FamilyContext | None:
    """Compile ``families.compile_family_map`` for this run's group/datasets.

    Returns None (a labeled absence recorded in stats.json) when the run has
    no charter family-map dataset at all — confirmatory mode refuses on that.
    """
    model = str(index["model"].iloc[0])
    group = GROUP_OF_MODEL.get(model)
    if group is None:
        raise AnalysisError(
            f"model {model!r} has no §7.6.1 campaign group — cannot compile "
            f"the §9.3 family map (roster: {sorted(GROUP_OF_MODEL)})"
        )
    datasets = tuple(sorted(set(index["dataset"].unique()) & KNOWN_DATASETS))
    if not datasets:
        return None
    try:
        table = compile_family_map(datasets, metrics=tuple(metrics), alpha=alpha)
    except FamilyMapError as exc:
        raise AnalysisError(f"family map compilation failed (§9.3): {exc}") from exc
    scoped = table[table["group"] == group]
    keys = frozenset(
        (int(r.contrast_id), str(r.metric), str(r.dataset))
        for r in scoped.itertuples(index=False)
    )
    rows: dict[tuple[int, str, str, str], MapRow] = {}
    for r in scoped.itertuples(index=False):
        if int(r.contrast_id) == FINGERPRINT_CONTRAST_ID:
            # #13's six sub-hypothesis rows share one (metric, dataset) key;
            # they route through compute_fingerprint / compute_equivalence
            # (G4a), never the baseline-pair pipeline.
            continue
        key = (int(r.contrast_id), str(r.metric), str(r.dataset), str(r.family))
        candidate = MapRow(
            tier=str(r.tier),
            family_id=str(r.family_id),
            upstream=str(r.upstream),
            sidedness=str(r.sidedness),
            unit=str(r.unit),
            correction=str(r.correction),
        )
        existing = rows.get(key)
        if existing is not None:
            # ADR-0087 duplicate (caller-passed 'faithfulness' collides with
            # the demotion row): the EXPLORATORY row wins — a demoted metric
            # may never re-enter the confirmatory chain via a CLI flag (G2).
            if existing.tier == "exploratory":
                continue
            if candidate.tier != "exploratory":
                raise AnalysisError(
                    f"§9.3 map key {key} is ambiguous across tiers "
                    f"({existing.tier!r} vs {candidate.tier!r}) with no "
                    "ADR-0087 exploratory row to prefer — refusing to route"
                )
        rows[key] = candidate
    return FamilyContext(
        group=group, datasets=datasets, table=scoped, keys=keys, rows=rows
    )


def _primary_endpoint(contrast_id: int) -> str:
    return chain_endpoint(contrast_id)


def _gate_leg_key(upstream: str, dataset: str, secondary_metric: str) -> str:
    """The ``<dataset>|<metric>`` primary-outcome key a secondary gates on.

    The headline (#4) gates per (dataset × the secondary's OWN metric); the
    estimand endpoints (#13/#14) key their outcomes under their registered
    estimand variable (decision d, 2026-08-16).
    """
    return f"{dataset}|{_UPSTREAM_LEG_METRIC.get(upstream, secondary_metric)}"


def _annotate_missing_leg(leg: str) -> str:
    """Human reason for a registered co-primary leg with no outcome (G5)."""
    if leg.endswith(f"|{PREDICATE_METRIC}"):
        return (
            f"registered co-primary leg {leg!r} has no outcome: "
            f"{_PREDICATE_FIX_HINT} — the set FAILS, "
            "it never shrinks (G5, 2026-08-16)"
        )
    return (
        f"registered co-primary leg {leg!r} has no outcome — the set FAILS, "
        "it never shrinks (G5, 2026-08-16)"
    )


def run_gatekeeping(
    contrast_stats: Sequence[Mapping[str, Any]],
    family_ctx: FamilyContext | None,
    *,
    alpha: float = 0.05,
    intra_set_rule: str = "all-datasets",
    extra_primaries: Sequence[PrimaryOutcome] = (),
) -> dict[str, Any]:
    """Execute the §9.3 Dmitrienko serial chain + Holm-within-family gating.

    Primaries: every computed primary-tier per-dataset p becomes a
    ``PrimaryOutcome`` under endpoint ``contrast-<id>``; the per-dataset
    co-primary SET spans dataset × metric (§9.1's metric pair are co-primary),
    so the outcome's dataset key is ``<dataset>|<metric>``.
    ``extra_primaries`` carries executor-produced endpoint outcomes (the #13
    fingerprint intersection-union p per dataset, keyed
    ``<dataset>|fingerprint``). The chain runs in the registered
    ``families.PRIMARY_CHAIN_ORDER`` restricted to the endpoints this run
    computed; ``chain_complete`` is False (and loudly listed) whenever any
    registered chain endpoint is missing.

    Tier routing consults the family-map ROW tier carried on each entry (G2)
    — an ADR-0087 exploratory row can never enter here as a primary or a
    Holm-family member.

    Secondaries (decision d, 2026-08-16): each computed secondary row joins
    its REGISTERED family (the map's 5-axis ``family_id``) and gates on its
    REGISTERED ``upstream`` endpoint from the map — never a driver hard-code.
    With a family context the chain is BOUND to the registration (G5):
    ``registered_sets`` (a missing registered co-primary leg FAILS the set —
    the predicate leg's absence fails #4's set with a reason naming the
    producer command), ``registered_family_sizes`` (Holm at the REGISTERED m), and
    ``upstream_by_family`` (topology enforced). Rows with no computable
    upstream outcome are listed under ``ungated`` — never silently dropped.
    """
    primaries: list[PrimaryOutcome] = list(extra_primaries)
    secondaries: list[SecondaryOutcome] = []
    ungated: list[dict[str, Any]] = []
    computed_primary_ids: set[int] = {
        int(p.endpoint.rsplit("-", 1)[1]) for p in extra_primaries
    }

    for entry in contrast_stats:
        if entry["tier"] != "primary":
            continue
        cid = int(entry["contrast_id"])
        computed_primary_ids.add(cid)
        for row in entry["per_dataset"]:
            if "p_value" not in row:
                continue
            key = f"{row['dataset']}|{entry['metric']}"
            primaries.append(
                PrimaryOutcome(
                    endpoint=_primary_endpoint(cid),
                    dataset=key,
                    p_value=float(row["p_value"]),
                )
            )

    if not primaries:
        return {
            "skipped": (
                "no primary-tier contrast was computed in this invocation — "
                "the §9.3 serial chain needs at least the headline (#4) "
                "outcomes to gate anything"
            )
        }

    supplied_primary_keys = {(p.endpoint, p.dataset) for p in primaries}
    group = family_ctx.group if family_ctx is not None else "?"
    for entry in contrast_stats:
        if entry["tier"] != "secondary":
            continue
        for row in entry["per_dataset"]:
            if "p_value" not in row:
                continue
            metric = str(entry["metric"])
            dataset = str(row["dataset"])
            if family_ctx is not None:
                map_row = family_ctx.map_row(
                    int(entry["contrast_id"]), metric, dataset,
                    str(entry.get("family")),
                )
                if map_row is None:
                    ungated.append(
                        {
                            "contrast_id": entry["contrast_id"],
                            "metric": metric,
                            "dataset": dataset,
                            "reason": (
                                "not a §9.3 family-map row for this run's "
                                "group — reported raw, unregistered"
                            ),
                        }
                    )
                    continue
                family_id = map_row.family_id
                upstream = map_row.upstream
            else:
                # No charter family map (non-charter datasets, design-input):
                # legacy flat behavior — headline-gated pseudo-family.
                family_id = f"{group}|{metric}|{dataset}"
                upstream = _primary_endpoint(HEADLINE_CONTRAST_ID)
            gate_key = _gate_leg_key(upstream, dataset, metric)
            if (upstream, gate_key) not in supplied_primary_keys:
                ungated.append(
                    {
                        "contrast_id": entry["contrast_id"],
                        "metric": metric,
                        "dataset": dataset,
                        "upstream": upstream,
                        "reason": (
                            f"registered upstream primary {upstream!r} has no "
                            f"computed outcome on {gate_key!r} — the gate "
                            "cannot open or close; reported raw, unregistered"
                        ),
                    }
                )
                continue
            secondaries.append(
                SecondaryOutcome(
                    contrast=f"#{entry['contrast_id']} {entry['name']}",
                    family_id=family_id,
                    upstream=upstream,
                    dataset=gate_key,
                    p_value=float(row["p_value"]),
                )
            )

    registered_order = [
        _primary_endpoint(cid)
        for cid in PRIMARY_CHAIN_ORDER
        if cid in computed_primary_ids
    ]
    missing_endpoints = [
        _primary_endpoint(cid)
        for cid in PRIMARY_CHAIN_ORDER
        if cid not in computed_primary_ids
    ]
    registered_sets: dict[str, tuple[str, ...]] | None = None
    registered_family_sizes: dict[str, int] | None = None
    upstream_by_family: dict[str, str] | None = None
    if family_ctx is not None:
        # G5 / decision d: bind the chain to the REGISTERED expectations.
        registered_sets = {
            endpoint: family_ctx.registered_set_legs(cid)
            for cid, endpoint in (
                (cid, _primary_endpoint(cid)) for cid in PRIMARY_CHAIN_ORDER
            )
            if cid in computed_primary_ids
            and family_ctx.registered_set_legs(cid)
        }
        registered_family_sizes = family_ctx.registered_family_sizes()
        upstream_by_family = family_ctx.upstream_by_family()
    try:
        trace: GatekeepingTrace = evaluate_chain(
            primaries,
            secondaries,
            alpha=alpha,
            primary_order=registered_order,
            intra_set_rule=intra_set_rule,  # type: ignore[arg-type]
            registered_sets=registered_sets,
            registered_family_sizes=registered_family_sizes,
            upstream_by_family=upstream_by_family,
        )
    except GatekeepingError as exc:
        raise AnalysisError(f"gatekeeping chain refused (§9.3): {exc}") from exc

    return {
        "alpha": trace.alpha,
        "intra_set_rule": trace.intra_set_rule,
        "primary_chain_order_registered": [
            _primary_endpoint(cid) for cid in PRIMARY_CHAIN_ORDER
        ],
        "primary_chain_order_executed": list(trace.primary_order or ()),
        "chain_complete": not missing_endpoints,
        "missing_primary_endpoints": missing_endpoints,
        "missing_endpoints_note": (
            None
            if not missing_endpoints
            else (
                "INCOMPLETE CHAIN: the registered Dmitrienko serial sequence "
                f"(families.PRIMARY_CHAIN_ORDER) includes {missing_endpoints} "
                "which this invocation could not compute — downstream "
                "endpoints of a missing gate cannot be claimed confirmatory"
            )
        ),
        "primaries": [
            {
                "endpoint": p.endpoint,
                "dataset_metric": p.dataset,
                "p_value": p.p_value,
                "alpha": p.alpha,
                "passed": p.passed,
                "status": p.status,
            }
            for p in trace.primaries
        ],
        "secondaries": [
            {
                "contrast": s.contrast,
                "family_id": s.family_id,
                "dataset_metric": s.dataset,
                "p_value": s.p_value,
                "status": s.status,
                "p_holm_within_family": s.p_holm,
                "significant": s.significant,
                "m_supplied": s.m_supplied,
                "m_registered": s.m_registered,
            }
            for s in trace.secondaries
        ],
        "set_decisions": [
            {
                "endpoint": d.endpoint,
                "rule": d.rule,
                "passed": d.passed,
                "binding_p": d.binding_p,
                "supplied_legs": list(d.supplied_legs),
                "registered_legs": (
                    list(d.registered_legs)
                    if d.registered_legs is not None
                    else None
                ),
                "missing_legs": list(d.missing_legs),
                "reason": d.reason,
                "missing_leg_reasons": [
                    _annotate_missing_leg(leg) for leg in d.missing_legs
                ],
            }
            for d in trace.set_decisions
        ],
        "events": [
            {
                "family_id": e.family_id,
                "upstream": e.upstream,
                "dataset_metric": e.dataset,
                "upstream_p": e.upstream_p,
                "opened": e.opened,
                "reason": e.reason,
            }
            for e in trace.events
        ],
        "ungated": ungated,
    }


# ---------------------------------------------------------------------------
# §9.5 conditional TOST for the family map's declared equivalence legs
# ---------------------------------------------------------------------------


#: Non-baseline axes a policy-vs-none pressure pair must agree on.
_POLICY_PAIR_MATCH_AXES: tuple[str, ...] = (
    "arm", "retriever", "topology", "engine", "model", "family",
    "budget_r", "rate_frac",
)
#: Axes for the #13 truncate leg (B11 vs B6): the baseline ids fix arm +
#: retriever, so those axes are excluded (they differ by construction).
_TRUNCATE_PAIR_MATCH_AXES: tuple[str, ...] = (
    "policy", "topology", "engine", "model", "family", "budget_r", "rate_frac",
)


def _match_pressure_pairs(
    cells: pd.DataFrame,
    refs: pd.DataFrame,
    match_axes: Sequence[str],
    label: str,
) -> list[tuple[str, str, list[str]]]:
    """(cell_row_key, ref_row_key, shared datasets) per matched axes-slot.

    Shared §9.5/#13 pressure matcher; groups the ``_coord_keyed`` copies so
    NaN pressure coordinates pair as absence (H4). Ambiguous slots fail loud.
    """
    pairs: list[tuple[str, str, list[str]]] = []
    ref_groups = {
        key: grp
        for key, grp in _coord_keyed(refs).groupby(list(match_axes), dropna=False)
    }
    for key, cell_grp in _coord_keyed(cells).groupby(list(match_axes), dropna=False):
        ref_grp = ref_groups.get(key)
        if ref_grp is None:
            continue
        cell_keys = sorted(cell_grp["row_key"].unique())
        ref_keys = sorted(ref_grp["row_key"].unique())
        if len(cell_keys) != 1 or len(ref_keys) != 1:
            raise AnalysisError(
                f"{label}: ambiguous cell pair (cells={cell_keys}, "
                f"refs={ref_keys}); refusing to guess"
            )
        datasets = sorted(set(cell_grp["dataset"]) & set(ref_grp["dataset"]))
        if datasets:
            pairs.append((cell_keys[0], ref_keys[0], datasets))
    return pairs


def _paired_pivot(
    per_query: pd.DataFrame,
    dataset: str,
    cell_key: str,
    ref_key: str,
    column: str,
    *,
    agg: str = "mean",
    by_window: bool = True,
) -> pd.DataFrame | None:
    """Wide (a, b) pivot of ``column`` for one dataset's pair.

    ``by_window=True`` pairs per (example_id, window_key) — the §9.5
    pressure carve-out keeps the per-example estimand while the WINDOW stays
    recoverable as the block-bootstrap resampling unit (decision c).
    ``by_window=False`` averages across windows first (one draw per example).
    Returns None when either side is absent entirely.
    """
    sub = per_query[per_query["dataset"] == dataset]
    group_cols = ["example_id", "window_key", "row_key"] if by_window else [
        "example_id", "row_key"
    ]
    wide = (
        sub.groupby(group_cols, observed=True)[column]
        .agg(agg)
        .unstack("row_key")
    )
    if cell_key not in wide.columns or ref_key not in wide.columns:
        return None
    return wide


def compute_equivalence(
    per_query_loader: Any,
    index: pd.DataFrame,
    family_ctx: FamilyContext | None,
    *,
    metric: str | None,
    margin: float | None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Conditional two-layer TOST (+ ROPE line) for the declared NONE legs.

    The §9.3 family map declares TOST rows for the three pre-registered NONE
    fingerprints (#13: recompute / offload / distribute). Each is either
    COMPUTED (policy-vs-none cell pair found, ``policy_event`` mask present,
    registered margin supplied) or listed as a labeled skip with the exact
    missing ingredient — declared legs are never silently absent.

    Decision c (2026-08-16, G18/G13): these legs live on PRESSURE cells, so
    pairing is per (example_id, window_key) and the WINDOW is passed to
    ``conditional_tost``/``rope_sensitivity`` as the block-bootstrap
    resampling unit — within-window dependence can never make equivalence
    artificially easy. Fewer than the registered floor of unique windows
    REFUSES (fail-loud, wrapped with the leg's context). G6: a paired row
    whose ``policy_event`` telemetry is MISSING is EXCLUDED and counted
    (``n_policy_event_missing``) — absence is never coerced to "no event".

    ``per_query_loader(row_keys) -> pd.DataFrame`` defers I/O so no window is
    read unless a leg is actually computable.
    """
    declared = [
        {"policy": policy, "correction": corr, "sidedness": sided, "predicted": pred}
        for policy, corr, sided, pred in FINGERPRINT_SUB_HYPOTHESES
        if corr == "tost"
    ]
    section: dict[str, Any] = {
        "declared_legs": declared,
        "source": "families.FINGERPRINT_SUB_HYPOTHESES (§9.3 TOST rows)",
        "resampling": (
            "window-block bootstrap (decision c 2026-08-16: window = "
            "resampling unit on §9.5 pressure legs)"
        ),
        "results": [],
        "skipped": [],
    }

    def skip(policy: str, reason: str) -> None:
        section["skipped"].append({"policy": policy, "reason": reason})

    if family_ctx is None:
        for leg in declared:
            skip(leg["policy"], "no §9.3 family-map dataset in this run")
        return section
    if margin is None or metric is None:
        for leg in declared:
            skip(
                leg["policy"],
                "no registered §9.5 margin/metric supplied "
                "(--tost-margin + --equivalence-metric)",
            )
        return section

    pressure = index[index["family"].isin(sorted(PRESSURE_FAMILIES))]
    for leg in declared:
        policy = leg["policy"]
        if policy not in _TOST_POLICY_AXIS_LEGS:
            skip(
                policy,
                "topology-slot leg (distribute): policy-axis pairing does not "
                "apply; DIST-topology pairing is not implemented in this driver",
            )
            continue
        cells = pressure[pressure["policy"] == policy]
        refs = pressure[pressure["policy"] == "none"]
        if cells.empty:
            skip(policy, f"no policy={policy!r} pressure cells in this run")
            continue
        matched = _match_pressure_pairs(
            cells, refs, _POLICY_PAIR_MATCH_AXES, f"equivalence leg {policy!r}"
        )
        found_pair = False
        for cell_key, ref_key, datasets in matched:
            datasets = sorted(set(datasets) & set(family_ctx.datasets))
            if not datasets:
                continue
            found_pair = True
            per_query = per_query_loader({cell_key, ref_key})
            if metric not in per_query.columns:
                skip(
                    policy,
                    f"metric {metric!r} absent from the pair's per-query "
                    f"records ({cell_key} vs {ref_key})",
                )
                continue
            if POLICY_EVENT_COLUMN not in per_query.columns:
                skip(
                    policy,
                    f"no {POLICY_EVENT_COLUMN!r} column in the pair's "
                    "per-query records — the §9.5 CONDITIONAL population "
                    "needs the S2 policy-event mask; an unconditional TOST "
                    "passes trivially and proves nothing",
                )
                continue
            for dataset in datasets:
                wide = _paired_pivot(
                    per_query, dataset, cell_key, ref_key, metric
                )
                if wide is None:
                    skip(
                        policy,
                        f"{dataset}: no {metric!r} values on both sides "
                        f"({cell_key} vs {ref_key})",
                    )
                    continue
                mask_wide = _paired_pivot(
                    per_query, dataset, cell_key, ref_key,
                    POLICY_EVENT_COLUMN, agg="max",
                )
                wide = wide.dropna(subset=[cell_key, ref_key])
                if wide.empty:
                    skip(policy, f"{dataset}: no overlapping example_id pairs")
                    continue
                # G6: the S2 mask is TELEMETRY — a paired example/window with
                # no policy_event record is EXCLUDED with a counted reason,
                # never treated as "no event" (fillna(0.0) was the defect).
                assert mask_wide is not None  # column presence checked above
                mask_series = mask_wide.reindex(wide.index)[cell_key]
                n_mask_missing = int(mask_series.isna().sum())
                keep = mask_series.notna()
                wide = wide[keep]
                if wide.empty:
                    skip(
                        policy,
                        f"{dataset}: every paired row lacks "
                        f"{POLICY_EVENT_COLUMN!r} telemetry "
                        f"({n_mask_missing} excluded, G6)",
                    )
                    continue
                a = wide[cell_key].to_numpy(dtype=float)
                b = wide[ref_key].to_numpy(dtype=float)
                mask = mask_series[keep].to_numpy(dtype=float) > 0.0
                window_ids = wide.index.get_level_values("window_key").to_numpy()
                try:
                    tost = conditional_tost(
                        a, b, mask, margin=margin, alpha=alpha,
                        seed=BOOTSTRAP_SEED, window_ids=window_ids,
                    )
                    rope = rope_sensitivity(
                        a, b, mask, rope=margin,
                        seed=ROPE_SEED, window_ids=window_ids,
                    )
                except ValueError as exc:
                    raise AnalysisError(
                        f"equivalence leg {policy!r} × {dataset} "
                        f"({cell_key} vs {ref_key}): {exc}"
                    ) from exc
                section["results"].append(
                    {
                        "policy": policy,
                        "predicted": leg["predicted"],
                        "dataset": dataset,
                        "metric": metric,
                        "cell_row_key": cell_key,
                        "reference_row_key": ref_key,
                        "margin": tost.margin,
                        "n_total": tost.n_total,
                        "n_policy_event_missing": n_mask_missing,
                        "n_events": tost.n_events,
                        "n_discordant": tost.n_discordant,
                        "mean_diff": tost.mean_diff,
                        "p_tost": tost.p_tost,
                        "domain_verdict": tost.domain_verdict,
                        "dominance": tost.dominance,
                        "dominance_ci": [
                            tost.dominance_ci_low,
                            tost.dominance_ci_high,
                        ],
                        "dominance_verdict": tost.dominance_verdict,
                        "equivalent": tost.equivalent,
                        "resampling": tost.resampling,
                        "n_windows": tost.n_windows,
                        "rope_sensitivity": {
                            "p_left": rope.p_left,
                            "p_rope": rope.p_rope,
                            "p_right": rope.p_right,
                            "verdict": rope.verdict,
                            "resampling": rope.resampling,
                            "note": "sensitivity LINE beside the TOST "
                            "conclusion, never the confirmatory gate (§9.5)",
                        },
                    }
                )
        if not found_pair:
            skip(
                policy,
                f"policy={policy!r} cells never share axes+coords+dataset "
                "with a policy='none' reference in this run",
            )
    return section


# ---------------------------------------------------------------------------
# G4a: #13 fingerprint superiority legs (Holm at registered m=3 + IU p)
# ---------------------------------------------------------------------------


def compute_fingerprint(
    per_query_loader: Any,
    index: pd.DataFrame,
    family_ctx: FamilyContext | None,
    *,
    metric: str | None,
    alpha: float = 0.05,
) -> tuple[dict[str, Any], list[PrimaryOutcome]]:
    """The #13 superiority legs per ``families.FINGERPRINT_SUB_HYPOTHESES``.

    Per dataset, the 3 registered one-sided legs (evict / compress-fp8 on the
    policy axis; truncate = the B11-vs-B6 arm pair under pressure, §7.3) run
    a paired Wilcoxon (registered ``zero_method='pratt'``) on the fingerprint
    quality instrument (``--equivalence-metric``, the same §9.5 instrument),
    tail = "the policy HARMS quality". Pairing is per example (averaged
    across windows — one draw per example, the §9.5 pressure carve-out's
    conservative reading for a rank test that cannot block-cluster). Holm
    runs at the REGISTERED m=3: missing legs are padded at p=1.0, and the
    intersection-union p (max of the 3 adjusted values, pads included) is
    the endpoint's chain contribution — an incomplete fingerprint therefore
    contributes p=1.0 and can never pass its chain step. Returns
    (section, per-dataset PrimaryOutcomes keyed ``<dataset>|fingerprint``).
    """
    declared = [
        {"leg": policy, "correction": corr, "sidedness": sided, "predicted": pred}
        for policy, corr, sided, pred in FINGERPRINT_SUB_HYPOTHESES
        if corr == "holm"
    ]
    section: dict[str, Any] = {
        "contrast_id": FINGERPRINT_CONTRAST_ID,
        "declared_legs": declared,
        "source": "families.FINGERPRINT_SUB_HYPOTHESES (§9.3 Holm rows)",
        "holm_m_registered": FINGERPRINT_HOLM_M,
        "legs": [],
        "per_dataset_intersection": [],
        "skipped": [],
        "note": (
            "intersection-union p = max of the 3 Holm-adjusted superiority "
            "legs (missing legs padded at p=1.0) — the #13 chain endpoint "
            "contribution; the 3 NONE predictions ride stats['equivalence']"
        ),
    }

    def skip(leg: str, reason: str) -> None:
        section["skipped"].append({"leg": leg, "reason": reason})

    if family_ctx is None:
        for leg in declared:
            skip(leg["leg"], "no §9.3 family-map dataset in this run")
        return section, []
    if metric is None:
        for leg in declared:
            skip(
                leg["leg"],
                "no fingerprint quality instrument supplied "
                "(--equivalence-metric; the §9.5 instrument is the "
                "fingerprint instrument)",
            )
        return section, []

    higher_is_better = _metric_direction(metric)
    # The registered claim: the coping policy HARMS quality (cell worse).
    alternative = "greater" if not higher_is_better else "less"
    pressure = index[index["family"].isin(sorted(PRESSURE_FAMILIES))]
    #: dataset -> {leg -> p}
    leg_p: dict[str, dict[str, float]] = {}

    def leg_pairs(leg: str) -> list[tuple[str, str, list[str]]] | None:
        if leg in _FINGERPRINT_POLICY_OF_LEG:
            policy_value = _FINGERPRINT_POLICY_OF_LEG[leg]
            cells = pressure[pressure["policy"] == policy_value]
            refs = pressure[pressure["policy"] == "none"]
            if cells.empty:
                skip(leg, f"no policy={policy_value!r} pressure cells in this run")
                return None
            return _match_pressure_pairs(
                cells, refs, _POLICY_PAIR_MATCH_AXES, f"fingerprint leg {leg!r}"
            )
        cell_b, ref_b = _FINGERPRINT_TRUNCATE_PAIR
        cells = pressure[pressure["baseline"] == cell_b]
        refs = pressure[pressure["baseline"] == ref_b]
        if cells.empty or refs.empty:
            skip(
                leg,
                f"no {cell_b}-vs-{ref_b} pressure pair in this run "
                "(truncation rides the arm axis, §7.3)",
            )
            return None
        return _match_pressure_pairs(
            cells, refs, _TRUNCATE_PAIR_MATCH_AXES, f"fingerprint leg {leg!r}"
        )

    for leg_info in declared:
        leg = leg_info["leg"]
        matched = leg_pairs(leg)
        if matched is None:
            continue
        found = False
        for cell_key, ref_key, datasets in matched:
            datasets = sorted(set(datasets) & set(family_ctx.datasets))
            if not datasets:
                continue
            found = True
            per_query = per_query_loader({cell_key, ref_key})
            if metric not in per_query.columns:
                skip(
                    leg,
                    f"metric {metric!r} absent from the pair's per-query "
                    f"records ({cell_key} vs {ref_key})",
                )
                continue
            for dataset in datasets:
                wide = _paired_pivot(
                    per_query, dataset, cell_key, ref_key, metric,
                    by_window=False,
                )
                if wide is None:
                    skip(
                        leg,
                        f"{dataset}: no {metric!r} values on both sides "
                        f"({cell_key} vs {ref_key})",
                    )
                    continue
                wide = wide.dropna(subset=[cell_key, ref_key])
                if wide.empty:
                    skip(leg, f"{dataset}: no overlapping example_id pairs")
                    continue
                a = wide[cell_key].to_numpy(dtype=float)
                b = wide[ref_key].to_numpy(dtype=float)
                result = paired_wilcoxon(
                    a, b, alternative=alternative, zero_method="pratt"
                )
                if dataset in leg_p and leg in leg_p[dataset]:
                    raise AnalysisError(
                        f"fingerprint leg {leg!r} × {dataset}: two matched "
                        "pressure pairs supply the same registered leg — "
                        "ambiguous; refusing to guess"
                    )
                leg_p.setdefault(dataset, {})[leg] = result.p_value
                section["legs"].append(
                    {
                        "leg": leg,
                        "predicted": leg_info["predicted"],
                        "dataset": dataset,
                        "metric": metric,
                        "cell_row_key": cell_key,
                        "reference_row_key": ref_key,
                        "n_pairs": result.n_pairs,
                        "n_nonzero": result.n_nonzero,
                        "zero_method": result.zero_method,
                        "executed_alternative": alternative,
                        "p_value": result.p_value,
                        "cliffs_delta_paired": result.cliffs_delta_paired,
                    }
                )
        if matched is not None and not found:
            skip(
                leg,
                f"leg {leg!r}: matched pressure cells share no charter "
                "dataset in this run",
            )

    primaries: list[PrimaryOutcome] = []
    for dataset in sorted(leg_p):
        supplied = leg_p[dataset]
        missing = sorted(
            {d["leg"] for d in declared} - set(supplied)
        )
        padded = list(supplied.values()) + [1.0] * len(missing)
        adjusted = holm(padded)
        p_iu = float(max(adjusted))
        for leg_row in section["legs"]:
            if leg_row["dataset"] != dataset:
                continue
            leg_idx = list(supplied).index(leg_row["leg"])
            leg_row["p_holm_within_fingerprint"] = float(adjusted[leg_idx])
        section["per_dataset_intersection"].append(
            {
                "dataset": dataset,
                "p_intersection_union": p_iu,
                "n_legs_supplied": len(supplied),
                "holm_m_registered": FINGERPRINT_HOLM_M,
                "missing_legs": missing,
                "note": (
                    "incomplete fingerprint: missing legs padded at p=1.0, "
                    "IU p is 1.0 by construction" if missing else
                    "complete registered fingerprint (3 legs)"
                ),
            }
        )
        primaries.append(
            PrimaryOutcome(
                endpoint=_primary_endpoint(FINGERPRINT_CONTRAST_ID),
                dataset=f"{dataset}|fingerprint",
                p_value=p_iu,
            )
        )
    return section, primaries


# ---------------------------------------------------------------------------
# G4b: #12 lambda_star_onset (falsification suite executor)
# ---------------------------------------------------------------------------


def lambda_star_onset_from_grid(
    rates: Sequence[float], powers: Sequence[float]
) -> dict[str, Any]:
    """Interpolated Chiu-Jain power-metric argmax on one rate grid (§9.2).

    ``rates`` are rate_frac grid points (fractions of the predicted λ*, so
    the prediction under test is onset at 1.0); ``powers`` the Chiu-Jain
    power metric at each point. Quadratic interpolation through the argmax
    and its neighbors gives the onset; an argmax at either grid EDGE is the
    registered INCONCLUSIVE-AT-RESOLUTION label (no interior maximum at this
    resolution — §9.2). Verdict: onset inside the multiplicative ×/÷1.15
    band around 1.0 -> WITHIN-BAND; outside -> OUTSIDE-BAND (publishable in
    either direction; the suite spends no α).
    """
    rate_arr = np.asarray(rates, dtype=float)
    power_arr = np.asarray(powers, dtype=float)
    if rate_arr.ndim != 1 or rate_arr.shape != power_arr.shape:
        raise AnalysisError(
            f"lambda-star grid shapes disagree: rates {rate_arr.shape} vs "
            f"powers {power_arr.shape}"
        )
    if not (np.all(np.isfinite(rate_arr)) and np.all(np.isfinite(power_arr))):
        raise AnalysisError("lambda-star grid holds non-finite values")
    if np.unique(rate_arr).size < LAMBDA_STAR_MIN_GRID_POINTS:
        raise AnalysisError(
            f"lambda-star onset needs >= {LAMBDA_STAR_MIN_GRID_POINTS} "
            f"distinct rate_frac grid points to interpolate an interior "
            f"argmax; got {sorted(np.unique(rate_arr))}"
        )
    order = np.argsort(rate_arr, kind="stable")
    rate_arr, power_arr = rate_arr[order], power_arr[order]
    k = int(np.argmax(power_arr))
    band_low, band_high = 1.0 / LAMBDA_STAR_BAND, LAMBDA_STAR_BAND
    if k == 0 or k == rate_arr.size - 1:
        return {
            "estimand": "lambda_star_onset",
            "onset_rate_frac": float(rate_arr[k]),
            "interpolated": False,
            "verdict": "INCONCLUSIVE-AT-RESOLUTION",
            "band": [band_low, band_high],
            "grid_rate_frac": rate_arr.tolist(),
            "grid_power": power_arr.tolist(),
            "reason": (
                "Chiu-Jain power-metric argmax sits at the grid EDGE — no "
                "interior maximum at this grid resolution (§9.2 registered "
                "label)"
            ),
        }
    x0, x1, x2 = rate_arr[k - 1 : k + 2]
    y0, y1, y2 = power_arr[k - 1 : k + 2]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a_coef = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b_coef = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom
    onset = float(-b_coef / (2 * a_coef)) if a_coef != 0.0 else float(x1)
    within = band_low <= onset <= band_high
    return {
        "estimand": "lambda_star_onset",
        "onset_rate_frac": onset,
        "interpolated": True,
        "verdict": "WITHIN-BAND" if within else "OUTSIDE-BAND",
        "band": [band_low, band_high],
        "grid_rate_frac": rate_arr.tolist(),
        "grid_power": power_arr.tolist(),
        "reason": (
            f"interpolated Chiu-Jain argmax at rate_frac={onset:.4g} "
            f"{'inside' if within else 'OUTSIDE'} the ×/÷{LAMBDA_STAR_BAND} "
            "band around the predicted λ* (rate_frac=1.0)"
        ),
    }


def compute_falsification_suite(
    per_query_loader: Any, index: pd.DataFrame
) -> dict[str, Any]:
    """#12 executor: the pressure-curve onset vs the λ* prediction (§9.2).

    FAIL-LOUD by design (G4b): a run without the required inputs names
    exactly which artifact is missing — the F2 rate grid (>= 3 rate_frac
    points per pressure cell), and per-window ``goodput_frac`` +
    ``latency_ms`` columns (the Chiu-Jain power metric = goodput-weighted
    offered rate over response time). The suite is falsification tier: it
    spends no α and its verdicts are labels, never gates (§9.2 exile).
    """
    f2 = index[index["family"] == "F2"].copy()
    f2 = f2[pd.to_numeric(f2["rate_frac"], errors="coerce").notna()]
    if f2.empty:
        raise AnalysisError(
            "contrast #12 (lambda_star_onset): MISSING ARTIFACT — no F2 "
            "pressure rows with a numeric rate_frac grid in this run's "
            "index; the §6.1 rate sweep (S0 campaign producer, task #116) "
            "has not landed"
        )
    group_axes = [
        "model", "engine", "arm", "retriever", "policy", "topology",
        "budget_r", "dataset",
    ]
    results: list[dict[str, Any]] = []
    for key, grp in f2.groupby(group_axes, dropna=False):
        rates = pd.to_numeric(grp["rate_frac"], errors="coerce")
        if rates.nunique() < LAMBDA_STAR_MIN_GRID_POINTS:
            continue
        per_query = per_query_loader(set(grp["row_key"].unique()))
        missing_cols = [
            c for c in _LAMBDA_STAR_REQUIRED_COLUMNS
            if c not in per_query.columns
        ]
        if missing_cols:
            raise AnalysisError(
                f"contrast #12 (lambda_star_onset): MISSING ARTIFACT — "
                f"per-query records for cells {sorted(grp['row_key'].unique())} "
                f"carry no {missing_cols} column(s); the Chiu-Jain power "
                "metric needs per-window goodput_frac and latency_ms (the "
                "#116/#126 regime bridge produces them)"
            )
        grid: dict[float, float] = {}
        joined = grp.merge(
            per_query, on=["row_key", "dataset", "window_key"], how="inner",
            suffixes=("", "_pq"),
        )
        for rate, rate_grp in joined.groupby(
            pd.to_numeric(joined["rate_frac"], errors="coerce")
        ):
            window_means = rate_grp.groupby("window_key")[
                list(_LAMBDA_STAR_REQUIRED_COLUMNS)
            ].mean()
            goodput = float(window_means["goodput_frac"].mean())
            latency = float(window_means["latency_ms"].mean())
            if latency <= 0.0:
                raise AnalysisError(
                    f"contrast #12: non-positive mean latency at "
                    f"rate_frac={rate} for {key} — power metric undefined"
                )
            grid[float(rate)] = float(rate) * goodput / latency
        onset = lambda_star_onset_from_grid(list(grid), list(grid.values()))
        onset["cell_axes"] = dict(zip(group_axes, [str(v) for v in key]))
        results.append(onset)
    if not results:
        raise AnalysisError(
            "contrast #12 (lambda_star_onset): MISSING ARTIFACT — no F2 "
            f"pressure cell carries >= {LAMBDA_STAR_MIN_GRID_POINTS} distinct "
            "rate_frac grid points; the §6.1 rate sweep grid "
            "({0.5,0.7,0.85,0.95,1.05,1.2}·λ*) has not been produced "
            "(task #116)"
        )
    return {
        "contrast_id": FLOOR_SUITE_CONTRAST_ID,
        "tier": "falsification",
        "label": (
            "§9.2 EXILE: standalone falsification suite — spends no α, "
            "gates nothing, publishable in either direction"
        ),
        "estimand": "lambda_star_onset",
        "band_multiplicative": LAMBDA_STAR_BAND,
        "results": results,
    }


# ---------------------------------------------------------------------------
# G4c: #14 truth_tax (the §9.2 estimand executor — task #119 replaces the
# fail-loud stub; the loud refusals for trees that predate the predicate
# REMAIN, naming exactly which artifact is missing)
# ---------------------------------------------------------------------------


def _predicate_join_key(obj: Mapping[str, Any]) -> tuple[Any, str, Any]:
    """The #127 identity triple (mirrors src.analysis.predicate)."""
    return (
        obj.get("example_id"),
        str(obj.get("repeat_index") or "0"),
        obj.get("record_index"),
    )


def _window_truth_tax(
    run_dir: Path,
    rec: Any,
    predicate_root: Path,
    floors: Mapping[str, Any],
) -> tuple[float | None, str | None]:
    """One window's §9.2 variable G − Y, or (None, exclusion reason).

    - §9.2 population = IN-REGIME cells: the window's ``regime.json`` (§6.1
      referee, campaign_layout.write_window_regime) must label it IN_REGIME;
      any other label EXCLUDES the window (counted upstream, per the
      registered population — an exclusion, not a failure). A missing/refused
      regime artifact fails loud: an uncertifiable window has no population
      membership.
    - Y = timely ∧ veridical via ``goodput.evaluate_window`` on the requests ⋈
      predicate join; ttft_ms/tpot_ms convert to seconds HERE (F2's one
      registered ms→s seam); duration = cell.json windows[] t_end − t_start.
    - An ok request without a predicate row fails loud (§9.10: the predicate
      must be scored for every completion); not-ok rows are non-veridical by
      registration (audit §2.6).
    """
    window_dir = run_dir / str(rec.window_dir)
    window_label = str(rec.window_dir)

    regime_path = window_dir / "regime.json"
    if not regime_path.is_file():
        raise AnalysisError(
            f"contrast #14 (truth_tax): MISSING ARTIFACT — {regime_path} "
            "absent; the §9.2 population is in-regime cells and needs the "
            "§6.1 regime referee per window "
            "(src.orchestration.campaign_layout.write_window_regime, #126)"
        )
    regime = json.loads(regime_path.read_text(encoding="utf-8"))
    label = regime.get("label")
    if label != IN_REGIME:
        return None, f"regime label {label!r} (population = in-regime cells, §9.2)"

    engine = str(rec.engine)
    floor = floors.get(engine)
    if not isinstance(floor, Mapping) or not {"ttft_s", "tpot_s"} <= set(floor):
        raise AnalysisError(
            f"contrast #14 (truth_tax): MISSING ARTIFACT — manifest "
            f"{_SLO_FLOORS_MANIFEST_KEY!r} carries no ttft_s/tpot_s floor "
            f"for engine {engine!r}; the §6.1 primary SLO pair is relative "
            "to the measured single-stream floor (E3 calibration, "
            "src/orchestration/calibration.summarize_floor)"
        )

    cell_meta = json.loads(
        (run_dir / str(rec.cell_json)).read_text(encoding="utf-8")
    )
    window_meta = (cell_meta.get("windows") or {}).get(str(rec.window_key))
    if (
        not isinstance(window_meta, Mapping)
        or window_meta.get("t_start") is None
        or window_meta.get("t_end") is None
    ):
        raise AnalysisError(
            f"contrast #14 (truth_tax): MISSING ARTIFACT — cell.json "
            f"windows[{rec.window_key!r}] carries no t_start/t_end for "
            f"{window_label}; the §1 windows[] table (campaign_layout, #126) "
            "supplies the pre-costed window duration"
        )
    duration_s = float(window_meta["t_end"]) - float(window_meta["t_start"])

    requests = _read_jsonl(window_dir / "requests.jsonl")
    if not requests:
        raise AnalysisError(
            f"contrast #14 (truth_tax): {window_dir / 'requests.jsonl'} has "
            "no rows — an empty window has no G or Y"
        )
    missing_cols = sorted(
        c for c in _TRUTH_TAX_REQUEST_COLUMNS
        if not any(c in r for r in requests)
    )
    if missing_cols:
        raise AnalysisError(
            f"contrast #14 (truth_tax): MISSING ARTIFACT — requests.jsonl "
            f"rows in {window_label} carry no {missing_cols} column(s); Y "
            "needs the #127 ok stamp and per-request ttft_ms/tpot_ms for "
            "the §6.1 SLO gate"
        )

    predicate_by_key: dict[tuple[Any, str, Any], Any] = {}
    pred_path = predicate_root / str(rec.window_dir) / PREDICATE_ROWS_NAME
    if not pred_path.is_file():
        raise AnalysisError(
            f"contrast #14 (truth_tax): MISSING ARTIFACT — {pred_path} "
            f"absent; {_PREDICATE_FIX_HINT}"
        )
    for obj in _read_jsonl(pred_path):
        predicate_by_key[_predicate_join_key(obj)] = obj.get("predicate")

    records: list[dict[str, Any]] = []
    unscored_ok: list[tuple[Any, str, Any]] = []
    for req in requests:
        key = _predicate_join_key(req)
        ok = bool(req.get("ok"))
        if key not in predicate_by_key:
            if ok:
                unscored_ok.append(key)
                continue
            verid: Any = float("nan")  # non-completion: non-veridical (§2.6)
        else:
            pred = predicate_by_key[key]
            verid = float("nan") if pred is None else bool(pred)
        ttft_ms = req.get("ttft_ms")
        tpot_ms = req.get("tpot_ms")
        records.append(
            {
                "ok": ok,
                "veridical": verid,
                # F2: THE ms→s conversion seam (evaluate_window is seconds).
                "ttft_s": (
                    float("nan") if ttft_ms is None else float(ttft_ms) / 1000.0
                ),
                "tpot_s": (
                    float("nan") if tpot_ms is None else float(tpot_ms) / 1000.0
                ),
            }
        )
    if unscored_ok:
        raise AnalysisError(
            f"contrast #14 (truth_tax): {len(unscored_ok)} completed (ok) "
            f"request(s) in {window_label} have NO predicate row (first: "
            f"{unscored_ok[:3]}) — the §8.5 predicate must be scored for "
            "every completion (§9.10); rebuild the predicate table against "
            "this tree"
        )

    baseline = SLOBaseline(
        ttft_s=float(floor["ttft_s"]), tpot_s=float(floor["tpot_s"])
    )
    try:
        metrics = evaluate_window(
            pd.DataFrame(records), baseline, duration_s=duration_s
        )
    except GoodputError as exc:
        raise AnalysisError(
            f"contrast #14 (truth_tax): {window_label}: {exc}"
        ) from exc
    return float(metrics.truth_tax_frac), None


def compute_truth_tax(
    run_dir: Path,
    index: pd.DataFrame,
    family_ctx: FamilyContext | None,
    *,
    predicate_root: Path | None,
    alpha: float = 0.05,
) -> tuple[dict[str, Any], list[PrimaryOutcome]]:
    """#14 executor: the §9.2 truth-tax estimand (G4c, task #119).

    Registered estimand (§9.2, verbatim): population = in-regime cells (§6.1
    3-layer referee); variable = G − Y; population summary = batch-means
    contrast across windows. The contrast rides the ENGINE slot ("cross-
    engine policy bundles at same NORMALIZED pressure", §7.8 #14): F2 cells
    agreeing on every non-engine axis + pressure coordinates pair each
    non-anchor engine against the vLLM anchor; per dataset the leg p-values
    are Holm-adjusted and the intersection-union p (max) is the chain
    endpoint contribution keyed ``<dataset>|truth_tax`` (the #13 executor's
    IU convention; ``ESTIMAND_HIGHER_IS_BETTER['truth_tax'] = False`` is the
    direction registry, untouched).

    Sidedness: the registry row says one-sided but NO engine-slot tail is
    registered anywhere (REGISTERED_CELL_DIRECTION has no #14 entry; §7.8 /
    §9.2 pin none) — the executor therefore runs the Welch contrast
    two-sided (the §9.6 power-sim precedent: run_power_sim's window stage
    runs batch_means_contrast two-sided), which can only be CONSERVATIVE
    (a two-sided p is never smaller than its one-sided half). The executed
    alternative and this note are recorded on every leg; registering the
    tail is an open pre-freeze owner item.

    FAIL-LOUD (G4): a run without the required inputs names exactly which
    artifact is missing — the §8.5 predicate table (task #119's producer),
    the manifest ``slo_floors`` (E3), per-window ``regime.json`` (#126) and
    cell.json windows[] t-bounds. Trees that predate the predicate get the
    same loud refusal the stub gave, now naming the producer command.
    """
    if predicate_root is None:
        raise AnalysisError(
            "contrast #14 (truth-tax estimand, §9.2) cannot be computed: its "
            "registered variable truth_tax = G − Y requires the per-query "
            f"§8.5 Y predicate, and {_PREDICATE_FIX_HINT}"
        )
    f2 = index[index["family"] == "F2"]
    if f2.empty:
        raise AnalysisError(
            "contrast #14 (truth_tax): MISSING ARTIFACT — no F2 pressure "
            "cells in this run's index; the §6.1 pressure grid (S0 campaign "
            "producer, task #116) has not landed"
        )
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    floors = manifest.get(_SLO_FLOORS_MANIFEST_KEY)
    if not isinstance(floors, Mapping) or not floors:
        raise AnalysisError(
            f"contrast #14 (truth_tax): MISSING ARTIFACT — {manifest_path} "
            f"carries no {_SLO_FLOORS_MANIFEST_KEY!r} mapping "
            "({engine: {ttft_s, tpot_s}}); the §6.1 SLO pair is relative to "
            "the measured single-stream floor (E3 calibration, "
            "src/orchestration/calibration.summarize_floor)"
        )

    section: dict[str, Any] = {
        "contrast_id": 14,
        "tier": "primary",
        "estimand": "truth_tax",
        "estimand_text": (
            "§9.2: population = in-regime cells (§6.1 3-layer referee); "
            "variable = G − Y; population summary = batch-means contrast "
            "across windows; slot = engine (policy bundle) at normalized "
            "pressure"
        ),
        "higher_is_better": ESTIMAND_HIGHER_IS_BETTER["truth_tax"],
        "anchor_engine": _TRUTH_TAX_ANCHOR_ENGINE,
        "registered_sidedness": "one-sided",
        "executed_alternative": "two-sided",
        "sidedness_note": (
            "registry sidedness is one-sided but no engine-slot tail is "
            "registered (REGISTERED_CELL_DIRECTION has no #14 entry; "
            "§7.8/§9.2 pin none) — executed two-sided per the §9.6 "
            "power-sim precedent (conservative); tail registration is an "
            "open pre-freeze owner item"
        ),
        "predicate_table": predicate_root.name,
        "legs": [],
        "per_dataset_intersection": [],
        "excluded_windows": [],
        "skipped": [],
    }

    def skip(reason: str) -> None:
        section["skipped"].append({"reason": reason})

    #: (dataset, leg-engine) -> p, guarded against duplicate supply.
    leg_p: dict[str, dict[str, float]] = {}
    tt_cache: dict[str, float | None] = {}

    def window_values(grp: pd.DataFrame) -> list[float]:
        values: list[float] = []
        for rec in grp.itertuples(index=False):
            cache_key = f"{rec.window_dir}"
            if cache_key not in tt_cache:
                value, excluded = _window_truth_tax(
                    run_dir, rec, predicate_root, floors
                )
                tt_cache[cache_key] = value
                if excluded is not None:
                    section["excluded_windows"].append(
                        {"window": str(rec.window_dir), "reason": excluded}
                    )
            if tt_cache[cache_key] is not None:
                values.append(float(tt_cache[cache_key]))  # type: ignore[arg-type]
        return values

    keyed = _coord_keyed(f2)
    for dataset, ds_grp in keyed.groupby("dataset"):
        if str(dataset) not in PREDICATE_DATASETS:
            skip(
                f"dataset {dataset!r} is outside the §8.5 predicate universe "
                "— never feeds Y"
            )
            continue
        if family_ctx is not None and str(dataset) not in family_ctx.datasets:
            skip(f"dataset {dataset!r} is not a §9.3 family-map dataset")
            continue
        for axes_key, grp in ds_grp.groupby(
            list(_TRUTH_TAX_GROUP_AXES), dropna=False
        ):
            engines = sorted(grp["engine"].dropna().unique())
            if _TRUTH_TAX_ANCHOR_ENGINE not in engines:
                skip(
                    f"{dataset} @ {dict(zip(_TRUTH_TAX_GROUP_AXES, axes_key))}: "
                    f"no {_TRUTH_TAX_ANCHOR_ENGINE!r} anchor cell among "
                    f"engines {engines}"
                )
                continue
            others = [e for e in engines if e != _TRUTH_TAX_ANCHOR_ENGINE]
            if not others:
                skip(
                    f"{dataset} @ {dict(zip(_TRUTH_TAX_GROUP_AXES, axes_key))}: "
                    "anchor engine only — no cross-engine partner"
                )
                continue
            anchor_grp = grp[grp["engine"] == _TRUTH_TAX_ANCHOR_ENGINE]
            anchor_values = window_values(anchor_grp)
            for engine in others:
                cell_grp = grp[grp["engine"] == engine]
                cell_values = window_values(cell_grp)
                if len(cell_values) < 2 or len(anchor_values) < 2:
                    skip(
                        f"{dataset}: {engine} vs "
                        f"{_TRUTH_TAX_ANCHOR_ENGINE}: needs >= 2 in-regime "
                        f"windows per side for a Welch variance estimate; "
                        f"got cell={len(cell_values)}, "
                        f"anchor={len(anchor_values)} (in-regime population, "
                        "§9.2)"
                    )
                    continue
                result = batch_means_contrast(
                    cell_values, anchor_values, alternative="two-sided"
                )
                if engine in leg_p.get(str(dataset), {}):
                    raise AnalysisError(
                        f"contrast #14: engine leg {engine!r} × {dataset} is "
                        "supplied by two matched pressure groups — "
                        "ambiguous; refusing to guess"
                    )
                leg_p.setdefault(str(dataset), {})[engine] = result.p_value
                section["legs"].append(
                    {
                        "dataset": str(dataset),
                        "engine": engine,
                        "anchor_engine": _TRUTH_TAX_ANCHOR_ENGINE,
                        "axes": {
                            k: str(v)
                            for k, v in zip(_TRUTH_TAX_GROUP_AXES, axes_key)
                        },
                        "n_windows_cell": result.n_windows_a,
                        "n_windows_anchor": result.n_windows_b,
                        "mean_truth_tax_cell": result.mean_a,
                        "mean_truth_tax_anchor": result.mean_b,
                        "mean_diff": result.mean_diff,
                        "statistic": result.statistic,
                        "df": result.df,
                        "p_value": result.p_value,
                        "ci95_low": result.ci95_low,
                        "ci95_high": result.ci95_high,
                        "executed_alternative": "two-sided",
                    }
                )

    primaries: list[PrimaryOutcome] = []
    for dataset in sorted(leg_p):
        supplied = leg_p[dataset]
        adjusted = holm(list(supplied.values()))
        p_iu = float(max(adjusted))
        for leg_row in section["legs"]:
            if leg_row["dataset"] != dataset:
                continue
            leg_idx = list(supplied).index(leg_row["engine"])
            leg_row["p_holm_within_dataset"] = float(adjusted[leg_idx])
        section["per_dataset_intersection"].append(
            {
                "dataset": dataset,
                "p_intersection_union": p_iu,
                "n_legs": len(supplied),
                "engines": sorted(supplied),
            }
        )
        primaries.append(
            PrimaryOutcome(
                endpoint=_primary_endpoint(14),
                dataset=f"{dataset}|truth_tax",
                p_value=p_iu,
            )
        )
    if not primaries:
        raise AnalysisError(
            "contrast #14 (truth_tax): no computable cross-engine leg — "
            + (
                "; ".join(s["reason"] for s in section["skipped"])
                or "no F2 window pair matched"
            )
        )
    return section, primaries


# ---------------------------------------------------------------------------
# G3: exploratory tier (BH-FDR, separated, non-confirmatory)
# ---------------------------------------------------------------------------


def build_exploratory_section(
    contrast_stats: Sequence[Mapping[str, Any]],
    family_ctx: FamilyContext | None,
) -> dict[str, Any]:
    """BH-FDR over the computed exploratory-tier rows (G3).

    The registered §9.3 exploratory tier (ADR-0087 faithfulness rows, #11,
    #16, #19) receives ``corrections.benjamini_hochberg`` within the
    computed exploratory set and lives in its own clearly-non-confirmatory
    section — never in the chain, never in a Holm family (G2).
    """
    rows: list[dict[str, Any]] = []
    for entry in contrast_stats:
        if entry["tier"] != "exploratory":
            continue
        for row in entry["per_dataset"]:
            if "p_value" not in row:
                continue
            rows.append(
                {
                    "contrast_id": entry["contrast_id"],
                    "name": entry["name"],
                    "metric": entry["metric"],
                    "dataset": row["dataset"],
                    "p_value": float(row["p_value"]),
                }
            )
    if rows:
        adjusted = benjamini_hochberg([r["p_value"] for r in rows])
        for row, p_adj in zip(rows, adjusted):
            row["p_bh_fdr"] = float(p_adj)
    n_registered = (
        int((family_ctx.table["correction"] == "bh-fdr").sum())
        if family_ctx is not None
        else None
    )
    return {
        "label": "EXPLORATORY — NON-CONFIRMATORY (§9.3 bh-fdr tier)",
        "correction": (
            "benjamini-hochberg across the computed exploratory rows "
            "(corrections.benjamini_hochberg; registered §9.3 exploratory "
            "tier — ungated, no α spent, no confirmatory sentence may cite "
            "these rows)"
        ),
        "n_computed": len(rows),
        "n_registered_rows_scoped": n_registered,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# G16: ADR-0086 realized-n gate (registered floor + step-down ladder)
# ---------------------------------------------------------------------------


def check_realized_n(
    contrast_stats: Sequence[Mapping[str, Any]],
    accepted_step_down: int | None,
) -> dict[str, Any]:
    """Confirmatory ADR-0086 gate: primary per-query rows below the floor.

    The ladder IS the registered data (``ADR0086_REALIZED_N_LADDER``): the
    first rung is the registered floor; ``accepted_step_down`` names the
    pre-declared rung the look explicitly steps down to (recorded — never
    silent). A primary row whose realized n is below the accepted floor
    REFUSES the look (before any output; the placeholder lock is released
    by the caller's failure path, so the one-look budget survives).
    """
    ladder = ADR0086_REALIZED_N_LADDER
    if accepted_step_down is not None and accepted_step_down not in ladder:
        raise AnalysisError(
            f"--accept-step-down {accepted_step_down} is not a rung of the "
            f"pre-declared ADR-0086 ladder {list(ladder)} — only registered "
            "rungs may be accepted (G16)"
        )
    floor = accepted_step_down if accepted_step_down is not None else ladder[0]
    violations: list[str] = []
    for entry in contrast_stats:
        if entry["tier"] != "primary" or entry.get("unit") == "window":
            continue
        for row in entry["per_dataset"]:
            realized = row.get("realized_n")
            if realized is None:
                continue
            if int(realized) < floor:
                violations.append(
                    f"contrast #{entry['contrast_id']} × {entry['metric']} × "
                    f"{row['dataset']}: realized n={realized} < floor={floor}"
                )
    if violations:
        detail = "\n".join(f"  {v}" for v in violations)
        raise AnalysisError(
            "ADR-0086 REALIZED-N REFUSAL (G16): primary rows below the "
            f"accepted floor ({floor}; registered ladder {list(ladder)}):\n"
            f"{detail}\n— a confirmatory look below the registered floor "
            "requires the pre-declared step-down (--accept-step-down "
            "<rung>), and no rung admits these n"
        )
    return {
        "checked": True,
        "ladder": list(ladder),
        "floor": floor,
        "step_down_accepted": accepted_step_down,
    }


# ---------------------------------------------------------------------------
# G14: atomic outputs + executing-code provenance stamp
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    """tmp + ``os.replace``: no reader ever observes a partial file (G14)."""
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def executing_provenance() -> dict[str, Any]:
    """The G14 stamp: EXECUTING code SHA + dirty flag + resampling seeds."""
    try:
        sha, dirty = _git_head_state()
    except Exception:  # noqa: BLE001 — provenance must never sink an analysis
        sha, dirty = None, None
    return {
        "executing_git_sha": sha,
        "executing_git_dirty": dirty,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "rope_seed": ROPE_SEED,
    }


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def _fmt_effect(row: Mapping[str, Any]) -> str:
    """Cliff's δ (paired) for Wilcoxon rows; proportion-diff + discordant n
    for McNemar (binary) rows — the two tests report different effect sizes."""
    if "cliffs_delta_paired" in row:
        return f"{row['cliffs_delta_paired']:.3f}"
    return f"{row['proportion_diff']:.3f} (n_disc={row['n_discordant']})"


def _fmt_p_holm(row: Mapping[str, Any]) -> str:
    """``None`` for primary-tier rows (§9.1: no cross-dataset correction)."""
    p_holm = row.get("p_holm_across_datasets")
    return f"{p_holm:.3g}" if p_holm is not None else "n/a (primary tier, full α)"


def build_summary_md(stats: Mapping[str, Any]) -> str:
    stamp = stats["mode_stamp"]
    run = stats["run"]
    lines: list[str] = [
        f"# Campaign analysis summary — **{stamp}**",
        "",
        f"- run: `{run['campaign']}/{run['session']}/{run['run_id']}` "
        f"(model `{run['model']}`)",
        f"- generated (UTC): {stats['generated_utc']}",
        f"- mode: {stats['one_look']['mode']}"
        + (
            f" · registered SHA `{stats['one_look']['registered_sha']}`"
            if stats["one_look"]["registered_sha"]
            else ""
        ),
        f"- sidedness policy (decision a, 2026-08-16): each row EXECUTES its "
        "REGISTERED sidedness from the §9.3 family-map row (paired Wilcoxon "
        "zero_method='pratt' for continuous metrics, McNemar exact-binomial "
        "for binary-unit rows, batch-means Welch t for loaded windows; the "
        "executed alternative is recorded per contrast); correction: none "
        "for primary-tier endpoints (full α per dataset, §9.1), diagnostic "
        "Holm across datasets for secondaries, BH-FDR for the exploratory "
        "tier (separated section) — the registered §9.3 Holm-within-family "
        "correction is executed in the gatekeeping section below",
        "",
        f"**Every number below is {stamp}.**",
        "",
    ]
    for entry in stats["contrasts"]:
        lines.append(
            f"## Contrast #{entry['contrast_id']} — {entry['name']} "
            f"[{entry['metric']}]"
        )
        lines.append("")
        lines.append(
            f"cell `{entry['cell_baseline']}` = `{entry['cell_row_key']}`  ·  "
            f"reference `{entry['reference_baseline']}` = "
            f"`{entry['reference_row_key']}`  ·  "
            f"{'higher' if entry['higher_is_better'] else 'lower'} is better  ·  "
            f"test: `{entry['test']}`"
        )
        lines.append("")
        if entry["unit"] == "window":
            lines.append(
                "| dataset | windows (cell/ref) | mean Δ (cell−ref) | 95% CI "
                "| p | p (Holm across datasets) |"
            )
            lines.append("|---|---|---|---|---|---|")
            for row in entry["per_dataset"]:
                if "skipped" in row:
                    lines.append(
                        f"| {row['dataset']} | — | — | — | SKIPPED: "
                        f"{row['skipped']} | — |"
                    )
                    continue
                lines.append(
                    f"| {row['dataset']} "
                    f"| {row['n_windows_cell']}/{row['n_windows_reference']} "
                    f"| {row['mean_diff']:.4g} "
                    f"| [{row['ci95_low']:.4g}, {row['ci95_high']:.4g}] "
                    f"| {row['p_value']:.3g} | {_fmt_p_holm(row)} |"
                )
        else:
            lines.append(
                "| dataset | n_pairs | median Δ (cell−ref) | W/L/T | effect | p | "
                "p (Holm across datasets) |"
            )
            lines.append("|---|---|---|---|---|---|---|")
            for row in entry["per_dataset"]:
                lines.append(
                    f"| {row['dataset']} | {row['n_pairs']} "
                    f"| {row['median_delta']:.4g} "
                    f"| {row['wins']}/{row['losses']}/{row['ties']} "
                    f"| {_fmt_effect(row)} "
                    f"| {row['p_value']:.3g} | {_fmt_p_holm(row)} |"
                )
        lines.append("")

    gate = stats.get("gatekeeping", {})
    lines.append("## Gatekeeping chain (§9.3 — Dmitrienko serial + Holm within family)")
    lines.append("")
    if "skipped" in gate:
        lines.append(f"SKIPPED: {gate['skipped']}")
    else:
        lines.append(
            f"- executed order: {' → '.join(gate['primary_chain_order_executed'])} "
            f"(registered: {' → '.join(gate['primary_chain_order_registered'])}; "
            f"chain {'COMPLETE' if gate['chain_complete'] else 'INCOMPLETE'})"
        )
        if gate.get("missing_endpoints_note"):
            lines.append(f"- **{gate['missing_endpoints_note']}**")
        lines.append(
            f"- intra-set rule: `{gate['intra_set_rule']}` at α={gate['alpha']:g}"
        )
        for decision in gate.get("set_decisions", ()):
            verdict = "PASSED" if decision["passed"] else "FAILED"
            lines.append(
                f"- co-primary set `{decision['endpoint']}`: **{verdict}** "
                f"(binding p={decision['binding_p']:.3g})"
            )
            for reason in decision.get("missing_leg_reasons", ()):
                lines.append(f"  - **{reason}**")
        lines.append("")
        lines.append("| endpoint | dataset×metric | p | passed | status |")
        lines.append("|---|---|---|---|---|")
        for p in gate["primaries"]:
            lines.append(
                f"| {p['endpoint']} | {p['dataset_metric']} "
                f"| {p['p_value']:.3g} | {p['passed']} | {p['status']} |"
            )
        if gate["secondaries"]:
            lines.append("")
            lines.append(
                "| secondary | family | dataset×metric | p | p (Holm within "
                "family) | status | significant |"
            )
            lines.append("|---|---|---|---|---|---|---|")
            for s in gate["secondaries"]:
                p_holm_s = (
                    f"{s['p_holm_within_family']:.3g}"
                    if s["p_holm_within_family"] is not None
                    else "n/a (descriptive)"
                )
                lines.append(
                    f"| {s['contrast']} | {s['family_id']} "
                    f"| {s['dataset_metric']} | {s['p_value']:.3g} "
                    f"| {p_holm_s} | {s['status']} | {s['significant']} |"
                )
        lines.append("")
        lines.append(f"Gate events (auditable trace): {len(gate['events'])} — see stats.json.")
    lines.append("")

    equiv = stats.get("equivalence", {})
    lines.append("## Equivalence legs (§9.5 — conditional two-layer TOST + ROPE line)")
    lines.append("")
    for result in equiv.get("results", ()):
        lines.append(
            f"- `{result['policy']}` × {result['dataset']} [{result['metric']}]: "
            f"n_events={result['n_events']} (discordant {result['n_discordant']}), "
            f"p_TOST={result['p_tost']:.3g}, domain={result['domain_verdict']}, "
            f"dominance={result['dominance_verdict']} → "
            f"{'EQUIVALENT' if result['equivalent'] else 'not equivalent'} "
            f"(ROPE: p_rope={result['rope_sensitivity']['p_rope']:.3g}, "
            f"{result['rope_sensitivity']['verdict']})"
        )
    for skipped_leg in equiv.get("skipped", ()):
        lines.append(
            f"- `{skipped_leg['policy']}`: SKIPPED — {skipped_leg['reason']}"
        )
    lines.append("")

    fingerprint = stats.get("fingerprint", {})
    lines.append(
        "## Fingerprint superiority legs (#13, §9.3 — Holm at registered "
        f"m={fingerprint.get('holm_m_registered', 3)} + intersection-union p)"
    )
    lines.append("")
    for leg_row in fingerprint.get("legs", ()):
        p_holm_fp = leg_row.get("p_holm_within_fingerprint")
        lines.append(
            f"- `{leg_row['leg']}` × {leg_row['dataset']} "
            f"[{leg_row['metric']}]: p={leg_row['p_value']:.3g}"
            + (f", p_holm={p_holm_fp:.3g}" if p_holm_fp is not None else "")
            + f" (alternative `{leg_row['executed_alternative']}`, "
            f"n_nonzero={leg_row['n_nonzero']})"
        )
    for iu in fingerprint.get("per_dataset_intersection", ()):
        lines.append(
            f"- **{iu['dataset']}: IU p = {iu['p_intersection_union']:.3g}** "
            f"({iu['n_legs_supplied']}/{iu['holm_m_registered']} legs; "
            f"{iu['note']})"
        )
    for skipped_leg in fingerprint.get("skipped", ()):
        lines.append(
            f"- `{skipped_leg['leg']}`: SKIPPED — {skipped_leg['reason']}"
        )
    lines.append("")

    exploratory = stats.get("exploratory", {})
    if exploratory.get("n_computed"):
        lines.append("## Exploratory tier (§9.3 BH-FDR) — **NON-CONFIRMATORY**")
        lines.append("")
        lines.append(f"- {exploratory['correction']}")
        lines.append("")
        lines.append("| contrast | metric | dataset | p | p (BH-FDR) |")
        lines.append("|---|---|---|---|---|")
        for row in exploratory["rows"]:
            lines.append(
                f"| #{row['contrast_id']} {row['name']} | {row['metric']} "
                f"| {row['dataset']} | {row['p_value']:.3g} "
                f"| {row['p_bh_fdr']:.3g} |"
            )
        lines.append("")

    truth_tax = stats.get("truth_tax")
    if truth_tax and "suppressed" not in truth_tax:
        lines.append("## Truth-tax estimand (#14, §9.2 — chain primary)")
        lines.append("")
        lines.append(f"- {truth_tax['estimand_text']}")
        lines.append(f"- sidedness: {truth_tax['sidedness_note']}")
        lines.append("")
        for leg in truth_tax.get("legs", ()):
            lines.append(
                f"- {leg['dataset']}: `{leg['engine']}` vs "
                f"`{leg['anchor_engine']}`: Δ(G−Y) = {leg['mean_diff']:+.4f} "
                f"(p={leg['p_value']:.3g}, windows "
                f"{leg['n_windows_cell']}/{leg['n_windows_anchor']})"
            )
        for iu in truth_tax.get("per_dataset_intersection", ()):
            lines.append(
                f"- **{iu['dataset']}: IU p = "
                f"{iu['p_intersection_union']:.3g}** "
                f"({iu['n_legs']} engine leg(s))"
            )
        for skipped_leg in truth_tax.get("skipped", ()):
            lines.append(f"- SKIPPED — {skipped_leg['reason']}")
        if truth_tax.get("excluded_windows"):
            lines.append(
                f"- excluded windows (§9.2 in-regime population): "
                f"{len(truth_tax['excluded_windows'])} (counted in stats.json)"
            )
        lines.append("")

    falsification = stats.get("falsification")
    if falsification:
        lines.append(
            "## Falsification suite (#12, §9.2 exile — spends no α)"
        )
        lines.append("")
        for result in falsification["results"]:
            lines.append(
                f"- onset rate_frac = {result['onset_rate_frac']:.4g} "
                f"-> **{result['verdict']}** ({result['reason']})"
            )
        lines.append("")

    blinding_info = stats.get("blinding", {})
    if blinding_info.get("sealed_map"):
        lines.append("## Blinding (§9.8)")
        lines.append("")
        lines.append(
            f"- sealed map: `{blinding_info['sealed_map']}` "
            f"(sha256 `{blinding_info.get('map_sha256')}`)"
        )
        if blinding_info.get("active"):
            lines.append(
                "- **LABELS SCRAMBLED** — arm-revealing labels masked, figures "
                "suppressed until the logged unblinding"
            )
        event = blinding_info.get("unblind_event")
        if event:
            lines.append(
                f"- UNBLINDED at {event['utc']} (logged to `{event['log']}`)"
            )
        lines.append("")

    skipped = stats["skipped"]
    if (
        skipped["pressure_rows"]
        or skipped["contrasts"]
        or skipped.get("confirmatory_exclusions")
    ):
        lines.append("## Skipped (labeled — never silently absent)")
        lines.append("")
        if skipped["pressure_rows"]:
            block = skipped["pressure_rows"]
            lines.append(
                f"**Pressure rows (F2/F3), {block['n_windows']} window(s) — "
                "batch-means only (§9.4), never per-query paired; not consumed "
                "by any computed contrast:**"
            )
            for key in block["row_keys"]:
                lines.append(f"- `{key}`")
            lines.append("")
        for entry in skipped["contrasts"]:
            lines.append(
                f"- contrast #{entry['contrast_id']} ({entry['name']}) "
                f"[{entry['label']}]: {entry['reason']}"
            )
        for excl in skipped.get("confirmatory_exclusions", ()):
            lines.append(
                f"- contrast #{excl['contrast_id']} × {excl['metric']} on "
                f"{excl['datasets']}: {excl['reason']}"
            )
        lines.append("")
    if stats["figures"]:
        lines.append("## Figures (registered statistics — stats.json rows)")
        lines.append("")
        for entry in stats["figures"]:
            if "file" in entry:
                lines.append(
                    f"- `{entry['file']}` ({entry['kind']}; consumes the "
                    "registered stats.json contrast rows; unpairable examples "
                    f"dropped+counted: {entry['n_dropped_nan_total']}; "
                    f"stamped {stamp} in-figure)"
                )
            else:
                lines.append(
                    f"- metric `{entry['skipped_metric']}` — NO FIGURE: "
                    f"{entry['reason']}"
                )
        lines.append("")
    lines.append("---")
    prov = stats.get("provenance", {})
    lines.append(
        f"Stamp: **{stamp}** · schema v{stats['schema_version']} · executing "
        f"code `{prov.get('executing_git_sha')}`"
        f"{' (DIRTY)' if prov.get('executing_git_dirty') else ''} · seeds "
        f"bootstrap={prov.get('bootstrap_seed')}/rope={prov.get('rope_seed')}"
    )
    lines.append("")
    return "\n".join(lines)


def _make_analysis_dir(run_dir: Path) -> Path:
    base = run_dir / ANALYSIS_DIRNAME
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    candidate = base / stamp
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = base / f"{stamp}-{suffix}"
    candidate.mkdir(parents=True)
    return candidate


def read_lock(run_dir: Path) -> dict[str, Any] | None:
    lock_path = run_dir / LOCK_NAME
    if not lock_path.is_file():
        return None
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OneLookError(
            f"{lock_path} exists but is not valid JSON ({exc}) — treating the "
            "run as LOCKED; a corrupt lock is not a license for a second look"
        ) from exc
    return lock if isinstance(lock, dict) else {"raw": lock}


def _one_look_refusal_message(run_dir: Path, lock: dict[str, Any]) -> str:
    if lock.get("phase") == "IN_PROGRESS":
        detail = (
            "a confirmatory analysis is CURRENTLY IN PROGRESS (started "
            f"{lock.get('started_utc', '<unknown time>')} under registered "
            f"SHA {lock.get('registered_sha', '<unknown>')!r})"
        )
    else:
        detail = (
            "this run's confirmatory analysis ran at "
            f"{lock.get('locked_utc', '<unknown time>')} under registered "
            f"SHA {lock.get('registered_sha', '<unknown>')!r}"
        )
    return (
        f"ONE-LOOK REFUSAL (§9.11): {run_dir / LOCK_NAME} already exists "
        f"— {detail}. The campaign data is analyzed once; a second look "
        "invalidates the registration. Design-input re-renders remain "
        "available via --design-input."
    )


def _acquire_confirmatory_lock(run_dir: Path, registered_sha: str) -> None:
    """Atomically claim the run's single confirmatory look (§9.11).

    Uses ``O_CREAT | O_EXCL`` so two concurrent ``--confirmatory`` invocations
    cannot both pass a read-then-write race (the old ``read_lock`` once /
    ``write_lock`` unconditionally-at-the-end pattern): exactly one process's
    ``open()`` succeeds, the other gets ``FileExistsError`` and refuses
    immediately, before running any of the expensive pipeline. The placeholder
    written here is IN_PROGRESS-only; ``run_analysis`` removes it on any
    exception raised by *this* process after acquiring it (see its
    try/except), so a crash mid-pipeline never burns the run's one registered
    look — only ``write_lock`` finalizing to ``phase: DONE`` does that.
    """
    lock_path = run_dir / LOCK_NAME
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        lock = read_lock(run_dir)
        assert lock is not None  # FileExistsError => the file is there
        raise OneLookError(_one_look_refusal_message(run_dir, lock)) from None
    placeholder = {
        "phase": "IN_PROGRESS",
        "registered_sha": registered_sha,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        os.write(fd, (json.dumps(placeholder, indent=2) + "\n").encode("utf-8"))
    except BaseException:
        os.close(fd)
        lock_path.unlink(missing_ok=True)
        raise
    else:
        os.close(fd)


def _release_placeholder_lock(run_dir: Path) -> None:
    """Undo ``_acquire_confirmatory_lock`` after a failed pipeline attempt.

    Only removes the lock if it is STILL the IN_PROGRESS placeholder — never
    a lock some other state finalized — so this can't ever delete a genuine
    completed-look record.
    """
    lock_path = run_dir / LOCK_NAME
    try:
        current = read_lock(run_dir)
    except OneLookError:
        # Corrupt JSON: read_lock's own fail-closed policy owns this state;
        # do not delete evidence we cannot parse.
        return
    if current is not None and current.get("phase") == "IN_PROGRESS":
        lock_path.unlink(missing_ok=True)


def write_lock(
    run_dir: Path, registered_sha: str, analysis_dir: Path, stats_path: Path
) -> Path:
    """Finalize the §9.11 lock over the placeholder from
    ``_acquire_confirmatory_lock``: write to a temp file, then
    ``os.replace`` it into place — an atomic overwrite, so no reader ever
    observes a partially-written lock file."""
    lock_path = run_dir / LOCK_NAME
    payload = {
        "policy": "PUBLICATION.md §9.11 ONE-LOOK: this run's confirmatory "
        "analysis has been executed exactly once",
        "phase": "DONE",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "registered_sha": registered_sha,
        "analysis_dir": analysis_dir.name,
        "stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
    }
    tmp_path = lock_path.with_name(f"{lock_path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, lock_path)
    return lock_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisResult:
    analysis_dir: Path
    stats_path: Path
    summary_path: Path
    figures: tuple[str, ...] = field(default_factory=tuple)


def run_analysis(
    run_dir: Path,
    *,
    contrast_ids: Sequence[int],
    metrics: Sequence[str] | None = None,
    mode: Mode,
    registered_sha: str | None = None,
    calibration_report: Path | None = None,
    tost_margin: float | None = None,
    equivalence_metric: str | None = None,
    alpha: float = 0.05,
    accepted_step_down: int | None = None,
    predicate_run_id: str | None = None,
) -> AnalysisResult:
    """Execute the pipeline; the CLI wraps this with the one-look flag checks.

    Confirmatory preconditions (checked BEFORE the §9.11 lock is acquired, so
    a refusal never touches the run's one-look budget): the G1 registration
    binding (SHA grammar/HEAD/clean-worktree/prereg/alpha/margins/metrics), a
    PASSING §9.7 calibration-report artifact, and a clean §9.10 ledger
    verification of the raw tree. ``metrics=None`` resolves to the mode's
    registered default: ``families.DEFAULT_METRICS`` (the §9.1 co-primary
    pair) in confirmatory mode, the driver's serving default otherwise.
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise AnalysisError(f"run directory does not exist: {run_dir}")

    # G1a: confirmatory metrics ARE the registered pair — the CLI list is a
    # design-input instrument only.
    metrics_overridden = metrics is not None
    if metrics is None:
        resolved_metrics: tuple[str, ...] = (
            REGISTERED_DEFAULT_METRICS if mode == "confirmatory" else DEFAULT_METRICS
        )
    else:
        resolved_metrics = tuple(metrics)
    if (
        mode == "confirmatory"
        and metrics_overridden
        and resolved_metrics != REGISTERED_DEFAULT_METRICS
    ):
        raise AnalysisError(
            f"confirmatory --metrics {list(resolved_metrics)} differs from "
            f"the REGISTERED §9.1 metric pair "
            f"{list(REGISTERED_DEFAULT_METRICS)} — the registered table may "
            "not follow the caller (G1); --metrics is design-input only"
        )
    metrics = resolved_metrics
    for metric in metrics:
        _metric_direction(metric)  # fail before any I/O on unknown direction

    stamp = CONFIRMATORY_STAMP if mode == "confirmatory" else DESIGN_STAMP
    preconditions: dict[str, Any] = {
        "registration": {"checked": False},
        "calibration": {"checked": False},
        "ledger": {"checked": False},
        "realized_n": {
            "checked": False,
            "ladder": list(ADR0086_REALIZED_N_LADDER),
        },
    }
    lock_acquired_here = False
    if mode == "confirmatory":
        if not registered_sha:
            raise OneLookError("confirmatory mode requires a registered SHA")
        # G1: bind the look to the frozen registration BEFORE anything runs.
        preconditions["registration"] = {
            "checked": True,
            **check_registration_binding(
                registered_sha,
                alpha=alpha,
                metrics=metrics,
                metrics_overridden=metrics_overridden,
            ),
        }
        # G1d/G9: margins come from the registered artifact, never the CLI.
        tost_margin, margin_record = resolve_registered_margin(
            tost_margin, equivalence_metric
        )
        preconditions["registration"]["tost_margin"] = margin_record
        if calibration_report is None:
            raise CalibrationGateError(
                "confirmatory mode requires --calibration-report: the §9.7 "
                "calibration artifact (A/A + effect-injection operating "
                "characteristics) is a hard precondition of the one "
                "registered look (§9.13 refusal pattern, as in prereg.py)"
            )
        report = load_calibration_report(calibration_report)
        preconditions["calibration"] = {
            "checked": True,
            **check_calibration(report, calibration_report),
        }
        preconditions["ledger"] = {"checked": True, **verify_run_ledger(run_dir)}
        # Atomic acquire-before-pipeline (§9.11): closes the TOCTOU window
        # where two concurrent --confirmatory invocations could both pass a
        # read-only check and both run the full (expensive) pipeline, with
        # the second writer silently clobbering the first's lock record.
        _acquire_confirmatory_lock(run_dir, registered_sha)
        lock_acquired_here = True
    else:
        if calibration_report is not None:
            # Allowed in design-input as a dry-run of the gate; recorded only.
            report = load_calibration_report(calibration_report)
            preconditions["calibration"] = {
                "checked": True,
                **check_calibration(report, calibration_report),
            }
        if read_lock(run_dir) is not None:
            print(
                f"WARNING: {run_dir / LOCK_NAME} exists — the confirmatory look "
                "is spent. These design-input outputs must never be quoted as "
                "confirmatory.",
                file=sys.stderr,
            )

    try:
        index = load_index(run_dir)

        # Task #119: locate + verify the §8.5 predicate table (the joined
        # decoupled-scoring output). None is legal — the registered predicate
        # legs then refuse/record through the existing missing-column paths.
        predicate_root, predicate_manifest = resolve_predicate_root(
            run_dir, predicate_run_id
        )

        # §9.8: a sealed arm map means the pipeline output stays scrambled
        # until the one-time logged unblinding (which the confirmatory look
        # performs). Tampered seals raise here, before anything is computed.
        blinding_state = load_blinding_state(run_dir)
        blinding_active = (
            blinding_state is not None
            and blinding_state.get("unblinded_utc") is None
            and mode == "design-input"
        )
        blinding_section: dict[str, Any] = {
            "sealed_map": (
                f"{BLINDING_DIRNAME}/{SEALED_MAP_NAME}"
                if blinding_state is not None
                else None
            ),
            "map_sha256": (
                blinding_state.get("map_sha256") if blinding_state else None
            ),
            "active": blinding_active,
            "unblind_event": None,
            "note": (
                "labels scrambled per §9.8 — figures suppressed and "
                "arm-revealing labels masked until the logged unblinding"
                if blinding_active
                else None
            ),
        }

        # Executor-backed ids (G4) never enter the baseline-pair pipeline:
        # #13 runs through compute_fingerprint (always), #12 through
        # compute_falsification_suite and #14 through compute_truth_tax
        # (each when requested).
        pair_ids = [
            cid
            for cid in contrast_ids
            if cid not in (FINGERPRINT_CONTRAST_ID, FLOOR_SUITE_CONTRAST_ID, 14)
        ]
        computable, skipped_contrasts = resolve_contrasts(pair_ids)
        per_query_contrasts = [
            c
            for c in computable
            if c.unit == "per_query" and c.family not in PRESSURE_FAMILIES
        ]
        window_contrasts = [c for c in computable if c.unit == "window"]

        family_ctx = build_family_context(index, metrics, alpha)
        if mode == "confirmatory" and family_ctx is None:
            raise AnalysisError(
                "confirmatory refusal (§9.3): none of this run's datasets "
                f"appear in the registered family map ({sorted(KNOWN_DATASETS)}) "
                "— no test may run that is not a row in the table"
            )

        pairs: list[ResolvedPair] = []
        for contrast in per_query_contrasts:
            pairs.extend(select_contrast_pairs(index, contrast))

        window_pairs: list[WindowPair] = []
        for contrast in window_contrasts:
            w_pairs, w_reasons = select_window_pairs(index, contrast)
            window_pairs.extend(w_pairs)
            for reason in w_reasons:
                skipped_contrasts.append(
                    SkippedContrast(
                        contrast_id=contrast.id,
                        name=contrast.name,
                        label="NO-WINDOW-PAIR-IN-RUN",
                        reason=reason,
                    )
                )

        wanted_keys = {
            k for p in pairs for k in (p.cell_row_key, p.reference_row_key)
        } | {
            k
            for p in window_pairs
            for k in (p.cell_row_key, p.reference_row_key)
        }
        per_query = (
            load_per_query(
                run_dir, index, wanted_keys, predicate_root=predicate_root
            )
            if wanted_keys
            else pd.DataFrame()
        )

        def _in_family(contrast_id: int, metric: str, dataset: str) -> bool:
            if family_ctx is None:
                return False
            return (contrast_id, metric, dataset) in family_ctx.keys

        contrast_stats: list[dict[str, Any]] = []
        confirmatory_exclusions: list[dict[str, Any]] = []

        def _family_filter(
            contrast_id: int, metric: str, datasets: tuple[str, ...]
        ) -> tuple[str, ...]:
            """Confirmatory §9.3 discipline: only family-map rows are tested."""
            if mode != "confirmatory":
                return datasets
            kept = tuple(d for d in datasets if _in_family(contrast_id, metric, d))
            dropped = [d for d in datasets if d not in kept]
            if dropped:
                confirmatory_exclusions.append(
                    {
                        "contrast_id": contrast_id,
                        "metric": metric,
                        "datasets": dropped,
                        "reason": "not a §9.3 family-map row — no test runs "
                        "that is not a row in the table",
                    }
                )
            return kept

        def _map_row_for(
            contrast_id: int, metric: str, datasets: tuple[str, ...],
            family: str,
        ) -> MapRow | None:
            """The registered row routing this (contrast, metric) — G2."""
            if family_ctx is None:
                return None
            for dataset in datasets:
                row = family_ctx.map_row(contrast_id, metric, dataset, family)
                if row is not None:
                    return row
            return None

        def _registered_metric_absent(contrast_id: int, metric: str) -> bool:
            """Confirmatory-only (G5/#119): a REGISTERED metric column with
            no per-query data is a recorded missing-leg state — the
            co-primary set FAILS via registered_sets; the invocation must
            not crash (the CLI never asked for this metric, the
            registration did)."""
            if mode != "confirmatory" or metric in per_query.columns:
                return False
            confirmatory_exclusions.append(
                {
                    "contrast_id": contrast_id,
                    "metric": metric,
                    "datasets": ["ALL"],
                    "reason": (
                        f"registered metric {metric!r} appears in no "
                        "per-query artifact"
                        + (
                            f" — {_PREDICATE_FIX_HINT}; the co-primary set "
                            "FAILS on the missing legs (G5), it never shrinks"
                            if metric == PREDICATE_METRIC
                            else " — the registered leg is missing; the "
                            "co-primary set FAILS on it (G5)"
                        )
                    ),
                }
            )
            return True

        for pair in pairs:
            for metric in metrics:
                if _registered_metric_absent(pair.contrast.id, metric):
                    continue
                allowed = _family_filter(pair.contrast.id, metric, pair.datasets)
                if not allowed:
                    continue
                entry = compute_pair_stats(
                    per_query,
                    dc_replace(pair, datasets=allowed),
                    metric,
                    map_row=_map_row_for(
                        pair.contrast.id, metric, allowed, pair.contrast.family
                    ),
                )
                for row in entry["per_dataset"]:
                    row["in_family_map"] = _in_family(
                        entry["contrast_id"], metric, row["dataset"]
                    )
                contrast_stats.append(entry)
        for w_pair in window_pairs:
            for metric in metrics:
                if _registered_metric_absent(w_pair.contrast.id, metric):
                    continue
                allowed = _family_filter(
                    w_pair.contrast.id, metric, w_pair.datasets
                )
                if not allowed:
                    continue
                entry = compute_window_pair_stats(
                    per_query,
                    dc_replace(w_pair, datasets=allowed),
                    metric,
                    map_row=_map_row_for(
                        w_pair.contrast.id, metric, allowed, w_pair.family
                    ),
                )
                for row in entry["per_dataset"]:
                    row["in_family_map"] = _in_family(
                        entry["contrast_id"], metric, row["dataset"]
                    )
                contrast_stats.append(entry)

        # G16 / ADR-0086: realized n recorded per row above; the confirmatory
        # look refuses below the accepted floor (the placeholder lock is
        # released by the failure path — the budget survives a refusal).
        if mode == "confirmatory":
            preconditions["realized_n"] = check_realized_n(
                contrast_stats, accepted_step_down
            )

        consumed = frozenset(
            k
            for p in window_pairs
            for k in (p.cell_row_key, p.reference_row_key)
        )
        pressure_block = pressure_row_skip(index, consumed)

        per_query_loader = lambda keys: load_per_query(  # noqa: E731
            run_dir, index, keys, predicate_root=predicate_root
        )

        # G4a: the #13 fingerprint superiority legs — their per-dataset
        # intersection-union p is the endpoint's chain contribution.
        fingerprint_section, fingerprint_primaries = compute_fingerprint(
            per_query_loader,
            index,
            family_ctx,
            metric=equivalence_metric,
            alpha=alpha,
        )

        # G4c (task #119): the #14 truth-tax estimand executor — only when
        # requested (fail-loud on missing inputs by design; the pre-predicate
        # refusal REMAINS, naming the producer command).
        truth_tax_section: dict[str, Any] | None = None
        truth_tax_primaries: list[PrimaryOutcome] = []
        if 14 in contrast_ids:
            truth_tax_section, truth_tax_primaries = compute_truth_tax(
                run_dir,
                index,
                family_ctx,
                predicate_root=predicate_root,
                alpha=alpha,
            )

        # §9.3 wiring: the registered chain + Holm-within-family corrections.
        gatekeeping_section = run_gatekeeping(
            contrast_stats,
            family_ctx,
            alpha=alpha,
            extra_primaries=[*fingerprint_primaries, *truth_tax_primaries],
        )

        # §9.5 wiring: conditional TOST for the declared equivalence legs.
        equivalence_section = compute_equivalence(
            per_query_loader,
            index,
            family_ctx,
            metric=equivalence_metric,
            margin=tost_margin,
            alpha=alpha,
        )

        # G4b: the #12 falsification suite, only when explicitly requested
        # (fail-loud on missing inputs by design).
        falsification_section = (
            compute_falsification_suite(per_query_loader, index)
            if FLOOR_SUITE_CONTRAST_ID in contrast_ids
            else None
        )

        # G3: the exploratory tier — BH-FDR, separated, non-confirmatory.
        exploratory_section = build_exploratory_section(
            contrast_stats, family_ctx
        )

        analysis_dir = _make_analysis_dir(run_dir)

        if blinding_active:
            assert blinding_state is not None
            mapping = {
                str(k): str(v) for k, v in dict(blinding_state["mapping"]).items()
            }
            contrast_stats = [
                apply_blinding_to_entry(e, mapping, index) for e in contrast_stats
            ]

        run_identity = {
            key: str(index[key].iloc[0]) for key in ("run_id", "campaign", "session", "model")
        }
        stats: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "mode_stamp": stamp,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "run": run_identity,
            "one_look": {
                "mode": mode,
                "registered_sha": registered_sha,
                "lock_file": LOCK_NAME if mode == "confirmatory" else None,
            },
            "provenance": executing_provenance(),
            "preconditions": preconditions,
            "requested_contrast_ids": list(dict.fromkeys(contrast_ids)),
            "metrics": list(metrics),
            "loader_notes": {
                "bool_coerced_fields": per_query.attrs.get(
                    "bool_coerced_fields", []
                )
                if not per_query.empty
                else [],
                # Task #119: which §8.5 predicate table (if any) was joined.
                "predicate_table": (
                    None
                    if predicate_root is None
                    else {
                        "path": f"{PREDICATE_DIRNAME}/{predicate_root.name}",
                        "scoring_run_id": (
                            predicate_manifest.get("scoring_run_id")
                            if predicate_manifest is not None
                            else None
                        ),
                        "config": (
                            predicate_manifest.get("config")
                            if predicate_manifest is not None
                            else None
                        ),
                    }
                ),
            },
            "family_map": (
                {
                    "group": family_ctx.group,
                    "datasets": list(family_ctx.datasets),
                    "n_rows": int(len(family_ctx.table)),
                    "source": "families.compile_family_map (§9.3)",
                }
                if family_ctx is not None
                else {
                    "absent_reason": (
                        "no charter family-map dataset in this run "
                        f"(known: {sorted(KNOWN_DATASETS)})"
                    )
                }
            ),
            "contrasts": contrast_stats,
            "gatekeeping": gatekeeping_section,
            "equivalence": equivalence_section,
            "fingerprint": fingerprint_section,
            "truth_tax": (
                {
                    "suppressed": (
                        "§9.8 blinding active — the #14 section carries "
                        "row_key/arm-bearing axes; suppressed until the "
                        "logged unblinding"
                    )
                }
                if blinding_active and truth_tax_section is not None
                else truth_tax_section
            ),
            "exploratory": exploratory_section,
            "falsification": falsification_section,
            "blinding": blinding_section,
            "skipped": {
                "pressure_rows": pressure_block,
                "contrasts": [
                    {
                        "contrast_id": s.contrast_id,
                        "name": s.name,
                        "label": s.label,
                        "reason": s.reason,
                    }
                    for s in skipped_contrasts
                ],
                "confirmatory_exclusions": confirmatory_exclusions,
            },
            "figures": [],  # rendered below, AFTER the logged unblinding (I10)
        }
        if blinding_active:
            # G12: mask EVERY arm-revealing section, not just the entries.
            apply_blinding_to_sections(stats, mapping, index)

        # §9.8: the confirmatory look IS the freeze — record the one-time
        # unblinding event AFTER the pipeline computed, BEFORE the outputs
        # are written (a pipeline failure above leaves the seal unspent).
        if mode == "confirmatory" and blinding_state is not None:
            log_path = run_dir / BLINDING_DIRNAME / UNBLIND_LOG_NAME
            try:
                blinding_mod.unblind(sealed_map_path(run_dir), log_path)
            except AlreadyUnblindedError as exc:
                raise AnalysisError(
                    f"blinding refusal (§9.8): {exc} — a confirmatory look "
                    "after a prior unblinding would not have been blind"
                ) from exc
            blinding_section["unblind_event"] = {
                "utc": datetime.now(timezone.utc).isoformat(),
                "log": f"{BLINDING_DIRNAME}/{UNBLIND_LOG_NAME}",
                "map_sha256": blinding_state.get("map_sha256"),
                "note": "one-time logged unblinding at the confirmatory "
                "freeze (§9.8); outputs below carry REAL labels",
            }

        # I10 (audit 2026-08-16): figures render ONLY AFTER the one-time
        # unblinding above is logged — no real-labeled artifact may precede
        # the recorded reveal. I1: they consume the registered contrast
        # entries (the very dicts serialized into stats.json), never a
        # per-query recompute; blinded design-input suppresses them entirely.
        figures = (
            render_figures(
                contrast_stats,
                metrics,
                tuple(per_query.columns),
                analysis_dir,
                stamp,
                index,
            )
            if contrast_stats and not blinding_active
            else []
        )
        stats["figures"] = figures

        stats_path = analysis_dir / STATS_JSON_NAME
        _atomic_write_text(stats_path, json.dumps(stats, indent=2) + "\n")
        summary_path = analysis_dir / SUMMARY_MD_NAME
        _atomic_write_text(summary_path, build_summary_md(stats))

        if mode == "confirmatory":
            assert registered_sha is not None
            write_lock(run_dir, registered_sha, analysis_dir, stats_path)
    except BaseException:
        # A crashed/failed attempt must not consume the one-look budget: only
        # a placeholder THIS process created is removed, and only while it is
        # still IN_PROGRESS (never a lock some other completed run finalized).
        if lock_acquired_here:
            _release_placeholder_lock(run_dir)
        raise

    return AnalysisResult(
        analysis_dir=analysis_dir,
        stats_path=stats_path,
        summary_path=summary_path,
        figures=tuple(
            str(entry["file"]) for entry in figures if "file" in entry
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Campaign analysis driver (D9). Consumes an ORGANIZED run "
            "(index/cells_index.csv from organize_results.py), computes the "
            "registered §7.8 per-query contrasts (default: #4, the B6-vs-B3 "
            "headline, per dataset) with paired Wilcoxon + W/L/T + Holm across "
            "datasets, and emits <run>/analysis/<timestamp>/{stats.json, "
            "summary.md, forest_<metric>.png, wlt_<metric>.png}. Every output "
            "is stamped DESIGN-INPUT-ONLY unless the one-look confirmatory "
            "flags are given (§9.11)."
        )
    )
    parser.add_argument(
        "run_dir", type=Path, help="organized run root: results/<campaign>/<session>/<run_id>"
    )
    parser.add_argument(
        "--contrasts",
        nargs="+",
        type=int,
        default=[HEADLINE_CONTRAST_ID],
        metavar="ID",
        help="§7.8 contrast ids (1-20); unknown ids fail loud; window-unit ids "
        "are reported as NOT-IMPLEMENTED-YET skips (default: 4)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        metavar="METRIC",
        help="per-query metric columns to test — DESIGN-INPUT ONLY (G1): "
        f"design-input default {' '.join(DEFAULT_METRICS)}; confirmatory mode "
        "always tests the registered §9.1 pair "
        f"({' '.join(REGISTERED_DEFAULT_METRICS)}) and refuses an override "
        "that differs",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--design-input",
        action="store_true",
        help="explicit default: outputs stamped DESIGN-INPUT-ONLY, repeatable",
    )
    mode.add_argument(
        "--confirmatory",
        action="store_true",
        help="the ONE confirmatory look (§9.11); requires --i-understand-one-look "
        "AND --registered-sha; writes/checks <run>/analysis_lock.json",
    )
    parser.add_argument(
        "--i-understand-one-look",
        action="store_true",
        help="confirmatory acknowledgment: this is the run's single registered look",
    )
    parser.add_argument(
        "--registered-sha",
        type=str,
        default=None,
        metavar="SHA",
        help="SHA of the frozen pre-registration this confirmatory look executes",
    )
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=None,
        metavar="JSON",
        help="§9.7 calibration-report artifact (CalibrationReport as JSON); "
        "REQUIRED for --confirmatory (the look refuses without a PASSING "
        "report); optional in design-input as a gate dry-run",
    )
    parser.add_argument(
        "--tost-margin",
        type=float,
        default=None,
        metavar="MARGIN",
        help="registered §9.5 domain margin (metric units) for the declared "
        "conditional-TOST equivalence legs; without it the legs are listed "
        "as labeled skips",
    )
    parser.add_argument(
        "--equivalence-metric",
        type=str,
        default=None,
        metavar="METRIC",
        help="per-query metric column the §9.5 equivalence legs test "
        "(paired with --tost-margin)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        metavar="ALPHA",
        help="significance level for the gatekeeping chain and family map "
        "(default: 0.05; confirmatory mode must match REGISTERED_ALPHA)",
    )
    parser.add_argument(
        "--accept-step-down",
        type=int,
        default=None,
        metavar="N",
        help="confirmatory-only: accept the pre-declared ADR-0086 realized-n "
        f"step-down to this ladder rung ({ADR0086_REALIZED_N_LADDER}); "
        "recorded in stats.json — never silent (G16)",
    )
    parser.add_argument(
        "--predicate-run-id",
        type=str,
        default=None,
        metavar="ID",
        help="which predicate/<scoring_run_id>/ table joins the §8.5 "
        "'predicate' metric (task #119; built by build_predicate_table.py). "
        "Default: auto-select when exactly ONE table exists; multiple tables "
        "refuse without this flag; none -> the registered predicate legs "
        "surface through the missing-column refusals",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.confirmatory:
        problems: list[str] = []
        if not args.i_understand_one_look:
            problems.append("--i-understand-one-look is missing")
        if not args.registered_sha:
            problems.append("--registered-sha SHA is missing")
        if args.calibration_report is None:
            problems.append(
                "--calibration-report JSON is missing (§9.7 calibration is a "
                "hard precondition of the one registered look)"
            )
        if problems:
            print(
                "REFUSED: --confirmatory is the run's single registered look "
                f"(§9.11) and demands explicit intent: {'; '.join(problems)}. "
                "Nothing was computed and no lock was written.",
                file=sys.stderr,
            )
            return 1
        mode: Mode = "confirmatory"
    else:
        if (
            args.i_understand_one_look
            or args.registered_sha
            or args.accept_step_down is not None
        ):
            print(
                "REFUSED: --i-understand-one-look/--registered-sha/"
                "--accept-step-down are confirmatory-mode flags; pass "
                "--confirmatory or drop them (design-input outputs never "
                "carry a registration).",
                file=sys.stderr,
            )
            return 1
        mode = "design-input"

    try:
        result = run_analysis(
            args.run_dir,
            contrast_ids=args.contrasts,
            metrics=args.metrics,
            mode=mode,
            registered_sha=args.registered_sha,
            calibration_report=args.calibration_report,
            tost_margin=args.tost_margin,
            equivalence_metric=args.equivalence_metric,
            alpha=args.alpha,
            accepted_step_down=args.accept_step_down,
            predicate_run_id=args.predicate_run_id,
        )
    except AnalysisError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    stamp = CONFIRMATORY_STAMP if mode == "confirmatory" else DESIGN_STAMP
    print(f"[campaign-analysis] mode    : {mode} (outputs stamped {stamp})")
    print(f"[campaign-analysis] stats   : {result.stats_path}")
    print(f"[campaign-analysis] summary : {result.summary_path}")
    for name in result.figures:
        print(f"[campaign-analysis] figure  : {result.analysis_dir / name}")
    if mode == "confirmatory":
        print(f"[campaign-analysis] LOCKED  : {Path(args.run_dir) / LOCK_NAME} (§9.11 one-look)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
