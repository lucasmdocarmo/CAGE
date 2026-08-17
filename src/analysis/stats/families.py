"""§7.8 contrast registry + the §9.3 family map compiler (PUBLICATION.md D9).

The 20 registered contrasts encoded as DATA. Contrast #21 (the S5
up/down-sweep) is FUTURE WORK per §7.8/§6.7 and is deliberately absent —
D9's family map inherits contrasts 1-20 only.

The confirmatory chain (§9.1, adopted 2026-08-02) holds THREE gatekept
primaries, tested in the declared fixed sequence ``PRIMARY_CHAIN_ORDER``
(Dmitrienko serial, audit §2.1):
- #4  B6-vs-B3 — the headline, a per-dataset co-primary SET (pooling
  prohibited; the pilot proved direction inversion, hence two-sided).
- #14 serving-yield (Y) cross-engine contrast — carries the §9.2 truth-tax
  estimand; its registered variable is ``truth_tax`` = G − Y (population =
  in-regime cells; batch-means contrast), never a generic per-window metric.
- #13 the coping-frontier fingerprint — the §8.11 table decomposed per §9.3
  into SIX registered sub-hypothesis rows (one intersection hypothesis in the
  chain): Holm within the 3 superiority predictions (evict/compress/truncate),
  conditional TOST for the 3 pre-registered NONE predictions
  (recompute/offload/distribute).
- #12 the pressure curve + floor-±15% suite is OUT of the primary chain
  (§9.1/§9.2 exile): ``tier="falsification"`` — a standalone suite,
  publishable in either direction, spends no α (``gatekept=False``); its
  registered variable is ``lambda_star_onset`` (interpolated Chiu-Jain argmax
  vs the min(λ_KV, λ_compute) prediction, ×/÷1.15 band).

``compile_family_map`` expands the registry into the §9.3 registered test
table: one row per contrast leg × campaign group × metric × dataset (a
contrast may carry ``extra_legs`` — #15 registers BOTH B11-vs-B6 (F2) and
B12-vs-B3 (F3, both reuse-ON, 2026-08-02 user call)), family membership rule
= group × metric × dataset × F-slot × UNIT (audit §2.2, engines as members;
the unit axis is the 2026-08-16 owner decision closing assertion G19 — no
Holm family may pool per-query Wilcoxon/McNemar and window-Welch p-values;
the F-slot axis is forced by the same decision's tier-natural upstreams, else
#15's F2/F3 window legs would straddle two gates in one family), correction =
none for gatekept primaries (full α per dataset), Holm within family for
secondaries, BH-FDR for the exploratory tier, holm/tost per fingerprint
sub-hypothesis. Contrasts with pinned ``metrics`` (the #12/#14 estimand
variables) ignore the caller's metric pair; caller metrics must come from the
``REGISTERED_METRICS`` roster (assertion G7 — the map compiles no row for a
metric name the driver does not register). Every gated secondary row carries
its REGISTERED ``upstream`` chain endpoint (2026-08-16 owner decision closing
assertion G10 — tier-natural parents: per-query secondaries gate on the #4
headline, F2/pressure window secondaries on #14 truth-tax, F3/reuse + DIST
secondaries on #13 fingerprint); primaries, falsification and exploratory
rows carry the ``UNGATED`` sentinel (a string, so the §9.13 renderer never
meets a null cell). No test runs that is not a row in this table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import pandas as pd

from src.analysis.cellspec import BASELINES, Family

Tier = Literal["primary", "secondary", "exploratory", "falsification"]
Sidedness = Literal["one-sided", "two-sided", "two one-sided (TOST)"]
Unit = Literal["per_query", "binary", "window"]
#: The Holm-family UNIT axis (2026-08-16 decision, G19): 'binary' rows share
#: the per-query pairing unit (McNemar pairs the same §9.4 per-query draws
#: Wilcoxon does), so the axis collapses to per_query vs window.
FamilyUnit = Literal["per_query", "window"]
Group = Literal["A", "B", "C", "D"]

CAMPAIGN_GROUPS: tuple[Group, ...] = ("A", "B", "C", "D")
# §5/D5 quality-instrumented datasets; SCBench/RULER/ShareGPT are instruments
# or external-validation slices, never family-map rows.
KNOWN_DATASETS: frozenset[str] = frozenset(
    {"squad_v2", "hotpotqa", "musique", "qasper"}
)
# §9.1 co-primary metric pair (audit §2.1): serving = paired TTFT delta,
# quality = the §8.5 per-dataset Y predicate. 2026-08-16 (assertion G7):
# metric names are unified ON THE DRIVER'S namespace ('ttft_ms', the
# run_campaign_analysis.HIGHER_IS_BETTER key) — the old registered-but-never-
# exercised 'ttft' spelling is gone, so the registered pair IS what runs.
SERVING_METRIC: str = "ttft_ms"
PREDICATE_METRIC: str = "predicate"
DEFAULT_METRICS: tuple[str, ...] = (SERVING_METRIC, PREDICATE_METRIC)

# G7 roster guard (2026-08-16): ``compile_family_map`` refuses caller metric
# names outside this roster, so the map can never register a test on a column
# the driver would refuse (or silently drift from the driver's namespace).
# DUPLICATED from the driver's HIGHER_IS_BETTER registry
# (scripts/4_analysis/run_campaign_analysis.py) because src/ must not import
# scripts/; the duplication is pinned by a cross-check test
# (tests/test_stats_engine.py — roster == driver HIGHER_IS_BETTER keys).
# Pinned estimand variables (#12/#14: lambda_star_onset, truth_tax) and the
# #13 'fingerprint' pseudo-metric are registry-internal, never caller-supplied,
# and therefore live outside the caller roster.
REGISTERED_METRICS: frozenset[str] = frozenset(
    {
        # serving (lower is better in the driver's direction registry)
        "ttft_ms",
        "latency_ms",
        "tpot_ms",
        "e2e_ms",
        # quality (higher is better)
        "f1_score",
        "exact_match",
        "f1_answerable",
        "exact_match_answerable",
        "grounding_score",
        "faithfulness",
        "context_relevance",
        "completeness_bertscore",
        "completeness_rouge_l",
        "predicate",
        "goodput_frac",
        "yield_frac",
    }
)
# 2026-08-07 owner decision (pre-freeze charter edit, ADR-0087): per-query
# continuous faithfulness is DEMOTED from the confirmatory tier (the §9.6
# power-sim honesty guard refused every additive-shift model on the real
# tie-heavy paired diffs) and instead rides the EXPLORATORY tier: one
# BH-FDR-corrected, ungated faithfulness row per generic per_query contrast
# leg. Confirmatory quality claims ride the §8.5 predicate (which on Qasper
# IS faithfulness binarized at the registered τ) and the window-level rows.
EXPLORATORY_FAITHFULNESS_METRIC: str = "faithfulness"

_CORRECTION_BY_TIER: dict[str, str] = {
    "primary": "none",
    "secondary": "holm",
    "exploratory": "bh-fdr",
    # §9.2 exile: the floor suite is a standalone falsification section — it
    # spends no α, so nothing corrects it; outcomes are labeled, never pooled.
    "falsification": "none",
}

FINGERPRINT_CONTRAST_ID: int = 13
FLOOR_SUITE_CONTRAST_ID: int = 12
# §9.3: "Fingerprint table (§8.11) decomposed into 6 sub-hypotheses: Holm for
# the 3 superiority predictions; TOST for the 3 pre-registered NONE
# predictions" (conditional policy-event population, §9.5). Entered in the
# chain as ONE intersection hypothesis under contrast #13.
# (policy, correction, sidedness, predicted fingerprint)
FINGERPRINT_SUB_HYPOTHESES: tuple[tuple[str, str, Sidedness, str], ...] = (
    ("evict", "holm", "one-sided", "context-loss hallucination"),
    ("compress", "holm", "one-sided", "evidence destruction (dual-reference delta)"),
    ("truncate", "holm", "one-sided", "abstention-shift or fabrication"),
    ("recompute", "tost", "two one-sided (TOST)", "NONE — latency-only"),
    ("offload", "tost", "two one-sided (TOST)", "NONE — transfer latency only"),
    ("distribute", "tost", "two one-sided (TOST)", "NONE — protocol/wire cost only"),
)


class FamilyMapError(ValueError):
    """Invalid registry content or ``compile_family_map`` input (fail closed)."""


@dataclass(frozen=True)
class ContrastLeg:
    """One additional registered comparison under an existing contrast number.

    §7.8 #15 is the only user: it registers BOTH B11-vs-B6 (F2) and
    B12-vs-B3 (F3) under one contrast slot ("truncation ratio").
    """

    baseline_a: str
    baseline_b: str
    slot: str
    family: Family
    groups: tuple[Group, ...]
    notes: str = ""


@dataclass(frozen=True)
class Contrast:
    """One registered §7.8 contrast — cells differing in exactly ONE slot.

    ``metrics=None`` means the compile-time default metric pair applies;
    a pinned tuple registers the contrast's own estimand variable(s) (§9.2)
    and the caller's metrics never touch it.
    """

    id: int
    name: str
    baseline_a: str | None
    baseline_b: str | None
    slot: str
    family: Family
    tier: Tier
    sidedness: Sidedness
    unit: Unit
    gatekept: bool
    groups: tuple[Group, ...]
    notes: str = ""
    metrics: tuple[str, ...] | None = None
    extra_legs: tuple[ContrastLeg, ...] = ()


def _validated(contrasts: tuple[Contrast, ...]) -> tuple[Contrast, ...]:
    ids = [c.id for c in contrasts]
    if ids != list(range(1, 21)):
        raise FamilyMapError(
            f"registry must hold exactly contrasts 1-20 in order (#21 is future "
            f"work, §7.8); got ids {ids}"
        )
    for c in contrasts:
        legs: list[tuple[str | None, str | None, tuple[Group, ...]]] = [
            (c.baseline_a, c.baseline_b, c.groups)
        ]
        legs.extend((leg.baseline_a, leg.baseline_b, leg.groups) for leg in c.extra_legs)
        for a, b, groups in legs:
            for bl in (a, b):
                if bl is not None and bl not in BASELINES:
                    raise FamilyMapError(f"contrast #{c.id}: unknown baseline {bl!r}")
            if not groups:
                raise FamilyMapError(f"contrast #{c.id}: no carrying group (§7.6.1)")
            unknown = set(groups) - set(CAMPAIGN_GROUPS)
            if unknown:
                raise FamilyMapError(
                    f"contrast #{c.id}: unknown groups {sorted(unknown)}"
                )
        if c.tier == "primary" and c.id not in (4, 13, 14):
            raise FamilyMapError(
                f"contrast #{c.id}: only the three §9.1 chain primaries "
                f"(4/13/14) may carry tier='primary' — the floor suite is "
                f"exiled (§9.2, tier='falsification')"
            )
        if c.tier == "falsification" and c.id != FLOOR_SUITE_CONTRAST_ID:
            raise FamilyMapError(
                f"contrast #{c.id}: only the floor-±15% suite (#12) carries "
                f"tier='falsification' (§9.2)"
            )
        if c.gatekept and c.tier in ("exploratory", "falsification"):
            raise FamilyMapError(
                f"contrast #{c.id}: the {c.tier} tier is outside the "
                f"confirmatory chain (§9.2/§9.3)"
            )
        if c.metrics is not None and not c.metrics:
            raise FamilyMapError(f"contrast #{c.id}: pinned metrics must be non-empty")
    return contrasts


CONTRASTS: tuple[Contrast, ...] = _validated(
    (
        Contrast(1, "What reuse buys on classic QA", "B1", "B2", "arm (reuse bit)",
                 "F1", "secondary", "one-sided", "per_query", True,
                 ("A", "B", "C", "D"),
                 "quality identical by construction at T=0 (identity gate); "
                 "serving delta is the claim"),
        Contrast(2, "The CAG pair: reuse's ceiling on shared context", "B4", "B3",
                 "arm (reuse bit)", "F1", "secondary", "one-sided", "per_query",
                 True, ("A", "B", "C", "D")),
        Contrast(3, "Retrieval's price vs oracle context", "B1", "B6",
                 "arm (context source)", "F1", "secondary", "one-sided",
                 "per_query", True, ("A", "B", "C", "D")),
        Contrast(4, "RAG vs CAG (headline)", "B6", "B3", "arm (context source)",
                 "F1", "primary", "two-sided", "per_query", True,
                 ("A", "B", "C", "D"),
                 "per-dataset co-primary SET (§9.1); pooling PROHIBITED — the "
                 "pilot proved direction inversion across workloads"),
        Contrast(5, "The ranking ablation", "B5", "B6", "retriever (rerank)",
                 "F1", "secondary", "one-sided", "per_query", True,
                 ("A", "B", "C", "D"), "BERGEN monotone chain: B5 < B6"),
        Contrast(6, "Dense vs BM25 (data-gated)", "B5", None,
                 "retriever (dense vs bm25)", "F1", "secondary", "two-sided",
                 "per_query", True, ("A",),
                 "fires only if the §7.2 offline gate shows a ≥5pp pool-"
                 "recall@100 gap; anchor model only"),
        Contrast(7, "Text compression's price", "B9", "B6", "arm (compression)",
                 "F1", "secondary", "one-sided", "per_query", True,
                 ("A", "B", "C", "D")),
        Contrast(8, "Does compression stack with reuse", "B10", "B3",
                 "arm (compression)", "F1", "secondary", "two-sided", "per_query",
                 True, ("A", "B", "C", "D"), "the no-precedent cell"),
        Contrast(9, "What the external KV store buys", "B8", "B6",
                 "arm (KV store)", "F1", "secondary", "two-sided", "per_query",
                 True, ("A", "B", "C", "D")),
        Contrast(10, "Four reuse mechanisms head-to-head", "B3", None, "engine",
                 "F1", "secondary", "two-sided", "per_query", True, ("A", "B"),
                 "engines as family members; HF = idea-gain zero point"),
        Contrast(11, "The locality law across datasets", None, None, "dataset",
                 "F1", "exploratory", "two-sided", "per_query", False,
                 ("A", "B"),
                 "curve-collapse claim (same arm, datasets as the dial); "
                 "descriptive by design"),
        Contrast(12, "The pressure curve + floor-±15% suite", None, None,
                 "budget_r × rate_frac grid (same cell)", "F2", "falsification",
                 "two-sided", "window", False, ("A", "B", "C", "D"),
                 "§9.2 EXILE: standalone falsification suite, OUT of the "
                 "primary chain; ×/÷1.15 band; INCONCLUSIVE-AT-RESOLUTION "
                 "label applies",
                 metrics=("lambda_star_onset",)),
        Contrast(13, "The within-vLLM coping frontier (fingerprints)", None,
                 None, "policy (recompute vs offload vs compress-fp8)", "F2",
                 "primary", "one-sided", "window", True, ("A", "B", "C", "D"),
                 "carries the §8.11 fingerprint table: decomposed into 3 "
                 "superiority (Holm) + 3 NONE (TOST, conditional population) "
                 "sub-hypotheses per §9.3"),
        Contrast(14, "Cross-engine policy bundles in serving yield Y", None,
                 None, "engine (policy bundle) at normalized pressure", "F2",
                 "primary", "one-sided", "window", True, ("A", "B", "C", "D"),
                 "carries the §9.2 truth-tax estimand: population = in-regime "
                 "cells, variable = G − Y, batch-means contrast",
                 metrics=("truth_tax",)),
        Contrast(15, "Truncation priced", "B11", "B6", "arm (truncation)", "F2",
                 "secondary", "one-sided", "window", True, ("A", "B"),
                 "latency saved vs truth lost (RAG-side, F2)",
                 extra_legs=(
                     ContrastLeg(
                         "B12", "B3", "arm (truncation ratio)", "F3",
                         ("A", "B"),
                         "stored-corpus truncation priced under prefix-ON "
                         "pressure; both reuse-ON (B12 bit resolved "
                         "2026-08-02) — one slot, truncation ratio",
                     ),
                 )),
        Contrast(16, "HF's OOM wall (motivation)", None, None,
                 "capacity boundary (unmanaged HF serving)", "F2",
                 "exploratory", "two-sided", "window", False, ("A", "B"),
                 "descriptive measurement — no hypothesis test"),
        Contrast(17, "Does eviction eat the shared cache", "B3", None,
                 "budget_r (rising pressure)", "F3", "secondary", "one-sided",
                 "window", True, ("A", "B", "C", "D"),
                 "per eviction geometry (radix LRU vs block eviction)"),
        Contrast(18, "Distribution's buy-back at transfer price", None, None,
                 "topology (single-pressured vs pd, declared iso-basis)",
                 "DIST", "secondary", "two-sided", "window", True, ("C", "D")),
        Contrast(19, "Does dedup survive the wire", "B3", None,
                 "pd hit-accounting (before/after the wire)", "DIST",
                 "exploratory", "two-sided", "window", False, ("C", "D"),
                 "§3.2 measurement target — either answer is a finding"),
        Contrast(20, "MLA×TP replication on V3", None, None,
                 "topology (tp degree)", "DIST", "secondary", "one-sided",
                 "window", True, ("D",),
                 "pure-TP protocol pinned (dp-attention = one labeled "
                 "exploratory observation cell, outside this map)"),
    )
)

HEADLINE_CONTRAST_ID: int = 4
# §9.1 (adopted 2026-08-02): THREE chain primaries — the floor suite (#12) is
# exiled to tier='falsification' and is NOT a primary endpoint.
PRIMARY_IDS: frozenset[int] = frozenset({4, 13, 14})
# The declared Dmitrienko SERIAL fixed sequence (audit §2.1: "Declare the
# fixed-sequence order of primaries explicitly"): headline co-primary set →
# truth-tax estimand → fingerprint intersection hypothesis. Consumed by
# ``gatekeeping.evaluate_chain(primary_order=...)``.
PRIMARY_CHAIN_ORDER: tuple[int, ...] = (4, 14, 13)


def chain_endpoint(contrast_id: int) -> str:
    """Endpoint name of a chain primary — the driver's naming (contrast-<id>)."""
    return f"contrast-{contrast_id}"


#: The registered chain endpoints — the only legal gating ``upstream`` values.
CHAIN_ENDPOINTS: frozenset[str] = frozenset(
    chain_endpoint(cid) for cid in PRIMARY_CHAIN_ORDER
)

#: ``upstream`` sentinel for rows outside the gated-secondary tier
#: (primary/falsification/exploratory). A STRING, not a null, for the same
#: reason ``correction`` uses the string "none": the §9.13 renderer
#: (``prereg._markdown_table``) cannot render nulls under the pinned pandas
#: (astype(str) keeps NaN), and a registration table cell must never be
#: blank. Semantics are exactly "no upstream gate".
UNGATED: str = "ungated"

# Registered gating topology (owner decision d, 2026-08-16; closes assertion
# G10 — the driver may no longer hard-code "everything gates on #4").
# Tier-natural parents:
# - per-query secondaries          → #4  (headline co-primary set)
# - F2/pressure window secondaries → #14 (truth-tax estimand)
# - F3/reuse + DIST secondaries    → #13 (fingerprint intersection)
PER_QUERY_SECONDARY_UPSTREAM: str = chain_endpoint(HEADLINE_CONTRAST_ID)
WINDOW_SECONDARY_UPSTREAM: dict[Family, str] = {
    "F2": chain_endpoint(14),
    "F3": chain_endpoint(FINGERPRINT_CONTRAST_ID),
    "DIST": chain_endpoint(FINGERPRINT_CONTRAST_ID),
}

# Import-time validation: every registered upstream must be a registered chain
# endpoint — a typo here would wire a family to a gate that can never open.
for _upstream_endpoint in (
    PER_QUERY_SECONDARY_UPSTREAM,
    *WINDOW_SECONDARY_UPSTREAM.values(),
):
    if _upstream_endpoint not in CHAIN_ENDPOINTS:
        raise FamilyMapError(
            f"registered upstream {_upstream_endpoint!r} is not a chain "
            f"endpoint {sorted(CHAIN_ENDPOINTS)} (§9.3 gating topology)"
        )
del _upstream_endpoint


def family_unit_of(unit: str) -> FamilyUnit:
    """Collapse a row's §9.4 test unit onto the Holm-family UNIT axis.

    ``binary`` shares the per-query pairing unit (a McNemar row pairs the
    same per-query draws a Wilcoxon row does); ``window`` is the batch-means
    unit. The family_id carries this axis so no Holm family can pool
    per-query and window p-values (2026-08-16 decision, assertion G19).
    """
    if unit == "window":
        return "window"
    if unit in ("per_query", "binary"):
        return "per_query"
    raise FamilyMapError(f"unknown test unit {unit!r} (per_query|binary|window)")


def registered_upstream(tier: str, family: Family, unit: str) -> str:
    """The registered chain endpoint gating a row; ``UNGATED`` otherwise.

    Only secondary-tier rows are gated on an upstream primary (§9.3);
    primaries gate each other through the serial order, the falsification
    suite spends no α, and the exploratory tier is ungated by construction.
    """
    if tier != "secondary":
        return UNGATED
    if family_unit_of(unit) == "per_query":
        return PER_QUERY_SECONDARY_UPSTREAM
    endpoint = WINDOW_SECONDARY_UPSTREAM.get(family)
    if endpoint is None:
        raise FamilyMapError(
            f"window-unit secondary in family {family!r} has no registered "
            f"upstream (§9.3 gating topology, 2026-08-16 decision)"
        )
    return endpoint


def compile_family_map(
    datasets: Sequence[str],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Expand the registry into the §9.3 registered test table.

    One row per contrast leg × carrying group × metric × dataset. Contrast
    #11's varied slot IS the dataset axis, so it emits a single
    ``cross-dataset`` row per group × metric. Unit per row follows §9.4:
    sub-pressure predicate rows are binary (McNemar); other sub-pressure rows
    per-query Wilcoxon; every pressure-family row is window-level batch means.

    ``metrics`` applies only to contrasts without pinned ``metrics`` (#12/#14
    register their §9.2 estimand variables instead) and must come from the
    ``REGISTERED_METRICS`` roster (G7 guard — the table must not follow the
    caller into an unregistered metric namespace). Contrast #13 expands into
    the six §9.3 fingerprint sub-hypothesis rows (3 Holm superiority + 3
    conditional-TOST NONE), never into the generic metric pair. Every row
    carries ``comparison`` (the cells compared), ``sub_hypothesis``
    (fingerprint rows only; empty elsewhere), ``family_unit`` (the Holm-family
    UNIT axis embedded in ``family_id`` — decision d 2026-08-16/G19) and
    ``upstream`` (the registered gating endpoint for secondary rows; the
    ``UNGATED`` sentinel on primary/falsification/exploratory rows —
    decision d 2026-08-16/G10).
    """
    if not datasets:
        raise FamilyMapError("datasets must be non-empty")
    if len(set(datasets)) != len(datasets):
        raise FamilyMapError(f"duplicate datasets in {list(datasets)}")
    unknown = set(datasets) - KNOWN_DATASETS
    if unknown:
        raise FamilyMapError(
            f"unknown datasets {sorted(unknown)}; charter datasets: "
            f"{sorted(KNOWN_DATASETS)}"
        )
    if not metrics:
        raise FamilyMapError("metrics must be non-empty")
    unregistered = set(metrics) - REGISTERED_METRICS
    if unregistered:
        raise FamilyMapError(
            f"unregistered metric names {sorted(unregistered)} (G7 roster "
            f"guard): the §9.3 map only compiles rows for the registered "
            f"roster (= the driver's HIGHER_IS_BETTER keys) "
            f"{sorted(REGISTERED_METRICS)}"
        )
    if not 0.0 < alpha < 1.0:
        raise FamilyMapError(f"alpha={alpha} must be in (0, 1)")

    rows: list[dict[str, object]] = []

    def emit(
        c: Contrast,
        *,
        family: Family,
        group: Group,
        dataset: str,
        comparison: str,
        metric: str,
        unit: str,
        correction: str,
        sidedness: str,
        sub_hypothesis: str,
        notes: str,
        tier: str | None = None,
        gatekept: bool | None = None,
    ) -> None:
        row_tier = c.tier if tier is None else tier
        f_unit = family_unit_of(unit)
        rows.append(
            {
                "contrast_id": c.id,
                "name": c.name,
                "tier": row_tier,
                "gatekept": c.gatekept if gatekept is None else gatekept,
                "family": family,
                "group": group,
                "metric": metric,
                "dataset": dataset,
                "comparison": comparison,
                "sub_hypothesis": sub_hypothesis,
                # Family membership = group × metric × dataset × F-slot ×
                # UNIT axis (decision d 2026-08-16/G19): the old 3-axis id
                # pooled per-query Wilcoxon and window-Welch p-values in one
                # Holm family (the pre-split m=12 family). The F-slot axis is
                # forced by the same decision's tier-natural upstreams: #15's
                # F2 leg gates on #14 while its F3 leg (and #17) gate on #13,
                # so a group|metric|dataset|window family would straddle two
                # gates — a §9.3 map error `_validate_compiled` refuses.
                "family_id": f"{group}|{metric}|{dataset}|{family}|{f_unit}",
                "family_unit": f_unit,
                "upstream": registered_upstream(row_tier, family, unit),
                "correction": correction,
                "sidedness": sidedness,
                "unit": unit,
                "alpha": alpha,
                "notes": notes,
            }
        )

    for c in CONTRASTS:
        leg_specs: list[tuple[str | None, str | None, str, Family, tuple[Group, ...], str]] = [
            (c.baseline_a, c.baseline_b, c.slot, c.family, c.groups, c.notes)
        ]
        leg_specs.extend(
            (leg.baseline_a, leg.baseline_b, leg.slot, leg.family, leg.groups,
             leg.notes or c.notes)
            for leg in c.extra_legs
        )
        for a, b, slot, family, groups, notes in leg_specs:
            comparison = f"{a} vs {b}" if a is not None and b is not None else slot
            row_datasets: tuple[str, ...] = (
                ("cross-dataset",) if slot == "dataset" else tuple(datasets)
            )
            for group in groups:
                for dataset in row_datasets:
                    if c.id == FINGERPRINT_CONTRAST_ID:
                        # §9.3 decomposition — the six registered rows ARE the
                        # tests that run; the generic metric pair never applies.
                        for policy, correction, sidedness, predicted in (
                            FINGERPRINT_SUB_HYPOTHESES
                        ):
                            emit(
                                c, family=family, group=group, dataset=dataset,
                                comparison=comparison,
                                metric="fingerprint",
                                unit=c.unit,
                                correction=correction,
                                sidedness=sidedness,
                                sub_hypothesis=f"{policy}: {predicted}",
                                notes=notes,
                            )
                        continue
                    for metric in (c.metrics if c.metrics is not None else tuple(metrics)):
                        unit = (
                            "binary"
                            if c.unit == "per_query" and metric == PREDICATE_METRIC
                            else c.unit
                        )
                        emit(
                            c, family=family, group=group, dataset=dataset,
                            comparison=comparison,
                            metric=metric,
                            unit=unit,
                            correction=_CORRECTION_BY_TIER[c.tier],
                            sidedness=c.sidedness,
                            sub_hypothesis="",
                            notes=notes,
                        )
                    if c.metrics is None and c.unit == "per_query":
                        # 2026-08-07 amendment (ADR-0087): exploratory
                        # per-query faithfulness row — BH-FDR, ungated,
                        # two-sided. Labeled p-values only; the claim ladder
                        # bars confirmatory sentences on this row.
                        emit(
                            c, family=family, group=group, dataset=dataset,
                            comparison=comparison,
                            metric=EXPLORATORY_FAITHFULNESS_METRIC,
                            unit="per_query",
                            correction=_CORRECTION_BY_TIER["exploratory"],
                            sidedness="two-sided",
                            sub_hypothesis="",
                            notes=(
                                "exploratory tier (2026-08-07 demotion "
                                "decision): guard-refused for powered "
                                "superiority; descriptive + labeled "
                                "exploratory only. " + notes
                            ).strip(),
                            tier="exploratory",
                            gatekept=False,
                        )
    frame = pd.DataFrame(rows)
    _validate_compiled(frame)
    return frame


def _validate_compiled(frame: pd.DataFrame) -> None:
    """Fail-closed structural invariants of the compiled §9.3 table.

    Redundant with construction by design (the emit path derives both axes),
    so a future edit that breaks either registered invariant fails HERE, not
    in a downstream Holm pool.
    """
    # Decision d 2026-08-16 (G19): one Holm family = one §9.4 unit — a family
    # must never pool per-query (Wilcoxon/McNemar) and window (Welch) rows.
    mixed = frame.groupby("family_id")["family_unit"].nunique()
    mixed = mixed[mixed > 1]
    if not mixed.empty:
        raise FamilyMapError(
            f"families mixing units: {sorted(mixed.index)} — one Holm family "
            f"= one §9.4 unit (2026-08-16 decision, G19)"
        )
    if frame["upstream"].isna().any():
        raise FamilyMapError(
            "null upstream cells — every row carries a chain endpoint or the "
            "UNGATED sentinel (§9.3/G10); a null would blank a registration "
            "table cell"
        )
    gated = frame.loc[frame["upstream"] != UNGATED]
    unknown_upstreams = set(gated["upstream"]) - set(CHAIN_ENDPOINTS)
    if unknown_upstreams:
        raise FamilyMapError(
            f"upstream values {sorted(unknown_upstreams)} are not registered "
            f"chain endpoints {sorted(CHAIN_ENDPOINTS)} (§9.3)"
        )
    non_secondary = gated.loc[gated["tier"] != "secondary"]
    if not non_secondary.empty:
        raise FamilyMapError(
            f"non-secondary rows carry an upstream gate (tiers "
            f"{sorted(set(non_secondary['tier']))}) — only secondaries are "
            f"gated on a chain primary (§9.3)"
        )
    straddling = gated.groupby("family_id")["upstream"].nunique()
    straddling = straddling[straddling > 1]
    if not straddling.empty:
        raise FamilyMapError(
            f"families straddling upstream gates: {sorted(straddling.index)} "
            f"— one family = one gate (§9.3)"
        )
