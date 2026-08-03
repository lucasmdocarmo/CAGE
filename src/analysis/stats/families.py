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
= group × metric × dataset (audit §2.2, engines as members), correction =
none for gatekept primaries (full α per dataset), Holm within family for
secondaries, BH-FDR for the exploratory tier, holm/tost per fingerprint
sub-hypothesis. Contrasts with pinned ``metrics`` (the #12/#14 estimand
variables) ignore the caller's metric pair. No test runs that is not a row
in this table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import pandas as pd

from src.analysis.cellspec import BASELINES, Family

Tier = Literal["primary", "secondary", "exploratory", "falsification"]
Sidedness = Literal["one-sided", "two-sided", "two one-sided (TOST)"]
Unit = Literal["per_query", "binary", "window"]
Group = Literal["A", "B", "C", "D"]

CAMPAIGN_GROUPS: tuple[Group, ...] = ("A", "B", "C", "D")
# §5/D5 quality-instrumented datasets; SCBench/RULER/ShareGPT are instruments
# or external-validation slices, never family-map rows.
KNOWN_DATASETS: frozenset[str] = frozenset(
    {"squad_v2", "hotpotqa", "musique", "qasper"}
)
# §9.1 co-primary metric pair (audit §2.1): serving = paired TTFT delta,
# quality = the §8.5 per-dataset Y predicate.
SERVING_METRIC: str = "ttft"
PREDICATE_METRIC: str = "predicate"
DEFAULT_METRICS: tuple[str, ...] = (SERVING_METRIC, PREDICATE_METRIC)

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
    register their §9.2 estimand variables instead). Contrast #13 expands into
    the six §9.3 fingerprint sub-hypothesis rows (3 Holm superiority + 3
    conditional-TOST NONE), never into the generic metric pair. Every row
    carries ``comparison`` (the cells compared) and ``sub_hypothesis``
    (fingerprint rows only; empty elsewhere).
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
    ) -> None:
        rows.append(
            {
                "contrast_id": c.id,
                "name": c.name,
                "tier": c.tier,
                "gatekept": c.gatekept,
                "family": family,
                "group": group,
                "metric": metric,
                "dataset": dataset,
                "comparison": comparison,
                "sub_hypothesis": sub_hypothesis,
                "family_id": f"{group}|{metric}|{dataset}",
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
    return pd.DataFrame(rows)
