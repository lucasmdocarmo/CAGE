"""§9.13 — PRE_REGISTRATION.md assembler (registration rung (b): repo SHA + OSF).

Assembles the registration document TEXT from the registered inputs: the §9.3
family-map table, the §9.5 equivalence margins, the §9.7 calibration report and
the machinery git SHA. Pure function — it never writes anywhere; the caller
owns the output path (and the commit that pins the SHA).

Charter gates enforced here (fail loud, §9.13): registration is BLOCKED until
calibration passes, so assembling from a failing ``CalibrationReport`` raises;
a malformed family map (the table D9 compiles every test from) raises.
Deterministic given inputs — no wall-clock content; the registration date is
carried by the commit + OSF timestamp, not by this text.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import pandas as pd

from src.analysis.stats.calibration import CalibrationReport

# The §9.3 registered-table schema = ``families.compile_family_map`` output
# (2026-08-02 fix: the assembler previously demanded a 'contrast' column no
# compiler emitted, so the §9.3→§9.13 pipeline did not compose).
FAMILY_MAP_COLUMNS: tuple[str, ...] = (
    "contrast_id",
    "name",
    "tier",
    "family",
    "group",
    "metric",
    "dataset",
    "correction",
    "sidedness",
    "unit",
    "alpha",
)
_TIERS = frozenset({"primary", "secondary", "exploratory", "falsification"})
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")

# §9.11 procedural clauses + §9.1/§9.2 primaries — registered verbatim, so the
# text lives here as constants, not caller input.
_PRIMARIES_TEXT = """\
- **Headline contrast — B6-vs-B3 (RAG-ranked vs CAG), a per-dataset co-primary SET
  (§9.1):** tested separately on each dataset with one pre-declared metric per dataset
  (serving: paired TTFT delta; quality: the §8.5 per-dataset Y predicate). **Pooling
  across datasets is PROHIBITED** — the pilot proved direction inversion, so the
  locality gradient IS the hypothesis; sidedness is declared per dataset cell in the
  family map below.
- **Co-primary — the Y truth tax, as an ICH-E9(R1)-style estimand (§9.2):**
  population = in-regime cells (D6 §6.1 three-layer definition); variable = G − Y
  (goodput minus serving yield); population summary = batch-means contrast across
  windows, with a named comparator and declared direction.
- **Fingerprint table (§8.11), decomposed into 6 sub-hypotheses (§9.3):** Holm for the
  3 superiority predictions; TOST for the 3 pre-registered NONE predictions.
- **Floor-±15% falsification suite: OUT of the confirmatory chain (§9.2)** —
  registered as a standalone falsification section (publishable in either direction;
  it must not spend or kill downstream α). Onset = interpolated Chiu-Jain power-metric
  argmax; band multiplicative ×/÷1.15; grid-resolution misses are labeled
  INCONCLUSIVE-AT-RESOLUTION; λ* = min(λ_KV, λ_compute) from a calibrated roofline
  service-time model.
"""

_ONE_LOOK_TEXT = """\
The pilot-archive dry-run is the only peek; confirmatory campaign data is analyzed
exactly once, after the campaign completes (§9.11). Budget exhaustion follows the
pre-declared cell priority ranking — no post-hoc selection of a partial dataset.
"""


class PreregError(ValueError):
    """Registration input violates the D9 charter (fail closed)."""


def _markdown_table(df: pd.DataFrame) -> str:
    # No tabulate dependency in this repo — render the pipe table directly.
    # Literal pipes in cell values (family_id is 'group|metric|dataset') are
    # escaped, else they would silently split table cells (2026-08-02 fix).
    cols = list(df.columns)
    cells = df.astype(str).apply(lambda s: s.str.replace("|", r"\|", regex=False))
    header = "| " + " | ".join(c.replace("|", r"\|") for c in cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    rows = cells[cols[0]]
    if len(cols) > 1:
        rows = rows.str.cat([cells[c] for c in cols[1:]], sep=" | ")
    return "\n".join([header, rule, *("| " + r + " |" for r in rows)])


def _validate_family_map(family_map: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FAMILY_MAP_COLUMNS if c not in family_map.columns]
    if missing:
        raise PreregError(f"family map missing required columns {missing} (§9.3)")
    if family_map.empty:
        raise PreregError("family map is empty — no test may run without a row (§9.3)")
    if family_map[list(FAMILY_MAP_COLUMNS)].isna().any().any():
        raise PreregError("family map contains missing cells (§9.3: every row complete)")
    bad_tiers = set(family_map["tier"].astype(str)) - _TIERS
    if bad_tiers:
        raise PreregError(f"unknown tiers {sorted(bad_tiers)}; allowed: {sorted(_TIERS)}")
    alphas = pd.to_numeric(family_map["alpha"], errors="coerce")
    if alphas.isna().any() or not ((alphas > 0.0) & (alphas < 1.0)).all():
        raise PreregError("family map alpha values must be numeric in (0, 1)")
    return family_map


def assemble_preregistration(
    family_map: pd.DataFrame,
    margins: Mapping[str, str],
    calibration_report: CalibrationReport,
    git_sha: str,
    *,
    power_table: pd.DataFrame | None = None,
    extra_exclusions: Sequence[str] = (),
    amendment_log_path: str = "MyDocs/registration/AMENDMENT_LOG.md",
) -> str:
    """Return the PRE_REGISTRATION.md text (§9.13). Caller writes it to disk.

    ``family_map`` is ``families.compile_family_map`` output (the §9.3
    registered table) — the two halves of the §9.3→§9.13 pipeline compose
    directly. ``margins`` maps endpoint/family -> its registered domain margin statement
    (§9.5 layer 1); the distribution-free |Cliff's δ| < 0.147 companion bound is
    emitted for every entry. ``power_table`` is ``simulate_campaign`` output;
    absent, the power section carries a BLOCKING placeholder — the document is
    assemblable for review but explicitly not registrable (§9.6 sets N).
    """
    _validate_family_map(family_map)
    if not margins:
        raise PreregError("margins are registration content (§9.5) — none provided")
    if not _GIT_SHA_RE.fullmatch(git_sha):
        raise PreregError(f"git_sha {git_sha!r} is not a 7-64 char lowercase hex SHA")
    if not calibration_report.aa.approximates_nominal:
        raise PreregError(
            "calibration A/A FAILED (FP rate CI excludes nominal α) — "
            "registration is BLOCKED until the machinery passes §9.7"
        )
    failed = [i for i in calibration_report.injections if i.meets_target is False]
    if failed:
        raise PreregError(
            f"calibration injection targets missed at effects "
            f"{[i.effect_size for i in failed]} — registration is BLOCKED (§9.7/§9.13)"
        )

    margin_rows = pd.DataFrame(
        {
            "endpoint": list(margins),
            "domain margin (metric units)": [margins[k] for k in margins],
            "distribution-free companion": ["|Cliff's δ| < 0.147 (negligible)"] * len(margins),
            "population": ["CONDITIONAL policy-event population, tie-aware (§9.5)"]
            * len(margins),
        }
    )
    if power_table is not None:
        power_section = (
            "Simulation-based (§9.6); code + seeds registered at the SHA above. The\n"
            "table below IS the per-window query count / window count decision input\n"
            "(0.8 power at the pre-declared MDEs).\n\n" + _markdown_table(power_table)
        )
    else:
        power_section = (
            "[BLOCKING — simulation power table absent: §9.6 output sets the missing N; "
            "this document must not be registered until it is filled.]"
        )

    exclusion_lines = [
        "- Engine-defect adjudication (§9.12): reproducible engine bug -> cell-level "
        "exclusion from QUALITY families with a public evidence trail (issue link, "
        "repro); serving metrics may still be reported, labeled.",
        "- In-regime labels are analysis-population definitions (§6.1): UNPRESSURED "
        "and PAST-CLIFF cells stay valid, labeled grid points, excluded from "
        "in-regime aggregates.",
        "- Y counts non-completions as non-veridical; quality-given-completion is "
        "always labeled conditional (§8.10 ceiling discipline).",
        *(f"- {rule}" for rule in extra_exclusions),
    ]

    sections = [
        "# PRE_REGISTRATION — CAGE campaign (D9, PUBLICATION.md §9)",
        "",
        f"Machinery SHA: `{git_sha}` (registered rung (b): repo SHA + OSF timestamp; "
        "the SHA below already executed the calibration suite embedded here).",
        "",
        "## 1. Primary endpoints",
        "",
        _PRIMARIES_TEXT,
        "## 2. Family map (§9.3 — no test runs that is not a row here)",
        "",
        _markdown_table(family_map),
        "",
        "Corrections: Holm within family for superiority claims; BH-FDR for the "
        "exploratory tier. Secondaries are gated on their upstream primary; "
        "otherwise reported descriptively.",
        "",
        "## 3. Equivalence margins (§9.5 — two-layer TOST)",
        "",
        _markdown_table(margin_rows),
        "",
        "Bayesian ROPE (Benavoli signed-rank, "
        "`src.analysis.stats.equivalence.rope_sensitivity`) reported as a "
        "sensitivity line beside every TOST conclusion.",
        "",
        "## 4. Power (§9.6)",
        "",
        power_section,
        "",
        "## 5. Exclusions & adjudication (§9.12)",
        "",
        "\n".join(exclusion_lines),
        "",
        "## 6. One-look policy & budget-exhaustion order (§9.11)",
        "",
        _ONE_LOOK_TEXT,
        "## 7. Calibration report (§9.7 — measured operating characteristics)",
        "",
        calibration_report.to_markdown(),
        "## 8. Amendment log",
        "",
        f"Any mid-campaign protocol change gets a dated, justified entry in "
        f"`{amendment_log_path}` (clinical-trial style); silent drift is a protocol "
        "violation (§9.11).",
        "",
    ]
    return "\n".join(sections)
