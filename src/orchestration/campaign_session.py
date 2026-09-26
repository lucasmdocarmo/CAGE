"""Campaign-mode bridge between the per-cell runner and the v2 tree producer.

Task #116 (walkthrough finding K-COV1): ``src/orchestration/campaign_layout.py``
is the RESULTS_LAYOUT-v2 producer but had ZERO production callers —
``scripts/3_run/run_experiment.py`` still wrote only the pilot-style
``<output_dir>/trial_N/{results.csv,metrics.json,qa_evidence.jsonl}`` tree.
This module is the seam that wires the runner INTO the campaign layout:

- **Activation**: ``CampaignCellSession.from_cli`` returns a session when the
  runner was given ``--campaign-root`` (or the ``CAGE_CAMPAIGN_ROOT`` env the
  shell runners export), else ``None`` — the pilot path stays byte-identical.
- **Cell identity** (RESULTS_LAYOUT §2): the row key is minted by
  ``CellSpec.to_row_key()`` — never hand-built. The tuple is derived from the
  runner's own vocabulary: ``--baseline``/``--baseline-label`` through
  ``cellspec.LEGACY_ALIASES`` (arm/retriever/policy/family/topology),
  ``--backend`` through ``ENGINE_OF_BACKEND``, ``--model`` through the D4
  HF-id→slug roster (``CAGE_MODEL_SLUG`` override). ``CAGE_CELL_*`` env vars
  override individual axes (the explicit-axes path for the campaign harness).
  Every derivation is fail-closed: an unmapped name refuses BEFORE any
  dataset/engine work.
- **Window emission**: one runner trial = one §1 measurement window
  ``window_<dataset>-<NN>`` (NN = trial number, %02d; plus the optional
  ``CAGE_WINDOW_ORDINAL_BASE`` shift — W4.4 per-task RULER steps share one
  (row_key, dataset) window space, each claiming its own disjoint ordinal
  range). ``CAGE_GPU_COUNT`` (W4.2) threads the serving stack's GPU count
  into cell.json via ``CellWriter`` for the §6.6b per-GPU basis. The trial's collected
  results rows become ``requests.jsonl`` (every row carrying the #127 join
  triple ``example_id``/``repeat_index``/``record_index`` plus the shared
  ``ok`` validity predicate), the staged ``qa_evidence.jsonl`` is re-emitted
  through the atomic layout writer, the telemetry series becomes
  ``cage_stats.jsonl``, backend metadata + the telemetry aggregate become
  ``engine_metrics.json``, and the pilot ``metrics.json`` experiment summary
  is written INTO the window as an auxiliary artifact — it is the completeness
  sentinel the shell resume gates (``cell_complete``) key on, campaign mode
  included (same ``metrics_json_valid`` rigor).
- **Regime referee** (V8 slice, close-out sheet row 1; §6.1): every emitted
  window gets its ``regime.json`` from ``campaign_layout.write_window_regime``
  at emission time: the measured window bounds, the telemetry the runner
  sampled during the window (``cage_stats.jsonl``) and the completed-over-
  issued attainment of its request rows (the quotient
  ``goodput.evaluate_window`` reports). An empty or short series is recorded
  as ``UNKNOWN_TELEMETRY`` on every cell (a label, never a numeric). A pd
  cell with a non-empty series routes the ``pools_bytes`` split of its cell
  record (the pin, or the value adopted from cell.json on resume) down the
  summed-pool lane; a series carrying two or more distinct roles on any
  other cell, or a pd series missing a budgeted role, RAISES (the writer's
  gates) and fails the cell. Written BEFORE the metrics sentinel, so a failed referee
  leaves an incomplete window for the resume reset, and before the hash
  journal, so the label is a sealed artifact.
- **Manifest** (§3): created once, at the first window emission of the run
  (create-if-absent; `write_manifest` itself refuses amendment), with non-null
  provenance — git SHA/dirty from the repo, seed, engine/engine_version,
  provider/hardware (``CAGE_PROVIDER``/``CAGE_HARDWARE``),
  ``dataset_manifests_sha256`` (env override or the sha256 of the
  ``CAGE_QUERY_MANIFEST`` file), the optional dataset roster
  (``CAGE_CAMPAIGN_DATASETS``), the run-level ``kv_cache_dtype`` and, when
  the cell pins ``CAGE_SLO_FLOORS_JSON`` (Batch 2 W4, ADR-0117), the §6.1
  single-stream floors under ``slo_floors`` (the #14 executor's SLO source;
  a reopened manifest must carry the SAME floors or the cell refuses).
- **Budget record** (W4): ``CAGE_BUDGET_PLAN_JSON`` carries the relaunch's
  cache_budget.BudgetPlan record on budgeted cells; it is cross-checked
  against the cell tuple here and persisted by ``CellWriter`` under
  cell.json ``budget_plan`` (the T2.5 rho_own basis). Absent pins keep the
  pre-W4 path byte-identical.
- **Write-time hash ledger** (S0-15 / §9.10 UPGRADE 5): every emitted artifact
  is sha256-hashed AT WRITE TIME into the append-only run-root journal
  ``write_time_hashes.jsonl``; ``seal_campaign_run`` refuses to seal when any
  current artifact's hash differs from its last journal entry (the file
  changed between write and seal) or was never journaled, THEN calls
  ``campaign_layout.seal_run`` (ledger.json written once at run end, §5 —
  the journal is the write-time half of the §9.10 sentence "content-hashed at
  write time; the ledger is committed immediately post-run").
- **Resume**: ``window_complete``/``reset_incomplete_windows`` give the runner
  per-window resume — complete windows are skipped (their metrics.json is
  loaded for trial aggregation, no re-serving, no duplicated rows);
  incomplete/partial windows (missing or unparseable metrics.json) are
  removed from disk AND undeclared from cell.json's windows[] table so
  ``CellWriter`` re-emits them cleanly.

Blinding boundary (#130): raw serving trees carry arm identity BY DESIGN;
blinding is scoring-side. This module adds no labels beyond what the layout
already specifies.

CLI (kept import-light — ``campaign_layout``/pandas load lazily):
``python3 -m src.orchestration.campaign_session cell-dir --baseline B
--baseline-label L --model M [--backend vllm]`` prints the absolute v2 cell
directory under ``$CAGE_CAMPAIGN_ROOT`` — the shell runners' resume gates
resolve cell dirs through THIS derivation so the gate and the writer can
never disagree on a row key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from src.analysis.cellspec import (
    CellSpec,
    CellSpecError,
    Engine,
    LEGACY_ALIASES,
    Model,
    from_legacy,
)
from typing import get_args as _get_args

from src.analysis.stats.ledger import hash_artifacts
from src.orchestration.cache_budget import KV_DTYPE_FACTOR

__all__ = [
    "BUDGET_PLAN_CELL_KEY",
    "BUDGET_PLAN_ENV",
    "CONSORT_COUNTERS",
    "CampaignCellSession",
    "CampaignSessionError",
    "ENGINE_OF_BACKEND",
    "HF_MODEL_SLUGS",
    "JOURNAL_NAME",
    "SLO_FLOORS_ENV",
    "SLO_FLOORS_MANIFEST_KEY",
    "append_write_time_hashes",
    "derive_cell_spec",
    "parse_budget_plan",
    "parse_slo_floors",
    "read_write_time_journal",
    "refuse_dropped_rows_summary",
    "resolve_model_slug",
    "seal_campaign_run",
    "validate_budget_plan",
    "validate_slo_floors",
]

#: Append-only write-time hash journal at the run root (S0-15). Lives BESIDE
#: manifest.json/ledger.json (never under cells/), so the §5 seal and the H7
#: EXTRA sweep — both scoped to cells/ + manifest.json — never flag it.
JOURNAL_NAME = "write_time_hashes.jsonl"

#: The pilot metrics.json experiment summary, re-emitted INTO each window as an
#: auxiliary artifact (organize_results indexes extra window *.json files); it
#: is the completeness sentinel the shell cell_complete gates parse.
WINDOW_METRICS_NAME = "metrics.json"

#: metrics.json["consort"] counters (run_experiment, task #127) that must ALL
#: read zero on a campaign window (ADR-0116, Batch 2 W3): a nonzero count is a
#: request the window's artifacts do not carry, invisible to the
#: requests-vs-evidence reconciliation because it is absent from both chains.
CONSORT_COUNTERS: tuple[str, ...] = (
    "n_dropped_prepare",
    "n_dropped_record",
    "n_dropped_turn",
    "evidence_write_failures",
)

#: Batch 2 W4 (ADR-0117): the two pins run_campaign threads through every
#: cell step's env (mirrored literals; tests/test_campaign_session.py pins
#: them equal to the driver's and to the analysis consumer's keys).
#: SLO_FLOORS_ENV carries the §6.1 single-stream floors as JSON
#: ({engine: {ttft_s, tpot_s, ...}}, seconds); the session writes them into
#: manifest.json[SLO_FLOORS_MANIFEST_KEY] when it creates the manifest and
#: refuses a reopened manifest whose floors differ (§3: amended never).
#: BUDGET_PLAN_ENV carries the relaunch's cache_budget.BudgetPlan record on
#: budgeted cells; CellWriter persists it under cell.json[BUDGET_PLAN_CELL_KEY]
#: (the T2.5 rho_own denominator + served dtype). An absent pin is absence:
#: the pre-W4 path stays byte-identical.
SLO_FLOORS_ENV = "CAGE_SLO_FLOORS_JSON"
BUDGET_PLAN_ENV = "CAGE_BUDGET_PLAN_JSON"
SLO_FLOORS_MANIFEST_KEY = "slo_floors"
BUDGET_PLAN_CELL_KEY = "budget_plan"
#: The engines a floor may be keyed by: the §7.3 server engines (the
#: in-process oracle serves no floor).
_FLOOR_ENGINES: frozenset[str] = frozenset(_get_args(Engine)) - {"hf"}
#: cellspec topology -> the planner topologies a BudgetPlan for that cell may
#: carry: a 'single' cell may be planned TP-sharded (session b serves
#: single-instance cells at serving_tp=4; run_campaign._budget_env plans
#: topology='tp' for them), the distributed legs plan their own topology.
_BUDGET_PLAN_TOPOLOGIES: dict[str, tuple[str, ...]] = {
    "single": ("single", "tp"),
    "tp": ("tp",),
    "pd": ("pd",),
}

#: run_experiment --backend token -> charter §7.3 engine axis value.
#: gemini/ollama are pilot-legacy backends with NO charter engine — campaign
#: mode refuses them (fail-closed, never a silent default).
ENGINE_OF_BACKEND: dict[str, str] = {
    "vllm": "vllm",
    "sglang": "sglang",
    "lmdeploy": "lmdeploy",
    "lmdeploy-turbomind": "lmdeploy",
    "hf-oracle": "hf",
    "hf_oracle": "hf",
}

#: D4 roster: HF model id -> charter model slug (the §3 manifest `model` and
#: the CellSpec model axis). CAGE_MODEL_SLUG overrides (e.g. an S0 shakedown
#: serving a small stand-in model labels its design-input run explicitly).
HF_MODEL_SLUGS: dict[str, str] = {
    "Qwen/Qwen3-14B": "qwen3-14b",
    "meta-llama/Llama-3.3-70B-Instruct": "llama-3.3-70b",
    "Qwen/Qwen3-Next-80B-A3B-Instruct": "qwen3-next-80b",
    "deepseek-ai/DeepSeek-V3": "deepseek-v3",
    "deepseek-ai/DeepSeek-V3-0324": "deepseek-v3",
}

_CHARTER_MODELS: frozenset[str] = frozenset(_get_args(Model))
_CELLSPEC_SCHEMA_VERSION = 1


#: The pilot-archive escape hatch of ``src.orchestration.ir`` (backlog F6),
#: mirrored here by literal so this import-light module never loads the IR
#: stack; ``tests/test_campaign_session.py`` pins the two equal. The campaign
#: path refuses on PRESENCE (review 2026-09-17, backlog A6): a stale
#: (pre-prefix) dense index serves retrieval out-of-distribution, and a window
#: measured that way would be mislabeled data.
STALE_INDEX_OPT_IN_ENV: str = "CAGE_ALLOW_STALE_INDEX"


class CampaignSessionError(RuntimeError):
    """Campaign-mode contract violation; carries EVERY problem found."""

    def __init__(self, problems: list[str] | str) -> None:
        if isinstance(problems, str):
            problems = [problems]
        self.problems = list(problems)
        lines = "\n".join(f"  [{i + 1}] {p}" for i, p in enumerate(self.problems))
        super().__init__(
            f"campaign mode refused — {len(self.problems)} problem(s) "
            f"(task #116 seam, docs/RESULTS_LAYOUT.md):\n{lines}"
        )


def _campaign_layout():
    """Lazy import: campaign_layout pulls pandas; the cell-dir CLI must stay light."""
    from src.orchestration import campaign_layout as cl

    return cl


# ---------------------------------------------------------------------------
# Identity derivation (single source for the runner AND the shell gates)
# ---------------------------------------------------------------------------


def resolve_model_slug(model: str, env: Mapping[str, str] | None = None) -> str:
    """Charter model slug for the runner's ``--model`` (fail-closed).

    Order: ``CAGE_MODEL_SLUG`` override > the model IS already a charter slug
    > the D4 HF-id roster. Anything else refuses — a campaign tree keyed to a
    non-roster model would be refused by organize_results anyway (§3), so the
    refusal happens HERE, before any serving.
    """
    env = os.environ if env is None else env
    override = (env.get("CAGE_MODEL_SLUG") or "").strip()
    slug = override or (model if model in _CHARTER_MODELS else HF_MODEL_SLUGS.get(model, ""))
    if slug not in _CHARTER_MODELS:
        raise CampaignSessionError(
            f"model {model!r} (CAGE_MODEL_SLUG={override or 'unset'}) does not "
            f"resolve to a D4 charter slug ({sorted(_CHARTER_MODELS)}); set "
            "CAGE_MODEL_SLUG explicitly for a non-roster stand-in model "
            "(design-input runs only)"
        )
    return slug


def _env_float(env: Mapping[str, str], key: str, problems: list[str]) -> Optional[float]:
    raw = (env.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        problems.append(f"{key}={raw!r} is not a float")
        return None


def _env_int(
    env: Mapping[str, str],
    key: str,
    problems: list[str],
    *,
    minimum: int,
    why: str,
) -> Optional[int]:
    """Optional fail-closed integer env (unset -> None; malformed -> problem)."""
    raw = (env.get(key) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        problems.append(f"{key}={raw!r} is not an integer — it carries {why}")
        return None
    if value < minimum:
        problems.append(f"{key}={value} must be >= {minimum} — it carries {why}")
        return None
    return value


def derive_cell_spec(
    *,
    baseline: str,
    baseline_label: Optional[str],
    backend: str,
    model: str,
    env: Mapping[str, str] | None = None,
) -> CellSpec:
    """The cell tuple for one runner invocation (RESULTS_LAYOUT §2, fail-closed).

    Axis sources, in precedence order:
    1. ``CAGE_CELL_ARM`` + ``CAGE_CELL_RETRIEVER`` (both or neither) — the
       explicit-axes path; defaults for the remaining axes are none/F1/single.
    2. ``cellspec.LEGACY_ALIASES`` on ``--baseline-label`` then ``--baseline``
       — the pilot-vocabulary path the existing shell runners speak.
    ``CAGE_CELL_POLICY`` / ``CAGE_CELL_FAMILY`` / ``CAGE_CELL_TOPOLOGY`` /
    ``CAGE_CELL_BUDGET_R`` / ``CAGE_CELL_RATE_FRAC`` override individual axes
    either way; ``CAGE_CELL_CORPUS_BUDGET`` carries the ADR-0106 B12 rung
    (absent = no rung coordinate, never a default). ``CellSpec.__post_init__``
    stays the one validity gate.
    """
    env = os.environ if env is None else env
    problems: list[str] = []

    engine = ENGINE_OF_BACKEND.get(backend)
    if engine is None:
        problems.append(
            f"backend {backend!r} has no charter engine (§7.3: "
            f"{sorted(set(ENGINE_OF_BACKEND.values()))}); campaign mode refuses "
            "gemini/ollama-class legacy backends"
        )

    arm = (env.get("CAGE_CELL_ARM") or "").strip() or None
    retriever = (env.get("CAGE_CELL_RETRIEVER") or "").strip() or None
    if (arm is None) != (retriever is None):
        problems.append(
            "CAGE_CELL_ARM and CAGE_CELL_RETRIEVER must be set together "
            "(an arm without its retriever axis is not a cell identity)"
        )

    policy_default, family_default, topology_default = "none", "F1", "single"
    if arm is None and retriever is None:
        legacy: Optional[CellSpec] = None
        for name in (baseline_label, baseline):
            if name and name in LEGACY_ALIASES:
                legacy = from_legacy(name)
                break
        if legacy is None:
            problems.append(
                f"neither --baseline-label {baseline_label!r} nor --baseline "
                f"{baseline!r} maps to a charter cell tuple "
                "(cellspec.LEGACY_ALIASES); set CAGE_CELL_ARM/CAGE_CELL_RETRIEVER "
                "explicitly or run this label on the pilot path"
            )
        else:
            arm, retriever = legacy.arm, legacy.retriever
            policy_default = legacy.policy
            family_default = legacy.family
            topology_default = legacy.topology

    policy = (env.get("CAGE_CELL_POLICY") or "").strip() or policy_default
    family = (env.get("CAGE_CELL_FAMILY") or "").strip() or family_default
    topology = (env.get("CAGE_CELL_TOPOLOGY") or "").strip() or topology_default
    budget_r = _env_float(env, "CAGE_CELL_BUDGET_R", problems)
    rate_frac = _env_float(env, "CAGE_CELL_RATE_FRAC", problems)
    corpus_budget_tokens = _env_int(
        env, "CAGE_CELL_CORPUS_BUDGET", problems, minimum=1,
        why="the B12 corpus-trunc rung budget in tokens (ADR-0106)",
    )

    try:
        model_slug = resolve_model_slug(model, env)
    except CampaignSessionError as exc:
        problems.extend(exc.problems)
        model_slug = None
    if problems:
        raise CampaignSessionError(problems)
    assert arm is not None and retriever is not None and engine is not None
    assert model_slug is not None
    try:
        return CellSpec(
            arm=arm,  # type: ignore[arg-type]
            retriever=retriever,  # type: ignore[arg-type]
            policy=policy,  # type: ignore[arg-type]
            topology=topology,  # type: ignore[arg-type]
            engine=engine,  # type: ignore[arg-type]
            model=model_slug,  # type: ignore[arg-type]
            family=family,  # type: ignore[arg-type]
            budget_r=budget_r,
            rate_frac=rate_frac,
            corpus_budget_tokens=corpus_budget_tokens,
        )
    except (CellSpecError, ValueError) as exc:
        raise CampaignSessionError(
            f"derived cell tuple is charter-illegal: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Write-time hash journal (S0-15: "hash ledger written at write time")
# ---------------------------------------------------------------------------


def append_write_time_hashes(run_root: Path, paths: Iterable[Path]) -> None:
    """sha256 each just-written artifact and append it to the run-root journal."""
    run_root = Path(run_root)
    entries = hash_artifacts(list(paths), base_dir=run_root)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    journal = run_root / JOURNAL_NAME
    with journal.open("a", encoding="utf-8") as fh:
        for rel, sha in entries.items():
            fh.write(
                json.dumps({"path": rel, "sha256": sha, "written_utc": stamp}) + "\n"
            )
        fh.flush()
        os.fsync(fh.fileno())


def read_write_time_journal(run_root: Path) -> dict[str, str]:
    """Journal as {relpath: LAST sha256} (later writes supersede earlier ones)."""
    journal = Path(run_root) / JOURNAL_NAME
    if not journal.is_file():
        raise CampaignSessionError(
            f"no {JOURNAL_NAME} at {journal} — the tree was not produced by the "
            "campaign-mode runner (write-time hashing, S0-15); seal such a tree "
            "directly via campaign_layout.seal_run if that is deliberate"
        )
    latest: dict[str, str] = {}
    problems: list[str] = []
    for lineno, line in enumerate(
        journal.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"{JOURNAL_NAME}:{lineno}: invalid JSON: {exc}")
            continue
        if (
            not isinstance(rec, dict)
            or not isinstance(rec.get("path"), str)
            or not isinstance(rec.get("sha256"), str)
        ):
            problems.append(f"{JOURNAL_NAME}:{lineno}: record needs path + sha256")
            continue
        latest[rec["path"]] = rec["sha256"]
    if problems:
        raise CampaignSessionError(problems)
    return latest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 16):
            digest.update(chunk)
    return digest.hexdigest()


def seal_campaign_run(run_root: Path) -> Path:
    """Cross-check the write-time journal, then seal the run (§5 ledger.json).

    Refuses (every problem listed) when any current sealed-scope artifact —
    manifest.json + every non-dot file under cells/ — was never journaled or
    hashes differently from its LAST journal entry: the file changed between
    write time and seal time, exactly the drift §9.10 UPGRADE 5 exists to
    catch on the node instead of at analysis load. Stale journal entries for
    files that no longer exist (reset/superseded windows) are ignored — the
    journal is a write log, not an inventory.
    """
    run_root = Path(run_root)
    journal = read_write_time_journal(run_root)
    problems: list[str] = []
    targets: list[Path] = []
    manifest = run_root / "manifest.json"
    if manifest.is_file():
        targets.append(manifest)
    cells = run_root / "cells"
    if cells.is_dir():
        for path in sorted(cells.rglob("*")):
            if not path.is_file():
                continue
            rel_parts = path.relative_to(run_root).parts
            if any(part.startswith(".") for part in rel_parts):
                continue
            if path.name.endswith(".tmp"):
                continue  # crash residue: seal_run itself refuses it loudly
            targets.append(path)
    for path in targets:
        rel = path.relative_to(run_root).as_posix()
        expected = journal.get(rel)
        if expected is None:
            problems.append(
                f"{rel}: not in {JOURNAL_NAME} — written outside the campaign "
                "writer (write-time hashing is the S0-15 contract)"
            )
        elif _sha256_file(path) != expected:
            problems.append(
                f"{rel}: content differs from its write-time hash — the file "
                "changed between write and seal (§9.10)"
            )
    if problems:
        raise CampaignSessionError(problems)
    return _campaign_layout().seal_run(run_root)


# ---------------------------------------------------------------------------
# JSON normalization (numpy -> native; fail loud on anything else)
# ---------------------------------------------------------------------------


def refuse_stale_index_summary(experiment_summary: Mapping[str, Any]) -> None:
    """Refuse a window whose runner summary says a stale index was served.

    ``run_experiment.py`` persists ``ir_index.stale_index_opt_in`` under
    ``metrics.json["experiment"]["stale_index_opt_in"]`` (False when no dense
    index was used). Only a literal ``False`` passes: ``True`` is
    out-of-distribution retrieval, a non-bool is never coerced, and an ABSENT
    key is a pre-A6 runner whose provenance cannot be trusted (backlog A6/F6).
    """
    experiment = experiment_summary.get("experiment")
    value = experiment.get("stale_index_opt_in") if isinstance(experiment, Mapping) else None
    if value is False:
        return
    raise CampaignSessionError(
        f"experiment.stale_index_opt_in is {value!r}; a campaign window must "
        "record False (a stale pre-prefix dense index served under "
        f"{STALE_INDEX_OPT_IN_ENV} is refused, backlog A6/F6; an absent key is "
        "a runner without the provenance field)"
    )


def refuse_dropped_rows_summary(experiment_summary: Mapping[str, Any]) -> None:
    """Refuse a window whose runner summary counted a dropped row (ADR-0116).

    ``run_experiment.py`` persists the per-query guard counters under
    ``metrics.json["consort"]``. On the campaign path the record and turn
    guards fail the cell in stage (``RecordStageError``), so those two
    counters read zero by construction and this seam is their belt; the
    prepare counter and the evidence-append counter are enforced HERE only.
    Checked BEFORE any artifact is written. Absence is not zero: a missing
    block, a missing counter or a non-integer value refuses (a runner without
    the counters cannot vouch for its population).
    """
    consort = experiment_summary.get("consort")
    if not isinstance(consort, Mapping):
        raise CampaignSessionError(
            "metrics.json carries no consort block (task #127 per-query drop "
            "counters); a campaign window without it cannot vouch for its "
            "population (ADR-0116)"
        )
    problems: list[str] = []
    for key in CONSORT_COUNTERS:
        value = consort.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            problems.append(
                f"consort.{key} is {value!r}, not an integer count (never coerced)"
            )
        elif value != 0:
            problems.append(
                f"consort.{key} = {value}: the runner dropped request(s) this "
                "window's artifacts do not carry; a campaign window's denominator "
                "is the offered schedule, so the window is refused (ADR-0116)"
            )
    if problems:
        raise CampaignSessionError(problems)


def _json_norm(value: Any) -> Any:
    """One JSON round trip: tuples become lists, keys sort, so a pin, its
    on-disk copy and a caller's tuple-spelled record compare equal."""
    return json.loads(json.dumps(value, sort_keys=True))


def validate_slo_floors(floors: Any) -> dict[str, dict[str, Any]]:
    """The §6.1 floors pin as a plain dict, or refuse (every problem named).

    Keys are §7.3 server engines; every value is an object whose ``ttft_s``
    and ``tpot_s`` are finite numbers > 0 in SECONDS (the goodput.SLOBaseline
    inputs). Extra provenance keys (n_requests, statistic, budget_fraction,
    source_sha256) pass through; an empty object pins nothing and refuses.
    """
    if not isinstance(floors, Mapping):
        raise CampaignSessionError(
            f"{SLO_FLOORS_ENV}: floors must be a JSON object "
            f"{{engine: {{ttft_s, tpot_s}}}}, got {type(floors).__name__}"
        )
    if not floors:
        raise CampaignSessionError(
            f"{SLO_FLOORS_ENV}: floors object is empty (no engine floor to pin; "
            "the #14 SLO pair needs one per served engine)"
        )
    problems: list[str] = []
    out: dict[str, dict[str, Any]] = {}
    for engine, floor in floors.items():
        if engine not in _FLOOR_ENGINES:
            problems.append(
                f"{SLO_FLOORS_ENV}: {engine!r} is not a §7.3 server engine "
                f"({sorted(_FLOOR_ENGINES)}; the hf oracle serves no floor)"
            )
            continue
        if not isinstance(floor, Mapping):
            problems.append(
                f"{SLO_FLOORS_ENV}: floor for {engine!r} must be an object with "
                f"ttft_s/tpot_s, got {floor!r}"
            )
            continue
        for key in ("ttft_s", "tpot_s"):
            value = floor.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                problems.append(
                    f"{SLO_FLOORS_ENV}: {engine}.{key} is {value!r}, must be a "
                    "finite number > 0 (seconds; the §6.1 single-stream floor)"
                )
        out[str(engine)] = dict(floor)
    if problems:
        raise CampaignSessionError(problems)
    return out


def parse_slo_floors(raw: str) -> dict[str, dict[str, Any]]:
    """``CAGE_SLO_FLOORS_JSON`` -> validated floors (fail closed)."""
    try:
        doc = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise CampaignSessionError(f"{SLO_FLOORS_ENV} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise CampaignSessionError(
            f"{SLO_FLOORS_ENV} must be a JSON object {{engine: {{ttft_s, tpot_s}}}}, "
            f"got {type(doc).__name__}"
        )
    return validate_slo_floors(doc)


def validate_budget_plan(plan: Any, spec: CellSpec) -> dict[str, Any]:
    """The BudgetPlan record pin, cross-checked against the cell tuple.

    The rho_own consumer reads ``budget_bytes_total`` (int >= 1) and
    ``kv_dtype`` (a cache_budget dtype); the record's ``model`` and ``engine``
    must be the cell's, its ``topology`` one the cell may be planned under
    (_BUDGET_PLAN_TOPOLOGIES) and its ``r`` the cell's ``budget_r`` when the
    cell carries one (the DIST overlay carries none: its r is the registered
    dist_budget_r and cannot be cross-checked here). A pd plan must carry
    ``pools_bytes`` as two integer byte budgets >= 1 (prefill, decode): the
    §6.5 split the regime referee's summed-pool lane consumes at emission,
    checked HERE so a malformed split refuses before serving. A record
    planned for another cell is mislabeled data and refuses.
    """
    if not isinstance(plan, Mapping):
        raise CampaignSessionError(
            f"{BUDGET_PLAN_ENV}: the budget plan must be a JSON object (a "
            f"cache_budget.BudgetPlan record), got {type(plan).__name__}"
        )
    if spec.engine == "hf":
        # Review S2: the same structural rule CellWriter applies, enforced at
        # activation so an hf cell never serves a trial under a budget pin.
        raise CampaignSessionError(
            f"{BUDGET_PLAN_ENV}: the in-process hf oracle serves no byte budget "
            "(P2: excluded from pressure sweeps); a budget plan on an hf cell "
            "is a contradiction"
        )
    problems: list[str] = []
    budget = plan.get("budget_bytes_total")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        problems.append(
            f"{BUDGET_PLAN_ENV}: budget_bytes_total is {budget!r}, must be an "
            "integer >= 1 (bytes; the rho_own denominator)"
        )
    kv_dtype = plan.get("kv_dtype")
    if not isinstance(kv_dtype, str) or kv_dtype not in KV_DTYPE_FACTOR:
        problems.append(
            f"{BUDGET_PLAN_ENV}: kv_dtype is {kv_dtype!r}, must be one of "
            f"{sorted(KV_DTYPE_FACTOR)} (the SERVED KV dtype)"
        )
    for key, want in (("model", spec.model), ("engine", spec.engine)):
        if plan.get(key) != want:
            problems.append(
                f"{BUDGET_PLAN_ENV}: {key} is {plan.get(key)!r} but this cell's "
                f"{key} is {want!r} (a budget planned for another cell is "
                "mislabeled data)"
            )
    topology = plan.get("topology")
    if topology not in _BUDGET_PLAN_TOPOLOGIES.get(spec.topology, ()):
        problems.append(
            f"{BUDGET_PLAN_ENV}: topology is {topology!r} but this cell's topology "
            f"is {spec.topology!r} (a 'single' cell may be planned TP-sharded; "
            "tp/pd cells plan their own topology)"
        )
    if topology == "pd":
        pools = plan.get("pools_bytes")
        if (
            not isinstance(pools, (list, tuple))
            or len(pools) != 2
            or any(isinstance(b, bool) or not isinstance(b, int) or b < 1 for b in pools)
        ):
            problems.append(
                f"{BUDGET_PLAN_ENV}: pools_bytes is {pools!r}; a pd plan must carry "
                "[prefill_bytes, decode_bytes] as two integers >= 1 (the §6.5 "
                "split, the regime referee's summed-pool budgets)"
            )
    r = plan.get("r")
    if isinstance(r, bool) or not isinstance(r, (int, float)) or not math.isfinite(r) or r <= 0:
        problems.append(f"{BUDGET_PLAN_ENV}: r is {r!r}, must be finite and > 0")
    elif spec.budget_r is not None and not math.isclose(
        # rel_tol 1e-6: the cell's budget_r arrives through CAGE_CELL_BUDGET_R,
        # which run_campaign formats with :g (6 significant digits), while the
        # plan carries the unformatted grid float (review S5).
        r, spec.budget_r, rel_tol=1e-6, abs_tol=0.0
    ):
        problems.append(
            f"{BUDGET_PLAN_ENV}: r is {r!r} but this cell's budget_r is "
            f"{spec.budget_r!r}"
        )
    if problems:
        raise CampaignSessionError(problems)
    try:
        return _json_norm(dict(plan))
    except TypeError as exc:
        raise CampaignSessionError(
            f"{BUDGET_PLAN_ENV}: the budget plan is not JSON-serializable: {exc}"
        ) from exc


def parse_budget_plan(raw: str, spec: CellSpec) -> dict[str, Any]:
    """``CAGE_BUDGET_PLAN_JSON`` -> validated record (fail closed)."""
    try:
        doc = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise CampaignSessionError(f"{BUDGET_PLAN_ENV} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise CampaignSessionError(
            f"{BUDGET_PLAN_ENV} must be a JSON object (a cache_budget.BudgetPlan "
            f"record), got {type(doc).__name__}"
        )
    return validate_budget_plan(doc, spec)


def _normalize_json(value: Any) -> Any:
    """numpy scalars/arrays -> native JSON values, recursively.

    Mirrors run_experiment._json_default's numpy handling for the evidence
    writer, WITHOUT the str() fallback: a value that is neither JSON-native
    nor numpy stays as-is so the atomic layout writer fails LOUD on it
    (silent default=str drift is audit H3's defect, not a fix).
    """
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _normalize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(v) for v in value]
    return value


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def _attainment(rows: list[Mapping[str, Any]]) -> Optional[float]:
    """§6.1 (c) attainment of one window: completed over issued, the quotient
    ``goodput.evaluate_window`` reports as ``WindowMetrics.attainment``
    (``n_completed / n_issued``), computed here from the normalized request
    rows so the regime referee and the yield analysis share one definition.
    Completed = the row's shared validity predicate ``ok`` is literally True
    (task #127); issued = every row (the dropped-rows refusal upstream makes
    the rows equal the offered schedule, ADR-0116). No rows -> None:
    attainment is undefined and the writer leaves the label null (absence,
    never 0.0)."""
    if not rows:
        return None
    return sum(1 for row in rows if row.get("ok") is True) / len(rows)


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------


class CampaignCellSession:
    """One runner invocation's campaign-mode state: ONE cell, N trial windows."""

    def __init__(
        self,
        *,
        run_root: Path,
        dataset: str,
        spec: CellSpec,
        num_trials: int,
        run_seed: int,
        kv_cache_dtype: Optional[str] = None,
        gpu_count: Optional[int] = None,
        window_ordinal_base: int = 0,
        env: Mapping[str, str] | None = None,
        slo_floors: Mapping[str, Any] | None = None,
        budget_plan: Mapping[str, Any] | None = None,
    ) -> None:
        self.env = os.environ if env is None else env
        self.run_root = Path(run_root)
        self.dataset = dataset
        self.spec = spec
        self.row_key = spec.to_row_key()
        self.num_trials = int(num_trials)
        self.run_seed = int(run_seed)
        self.kv_cache_dtype = kv_cache_dtype
        # Batch 2 W4 (ADR-0117): the §6.1 floors this cell pins (written into
        # manifest.json at creation, compared on reopen) and the BudgetPlan
        # record CellWriter persists into cell.json; None = pin not in use
        # (pilot/shell producers), the pre-W4 path byte-identical.
        self.slo_floors = None if slo_floors is None else validate_slo_floors(slo_floors)
        self.budget_plan = (
            None if budget_plan is None else validate_budget_plan(budget_plan, spec)
        )
        if (
            self.slo_floors is not None
            and spec.engine != "hf"
            and spec.engine not in self.slo_floors
        ):
            # Review S3: the floor of THIS cell's engine is what contrast #14
            # reads for its windows; a pin copied from another engine's plan
            # would spend the GPU time before the analysis refused.
            raise CampaignSessionError(
                f"{SLO_FLOORS_ENV} pins floors for {sorted(self.slo_floors)} but "
                f"this cell serves on {spec.engine!r}; every window needs the "
                "floor of the engine it was served on (§6.1)"
            )
        # W4.2: the GPU count this cell's serving stack launched with
        # (CAGE_GPU_COUNT, threaded from the run_campaign plan step) —
        # persisted into cell.json by CellWriter, which owns the
        # topology/count refusal rules; None = seam not in use (pilot/shell
        # producers), the analysis consumer's labeled skip stays the outcome.
        self.gpu_count = gpu_count
        # W4.4: per-task RULER steps share one (row_key, dataset) window
        # space; this base shifts every emitted/checked ordinal so each
        # task's runner invocation owns (base, base+num_trials] exclusively.
        self.window_ordinal_base = int(window_ordinal_base)
        self._run: Any = None  # lazy CampaignRun
        self._manifest_created = False

        cl = _campaign_layout()
        problems: list[str] = []
        if gpu_count is not None and (
            isinstance(gpu_count, bool)
            or not isinstance(gpu_count, int)
            or gpu_count < 1
        ):
            problems.append(
                f"gpu_count={gpu_count!r} must be an integer >= 1 "
                "(CAGE_GPU_COUNT: the serving stack's GPU count, W4.2)"
            )
        if (
            isinstance(window_ordinal_base, bool)
            or not isinstance(window_ordinal_base, int)
            or window_ordinal_base < 0
        ):
            problems.append(
                f"window_ordinal_base={window_ordinal_base!r} must be an "
                "integer >= 0 (CAGE_WINDOW_ORDINAL_BASE: the step's claimed "
                "window-ordinal range starts after it)"
            )
        if dataset not in cl.DATASET_IDS:
            problems.append(
                f"dataset {dataset!r} is not a RESULTS_LAYOUT §1 dataset id "
                f"({sorted(cl.DATASET_IDS)}) — campaign windows are named "
                "window_<dataset>-<NN>, so a non-roster dataset cannot be emitted"
            )
        if self.num_trials < 1:
            problems.append(f"num_trials={num_trials!r} must be >= 1")
        parts = self.run_root.parts
        if len(parts) < 3:
            problems.append(
                f"campaign root {self.run_root} is too shallow — expected "
                "results/<campaign>/<session>/<run_id>"
            )
        else:
            self.run_id = self.run_root.name
            self.session = self.run_root.parent.name
            self.campaign = self.run_root.parent.parent.name
            if not cl.RUN_ID_RE.match(self.run_id):
                problems.append(
                    f"run_id (campaign-root basename) {self.run_id!r} violates "
                    f"the §1 grammar {cl.RUN_ID_RE.pattern}"
                )
            if self.session not in cl.SESSIONS:
                problems.append(
                    f"session (campaign-root parent) {self.session!r} is not a "
                    f"§1 session ({sorted(cl.SESSIONS)})"
                )
        if problems:
            raise CampaignSessionError(problems)
        self._check_existing_tree()

    def _check_existing_tree(self) -> None:
        """W4 activation checks (review S1): a pinned cell refuses BEFORE it
        serves a trial when the run's manifest carries other floors, or when
        the cell's recorded budget contradicts the pin, or when the cell is
        already populated without a budget record (review S4). emit_window
        and CellWriter apply the same rules again at write time."""
        manifest_path = self.run_root / "manifest.json"
        if self.slo_floors is not None and manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CampaignSessionError(
                    f"{manifest_path} is not readable JSON ({exc}); refusing to "
                    "extend the run"
                ) from exc
            if not isinstance(manifest, Mapping):
                raise CampaignSessionError(
                    f"{manifest_path} is not a JSON object; refusing to extend the run"
                )
            self._check_manifest_floors(manifest)
        meta_path = self.cell_dir / "cell.json"
        if self.budget_plan is None or not meta_path.is_file():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignSessionError(
                f"{meta_path} is not valid JSON ({exc}); the cell cannot be "
                "resumed; re-run it with CAGE_FORCE_RERUN=1 to wipe it"
            ) from exc
        if not isinstance(meta, Mapping):
            return
        recorded = meta.get(BUDGET_PLAN_CELL_KEY)
        windows = meta.get("windows")
        if recorded is None:
            if isinstance(windows, Mapping) and windows:
                raise CampaignSessionError(
                    f"{meta_path} carries {len(windows)} window(s) but no "
                    f"{BUDGET_PLAN_CELL_KEY!r} record; a populated cell without a "
                    "budget record cannot be extended under a budget pin (its "
                    "windows were served under an unrecorded budget); wipe it "
                    "with CAGE_FORCE_RERUN=1 or use a new run_id"
                )
            return
        if _json_norm(recorded) != _json_norm(self.budget_plan):
            got = recorded.get("budget_bytes_total") if isinstance(recorded, Mapping) else recorded
            raise CampaignSessionError(
                f"{meta_path} recorded {BUDGET_PLAN_CELL_KEY!r} (budget_bytes_total="
                f"{got!r}) contradicts this cell's {BUDGET_PLAN_ENV} pin "
                f"(budget_bytes_total={self.budget_plan.get('budget_bytes_total')!r}); "
                "a re-run under another budget is a new cell tree, not an extension"
            )

    # -- activation -------------------------------------------------------

    @classmethod
    def from_cli(cls, args: Any, env: Mapping[str, str] | None = None) -> Optional["CampaignCellSession"]:
        """Build the session from run_experiment's parsed args, or None.

        Activation: ``--campaign-root`` flag, else the ``CAGE_CAMPAIGN_ROOT``
        env var the shell runners export. Inactive -> None (pilot path,
        untouched). Active -> every derivation/validation runs HERE, before
        any dataset/engine work (fail-closed early, the runner's doctrine).
        """
        env = os.environ if env is None else env
        root = getattr(args, "campaign_root", None) or (env.get("CAGE_CAMPAIGN_ROOT") or "").strip()
        if not root:
            return None
        if STALE_INDEX_OPT_IN_ENV in env:
            raise CampaignSessionError(
                f"{STALE_INDEX_OPT_IN_ENV} is set ({env[STALE_INDEX_OPT_IN_ENV]!r}); "
                "campaign mode never serves a stale (pre-prefix) dense index "
                "(backlog A6/F6), unset it or rebuild with --rebuild-ir-index"
            )
        if getattr(args, "top_k_sweep", False):
            raise CampaignSessionError(
                "--top-k-sweep varies top_k INSIDE one cell identity; a campaign "
                "cell is ONE tuple — run each top_k as its own cell instead"
            )
        spec = derive_cell_spec(
            baseline=args.baseline,
            baseline_label=getattr(args, "baseline_label", None),
            backend=args.backend,
            model=args.model,
            env=env,
        )
        problems: list[str] = []
        gpu_count = _env_int(
            env, "CAGE_GPU_COUNT", problems, minimum=1,
            why="the serving stack's GPU count (W4.2, run_campaign threads it)",
        )
        base = _env_int(
            env, "CAGE_WINDOW_ORDINAL_BASE", problems, minimum=0,
            why="the per-task window-ordinal range start (W4.4 RULER steps)",
        )
        if problems:
            raise CampaignSessionError(problems)
        # W4: the two plan pins (an empty value is absence, like every other
        # optional seam env; a malformed one refuses naming the env).
        raw_floors = (env.get(SLO_FLOORS_ENV) or "").strip()
        slo_floors = parse_slo_floors(raw_floors) if raw_floors else None
        raw_plan = (env.get(BUDGET_PLAN_ENV) or "").strip()
        budget_plan = parse_budget_plan(raw_plan, spec) if raw_plan else None
        return cls(
            run_root=Path(root),
            dataset=args.dataset,
            spec=spec,
            num_trials=getattr(args, "num_trials", 1),
            run_seed=args.seed,
            kv_cache_dtype=getattr(args, "kv_cache_dtype", None),
            gpu_count=gpu_count,
            window_ordinal_base=base or 0,
            env=env,
            slo_floors=slo_floors,
            budget_plan=budget_plan,
        )

    # -- paths / resume ---------------------------------------------------

    @property
    def cell_dir(self) -> Path:
        return self.run_root / "cells" / self.row_key

    def window_key(self, ordinal: int) -> str:
        # ``ordinal`` is the runner's TRIAL number (1..num_trials); the W4.4
        # base shifts it into this step's claimed on-disk ordinal range, so
        # every path below (resume checks, resets, emission) is range-scoped.
        return f"{self.dataset}-{self.window_ordinal_base + int(ordinal):02d}"

    def window_dir(self, ordinal: int) -> Path:
        return self.cell_dir / f"window_{self.window_key(ordinal)}"

    def window_complete(self, ordinal: int) -> bool:
        """Same semantics as the shell gates' metrics_json_valid (J2):
        the window's metrics.json must exist AND parse."""
        path = self.window_dir(ordinal) / WINDOW_METRICS_NAME
        if not path.is_file():
            return False
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            print(
                f"[campaign] invalid {WINDOW_METRICS_NAME} (unparseable JSON) -> "
                f"treating window as INCOMPLETE: {path}"
            )
            return False
        return True

    def load_window_summary(self, ordinal: int) -> dict[str, Any]:
        path = self.window_dir(ordinal) / WINDOW_METRICS_NAME
        return json.loads(path.read_text(encoding="utf-8"))

    def reset_incomplete_windows(self) -> list[int]:
        """Remove half-written windows (dir and/or windows[] entry) for THIS
        dataset's trial ordinals so re-emission cannot collide (resume path).

        A window is kept only when its metrics.json parses (window_complete).
        Other datasets' windows in the same cell are never touched. A corrupt
        cell.json refuses loudly — CellWriter could not extend it either;
        CAGE_FORCE_RERUN wipes the cell deliberately.
        """
        reset: list[int] = []
        cell_dir = self.cell_dir
        meta_path = cell_dir / "cell.json"
        meta: Optional[dict[str, Any]] = None
        if meta_path.is_file():
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise CampaignSessionError(
                    f"{meta_path} is not valid JSON ({exc}) — the cell cannot be "
                    "resumed; re-run it with CAGE_FORCE_RERUN=1 to wipe it"
                ) from exc
            if isinstance(loaded, dict):
                meta = loaded
        windows = meta.get("windows") if isinstance(meta, dict) else None
        changed = False
        for ordinal in range(1, self.num_trials + 1):
            if self.window_complete(ordinal):
                continue
            wdir = self.window_dir(ordinal)
            touched = False
            if wdir.exists():
                shutil.rmtree(wdir)
                touched = True
            key = self.window_key(ordinal)
            if isinstance(windows, dict) and key in windows:
                windows.pop(key)
                changed = True
                touched = True
            if touched:
                reset.append(ordinal)
        if changed and meta is not None:
            _atomic_write_json(meta_path, meta)
            append_write_time_hashes(self.run_root, [meta_path])
        if reset:
            print(
                f"[campaign] reset incomplete window(s) {reset} of "
                f"cells/{self.row_key} (dataset {self.dataset}) for re-emission"
            )
        return reset

    # -- manifest ---------------------------------------------------------

    def _resolve_engine_version(self, backend_metadata: Mapping[str, Any]) -> str:
        override = (self.env.get("CAGE_ENGINE_VERSION") or "").strip()
        if override:
            return override
        for key in ("server_version", "client_library_version"):
            value = backend_metadata.get(key)
            if isinstance(value, str) and value:
                return value
        raise CampaignSessionError(
            "engine_version unresolvable: CAGE_ENGINE_VERSION unset and the "
            "backend reported neither server_version nor client_library_version "
            "(§3: a run without engine provenance is not a run)"
        )

    def _resolve_dataset_manifests_sha256(self) -> str:
        override = (self.env.get("CAGE_DATASET_MANIFESTS_SHA256") or "").strip()
        if override:
            return override
        manifest_path = (self.env.get("CAGE_QUERY_MANIFEST") or "").strip()
        if manifest_path and Path(manifest_path).is_file():
            return _sha256_file(Path(manifest_path))
        raise CampaignSessionError(
            "dataset_manifests_sha256 unresolvable: set "
            "CAGE_DATASET_MANIFESTS_SHA256 explicitly or point "
            "CAGE_QUERY_MANIFEST at the pre-drawn query manifest file "
            "(§3: the manifest pins the exact query/corpus builds)"
        )

    def _manifest_extra(self) -> dict[str, Any]:
        extra: dict[str, Any] = {
            # Run-level server launch dtype (S0-15 kv_dtype provenance); a
            # policy cell's per-cell dtype lives in its own metrics.json and
            # its policy axis — a run-level field cannot carry per-cell values.
            "kv_cache_dtype": (
                self.kv_cache_dtype
                or (self.env.get("VLLM_KV_CACHE_DTYPE") or "").strip()
                or "auto"
            ),
        }
        if self.slo_floors is not None:
            # W4: the §6.1 floors, keyed as the #14 executor reads them
            # (run_campaign_analysis._SLO_FLOORS_MANIFEST_KEY).
            extra[SLO_FLOORS_MANIFEST_KEY] = self.slo_floors
        roster_raw = (self.env.get("CAGE_CAMPAIGN_DATASETS") or "").strip()
        if roster_raw:
            cl = _campaign_layout()
            roster = [d for d in roster_raw.replace(",", " ").split() if d]
            unknown = sorted(set(roster) - cl.DATASET_IDS)
            if unknown:
                raise CampaignSessionError(
                    f"CAGE_CAMPAIGN_DATASETS names non-§1 dataset id(s) {unknown} "
                    f"({sorted(cl.DATASET_IDS)})"
                )
            if self.dataset not in roster:
                raise CampaignSessionError(
                    f"dataset {self.dataset!r} is not in the declared "
                    f"CAGE_CAMPAIGN_DATASETS roster {roster} — "
                    "organize_results would refuse this window"
                )
            extra["datasets"] = roster
        return extra

    def _require_env(self, key: str, why: str, problems: list[str]) -> str:
        value = (self.env.get(key) or "").strip()
        if not value:
            problems.append(f"{key} is unset/empty — §3 requires {why}")
        return value

    def _check_manifest_floors(self, manifest: Mapping[str, Any]) -> None:
        """W4 reopen rule: a pinned cell extends only a run whose manifest
        carries the SAME floors (§3: amended never; one run, one floor set).
        A cell without a pin extends any run (pre-W4 producers)."""
        if self.slo_floors is None:
            return
        recorded = manifest.get(SLO_FLOORS_MANIFEST_KEY)
        if recorded is None:
            raise CampaignSessionError(
                f"manifest.json carries no {SLO_FLOORS_MANIFEST_KEY!r} but this "
                f"cell pins {SLO_FLOORS_ENV}; §3: the manifest is amended never, "
                "so the floors can never be added to this run (a re-run with the "
                "pinned plan is a new run_id)"
            )
        if not isinstance(recorded, Mapping):
            raise CampaignSessionError(
                f"manifest.json {SLO_FLOORS_MANIFEST_KEY!r} is {recorded!r}, not an "
                "object; refusing to extend a run whose floors cannot be read"
            )
        if _json_norm(recorded) != _json_norm(self.slo_floors):
            differing = sorted(
                engine
                for engine in set(recorded) | set(self.slo_floors)
                if _json_norm(recorded.get(engine)) != _json_norm(self.slo_floors.get(engine))
            )
            raise CampaignSessionError(
                f"manifest.json {SLO_FLOORS_MANIFEST_KEY!r} differs from this "
                f"cell's {SLO_FLOORS_ENV} pin for engine(s) {differing}; one run "
                "serves ONE floor set (§3 amended never); a cell planned against "
                "other floors belongs to a new run_id"
            )

    def _ensure_run(self, backend_metadata: Mapping[str, Any]) -> Any:
        """Open the CampaignRun, creating manifest.json on first use (§3)."""
        if self._run is not None:
            return self._run
        cl = _campaign_layout()
        manifest_path = self.run_root / "manifest.json"
        if manifest_path.is_file():
            run = cl.CampaignRun(self.run_root)
            self._check_manifest_floors(run.manifest)
            self._run = run
            return self._run
        problems: list[str] = []
        provider = self._require_env("CAGE_PROVIDER", "the provider (neocloud name or gcp)", problems)
        hardware = self._require_env("CAGE_HARDWARE", "the machine shape / GPU SKU x count", problems)
        if problems:
            raise CampaignSessionError(problems)
        self._run = cl.CampaignRun.create(
            self.run_root,
            campaign=self.campaign,
            session=self.session,
            run_id=self.run_id,
            model=self.spec.model,
            engine=self.spec.engine,
            engine_version=self._resolve_engine_version(backend_metadata),
            seed=self.run_seed,
            provider=provider,
            hardware=hardware,
            dataset_manifests_sha256=self._resolve_dataset_manifests_sha256(),
            cellspec_schema_version=_CELLSPEC_SCHEMA_VERSION,
            extra=self._manifest_extra(),
        )
        self._manifest_created = True
        print(f"[campaign] manifest.json created at run start: {manifest_path}")
        return self._run

    # -- window emission --------------------------------------------------

    @staticmethod
    def _identity(row: Mapping[str, Any]) -> tuple[Any, Any, Any]:
        return (row.get("example_id"), row.get("repeat_index"), row.get("record_index"))

    def _normalize_request_row(self, row: Mapping[str, Any], index: int) -> dict[str, Any]:
        out = {str(k): _normalize_json(v) for k, v in row.items()}
        example_id = out.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise CampaignSessionError(
                f"results row {index} lacks a non-empty example_id — the §8 join "
                "key; an unjoinable row is an unaccountable row (#127)"
            )
        # #127 join triple: absent components stay explicit None (absence of a
        # schedule is not index 0), never omitted keys.
        out.setdefault("repeat_index", None)
        out.setdefault("record_index", None)
        if "ok" not in out:
            # The loaders' shared validity predicate (task #127):
            # NOT error AND NOT empty_generation.
            out["ok"] = (not out.get("error")) and not bool(out.get("empty_generation"))
        return out

    def _role_budgets(
        self,
        plan: Mapping[str, Any] | None,
        cage_stats_rows: list[Mapping[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Per-role byte budgets for the regime writer's summed-pool lane
        (§6.5, T2.3), or None.

        ``plan`` is the cell's effective BudgetPlan record (the CellWriter's:
        the session pin, or the value adopted from cell.json on a resume
        without the pin). Only a pd cell with a NON-EMPTY staged series routes
        there, from the record's ``pools_bytes`` (prefill, decode): the §6.5
        split is an explicit recorded input, never inferred from the stream.
        An empty series returns None so the writer records UNKNOWN_TELEMETRY
        like every other cell (budgets on an empty stream are a caller bug
        there). A pd cell without any record returns None, so the writer
        refuses a multi-role series (T4.1) and the cell fails instead of
        certifying a pooled window against budgets nobody recorded; a pd
        record without the split refuses here by name (validate_budget_plan
        already refuses it at activation; adopted records are re-checked).
        Non-pd cells always return None. The role keys are the planner's
        (cache_budget: prefill, decode); the writer refuses a stream whose
        roles do not match them exactly, a missing role included."""
        if self.spec.topology != "pd" or plan is None or not cage_stats_rows:
            return None
        pools = plan.get("pools_bytes")
        if not isinstance(pools, (list, tuple)) or len(pools) != 2:
            raise CampaignSessionError(
                f"{BUDGET_PLAN_ENV}: a pd cell's plan must carry pools_bytes "
                f"[prefill, decode] for the §6.5 summed-pool regime lane, got "
                f"{pools!r}"
            )
        return {"prefill": pools[0], "decode": pools[1]}

    def _load_staging_jsonl(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not path.is_file():
            return rows
        problems: list[str] = []
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"{path}:{lineno}: invalid JSON: {exc}")
                continue
            if not isinstance(rec, dict):
                problems.append(f"{path}:{lineno}: record is not an object")
                continue
            rows.append(rec)
        if problems:
            raise CampaignSessionError(problems)
        return rows

    def _reconcile_evidence(
        self,
        requests_rows: list[dict[str, Any]],
        evidence_rows: list[dict[str, Any]],
        experiment_summary: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Make the two per-query chains reconcile or refuse loudly (verify (b)).

        Open-loop arrivals dropped by the cap / dispatch errors produce a
        results stub but never reach record_result, so no evidence row exists;
        emit a synthesized evidence stub carrying the SAME identity triple and
        the shared validity fields (ok=False + the error). A count mismatch
        that is NOT explained by error stubs — a lost append on a VALID row —
        refuses: the window would fail verify_results reconciliation anyway,
        and on-pod is the right place to hear it.
        """
        have = {self._identity(r) for r in evidence_rows}
        for row in requests_rows:
            ident = self._identity(row)
            if ident in have:
                continue
            if row.get("error"):
                evidence_rows.append(
                    {
                        "example_id": row["example_id"],
                        "repeat_index": row.get("repeat_index"),
                        "record_index": row.get("record_index"),
                        "ok": False,
                        "error": row.get("error"),
                        "empty_generation": False,
                        "arrival_s": row.get("arrival_s"),
                        "synthesized": "dispatch-stub (no generation to evidence)",
                    }
                )
                have.add(ident)
        if len(requests_rows) != len(evidence_rows):
            consort = (experiment_summary.get("consort") or {}) if isinstance(
                experiment_summary, Mapping
            ) else {}
            raise CampaignSessionError(
                f"window would fail verify_results reconciliation: "
                f"{len(requests_rows)} results row(s) vs {len(evidence_rows)} "
                f"evidence row(s) after dispatch-stub reconciliation "
                f"(consort: {dict(consort)}) — a lost evidence append on a "
                "valid row is §9.10-accountable, not paperable-over"
            )
        return evidence_rows

    def emit_window(
        self,
        *,
        ordinal: int,
        trial_seed: int,
        results_rows: list[Mapping[str, Any]],
        staging_dir: Path,
        experiment_summary: Mapping[str, Any],
        backend_metadata: Mapping[str, Any],
        telemetry_snapshot: Optional[Mapping[str, Any]],
        t_start: float,
        t_end: float,
    ) -> Any:
        """Emit ONE §1 measurement window through campaign_layout's writers."""
        refuse_stale_index_summary(experiment_summary)
        refuse_dropped_rows_summary(experiment_summary)
        cl = _campaign_layout()
        run = self._ensure_run(backend_metadata)
        cell = run.cell(
            self.spec, gpu_count=self.gpu_count, budget_plan=self.budget_plan
        )

        requests_rows_n = [
            self._normalize_request_row(row, i) for i, row in enumerate(results_rows)
        ]
        staging_dir = Path(staging_dir)
        evidence_rows = [
            _normalize_json(r)
            for r in self._load_staging_jsonl(staging_dir / "qa_evidence.jsonl")
        ]
        evidence_rows = self._reconcile_evidence(
            requests_rows_n, evidence_rows, experiment_summary
        )
        cage_stats_rows = self._load_staging_jsonl(staging_dir / "telemetry_series.jsonl")
        engine_metrics = _normalize_json(
            {
                "backend": dict(backend_metadata),
                "vllm_telemetry": (
                    dict(telemetry_snapshot) if telemetry_snapshot is not None else None
                ),
            }
        )
        qa_evidence = (
            None if self.dataset in cl.QA_EVIDENCE_EXEMPT_DATASETS else evidence_rows
        )

        # The trial number shifts into this step's claimed ordinal range
        # (W4.4); base 0 keeps every pre-W4.4 emission byte-identical.
        shifted = self.window_ordinal_base + int(ordinal)
        handle = cell.add_window(
            self.dataset,
            seed=int(trial_seed),
            rep=shifted,
            t_start=float(t_start),
            t_end=float(t_end),
            requests=requests_rows_n,
            cage_stats=cage_stats_rows,
            engine_metrics=engine_metrics,
            qa_evidence=qa_evidence,
            ordinal=shifted,
        )
        # §6.1 regime referee (V8 slice): label the window from the telemetry
        # sampled during it plus completed-over-issued attainment. Runs BEFORE
        # the metrics sentinel (a failed referee leaves an incomplete window
        # that reset_incomplete_windows re-emits) and before the hash journal
        # below (regime.json is a sealed artifact). An empty series is
        # recorded as UNKNOWN_TELEMETRY by the writer on every cell; a series
        # with two or more roles on a non-pd cell, or a pd series missing a
        # budgeted role, raises there (T4.1), never a pooled label.
        cl.write_window_regime(
            handle.window_dir,
            t_start=float(t_start),
            t_end=float(t_end),
            attainment=_attainment(requests_rows_n),
            role_budgets=self._role_budgets(cell.budget_plan, cage_stats_rows),
        )
        metrics_path = _atomic_write_json(
            handle.window_dir / WINDOW_METRICS_NAME, _normalize_json(experiment_summary)
        )

        # S0-15: hash ledger written at write time — journal every artifact of
        # this emission (window files + the updated cell.json + the manifest
        # when this emission created it).
        journal_paths = sorted(p for p in handle.window_dir.iterdir() if p.is_file())
        journal_paths.append(self.cell_dir / "cell.json")
        if self._manifest_created:
            journal_paths.append(self.run_root / "manifest.json")
            self._manifest_created = False
        append_write_time_hashes(self.run_root, journal_paths)

        print(
            f"[campaign] window {handle.window_key} emitted -> {handle.window_dir} "
            f"({len(requests_rows_n)} request row(s); sentinel {metrics_path.name})"
        )
        return handle


# ---------------------------------------------------------------------------
# CLI — the shell runners' row-key oracle (kept import-light)
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Campaign-session helpers (task #116). Subcommand cell-dir prints "
            "the v2 cell directory ($CAGE_CAMPAIGN_ROOT/cells/<row_key>) for one "
            "runner cell — the row key is minted by CellSpec, never hand-built, "
            "so the shell resume gates and the writer share ONE derivation."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)
    cell_dir = sub.add_parser("cell-dir", help="print the v2 cell directory for one cell")
    cell_dir.add_argument("--baseline", required=True)
    cell_dir.add_argument("--baseline-label", default=None)
    cell_dir.add_argument("--model", required=True)
    cell_dir.add_argument("--backend", default="vllm")
    args = parser.parse_args(argv)

    if args.command == "cell-dir":
        root = (os.environ.get("CAGE_CAMPAIGN_ROOT") or "").strip()
        if not root:
            print("ERROR: CAGE_CAMPAIGN_ROOT is unset (campaign mode only)", file=sys.stderr)
            return 2
        try:
            spec = derive_cell_spec(
                baseline=args.baseline,
                baseline_label=args.baseline_label,
                backend=args.backend,
                model=args.model,
            )
        except CampaignSessionError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(Path(root) / "cells" / spec.to_row_key())
        return 0
    return 2  # pragma: no cover — argparse enforces the subcommand


if __name__ == "__main__":
    raise SystemExit(main())
