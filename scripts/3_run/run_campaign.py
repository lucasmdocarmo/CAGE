#!/usr/bin/env python3
"""
Order:     stage 3 — THE campaign sweep driver (tranche P1, audit gap G1); 'plan' runs locally before any provisioning, 'run' executes on the pod after 2_serving; ends by invoking seal_campaign_run.py
Objective: Enumerate ONE session's registered D6 grid into a reviewable execution-plan JSON ('plan', pure), then execute the plan cell-by-cell with engine-relaunch boundaries, per-window resume and fail-continue ('run')
Cloud:     both

The single command that runs a campaign session (charter §6.1/§6.8/§7.6.1).

PLAN / EXECUTE split (the design's load-bearing decision): ``plan`` is PURE —
it reads the P6 floor table, enumerates the session's REGISTERED grid into an
ordered list of steps, and writes nothing except ``--out``. No subprocess, no
GPU, no network. That purity is what makes the driver testable offline and
the plan JSON the artifact an operator reviews BEFORE any GPU spends a cent.
``run`` consumes a plan and shells the real scripts:

    python3 scripts/3_run/run_campaign.py plan --session a \\
        --floor-table results/preflight/floor_table_a.json \\
        --window-duration-s 300 --out plan_a.json
    python3 scripts/3_run/run_campaign.py run --plan plan_a.json \\
        --campaign-root results/<campaign>/a/<run_id> [--seal]

Frozen Wave-1 contracts CONSUMED here (never modified):

- ``run_experiment.py --campaign-root`` emits v2 windows; cell identity
  reaches it via the ``CAGE_CELL_*`` env seam
  (src/orchestration/campaign_session.derive_cell_spec, explicit-axes path).
- Launcher budget envs: ``CAGE_KV_BUDGET_BYTES`` (vLLM primary knob) /
  ``CAGE_SGLANG_MAX_TOTAL_TOKENS`` (SGLang token dial) — values derived via
  ``cache_budget.plan_budget`` from the floor table's demand rows, never
  re-derived by hand.
- ``cellspec`` mints every row key (``CellSpec.to_row_key()`` — never
  hand-built) and is the one legality gate for tuples.
- ``campaign_layout.SESSIONS``/``DATASET_IDS`` are the session/dataset
  vocabularies; ``seal_campaign_run.py`` is the one sealer.

Grid registration (§7.6.1 — Group A / session 'a' is the only grid this
driver registers so far; sessions b/cd-act1/cd-act2 refuse loudly until their
registrations land with the Wave-3 topology work):

- F1  locality (prefix ON, sub-pressure): B1-B12 × {vllm, sglang} × the four
  QA datasets — no budget/rate grid (one config per cell).
- F1 HF oracle: EXACTLY the reduced 10-cell set {B3 × all 4 datasets;
  B1, B2, B6 × squad_v2 + qasper} on the in-process hf engine.
- F2  pressure (prefix OFF): FRESH set × {vllm, sglang} × the FULL §6.1
  factorial r ∈ {1.5, 1.0, 0.75, 0.5, 0.25} × 6 rate fractions of λ*.
- F3  interaction (prefix ON × pressure): REUSE set × {vllm, sglang} × the
  §6.8 reduced 3×3 grid r ∈ {1.0, 0.5, 0.25} × {0.85, 0.95, 1.05}·λ*.
- Replications: 3 measurement windows per grid point (D6 §6.3).
- BLOCKED cells are ENUMERATED (never silently dropped) but carry a non-null
  ``blocked_on``: DIST tp-overlay cells until their registration lands, PD
  cells on any engine without a PD launcher (today: everything but vllm),
  and any cell whose arm demands a serving lever its engine's FROZEN
  launcher cannot provide (today: retr-store on sglang —
  manage_sglang_server.sh has no KV-store connector knob). ``run`` refuses a
  plan with blocked cells unless the operator passes ``--skip-blocked``
  explicitly (loud partial execution: skipped cells are reported per-cell
  and gate ``--seal``).
- PD cells (T3.2): a vllm DIST/pd cell is EXECUTABLE — its relaunch step
  invokes manage_vllm_pd.sh with the §6.5 per-role byte budgets (the
  registered dist_pd_split of floor(r×D) at the registered dist_budget_r)
  plus CAGE_TELEMETRY_ENDPOINTS for role-tagged telemetry. The cell carries
  ``blocked_on=null`` but ``gate: "--allow-pd + PD preflight smoke"``:
  ``run`` executes pd cells ONLY under the explicit ``--allow-pd`` flag,
  because the PD data path is unverified until the Run-C-prime preflight PD
  smoke passes (and until then the campaign provenance gate refusing is the
  correct downstream outcome).

Cell BEHAVIOR realization (repair of the T1.2 verifier blocker): the runner's
``--baseline`` token selects only the serving PIPELINE; the arm's remaining
behavior rides explicit argv/launch-env, mirrored from the shell drivers —
corpus arms get ``--corpus-prefix-budget`` (run_prefix_envelope.sh), retr-comp
gets ``--context-source retrieved`` (run_compression.sh's Phase-2-confound
fix), the reranker is pinned EXPLICITLY on every retrieval arm so B5-vs-B6
stays the one pre-registered reranker ablation, trunc arms carry their
registered truncation knob, corpus-comp launches the server with the fp8 KV
dtype env, and retr-store launches vLLM with the LMCache connector env
(run_kv_store.sh). Identity STILL rides only the CAGE_CELL_* seam — behavior
flags never mint identity.

Ordering minimizes engine relaunches: cells sort on (engine, prefix_mode,
model, budget_r, kv_dtype, connector, rate); every serving-config change is
an explicit ``relaunch`` step in the plan carrying the launcher argv + launch
env (budget, KV dtype, connector), so the relaunch count is exactly the
number of distinct EXECUTABLE serving configs (blocked cells launch nothing).

Failure doctrine: a failed cell writes a ``.STATUS-<dataset>`` sentinel
(dot-named so the §5 seal scope — which refuses any non-journaled file under
cells/ — skips it; dataset-suffixed because F1 row keys are shared by four
datasets and a forensic record must name WHICH one failed) and execution
CONTINUES; a later successful (or already-complete) pass of the same
cell×dataset REMOVES the sentinel — a stale "failed" record on healthy data
is a false forensic trail. The final summary matrix prints per-cell outcomes
and the exit code is nonzero if anything failed. A failed RELAUNCH fails
every cell up to the next relaunch boundary (a cell must never run against
the wrong serving config — that would be a silently mislabeled cell).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.cellspec import (  # noqa: E402
    BASELINES,
    CellSpec,
    FRESH_SET,
    REUSE_SET,
)
from src.orchestration.campaign_layout import (  # noqa: E402
    DATASET_IDS,
    RUN_ID_RE,
    SESSIONS,
    WINDOW_DIR_RE,
)
from src.orchestration.cache_budget import (  # noqa: E402
    CacheBudgetError,
    plan_budget,
)
from src.orchestration.load_generator import (  # noqa: E402
    D6_RATE_FRACTIONS,
    D6_REDUCED_RATE_FRACTIONS,
)

__all__ = [
    "PLAN_SCHEMA",
    "PlanError",
    "RunError",
    "SessionGrid",
    "SESSION_GRIDS",
    "PlannedCell",
    "build_plan",
    "enumerate_cells",
    "load_floor_table",
    "load_plan",
    "main",
]

# v2 (2026-09-01): pd steps added required keys gate/topology/pd — a v1 plan
# predates them, so 'run' must refuse it via the schema check and the operator
# re-plans (never silently execute a plan missing pd semantics).
PLAN_SCHEMA = "cage-campaign-plan-v2"
FLOOR_TABLE_SCHEMA = "floor-table-v1"

#: §6.1 pre-registered budget ratios r = B/D (Group-A anchor factorial; r=1.5
#: is the comfortable control rung, never in-regime by design).
FULL_BUDGET_LEVELS: Tuple[float, ...] = (1.5, 1.0, 0.75, 0.5, 0.25)
#: §6.8 pruning rule: the reduced grid for every non-anchor pressure family.
REDUCED_BUDGET_LEVELS: Tuple[float, ...] = (1.0, 0.5, 0.25)

#: Rate fractions come from the DISPATCHER's registered constants
#: (src/orchestration/load_generator.py) — the plan and the load generator can
#: never disagree about the grid (same rule build_floor_table.py follows).
FULL_RATE_FRACTIONS: Tuple[float, ...] = tuple(D6_RATE_FRACTIONS)
REDUCED_RATE_FRACTIONS: Tuple[float, ...] = tuple(D6_REDUCED_RATE_FRACTIONS)

#: §7.6.1 F1 row: the four quality-instrumented QA datasets (D5 items 1-4).
QA_DATASETS: Tuple[str, ...] = ("squad_v2", "hotpotqa", "musique", "qasper")

#: D6 §6.3: >= 3 replications per grid point; one replication = one §1
#: measurement window (= one runner trial).
REPLICATIONS = 3

#: The task text the operator sees on an enumerated-but-unrunnable DIST
#: tp-overlay cell (the T3.1 TP env exists, but the tp-overlay grid/driver
#: registration has not landed — enumerated blocked, never guessed at).
TP_DIST_BLOCKED_ON = "tp-topology overlay registration (Wave-3)"

#: The gate label every EXECUTABLE pd cell carries (blocked_on stays null —
#: the launcher exists — but 'run' still refuses without the operator's
#: explicit --allow-pd consent until the Run-C-prime preflight PD smoke).
PD_GATE = "--allow-pd + PD preflight smoke"

#: manage_vllm_pd.sh default ports, MIRRORED here (the launcher's
#: CAGE_PD_PREFILL_PORT/CAGE_PD_DECODE_PORT defaults) so the emitted
#: telemetry endpoints point at the instances the launcher actually starts —
#: a port override must ride BOTH sides together.
PD_PREFILL_PORT = 8100
PD_DECODE_PORT = 8200

#: CAGE_TELEMETRY_ENDPOINTS value for pd relaunches (T4.1 role=url grammar,
#: run_experiment.parse_telemetry_endpoints): one role-tagged sampler per
#: instance, so PD windows carry per-role serving telemetry.
PD_TELEMETRY_ENDPOINTS = (
    f"prefill=http://localhost:{PD_PREFILL_PORT},"
    f"decode=http://localhost:{PD_DECODE_PORT}"
)

#: charter model slug -> HF id, as the launchers and run_experiment --model
#: expect it (the runner maps the id back to the slug via
#: campaign_session.HF_MODEL_SLUGS; deepseek's canonical id is the -0324
#: checkpoint per §7.6 Group D).
HF_ID_OF_SLUG: Dict[str, str] = {
    "qwen3-14b": "Qwen/Qwen3-14B",
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen3-next-80b": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3-0324",
}

#: engine axis value -> run_experiment --backend token (frozen runner CLI).
BACKEND_OF_ENGINE: Dict[str, str] = {
    "vllm": "vllm",
    "sglang": "sglang",
    "hf": "hf-oracle",
}

#: arm -> the runner --baseline token that selects the base serving PIPELINE
#: (pilot pipeline vocabulary; frozen argparse choices). The token alone is
#: NOT the arm's behavior — the shell drivers add behavior-bearing flags/env
#: beyond it (run_prefix_envelope.sh, run_compression.sh, run_kv_store.sh),
#: and _behavior_argv/_serving_config mirror exactly those additions. Cell
#: IDENTITY never comes from this token — it rides the CAGE_CELL_* env seam
#: (derive_cell_spec's explicit-axes path wins over any label).
ARM_RUNNER_BASELINE: Dict[str, str] = {
    "gold-fresh": "no_cache",
    "gold-reuse": "prefix_cache",
    "corpus-reuse": "prefix_cache",
    "corpus-fresh": "no_cache",
    "retr-fresh": "rag",
    "retr-reuse": "hybrid",
    "retr-store": "rag",
    "retr-comp": "compressed_rag",
    "corpus-comp": "compressed_cag",
    "retr-trunc": "rag",
    "corpus-trunc": "prefix_cache",
}

#: The ONE pre-registered cross-encoder reranker (§7.1 ranking rule: ablated
#: exactly once, B5 vs B6). Pinned EXPLICITLY in every retrieval cell's argv —
#: inheriting the runner's default silently would let a runner-default drift
#: rewrite the pre-registered ablation.
RERANKER_MODEL = "BAAI/bge-reranker-large"

#: CellSpec retriever axis -> the runner's retrieval argv. The runner splits
#: the axis into --retriever (pipeline) + --reranker-model (ranking stage);
#: 'rerank' = dense retrieval + the pinned cross-encoder, 'dense' = the same
#: retrieval with ranking DISABLED ('none' is the runner's documented off
#: switch, normalize_reranker_model). bm25/rrf have no registered argv here
#: yet — enumerating them refuses (fail closed) rather than guessing.
RETRIEVER_ARGV: Dict[str, Tuple[str, ...]] = {
    "dense": ("--retriever", "dense", "--reranker-model", "none"),
    "rerank": ("--retriever", "dense", "--reranker-model", RERANKER_MODEL),
}

#: Arms serving the shared corpus-as-prefix block (true CAG, Chan et al.
#: 2412.15605): all get --corpus-prefix-budget. corpus-trunc is handled apart
#: (its budget is the TRUNCATED one — that difference IS the B12-vs-B3 slot).
CORPUS_BLOCK_ARMS = frozenset({"corpus-reuse", "corpus-fresh", "corpus-comp"})

#: arm -> serving-side levers that must be applied AT SERVER LAUNCH (they are
#: part of the relaunch-boundary identity, not the cell argv):
#: - corpus-comp: fp8 KV cache dtype (run_compression.sh launches the server
#:   with VLLM_KV_CACHE_DTYPE=fp8; run_experiment --kv-cache-dtype is
#:   record-only per its own help text).
#: - retr-store: the LMCache connector (run_kv_store.sh: the connector must
#:   ride the server launch via VLLM_KV_TRANSFER_CONFIG; Group A/B store impl
#:   is LMCache per §7.1 B8).
ARM_KV_DTYPE: Dict[str, str] = {"corpus-comp": "fp8"}
ARM_CONNECTOR: Dict[str, str] = {"retr-store": "lmcache"}

#: The exact connector JSON run_kv_store.sh passes at vLLM launch — verbatim,
#: never re-derived (a drifted kv_role would silently change the arm).
LMCACHE_KV_TRANSFER_CONFIG = (
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
)

#: engine -> (env var, value) realizing each serving lever at launch. An
#: engine ABSENT from a lever's map cannot serve that lever with the frozen
#: launcher fleet — its cells are enumerated BLOCKED (never silently served
#: without the lever: that is the mislabeled-duplicate failure mode).
#: SGLang's fp8 token is fp8_e5m2 (the launcher's own documented example);
#: the cell argv still records the semantic class 'fp8' (runner choices).
KV_DTYPE_LAUNCH_ENV: Dict[str, Tuple[str, str]] = {
    "vllm": ("VLLM_KV_CACHE_DTYPE", "fp8"),
    "sglang": ("SGLANG_KV_CACHE_DTYPE", "fp8_e5m2"),
}
CONNECTOR_LAUNCH_ENV: Dict[str, Tuple[str, str]] = {
    "vllm": ("VLLM_KV_TRANSFER_CONFIG", LMCACHE_KV_TRANSFER_CONFIG),
}

#: blocked_on text for retr-store cells on an engine with no connector knob
#: (today: sglang — manage_sglang_server.sh exposes budget/dtype/prefix only).
SGLANG_STORE_BLOCKED_ON = (
    "sglang KV-store connector wiring (manage_sglang_server.sh, Wave-3)"
)

#: cache_budget primary-knob flag -> the launcher env that carries it
#: (frozen launcher contract; the vLLM blocks-override is the FALLBACK knob
#: and is never emitted here — setting both is a launcher refusal).
_BUDGET_FLAG_ENV: Dict[str, str] = {
    "--kv-cache-memory-bytes": "CAGE_KV_BUDGET_BYTES",
    "--max-total-tokens": "CAGE_SGLANG_MAX_TOTAL_TOKENS",
}

#: Production launcher argv prefixes, per engine (repo-root-relative; 'run'
#: executes with cwd=REPO_ROOT). hf has NO launcher: the oracle runs
#: in-process inside run_experiment (backend hf-oracle), so hf cells never
#: get a relaunch boundary.
#: PD launcher registry key — vllm's PD topology rides its OWN launcher
#: (manage_vllm_pd.sh), never the single-instance script.
PD_LAUNCHER_KEY = "vllm:pd"

DEFAULT_LAUNCHER_CMDS: Dict[str, Tuple[str, ...]] = {
    "vllm": ("bash", "scripts/2_serving/manage_vllm_server.sh"),
    "sglang": ("bash", "scripts/2_serving/manage_sglang_server.sh"),
    PD_LAUNCHER_KEY: ("bash", "scripts/2_serving/manage_vllm_pd.sh"),
}
DEFAULT_RUNNER_CMD: Tuple[str, ...] = ("python3", "scripts/3_run/run_experiment.py")
DEFAULT_SEAL_CMD: Tuple[str, ...] = ("python3", "scripts/3_run/seal_campaign_run.py")

_LAMBDA_PENDING_BASIS = "kv-bound-only [pending calibration]"
_LAMBDA_CALIBRATED_BASIS = "calibrated min(lambda_KV, lambda_compute)"

_PRESSURE_FAMILIES = frozenset({"F2", "F3"})


class PlanError(ValueError):
    """A plan that cannot be enumerated honestly (fail closed)."""


class RunError(RuntimeError):
    """A run invocation refused before executing anything."""


def _baseline_num(baseline_id: str) -> int:
    return int(baseline_id[1:])


def _ordered(baseline_ids: frozenset) -> Tuple[str, ...]:
    """Deterministic B-number order for a baseline-id set (B2 < B10)."""
    return tuple(sorted(baseline_ids, key=_baseline_num))


# ---------------------------------------------------------------------------
# Session grid registration (§7.6 / §7.6.1 — THE registered enumeration
# source; no hand lists anywhere downstream)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionGrid:
    """One session's registered grid — every axis the enumerator may use."""

    session: str
    group: str  # §7.6 group letter, for the plan header
    model: str  # charter model slug (one run = one model, RESULTS_LAYOUT §3)
    f1_baselines: Tuple[str, ...]
    f1_engines: Tuple[str, ...]
    f1_datasets: Tuple[str, ...]
    # (baseline_id, datasets) rows of the reduced HF-oracle slice (§7.3/§7.4:
    # hf is the sub-pressure F1 oracle ONLY — structurally enforced below).
    hf_oracle_cells: Tuple[Tuple[str, Tuple[str, ...]], ...]
    f2_baselines: Tuple[str, ...]
    f2_engines: Tuple[str, ...]
    f2_budgets: Tuple[float, ...]
    f2_rates: Tuple[float, ...]
    f2_dataset: str
    f3_baselines: Tuple[str, ...]
    f3_engines: Tuple[str, ...]
    f3_budgets: Tuple[float, ...]
    f3_rates: Tuple[float, ...]
    f3_dataset: str
    # (baseline_id, engine, topology) distributed-overlay cells — enumerated,
    # always blocked on the Wave-3 P/D launcher.
    dist_cells: Tuple[Tuple[str, str, str], ...] = ()
    replications: int = REPLICATIONS
    # Registered BEHAVIOR knobs (reviewable in the plan header — these are
    # design registrations, not measurements, so they live HERE, never as
    # silent defaults inside step builders):
    # - corpus_prefix_budget_tokens: the shared corpus-block token budget for
    #   the corpus arms (B3/B4/B10); the pilot convention is 2800
    #   (run_prefix_envelope.sh CORPUS_BUDGET, fits the uniform 4096 max-len
    #   regime with query+generation headroom).
    # - corpus_trunc_budget_tokens: B12's TRUNCATED corpus budget ("store
    #   less than you know") — the one-slot B12-vs-B3 contrast; ~2x matches
    #   the compression arms' ratio (run_compression.sh's 2x2 convention).
    # - retr_trunc_kept_docs: B11's rank-truncation ("read less of what you
    #   found") — serve only the top-N of the ranked list via
    #   --max-context-docs; kept/dropped labels come from the runner.
    # - dist_pd_split / dist_budget_r (T3.2): the §6.5 P/D split is an
    #   EXPLICIT recorded input, never a silent default INSIDE the budget
    #   planner — registering it HERE (reviewable in the plan header path)
    #   is what makes the pd relaunch honest. dist_budget_r is the budget
    #   rung the DIST overlay serves at (protocol-cost isolation runs at the
    #   comfortable r=1.0 rung, not under pressure); the pd byte budgets are
    #   the split of floor(dist_budget_r × D) from the floor table.
    corpus_prefix_budget_tokens: int = 2800
    corpus_trunc_budget_tokens: int = 1400
    retr_trunc_kept_docs: int = 1
    dist_pd_split: float = 0.5
    dist_budget_r: float = 1.0

    def __post_init__(self) -> None:
        problems: List[str] = []
        for name in (
            "corpus_prefix_budget_tokens",
            "corpus_trunc_budget_tokens",
            "retr_trunc_kept_docs",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                problems.append(f"{name}={value!r} must be an integer >= 1")
        if (
            isinstance(self.corpus_trunc_budget_tokens, int)
            and isinstance(self.corpus_prefix_budget_tokens, int)
            and self.corpus_trunc_budget_tokens >= self.corpus_prefix_budget_tokens
        ):
            problems.append(
                f"corpus_trunc_budget_tokens={self.corpus_trunc_budget_tokens} "
                f"must be < corpus_prefix_budget_tokens="
                f"{self.corpus_prefix_budget_tokens} — B12 truncates the B3 "
                "corpus, or it is not a truncation arm"
            )
        # T3.2 pd registration knobs: same fail-closed rules the budget
        # planner enforces (a grid carrying an illegal split would refuse at
        # relaunch-build time anyway; refusing HERE names the registration).
        if not (
            isinstance(self.dist_pd_split, float)
            and 0.0 < self.dist_pd_split < 1.0
        ):
            problems.append(
                f"dist_pd_split={self.dist_pd_split!r} must be a float strictly "
                "inside (0, 1) — §6.5 records the P/D split explicitly"
            )
        if not (
            isinstance(self.dist_budget_r, (int, float))
            and not isinstance(self.dist_budget_r, bool)
            and math.isfinite(self.dist_budget_r)
            and self.dist_budget_r > 0
        ):
            problems.append(
                f"dist_budget_r={self.dist_budget_r!r} must be finite and > 0"
            )
        for name in ("f1_datasets",):
            unknown = sorted(set(getattr(self, name)) - DATASET_IDS)
            if unknown:
                problems.append(f"{name} has non-§1 dataset id(s) {unknown}")
        for name in ("f2_dataset", "f3_dataset"):
            if getattr(self, name) not in DATASET_IDS:
                problems.append(f"{name}={getattr(self, name)!r} is not a §1 dataset id")
        # hf is the sub-pressure F1 oracle only (§7.3): a registration that
        # put it on a pressure family would be a structural impossibility,
        # not an input error — assert, per the fail-closed doctrine.
        assert "hf" not in self.f2_engines and "hf" not in self.f3_engines, (
            "SessionGrid registered engine=hf under a pressure family — "
            "structurally impossible (§7.3: hf is the sub-pressure F1 oracle)"
        )
        if self.replications < 1:
            problems.append(f"replications={self.replications} must be >= 1")
        if problems:
            raise PlanError("; ".join(problems))


#: Session 'a' = charter §7.6 Group A (Qwen3-14B anchor, single 80 GB GPU):
#: the full controlled grid — the only group at full pressure density (§6.8).
#: F2's pressure workload is Qasper (D5 item 4: THE quality-instrumented
#: pressure workload); F3's primary interaction workload is Qasper's
#: multi-question-per-paper reuse (D5 F3; the HotpotQA engineered-overlap
#: store rides the workload manifest, not a separate cell axis). Group A
#: carries no distributed overlay (§7.6.1 bottom row: "—").
#: LMDeploy F1 carriage on the anchor is a Wave-3 registration alongside its
#: launcher wiring; vllm+sglang are the two engines registered here.
SESSION_GRIDS: Dict[str, SessionGrid] = {
    "a": SessionGrid(
        session="a",
        group="A",
        model="qwen3-14b",
        f1_baselines=_ordered(frozenset(BASELINES)),
        f1_engines=("vllm", "sglang"),
        f1_datasets=QA_DATASETS,
        hf_oracle_cells=(
            ("B1", ("squad_v2", "qasper")),
            ("B2", ("squad_v2", "qasper")),
            ("B3", QA_DATASETS),
            ("B6", ("squad_v2", "qasper")),
        ),
        f2_baselines=_ordered(FRESH_SET),
        f2_engines=("vllm", "sglang"),
        f2_budgets=FULL_BUDGET_LEVELS,
        f2_rates=FULL_RATE_FRACTIONS,
        f2_dataset="qasper",
        f3_baselines=_ordered(REUSE_SET),
        f3_engines=("vllm", "sglang"),
        f3_budgets=REDUCED_BUDGET_LEVELS,
        f3_rates=REDUCED_RATE_FRACTIONS,
        f3_dataset="qasper",
        dist_cells=(),
        corpus_prefix_budget_tokens=2800,
        corpus_trunc_budget_tokens=1400,
        retr_trunc_kept_docs=1,
    ),
}


def get_session_grid(session: str) -> SessionGrid:
    """The registered grid for a session — fail closed on both unknowns."""
    if session not in SESSIONS:
        raise PlanError(
            f"unknown session {session!r} — the §1 session vocabulary is "
            f"{sorted(SESSIONS)} (campaign_layout.SESSIONS)"
        )
    grid = SESSION_GRIDS.get(session)
    if grid is None:
        raise PlanError(
            f"session {session!r} is in the §1 vocabulary but its grid is NOT "
            f"yet registered in this driver (registered: {sorted(SESSION_GRIDS)}); "
            "sessions b/cd-* land with the Wave-3 topology registrations — "
            "refusing to fabricate a grid"
        )
    return grid


# ---------------------------------------------------------------------------
# Floor table (T2.4 floor-table-v1) — REQUIRED demand/λ* source
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FloorTable:
    """Validated floor-table-v1 content: demand + λ* rows keyed by r."""

    path: Path
    sha256: str
    model: str
    engine: str
    kv_dtype: str
    grid: str
    rows: Dict[float, Dict[str, Any]]

    def row(self, r: float) -> Dict[str, Any]:
        for key, row in self.rows.items():
            if math.isclose(key, r, rel_tol=0.0, abs_tol=1e-9):
                return row
        raise PlanError(
            f"floor table {self.path} (grid={self.grid!r}) has no row for "
            f"r={r:g} — its r values are {sorted(self.rows, reverse=True)}; "
            "rebuild it with the grid that covers this session (§6.1/§6.8)"
        )


def load_floor_table(path: Path) -> FloorTable:
    """Load + validate the REQUIRED T2.4 floor-table-v1 JSON (fail closed).

    Demand D is engine-independent by design (P2: bytes are the same physical
    quantity on every engine), so one table serves both pressure engines of a
    session; its ``engine`` field is recorded in the plan as the calibration
    binding, never used to gate enumeration.
    """
    path = Path(path)
    if not path.is_file():
        raise PlanError(
            f"floor table not found: {path} — the plan REQUIRES the P6 "
            "pre-measurement prediction artifact (build_floor_table.py); "
            "a plan without it would fabricate demand/λ*"
        )
    raw = path.read_bytes()
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PlanError(f"floor table {path} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != FLOOR_TABLE_SCHEMA:
        raise PlanError(
            f"floor table {path} schema={doc.get('schema') if isinstance(doc, dict) else type(doc).__name__!r} "
            f"is not {FLOOR_TABLE_SCHEMA!r} (T2.4 contract)"
        )
    inputs = doc.get("generated_inputs")
    rows_raw = doc.get("rows")
    if not isinstance(inputs, dict) or not isinstance(rows_raw, list) or not rows_raw:
        raise PlanError(
            f"floor table {path} is missing generated_inputs and/or a non-empty rows[]"
        )
    rows: Dict[float, Dict[str, Any]] = {}
    problems: List[str] = []
    for i, row in enumerate(rows_raw):
        if not isinstance(row, dict):
            problems.append(f"rows[{i}] is not an object")
            continue
        bad = False
        for req in ("r", "demand_bytes", "lambda_star_pred_rps"):
            value = row.get(req)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                problems.append(f"rows[{i}].{req}={value!r} must be a number")
                bad = True
        if not bad:
            rows[float(row["r"])] = row
    if problems:
        raise PlanError(f"floor table {path}: " + "; ".join(problems))
    return FloorTable(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        model=str(inputs.get("model")),
        engine=str(inputs.get("engine")),
        kv_dtype=str(inputs.get("kv_dtype")),
        grid=str(inputs.get("grid")),
        rows=rows,
    )


# ---------------------------------------------------------------------------
# Enumeration (registration -> cells) + ordering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedCell:
    """One enumerated cell: the tuple, its numbered baseline, its workload."""

    spec: CellSpec
    baseline_id: str
    dataset: str
    blocked_on: Optional[str] = None


def _launch_blocked_on(spec: CellSpec) -> Optional[str]:
    """Non-null when the arm demands a serving lever this engine's FROZEN
    launcher cannot provide.

    Such a cell is enumerated BLOCKED (the operator sees the debt in the
    plan) — the alternative, serving it without the lever, would write a
    byte-identical mislabeled duplicate of the lever-free arm (retr-store
    without its connector IS plain rag; corpus-comp without fp8 IS
    corpus-reuse), the exact fail-closed violation this module exists to
    prevent.
    """
    kv_dtype = ARM_KV_DTYPE.get(spec.arm)
    connector = ARM_CONNECTOR.get(spec.arm)
    if kv_dtype is None and connector is None:
        return None
    if spec.engine == "hf":
        lever = "the fp8 KV-dtype lever" if kv_dtype else "a KV-store connector"
        return f"in-process hf oracle has no launch seam for {lever}"
    if connector is not None and spec.engine not in CONNECTOR_LAUNCH_ENV:
        if spec.engine == "sglang":
            return SGLANG_STORE_BLOCKED_ON
        return f"{spec.engine} launcher has no KV-store connector knob"
    if kv_dtype is not None and spec.engine not in KV_DTYPE_LAUNCH_ENV:
        return f"{spec.engine} launcher has no KV-cache dtype knob"
    return None


def enumerate_cells(grid: SessionGrid) -> List[PlannedCell]:
    """Enumerate every registered cell of a session (§7.6.1 matrix, one row).

    All legality is delegated to ``CellSpec.__post_init__`` — the enumerator
    builds tuples, never validates axes itself. Empty enumeration refuses:
    a session with zero cells is a registration bug, not a no-op.
    """
    cells: List[PlannedCell] = []

    def _planned(spec: CellSpec, bid: str, dataset: str) -> PlannedCell:
        # Launch-realizability is decided AT ENUMERATION so the plan carries
        # the debt visibly (blocked_on) instead of 'run' discovering it.
        return PlannedCell(
            spec=spec,
            baseline_id=bid,
            dataset=dataset,
            blocked_on=_launch_blocked_on(spec),
        )

    # F1 — locality, prefix ON, sub-pressure (no budget/rate coordinates).
    for bid in grid.f1_baselines:
        for engine in grid.f1_engines:
            for dataset in grid.f1_datasets:
                cells.append(
                    _planned(
                        CellSpec.from_baseline(
                            bid, engine=engine, model=grid.model, family="F1"  # type: ignore[arg-type]
                        ),
                        bid,
                        dataset,
                    )
                )

    # F1 HF-oracle reduced slice (§7.3: hf carries family F1 only —
    # CellSpec itself refuses anything else, making the guard structural).
    for bid, datasets in grid.hf_oracle_cells:
        for dataset in datasets:
            cells.append(
                _planned(
                    CellSpec.from_baseline(
                        bid, engine="hf", model=grid.model, family="F1"  # type: ignore[arg-type]
                    ),
                    bid,
                    dataset,
                )
            )

    # F2 — pressure, prefix OFF, budget × rate factorial (§6.1).
    for bid in grid.f2_baselines:
        for engine in grid.f2_engines:
            for r in grid.f2_budgets:
                for frac in grid.f2_rates:
                    cells.append(
                        _planned(
                            CellSpec.from_baseline(
                                bid,
                                engine=engine,  # type: ignore[arg-type]
                                model=grid.model,  # type: ignore[arg-type]
                                family="F2",
                                budget_r=r,
                                rate_frac=frac,
                            ),
                            bid,
                            grid.f2_dataset,
                        )
                    )

    # F3 — interaction, prefix ON × pressure, reduced grid (§6.8).
    for bid in grid.f3_baselines:
        for engine in grid.f3_engines:
            for r in grid.f3_budgets:
                for frac in grid.f3_rates:
                    cells.append(
                        _planned(
                            CellSpec.from_baseline(
                                bid,
                                engine=engine,  # type: ignore[arg-type]
                                model=grid.model,  # type: ignore[arg-type]
                                family="F3",
                                budget_r=r,
                                rate_frac=frac,
                            ),
                            bid,
                            grid.f3_dataset,
                        )
                    )

    # DIST — topology overlay. pd cells on vllm are EXECUTABLE since T3.2
    # (manage_vllm_pd.sh + pd_proxy.py); they carry the --allow-pd gate label
    # instead of blocked_on. Everything else stays enumerated-but-blocked:
    # pd on an engine with no PD launcher, and the tp overlay until its
    # registration lands ('run' refuses blocked cells loudly).
    for bid, engine, topology in grid.dist_cells:
        spec = CellSpec.from_baseline(
            bid,
            engine=engine,  # type: ignore[arg-type]
            model=grid.model,  # type: ignore[arg-type]
            family="DIST",
            topology=topology,  # type: ignore[arg-type]
        )
        if topology == "pd":
            if engine != "vllm":
                blocked: Optional[str] = (
                    f"{engine} PD launcher (manage_vllm_pd.sh is vLLM-only; "
                    "other engines' PD rides its own Wave-3 wiring)"
                )
            else:
                blocked = _launch_blocked_on(spec)  # arm levers still gate
        else:
            blocked = TP_DIST_BLOCKED_ON
        cells.append(
            PlannedCell(
                spec=spec,
                baseline_id=bid,
                dataset=grid.f1_datasets[0],
                blocked_on=blocked,
            )
        )

    if not cells:
        raise PlanError(
            f"session {grid.session!r} enumerated ZERO cells — an empty "
            "registration is a bug, not a no-op (fail closed)"
        )
    return cells


def _prefix_off(spec: CellSpec) -> bool:
    """Pinned rule: F2 is the prefix-OFF family; every other family serves ON."""
    return spec.family == "F2"


def _coord_key(value: Optional[float]) -> Tuple[int, float]:
    # None (sub-pressure) sorts BEFORE any budget so each engine's F1 block
    # runs first on the plain (budget-free) server config.
    return (0, 0.0) if value is None else (1, float(value))


def _lever_key(value: Optional[str]) -> str:
    # None (no lever) sorts before any lever value; a plain string keeps the
    # order total without inventing a numeric encoding.
    return "" if value is None else value


def _sort_key(cell: PlannedCell) -> Tuple[Any, ...]:
    """Relaunch-minimizing order:
    (engine, prefix_mode, model, budget_r, kv_dtype, connector, rate).

    The launch levers (kv_dtype, connector) sort BEFORE rate: they are part
    of the serving config while rate is client-side, so lever-bearing cells
    must group contiguously across rate levels or every rate change would
    straddle a lever boundary and force extra relaunches. Trailing components
    (family, baseline number, dataset) only make the order
    total/deterministic — they never split a serving config.
    """
    spec = cell.spec
    return (
        spec.engine,
        # Topology sorts right after engine: pd cells serve a DIFFERENT
        # process stack, so they must group contiguously and never interleave
        # single-topology cells (which would force extra relaunches).
        spec.topology,
        1 if _prefix_off(spec) else 0,
        spec.model,
        _coord_key(spec.budget_r),
        _lever_key(ARM_KV_DTYPE.get(spec.arm)),
        _lever_key(ARM_CONNECTOR.get(spec.arm)),
        _coord_key(spec.rate_frac),
        spec.family,
        _baseline_num(cell.baseline_id),
        cell.dataset,
    )


#: (engine, prefix_off, model, budget_r, kv_dtype, connector, topology)
_ServingConfig = Tuple[
    str, bool, str, Optional[float], Optional[str], Optional[str], str
]


def _serving_config(cell: PlannedCell) -> Optional[_ServingConfig]:
    """The relaunch-boundary identity; None for the in-process hf oracle.

    Rate is CLIENT-side load (never a server dial), so it is deliberately
    absent — rate changes must not force a relaunch. The launch levers
    (kv_dtype for corpus-comp, connector for retr-store) ARE server dials
    (run_compression.sh / run_kv_store.sh apply them at launch), so they are
    relaunch-boundary dimensions — and so is topology (T3.2: a pd stack is a
    different process set than a single server; sharing a boundary would run
    one topology's cells against the other's serving stack).
    """
    spec = cell.spec
    if spec.engine == "hf":
        return None
    return (
        spec.engine,
        _prefix_off(spec),
        spec.model,
        spec.budget_r,
        ARM_KV_DTYPE.get(spec.arm),
        ARM_CONNECTOR.get(spec.arm),
        spec.topology,
    )


# ---------------------------------------------------------------------------
# Step builders (plan JSON content)
# ---------------------------------------------------------------------------


def _budget_env(
    engine: str,
    model: str,
    r: float,
    floor: FloorTable,
    served_kv_dtype: Optional[str] = None,
) -> Tuple[Dict[str, str], int]:
    """Launcher budget env for one serving config, via cache_budget.plan_budget.

    Demand comes from the floor table row at this r (T2.4 is the ONE demand
    source); the primary knob of the resulting BudgetPlan maps onto the
    launcher env (frozen contract). The BYTE budget stays anchored to
    floor(r × D) for every config at the same r (the iso-bytes comparison
    anchor that makes B10's double-saving measurable), but token-denominated
    dials (SGLang --max-total-tokens) must convert bytes at the SERVED KV
    dtype — an fp8 server planned with bf16 arithmetic would realize only
    HALF the byte budget (§6.5 violation). Returns (env, budget_bytes_total).
    """
    row = floor.row(r)
    if served_kv_dtype is None and floor.kv_dtype != "bf16":
        # A floor table built with --kv-dtype fp8 halves bytes/token, so
        # deriving a PLAIN (non-fp8-lever) launch's token dial from it would
        # DOUBLE the token cap on a bf16 server — a silent §6.5 budget breach
        # (2026-08-31 verifier minor). P5: BF16 first; fp8 is the lever's own
        # dtype, carried via served_kv_dtype only.
        raise PlanError(
            f"floor table {floor.path} was built with kv_dtype="
            f"{floor.kv_dtype!r} but this launch is not the fp8 lever — "
            "plain launches require a bf16 floor table (rebuild with "
            "--kv-dtype bf16)"
        )
    try:
        plan = plan_budget(
            model=model,
            engine=engine,
            r=r,
            demand=int(row["demand_bytes"]),
            kv_dtype=served_kv_dtype or floor.kv_dtype,
            tp=1,
            topology="single",
        )
    except CacheBudgetError as exc:
        raise PlanError(f"budget planning refused for engine={engine} r={r:g}: {exc}") from exc
    env: Dict[str, str] = {}
    for arg in plan.engine_args:
        if arg.kind != "primary":
            continue
        flag, value = arg.args
        env_name = _BUDGET_FLAG_ENV.get(flag)
        if env_name is None:
            raise PlanError(
                f"budget planner emitted primary knob {flag!r} with no launcher "
                "env mapping — extend _BUDGET_FLAG_ENV before planning this engine"
            )
        env[env_name] = value
    if not env:
        raise PlanError(
            f"budget planner emitted no primary knob for engine={engine} r={r:g}"
        )
    return env, plan.budget_bytes_total


def _pd_budget_env(
    grid: SessionGrid, engine: str, model: str, floor: FloorTable
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """PD launcher env: the §6.5 split of floor(dist_budget_r × D) into the
    two REQUIRED per-role byte budgets, plus the role-tagged telemetry
    endpoints. Returns (env, pd-record-for-the-plan)."""
    if floor.kv_dtype != "bf16":
        # Same guard as _budget_env: DIST pd cells launch PLAIN (no fp8
        # lever), so a non-bf16 floor table would mis-denominate the pools.
        raise PlanError(
            f"floor table {floor.path} was built with kv_dtype="
            f"{floor.kv_dtype!r} but the pd relaunch is a plain launch — "
            "rebuild with --kv-dtype bf16"
        )
    row = floor.row(grid.dist_budget_r)
    try:
        plan = plan_budget(
            model=model,
            engine=engine,
            r=grid.dist_budget_r,
            demand=int(row["demand_bytes"]),
            kv_dtype=floor.kv_dtype,
            tp=1,
            topology="pd",
            pd_split=grid.dist_pd_split,  # §6.5: explicit, registered, exact-sum
        )
    except CacheBudgetError as exc:
        raise PlanError(
            f"pd budget planning refused for engine={engine} "
            f"r={grid.dist_budget_r:g} split={grid.dist_pd_split:g}: {exc}"
        ) from exc
    assert plan.pools_bytes is not None  # topology='pd' always emits pools
    prefill_bytes, decode_bytes = plan.pools_bytes
    env = {
        # The launcher REFUSES to start without both (manage_vllm_pd.sh).
        "CAGE_KV_BUDGET_BYTES_PREFILL": str(prefill_bytes),
        "CAGE_KV_BUDGET_BYTES_DECODE": str(decode_bytes),
        # T4.1 role-tagged telemetry — one sampler per instance.
        "CAGE_TELEMETRY_ENDPOINTS": PD_TELEMETRY_ENDPOINTS,
    }
    record = {
        "split": grid.dist_pd_split,
        "budget_r": grid.dist_budget_r,
        "budget_bytes_total": plan.budget_bytes_total,
        "prefill_bytes": prefill_bytes,
        "decode_bytes": decode_bytes,
    }
    return env, record


def _relaunch_step(
    config: _ServingConfig,
    grid: SessionGrid,
    floor: FloorTable,
    launcher_cmds: Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    engine, prefix_off, model, budget_r, kv_dtype, connector, topology = config
    if topology == "pd":
        # T3.2: the pd stack has its OWN launcher (two role instances +
        # proxy); its `start` verb is self-cleaning, so a relaunch boundary
        # is one `start` (the script has no restart verb by design).
        launcher = launcher_cmds.get(PD_LAUNCHER_KEY)
        if launcher is None:
            raise PlanError(
                f"no PD launcher command registered under {PD_LAUNCHER_KEY!r} "
                f"(known: {sorted(launcher_cmds)})"
            )
        argv = list(launcher) + ["start", HF_ID_OF_SLUG[model]]
        if prefix_off:
            argv.append("--no-prefix-cache")
        env, pd_record = _pd_budget_env(grid, engine, model, floor)
        return {
            "kind": "relaunch",
            "engine": engine,
            "model": model,
            "prefix_mode": "OFF" if prefix_off else "ON",
            # budget_r stays the CELL coordinate (None — DIST is the
            # topology overlay, not a pressure family); the pd byte budgets
            # ride the dedicated record + env.
            "budget_r": budget_r,
            "budget_bytes": pd_record["budget_bytes_total"],
            "kv_dtype": kv_dtype,
            "connector": connector,
            "topology": topology,
            "pd": pd_record,
            "argv": argv,
            "env": env,
        }
    launcher = launcher_cmds.get(engine)
    if launcher is None:
        raise PlanError(
            f"no launcher command registered for engine {engine!r} "
            f"(known: {sorted(launcher_cmds)})"
        )
    argv = list(launcher) + ["restart", HF_ID_OF_SLUG[model]]
    if prefix_off:
        argv.append("--no-prefix-cache")
    env: Dict[str, str] = {}
    budget_bytes: Optional[int] = None
    if budget_r is not None:
        env, budget_bytes = _budget_env(
            engine, model, budget_r, floor, served_kv_dtype=kv_dtype
        )
    # Launch levers ride the SERVER environment (run_compression.sh /
    # run_kv_store.sh conventions) — an unmapped engine here is a driver bug:
    # enumeration already blocked such cells, so no relaunch may reach this.
    if kv_dtype is not None:
        mapping = KV_DTYPE_LAUNCH_ENV.get(engine)
        if mapping is None:
            raise PlanError(
                f"relaunch for engine={engine} demands kv_dtype={kv_dtype!r} but "
                "the launcher has no dtype env — enumeration should have "
                "BLOCKED these cells (driver invariant violated)"
            )
        env[mapping[0]] = mapping[1]
    if connector is not None:
        mapping = CONNECTOR_LAUNCH_ENV.get(engine)
        if mapping is None:
            raise PlanError(
                f"relaunch for engine={engine} demands connector={connector!r} "
                "but the launcher has no connector env — enumeration should "
                "have BLOCKED these cells (driver invariant violated)"
            )
        env[mapping[0]] = mapping[1]
    return {
        "kind": "relaunch",
        "engine": engine,
        "model": model,
        "prefix_mode": "OFF" if prefix_off else "ON",
        "budget_r": budget_r,
        "budget_bytes": budget_bytes,
        "kv_dtype": kv_dtype,
        "connector": connector,
        "topology": topology,
        "pd": None,  # single-topology relaunch: no §6.5 role split
        "argv": argv,
        "env": env,
    }


def _cell_identity_env(spec: CellSpec) -> Dict[str, str]:
    """The CAGE_CELL_* seam (campaign_session.derive_cell_spec explicit-axes
    path): every axis explicit; absent pressure coords stay ABSENT keys,
    never "0" (absence is not zero)."""
    env = {
        "CAGE_CELL_ARM": spec.arm,
        "CAGE_CELL_RETRIEVER": spec.retriever,
        "CAGE_CELL_POLICY": spec.policy,
        "CAGE_CELL_FAMILY": spec.family,
        "CAGE_CELL_TOPOLOGY": spec.topology,
    }
    if spec.budget_r is not None:
        env["CAGE_CELL_BUDGET_R"] = f"{spec.budget_r:g}"
    if spec.rate_frac is not None:
        env["CAGE_CELL_RATE_FRAC"] = f"{spec.rate_frac:g}"
    return env


def _behavior_argv(spec: CellSpec, grid: SessionGrid) -> List[str]:
    """The arm's behavior-bearing runner flags BEYOND the --baseline token.

    Mirrors the shell drivers' conventions exactly (the --baseline token
    selects only the base pipeline; without these flags ~7 of 12 baselines
    would serve byte-identically to their lever-free siblings — the verified
    T1.2 blocker):

    - corpus arms: --corpus-prefix-budget (run_prefix_envelope.sh cag_true_*;
      corpus-trunc gets the TRUNCATED budget — that difference IS B12-vs-B3).
    - retrieval arms: the pipeline + reranker EXPLICIT (never the runner's
      default), so B5 (dense, reranker OFF) vs B6 (ranked) stays the one
      pre-registered reranker ablation.
    - retr-comp: --context-source retrieved (run_compression.sh: without it
      the arm compresses GOLD context — the Phase-2 confound).
    - retr-trunc: --max-context-docs (rank truncation, "read less of what
      you found").
    - corpus-comp: --kv-cache-dtype fp8 is RECORD-ONLY provenance (the
      runner's own help text); the launch lever rides the relaunch env.
    """
    extra: List[str] = []
    if spec.arm in CORPUS_BLOCK_ARMS:
        extra += ["--corpus-prefix-budget", str(grid.corpus_prefix_budget_tokens)]
    elif spec.arm == "corpus-trunc":
        extra += ["--corpus-prefix-budget", str(grid.corpus_trunc_budget_tokens)]
    if spec.retriever != "none":
        retrieval = RETRIEVER_ARGV.get(spec.retriever)
        if retrieval is None:
            raise PlanError(
                f"retriever {spec.retriever!r} has no registered runner argv "
                f"(registered: {sorted(RETRIEVER_ARGV)}) — refusing to guess a "
                "retrieval pipeline (fail closed)"
            )
        extra += list(retrieval)
    if spec.arm == "retr-comp":
        extra += ["--context-source", "retrieved"]
    if spec.arm == "retr-trunc":
        extra += ["--max-context-docs", str(grid.retr_trunc_kept_docs)]
    kv_dtype = ARM_KV_DTYPE.get(spec.arm)
    if kv_dtype is not None:
        extra += ["--kv-cache-dtype", kv_dtype]
    return extra


def _cell_step(
    cell: PlannedCell,
    grid: SessionGrid,
    floor: FloorTable,
    runner_cmd: Sequence[str],
    seed: int,
    window_duration_s: float,
) -> Dict[str, Any]:
    spec = cell.spec
    argv = list(runner_cmd) + [
        "--baseline",
        ARM_RUNNER_BASELINE[spec.arm],
        "--baseline-label",
        f"{cell.baseline_id}_{spec.arm}",
        "--model",
        HF_ID_OF_SLUG[spec.model],
        "--backend",
        BACKEND_OF_ENGINE[spec.engine],
        "--dataset",
        cell.dataset,
        "--num-trials",
        str(grid.replications),
        "--seed",
        str(seed),
    ]
    argv += _behavior_argv(spec, grid)
    offered_rate: Optional[float] = None
    lambda_star: Optional[float] = None
    rate_basis: Optional[str] = None
    if spec.family in _PRESSURE_FAMILIES:
        assert spec.budget_r is not None and spec.rate_frac is not None
        row = floor.row(spec.budget_r)
        lambda_star = float(row["lambda_star_pred_rps"])
        offered_rate = spec.rate_frac * lambda_star
        # lambda_compute null => the λ* here is the KV bound alone — the plan
        # carries that label loudly (§6.1: λ* = min(λ_KV, λ_compute)); it is
        # NEVER hidden behind a bare number.
        rate_basis = (
            _LAMBDA_PENDING_BASIS
            if row.get("lambda_compute_rps") is None
            else _LAMBDA_CALIBRATED_BASIS
        )
        argv += [
            "--workload-mode",
            "open_loop",
            "--rate",
            f"{offered_rate:.6g}",
            "--duration-s",
            f"{window_duration_s:g}",
        ]
    return {
        "kind": "cell",
        "family": spec.family,
        "baseline": cell.baseline_id,
        "dataset": cell.dataset,
        "row_key": spec.to_row_key(),
        "cellspec": spec.to_flat_dict(),
        "windows": grid.replications,
        "serving": (
            None
            if spec.engine == "hf"
            else {
                "engine": spec.engine,
                "prefix_mode": "OFF" if _prefix_off(spec) else "ON",
                "budget_r": spec.budget_r,
                # Launch levers: part of the serving identity (relaunch
                # boundary), recorded here so the operator sees WHICH server
                # a cell requires — including on blocked cells, where this
                # names the unrealizable requirement.
                "kv_dtype": ARM_KV_DTYPE.get(spec.arm),
                "connector": ARM_CONNECTOR.get(spec.arm),
                # T3.2: topology is a relaunch-boundary dimension too (pd =
                # a different process stack than a single server).
                "topology": spec.topology,
            }
        ),
        "offered_rate_rps": offered_rate,
        "lambda_star_pred_rps": lambda_star,
        "rate_basis": rate_basis,
        "argv": argv,
        "env": _cell_identity_env(spec),
        "blocked_on": cell.blocked_on,
        # T3.2: executable pd cells are gated behind the operator's explicit
        # --allow-pd consent (blocked_on stays null — the launcher exists,
        # but the data path is unverified until the PD preflight smoke).
        "gate": PD_GATE if spec.topology == "pd" else None,
    }


def build_plan(
    session: str,
    floor: FloorTable,
    *,
    window_duration_s: float,
    seed: int = 42,
    runner_cmd: Sequence[str] = DEFAULT_RUNNER_CMD,
    launcher_cmds: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, Any]:
    """PURE plan builder: registered grid -> ordered step list + counts.

    Raises :class:`PlanError` on every dishonest input (unknown/unregistered
    session, floor table not matching the session model or missing r rows,
    non-positive window duration). Never touches the filesystem.
    """
    grid = get_session_grid(session)
    if floor.model != grid.model:
        raise PlanError(
            f"floor table {floor.path} is for model {floor.model!r} but session "
            f"{session!r} runs {grid.model!r} — demand from the wrong model's KV "
            "arithmetic would mislabel every budget"
        )
    if not (isinstance(window_duration_s, (int, float)) and math.isfinite(window_duration_s) and window_duration_s > 0):
        raise PlanError(
            f"window_duration_s={window_duration_s!r} must be finite and > 0 "
            "(§6.1: fixed pre-costed window durations — never defaulted)"
        )
    launcher_cmds = dict(launcher_cmds or DEFAULT_LAUNCHER_CMDS)

    cells = sorted(enumerate_cells(grid), key=_sort_key)

    # Pre-resolve every budget row so a floor-table gap refuses BEFORE any
    # step is emitted (all problems at plan time, none at 3 a.m. on the pod).
    for r in sorted({c.spec.budget_r for c in cells if c.spec.budget_r is not None}):
        floor.row(r)

    steps: List[Dict[str, Any]] = []
    current: Optional[_ServingConfig] = None
    relaunches = 0
    for cell in cells:
        config = _serving_config(cell)
        # Blocked cells launch NOTHING: they never run, so their (possibly
        # unrealizable) serving config must not emit a relaunch step nor
        # disturb the boundary the surrounding executable cells run under.
        if cell.blocked_on is None and config is not None and config != current:
            steps.append(_relaunch_step(config, grid, floor, launcher_cmds))
            current = config
            relaunches += 1
        steps.append(
            _cell_step(
                cell,
                grid,
                floor,
                runner_cmd,
                seed,
                window_duration_s,
            )
        )
    for i, step in enumerate(steps):
        step["index"] = i

    cell_steps = [s for s in steps if s["kind"] == "cell"]
    by_family: Dict[str, Dict[str, int]] = {}
    for s in cell_steps:
        fam = by_family.setdefault(s["family"], {"cells": 0, "windows": 0})
        fam["cells"] += 1
        fam["windows"] += s["windows"]
    blocked = [s["row_key"] for s in cell_steps if s["blocked_on"]]

    return {
        "schema": PLAN_SCHEMA,
        "session": session,
        "group": grid.group,
        "model": grid.model,
        "model_hf_id": HF_ID_OF_SLUG[grid.model],
        "seed": seed,
        "window_duration_s": float(window_duration_s),
        "replications": grid.replications,
        "floor_table": {
            "path": str(floor.path),
            "sha256": floor.sha256,
            "engine": floor.engine,
            "kv_dtype": floor.kv_dtype,
            "grid": floor.grid,
            # P2: demand is the same physical bytes on every engine — one
            # table serves both pressure engines; recorded, not assumed.
            "demand_basis": "engine-independent bytes (charter P2)",
        },
        # The registered behavior knobs — surfaced in the header so the
        # operator reviews THE values that realize arm behavior, not just
        # the cell list (they also appear inline in every affected argv).
        "behavior_knobs": {
            "reranker_model": RERANKER_MODEL,
            "corpus_prefix_budget_tokens": grid.corpus_prefix_budget_tokens,
            "corpus_trunc_budget_tokens": grid.corpus_trunc_budget_tokens,
            "retr_trunc_kept_docs": grid.retr_trunc_kept_docs,
            "lmcache_kv_transfer_config": LMCACHE_KV_TRANSFER_CONFIG,
        },
        "counts": {
            "cells": len(cell_steps),
            "windows": sum(s["windows"] for s in cell_steps),
            "relaunches": relaunches,
            "blocked": len(blocked),
            "by_family": by_family,
        },
        "blocked_row_keys": blocked,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "charter_refs": ["6.1", "6.8", "7.6", "7.6.1", "P6"],
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Plan load/validation (the 'run' side of the schema contract)
# ---------------------------------------------------------------------------

_CELL_STEP_KEYS = (
    "family",
    "baseline",
    "dataset",
    "row_key",
    "cellspec",
    "windows",
    "argv",
    "env",
    "blocked_on",
    "gate",
)
_RELAUNCH_STEP_KEYS = (
    "engine",
    "model",
    "prefix_mode",
    "budget_r",
    "kv_dtype",
    "connector",
    "topology",
    "pd",
    "argv",
    "env",
)


def load_plan(path: Path) -> Dict[str, Any]:
    """Load + fail-closed-validate a plan JSON (schema, step shape, row keys).

    Every cell's ``row_key`` is re-minted from its embedded cellspec via
    ``CellSpec.from_flat_dict(...).to_row_key()`` — a plan whose keys were
    hand-edited (or drifted from cellspec) refuses instead of writing a tree
    the organizer would reject later.
    """
    path = Path(path)
    if not path.is_file():
        raise RunError(f"plan not found: {path}")
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunError(f"plan {path} is not valid JSON: {exc}") from exc
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise RunError(
            f"plan {path} schema is not {PLAN_SCHEMA!r} — refusing to guess at "
            "an unknown plan format"
        )
    problems: List[str] = []
    steps = plan.get("steps")
    if not isinstance(steps, list):
        raise RunError(f"plan {path} has no steps[] list")
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("kind") not in ("cell", "relaunch"):
            problems.append(f"steps[{i}]: kind must be 'cell' or 'relaunch'")
            continue
        keys = _CELL_STEP_KEYS if step["kind"] == "cell" else _RELAUNCH_STEP_KEYS
        missing = [k for k in keys if k not in step]
        if missing:
            problems.append(f"steps[{i}] ({step['kind']}): missing key(s) {missing}")
            continue
        if not isinstance(step["argv"], list) or not all(
            isinstance(a, str) for a in step["argv"]
        ):
            problems.append(f"steps[{i}]: argv must be a list of strings")
        if not isinstance(step["env"], dict):
            problems.append(f"steps[{i}]: env must be an object")
        if step["kind"] == "cell":
            try:
                minted = CellSpec.from_flat_dict(step["cellspec"]).to_row_key()
            except Exception as exc:  # cellspec is the one legality gate
                problems.append(f"steps[{i}]: cellspec is charter-illegal: {exc}")
                continue
            if minted != step["row_key"]:
                problems.append(
                    f"steps[{i}]: row_key {step['row_key']!r} != cellspec-minted "
                    f"{minted!r} — keys are minted by CellSpec, never hand-built"
                )
    if problems:
        raise RunError(
            f"plan {path} failed validation ({len(problems)} problem(s)):\n"
            + "\n".join(f"  [{j + 1}] {p}" for j, p in enumerate(problems))
        )
    return plan


# ---------------------------------------------------------------------------
# 'run' — execute a plan
# ---------------------------------------------------------------------------


def count_complete_windows(campaign_root: Path, row_key: str, dataset: str) -> int:
    """Complete windows already on disk for one cell: window dir (reader
    grammar) + its metrics.json completeness sentinel (the same sentinel the
    runner's own resume and the shell gates key on)."""
    cell_dir = Path(campaign_root) / "cells" / row_key
    if not cell_dir.is_dir():
        return 0
    n = 0
    for entry in cell_dir.iterdir():
        match = WINDOW_DIR_RE.match(entry.name)
        if (
            entry.is_dir()
            and match is not None
            and match.group(1) == dataset
            and (entry / "metrics.json").is_file()
        ):
            n += 1
    return n


def _sentinel_path(campaign_root: Path, row_key: str, dataset: str) -> Path:
    # Dot-named DELIBERATELY: the §5 seal scope (campaign_layout.seal_run and
    # seal_campaign_run's journal cross-check) skips dot entries — a visible
    # sentinel under cells/ must never poison a later --seal-partial seal.
    # Dataset-suffixed DELIBERATELY: F1 row keys are shared by all four QA
    # datasets, so a bare per-cell sentinel could not say WHICH pass failed.
    return Path(campaign_root) / "cells" / row_key / f".STATUS-{dataset}"


def _write_failed_sentinel(
    campaign_root: Path, row_key: str, dataset: str, reason: str
) -> None:
    path = _sentinel_path(campaign_root, row_key, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.write_text(
        f"STATUS=failed dataset={dataset} reason={reason} utc={stamp}\n",
        encoding="utf-8",
    )


def _clear_failed_sentinel(campaign_root: Path, row_key: str, dataset: str) -> None:
    # A stale STATUS=failed over data that later completed (resume rerun,
    # --force-rerun) is a FALSE forensic record — remove it on success and on
    # a verified-complete skip.
    _sentinel_path(campaign_root, row_key, dataset).unlink(missing_ok=True)


def _exec(argv: Sequence[str], extra_env: Mapping[str, str]) -> int:
    env = dict(os.environ)
    env.update(extra_env)
    proc = subprocess.run(list(argv), env=env, cwd=str(REPO_ROOT))
    return proc.returncode


@dataclass
class _Outcome:
    row_key: str
    dataset: str
    outcome: str  # ok | skipped-complete | skipped-blocked | failed | skipped-launch-failed


def run_plan(
    plan: Mapping[str, Any],
    campaign_root: Path,
    *,
    force_rerun: bool = False,
    skip_blocked: bool = False,
    allow_pd: bool = False,
    seal: bool = False,
    seal_partial: bool = False,
    seal_cmd: Sequence[str] = DEFAULT_SEAL_CMD,
) -> int:
    """Execute a validated plan sequentially; returns the process exit code.

    Refusals BEFORE anything executes: blocked cells present without the
    EXPLICIT ``skip_blocked`` consent (a silent partial run is forbidden;
    with consent every blocked cell is still reported per-cell and gates a
    plain ``--seal``), executable PD cells present without the EXPLICIT
    ``allow_pd`` consent (T3.2: the PD data path is unverified until the
    Run-C-prime preflight PD smoke), zero executable cell steps, campaign
    root not matching the plan's session or the §1 run_id grammar. During
    execution a cell failure CONTINUES the run (sentinel + summary + nonzero
    exit); a relaunch failure fails every cell up to the next relaunch
    boundary — running them against a stale serving config would mislabel
    the data, which is worse than not running.
    """
    campaign_root = Path(campaign_root)
    steps: List[Dict[str, Any]] = list(plan["steps"])
    cell_steps = [s for s in steps if s["kind"] == "cell"]

    blocked = [s for s in cell_steps if s.get("blocked_on")]
    if blocked and not skip_blocked:
        listing = "\n".join(
            f"  {s['row_key']}  [blocked_on: {s['blocked_on']}]" for s in blocked
        )
        raise RunError(
            f"plan contains {len(blocked)} blocked cell(s) — refusing to run "
            "ANY of it (no partial SILENT skips; pass --skip-blocked to run "
            f"the executable subset loudly):\n{listing}"
        )
    # T3.2 PD consent gate: executable pd cells (blocked_on null, gate label
    # set) run ONLY under the operator's explicit --allow-pd — the pd
    # launcher exists, but every engine-facing name on its path plus the
    # NIXL transfer itself is [VERIFY-LIVE at Run-C-prime preflight], so an
    # un-consented pd execution would spend GPU time on an unverified stack.
    pd_cells = [
        s
        for s in cell_steps
        if not s.get("blocked_on")
        and isinstance(s.get("cellspec"), dict)
        and s["cellspec"].get("topology") == "pd"
    ]
    if pd_cells and not allow_pd:
        listing = "\n".join(
            f"  {s['row_key']}  [gate: {s.get('gate')}]" for s in pd_cells
        )
        raise RunError(
            f"plan contains {len(pd_cells)} PD (prefill/decode "
            "disaggregation) cell(s) — refusing to run ANY of it without the "
            "EXPLICIT --allow-pd consent. The PD data path is unverified "
            "until the Run-C-prime preflight PD smoke passes; pass "
            f"--allow-pd only after that gate:\n{listing}"
        )
    if not [s for s in cell_steps if not s.get("blocked_on")]:
        raise RunError(
            "plan contains zero executable cells — a dry plan is reviewable, "
            "but 'run' on it is an operator error"
        )
    session = plan.get("session")
    if campaign_root.parent.name != session:
        raise RunError(
            f"campaign root {campaign_root} sits under session dir "
            f"{campaign_root.parent.name!r} but the plan is for session "
            f"{session!r} — a cross-session tree would be refused downstream"
        )
    if not RUN_ID_RE.match(campaign_root.name):
        raise RunError(
            f"campaign-root basename {campaign_root.name!r} violates the §1 "
            f"run_id grammar {RUN_ID_RE.pattern}"
        )

    outcomes: List[_Outcome] = []
    server_ok = True  # state of the current serving config (non-hf cells)
    for step in steps:
        if step["kind"] == "relaunch":
            rc = _exec(step["argv"], step["env"])
            server_ok = rc == 0
            if not server_ok:
                print(
                    f"[run_campaign] RELAUNCH FAILED (exit {rc}): engine="
                    f"{step['engine']} prefix={step['prefix_mode']} "
                    f"budget_r={step.get('budget_r')} — failing its cells "
                    "until the next relaunch boundary"
                )
            continue
        row_key, dataset = step["row_key"], step["dataset"]
        if step.get("blocked_on"):
            # Reaches here only under the operator's explicit --skip-blocked:
            # reported per-cell, executes nothing, gates a plain --seal below.
            outcomes.append(_Outcome(row_key, dataset, "skipped-blocked"))
            continue
        if step.get("serving") is not None and not server_ok:
            _write_failed_sentinel(campaign_root, row_key, dataset, "relaunch-failed")
            outcomes.append(_Outcome(row_key, dataset, "skipped-launch-failed"))
            continue
        done = count_complete_windows(campaign_root, row_key, dataset)
        if not force_rerun and done >= int(step["windows"]):
            _clear_failed_sentinel(campaign_root, row_key, dataset)
            outcomes.append(_Outcome(row_key, dataset, "skipped-complete"))
            continue
        argv = list(step["argv"]) + ["--campaign-root", str(campaign_root)]
        rc = _exec(argv, step["env"])
        if rc == 0:
            _clear_failed_sentinel(campaign_root, row_key, dataset)
            outcomes.append(_Outcome(row_key, dataset, "ok"))
        else:
            _write_failed_sentinel(campaign_root, row_key, dataset, f"runner-exit-{rc}")
            outcomes.append(_Outcome(row_key, dataset, "failed"))

    # ---- summary matrix (per-cell outcomes; the operator's at-a-glance) ----
    counts: Dict[str, int] = {}
    print("\n[run_campaign] ===== summary matrix =====")
    for o in outcomes:
        counts[o.outcome] = counts.get(o.outcome, 0) + 1
        print(f"  [{o.outcome:>21}] {o.row_key} ({o.dataset})")
    print(f"[run_campaign] totals: {counts}")

    any_failed = any(
        o.outcome in ("failed", "skipped-launch-failed") for o in outcomes
    )
    any_blocked_skipped = any(o.outcome == "skipped-blocked" for o in outcomes)
    if seal:
        # Blocked skips gate the seal exactly like failures: the tree is NOT
        # the full registered session, and a plain seal marks it done.
        if (any_failed or any_blocked_skipped) and not seal_partial:
            print(
                "[run_campaign] --seal SKIPPED: run has failures and/or "
                "blocked-skipped cells and --seal-partial was not given "
                "(a seal marks the tree done)"
            )
        else:
            rc = _exec(list(seal_cmd) + [str(campaign_root)], {})
            if rc != 0:
                print(f"[run_campaign] seal FAILED (exit {rc})")
                return 1
            print("[run_campaign] sealed.")
    return 1 if any_failed else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_plan(args: argparse.Namespace) -> int:
    floor = load_floor_table(Path(args.floor_table))
    launcher_cmds: Optional[Dict[str, Tuple[str, ...]]] = None
    if args.launcher_cmd:
        override = tuple(shlex.split(args.launcher_cmd))
        launcher_cmds = {engine: override for engine in DEFAULT_LAUNCHER_CMDS}
    plan = build_plan(
        args.session,
        floor,
        window_duration_s=args.window_duration_s,
        seed=args.seed,
        runner_cmd=tuple(shlex.split(args.runner_cmd)),
        launcher_cmds=launcher_cmds,
    )
    text = json.dumps(plan, indent=2, sort_keys=False) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        c = plan["counts"]
        print(
            f"[run_campaign] plan written: {args.out} — {c['cells']} cells, "
            f"{c['windows']} windows, {c['relaunches']} relaunches, "
            f"{c['blocked']} blocked"
        )
    else:
        print(text, end="")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    plan = load_plan(Path(args.plan))
    seal_cmd = (
        tuple(shlex.split(args.seal_cmd)) if args.seal_cmd else DEFAULT_SEAL_CMD
    )
    return run_plan(
        plan,
        Path(args.campaign_root),
        force_rerun=args.force_rerun,
        skip_blocked=args.skip_blocked,
        allow_pd=args.allow_pd,
        seal=args.seal,
        seal_partial=args.seal_partial,
        seal_cmd=seal_cmd,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_campaign",
        description=(
            "THE campaign sweep driver: 'plan' enumerates a session's "
            "registered D6 grid into a reviewable execution plan (pure, "
            "offline); 'run' executes a plan against a campaign root."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="enumerate the registered grid (pure)")
    p_plan.add_argument("--session", required=True, help=f"one of {sorted(SESSIONS)}")
    p_plan.add_argument(
        "--floor-table",
        required=True,
        help="REQUIRED T2.4 floor-table-v1 JSON (build_floor_table.py) — the "
        "one demand/λ* source; a missing/invalid file refuses",
    )
    p_plan.add_argument(
        "--window-duration-s",
        required=True,
        type=float,
        help="fixed pre-costed measurement-window duration for pressure cells "
        "(§6.1; REQUIRED — never defaulted)",
    )
    p_plan.add_argument("--seed", type=int, default=42)
    p_plan.add_argument(
        "--runner-cmd",
        default=" ".join(DEFAULT_RUNNER_CMD),
        help="argv prefix for cell steps (test seam; default = the real runner)",
    )
    p_plan.add_argument(
        "--launcher-cmd",
        default=None,
        help="argv prefix override for ALL relaunch steps (test seam; default "
        "= the real per-engine 2_serving launchers)",
    )
    p_plan.add_argument("--out", default=None, help="write the plan JSON here (else stdout)")
    p_plan.set_defaults(func=_cmd_plan)

    p_run = sub.add_parser("run", help="execute a plan")
    p_run.add_argument("--plan", required=True, help="plan JSON from 'plan'")
    p_run.add_argument(
        "--campaign-root",
        required=True,
        help="RESULTS_LAYOUT run root results/<campaign>/<session>/<run_id>",
    )
    p_run.add_argument(
        "--force-rerun",
        action="store_true",
        help="run cells even when all their windows are already complete",
    )
    p_run.add_argument(
        "--skip-blocked",
        action="store_true",
        help="EXPLICIT consent to run the executable subset of a plan whose "
        "blocked cells (blocked_row_keys) cannot run yet; every skipped cell "
        "is reported and a plain --seal is gated (use --seal-partial)",
    )
    p_run.add_argument(
        "--allow-pd",
        action="store_true",
        help="EXPLICIT consent to execute PD (prefill/decode disaggregation) "
        "cells — pass ONLY after the Run-C-prime preflight PD smoke has "
        "passed on the provisioned node (gate: --allow-pd + PD preflight "
        "smoke); without it a plan containing pd cells refuses to run",
    )
    p_run.add_argument(
        "--seal",
        action="store_true",
        help="invoke seal_campaign_run.py at the end (fully-passed runs only "
        "unless --seal-partial)",
    )
    p_run.add_argument("--seal-partial", action="store_true")
    p_run.add_argument(
        "--seal-cmd",
        default=None,
        help="argv prefix override for the sealer (test seam)",
    )
    p_run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (PlanError, RunError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
