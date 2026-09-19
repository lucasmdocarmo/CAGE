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

Grid registration (§7.6.1 — sessions 'a' (Group A) and 'b' (Group B,
Run C-prime scope) are registered; sessions cd-act1/cd-act2 refuse loudly
until their registrations land):

- F1  locality (prefix ON, sub-pressure): B1-B12 × {vllm, sglang} × the four
  QA datasets — no budget/rate grid (one config per cell).
- F1 HF oracle: EXACTLY the reduced 10-cell set {B3 × all 4 datasets;
  B1, B2, B6 × squad_v2 + qasper} on the in-process hf engine.
- F2  pressure (prefix OFF): FRESH set × {vllm, sglang} × the session's
  registered factorial — session a: the FULL §6.1 5×6 factorial PLUS the
  §6.4 anchor fine r-grid (ANCHOR_FINE_BUDGET_LEVELS at the two chassis
  rates 0.85/1.05·λ*, deduplicated against the factorial — a coordinate on
  both grids is ONE cell carrying both memberships in its ``grids`` marker);
  session b: the §6.8 reduced 3×3 grid.
- F2 RULER pairing (session a; D5 item 5): the length INSTRUMENT rides the
  SAME F2 grid points as gold-fresh — B1 × both pressure engines × every
  registered F2 coordinate × the 4-task charter subset (RULER_F2_TASKS,
  literals pinned to src/data/ruler._TASKS), dataset ``ruler`` at SHAPE-32K
  (32,512-in / 256-out, §5.1 item 1). Every RULER cell is PAIRED with the
  matched real-text Qasper cell at the identical coordinate BY CONSTRUCTION
  (the registration refuses a ruler baseline absent from f2_baselines).
  Per-task steps share the cell row key (task is not a CellSpec axis); each
  task claims its own window-ordinal RANGE via ``window_ordinal_base``
  (task_index × replications), threaded to the runner as
  CAGE_WINDOW_ORDINAL_BASE so per-task resume can never collide.
- F3  interaction (prefix ON × pressure): REUSE set × {vllm, sglang} × the
  §6.8 reduced 3×3 grid r ∈ {1.0, 0.5, 0.25} × {0.85, 0.95, 1.05}·λ*.
- Replications: 3 measurement windows per grid point (D6 §6.3).
- corpus-fresh prefix OFF (ADR-0103, owner decision 2026-09-16): B4 is
  served with the engine prefix cache OFF through a PER-ARM RELAUNCH,
  uniformly on every engine, in EVERY family (F1 and F3; PREFIX_OFF_ARMS,
  consulted only via ``_prefix_off``). Its family carriage is unchanged
  (REUSE bit, rides F3 beside B3). The runner's ``no_cache`` token only
  labels telemetry, so before ADR-0103 B4 and B3 were served by one
  prefix-ON server: the mislabeled-duplicate failure class. Cost: one extra
  budget-free prefix-OFF relaunch per engine for F1; the F3 B4 cells share
  the F2 plain prefix-OFF boundaries at the same r (F3 budgets are a subset
  of F2 budgets and family is not a serving dimension).
- rerank pool (ADR-0104, owner decision 2026-09-16): B6 and every arm
  inheriting the ranked pipeline retrieve a dense candidate POOL of
  RERANK_POOL (10), rerank it whole with the pinned cross-encoder and serve
  the top 3; B5 serves the dense top 3 unranked. The pool is pinned in the
  cell argv (--rerank-pool, RETRIEVER_ARGV['rerank']); the runner refuses a
  pool without a reranker, so B5 can never carry one silently.
- retrieval pins (backlog A5): every retrieval cell pins --top-k
  (RETRIEVAL_TOP_K), --embedding-model and --embedding-revision (both read
  from the ADR-0099 freeze slot INSTRUMENT_REVISIONS.dense_retriever of the
  registration artifact, never a second hard-coded copy; the runner loads
  the encoder AT that revision and rebuilds an index built at another) and
  --ir-index-dir (SessionGrid.ir_index_root),
  and its env pins CAGE_DISTRACTOR_DOCS (DISTRACTOR_DOCS; behavior, not
  identity: derive_cell_spec ignores it). The runner's argparse defaults
  never reach a campaign cell, and load_plan refuses a plan whose retrieval
  cells lack the pins or whose header lacks the freeze pin.
- per-cell Redis namespaces (backlog F5a): every B7 (retr-reuse) cell
  carries --redis-key-prefix cage:<sha1(row_key)[:12]>
  (redis_key_prefix_for_row) and --flush-redis-namespace, so no two cells
  ever share retrieval-artifact cache entries and each cell starts empty.
  The runner-side key additionally folds the index's corpus hash (F5b).
- max_model_len (backlog A10, Tier A): ONE request-length cap per session
  (SessionGrid.max_model_len, default DEFAULT_MAX_MODEL_LEN = 32768: RULER
  SHAPE-32K is 32,512 in + 256 out and long Qasper papers exceed the pilot
  4096), carried on EVERY relaunch of BOTH engines as the env
  MAX_MODEL_LEN_ENV (VLLM_MAX_MODEL_LEN: vLLM --max-model-len, SGLang
  --context-length, the pd launcher applies it to both roles), recorded on
  the relaunch step and in the header ``serving_shapes``. The KV pool is
  byte-budgeted separately (gate (j)); this caps request length only. The
  shell default (_serving_config.sh, 4096) stays for the pilot scripts and
  never reaches a campaign relaunch: load_plan refuses a relaunch without
  the record or the env, and a grid registering RULER tasks refuses a value
  below SHAPE-32K.
- gpu_count (W4.2, feeds §6.6b / contrast #18): every cell step carries the
  integer GPU count its serving stack launches with — topology 'single' →
  the session's registered ``serving_tp`` (1 on the anchor; TP-sharded
  single-instance serving counts its ranks, §7.6 Group B e5), 'tp' → the
  registered ``dist_tp_size``, 'pd' → the two registered role GPU counts
  summed. Threaded to the runner as CAGE_GPU_COUNT and persisted into
  cell.json by the campaign writer; underivable counts appear ONLY on
  blocked cells (an executable cell without one refuses at plan time).
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
registered truncation knob (B12: ONE cell per rung of the ADR-0106 descending
corpus ladder, ``--corpus-prefix-budget <rung> --corpus-rung <rung>`` plus the
CAGE_CELL_CORPUS_BUDGET identity coordinate; the runner refuses a manifest
lacking the rung and serves out-of-corpus queries labeled, never dropped),
corpus-comp launches the server with the fp8 KV
dtype env, and retr-store launches vLLM with the LMCache connector env
(run_kv_store.sh). Identity STILL rides only the CAGE_CELL_* seam — behavior
flags never mint identity.

Cold start per window (ADR-0102, owner decision 2026-09-16): every
server-engine cell (SERVER_ENGINES; the in-process hf oracle is exempt)
carries ``--reset-cache-between-trials`` (RESET_CACHE_PER_WINDOW) and
``--warmup-pool-queries 20`` (WARMUP_POOL_QUERIES): the runner flushes the
engine cache before EVERY window, refuses the window if the flush fails, and
warms up on W_warm = 20 requests drawn deterministically from a pool DISJOINT
from every trial's measured set (results discarded; summary + ids sha256 in
the window metadata). Both constants surface in the plan header, and
``load_plan`` refuses a stale plan whose server-engine cell lacks them.

Engine endpoints (Batch 2 finding W2, 2026-09-18, option A): every
server-engine cell carries ``--api-base http://localhost:<port>`` and every
relaunch exports the launcher port env (VLLM_PORT / SGLANG_PORT; the pd
relaunch CAGE_PD_PROXY_PORT), BOTH derived from the one port table
ENGINE_PORTS that mirrors the frozen launchers' defaults, so the server a
relaunch starts and the endpoint its cells dial agree by construction. The
runner's --api-base default is the vLLM port and never reaches a campaign
cell; ``load_plan`` refuses a stale plan per cell and per relaunch, and
``run`` refuses while a CAGE_<ENGINE>_API_BASE override is exported (the
runner resolves it before the pin). The in-process hf oracle dials nothing.

Per-row N (backlog A9, DECISION.md amendment A1 / A5 of
MyDocs/registration/power_decision_2026-08-07): every cell carries
``--num-queries <n>`` for its ROW CLASS: primary predicate cells (the #4
contrast cells, baselines PRIMARY_BASELINES on the pinned PRIMARY_ENGINE in
F1) n = 2,000 paired queries per dataset; every other per-query F1 cell
(secondary) n = 800; HF-oracle / T=0 identity and the TTFT-only DIST
topology cells n = 300; loaded/window cells (F2, F3, RULER) W = 200 requests
per window (the open-loop generator draws from the measured set, so W IS the
registered pool size). The five constants live on the SessionGrid (header
``per_row_n``); ``achievable_n`` LOWERS a dataset's class n with a header
caveat (A5: Qasper registers its own achievable n). With a registered query
manifest the runner measures the FIRST n ids of each trial (nested prefix
subsets), so the plan REFUSES a manifest whose trials carry fewer than n ids
for that dataset's cells. The 2,000 to 1,600 to 1,200 step-down is an
ANALYSIS-time realized-n policy, never a plan knob.

Ordering minimizes engine relaunches: cells sort on (engine, prefix_mode,
model, budget_r, kv_dtype, connector, rate); every serving-config change is
an explicit ``relaunch`` step in the plan carrying the launcher argv + launch
env (budget, KV dtype, connector), so the relaunch count is exactly the
number of distinct EXECUTABLE serving configs (blocked cells launch nothing).

Failure doctrine: a failed cell writes a ``.STATUS-<dataset>`` sentinel
(dot-named so the §5 seal scope — which refuses any non-journaled file under
cells/ — skips it; dataset-suffixed because F1 row keys are shared by four
datasets and a forensic record must name WHICH one failed; per-task RULER
steps additionally suffix ``-from-<NN>`` — their claimed ordinal range —
because the tasks share one dataset and a task-2 success must never clear a
task-1 failure record) and execution CONTINUES; a later successful (or already-complete) pass of the same
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
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

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
    MODEL_KV,
    CacheBudgetError,
    plan_budget,
)
from src.orchestration.load_generator import (  # noqa: E402
    D6_RATE_FRACTIONS,
    D6_REDUCED_RATE_FRACTIONS,
)
from src.data import ruler as _ruler  # noqa: E402  (task-literal pin only)
from src.data.manifest import ManifestError, trunc_rung_for  # noqa: E402
from src.orchestration.ir import STALE_INDEX_OPT_IN_ENV  # noqa: E402

__all__ = [
    "PLAN_SCHEMA",
    "PlanError",
    "RunError",
    "SessionGrid",
    "SESSION_GRIDS",
    "PlannedCell",
    "build_plan",
    "cell_num_queries",
    "class_n",
    "classify_row",
    "engine_api_base",
    "enumerate_cells",
    "load_floor_table",
    "load_plan",
    "main",
    "row_class",
]

# v2 (2026-09-01): pd steps added required keys gate/topology/pd — a v1 plan
# predates them, so 'run' must refuse it via the schema check and the operator
# re-plans (never silently execute a plan missing pd semantics).
# v3 (2026-09-02, Wave-4 W4.2/W4.3/W4.6 delta, per the v2 precedent): cell
# steps gained REQUIRED keys ``gpu_count`` (§6.6b producer), ``grids`` (the
# §6.4 fine-grid membership marker), ``ruler_task`` and
# ``window_ordinal_base`` (the D5#5 per-task RULER pairing); relaunch steps
# gained REQUIRED key ``tp`` (the tensor-parallel degree the launcher is
# given — session-b/Group-B serving shapes). A v2 plan predates ALL of these,
# so 'run' must refuse it and the operator re-plans.
# v4 (2026-09-16, adversarial-review repair, per the v2/v3 precedent): cell
# argv gained the ADR-0102 cold-start flags, the ADR-0104 --rerank-pool on
# ranked cells, the ADR-0106 --corpus-rung on B12 cells and the per-dataset
# --query-manifest registration (plan header ``query_manifests``); the
# ADR-0103 prefix-OFF relaunch became part of the serving identity. A v3
# plan predates ALL of these: its B4 cells would run under prefix-ON
# relaunches and its ranked cells the legacy rerank-exactly-top-k pipeline
# under the SAME row keys (mislabeled duplicates), so 'run' refuses it via
# the schema check AND load_plan's per-clause _stale_plan_problems.
# v5 (2026-09-17, backlog A9 --num-queries half, per the v2/v3/v4 precedent):
# cell steps gained REQUIRED keys ``row_class`` and ``num_queries`` (the
# DECISION.md A1 per-row N) and every cell argv gained --num-queries; the
# header gained ``per_row_n``. A v4 plan predates them: its cells would run
# the runner's default query count under row keys that now register a per
# class n (mislabeled n), so 'run' refuses it and the operator re-plans.
# A5/F5a (2026-09-17, backlog A5 retrieval pins + F5a per-cell Redis
# namespaces): retrieval cell argv gained --top-k / --embedding-model /
# --embedding-revision / --ir-index-dir (+ --redis-key-prefix /
# --flush-redis-namespace on B7), the cell env CAGE_DISTRACTOR_DOCS, and the
# header's behavior_knobs the freeze pin (embedding_model,
# embedding_model_revision, ir_index_root). No schema bump: a v5 plan built
# before A5 (or before the revision became enforced, review 2026-09-17) is
# refused by load_plan's header check and the per-clause
# _stale_plan_problems (its retrieval cells lack the pins), so it can never
# run runner defaults under registered row keys.
# A10 (2026-09-17, backlog Tier A uniform max_model_len): relaunch steps
# gained the REQUIRED key ``max_model_len`` and the env VLLM_MAX_MODEL_LEN;
# the header ``serving_shapes`` gained ``max_model_len``. No schema bump: a
# v5 plan built before A10 is refused by the relaunch key check and by
# load_plan's header + per-relaunch checks (its launchers would fall back to
# the pilot shell default 4096 and refuse every RULER request), so it can
# never run a non-uniform or under-sized regime under registered row keys.
# W1 (2026-09-18, Batch 2 finding W1 / ADR-0055 decoupled scoring): cell
# steps gained the env CAGE_SKIP_QUALITY=1 and the header behavior_knobs the
# quality_scoring record. No schema bump: a v5 plan built before W1 is
# refused by _stale_plan_problems (its cells would run inline model scoring
# inside the measured window), so the operator re-plans.
# W2 (2026-09-18, Batch 2 finding W2, owner picked option A): every
# server-engine cell argv gained --api-base and every relaunch env the
# launcher port env (VLLM_PORT / SGLANG_PORT / CAGE_PD_PROXY_PORT), both
# derived from the ONE port table ENGINE_PORTS; the relaunch record gained
# ``api_base`` and the header serving_shapes the table. No schema bump: a v5
# plan built before W2 is refused per cell and per relaunch (its SGLang cells
# would ride the runner's --api-base default, the vLLM port), so the operator
# re-plans.
PLAN_SCHEMA = "cage-campaign-plan-v5"
FLOOR_TABLE_SCHEMA = "floor-table-v1"

#: §6.1 pre-registered budget ratios r = B/D (Group-A anchor factorial; r=1.5
#: is the comfortable control rung, never in-regime by design).
FULL_BUDGET_LEVELS: Tuple[float, ...] = (1.5, 1.0, 0.75, 0.5, 0.25)
#: §6.8 pruning rule: the reduced grid for every non-anchor pressure family.
REDUCED_BUDGET_LEVELS: Tuple[float, ...] = (1.0, 0.5, 0.25)

#: §6.4 Graft C — the anchor fine 7-level r-grid, Qwen3-14B/Group A ONLY
#: (§6.8: "plus the §6.4 fine 7-level r-grid run on Group A (anchor) ONLY";
#: §7.6.1 F2 row A: "FRESH set × full 5×6 + §6.4 r-grid"). Scope derivation:
#: §6.4 produces the quality-vs-stored-bytes curves per arm family AND the
#: iso-KV-bytes CROSS-ENGINE comparison, so the fine grid rides the F2 FRESH
#: set on BOTH registered pressure engines. 5 of the 7 levels coincide with
#: FULL_BUDGET_LEVELS; the fine grid's NEW coordinates are exactly
#: {1.25, 0.375} × the two rates below (deduplicated at enumeration — the
#: shared coordinates are ONE cell each, carrying both grid memberships).
ANCHOR_FINE_BUDGET_LEVELS: Tuple[float, ...] = (1.5, 1.25, 1.0, 0.75, 0.5, 0.375, 0.25)
#: §6.4: the fine grid runs "at two chassis-validated rates (0.85·λ* and
#: 1.05·λ*)" — both are members of the registered dispatcher factorial, so
#: the fine grid can never offer a rate the D6 generator does not register.
ANCHOR_FINE_RATE_FRACTIONS: Tuple[float, ...] = (0.85, 1.05)
assert set(ANCHOR_FINE_RATE_FRACTIONS) <= set(D6_RATE_FRACTIONS), (
    "§6.4 fine rates drifted outside load_generator.D6_RATE_FRACTIONS"
)

#: Grid-membership labels stamped on every F2 cell step (``grids``): which
#: registered grid(s) the coordinate belongs to — reviewable in the plan and
#: the analysis filter for the §6.4 curves (a coordinate on both grids is ONE
#: cell serving both memberships; enumerating it twice would double-run it).
GRID_D6_FACTORIAL = "d6-factorial"
GRID_ANCHOR_FINE = "anchor-fine-6.4"

#: Rate fractions come from the DISPATCHER's registered constants
#: (src/orchestration/load_generator.py) — the plan and the load generator can
#: never disagree about the grid (same rule build_floor_table.py follows).
FULL_RATE_FRACTIONS: Tuple[float, ...] = tuple(D6_RATE_FRACTIONS)
REDUCED_RATE_FRACTIONS: Tuple[float, ...] = tuple(D6_REDUCED_RATE_FRACTIONS)

#: D5 item 5 RULER pairing — the 4-task charter subset (NIAH-MK/MQ, VT, QA;
#: "subset tasks ...; per-task reporting (never the mean)"). ``niah_single``
#: stays a loader/CLI default for pilots but is NOT a charter subset member,
#: so the grid does not register it. Literals are pinned STRUCTURALLY against
#: the loader's own registered task tuple below — a drifted spelling refuses
#: at import, never at 3 a.m. on the pod.
RULER_F2_TASKS: Tuple[str, ...] = (
    "niah_multikey",
    "niah_multiquery",
    "variable_tracking",
    "qa",
)
assert set(RULER_F2_TASKS) <= set(_ruler._TASKS), (
    f"RULER_F2_TASKS drifted from src/data/ruler._TASKS: "
    f"{sorted(set(RULER_F2_TASKS) - set(_ruler._TASKS))}"
)
#: SHAPE-32K (charter §5.1 item 1, PINNED 2026-08-02): input 32,512 + output
#: 256 = 32,768 total. Restated from the loader's own pin so a drift refuses.
RULER_CONTEXT_TOKENS: int = 32_512
RULER_OUTPUT_TOKENS: int = 256
assert RULER_CONTEXT_TOKENS == _ruler.MAX_CONTEXT_TOKENS, (
    "SHAPE-32K input drifted from src/data/ruler.MAX_CONTEXT_TOKENS"
)
assert RULER_OUTPUT_TOKENS == _ruler.OUTPUT_TOKENS_HINT, (
    "SHAPE-32K output drifted from src/data/ruler.OUTPUT_TOKENS_HINT"
)

#: Backlog A10 (Tier A): the ONE launcher env that carries the per-session
#: request-length cap. Every launcher of the frozen fleet reads it from the
#: environment: scripts/lib/_serving_config.sh exports it (pilot default
#: 4096, untouched), manage_vllm_server.sh and manage_vllm_pd.sh pass it as
#: --max-model-len (the pd launcher to BOTH role instances),
#: manage_sglang_server.sh maps it to --context-length, and
#: manage_vllm_cluster.py reads it per instance. The relaunch env sets it
#: EXPLICITLY on every relaunch so the shell default never reaches a
#: campaign server.
MAX_MODEL_LEN_ENV: str = "VLLM_MAX_MODEL_LEN"
#: Backlog A10: the registered default of SessionGrid.max_model_len. RULER
#: SHAPE-32K requests are RULER_CONTEXT_TOKENS + RULER_OUTPUT_TOKENS =
#: 32,768 tokens and long Qasper papers exceed the pilot 4096; the
#: uniform-regime rule (_serving_config.sh) demands ONE value per session so
#: no cell is served under a different length regime. The KV pool is
#: byte-budgeted separately (CAGE_KV_BUDGET_BYTES / max-total-tokens, gate
#: (j)), so this caps request length only. Pinned structurally to SHAPE-32K
#: below: a drift of either side refuses at import.
DEFAULT_MAX_MODEL_LEN: int = 32_768
assert DEFAULT_MAX_MODEL_LEN >= RULER_CONTEXT_TOKENS + RULER_OUTPUT_TOKENS, (
    "DEFAULT_MAX_MODEL_LEN is below RULER SHAPE-32K (backlog A10)"
)

#: §7.6.1 F1 row: the four quality-instrumented QA datasets (D5 items 1-4).
QA_DATASETS: Tuple[str, ...] = ("squad_v2", "hotpotqa", "musique", "qasper")

#: D6 §6.3: >= 3 replications per grid point; one replication = one §1
#: measurement window (= one runner trial).
REPLICATIONS = 3

#: ADR-0102 (owner decision 2026-09-16): COLD START PER WINDOW. A campaign
#: trial IS a window, so every server-engine cell passes
#: ``--reset-cache-between-trials`` and the runner flushes the engine cache
#: before EVERY window (trial 1 included: the server may still hold the
#: previous cell's cache) and REFUSES the window when the flush fails (a
#: window that starts warm when the plan says cold is a mislabeled row).
#: Registered here, never a silent default inside a step builder.
RESET_CACHE_PER_WINDOW: bool = True
#: ADR-0102: the registered warm-up W_warm = 20 requests, served after every
#: per-window cache reset and drawn deterministically (seed + trial ordinal)
#: from a pool DISJOINT from every trial's measured set
#: (``run_experiment.py --warmup-pool-queries``). Results are discarded; only
#: a summary line + the ids' sha256 land in the window metadata. The legacy
#: ``--warmup-queries`` flag replays the MEASURED set (cache-warms it) and
#: must never ride a campaign cell.
WARMUP_POOL_QUERIES: int = 20
#: ADR-0102: the engines with a server-side cache to flush; the in-process
#: hf oracle gets neither cold-start flag. lmdeploy is listed for the day its
#: cells register (BACKEND_OF_ENGINE has no lmdeploy token today).
SERVER_ENGINES: Tuple[str, ...] = ("vllm", "sglang", "lmdeploy")

#: ADR-0055 ("serving writes, scoring reads", accepted 2026-08-04; Batch 2
#: finding W1, 2026-09-18): the runner's decoupled-scoring switch. Under
#: CAGE_SKIP_QUALITY=1 the serving loop uses the MODEL-FREE evaluator (F1,
#: EM and the abstention detector inline; LettuceDetect, NLI, BERTScore,
#: ROUGE and the similarity embedder run post-serving through
#: rescore_quality.py --full --scoring-run-id <id>, the campaign v2 mode).
#: The runner's own default is 0 (inline model scoring,
#: the pilot path), so EVERY cell step pins the env: a campaign window
#: produced by inline scoring spends pod time on CPU scoring, dilutes the
#: window's occupancy average with engine-idle time and contradicts the
#: accepted ADR. BEHAVIOR, not identity: derive_cell_spec ignores it.
SKIP_QUALITY_ENV: str = "CAGE_SKIP_QUALITY"
SKIP_QUALITY_VALUE: str = "1"
DECOUPLED_SCORING_ADR: str = "ADR-0055"

#: Backlog A9 / DECISION.md amendment A1 (MyDocs/registration/
#: power_decision_2026-08-07/DECISION.md): the registered per-row N. These
#: seed the SessionGrid fields of the same name (the grid is what the plan
#: header records; a session may only LOWER a dataset's n via achievable_n,
#: A5). Never a silent default inside a step builder.
#: - N_PRIMARY: MDE-0.05 primary predicate cells (the #4 B6/B3 contrast on
#:   the pinned primary engine, family F1): 2,000 paired queries per dataset.
#: - N_SECONDARY: secondary-only per-query F1 cells (MDE 0.10 at alpha/12): 800.
#: - N_IDENTITY: TTFT-only, HF-oracle and T=0 identity cells: 300.
#: - WINDOW_REQUESTS: loaded/window cells (F2, F3, RULER): W = 200 requests
#:   per window; the open-loop generator draws from the measured set
#:   (run_experiment execute_open_loop_measured: schedule index maps modulo
#:   the prepared measured set), so --num-queries IS the window pool size.
N_PRIMARY: int = 2000
N_SECONDARY: int = 800
N_IDENTITY: int = 300
WINDOW_REQUESTS: int = 200
#: A1: the primary predicate cells are pinned to ONE engine per group (the
#: charter primary engine, vLLM) and to the #4 contrast pair B3 (corpus-reuse)
#: vs B6 (retr-reuse ranked).
PRIMARY_ENGINE: str = "vllm"
PRIMARY_BASELINES: Tuple[str, ...] = ("B3", "B6")
#: The row-class vocabulary (``row_class`` on every cell step).
ROW_CLASSES: Tuple[str, ...] = ("primary", "secondary", "identity", "window")
#: The decision record every per_row_n header cites.
PER_ROW_N_DECISION: str = (
    "MyDocs/registration/power_decision_2026-08-07/DECISION.md amendment A1 "
    "(per-row N table) and A5 (achievable-n branch); the 2000/1600/1200 "
    "step-down is an analysis-time realized-n policy, not a plan knob"
)

#: ADR-0103 (owner decision 2026-09-16): the arms served with the engine
#: prefix cache OFF in EVERY family, through a per-arm RELAUNCH, uniformly on
#: every engine. corpus-fresh (B4) recomputes the corpus block on every
#: request: its runner token ``no_cache`` only LABELS telemetry
#: (src/orchestration/baselines.py), so absent this rule B4 and B3 would be
#: served by the same prefix-ON server and differ by label alone (the
#: mislabeled-duplicate failure class). Family carriage is unchanged: B4's
#: REUSE bit still rides F3 beside B3 (cellspec._ARMS_BY_FAMILY). The rule
#: is consulted ONLY through _prefix_off (sort key, serving-config identity,
#: relaunch argv, cell serving record), never re-derived by a step builder.
PREFIX_OFF_ARMS: FrozenSet[str] = frozenset({"corpus-fresh"})

#: The task text the operator sees on an enumerated-but-unrunnable DIST
#: tp-overlay cell: the T3.1 TP env exists, but THIS session registered no
#: ``dist_tp_size``, so the TP degree the leg would launch with is
#: underivable — enumerated blocked, never guessed at.
TP_DIST_BLOCKED_ON = (
    "tp-topology overlay: no registered dist_tp_size on this SessionGrid — "
    "register the TP degree before planning the tp leg"
)

#: engine -> the launcher env carrying the T3.1 tensor-parallel degree
#: (scripts/lib/_serving_config.sh, one source of truth; value 1 means the
#: flag is OMITTED, so the env is only ever emitted for degrees >= 2). An
#: engine absent here cannot serve a TP-sharded config with the frozen
#: launcher fleet — its tp cells are enumerated BLOCKED.
TP_LAUNCH_ENV: Dict[str, str] = {
    "vllm": "CAGE_VLLM_TENSOR_PARALLEL",
    "sglang": "CAGE_SGLANG_TP",
}

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

#: Batch 2 finding W2 (2026-09-18; owner picked option A of A/B/C; ADR-0059
#: amendment of the same date): the ONE port table the launcher env AND every
#: server cell's --api-base derive from, so the server a relaunch starts and
#: the endpoint its cells dial cannot drift apart. Values MIRROR the frozen
#: launchers' defaults (manage_vllm_server.sh PORT="${VLLM_PORT:-8000}",
#: manage_sglang_server.sh PORT="${SGLANG_PORT:-30000}"), pinned structurally
#: by tests/test_run_campaign.py against the scripts. Before W2 no cell
#: carried an endpoint: every cell rode the runner's --api-base default
#: (http://localhost:8000, the vLLM port), so every SGLang cell was sent to a
#: port no SGLang server listens on. An engine absent here has no registered
#: port and its cells refuse at plan time (lmdeploy: no campaign launcher
#: today). BEHAVIOR, not identity: derive_cell_spec never reads argv.
ENGINE_PORTS: Dict[str, int] = {"vllm": 8000, "sglang": 30000}
#: engine -> the launcher env carrying the port (frozen launcher contract),
#: exported on every single-instance relaunch (the tp overlay rides the same
#: launcher) so the launcher's shell default never reaches a campaign server
#: (the A10 rule for max_model_len).
PORT_LAUNCH_ENV: Dict[str, str] = {"vllm": "VLLM_PORT", "sglang": "SGLANG_PORT"}
assert set(ENGINE_PORTS) == set(PORT_LAUNCH_ENV), (
    "ENGINE_PORTS and PORT_LAUNCH_ENV register different engines (Batch 2 W2)"
)
#: manage_vllm_pd.sh: the proxy the runner dials in the pd topology
#: (CAGE_PD_PROXY_PORT, default 8000, mirrored like PD_PREFILL_PORT /
#: PD_DECODE_PORT); pd cells pin --api-base to it and the pd relaunch exports
#: the env.
PD_PROXY_PORT = 8000
PD_PROXY_PORT_ENV = "CAGE_PD_PROXY_PORT"
#: manage_vllm_pd.sh role instance ports: the SAME mirrored defaults the
#: telemetry endpoints above are built from, exported on the pd relaunch so
#: the launcher, the proxy's upstreams and the recorded telemetry endpoints
#: agree by construction (W2 review F3: an operator shell CAGE_PD_PREFILL_PORT
#: would move the instance while the record kept naming 8100).
PD_ROLE_PORT_ENVS: Dict[str, int] = {
    "CAGE_PD_PREFILL_PORT": PD_PREFILL_PORT,
    "CAGE_PD_DECODE_PORT": PD_DECODE_PORT,
}
#: The loopback host every campaign endpoint is dialed on: the runner and the
#: servers share the pod, the launchers health-check on it, and
#: PD_TELEMETRY_ENDPOINTS already spells it out.
API_BASE_HOST = "http://localhost"
#: The runner's per-engine endpoint OVERRIDES: run_experiment.py resolves
#: CAGE_<ENGINE>_API_BASE BEFORE --api-base for sglang/lmdeploy (adapter and
#: cache flush alike). _exec inherits the operator's shell, so an exported
#: override would beat the plan's pin on every cell of that engine: 'run'
#: refuses on PRESENCE, the STALE_INDEX_OPT_IN_ENV rule.
API_BASE_OVERRIDE_ENVS: Tuple[str, ...] = (
    "CAGE_SGLANG_API_BASE",
    "CAGE_LMDEPLOY_API_BASE",
)
#: The finding every endpoint refusal and header record cites.
ENGINE_PORTS_FINDING = "Batch 2 W2"
#: Every launcher port env the driver pins -> its registered value. 'run'
#: refuses a shell value that DIFFERS (an equal value is fine): the preflight
#: gate dials the shell's SGLANG_PORT for its URL (scripts/checks/
#: preflight_check.sh), while every relaunch exports the registered port and
#: the step env wins, so a differing shell value would make the preflight
#: evidence come from a port the campaign never serves on (W2 review F5).
SHELL_PORT_ENVS: Dict[str, int] = {
    **{PORT_LAUNCH_ENV[engine]: port for engine, port in ENGINE_PORTS.items()},
    PD_PROXY_PORT_ENV: PD_PROXY_PORT,
    **PD_ROLE_PORT_ENVS,
}

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

#: ADR-0104 (owner decision 2026-09-16): the dense candidate-POOL size the
#: ranked pipeline reranks before serving. B6 and every arm inheriting the
#: ranked pipeline (B7-B9, B11) retrieve the dense top RERANK_POOL, rerank
#: the whole pool with RERANKER_MODEL and serve the runner's --top-k (3,
#: unchanged); B5 serves the dense top 3 UNRANKED and carries no pool. The
#: pool rides the cell argv as --rerank-pool (never a runner default: the
#: runner's unset default is the LEGACY rerank-exactly-top-k behavior, so a
#: silently dropped flag would collapse B6's pool to the old pipeline
#: without any label changing). The runner refuses a pool without a
#: reranker, so the flag can never leak onto B5 unnoticed.
RERANK_POOL = 10

#: CellSpec retriever axis -> the runner's retrieval argv. The runner splits
#: the axis into --retriever (pipeline) + --reranker-model (ranking stage)
#: + --rerank-pool (ADR-0104 candidate pool, ranked pipeline only);
#: 'rerank' = dense retrieval + the pinned cross-encoder over a RERANK_POOL
#: pool, 'dense' = the same retrieval with ranking DISABLED ('none' is the
#: runner's documented off switch, normalize_reranker_model) and no pool.
#: bm25/rrf have no registered argv here yet: enumerating them refuses
#: (fail closed) rather than guessing.
RETRIEVER_ARGV: Dict[str, Tuple[str, ...]] = {
    "dense": ("--retriever", "dense", "--reranker-model", "none"),
    "rerank": (
        "--retriever", "dense",
        "--reranker-model", RERANKER_MODEL,
        "--rerank-pool", str(RERANK_POOL),
    ),
}

#: Backlog A5 (retrieval pins): the number of retrieved documents every
#: retrieval cell SERVES (the runner's --top-k). The ranked pipeline reranks
#: a RERANK_POOL of candidates and serves this many (ADR-0104); B5 serves the
#: dense top RETRIEVAL_TOP_K unranked. Pinned in every retrieval cell's argv:
#: the runner's argparse default (3 today) must never be what a campaign
#: cell silently inherits, or a runner-default drift would rewrite the
#: served context width under unchanged row keys.
RETRIEVAL_TOP_K: int = 3

#: Backlog A5: the Decision 3B distractor pool size (the first N content
#: deduped, gold-excluded paragraphs of the split widen the retrieval corpus
#: so retrieval is a real search problem). The runner reads it from the env
#: (CAGE_DISTRACTOR_DOCS, default 1000); the driver pins it in every
#: retrieval cell's env. BEHAVIOR, not identity: derive_cell_spec ignores
#: it (identity rides the CAGE_CELL_* seam and nothing else).
DISTRACTOR_DOCS: int = 1000

#: Backlog A5: the runner's IR index root (its --ir-index-dir default). A
#: session registers the root it serves from (SessionGrid.ir_index_root,
#: reviewable in the plan header) and every retrieval cell pins it
#: explicitly; per-dataset/per-model index dirs hang below it (the runner's
#: default_index_dir), and the index content hash guards staleness.
DEFAULT_IR_INDEX_ROOT: str = "./experiments/ir_index"

#: ADR-0099 / backlog A5: the dense retriever's registered pin lives in the
#: freeze artifact's INSTRUMENT_REVISIONS.dense_retriever slot
#: (MyDocs/registration/freeze_resolutions.json; the same resolution chain
#: build_retrieval_gate_table.py consumes). The driver READS the model id
#: from that slot at plan time (resolve_retrieval_pins) and records model +
#: revision in the plan header; it never carries a second copy of the id.
#: The artifact's 'embedding' entry registers the QUALITY module's
#: similarity embedder, a different instrument: never consumed here.
FREEZE_FILE_ENV_VAR: str = "CAGE_FREEZE_RESOLUTIONS"
DEFAULT_FREEZE_FILE: Path = (
    REPO_ROOT / "MyDocs" / "registration" / "freeze_resolutions.json"
)
FREEZE_DENSE_RETRIEVER_SLOT: str = "dense_retriever"
DENSE_RETRIEVER_ADR: str = "ADR-0099"

#: Backlog F5a (per-cell Redis namespaces): the arms whose runner pipeline
#: consults the Redis retrieval-artifact cache (B7 retr-reuse = the runner's
#: 'hybrid' pipeline). Each such cell gets --redis-key-prefix minted from
#: its row key (redis_key_prefix_for_row) plus --flush-redis-namespace, so
#: no two cells ever share cache entries and every cell starts empty. The
#: prefix root and the short-sha width are named here, never inline.
REDIS_CACHE_ARMS: FrozenSet[str] = frozenset({"retr-reuse"})
REDIS_KEY_PREFIX_ROOT: str = "cage"
REDIS_NAMESPACE_SHA_CHARS: int = 12
REDIS_NAMESPACE_RULE: str = "cage:<sha1(row_key)[:12]>"

#: Arms serving the shared corpus-as-prefix block (true CAG, Chan et al.
#: 2412.15605): all get --corpus-prefix-budget. corpus-trunc is handled apart
#: (its budget is a TRUNCATED rung: that difference IS the B12-vs-B3 slot).
CORPUS_BLOCK_ARMS = frozenset({"corpus-reuse", "corpus-fresh", "corpus-comp"})

#: ADR-0106 (owner decision 2026-09-16, charter §7.7(d)): B12 (corpus-trunc)
#: is a DESCENDING corpus-budget ladder whose top point is B3's own
#: full-budget cell (never duplicated). The registered rungs live on
#: SessionGrid.corpus_trunc_budgets; each enumerates ONE cell wherever B12
#: is carried, the rung rides the identity seam (CAGE_CELL_CORPUS_BUDGET,
#: CellSpec.corpus_budget_tokens) and the argv as the EXPLICIT --corpus-rung
#: (the runner's A4 guard refuses a manifest whose ladder lacks the rung).
#: The arm the ladder applies to, and the ADR the plan header cites.
CORPUS_TRUNC_ARM = "corpus-trunc"
CORPUS_TRUNC_ADR = "ADR-0106"

#: ADR-0106 (repair 2026-09-16): a B12 rung serves ONLY from a query
#: manifest carrying the ladder (the runner's A4 guard refuses any other
#: source, and the non-manifest fallback DROPS out-of-corpus queries), so
#: every rung cell whose dataset has no manifest registered at plan time is
#: blocked_on this text: visible debt in the plan and a 'run' refusal
#: without --skip-blocked, never a silently unrunnable step. Manifests are
#: registered per dataset with ``plan --query-manifest <dataset>=<path>``
#: and validated at plan time (dataset, block budget, every rung).
TRUNC_MANIFEST_BLOCKED_ON_FMT = (
    "query-manifest:{dataset} ({adr}: a B12 rung serves only from a query "
    "manifest carrying the ladder; plan with --query-manifest {dataset}=<path>)"
)


def trunc_manifest_blocked_on(dataset: str) -> str:
    """The blocked_on text for a B12 rung cell with no manifest for ``dataset``."""
    return TRUNC_MANIFEST_BLOCKED_ON_FMT.format(dataset=dataset, adr=CORPUS_TRUNC_ADR)

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


@dataclass(frozen=True)
class RetrievalPins:
    """The freeze-artifact half of the A5 retrieval pins (ADR-0099).

    ``embedding_model`` is what every retrieval cell pins as
    --embedding-model; ``embedding_revision`` is the frozen HF commit the
    slot records (provenance, recorded in the plan header); ``freeze_file``
    is the resolved artifact path the values were read from.
    """

    embedding_model: str
    embedding_revision: str
    freeze_file: str


def resolve_retrieval_pins(freeze_file: Optional[Path] = None) -> RetrievalPins:
    """Read the dense retriever pin from the freeze artifact (fail closed).

    Source order: the explicit ``freeze_file``, else ``$CAGE_FREEZE_RESOLUTIONS``
    (FREEZE_FILE_ENV_VAR), else DEFAULT_FREEZE_FILE. Consumes ONLY
    ``INSTRUMENT_REVISIONS.dense_retriever`` (FREEZE_DENSE_RETRIEVER_SLOT):
    the artifact's ``embedding`` entry is the quality module's similarity
    embedder and is never a fallback. Every gap refuses with PlanError
    naming the fix; there is no default model id in this driver.
    """
    if freeze_file is None:
        override = (os.environ.get(FREEZE_FILE_ENV_VAR) or "").strip()
        freeze_file = Path(override) if override else DEFAULT_FREEZE_FILE
    freeze_file = Path(freeze_file)
    slot = f"INSTRUMENT_REVISIONS.{FREEZE_DENSE_RETRIEVER_SLOT}"
    fix = (
        f"point --freeze-file / ${FREEZE_FILE_ENV_VAR} at the frozen registration "
        f"artifact (default {DEFAULT_FREEZE_FILE}); a campaign plan never pins "
        f"an unregistered retriever ({DENSE_RETRIEVER_ADR}, backlog A5)"
    )
    if not freeze_file.is_file():
        raise PlanError(
            f"{freeze_file}: freeze artifact missing: the retrieval cells' "
            f"--embedding-model pin ({slot}) cannot be read; {fix}"
        )
    try:
        data = json.loads(freeze_file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PlanError(f"{freeze_file}: freeze artifact cannot be read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PlanError(f"{freeze_file}: freeze artifact is not valid JSON: {exc}") from exc
    revisions = data.get("INSTRUMENT_REVISIONS") if isinstance(data, dict) else None
    if not isinstance(revisions, Mapping):
        raise PlanError(
            f"{freeze_file}: no INSTRUMENT_REVISIONS mapping: the artifact carries "
            f"no instrument pins to consume; {fix}"
        )
    entry = revisions.get(FREEZE_DENSE_RETRIEVER_SLOT)
    if not isinstance(entry, Mapping):
        raise PlanError(
            f"{freeze_file}: INSTRUMENT_REVISIONS has no "
            f"{FREEZE_DENSE_RETRIEVER_SLOT!r} mapping: the dense retriever pin is "
            f"absent. Fix: record {{model, revision, resolved}} under {slot}. Do "
            f"NOT repurpose INSTRUMENT_REVISIONS.embedding (the quality module's "
            f"similarity embedder, a different instrument); {fix}"
        )
    model = entry.get("model")
    revision = entry.get("revision")
    if not isinstance(model, str) or not model.strip():
        raise PlanError(
            f"{freeze_file}: {slot}.model is {model!r}: a pin without a model "
            f"pins nothing; {fix}"
        )
    if not isinstance(revision, str) or not revision.strip():
        raise PlanError(
            f"{freeze_file}: {slot}.revision is {revision!r}: the dense retriever "
            f"revision pin is absent; resolve and record the HF commit hash; {fix}"
        )
    return RetrievalPins(
        embedding_model=model.strip(),
        embedding_revision=revision.strip(),
        freeze_file=str(freeze_file.resolve()),
    )


def redis_key_prefix_for_row(row_key: str) -> str:
    """The F5a Redis namespace of one cell: ``cage:<sha1(row_key)[:12]>``.

    Minted from the CellSpec row key (unique per cell by construction), so
    two cells can never share retrieval-artifact entries; the runner's
    RedisClient composes ``<prefix>:<namespace>:<key>`` below it. An empty
    row key refuses (a namespace must name a cell).
    """
    if not isinstance(row_key, str) or not row_key.strip():
        raise PlanError(
            f"redis namespace needs a row key, got {row_key!r} (backlog F5a)"
        )
    digest = hashlib.sha1(row_key.encode("utf-8")).hexdigest()
    return f"{REDIS_KEY_PREFIX_ROOT}:{digest[:REDIS_NAMESPACE_SHA_CHARS]}"


def engine_api_base(engine: str, topology: str = "single") -> str:
    """The endpoint a cell of ``engine`` dials (Batch 2 W2), from the one
    port table: ``http://localhost:<ENGINE_PORTS[engine]>`` for the single
    and tp topologies (both ride the single-instance launcher), the pd proxy
    (``PD_PROXY_PORT``) for vLLM pd (manage_vllm_pd.sh is the only PD
    launcher). The in-process hf oracle has no endpoint and refuses here, as
    does any engine with no registered port or pd proxy (fail closed: the
    runner's --api-base default is the vLLM port and must never be what
    another engine's cell silently inherits).
    """
    if topology == "pd":
        if engine != "vllm":
            raise PlanError(
                f"engine {engine!r} has no registered pd proxy (manage_vllm_pd.sh "
                f"is vLLM-only; {ENGINE_PORTS_FINDING}): refusing to pin its pd "
                "cells to the vLLM proxy"
            )
        return f"{API_BASE_HOST}:{PD_PROXY_PORT}"
    port = ENGINE_PORTS.get(engine)
    if port is None:
        raise PlanError(
            f"engine {engine!r} has no registered port (ENGINE_PORTS: "
            f"{sorted(ENGINE_PORTS)}; {ENGINE_PORTS_FINDING}): refusing to let "
            "its cells inherit the runner's --api-base default"
        )
    return f"{API_BASE_HOST}:{port}"


def _ordered(baseline_ids: frozenset) -> Tuple[str, ...]:
    """Deterministic B-number order for a baseline-id set (B2 < B10)."""
    return tuple(sorted(baseline_ids, key=_baseline_num))


# ---------------------------------------------------------------------------
# Session grid registration (§7.6 / §7.6.1 — THE registered enumeration
# source; no hand lists anywhere downstream)
# ---------------------------------------------------------------------------


def _grid_datasets(grid: "SessionGrid") -> FrozenSet[str]:
    """Every dataset a cell of this grid can carry (QA rows, hf slice, F2/F3,
    and the D5#5 RULER instrument when the session registers it)."""
    names = set(grid.f1_datasets)
    for _bid, datasets in grid.hf_oracle_cells:
        names.update(datasets)
    names.update({grid.f2_dataset, grid.f3_dataset})
    if grid.f2_ruler_baselines:
        names.add("ruler")
    return frozenset(names)


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
    # - corpus_trunc_budgets: B12's TRUNCATED corpus budgets ("store less
    #   than you know"), the ADR-0106 descending ladder below B3's full
    #   budget: (1400, 700) = 2x and 4x truncation (2x matches the
    #   compression arms' ratio, run_compression.sh's 2x2 convention). One
    #   corpus-trunc cell is enumerated PER RUNG; the 2800 point of the
    #   ladder is B3's own cell and is never a rung.
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
    corpus_trunc_budgets: Tuple[int, ...] = (1400, 700)
    retr_trunc_kept_docs: int = 1
    dist_pd_split: float = 0.5
    dist_budget_r: float = 1.0
    # Backlog A5: the IR index root every retrieval cell of this session
    # serves from (--ir-index-dir, EXPLICIT on every retrieval cell; the
    # runner's own default is DEFAULT_IR_INDEX_ROOT and never reaches a
    # cell silently). A registered PATH, reviewable in the plan header.
    ir_index_root: str = DEFAULT_IR_INDEX_ROOT
    # §6.4 anchor fine r-grid overlay on F2 (Group A ONLY per §6.8): extra
    # (budget × rate) coordinates enumerated ON TOP of the factorial,
    # deduplicated by coordinate. Both empty on every non-anchor session.
    f2_fine_budgets: Tuple[float, ...] = ()
    f2_fine_rates: Tuple[float, ...] = ()
    # D5 item 5 RULER pairing (W4.4 grid half): the instrument's F2 baselines
    # (MUST be a subset of f2_baselines so the matched real-text Qasper twin
    # exists BY CONSTRUCTION — the HELMET validity boundary) and the
    # registered task subset (RULER_F2_TASKS literals, per-task steps).
    f2_ruler_baselines: Tuple[str, ...] = ()
    f2_ruler_tasks: Tuple[str, ...] = ()
    # W4.2 serving GPU counts (feed cell.json gpu_count, §6.6b basis):
    # - serving_tp: the tensor-parallel degree every SINGLE-topology relaunch
    #   of this session launches with (1 = the plain single-GPU launch,
    #   byte-identical to pre-W4.2 plans; Group B serves TP=4 per §7.6 e5 —
    #   a TP-sharded single-instance config counts its ranks as gpu_count).
    # - dist_tp_size: the DIST tp-overlay leg's TP degree (None = the tp leg
    #   is unregistered — its cells stay enumerated-but-BLOCKED).
    # - dist_pd_role_gpus: (prefill, decode) GPU counts of the pd leg's role
    #   instances; gpu_count = their sum. The frozen pd launcher applies ONE
    #   CAGE_VLLM_TENSOR_PARALLEL to BOTH roles, so unequal counts refuse.
    serving_tp: int = 1
    dist_tp_size: Optional[int] = None
    dist_pd_role_gpus: Tuple[int, int] = (1, 1)
    # Backlog A10: the ONE request-length cap every relaunch of this session
    # launches with (env MAX_MODEL_LEN_ENV, both engines): RULER SHAPE-32K
    # (32,512 in + 256 out) plus long Qasper papers; uniform per session
    # (_serving_config.sh uniform-regime rule). The KV pool is
    # byte-budgeted separately by gate (j), so this caps request length
    # only. A grid registering RULER tasks refuses a value below SHAPE-32K.
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    # Backlog A9 / DECISION.md A1 per-row N (see the module constants of the
    # same names for the table): the class n every cell's --num-queries is
    # drawn from, the primary-predicate pin (engine + baseline pair), and
    # the A5 per-dataset achievable n, which may only LOWER a class n for
    # that dataset (header caveat; e.g. qasper whose dev split cannot supply
    # 2,000 paper-first draws). Registered per session so the header reviews
    # THE values, never a runner default.
    n_primary: int = N_PRIMARY
    n_secondary: int = N_SECONDARY
    n_identity: int = N_IDENTITY
    window_requests: int = WINDOW_REQUESTS
    primary_engine: str = PRIMARY_ENGINE
    primary_baselines: Tuple[str, ...] = PRIMARY_BASELINES
    achievable_n: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        problems: List[str] = []
        # A9 per-row N registration (fail closed on shapes; the classifier
        # and the manifest-coverage check refuse per cell at plan time).
        for name in ("n_primary", "n_secondary", "n_identity", "window_requests"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                problems.append(
                    f"{name}={value!r} must be an integer >= 1 (DECISION.md A1 "
                    "per-row N)"
                )
        if self.primary_engine not in BACKEND_OF_ENGINE or self.primary_engine == "hf":
            problems.append(
                f"primary_engine={self.primary_engine!r} must be a server engine "
                f"of the runner vocabulary ({sorted(e for e in BACKEND_OF_ENGINE if e != 'hf')}); "
                "the primary predicate cells are per-query F1 cells on the "
                "pinned engine, never the in-process oracle (A1)"
            )
        elif self.primary_engine not in self.f1_engines:
            problems.append(
                f"primary_engine={self.primary_engine!r} is not one of this "
                f"session's f1_engines {list(self.f1_engines)}: the primary "
                "class would enumerate ZERO cells (A1 pins the #4 contrast to "
                "an engine the session actually serves)"
            )
        if not isinstance(self.primary_baselines, tuple) or not self.primary_baselines:
            problems.append(
                f"primary_baselines={self.primary_baselines!r} must be a non-empty "
                "tuple of baseline ids (A1: the #4 contrast pair)"
            )
        else:
            unknown_primary = sorted(set(self.primary_baselines) - set(BASELINES))
            if unknown_primary:
                problems.append(
                    f"primary_baselines has unregistered baseline id(s) "
                    f"{unknown_primary} (registered: {sorted(BASELINES)})"
                )
        if not isinstance(self.achievable_n, Mapping):
            problems.append(
                f"achievable_n={self.achievable_n!r} must be a mapping dataset -> n"
            )
        else:
            grid_datasets = _grid_datasets(self)
            for dataset, value in self.achievable_n.items():
                if dataset not in grid_datasets:
                    problems.append(
                        f"achievable_n[{dataset!r}] names a dataset no cell of "
                        f"this session carries (registered: {sorted(grid_datasets)})"
                    )
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    problems.append(
                        f"achievable_n[{dataset!r}]={value!r} must be an integer >= 1"
                    )
                elif isinstance(self.n_primary, int) and value > self.n_primary:
                    problems.append(
                        f"achievable_n[{dataset!r}]={value} exceeds n_primary="
                        f"{self.n_primary}: the A5 branch only LOWERS a class n "
                        "for a dataset whose split cannot supply it"
                    )
        for name in (
            "corpus_prefix_budget_tokens",
            "retr_trunc_kept_docs",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                problems.append(f"{name}={value!r} must be an integer >= 1")
        if not isinstance(self.ir_index_root, str) or not self.ir_index_root.strip():
            problems.append(
                f"ir_index_root={self.ir_index_root!r} must be a non-empty path "
                "(backlog A5: every retrieval cell pins --ir-index-dir to it)"
            )
        # ADR-0106 ladder: non-empty, integer rungs, strictly descending, every
        # rung < the full budget (the full budget is B3's own cell; a rung at or
        # above it is not a truncation and would duplicate B3 under a B12 label).
        ladder = self.corpus_trunc_budgets
        if not isinstance(ladder, tuple) or not ladder:
            problems.append(
                f"corpus_trunc_budgets={ladder!r} must be a non-empty tuple of "
                "descending rung budgets (ADR-0106)"
            )
        else:
            previous: Optional[int] = None
            for rung in ladder:
                if not isinstance(rung, int) or isinstance(rung, bool) or rung < 1:
                    problems.append(
                        f"corpus_trunc_budgets entry {rung!r} must be an integer >= 1"
                    )
                    continue
                if (
                    isinstance(self.corpus_prefix_budget_tokens, int)
                    and rung >= self.corpus_prefix_budget_tokens
                ):
                    problems.append(
                        f"corpus_trunc_budgets rung {rung} must be < "
                        f"corpus_prefix_budget_tokens={self.corpus_prefix_budget_tokens} "
                        "(the full budget is B3's own cell, ADR-0106, and a "
                        "rung above it is not a truncation)"
                    )
                if previous is not None and rung >= previous:
                    problems.append(
                        f"corpus_trunc_budgets={ladder} must be strictly descending "
                        "(ADR-0106: a descending ladder, no duplicate rungs)"
                    )
                previous = rung
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
        # §6.4 fine overlay: both axes registered together or not at all — a
        # budget list with no rates (or vice versa) enumerates nothing and
        # silently drops the registered graft.
        if bool(self.f2_fine_budgets) != bool(self.f2_fine_rates):
            problems.append(
                "f2_fine_budgets and f2_fine_rates must be registered together "
                "(§6.4: the fine grid is budgets × rates; one side alone is an "
                "empty registration)"
            )
        # D5#5 RULER pairing: tasks and baselines travel together, tasks must
        # be loader-registered literals, and every ruler baseline needs its
        # matched Qasper twin in f2_baselines (the HELMET validity boundary).
        if bool(self.f2_ruler_baselines) != bool(self.f2_ruler_tasks):
            problems.append(
                "f2_ruler_baselines and f2_ruler_tasks must be registered "
                "together (D5#5: the RULER instrument is baselines × tasks)"
            )
        unknown_tasks = sorted(set(self.f2_ruler_tasks) - set(_ruler._TASKS))
        if unknown_tasks:
            problems.append(
                f"f2_ruler_tasks has task(s) {unknown_tasks} not registered in "
                f"src/data/ruler._TASKS ({list(_ruler._TASKS)}) — the loader "
                "would refuse them; fix the registration, not the loader"
            )
        if len(set(self.f2_ruler_tasks)) != len(self.f2_ruler_tasks):
            problems.append(
                f"f2_ruler_tasks {list(self.f2_ruler_tasks)} has duplicates — "
                "each task claims one window-ordinal range; a duplicate would "
                "double-claim it"
            )
        unpaired = sorted(set(self.f2_ruler_baselines) - set(self.f2_baselines))
        if unpaired:
            problems.append(
                f"f2_ruler_baselines {unpaired} absent from f2_baselines — "
                "D5#5 mandates every RULER cell be PAIRED with a matched "
                "real-text Qasper cell at the same coordinate; an unpaired "
                "ruler baseline has no twin"
            )
        # W4.2 GPU-count registration knobs (fail-closed shapes; the actual
        # topology/count derivation refuses per cell at plan/write time).
        if not isinstance(self.serving_tp, int) or isinstance(self.serving_tp, bool) or self.serving_tp < 1:
            problems.append(
                f"serving_tp={self.serving_tp!r} must be an integer >= 1 "
                "(the TP degree single-topology relaunches launch with)"
            )
        if self.dist_tp_size is not None and (
            not isinstance(self.dist_tp_size, int)
            or isinstance(self.dist_tp_size, bool)
            or self.dist_tp_size < 2
        ):
            problems.append(
                f"dist_tp_size={self.dist_tp_size!r} must be None (tp leg "
                "unregistered) or an integer >= 2 — a TP=1 'tp' leg is a "
                "topology/count contradiction"
            )
        # Backlog A10: a positive integer; with RULER tasks registered, at
        # least SHAPE-32K (a shorter cap would refuse every RULER request at
        # the server, or silently truncate it: mislabeled instrument data).
        if (
            not isinstance(self.max_model_len, int)
            or isinstance(self.max_model_len, bool)
            or self.max_model_len < 1
        ):
            problems.append(
                f"max_model_len={self.max_model_len!r} must be an integer >= 1 "
                "(backlog A10: the uniform per-session request-length cap "
                f"every relaunch carries as {MAX_MODEL_LEN_ENV})"
            )
        elif self.f2_ruler_tasks and self.max_model_len < (
            RULER_CONTEXT_TOKENS + RULER_OUTPUT_TOKENS
        ):
            problems.append(
                f"max_model_len={self.max_model_len} is below RULER SHAPE-32K "
                f"({RULER_CONTEXT_TOKENS} + {RULER_OUTPUT_TOKENS} = "
                f"{RULER_CONTEXT_TOKENS + RULER_OUTPUT_TOKENS}) but this grid "
                f"registers RULER tasks {list(self.f2_ruler_tasks)} (backlog "
                "A10: every RULER request would exceed the server cap)"
            )
        if (
            not isinstance(self.dist_pd_role_gpus, tuple)
            or len(self.dist_pd_role_gpus) != 2
            or any(
                not isinstance(v, int) or isinstance(v, bool) or v < 1
                for v in self.dist_pd_role_gpus
            )
        ):
            problems.append(
                f"dist_pd_role_gpus={self.dist_pd_role_gpus!r} must be a "
                "(prefill, decode) pair of integers >= 1"
            )
        elif self.dist_pd_role_gpus[0] != self.dist_pd_role_gpus[1]:
            problems.append(
                f"dist_pd_role_gpus={self.dist_pd_role_gpus!r} must be equal — "
                "manage_vllm_pd.sh applies ONE CAGE_VLLM_TENSOR_PARALLEL to "
                "BOTH role instances (frozen launcher contract); unequal role "
                "counts are unrealizable"
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
        # §6.4 anchor fine r-grid (Group A ONLY, §6.8) — see the constants'
        # scope-derivation note; new coordinates = {1.25, 0.375} × {0.85, 1.05}.
        f2_fine_budgets=ANCHOR_FINE_BUDGET_LEVELS,
        f2_fine_rates=ANCHOR_FINE_RATE_FRACTIONS,
        # D5#5 RULER pairing: B1 (gold-fresh) ONLY — the instrument's payload
        # IS the served context, which is exactly the gold-context arm's
        # shape; the retrieval arms (B5/B6/B9/B11) retrieve from a dataset
        # corpus RULER does not define, so registering them would fabricate a
        # retrieval workload the instrument never specified (conservative
        # pin, recorded as an ADR input). Every ruler cell's Qasper twin is
        # the B1 qasper cell at the identical (engine, r, rate) coordinate.
        f2_ruler_baselines=("B1",),
        f2_ruler_tasks=RULER_F2_TASKS,
        f3_baselines=_ordered(REUSE_SET),
        f3_engines=("vllm", "sglang"),
        f3_budgets=REDUCED_BUDGET_LEVELS,
        f3_rates=REDUCED_RATE_FRACTIONS,
        f3_dataset="qasper",
        dist_cells=(),
        corpus_prefix_budget_tokens=2800,
        corpus_trunc_budgets=(1400, 700),
        retr_trunc_kept_docs=1,
    ),
    # Session 'b' = charter §7.6 Group B (Llama-3.3-70B) + the Run-C-prime
    # DIST overlay (standing Plan-B scope, W4.6). Conservative pins where the
    # charter is ambiguous — each RECORDED here as an ADR input:
    # - DIST overlay: §7.6.1's Distributed row for B is "—", but the standing
    #   Plan-B / Run-C-prime scope carries the transfer pair {B1, B3} (the
    #   §7.6.1 Group-C pair) on THIS session's hardware. Registered on vLLM
    #   ONLY: contrast #18 is topology-paired WITHIN one engine, and SGLang
    #   PD is T3.4-gated — sglang pd cells are NOT registered (registering
    #   them blocked would imply a pending leg the scope does not carry).
    # - tp leg TP=8 / pd leg 4+4 roles (task-pinned Plan-B scope): both legs
    #   serve 8 GPUs — the iso-GPU version of the #18 pair; both serve
    #   floor(dist_budget_r × D) total bytes (§6.6a iso-aggregate-bytes).
    #   BOTH legs' budget envs are denominated PER-RANK under the planner's
    #   registered --kv-cache-memory-bytes convention (tp leg: total // 8;
    #   pd leg: each §6.5 role pool // 4 — see _pd_budget_env). Per-rank vs
    #   whole-pool live semantics is the plan's verify_live entry: gate (j)
    #   validates realized bytes at S0 BEFORE campaign data, and would fail
    #   BOTH legs together (never silently skew one side of the pair).
    # - serving_tp=4 for every non-DIST cell per §7.6 Group B / §7.7(e)
    #   (70B BF16 needs TP=4 for a comfortable budget sweep; TP=2 is the
    #   pinched sensitivity point, NOT registered here).
    # - F1 = B1-B12 × {vllm, sglang} × the four QA datasets ("quality
    #   slice"). The §7.6 "+ SCBench slice" is NOT yet registered: its
    #   2-subset selection (scbench_kv / scbench_qa_eng via
    #   CAGE_SCBENCH_SUBSET) needs the same per-subset window-range seam the
    #   RULER tasks got, and registering it as ONE undifferentiated dataset
    #   would leave the subset an unregistered degree of freedom — named
    #   deferral, additive later.
    # - HF oracle: the same reduced 10-cell slice as session a (batch-1
    #   device_map rides the same 4-GPU box, §7.7(e); the charter pins no
    #   Group-B-specific oracle density — conservative reuse of the anchor
    #   slice).
    # - LMDeploy/TurboMind F1 carriage (§7.6 B "all 4") lands with its
    #   launcher wiring, exactly as on session a.
    # - F2/F3 densities per §7.6.1 row B: FRESH × 3×3 and REUSE × 3×3
    #   (§6.8 pruning rule); no fine grid, no RULER pairing (anchor-only
    #   grafts).
    "b": SessionGrid(
        session="b",
        group="B",
        model="llama-3.3-70b",
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
        f2_budgets=REDUCED_BUDGET_LEVELS,
        f2_rates=REDUCED_RATE_FRACTIONS,
        f2_dataset="qasper",
        f3_baselines=_ordered(REUSE_SET),
        f3_engines=("vllm", "sglang"),
        f3_budgets=REDUCED_BUDGET_LEVELS,
        f3_rates=REDUCED_RATE_FRACTIONS,
        f3_dataset="qasper",
        dist_cells=(
            ("B1", "vllm", "tp"),
            ("B1", "vllm", "pd"),
            ("B3", "vllm", "tp"),
            ("B3", "vllm", "pd"),
        ),
        corpus_prefix_budget_tokens=2800,
        corpus_trunc_budgets=(1400, 700),
        retr_trunc_kept_docs=1,
        dist_pd_split=0.5,
        dist_budget_r=1.0,
        serving_tp=4,
        dist_tp_size=8,
        dist_pd_role_gpus=(4, 4),
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
            "the cd-* sessions land with the Group-C/D topology registrations — "
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
    """One enumerated cell: the tuple, its numbered baseline, its workload.

    ``grids`` is the F2 grid-membership marker (None outside F2);
    ``ruler_task``/``window_ordinal_base`` carry the D5#5 per-task RULER
    pairing — per-task steps share a row key (task is not a CellSpec axis),
    so each claims its own window-ordinal range ``(base, base+windows]``.
    """

    spec: CellSpec
    baseline_id: str
    dataset: str
    blocked_on: Optional[str] = None
    grids: Optional[Tuple[str, ...]] = None
    ruler_task: Optional[str] = None
    window_ordinal_base: int = 0


def _cell_gpu_count(grid: SessionGrid, spec: CellSpec) -> Optional[int]:
    """W4.2 derivation — the integer GPU count the cell's serving stack
    launches with (None ONLY for a blocked tp cell whose degree is
    unregistered; an executable cell without one refuses at step build).

    - single: the session's registered serving_tp (1 on the anchor; a
      TP-sharded single-instance config counts its ranks — §7.6 B e5). The
      in-process hf oracle rides the same box (batch-1 device_map, §7.7(e)).
    - tp: the registered dist_tp_size (None => the leg is unregistered).
    - pd: the two registered role GPU counts summed (§6.5 roles).
    """
    if spec.topology == "tp":
        return grid.dist_tp_size
    if spec.topology == "pd":
        return grid.dist_pd_role_gpus[0] + grid.dist_pd_role_gpus[1]
    return grid.serving_tp


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

    def _rung_specs(bid: str, **axes: Any) -> List[CellSpec]:
        # ADR-0106: the corpus-trunc baseline enumerates ONE cell per
        # registered rung (descending ladder order); every other baseline
        # is exactly one cell, carrying no rung coordinate.
        if BASELINES[bid].arm == CORPUS_TRUNC_ARM:  # type: ignore[index]
            return [
                CellSpec.from_baseline(
                    bid, model=grid.model, corpus_budget_tokens=rung, **axes  # type: ignore[arg-type]
                )
                for rung in grid.corpus_trunc_budgets
            ]
        return [CellSpec.from_baseline(bid, model=grid.model, **axes)]  # type: ignore[arg-type]

    # F1 — locality, prefix ON, sub-pressure (no budget/rate coordinates).
    for bid in grid.f1_baselines:
        for engine in grid.f1_engines:
            for dataset in grid.f1_datasets:
                for spec in _rung_specs(bid, engine=engine, family="F1"):
                    cells.append(_planned(spec, bid, dataset))

    # F1 HF-oracle reduced slice (§7.3: hf carries family F1 only —
    # CellSpec itself refuses anything else, making the guard structural).
    for bid, datasets in grid.hf_oracle_cells:
        for dataset in datasets:
            for spec in _rung_specs(bid, engine="hf", family="F1"):
                cells.append(_planned(spec, bid, dataset))

    # F2 — pressure, prefix OFF. Grid points = the §6.1/§6.8 factorial PLUS
    # the §6.4 anchor fine overlay, DEDUPLICATED by coordinate: a coordinate
    # on both grids is ONE cell carrying both memberships (enumerating it
    # twice would double-run the same tuple). Insertion order: factorial
    # first, then the fine-only additions (the sort below reorders anyway).
    f2_points: Dict[Tuple[float, float], Tuple[str, ...]] = {}
    for r in grid.f2_budgets:
        for frac in grid.f2_rates:
            f2_points[(r, frac)] = (GRID_D6_FACTORIAL,)
    for r in grid.f2_fine_budgets:
        for frac in grid.f2_fine_rates:
            memberships = f2_points.get((r, frac), ())
            f2_points[(r, frac)] = memberships + (GRID_ANCHOR_FINE,)

    def _f2_planned(
        bid: str,
        engine: str,
        r: float,
        frac: float,
        memberships: Tuple[str, ...],
        dataset: str,
        *,
        ruler_task: Optional[str] = None,
        window_ordinal_base: int = 0,
    ) -> PlannedCell:
        spec = CellSpec.from_baseline(
            bid,
            engine=engine,  # type: ignore[arg-type]
            model=grid.model,  # type: ignore[arg-type]
            family="F2",
            budget_r=r,
            rate_frac=frac,
        )
        return PlannedCell(
            spec=spec,
            baseline_id=bid,
            dataset=dataset,
            blocked_on=_launch_blocked_on(spec),
            grids=memberships,
            ruler_task=ruler_task,
            window_ordinal_base=window_ordinal_base,
        )

    for bid in grid.f2_baselines:
        for engine in grid.f2_engines:
            for (r, frac), memberships in f2_points.items():
                cells.append(
                    _f2_planned(bid, engine, r, frac, memberships, grid.f2_dataset)
                )

    # F2 RULER pairing (D5#5): the instrument rides the SAME F2 grid points,
    # dataset 'ruler' at SHAPE-32K, one step per registered task. Per-task
    # steps share the row key (task is not a CellSpec axis), so each claims
    # its own window-ordinal range via window_ordinal_base — the ordinal
    # ranges are disjoint by construction (task_index × replications).
    for bid in grid.f2_ruler_baselines:
        for engine in grid.f2_engines:
            for (r, frac), memberships in f2_points.items():
                for task_index, task in enumerate(grid.f2_ruler_tasks):
                    cells.append(
                        _f2_planned(
                            bid,
                            engine,
                            r,
                            frac,
                            memberships,
                            "ruler",
                            ruler_task=task,
                            window_ordinal_base=task_index * grid.replications,
                        )
                    )

    # F3 — interaction, prefix ON × pressure, reduced grid (§6.8).
    for bid in grid.f3_baselines:
        for engine in grid.f3_engines:
            for r in grid.f3_budgets:
                for frac in grid.f3_rates:
                    for spec in _rung_specs(
                        bid, engine=engine, family="F3", budget_r=r, rate_frac=frac
                    ):
                        cells.append(_planned(spec, bid, grid.f3_dataset))

    # DIST — topology overlay. pd cells on vllm are EXECUTABLE since T3.2
    # (manage_vllm_pd.sh + pd_proxy.py); they carry the --allow-pd gate label
    # instead of blocked_on. tp cells are EXECUTABLE (W4.6) when the session
    # registered a dist_tp_size AND the engine has a T3.1 TP launch env.
    # Everything else stays enumerated-but-blocked: pd on an engine with no
    # PD launcher, and the tp overlay on an unregistered degree or a
    # TP-env-less engine ('run' refuses blocked cells loudly).
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
        elif grid.dist_tp_size is None:
            blocked = TP_DIST_BLOCKED_ON
        elif engine not in TP_LAUNCH_ENV:
            blocked = (
                f"{engine} launcher has no T3.1 tensor-parallel env "
                f"(registered: {sorted(TP_LAUNCH_ENV)})"
            )
        else:
            blocked = _launch_blocked_on(spec)  # arm levers still gate
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
    """The ONE prefix-cache rule: a cell serves with the engine prefix cache
    OFF when its family is F2 (the prefix-OFF pressure family) OR its arm is
    in PREFIX_OFF_ARMS (ADR-0103: corpus-fresh, B4, in every family); every
    other cell serves ON.

    ADR-0103 closes a mislabeled-duplicate exposure (the failure class the
    module docstring names): the B4 runner token ``no_cache`` only labels
    telemetry, so without this clause B4 and B3 shared one prefix-ON server
    and were byte-identical serving twins under different names. Making the
    prefix mode part of the arm's serving config gives B4 its own relaunch
    group (the grouping key already carries prefix_off) while its family
    carriage is untouched.
    """
    return spec.family == "F2" or spec.arm in PREFIX_OFF_ARMS


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
        # ADR-0106: a B12 slice's rung cells run in DESCENDING budget order
        # (the ladder as the charter reads it); non-rung cells carry 0 so
        # their order is byte-identical to pre-ADR-0106 plans.
        0 if spec.corpus_budget_tokens is None else -spec.corpus_budget_tokens,
        # Per-task RULER steps share every component above (one row key, one
        # dataset) — the ordinal base keeps the order total and puts the
        # tasks' window ranges in ascending order (deterministic plans).
        cell.window_ordinal_base,
        _lever_key(cell.ruler_task),
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
# Per-row N (backlog A9, DECISION.md A1/A5): row class -> --num-queries
# ---------------------------------------------------------------------------


def classify_row(
    spec: CellSpec,
    baseline_id: str,
    dataset: str,
    *,
    primary_engine: str,
    primary_baselines: Sequence[str],
) -> str:
    """The DECISION.md A1 row class of one cell, as a pure function of the
    cell tuple and the primary-predicate pin (so load_plan can re-derive it
    from the plan header without a SessionGrid):

    - engine hf -> "identity" (the HF-oracle / T=0 identity cells);
    - family F2/F3, or the RULER instrument dataset -> "window" (loaded
      cells: W requests per window);
    - family DIST -> "identity" (the TTFT-only topology contrast #18: a
      closed-loop latency pair, never a per-query predicate cell);
    - family F1 with baseline in ``primary_baselines`` on ``primary_engine``
      -> "primary" (the #4 contrast cells); every other F1 cell ->
      "secondary".

    Any other family refuses (PlanError): a new family must register its
    class here, never inherit one silently.
    """
    if spec.engine == "hf":
        return "identity"
    if spec.family in _PRESSURE_FAMILIES or dataset == "ruler":
        return "window"
    if spec.family == "DIST":
        return "identity"
    if spec.family == "F1":
        if baseline_id in primary_baselines and spec.engine == primary_engine:
            return "primary"
        return "secondary"
    raise PlanError(
        f"cell {spec.to_row_key()} has family {spec.family!r} with no registered "
        f"A1 row class (registered families: F1, F2, F3, DIST) - refusing to "
        "guess its --num-queries"
    )


def row_class(grid: SessionGrid, cell: PlannedCell) -> str:
    """``classify_row`` under the session's registered primary pin."""
    return classify_row(
        cell.spec,
        cell.baseline_id,
        cell.dataset,
        primary_engine=grid.primary_engine,
        primary_baselines=grid.primary_baselines,
    )


def class_n(
    cls: str,
    dataset: str,
    *,
    n_primary: int,
    n_secondary: int,
    n_identity: int,
    window_requests: int,
    achievable_n: Mapping[str, int],
) -> int:
    """The registered n for a row class, LOWERED to the dataset's achievable
    n when one is registered (A5); never raised by it."""
    table = {
        "primary": n_primary,
        "secondary": n_secondary,
        "identity": n_identity,
        "window": window_requests,
    }
    if cls not in table:
        raise PlanError(f"unknown row class {cls!r} (registered: {list(ROW_CLASSES)})")
    n = table[cls]
    ceiling = achievable_n.get(dataset)
    if ceiling is not None and ceiling < n:
        return int(ceiling)
    return n


def cell_num_queries(grid: SessionGrid, cell: PlannedCell) -> Tuple[str, int]:
    """(row_class, --num-queries) for one cell under the session's registration."""
    cls = row_class(grid, cell)
    return cls, class_n(
        cls,
        cell.dataset,
        n_primary=grid.n_primary,
        n_secondary=grid.n_secondary,
        n_identity=grid.n_identity,
        window_requests=grid.window_requests,
        achievable_n=grid.achievable_n,
    )


def _achievable_n_caveat(grid: SessionGrid) -> Optional[str]:
    """The header caveat naming every dataset whose class n is lowered (A5)."""
    if not grid.achievable_n:
        return None
    lowered = ", ".join(
        f"{ds}={n}" for ds, n in sorted(grid.achievable_n.items())
    )
    return (
        f"achievable_n lowers the registered class n for {lowered} (DECISION.md "
        "A5: the dataset's evaluation split cannot supply the registered n, so "
        "it registers its own achievable n; achieved power is restated at the "
        "realized n with the section 9.6 labeled caveat - a pre-declared "
        "branch, not a protocol deviation)"
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
    tp: int = 1,
) -> Tuple[Dict[str, str], int]:
    """Launcher budget env for one serving config, via cache_budget.plan_budget.

    Demand comes from the floor table row at this r (T2.4 is the ONE demand
    source); the primary knob of the resulting BudgetPlan maps onto the
    launcher env (frozen contract). The BYTE budget stays anchored to
    floor(r × D) for every config at the same r (the iso-bytes comparison
    anchor that makes B10's double-saving measurable), but token-denominated
    dials (SGLang --max-total-tokens) must convert bytes at the SERVED KV
    dtype — an fp8 server planned with bf16 arithmetic would realize only
    HALF the byte budget (§6.5 violation). ``tp`` > 1 plans the TP-sharded
    launch (plan_budget topology='tp': GQA shards → the primary knob carries
    the PER-RANK slice; MLA replicates, #20) — the TOTAL byte budget is
    still floor(r × D). Returns (env, budget_bytes_total).
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
            tp=tp,
            topology="single" if tp == 1 else "tp",
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
    grid: SessionGrid, engine: str, model: str, floor: FloorTable, role_tp: int
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """PD launcher env: the §6.5 split of floor(dist_budget_r × D) into the
    two REQUIRED per-role byte budgets, plus the role-tagged telemetry
    endpoints. ``role_tp`` is the TP degree BOTH role instances launch with
    (the frozen one-env launcher contract): the launcher passes each budget
    env VERBATIM as ``--kv-cache-memory-bytes`` on that instance, and the
    flag's registered convention is PER-RANK (the same convention the paired
    tp leg's _budget_env emits; live per-rank-vs-whole-pool semantics stays
    the plan's verify_live entry, closed by gate (j) at S0) — so a TP-sharded
    role's env carries the per-rank SLICE of its §6.5 pool, never the pool
    total. Returns (env, pd-record-for-the-plan)."""
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
    # W4.6 repair (2026-09-02 verifier major): handing a role's POOL TOTAL to
    # a role instance launched at TP=role_tp would — under the flag's
    # registered per-rank convention — realize ~role_tp× the §6.5 pools and
    # break the #18 pair's §6.6a iso-aggregate-bytes against the tp leg. The
    # env therefore carries the SAME per-rank shard rule the planner applies
    # under topology='tp', sourced from its own model table (GQA shards:
    # pool // tp, realized sum checked by gate (j); MLA replicates: per-rank
    # == pool, the replication cost IS registered contrast #20) — never
    # re-derived here from model arithmetic.
    sharded = role_tp >= 2 and not MODEL_KV[model].mla_tp_replicated
    if sharded:
        prefill_env = prefill_bytes // role_tp
        decode_env = decode_bytes // role_tp
    else:
        prefill_env, decode_env = prefill_bytes, decode_bytes
    env = {
        # The launcher REFUSES to start without both (manage_vllm_pd.sh).
        "CAGE_KV_BUDGET_BYTES_PREFILL": str(prefill_env),
        "CAGE_KV_BUDGET_BYTES_DECODE": str(decode_env),
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
    if role_tp >= 2:
        # The per-rank env basis + what gate (j) must observe as the realized
        # pool sum. Recorded ONLY when a TP degree shaped the env — (1, 1)
        # role plans stay byte-identical to pre-W4.6 ones (the same rule the
        # TP env itself follows in _relaunch_step).
        record["prefill_bytes_per_rank"] = prefill_env
        record["decode_bytes_per_rank"] = decode_env
        record["expected_bytes_total"] = (
            (prefill_env + decode_env) * role_tp
            if sharded
            else prefill_env + decode_env  # MLA: one latent copy is counted
        )
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
        # Per-role TP (W4.6): the frozen pd launcher applies ONE
        # CAGE_VLLM_TENSOR_PARALLEL to BOTH role instances (validated equal
        # at registration); degree 1 omits the env — the launcher omits the
        # flag, keeping (1,1) plans byte-identical to pre-W4.6 ones. The
        # degree also shapes the budget env (per-rank slices — see
        # _pd_budget_env), so it is derived BEFORE the env is built.
        role_tp = grid.dist_pd_role_gpus[0]
        env, pd_record = _pd_budget_env(grid, engine, model, floor, role_tp)
        if role_tp >= 2:
            env["CAGE_VLLM_TENSOR_PARALLEL"] = str(role_tp)
        # Backlog A10: the uniform request-length cap, applied by the pd
        # launcher to BOTH role instances (one env, frozen contract).
        env[MAX_MODEL_LEN_ENV] = str(grid.max_model_len)
        # Batch 2 W2: the proxy port the cells under this relaunch dial, and
        # the role ports the proxy's upstreams and the telemetry endpoints
        # above name (one table, exported, never left to the shell).
        env[PD_PROXY_PORT_ENV] = str(PD_PROXY_PORT)
        for role_env, role_port in PD_ROLE_PORT_ENVS.items():
            env[role_env] = str(role_port)
        return {
            "kind": "relaunch",
            "engine": engine,
            "model": model,
            "prefix_mode": "OFF" if prefix_off else "ON",
            "max_model_len": grid.max_model_len,
            # budget_r stays the CELL coordinate (None — DIST is the
            # topology overlay, not a pressure family); the pd byte budgets
            # ride the dedicated record + env.
            "budget_r": budget_r,
            "budget_bytes": pd_record["budget_bytes_total"],
            "kv_dtype": kv_dtype,
            "connector": connector,
            "topology": topology,
            "tp": role_tp,  # per ROLE instance (both roles, launcher contract)
            "pd": pd_record,
            "api_base": engine_api_base(engine, topology),  # W2: the proxy
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
    if topology == "tp":
        # W4.6: the DIST tp leg rides the SINGLE-INSTANCE launcher at the
        # registered dist_tp_size, serving floor(dist_budget_r × D) total
        # bytes — the SAME total the pd leg splits into roles, so the #18
        # pair is iso-aggregate-bytes (§6.6a) AND iso-GPU by registration.
        # Enumeration blocked these cells unless dist_tp_size was registered
        # and the engine has a TP env, so both lookups are driver invariants.
        tp_size = grid.dist_tp_size
        if tp_size is None:
            raise PlanError(
                "relaunch for the tp overlay with no registered dist_tp_size "
                "— enumeration should have BLOCKED these cells (driver "
                "invariant violated)"
            )
        env, budget_bytes = _budget_env(
            engine, model, grid.dist_budget_r, floor,
            served_kv_dtype=kv_dtype, tp=tp_size,
        )
        launched_tp = tp_size
    else:
        if budget_r is not None:
            env, budget_bytes = _budget_env(
                engine, model, budget_r, floor,
                served_kv_dtype=kv_dtype, tp=grid.serving_tp,
            )
        launched_tp = grid.serving_tp
    # T3.1 TP env (single-instance launchers): emitted for degrees >= 2 only
    # — degree 1 means the launcher omits the flag, so single-GPU relaunch
    # env stays byte-identical to pre-W4.6 plans. Budget-free relaunches
    # (F1 on a TP-sharded session) still need the degree: a 70B F1 server
    # launched without it would silently serve TP=1.
    if launched_tp >= 2:
        tp_env = TP_LAUNCH_ENV.get(engine)
        if tp_env is None:
            raise PlanError(
                f"relaunch for engine={engine} demands tensor-parallel degree "
                f"{launched_tp} but the launcher has no T3.1 TP env "
                f"(registered: {sorted(TP_LAUNCH_ENV)}) — register the env "
                "before planning this session"
            )
        env[tp_env] = str(launched_tp)
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
    # Backlog A10: the uniform request-length cap rides EVERY relaunch of
    # BOTH engines (vLLM --max-model-len, SGLang --context-length), budget-
    # free F1 relaunches included: a server launched without it would fall
    # back to the pilot shell default 4096 and refuse every RULER request.
    env[MAX_MODEL_LEN_ENV] = str(grid.max_model_len)
    # Batch 2 W2: the launcher port, from the ONE table the cells under this
    # relaunch pin --api-base from (the launcher's shell default never
    # reaches a campaign server; the tp overlay rides the same launcher).
    api_base = engine_api_base(engine, topology)  # refuses an unregistered engine
    env[PORT_LAUNCH_ENV[engine]] = str(ENGINE_PORTS[engine])
    return {
        "kind": "relaunch",
        "engine": engine,
        "model": model,
        "prefix_mode": "OFF" if prefix_off else "ON",
        "max_model_len": grid.max_model_len,
        "budget_r": budget_r,
        "budget_bytes": budget_bytes,
        "kv_dtype": kv_dtype,
        "connector": connector,
        "topology": topology,
        "tp": launched_tp,  # the T3.1 degree this serving stack launches with
        "pd": None,  # single/tp relaunch: no §6.5 role split
        "api_base": api_base,  # W2: what the cells under this relaunch dial
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
    if spec.corpus_budget_tokens is not None:
        # ADR-0106: the B12 rung is an identity coordinate (one row per rung).
        env["CAGE_CELL_CORPUS_BUDGET"] = str(spec.corpus_budget_tokens)
    return env


def _behavior_argv(spec: CellSpec, grid: SessionGrid, pins: RetrievalPins) -> List[str]:
    """The arm's behavior-bearing runner flags BEYOND the --baseline token.

    Mirrors the shell drivers' conventions exactly (the --baseline token
    selects only the base pipeline; without these flags ~7 of 12 baselines
    would serve byte-identically to their lever-free siblings — the verified
    T1.2 blocker):

    - corpus arms: --corpus-prefix-budget (run_prefix_envelope.sh cag_true_*;
      corpus-trunc gets its TRUNCATED rung budget PLUS the explicit
      --corpus-rung (ADR-0106): the runner's A4 guard requires a manifest
      carrying that rung and serves every query, labeling in/out-of-corpus).
    - retrieval arms: the pipeline + reranker EXPLICIT (never the runner's
      default), so B5 (dense, reranker OFF) vs B6 (ranked) stays the one
      pre-registered reranker ablation.
    - retr-comp: --context-source retrieved (run_compression.sh: without it
      the arm compresses GOLD context — the Phase-2 confound).
    - retr-trunc: --max-context-docs (rank truncation, "read less of what
      you found").
    - corpus-comp: --kv-cache-dtype fp8 is RECORD-ONLY provenance (the
      runner's own help text); the launch lever rides the relaunch env.
    - retrieval arms (backlog A5): --top-k RETRIEVAL_TOP_K, --embedding-model
      and --embedding-revision (the freeze-slot pins; the runner loads the
      encoder at that revision) and --ir-index-dir (the registered root), so
      no runner argparse default ever reaches a campaign cell.
    - retr-reuse (backlog F5a): --redis-key-prefix minted from the row key
      plus --flush-redis-namespace (a private, initially empty namespace).
    """
    extra: List[str] = []
    if spec.arm in CORPUS_BLOCK_ARMS:
        extra += ["--corpus-prefix-budget", str(grid.corpus_prefix_budget_tokens)]
    elif spec.arm == CORPUS_TRUNC_ARM:
        rung = spec.corpus_budget_tokens
        if rung is None or rung not in grid.corpus_trunc_budgets:
            raise PlanError(
                f"corpus-trunc cell {spec.to_row_key()} carries rung {rung!r}, "
                f"not one of the registered ladder {grid.corpus_trunc_budgets} "
                "(ADR-0106); enumeration should have minted one cell per "
                "rung (driver invariant violated)"
            )
        extra += ["--corpus-prefix-budget", str(rung), "--corpus-rung", str(rung)]
    if spec.retriever != "none":
        retrieval = RETRIEVER_ARGV.get(spec.retriever)
        if retrieval is None:
            raise PlanError(
                f"retriever {spec.retriever!r} has no registered runner argv "
                f"(registered: {sorted(RETRIEVER_ARGV)}) — refusing to guess a "
                "retrieval pipeline (fail closed)"
            )
        extra += list(retrieval)
        extra += [
            "--top-k", str(RETRIEVAL_TOP_K),
            "--embedding-model", pins.embedding_model,
            "--embedding-revision", pins.embedding_revision,
            "--ir-index-dir", grid.ir_index_root,
        ]
        if spec.arm in REDIS_CACHE_ARMS:
            extra += [
                "--redis-key-prefix", redis_key_prefix_for_row(spec.to_row_key()),
                "--flush-redis-namespace",
            ]
    if spec.arm == "retr-comp":
        extra += ["--context-source", "retrieved"]
    if spec.arm == "retr-trunc":
        extra += ["--max-context-docs", str(grid.retr_trunc_kept_docs)]
    kv_dtype = ARM_KV_DTYPE.get(spec.arm)
    if kv_dtype is not None:
        extra += ["--kv-cache-dtype", kv_dtype]
    return extra


def _cold_start_argv(spec: CellSpec) -> List[str]:
    """ADR-0102 cold-start-per-window runner flags for one cell.

    Server engines (SERVER_ENGINES) get ``--reset-cache-between-trials`` when
    RESET_CACHE_PER_WINDOW and ``--warmup-pool-queries WARMUP_POOL_QUERIES``
    when the registered W_warm is positive; the in-process hf oracle has no
    server cache to flush and gets neither. A window protocol, not arm
    behavior: identity still rides only the CAGE_CELL_* seam.
    """
    if spec.engine not in SERVER_ENGINES:
        return []
    extra: List[str] = []
    if RESET_CACHE_PER_WINDOW:
        extra.append("--reset-cache-between-trials")
    if WARMUP_POOL_QUERIES > 0:
        extra += ["--warmup-pool-queries", str(WARMUP_POOL_QUERIES)]
    return extra


def _argv_flag_value(argv: Sequence[str], flag: str) -> Optional[str]:
    """The value following ``flag`` in argv (None when absent or trailing)."""
    if flag not in argv:
        return None
    i = list(argv).index(flag)
    return argv[i + 1] if i + 1 < len(argv) else None


def _stale_plan_problems(
    step: Mapping[str, Any],
    spec: CellSpec,
    preceding_relaunch: Optional[Mapping[str, Any]],
    label: str,
    per_row_n: Optional[Mapping[str, Any]] = None,
    retrieval: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """load_plan's fail-closed per-cell check against EVERY stale plan shape
    today's ADRs fail-close against (v4; one clause per ADR). A stale plan
    that passes the schema literal but carries pre-ADR argv would run
    mislabeled duplicates, so 'run' refuses it and the operator re-plans:

    - ADR-0102: a server-engine cell carries --reset-cache-between-trials
      and --warmup-pool-queries == WARMUP_POOL_QUERIES (the VALUE, not just
      the flag).
    - ADR-0103: the cell's serving record and the relaunch it runs under
      agree with ``_prefix_off`` (the one prefix rule): prefix_mode and
      the launcher's --no-prefix-cache.
    - ADR-0104: a 'rerank' cell carries --rerank-pool RERANK_POOL; every
      other retriever carries no pool.
    - ADR-0106: a corpus-trunc cell carries --corpus-rung and
      --corpus-prefix-budget equal to its identity rung, and an EXECUTABLE
      one carries the --query-manifest that serves the rung.
    - A9 (per-row N): EVERY cell (server engine or hf) carries --num-queries
      equal to its ``num_queries`` record, and, given the plan header's
      ``per_row_n``, that record and ``row_class`` re-derive from the
      registered table (a primary cell relabeled to 800 is a mislabeled n).
    - A5 (retrieval pins), given the plan header's ``retrieval`` knobs
      (embedding_model, embedding_model_revision, ir_index_root): every
      retrieval cell carries --top-k RETRIEVAL_TOP_K, --embedding-model and
      --embedding-revision == the header's freeze pins, --ir-index-dir ==
      the registered root and the env CAGE_DISTRACTOR_DOCS ==
      DISTRACTOR_DOCS; a non-retrieval cell carries none of them.
    - F5a: a retr-reuse cell carries --redis-key-prefix ==
      redis_key_prefix_for_row(row_key) and --flush-redis-namespace; every
      other cell carries neither (a shared namespace is shared cache hits).
    - ADR-0055 (Batch 2 W1): EVERY cell env carries SKIP_QUALITY_ENV ==
      SKIP_QUALITY_VALUE (a pre-W1 plan, or one hand-edited to "0", would
      score inline inside the measured window).
    - Batch 2 W2 (engine endpoints): every non-hf cell carries --api-base ==
      engine_api_base(engine, topology) (a pre-W2 plan, or one whose SGLang
      cell was hand-pointed at the vLLM port, dials a server the plan never
      registered), and an EXECUTABLE server cell dials exactly the
      ``api_base`` its preceding relaunch records (review F1: a cell moved
      under another engine's boundary would be served by whatever survived
      an earlier boundary); an hf cell carries none, and a BLOCKED cell with
      no registered endpoint carries none.
    """
    row = step.get("row_key")
    argv: Sequence[str] = step.get("argv") or []
    env: Mapping[str, Any] = step.get("env") or {}
    problems: List[str] = []
    stale = ": stale plan, re-plan"

    got_api = _argv_flag_value(argv, "--api-base")
    if spec.engine == "hf":
        if got_api is not None:
            problems.append(
                f"{label}: hf cell {row!r} carries --api-base {got_api!r} (the "
                f"in-process oracle dials nothing; {ENGINE_PORTS_FINDING})" + stale
            )
    else:
        try:
            want_api = engine_api_base(spec.engine, spec.topology)
        except PlanError as exc:
            if step.get("blocked_on") is None:
                problems.append(f"{label}: cell {row!r}: {exc}" + stale)
            elif got_api is not None:
                problems.append(
                    f"{label}: blocked cell {row!r} carries --api-base {got_api!r} "
                    f"but its engine has no registered endpoint ({exc})" + stale
                )
        else:
            if got_api != want_api:
                problems.append(
                    f"{label}: cell {row!r} carries --api-base {got_api!r}, the "
                    f"registered endpoint of engine {spec.engine!r} is "
                    f"{want_api!r} ({ENGINE_PORTS_FINDING}: the runner's "
                    "--api-base default is the vLLM port and never reaches a "
                    "campaign cell)" + stale
                )

    got_skip = env.get(SKIP_QUALITY_ENV)
    if got_skip != SKIP_QUALITY_VALUE:
        problems.append(
            f"{label}: cell {row!r} env {SKIP_QUALITY_ENV} is {got_skip!r}, the "
            f"registered pin is {SKIP_QUALITY_VALUE!r} ({DECOUPLED_SCORING_ADR}: "
            "scoring is a separate post-serving pass; inline model scoring "
            "inside the measured window never rides a campaign cell)" + stale
        )

    got_n = _argv_flag_value(argv, "--num-queries")
    if got_n is None:
        problems.append(
            f"{label}: cell {row!r} lacks --num-queries (A9 per-row N: the "
            "runner's default query count must never reach a campaign cell)"
            + stale
        )
    elif got_n != str(step.get("num_queries")):
        problems.append(
            f"{label}: cell {row!r} carries --num-queries {got_n!r} but its "
            f"num_queries record is {step.get('num_queries')!r} (A9)" + stale
        )
    if per_row_n is not None:
        try:
            expected_cls = classify_row(
                spec,
                str(step.get("baseline")),
                str(step.get("dataset")),
                primary_engine=str(per_row_n["primary_engine"]),
                primary_baselines=tuple(per_row_n["primary_baselines"]),
            )
            expected_n = class_n(
                expected_cls,
                str(step.get("dataset")),
                n_primary=int(per_row_n["n_primary"]),
                n_secondary=int(per_row_n["n_secondary"]),
                n_identity=int(per_row_n["n_identity"]),
                window_requests=int(per_row_n["window_requests"]),
                achievable_n=dict(per_row_n.get("achievable_n") or {}),
            )
        except (KeyError, TypeError, ValueError, PlanError) as exc:
            problems.append(
                f"{label}: cell {row!r} row class cannot be re-derived from the "
                f"plan header per_row_n: {exc}" + stale
            )
        else:
            if step.get("row_class") != expected_cls:
                problems.append(
                    f"{label}: cell {row!r} row_class is {step.get('row_class')!r} "
                    f"but the A1 table says {expected_cls!r}" + stale
                )
            if step.get("num_queries") != expected_n:
                problems.append(
                    f"{label}: cell {row!r} num_queries is {step.get('num_queries')!r} "
                    f"but the registered {expected_cls} n is {expected_n} (A9)"
                    + stale
                )

    if spec.engine in SERVER_ENGINES:
        if RESET_CACHE_PER_WINDOW and "--reset-cache-between-trials" not in argv:
            problems.append(
                f"{label}: server-engine cell {row!r} lacks "
                "--reset-cache-between-trials (ADR-0102 cold start per window)"
                + stale
            )
        if WARMUP_POOL_QUERIES > 0:
            got = _argv_flag_value(argv, "--warmup-pool-queries")
            if got != str(WARMUP_POOL_QUERIES):
                problems.append(
                    f"{label}: server-engine cell {row!r} carries "
                    f"--warmup-pool-queries {got!r}, registered W_warm is "
                    f"{WARMUP_POOL_QUERIES} (ADR-0102 disjoint-pool warm-up)"
                    + stale
                )

    expected_mode = "OFF" if _prefix_off(spec) else "ON"
    serving = step.get("serving")
    if isinstance(serving, dict):
        if serving.get("prefix_mode") != expected_mode:
            problems.append(
                f"{label}: cell {row!r} serving.prefix_mode is "
                f"{serving.get('prefix_mode')!r} but the one prefix rule "
                f"(_prefix_off, ADR-0103) says {expected_mode}" + stale
            )
        if step.get("blocked_on") is None:
            if preceding_relaunch is None:
                problems.append(
                    f"{label}: executable server cell {row!r} has no relaunch "
                    "step before it (ADR-0103: the serving config is a relaunch "
                    "boundary)" + stale
                )
            else:
                r_argv: Sequence[str] = preceding_relaunch.get("argv") or []
                if preceding_relaunch.get("prefix_mode") != expected_mode:
                    problems.append(
                        f"{label}: cell {row!r} runs under a relaunch with "
                        f"prefix_mode {preceding_relaunch.get('prefix_mode')!r}, "
                        f"the one prefix rule says {expected_mode} (ADR-0103)"
                        + stale
                    )
                if ("--no-prefix-cache" in r_argv) != (expected_mode == "OFF"):
                    problems.append(
                        f"{label}: cell {row!r} needs prefix {expected_mode} but "
                        "its relaunch argv "
                        + ("carries" if "--no-prefix-cache" in r_argv else "lacks")
                        + " --no-prefix-cache (ADR-0103)" + stale
                    )
                # Batch 2 W2 (review F1): the endpoint the cell dials is the
                # one the relaunch it runs under serves on.
                if got_api != preceding_relaunch.get("api_base"):
                    problems.append(
                        f"{label}: cell {row!r} dials {got_api!r} but runs under a "
                        f"relaunch serving {preceding_relaunch.get('api_base')!r} "
                        f"({ENGINE_PORTS_FINDING}: a cell served by a boundary it "
                        "does not dial is mislabeled data)" + stale
                    )

    if spec.retriever == "rerank":
        pool = _argv_flag_value(argv, "--rerank-pool")
        if pool != str(RERANK_POOL):
            problems.append(
                f"{label}: ranked cell {row!r} carries --rerank-pool {pool!r}, "
                f"registered pool is {RERANK_POOL} (ADR-0104: without it the "
                "legacy rerank-exactly-top-k pipeline runs under the pooled "
                "row key)" + stale
            )
    elif "--rerank-pool" in argv:
        problems.append(
            f"{label}: cell {row!r} (retriever {spec.retriever!r}) carries "
            "--rerank-pool (ADR-0104: the pool rides ranked cells only)" + stale
        )

    if retrieval is not None:
        if spec.retriever != "none":
            expected_flags = {
                "--top-k": str(RETRIEVAL_TOP_K),
                "--embedding-model": str(retrieval.get("embedding_model")),
                "--embedding-revision": str(retrieval.get("embedding_model_revision")),
                "--ir-index-dir": str(retrieval.get("ir_index_root")),
            }
            for flag, want in expected_flags.items():
                got = _argv_flag_value(argv, flag)
                if got != want:
                    problems.append(
                        f"{label}: retrieval cell {row!r} carries {flag} {got!r}, "
                        f"the registered pin is {want!r} (backlog A5: runner "
                        "defaults never reach a campaign cell)" + stale
                    )
            got_env = env.get("CAGE_DISTRACTOR_DOCS")
            if got_env != str(DISTRACTOR_DOCS):
                problems.append(
                    f"{label}: retrieval cell {row!r} env CAGE_DISTRACTOR_DOCS is "
                    f"{got_env!r}, the registered pin is {DISTRACTOR_DOCS} "
                    "(backlog A5)" + stale
                )
        else:
            for flag in ("--top-k", "--embedding-model", "--embedding-revision", "--ir-index-dir"):
                if flag in argv:
                    problems.append(
                        f"{label}: cell {row!r} (retriever 'none') carries {flag} "
                        "(backlog A5: retrieval pins ride retrieval cells only)"
                        + stale
                    )
            if "CAGE_DISTRACTOR_DOCS" in env:
                problems.append(
                    f"{label}: cell {row!r} (retriever 'none') carries "
                    "CAGE_DISTRACTOR_DOCS (backlog A5)" + stale
                )
        if spec.arm in REDIS_CACHE_ARMS:
            want_prefix = redis_key_prefix_for_row(str(row))
            got_prefix = _argv_flag_value(argv, "--redis-key-prefix")
            if got_prefix != want_prefix:
                problems.append(
                    f"{label}: retr-reuse cell {row!r} carries --redis-key-prefix "
                    f"{got_prefix!r}, its own namespace is {want_prefix!r} "
                    "(backlog F5a: no two cells share cache entries)" + stale
                )
            if "--flush-redis-namespace" not in argv:
                problems.append(
                    f"{label}: retr-reuse cell {row!r} lacks "
                    "--flush-redis-namespace (backlog F5a: every cell starts "
                    "with an empty namespace)" + stale
                )
        else:
            for flag in ("--redis-key-prefix", "--flush-redis-namespace"):
                if flag in argv:
                    problems.append(
                        f"{label}: cell {row!r} (arm {spec.arm!r}) carries {flag} "
                        "(backlog F5a: Redis namespaces ride retr-reuse cells only)"
                        + stale
                    )

    if spec.arm == CORPUS_TRUNC_ARM:
        rung = str(spec.corpus_budget_tokens)
        for flag in ("--corpus-rung", "--corpus-prefix-budget"):
            got = _argv_flag_value(argv, flag)
            if got != rung:
                problems.append(
                    f"{label}: corpus-trunc cell {row!r} carries {flag} {got!r}, "
                    f"its identity rung is {rung} ({CORPUS_TRUNC_ADR})" + stale
                )
        if step.get("blocked_on") is None and "--query-manifest" not in argv:
            problems.append(
                f"{label}: executable corpus-trunc cell {row!r} lacks "
                f"--query-manifest ({CORPUS_TRUNC_ADR}: a rung serves only from "
                "a manifest carrying the ladder)" + stale
            )
    return problems


def _cell_step(
    cell: PlannedCell,
    grid: SessionGrid,
    floor: FloorTable,
    runner_cmd: Sequence[str],
    seed: int,
    window_duration_s: float,
    pins: RetrievalPins,
    query_manifest: Optional[str] = None,
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
    if spec.engine != "hf":
        # Batch 2 W2: the endpoint from the ONE port table the relaunch's
        # launcher env derives from (never the runner's --api-base default,
        # which is the vLLM port); the in-process oracle dials nothing. A
        # BLOCKED cell on an engine/topology with no registered endpoint
        # (today: a pd cell off vLLM) carries none, the gpu_count rule: the
        # debt stays visible in the plan, never guessed; an executable one
        # refuses.
        try:
            api_base: Optional[str] = engine_api_base(spec.engine, spec.topology)
        except PlanError:
            if cell.blocked_on is None:
                raise
            api_base = None
        if api_base is not None:
            argv += ["--api-base", api_base]
    # A9 per-row N: EVERY cell carries its registered --num-queries (the
    # runner's default query count must never reach a campaign cell); with a
    # manifest the runner measures the FIRST n ids of each trial.
    cls, num_queries = cell_num_queries(grid, cell)
    argv += ["--num-queries", str(num_queries)]
    argv += _behavior_argv(spec, grid, pins)
    argv += _cold_start_argv(spec)
    if query_manifest is not None:
        # The uniform yardstick (build_query_manifest.py): every cell of a
        # dataset with a registered manifest measures the manifest's
        # pre-drawn ids (the runner's --query-manifest sets
        # CAGE_QUERY_MANIFEST); a B12 rung is served from its ladder.
        argv += ["--query-manifest", query_manifest]
    if cell.ruler_task is not None:
        # D5#5 RULER instrument step: the task literal + SHAPE-32K, EXPLICIT
        # on every step (the loader's 4096-token default and niah_single
        # fallback are pilot conveniences, never a registered grid cell).
        argv += [
            "--ruler-task",
            cell.ruler_task,
            "--ruler-context-tokens",
            str(RULER_CONTEXT_TOKENS),
            "--max-tokens",
            str(RULER_OUTPUT_TOKENS),
        ]
    # W4.2: the GPU count this cell's serving stack launches with — an
    # EXECUTABLE cell without one would strand §6.6b downstream, so it
    # refuses here; a blocked cell carries the underivable count as an
    # explicit null (the debt stays visible in the plan, never guessed).
    gpu_count = _cell_gpu_count(grid, spec)
    if gpu_count is None and cell.blocked_on is None:
        raise PlanError(
            f"cell {spec.to_row_key()} is executable but its gpu_count is "
            "underivable (topology/count registration gap) — driver invariant "
            "violated"
        )
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
    env = _cell_identity_env(spec)
    if spec.retriever != "none":
        # Backlog A5: the Decision 3B distractor pool size, BEHAVIOR (not
        # identity: derive_cell_spec ignores it) pinned so the runner's env
        # default never reaches a campaign cell.
        env["CAGE_DISTRACTOR_DOCS"] = str(DISTRACTOR_DOCS)
    if gpu_count is not None:
        # NOT identity (derive_cell_spec ignores it): the serving-stack fact
        # the campaign writer persists into cell.json (W4.2 → §6.6b / #18).
        env["CAGE_GPU_COUNT"] = str(gpu_count)
    if cell.window_ordinal_base:
        # Per-task RULER steps share a row key; each task's runner invocation
        # emits windows (base, base+replications] so per-task resume can
        # never collide (campaign_session reads this env).
        env["CAGE_WINDOW_ORDINAL_BASE"] = str(cell.window_ordinal_base)
    # ADR-0055 (Batch 2 W1): decoupled scoring on EVERY cell, hf oracle and
    # blocked cells included. The runner's default is inline model scoring,
    # and _exec applies this env on top of the operator's shell, so the pin
    # wins over an exported inline switch. Behavior, never identity.
    env[SKIP_QUALITY_ENV] = SKIP_QUALITY_VALUE
    return {
        "kind": "cell",
        "family": spec.family,
        "baseline": cell.baseline_id,
        "dataset": cell.dataset,
        "row_key": spec.to_row_key(),
        "cellspec": spec.to_flat_dict(),
        "windows": grid.replications,
        # A9: the DECISION.md A1 row class and the n this cell measures per
        # window (--num-queries above); both re-derived by load_plan.
        "row_class": cls,
        "num_queries": num_queries,
        "gpu_count": gpu_count,
        # F2 grid-membership marker (§6.4): null outside F2 — absence stays
        # absence; F1/F3/DIST cells sit on no registered budget×rate grid.
        "grids": None if cell.grids is None else list(cell.grids),
        "ruler_task": cell.ruler_task,
        "window_ordinal_base": cell.window_ordinal_base,
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
        "env": env,
        "blocked_on": cell.blocked_on,
        # T3.2: executable pd cells are gated behind the operator's explicit
        # --allow-pd consent (blocked_on stays null — the launcher exists,
        # but the data path is unverified until the PD preflight smoke).
        "gate": PD_GATE if spec.topology == "pd" else None,
    }


def _register_query_manifests(
    grid: SessionGrid, query_manifests: Optional[Mapping[str, Path]]
) -> Dict[str, Dict[str, Any]]:
    """Validate the operator's per-dataset manifest registration (ADR-0106
    repair): each file must exist and parse, be for its dataset, carry the
    grid's corpus_prefix_budget_tokens as block_budget (B3/B4/B10 serve the
    full block through the same manifest) and every registered rung. Returns
    the plan-header records (absolute path, sha256, validated fields)."""
    out: Dict[str, Dict[str, Any]] = {}
    if not query_manifests:
        return out
    known = _grid_datasets(grid)
    for dataset, raw_path in query_manifests.items():
        if dataset not in known:
            raise PlanError(
                f"--query-manifest {dataset}=...: {dataset!r} is not a dataset "
                f"of session {grid.session!r} (registered: {sorted(known)})"
            )
        path = Path(raw_path)
        if not path.is_file():
            raise PlanError(f"--query-manifest {dataset}: manifest not found: {path}")
        raw = path.read_bytes()
        try:
            manifest = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PlanError(
                f"--query-manifest {dataset}: {path} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(manifest, dict):
            raise PlanError(f"--query-manifest {dataset}: {path} is not a JSON object")
        if manifest.get("dataset") != dataset:
            raise PlanError(
                f"--query-manifest {dataset}: {path} is for dataset "
                f"{manifest.get('dataset')!r}; refusing a mismatched yardstick"
            )
        block_budget = manifest.get("block_budget")
        if block_budget != grid.corpus_prefix_budget_tokens:
            raise PlanError(
                f"--query-manifest {dataset}: {path} has block_budget "
                f"{block_budget!r} but the grid registers corpus_prefix_budget_tokens="
                f"{grid.corpus_prefix_budget_tokens} (the runner's A4 guard would "
                "refuse every corpus cell); rebuild the manifest at the grid's budget"
            )
        for rung in grid.corpus_trunc_budgets:
            try:
                trunc_rung_for(manifest, rung)
            except ManifestError as exc:
                raise PlanError(
                    f"--query-manifest {dataset}: {path} lacks the registered "
                    f"B12 rung {rung} ({CORPUS_TRUNC_ADR}): {exc}; rebuild with "
                    f"build_query_manifest.py --trunc-budgets "
                    f"{','.join(str(r) for r in grid.corpus_trunc_budgets)}"
                ) from exc
        # A9: the per-trial id counts (the runner selects the first n ids of
        # each trial, so coverage is checked per trial, never on the pool).
        trials = manifest.get("trials")
        if not isinstance(trials, dict) or not trials:
            raise PlanError(
                f"--query-manifest {dataset}: {path} has no trials{{}} object "
                "(every window reads its trial's pre-drawn ids by number)"
            )
        trial_sizes: Dict[str, int] = {}
        for trial_key, ids in trials.items():
            if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
                raise PlanError(
                    f"--query-manifest {dataset}: {path} trial {trial_key!r} is not "
                    "a list of id strings; refusing to read it as empty"
                )
            trial_sizes[str(trial_key)] = len(ids)
        out[dataset] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "block_budget": int(block_budget),
            "trunc_rungs": list(grid.corpus_trunc_budgets),
            "trial_sizes": trial_sizes,
        }
    return out


def _check_manifest_coverage(
    grid: SessionGrid,
    manifests: Mapping[str, Mapping[str, Any]],
    cells: Sequence[PlannedCell],
) -> None:
    """A9 fail-closed coverage: every registered manifest must carry, for
    each of the grid's ``replications`` trials, at least as many ids as the
    most demanding cell of its dataset registers as --num-queries (the
    runner measures the first n ids of trial t and refuses a shorter trial,
    so a shortfall is caught HERE, before any GPU spends). ``achievable_n``
    is the ONLY way to lower the requirement (A5)."""
    demand: Dict[str, Tuple[int, str, str]] = {}  # dataset -> (n, class, row_key)
    for cell in cells:
        cls, n = cell_num_queries(grid, cell)
        current = demand.get(cell.dataset)
        if current is None or n > current[0]:
            demand[cell.dataset] = (n, cls, cell.spec.to_row_key())
    for dataset, rec in manifests.items():
        need = demand.get(dataset)
        if need is None:
            continue  # a manifest for a dataset without cells is not a shortfall
        n, cls, row_key = need
        sizes: Mapping[str, int] = rec["trial_sizes"]
        for trial in range(1, grid.replications + 1):
            have = sizes.get(str(trial))
            if have is None:
                raise PlanError(
                    f"--query-manifest {dataset}: {rec['path']} has no trial {trial} "
                    f"but the grid registers {grid.replications} windows per cell "
                    f"(one manifest trial each); the {cls} cells of {dataset} "
                    f"(e.g. {row_key}) register n={n}. Rebuild with "
                    f"build_query_manifest.py --num-queries {n} --num-trials "
                    f"{grid.replications}"
                )
            if have < n:
                raise PlanError(
                    f"--query-manifest {dataset}: {rec['path']} trial {trial} carries "
                    f"{have} ids but the {cls} cells of {dataset} (e.g. {row_key}) "
                    f"register n={n}: shortfall {n - have} (A9 per-row N; the runner "
                    "measures the first n ids of each trial and would refuse). "
                    f"Rebuild with build_query_manifest.py --num-queries {n} "
                    f"--num-trials {grid.replications}, or register "
                    f"SessionGrid.achievable_n[{dataset!r}] (DECISION.md A5) if the "
                    "split cannot supply it"
                )


def build_plan(
    session: str,
    floor: FloorTable,
    *,
    window_duration_s: float,
    seed: int = 42,
    runner_cmd: Sequence[str] = DEFAULT_RUNNER_CMD,
    launcher_cmds: Optional[Mapping[str, Sequence[str]]] = None,
    query_manifests: Optional[Mapping[str, Path]] = None,
    freeze_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """PURE plan builder: registered grid -> ordered step list + counts.

    Raises :class:`PlanError` on every dishonest input (unknown/unregistered
    session, floor table not matching the session model or missing r rows,
    non-positive window duration, a query manifest that does not exist, is
    for another dataset, or lacks the registered corpus budget / a rung).
    Reads only the registered manifest files (their sha256 is recorded).

    ``query_manifests`` maps dataset -> manifest path (``plan
    --query-manifest <dataset>=<path>``): every cell of that dataset carries
    ``--query-manifest``; a B12 rung cell of a dataset WITHOUT one is
    blocked_on the missing registration (see TRUNC_MANIFEST_BLOCKED_ON_FMT).

    ``freeze_file`` is the registration artifact the A5 retrieval pins are
    read from (``plan --freeze-file``; None = $CAGE_FREEZE_RESOLUTIONS, else
    DEFAULT_FREEZE_FILE). Reads only that artifact and the manifests.
    """
    grid = get_session_grid(session)
    pins = resolve_retrieval_pins(freeze_file)
    manifests = _register_query_manifests(grid, query_manifests)
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
    # ADR-0106 repair: a rung cell with no manifest for its dataset is
    # blocked (never a silent fallback); an already-blocked cell keeps its
    # first (launch-side) reason.
    cells = [
        replace(c, blocked_on=trunc_manifest_blocked_on(c.dataset))
        if c.spec.arm == CORPUS_TRUNC_ARM
        and c.dataset not in manifests
        and c.blocked_on is None
        else c
        for c in cells
    ]

    # Pre-resolve every budget row so a floor-table gap refuses BEFORE any
    # step is emitted (all problems at plan time, none at 3 a.m. on the pod).
    for r in sorted({c.spec.budget_r for c in cells if c.spec.budget_r is not None}):
        floor.row(r)
    # A9: every registered manifest must supply each cell's n per trial.
    _check_manifest_coverage(grid, manifests, cells)

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
        manifest_rec = manifests.get(cell.dataset)
        steps.append(
            _cell_step(
                cell,
                grid,
                floor,
                runner_cmd,
                seed,
                window_duration_s,
                pins,
                query_manifest=None if manifest_rec is None else manifest_rec["path"],
            )
        )
    for i, step in enumerate(steps):
        step["index"] = i

    cell_steps = [s for s in steps if s["kind"] == "cell"]
    # F5a invariant: distinct row keys mint distinct namespaces (a short-sha
    # collision would silently share cache entries between two cells).
    namespace_owner: Dict[str, str] = {}
    for s in cell_steps:
        prefix = _argv_flag_value(s["argv"], "--redis-key-prefix")
        if prefix is None:
            continue
        owner = namespace_owner.setdefault(prefix, s["row_key"])
        if owner != s["row_key"]:
            raise PlanError(
                f"Redis namespace {prefix!r} is minted by two cells ({owner!r} and "
                f"{s['row_key']!r}): short-sha collision, widen "
                "REDIS_NAMESPACE_SHA_CHARS (backlog F5a)"
            )
    by_family: Dict[str, Dict[str, int]] = {}
    for s in cell_steps:
        fam = by_family.setdefault(s["family"], {"cells": 0, "windows": 0})
        fam["cells"] += 1
        fam["windows"] += s["windows"]
    blocked = [s["row_key"] for s in cell_steps if s["blocked_on"]]
    cells_by_class: Dict[str, int] = {}
    for s in cell_steps:
        cells_by_class[s["row_class"]] = cells_by_class.get(s["row_class"], 0) + 1

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
            # ADR-0104: the ranked pipeline reranks a pool of RERANK_POOL
            # dense candidates and serves the runner's --top-k (3).
            "rerank_pool": RERANK_POOL,
            "rerank_pool_adr": "ADR-0104",
            "corpus_prefix_budget_tokens": grid.corpus_prefix_budget_tokens,
            # ADR-0106: the B12 ladder, the registered rungs, and the full
            # ladder as read (B3's own budget first, then each B12 rung).
            "corpus_trunc_budgets": list(grid.corpus_trunc_budgets),
            "corpus_trunc_ladder": [
                grid.corpus_prefix_budget_tokens, *grid.corpus_trunc_budgets
            ],
            "corpus_trunc_adr": CORPUS_TRUNC_ADR,
            "retr_trunc_kept_docs": grid.retr_trunc_kept_docs,
            # Backlog A5: the retrieval pins every retrieval cell carries
            # (argv --top-k / --embedding-model / --ir-index-dir and the env
            # CAGE_DISTRACTOR_DOCS); the embedding model is READ from the
            # ADR-0099 freeze slot, recorded here with its revision + source.
            "retrieval_top_k": RETRIEVAL_TOP_K,
            "distractor_docs": DISTRACTOR_DOCS,
            "embedding_model": pins.embedding_model,
            "embedding_model_revision": pins.embedding_revision,
            "embedding_model_freeze_slot": (
                f"INSTRUMENT_REVISIONS.{FREEZE_DENSE_RETRIEVER_SLOT}"
            ),
            "embedding_model_freeze_file": pins.freeze_file,
            "embedding_model_adr": DENSE_RETRIEVER_ADR,
            "ir_index_root": grid.ir_index_root,
            "retrieval_pins_backlog": "A5",
            # Backlog F5a: per-cell Redis namespaces on the cache-consulting
            # arms (minted from the row key, flushed at cell start).
            "redis_cache_arms": sorted(REDIS_CACHE_ARMS),
            "redis_namespace_rule": REDIS_NAMESPACE_RULE,
            "redis_namespace_backlog": "F5a",
            "lmcache_kv_transfer_config": LMCACHE_KV_TRANSFER_CONFIG,
            # ADR-0102 cold start per window (server-engine cells only).
            "reset_cache_per_window": RESET_CACHE_PER_WINDOW,
            "warmup_pool_queries": WARMUP_POOL_QUERIES,
            "cold_start_adr": "ADR-0102",
            # ADR-0103: the arms served prefix OFF by relaunch in every
            # family (corpus-fresh); F2 stays the prefix-OFF family.
            "prefix_off_arms": sorted(PREFIX_OFF_ARMS),
            "prefix_off_adr": "ADR-0103",
            # ADR-0055 (Batch 2 W1): every cell env pins decoupled scoring;
            # re-checked per cell by load_plan.
            "quality_scoring": "decoupled",
            "quality_scoring_env": SKIP_QUALITY_ENV,
            "quality_scoring_adr": DECOUPLED_SCORING_ADR,
        },
        # W4.2/W4.6: the registered serving shapes — reviewable in the header
        # like the behavior knobs (design registrations, not measurements).
        "serving_shapes": {
            "serving_tp": grid.serving_tp,
            "dist_tp_size": grid.dist_tp_size,
            "dist_pd_role_gpus": list(grid.dist_pd_role_gpus),
            # Backlog A10: the ONE request-length cap every relaunch of this
            # session carries (env MAX_MODEL_LEN_ENV, both engines);
            # re-checked per relaunch by load_plan.
            "max_model_len": grid.max_model_len,
            "max_model_len_env": MAX_MODEL_LEN_ENV,
            "max_model_len_backlog": "A10",
            # Batch 2 W2: the ONE port table every relaunch's launcher env
            # and every server cell's --api-base derive from (mirrors the
            # frozen launchers' defaults); re-checked per relaunch and per
            # cell by load_plan; the override envs 'run' refuses on presence.
            "engine_ports": dict(ENGINE_PORTS),
            "port_launch_env": dict(PORT_LAUNCH_ENV),
            "pd_proxy_port": PD_PROXY_PORT,
            "pd_proxy_port_env": PD_PROXY_PORT_ENV,
            "api_base_override_envs": list(API_BASE_OVERRIDE_ENVS),
            "engine_ports_finding": ENGINE_PORTS_FINDING,
        },
        # §6.4 anchor fine grid registration (null on non-anchor sessions).
        "fine_grid": (
            None
            if not grid.f2_fine_budgets
            else {
                "budget_levels": list(grid.f2_fine_budgets),
                "rate_fractions": list(grid.f2_fine_rates),
                "membership_labels": [GRID_D6_FACTORIAL, GRID_ANCHOR_FINE],
            }
        ),
        # D5#5 RULER pairing registration (null when the session carries none).
        "ruler_f2": (
            None
            if not grid.f2_ruler_baselines
            else {
                "baselines": list(grid.f2_ruler_baselines),
                "tasks": list(grid.f2_ruler_tasks),
                "context_tokens": RULER_CONTEXT_TOKENS,
                "output_tokens": RULER_OUTPUT_TOKENS,
            }
        ),
        # A9 / DECISION.md A1: the registered per-row N this plan's
        # --num-queries values are drawn from, the primary-predicate pin and
        # the A5 achievable-n override (with its caveat) - reviewable here
        # like the behavior knobs, and re-derived per cell by load_plan.
        "per_row_n": {
            "n_primary": grid.n_primary,
            "n_secondary": grid.n_secondary,
            "n_identity": grid.n_identity,
            "window_requests": grid.window_requests,
            "primary_engine": grid.primary_engine,
            "primary_baselines": list(grid.primary_baselines),
            "achievable_n": dict(grid.achievable_n),
            "achievable_n_caveat": _achievable_n_caveat(grid),
            "cells_by_class": cells_by_class,
            "decision": PER_ROW_N_DECISION,
        },
        # ADR-0106 repair: the per-dataset query manifests this plan serves
        # (path + sha256 + what was validated); {} when none is registered.
        "query_manifests": manifests,
        "counts": {
            "cells": len(cell_steps),
            "windows": sum(s["windows"] for s in cell_steps),
            "relaunches": relaunches,
            "blocked": len(blocked),
            "by_family": by_family,
        },
        "blocked_row_keys": blocked,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "charter_refs": ["6.1", "6.4", "6.8", "7.6", "7.6.1", "D5#5", "P6"],
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
    # v3 additions (see the PLAN_SCHEMA comment): the §6.6b producer, the
    # §6.4 membership marker, and the D5#5 per-task RULER pairing keys.
    "gpu_count",
    "grids",
    "ruler_task",
    "window_ordinal_base",
    # v5 additions (A9 per-row N): the A1 row class and the n measured.
    "row_class",
    "num_queries",
)
_RELAUNCH_STEP_KEYS = (
    "engine",
    "model",
    "prefix_mode",
    "budget_r",
    "kv_dtype",
    "connector",
    "topology",
    "tp",  # v3: the T3.1 degree the launcher is given (W4.6 serving shapes)
    "pd",
    "argv",
    "env",
    # A10: the uniform request-length cap this relaunch launches with
    # (also carried in env as MAX_MODEL_LEN_ENV; both re-checked below).
    "max_model_len",
    # Batch 2 W2: the endpoint this relaunch serves on; every executable
    # cell under it is checked against the record (review F1).
    "api_base",
)


def _stale_relaunch_problems(
    step: Mapping[str, Any], label: str, header_max_model_len: int
) -> List[str]:
    """load_plan's fail-closed per-relaunch check (backlog A10): the
    ``max_model_len`` record is a positive integer equal to the header's
    (ONE value per session, the uniform-regime rule) and the env carries
    MAX_MODEL_LEN_ENV with exactly that value (a relaunch without the env
    would launch at the pilot shell default 4096; a drifted env would make
    the record lie about the server every cell under it ran against).

    Batch 2 W2: the env carries the launcher port (PORT_LAUNCH_ENV, or
    PD_PROXY_PORT_ENV for pd) equal to the ONE registered port of the
    engine, the port the cells under this relaunch pin --api-base to (a
    relaunch without it would serve on the launcher's shell default; a
    drifted one would serve on a port no cell dials)."""
    problems: List[str] = []
    stale = ": stale plan, re-plan"
    env: Mapping[str, Any] = step.get("env") or {}
    engine = str(step.get("engine"))
    if step.get("topology") == "pd":
        port_env: Optional[str] = PD_PROXY_PORT_ENV
        want_port: Optional[int] = PD_PROXY_PORT
    else:
        port_env = PORT_LAUNCH_ENV.get(engine)
        want_port = ENGINE_PORTS.get(engine)
    if port_env is None or want_port is None:
        problems.append(
            f"{label}: relaunch engine {engine!r} has no registered port "
            f"(ENGINE_PORTS: {sorted(ENGINE_PORTS)}; {ENGINE_PORTS_FINDING})"
            + stale
        )
    else:
        if env.get(port_env) != str(want_port):
            problems.append(
                f"{label}: relaunch env {port_env} is {env.get(port_env)!r}, the "
                f"registered port is {want_port} ({ENGINE_PORTS_FINDING}: the cells "
                "under this relaunch pin --api-base to that port; the launcher's "
                "shell default never reaches a campaign server)" + stale
            )
        # Review F1: the record the cells under this relaunch are checked
        # against must itself be the registered endpoint.
        want_api = f"{API_BASE_HOST}:{want_port}"
        if step.get("api_base") != want_api:
            problems.append(
                f"{label}: relaunch api_base record is {step.get('api_base')!r}, "
                f"the registered endpoint is {want_api!r} ({ENGINE_PORTS_FINDING}: "
                "the cells under this relaunch are checked against the record)"
                + stale
            )
        if step.get("topology") == "pd":
            # Review F3: the role ports the recorded telemetry endpoints name.
            for role_env, role_port in PD_ROLE_PORT_ENVS.items():
                if env.get(role_env) != str(role_port):
                    problems.append(
                        f"{label}: pd relaunch env {role_env} is "
                        f"{env.get(role_env)!r}, the registered role port is "
                        f"{role_port} ({ENGINE_PORTS_FINDING}: the recorded "
                        "telemetry endpoints name that port)" + stale
                    )
    value = step.get("max_model_len")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        problems.append(
            f"{label}: relaunch max_model_len is {value!r}, must be an integer "
            ">= 1 (backlog A10)" + stale
        )
        return problems
    if value != header_max_model_len:
        problems.append(
            f"{label}: relaunch max_model_len {value} differs from the header's "
            f"{header_max_model_len}: a session serves ONE uniform "
            "max_model_len (backlog A10, _serving_config.sh uniform-regime "
            "rule)" + stale
        )
    got = env.get(MAX_MODEL_LEN_ENV)
    if got != str(value):
        problems.append(
            f"{label}: relaunch env {MAX_MODEL_LEN_ENV} is {got!r} but its "
            f"max_model_len record is {value} (backlog A10: the launcher reads "
            "the env; without it the pilot shell default 4096 would serve the "
            "cells under this relaunch)" + stale
        )
    return problems


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
    per_row_n = plan.get("per_row_n")
    if not isinstance(per_row_n, dict):
        raise RunError(
            f"plan {path} has no per_row_n header (A9: the registered per-row N "
            "every cell's --num-queries is re-derived from) - stale plan, re-plan"
        )
    knobs = plan.get("behavior_knobs")
    if not isinstance(knobs, dict):
        raise RunError(
            f"plan {path} has no behavior_knobs header (the registered behavior "
            "knobs every cell argv is checked against) - stale plan, re-plan"
        )
    for key in ("embedding_model", "embedding_model_revision", "ir_index_root"):
        value = knobs.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RunError(
                f"plan {path} behavior_knobs lacks {key} (backlog A5: the "
                "retrieval pins every retrieval cell's argv is re-checked "
                "against; a pre-A5 plan would run runner defaults under "
                "registered row keys) - stale plan, re-plan"
            )
    retrieval_knobs = {
        "embedding_model": knobs["embedding_model"],
        "embedding_model_revision": knobs["embedding_model_revision"],
        "ir_index_root": knobs["ir_index_root"],
    }
    shapes = plan.get("serving_shapes")
    header_max_model_len = (
        shapes.get("max_model_len") if isinstance(shapes, dict) else None
    )
    if (
        not isinstance(header_max_model_len, int)
        or isinstance(header_max_model_len, bool)
        or header_max_model_len < 1
    ):
        raise RunError(
            f"plan {path} serving_shapes lacks an integer max_model_len >= 1 "
            "(backlog A10: the uniform request-length cap every relaunch is "
            "re-checked against; a pre-A10 plan would launch at the pilot "
            "shell default 4096 and refuse every RULER request) - stale plan, "
            "re-plan"
        )
    preceding_relaunch: Optional[Dict[str, Any]] = None
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("kind") not in ("cell", "relaunch"):
            problems.append(f"steps[{i}]: kind must be 'cell' or 'relaunch'")
            continue
        if step["kind"] == "relaunch":
            preceding_relaunch = step
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
        if step["kind"] == "relaunch":
            problems.extend(
                _stale_relaunch_problems(step, f"steps[{i}]", header_max_model_len)
            )
        if step["kind"] == "cell":
            try:
                spec = CellSpec.from_flat_dict(step["cellspec"])
                minted = spec.to_row_key()
            except Exception as exc:  # cellspec is the one legality gate
                problems.append(f"steps[{i}]: cellspec is charter-illegal: {exc}")
                continue
            if minted != step["row_key"]:
                problems.append(
                    f"steps[{i}]: row_key {step['row_key']!r} != cellspec-minted "
                    f"{minted!r} — keys are minted by CellSpec, never hand-built"
                )
            problems.extend(
                _stale_plan_problems(
                    step,
                    spec,
                    preceding_relaunch,
                    f"steps[{i}]",
                    per_row_n=per_row_n,
                    retrieval=retrieval_knobs,
                )
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


def count_complete_windows(
    campaign_root: Path,
    row_key: str,
    dataset: str,
    *,
    ordinal_base: int = 0,
    expected: Optional[int] = None,
) -> int:
    """Complete windows already on disk for one cell: window dir (reader
    grammar) + its metrics.json completeness sentinel (the same sentinel the
    runner's own resume and the shell gates key on).

    With ``expected`` given, only ordinals in the step's claimed range
    ``(ordinal_base, ordinal_base + expected]`` count — the D5#5 per-task
    RULER steps share one (row_key, dataset) window space, so a flat count
    would let one task's complete windows mark ANOTHER task done (a silently
    skipped registered cell). ``expected=None`` keeps the legacy flat count.
    """
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
            if expected is not None:
                ordinal = int(match.group(2))
                if not (ordinal_base < ordinal <= ordinal_base + expected):
                    continue
            n += 1
    return n


def _sentinel_path(
    campaign_root: Path, row_key: str, dataset: str, ordinal_base: int = 0
) -> Path:
    # Dot-named DELIBERATELY: the §5 seal scope (campaign_layout.seal_run and
    # seal_campaign_run's journal cross-check) skips dot entries — a visible
    # sentinel under cells/ must never poison a later --seal-partial seal.
    # Dataset-suffixed DELIBERATELY: F1 row keys are shared by all four QA
    # datasets, so a bare per-cell sentinel could not say WHICH pass failed.
    # Ordinal-base-suffixed (per-task RULER steps only) for the same reason:
    # the tasks share one dataset, and a task-2 success must never clear a
    # task-1 failure record (base 0 keeps the pre-W4.4 spelling verbatim).
    name = f".STATUS-{dataset}"
    if ordinal_base:
        name += f"-from-{ordinal_base + 1:02d}"
    return Path(campaign_root) / "cells" / row_key / name


def _write_failed_sentinel(
    campaign_root: Path,
    row_key: str,
    dataset: str,
    reason: str,
    ordinal_base: int = 0,
) -> None:
    path = _sentinel_path(campaign_root, row_key, dataset, ordinal_base)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.write_text(
        f"STATUS=failed dataset={dataset} reason={reason} utc={stamp}\n",
        encoding="utf-8",
    )


def _clear_failed_sentinel(
    campaign_root: Path, row_key: str, dataset: str, ordinal_base: int = 0
) -> None:
    # A stale STATUS=failed over data that later completed (resume rerun,
    # --force-rerun) is a FALSE forensic record — remove it on success and on
    # a verified-complete skip.
    _sentinel_path(campaign_root, row_key, dataset, ordinal_base).unlink(
        missing_ok=True
    )


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
    # Backlog A6 / F6 (review 2026-09-17): _exec inherits the operator's shell
    # environment, so the pilot-archive escape hatch would reach every
    # retrieval cell and serve a pre-prefix index under a WARNING. Refused on
    # PRESENCE (any value) before the first step; campaign indices are
    # rebuilt with --rebuild-ir-index, never served stale.
    if STALE_INDEX_OPT_IN_ENV in os.environ:
        raise RunError(
            f"{STALE_INDEX_OPT_IN_ENV} is set in the environment "
            f"({os.environ[STALE_INDEX_OPT_IN_ENV]!r}); the campaign path never "
            "serves a stale (pre-prefix) dense index, unset it before 'run'"
        )
    # Batch 2 W2: the runner resolves CAGE_<ENGINE>_API_BASE BEFORE the
    # plan's --api-base pin (adapter and cache flush alike), so an exported
    # override would send every cell of that engine to an endpoint the plan
    # never registered. Refused on PRESENCE (any value) before the first step.
    for name in API_BASE_OVERRIDE_ENVS:
        if name in os.environ:
            raise RunError(
                f"{name} is set in the environment ({os.environ[name]!r}); the "
                "runner resolves it before the plan's --api-base pin, so a "
                "campaign cell could dial an endpoint the plan never registered "
                f"({ENGINE_PORTS_FINDING}); unset it before 'run'"
            )
    # Batch 2 W2 (review F5): the preflight gate dials the shell's launcher
    # port env while every relaunch exports the registered port (the step env
    # wins), so a shell value that DIFFERS would make the preflight evidence
    # come from a port the campaign never serves on. Refused on mismatch
    # before the first step; an equal value is fine.
    for name, want in SHELL_PORT_ENVS.items():
        got = os.environ.get(name)
        if got is not None and got.strip() != str(want):
            raise RunError(
                f"{name} is {got!r} in the environment but the registered port is "
                f"{want} ({ENGINE_PORTS_FINDING}): the preflight dials the shell "
                "value while every relaunch pins the registered one; unset it or "
                "set it to the registered port before 'run'"
            )
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
        # Per-task RULER steps claim disjoint window-ordinal ranges within a
        # shared (row_key, dataset) space — resume counting and the failure
        # sentinel are both scoped to THIS step's range (base 0 = legacy).
        base = int(step.get("window_ordinal_base") or 0)
        if step.get("blocked_on"):
            # Reaches here only under the operator's explicit --skip-blocked:
            # reported per-cell, executes nothing, gates a plain --seal below.
            outcomes.append(_Outcome(row_key, dataset, "skipped-blocked"))
            continue
        if step.get("serving") is not None and not server_ok:
            _write_failed_sentinel(
                campaign_root, row_key, dataset, "relaunch-failed", base
            )
            outcomes.append(_Outcome(row_key, dataset, "skipped-launch-failed"))
            continue
        done = count_complete_windows(
            campaign_root,
            row_key,
            dataset,
            ordinal_base=base,
            expected=int(step["windows"]),
        )
        if not force_rerun and done >= int(step["windows"]):
            _clear_failed_sentinel(campaign_root, row_key, dataset, base)
            outcomes.append(_Outcome(row_key, dataset, "skipped-complete"))
            continue
        argv = list(step["argv"]) + ["--campaign-root", str(campaign_root)]
        rc = _exec(argv, step["env"])
        if rc == 0:
            _clear_failed_sentinel(campaign_root, row_key, dataset, base)
            outcomes.append(_Outcome(row_key, dataset, "ok"))
        else:
            _write_failed_sentinel(
                campaign_root, row_key, dataset, f"runner-exit-{rc}", base
            )
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


def parse_query_manifest_args(items: Sequence[str]) -> Dict[str, Path]:
    """``DATASET=PATH`` registrations -> {dataset: path}; malformed or
    duplicate entries refuse (PlanError)."""
    out: Dict[str, Path] = {}
    for item in items:
        dataset, sep, raw_path = item.partition("=")
        if not sep or not dataset.strip() or not raw_path.strip():
            raise PlanError(
                f"--query-manifest {item!r}: expected DATASET=PATH"
            )
        dataset = dataset.strip()
        if dataset in out:
            raise PlanError(f"--query-manifest {dataset}: registered twice")
        out[dataset] = Path(raw_path.strip())
    return out


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
        query_manifests=parse_query_manifest_args(args.query_manifest),
        freeze_file=Path(args.freeze_file) if args.freeze_file else None,
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
    p_plan.add_argument(
        "--query-manifest",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="register the uniform query manifest (build_query_manifest.py) "
        "for one dataset; repeatable. Every cell of that dataset carries "
        "--query-manifest; a B12 rung cell of a dataset WITHOUT one is "
        "blocked_on the missing registration (ADR-0106)",
    )
    p_plan.add_argument(
        "--freeze-file",
        default=None,
        help="the frozen registration artifact the A5 retrieval pins are read "
        f"from (INSTRUMENT_REVISIONS.{FREEZE_DENSE_RETRIEVER_SLOT}, "
        f"{DENSE_RETRIEVER_ADR}); default ${FREEZE_FILE_ENV_VAR} else "
        f"{DEFAULT_FREEZE_FILE}; a missing artifact refuses",
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
