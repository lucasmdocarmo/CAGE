#!/usr/bin/env python3
"""
Order:     stage 3: THE campaign sweep driver (tranche P1, audit gap G1); 'plan' runs once the per-engine cal-v1 floor artifacts exist (calibrate_cell.py, before any cell), 'run' executes on the pod after 2_serving; ends by invoking seal_campaign_run.py
Objective: Enumerate ONE session's registered D6 grid into a reviewable execution-plan JSON ('plan', pure), then execute the plan cell-by-cell with engine-relaunch boundaries, per-window resume and fail-continue ('run')
Cloud:     both

The single command that runs a campaign session (charter §6.1/§6.8/§7.6.1).

PLAN / EXECUTE split (the design's load-bearing decision): ``plan`` is PURE —
it reads the P6 floor table and the per-engine cal-v1 floor artifacts,
enumerates the session's REGISTERED grid into an ordered list of steps, and
writes nothing except ``--out``. No subprocess, no GPU, no network. That
purity is what makes the driver testable offline and the plan JSON the
artifact an operator reviews BEFORE any campaign cell spends a cent (the
calibration artifacts need a live engine, so ``plan`` runs after
calibrate_cell.py on the pod, or locally on the pulled artifacts).
``run`` consumes a plan and shells the real scripts:

    python3 scripts/3_run/run_campaign.py plan --session a \\
        --floor-table results/preflight/floor_table_a.json \\
        --calibration vllm=results/calibration/vllm.json \\
        --calibration sglang=results/calibration/sglang.json \\
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
- F1 HF oracle: EXACTLY the reduced 8-cell set {B3 × all 4 datasets;
  B1, B6 × squad_v2 + qasper} on the in-process hf engine (ADR-0152,
  S0F-60: the oracle has no engine reuse, so a B2 oracle cell was B1 under
  another name on the 2026-10-08 landing; dropped from the registration).
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
- prefix OFF arms (ADR-0103, owner decision 2026-09-16; extended by
  ADR-0150, owner decision 2026-10-09, S0F-58): every arm the charter's
  7.1 table marks reuse "off" (gold-fresh B1, corpus-fresh B4, retr-fresh
  B5/B6, retr-comp B9, retr-trunc B11) is served with the engine prefix
  cache OFF through a PER-ARM RELAUNCH, uniformly on every engine, in EVERY
  family (PREFIX_OFF_ARMS, consulted only via ``_prefix_off``). Family
  carriage is unchanged (B4's REUSE bit still rides F3 beside B3). The
  runner's ``no_cache`` / ``rag`` tokens only label telemetry, so before
  ADR-0103 B4 and B3, and before ADR-0150 B1 and B2 as well as B6 and B7,
  were served by one prefix-ON server and were byte-identical serving twins
  under different names (proven on the 2026-10-08 landing: identical
  per-request cached-token sequences). Cost: one budget-free prefix-OFF
  relaunch per engine for F1, shared by every OFF arm; the F3 B4 cells share
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
- slo_floors and budget_plan producers (Batch 2 finding W4, owner decision
  2026-09-24, options 1A + 2A; ADR-0117): ``plan --calibration ENGINE=PATH``
  registers ONE cal-v1 floor artifact (calibrate_cell.py) per executable
  server engine, validated (procedure version, model, engine, the r = 1.5
  floor rung unless ``--calibration-budget-fraction`` registers another,
  30 median single-stream requests, confirmatory: false), recorded with its
  sha256 in the header ``calibration`` and pinned on EVERY cell step (hf
  oracle and blocked cells included) as the env CAGE_SLO_FLOORS_JSON; the
  campaign session writes the pin into manifest.json["slo_floors"] when it
  creates the manifest (the first emitting cell, the hf oracle on both
  registered sessions; amended never, compared on reopen). Every BUDGETED
  executable server cell (F2/F3 pressure coordinates, the DIST legs at
  dist_budget_r) additionally pins the cache_budget.BudgetPlan record of
  the relaunch it runs under (the same object the launcher env was derived
  from, carried on the relaunch step as ``budget_plan``) as the env
  CAGE_BUDGET_PLAN_JSON; the session threads it to CellWriter, which
  persists it under cell.json["budget_plan"] (the rho_own basis). F1, hf
  and blocked cells carry null and no env (absence stays absence).
  ``load_plan`` refuses a plan without the header, a cell whose floors pin
  differs from the header, a budgeted cell whose record differs from its
  relaunch's, or a budget-free cell carrying one; ``run`` refuses while
  either env is exported in the shell.
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
  smoke passes. S0F-22 Batch 1 (ADR-0133): the pd cell also carries the
  role telemetry pair (PD_TELEMETRY_FLAG + CAGE_TELEMETRY_ENDPOINTS), the
  proxy asks the prefill for its KV transfer ticket and relays it, and the
  runner's pd gate refuses any served row (no error) without an engine-shaped
  ticket; Batch 2 (ADR-0134) adds the per-window decode counter proof, which
  is why the decode telemetry endpoint is mandatory on a pd cell.

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

Telemetry (S0F-26, ADR-0136): every server-engine cell (SERVER_ENGINES,
blocked cells included) carries ``--vllm-telemetry`` (TELEMETRY_FLAG) exactly
once, and the in-process hf oracle never does (it serves no /metrics; a
sampler there would dial the runner's default port). ``load_plan`` refuses a
stale plan on either count. The runner refuses a campaign cell on vLLM or
SGLang without the flag, probes the endpoint for the KV usage gauge after the
warm-up and before the measured stage, and verify_results check (m) names a
window that still read UNKNOWN_TELEMETRY.

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

Dress rehearsal (2026-10-07, ADR-0144): ``plan --rehearsal-n N`` derives,
from the REGISTERED grid of ``--session``, the grid of a full-chain rehearsal
run: every baseline, engine, HF-oracle cell, RULER task and B12 rung kept;
the datasets restricted to those with a registered ``--query-manifest``; F2
collapsed to the lower-median (budget, rate) of its factorial plus the first
fine-only coordinate when the session registers a fine grid; F3 collapsed to
its lower medians; one window per cell; every row class at n = N; no
achievable_n override. ``rehearsal_grid`` is the one rule (never a hand list),
the plan header records it under ``rehearsal`` (null on a registered plan),
and the registered grids are never modified. The rehearsal exists so stages
6 to 14 of the experiment master run once, end to end, before a registered
session spends its hours; its numbers are DESIGN-INPUT-ONLY by construction.

Ordering minimizes engine relaunches: cells sort on (engine, prefix_mode,
model, budget_r, kv_dtype, connector, rate); every serving-config change is
an explicit ``relaunch`` step in the plan carrying the launcher argv + launch
env (budget, KV dtype, connector), so the relaunch count is exactly the
number of distinct EXECUTABLE serving configs (blocked cells launch nothing).

Engine handoff (Batch 1 finding V2, 2026-10-05): a relaunch step's verb
(``restart`` on the single-instance launchers, the self-cleaning ``start`` of
the pd launcher) stops only ITS OWN engine family, so before V2 the plan
order hf, sglang, vllm left the SGLang server resident on the GPU when the
first vLLM relaunch ran, and that relaunch failed after its readiness budget
(the session a dry trace skipped 430 of 870 cells this way); nothing stopped
the last engine when the run ended. Every relaunch step now records its
``launcher_key`` (the engine, or PD_LAUNCHER_KEY for the pd stack) and its
``stop_argv`` (``<launcher> stop``: every launcher of the fleet accepts the
bare verb and exits 0 with nothing running). ``stop_boundaries`` is the pure
rule, derived from the relaunch sequence and counted in the plan header
(``counts.engine_stops``, the number ``plan`` prints): the
previous family is stopped BEFORE a relaunch of a different launcher and the
last family is stopped when the run ends; same-launcher relaunches (prefix
ON then OFF) stop nothing in between. ``run`` additionally stops every
launcher the plan uses once before its first step (clean room: a resident
engine from an aborted run). A stop runs under the env of the relaunch it
closes; a failed stop is printed, counted, makes the exit code nonzero and
never gates the seal (the data tree is complete); a launcher the plan never
uses is never touched (the stop argv come from the plan, never from a
default table, so a stubbed test plan never runs a real launcher).

Window bound (Batch 1 finding V3, 2026-10-05): every window cell (row class
``window``: F2, F3 and the RULER instrument) is bounded by
``--arrival-count`` EQUAL to its own ``--num-queries`` (W requests issued,
one per prepared request), never by ``--duration-s``. Before V3 the driver
emitted the duration form and the runner's replay guard
(load_generator.ensure_no_measured_replay) refused every pressure window
whose rate x duration exceeded the W prepared requests unless
CAGE_ALLOW_REPLAY=1 labeled the run non-confirmatory. S0 proved the count
form live (``--arrival-count 50``, 6.5 s spans, 50 of 50 served).
``window_duration_s`` stays a REQUIRED ``plan`` input and header field as the
pre-costed duration ESTIMATE (the cost model), bounding nothing.

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
import statistics
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

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
    KV_DTYPE_FACTOR,
    MODEL_KV,
    BudgetPlan,
    CacheBudgetError,
    demand_bytes,
    plan_budget,
)
from src.orchestration.calibration import (  # noqa: E402
    FLOOR_N_REQUESTS,
    FLOOR_STATISTIC,
    PROBE_ATTAINMENT_MIN,
    PROBE_BISECT_STEPS,
    PROBE_LADDER_FACTOR,
    PROBE_MAX_STEPS,
    PROBE_WARMUP_S,
    PROBE_WINDOW_S,
    PROCEDURE_VERSION,
    START_QPS_RULE,
    CalibrationError,
    FloorMeasurement,
    ProbeStep,
    decide_lambda_star,
    floor_start_qps,
    geometric_rate_ladder,
)
from src.orchestration.campaign_session import ENGINE_OF_BACKEND  # noqa: E402
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
    "CalibrationFloor",
    "RungCalibration",
    "budgeted_rungs",
    "build_plan",
    "budget_plan_record",
    "calibrate_rungs",
    "cell_num_queries",
    "class_n",
    "classify_row",
    "demand_seq_tokens",
    "engine_api_base",
    "enumerate_cells",
    "load_calibration",
    "load_floor_table",
    "load_plan",
    "load_rung_calibration",
    "main",
    "interpolation_alpha",
    "rung_lambda_for_class",
    "smallest_pressure_class",
    "window_span_summary",
    "preflight_registration_check",
    "alpha_tolerance",
    "resolve_alpha",
    "pool_shortfalls",
    "mac_pool_shortfalls",
    "request_cap_tokens",
    "parse_calibration_args",
    "plan_launchers",
    "rehearsal_grid",
    "row_class",
    "slo_floors_env_value",
    "stop_boundaries",
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
# W4 (2026-09-24, Batch 2 finding W4, owner picked options 1A + 2A; ADR-0117):
# the header gained ``calibration`` (the per-engine cal-v1 floors + sha256),
# every cell env the pin CAGE_SLO_FLOORS_JSON, cell AND relaunch steps the
# REQUIRED key ``budget_plan`` (the cache_budget.BudgetPlan record; null on
# budget-free steps) and budgeted cell envs the pin CAGE_BUDGET_PLAN_JSON. No
# schema bump: a v5 plan built before W4 is refused by the header check, the
# required-key check and _stale_plan_problems (its manifest would carry no
# slo_floors and its cell.json no budget_plan, so contrast #14 would refuse
# and rho_own would skip on the whole tree), so the operator re-plans.
# S0F-26 (2026-10-02, ADR-0136): every server-engine cell argv gained
# --vllm-telemetry (only pd cells carried it since S0F-22 Batch 1). No schema
# bump: a v5 plan built before it is refused by _stale_plan_problems (its
# windows would read UNKNOWN_TELEMETRY), so the operator re-plans.
# V2 (2026-10-05, Batch 1 finding V2, engine handoff): relaunch steps gained
# the REQUIRED keys ``launcher_key`` and ``stop_argv``; the header counts
# gained ``engine_stops``. No schema bump: a v5 plan built before V2 is
# refused by the required-key check (its run would leave the previous engine
# family resident across a launcher change), so the operator re-plans.
# V3 (2026-10-05, Batch 1 finding V3, window bound): window cell argv carry
# --arrival-count == --num-queries and no --duration-s; the header
# behavior_knobs gained ``window_bound``. No schema bump: a v5 plan built
# before V3 is refused by _stale_plan_problems (its window cells would refuse
# at the runner's replay guard, or replay under CAGE_ALLOW_REPLAY=1), so the
# operator re-plans.
# v6 (2026-10-09, Batch B, ADR-0153/0154/0155, per the v2 precedent): cell
# steps gained the REQUIRED keys ``demand_class`` (the per-arm demand class
# the serving budget was sized on, S0F-63), ``lambda_star_rps``,
# ``lambda_star_source`` and ``lambda_kv_pred_rps`` (the offered-rate basis is
# the rung calibration measured on the cell's own workload, S0F-62; the floor
# table's KV-bound rate is recorded as the P6 prediction, never offered);
# relaunch steps gained ``demand_class``; the plan gained the step kind
# ``dry_window`` (one per engine and demand class, S0F-61) and the header
# ``rung_calibration`` and ``demand_classes``. A v5 plan carries the pending
# KV-bound rates the 2026-10-08 landing ran 3 to 6 times below capacity on,
# so 'run' refuses it and the operator re-plans with the rung artifacts.
# ADR-0156 (2026-10-09, Batch C): cell steps gained ``window_span_s_expected``
# and ``window_span_below_floor`` (S0F-68 audit), the rung artifact became v2
# (the smallest-class ladder) and the header gained ``window_spans``. No
# schema bump: a v6 plan built before it is refused by the required-key check
# and by _stale_plan_problems (its derived rates rest on the KV-bound ratio),
# so the operator re-plans with v2 artifacts.
PLAN_SCHEMA = "cage-campaign-plan-v6"
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
#: Batch 1 finding V1 (2026-10-05): the chat wrapper the rendered request adds
#: around the haystack (the system instruction, "Context 1:", "Question:", the
#: template's role tokens and the generation prompt), measured on 2026-10-05
#: with the Qwen/Qwen3-14B tokenizer and the vLLM adapter's pinned template
#: kwargs. At a 2,048-token haystack, 30 items per task: niah_multikey 85 to
#: 87, niah_multiquery 99 to 106, variable_tracking 92 to 96, qa 75 to 91. At
#: the real haystack (32,384), the 200 items of each of the three campaign
#: seeds (42, 43, 44) per task (independent review 2026-10-05): maxima 88,
#: 107, 97 and 99 (qa questions 5 to 31 tokens); system plus template alone
#: 63 with an empty user message, 73 with a minimal one. The allowance
#: covers the maximum (107) with 21 to spare; the worst rendered request
#: observed was 32,481 of the 32,512 cap. The runner renders every prompt
#: (measured and warm-up pool) and refuses the cell when one
#: exceeds the cap (--ruler-rendered-input-cap), so the allowance is a sizing
#: input, never a guarantee. The haystack target every RULER cell carries as
#: --ruler-context-tokens is the registered input shape minus the allowance;
#: before V1 the haystack target WAS the input shape, counted in words, and
#: every RULER request exceeded the server cap.
RULER_WRAPPER_ALLOWANCE: int = 128
RULER_HAYSTACK_TOKENS: int = RULER_CONTEXT_TOKENS - RULER_WRAPPER_ALLOWANCE
RULER_SIZING_FINDING: str = "Batch 1 V1"
assert 0 < RULER_WRAPPER_ALLOWANCE < RULER_CONTEXT_TOKENS

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

#: Batch 2 finding W4 (2026-09-24; owner picked options 1A + 2A of the Spec
#: Block; ADR-0117): the two producer pins. Before W4 nothing produced
#: manifest.json["slo_floors"] (contrast #14 refused on every tree) or
#: cell.json["budget_plan"] (the rho_own leg skipped on every window); the
#: floors existed only in the standalone cal-v1 artifact calibrate_cell.py
#: writes and the budget plan was computed per relaunch and dropped.
#: - SLO_FLOORS_ENV: the §6.1 single-stream floors of every registered engine
#:   as compact JSON ({engine: {ttft_s, tpot_s, n_requests, statistic,
#:   budget_fraction, source_sha256}}, seconds), pinned on EVERY cell step
#:   because the manifest is created by whichever cell emits first and is
#:   amended never; campaign_session writes it into the manifest under
#:   SLO_FLOORS_MANIFEST_KEY and compares on reopen.
#: - BUDGET_PLAN_ENV: the relaunch's cache_budget.BudgetPlan record (asdict,
#:   plus the floor table's sha256) on BUDGETED cells only; campaign_session
#:   threads it to CellWriter, which persists it under BUDGET_PLAN_CELL_KEY.
#: BEHAVIOR/provenance, never identity: derive_cell_spec ignores both. The
#: literals are mirrored in campaign_session (pinned equal by tests).
SLO_FLOORS_ENV: str = "CAGE_SLO_FLOORS_JSON"
BUDGET_PLAN_ENV: str = "CAGE_BUDGET_PLAN_JSON"
CELL_PIN_ENVS: Tuple[str, ...] = (SLO_FLOORS_ENV, BUDGET_PLAN_ENV, "CAGE_TELEMETRY_ENDPOINTS")
SLO_FLOORS_MANIFEST_KEY: str = "slo_floors"
BUDGET_PLAN_CELL_KEY: str = "budget_plan"
#: Charter §6.1: the floor is measured at the comfortable control rung
#: r = 1.5, concurrency 1 (goodput.SLOBaseline docstring); an artifact at
#: another rung refuses unless the operator registers that rung explicitly
#: with --calibration-budget-fraction (recorded in the header).
FLOOR_BUDGET_FRACTION: float = 1.5
SLO_FLOORS_FINDING: str = "Batch 2 W4"
SLO_FLOORS_ADR: str = "ADR-0117"

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
#:   per window; --num-queries IS the window pool size AND, since V3, the
#:   window's --arrival-count (one arrival per prepared request: the open-loop
#:   generator's replay guard refuses a schedule longer than the pool unless
#:   CAGE_ALLOW_REPLAY=1 labels the run non-confirmatory, so a duration-bound
#:   window at a real rate never ran).
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

#: ADR-0103 (owner decision 2026-09-16) extended by ADR-0150 (owner decision
#: 2026-10-09, S0F-58): the arms served with the engine prefix cache OFF in
#: EVERY family, through a per-arm RELAUNCH, uniformly on every engine: every
#: arm the charter's 7.1 table marks reuse "off". The runner tokens
#: ``no_cache`` and ``rag`` only LABEL telemetry (src/orchestration/
#: baselines.py), so absent this rule B1 and B2, B4 and B3, B6 and B7 would be
#: served by the same prefix-ON server and differ by label alone (the
#: mislabeled-duplicate failure class; the 2026-10-08 landing showed gold-fresh
#: and gold-reuse with identical per-request cached-token sequences). Family
#: carriage is unchanged (cellspec._ARMS_BY_FAMILY). The rule is consulted ONLY
#: through _prefix_off (sort key, serving-config identity, relaunch argv, cell
#: serving record), never re-derived by a step builder. retr-store (B8) keeps
#: the cache ON: its reuse is the external store, the charter's own column.
PREFIX_OFF_ARMS: FrozenSet[str] = frozenset(
    {"gold-fresh", "corpus-fresh", "retr-fresh", "retr-comp", "retr-trunc"}
)
PREFIX_OFF_ADRS: str = "ADR-0103, ADR-0150"

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

#: CAGE_TELEMETRY_ENDPOINTS value for pd relaunches AND pd cells (T4.1
#: role=url grammar, run_experiment.parse_telemetry_endpoints): one
#: role-tagged sampler per instance, so PD windows carry per-role serving
#: telemetry. S0F-22 Batch 1 (integration audit distributed-4): the pd CELL
#: step carries the pair too (the env plus PD_TELEMETRY_FLAG), because the
#: runner reads the env, refuses it without the flag, and samples nothing
#: without either; without them every pd window read UNKNOWN_TELEMETRY and
#: the runner had no address for the decode role's /metrics (the Batch 2
#: transfer proof). Pinned by load_plan per cell; a single-topology cell
#: carries neither (its sampler, when any, dials --api-base).
PD_TELEMETRY_ENDPOINTS_ENV = "CAGE_TELEMETRY_ENDPOINTS"
#: S0F-26 (ADR-0136): the runner's telemetry flag rides EVERY server-engine
#: cell (SERVER_ENGINES, blocked cells included: the ADR-0102 rule) exactly
#: once, and never an hf cell. Before S0F-26 only pd cells carried it, so
#: every other campaign window would have read UNKNOWN_TELEMETRY (S0 ran its
#: cells by hand, with the flag). The in-process oracle serves no /metrics: a
#: sampler there would dial the runner's --api-base default and record
#: whatever engine listens on it. Pinned per cell by load_plan. The runner
#: refuses a campaign cell on vLLM or SGLang without the flag and probes the
#: endpoint for the KV usage gauge before the measured stage.
TELEMETRY_FLAG = "--vllm-telemetry"
PD_TELEMETRY_FLAG = TELEMETRY_FLAG  # the S0F-22 Batch 1 name, one flag
TELEMETRY_FINDING = "S0F-26"
TELEMETRY_ADR = "ADR-0136"
PD_TELEMETRY_ENDPOINTS = (
    f"prefill=http://localhost:{PD_PREFILL_PORT},"
    f"decode=http://localhost:{PD_DECODE_PORT}"
)
PD_TELEMETRY_FINDING = "S0F-22 Batch 1 / integration audit distributed-4"

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

#: Batch 1 finding V2 (2026-10-05): the launcher verb that stops one family's
#: engine(s). Every launcher of the fleet dispatches it with no model
#: argument (manage_vllm_server.sh and manage_sglang_server.sh ``stop)`` ->
#: stop_server; manage_vllm_pd.sh ``stop)`` -> stop_stack) and every kill in
#: those bodies is guarded, so the verb exits 0 with nothing running. The
#: relaunch step records ``stop_argv`` = its launcher prefix + this verb; run
#: executes it at the boundaries stop_boundaries names.
LAUNCHER_STOP_VERB: str = "stop"
ENGINE_STOP_FINDING: str = "Batch 1 V2"
#: run's exit code when every cell passed but an engine stop failed (review
#: F4): distinct from 1 (a failed cell) so the operator reads "the data is
#: complete, the pod may still hold an engine" without opening the log.
EXIT_STOP_FAILED: int = 2
#: Batch 1 finding V3 (2026-10-05): how a window cell is bounded. The value
#: the header records; the per-cell rule is --arrival-count == --num-queries.
WINDOW_BOUND: str = "arrival-count"
WINDOW_BOUND_FINDING: str = "Batch 1 V3"

#: The floor table's own basis label for its KV-bound rate (build_floor_table
#: LAMBDA_STAR_BASIS). Since ADR-0154 that rate is RECORDED on every pressure
#: cell as the P6 prediction (``lambda_kv_pred_rps``) and never offered: the
#: 2026-10-08 landing offered it (3 to 6 times below the measured capacity on
#: a 4.0 s assumed service time) and no window reached the regime.
_LAMBDA_PENDING_BASIS = "kv-bound-only [pending calibration]"
#: ADR-0154 (owner GO 2026-10-09, S0F-62): the offered rate of a pressure cell
#: is rate_frac x lambda*(engine, r) MEASURED by ``calibrate-rungs`` on the
#: gold-fresh F2 workload of the session, under the relaunch the plan itself
#: emits for that rung (prefix OFF, the gold demand class's byte budget), with
#: the registered cal-v2 ladder rule. ADR-0156 (owner decision 2026-10-09,
#: "apply recommended fixes for best solution possible") adds the SECOND
#: anchor: the engine's smallest executable demand class gets its own ladder
#: under its own class budget at every rung, and every class between the two
#: anchors is interpolated on the log-log line through them (the KV-bound
#: ratio of Batch B was an assumption the landing's scheduler-bound evidence
#: contradicted at loose r). The three basis labels every pressure cell
#: carries name which of the three its lambda* is.
LAMBDA_BASIS_RUNG = "cal-v2 rung ESTIMATED (gold-fresh F2 workload, this engine, this r)"
LAMBDA_BASIS_SMALL = "cal-v2 rung ESTIMATED (the engine's smallest executable demand class, this r)"
#: lambda(s) = lambda_g x (s_g / s)^alpha with alpha = ln(lambda_m / lambda_g)
#: / ln(s_g / s_m) per (engine, r): alpha = 1 is the KV-bound limit (D linear
#: in tokens, Batch B's assumption), alpha = 0 a pure request cap. Never an
#: extrapolation: the anchors are the largest and the smallest class.
LAMBDA_BASIS_INTERPOLATED = (
    "interpolated: log-log between the gold-fresh and the smallest-class rung lambda* "
    "(alpha recorded; 1 = KV-bound) [D]; checked live by the dry window (ADR-0153)"
)
RUNG_CALIBRATION_SCHEMA = "cage-rung-calibration-v2"
RUNG_CALIBRATION_ADR = "ADR-0154"
RUNG_CALIBRATION_FINDING = "S0F-62"
TWO_ANCHOR_ADR = "ADR-0156"
#: The ladder's first rung on the loosest budget: the floor's single-stream
#: service rate (START_QPS_RULE) at this decode length. The campaign's QA
#: answers stop at the newline (17 to 29 output tokens on the landing's vLLM
#: rows; 256 is the runner's cap, not the served length), so the registered
#: rule at 256 tokens would start a 25 minute climb from 0.35 qps. 32 tokens
#: is the sizing input of the FIRST rung only; the climb finds lambda*.
LADDER_START_DECODE_TOKENS: int = 32
#: Tighter rungs start the climb below the looser rung's lambda* (lambda*
#: cannot rise as the budget shrinks): two ladder steps below it.
LADDER_CHAIN_DIVISOR: float = PROBE_LADDER_FACTOR ** 2
#: Calibration windows replay the measured pool past W arrivals when the
#: ladder rate x PROBE_WINDOW_S exceeds the pool; calibration data never
#: enters confirmatory analysis (calibration.py doctrine), so the runner's
#: labeled non-confirmatory switch rides every ladder step.
LADDER_REPLAY_ENV = "CAGE_ALLOW_REPLAY"

#: ADR-0155 (owner GO 2026-10-09, S0F-63): per-arm DEMAND CLASSES. Charter
#: §6.1 sizes D at the target concurrency "per §7.6.1 cell family"; the floor
#: table carries ONE shape (the gold passages), so at the same byte budget a
#: retrieval arm (about a quarter of the tokens) held four times the sequences
#: and never filled its pool (fig04 of the landing: KV usage under 0.25 on
#: three of F2's four arms). Each arm's demand is the floor table's demand
#: scaled by its served sequence tokens: D_class = floor(D_floor x s_class /
#: s_floor), the same arithmetic as cache_budget.demand_bytes (D is linear in
#: the sequence length), so the gold class stays byte-identical to the floor
#: table and every class holds the same number of sequences at a given r.
#: The values are MEASURED means of prompt plus output tokens over the Qasper
#: windows of the 2026-10-08 landing (experiments/S0/2026-10-08_1322, Qwen3-14B
#: tokenizer, vLLM output counts; the SGLang rows served 2 tokens of thinking
#: scaffolding, S0F-59). retr-store (B8) launched on no engine of that landing
#: and takes retr-fresh's shape (the same ranked retrieval, B8 = B6 plus the
#: store). corpus-trunc is per rung (the rung IS the served block).
DEMAND_CLASS_ADR = "ADR-0155"
DEMAND_CLASS_FINDING = "S0F-63"
DEMAND_SEQ_TOKENS_SOURCE = (
    "mean served sequence tokens (prompt + output) over the Qasper windows of "
    "experiments/S0/2026-10-08_1322 (Qwen/Qwen3-14B tokenizer, vLLM output "
    "counts); retr-store takes retr-fresh's shape [D]"
)
DEMAND_SEQ_TOKENS_2026_10_08: Dict[str, int] = {
    "gold-fresh": 4779,
    "gold-reuse": 4779,
    "corpus-fresh": 2336,
    "corpus-reuse": 2336,
    "corpus-comp": 2336,
    "retr-fresh": 1127,
    "retr-reuse": 1126,
    "retr-store": 1127,
    "retr-comp": 702,
    "retr-trunc": 348,
}
CORPUS_TRUNC_DEMAND_SEQ_TOKENS_2026_10_08: Dict[int, int] = {1400: 1205, 700: 480}
#: ADR-0158 (S0F-71 remedy (a), owner 2026-10-09 "proceed with recommendations"):
#: the request-length cap is PER DEMAND CLASS. A budgeted relaunch's KV pool is
#: floor(r x c x s_class) tokens (ADR-0155), and vLLM refuses to start when
#: that pool cannot hold ONE request of the cap (S0 2026-10-08 log); a uniform
#: 32,768 cap (A10) therefore refused every class below gold at the tight
#: rungs. The anchor class keeps the session cap (RULER SHAPE-32K runs on it);
#: every other class gets the smallest power of two at or above
#: REQUEST_CAP_MARGIN x its served MAXIMUM on the 2026-10-08 landing, never
#: above the session cap. Served maxima (prompt + output tokens, vLLM counts,
#: ok rows, 318 requests files, computed 2026-10-09 [V]; the anchor's figure
#: includes the RULER rows; the QA gold maximum was 18,237): the table below,
#: keyed by the class's served tokens (DEMAND_SEQ_TOKENS_2026_10_08 values).
#: Limitation [A]: arms of different classes inside one contrast serve under
#: different caps (B11 at 4,096 vs B6 at 8,192); the cap bounds request length
#: only, the byte budget is untouched, and every served request sits at or
#: below half its cap.
DEMAND_CLASS_MAX_SERVED_TOKENS_2026_10_08: Dict[int, int] = {
    4779: 32574,  # gold-fresh (RULER rows included) and gold-reuse
    2336: 3211,   # corpus-fresh, corpus-reuse, corpus-comp
    1205: 1551,   # corpus-trunc, 1400-token rung
    1127: 2607,   # retr-fresh; retr-store takes the same shape [D]
    1126: 2607,   # retr-reuse
    702: 1607,    # retr-comp
    480: 1551,    # corpus-trunc, 700-token rung
    348: 1068,    # retr-trunc
}
REQUEST_CAP_MARGIN: float = 2.0
REQUEST_CAP_ADR = "ADR-0158"
REQUEST_CAP_RULE = (
    "per demand class: the anchor class serves the session max_model_len (RULER); every "
    f"other class the smallest power of two >= {REQUEST_CAP_MARGIN:g} x its served maximum "
    "on the 2026-10-08 landing, never above the session cap; the cap bounds request "
    f"length only and the pool must hold one request of it ({REQUEST_CAP_ADR})"
)


def request_cap_tokens(grid: "SessionGrid", seq_tokens: Optional[int]) -> int:
    """The request-length cap (VLLM_MAX_MODEL_LEN: vLLM --max-model-len, SGLang
    --context-length) a relaunch of demand class ``seq_tokens`` launches with
    (ADR-0158). None (a budget-free relaunch) and the anchor class take the
    session cap; a class absent from the maxima table refuses (PlanError)."""
    session_cap = int(grid.max_model_len)
    if seq_tokens is None or int(seq_tokens) == int(grid.demand_seq_tokens[DEMAND_ANCHOR_ARM]):
        return session_cap
    served_max = grid.demand_class_max_served_tokens.get(int(seq_tokens))
    if isinstance(served_max, bool) or not isinstance(served_max, int) or served_max < 1:
        raise PlanError(
            f"demand class {seq_tokens} tokens has no served maximum registered on the grid "
            f"(demand_class_max_served_tokens; {REQUEST_CAP_ADR}): register it from the "
            "landing's rows before planning this class"
        )
    cap = 1
    while cap < REQUEST_CAP_MARGIN * served_max:
        cap *= 2
    return min(session_cap, cap)
#: The anchor arm of the floor table: its registered shape must equal the
#: floor table's ``avg_seq_tokens`` (the P6 artifact and the registration
#: describe the same sequence) or the plan refuses.
DEMAND_ANCHOR_ARM = "gold-fresh"

#: ADR-0153 (owner GO 2026-10-09, S0F-61): the DRY WINDOW. One per budgeted
#: serving configuration minus r (engine, prefix mode, demand class, kv dtype,
#: connector, topology) with pressure cells, planned at the head of its first
#: budgeted relaunch (the tightest r the class is carried at, since budgets
#: sort ascending): the class's first pressure cell at that r, offered at
#: DRY_WINDOW_RATE_FRAC x lambda*, one window, written to the sibling run root
#: ``<run_id>-dry`` (the campaign tree never sees it). Its regime.json must
#: read IN_REGIME; any other label fails every pressure cell of that (engine,
#: class) with a named sentinel, the budget-free cells still run, and the exit
#: code is EXIT_DRY_WINDOW_FAILED. Owner-delegated decision 2026-10-09: the
#: class fails and the run continues (F1 data does not depend on lambda*).
DRY_WINDOW_RATE_FRAC: float = 0.95
assert DRY_WINDOW_RATE_FRAC in D6_RATE_FRACTIONS and DRY_WINDOW_RATE_FRAC in D6_REDUCED_RATE_FRACTIONS, (
    "the dry window rate must be a registered dispatcher fraction on every grid"
)
DRY_WINDOW_EXPECTED_LABEL = "IN_REGIME"
DRY_WINDOW_ROOT_SUFFIX = "-dry"
#: ADR-0153 (review MEDIUM 6, 2026-10-09): the dry window runs in DURATION
#: mode for the registered probe window length with the pool replayed
#: (non-confirmatory, like the ladder), so its regime reading rests on 60 or
#: more sampler ticks whatever the class's derived rate; a W-arrival window
#: at the small classes' rates would span a few seconds (S0F-68).
DRY_WINDOW_DURATION_S: float = PROBE_WINDOW_S
DRY_WINDOW_ADR = "ADR-0153"
DRY_WINDOW_FINDING = "S0F-61"
DRY_WINDOW_GATE = "dry window: the first window of this engine and demand class must read IN_REGIME"
#: run's exit code when every executed cell passed but a dry window failed
#: its class (distinct from 1, a failed cell, and 2, a failed engine stop).
EXIT_DRY_WINDOW_FAILED: int = 3

#: ADR-0156 (S0F-68): the expected span of a pressure window is W / offered
#: rate (V3: W arrivals, one per prepared request). The regime inputs read one
#: telemetry sample per TELEMETRY_SAMPLE_INTERVAL_S (the runner's sampler,
#: run_experiment.py ``interval=1.0``, mirrored here) and the registered
#: warm-up transient is PROBE_WARMUP_S: a window shorter than that transient
#: is all ramp. Every pressure cell records its span and whether it sits
#: below the floor; the plan header and the master count them. A COUNT, never
#: a refusal: W is the registered per-row N of the power decision, and
#: changing it is the owner's one-way door (the dry window, 75 s, is the live
#: regime proof per serving configuration meanwhile).
TELEMETRY_SAMPLE_INTERVAL_S: float = 1.0
WINDOW_SPAN_FLOOR_S: float = PROBE_WARMUP_S
WINDOW_SPAN_FINDING = "S0F-68"

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
    # ADR-0155 demand classes: arm -> mean served sequence tokens (prompt +
    # output) the arm's byte budget is sized on, and the per-rung shapes of
    # the corpus-trunc ladder. Registered per session; the defaults are the
    # 2026-10-08 landing's measurements with the Qwen3-14B tokenizer. On
    # session b (Llama-3.3-70B) the same counts are registered [A: the Llama
    # tokenizer counts the same passages within about 10 percent; re-measure
    # at the session b rehearsal], visible in the plan header.
    demand_seq_tokens: Mapping[str, int] = field(
        default_factory=lambda: dict(DEMAND_SEQ_TOKENS_2026_10_08)
    )
    corpus_trunc_demand_seq_tokens: Mapping[int, int] = field(
        default_factory=lambda: dict(CORPUS_TRUNC_DEMAND_SEQ_TOKENS_2026_10_08)
    )
    # ADR-0158: the served MAXIMUM per demand class (keyed by the class's served
    # tokens), the basis of the per-class request cap; a grid registering a
    # class without one refuses here, at registration.
    demand_class_max_served_tokens: Mapping[int, int] = field(
        default_factory=lambda: dict(DEMAND_CLASS_MAX_SERVED_TOKENS_2026_10_08)
    )

    def __post_init__(self) -> None:
        problems: List[str] = []
        # ADR-0155: every entry an integer >= 1; the anchor arm present; one
        # shape per registered corpus-trunc rung.
        if not isinstance(self.demand_seq_tokens, Mapping) or not self.demand_seq_tokens:
            problems.append(
                f"demand_seq_tokens={self.demand_seq_tokens!r} must be a non-empty "
                f"mapping arm -> served sequence tokens ({DEMAND_CLASS_ADR})"
            )
        else:
            for arm, value in self.demand_seq_tokens.items():
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    problems.append(
                        f"demand_seq_tokens[{arm!r}]={value!r} must be an integer >= 1"
                    )
            if DEMAND_ANCHOR_ARM not in self.demand_seq_tokens:
                problems.append(
                    f"demand_seq_tokens lacks the anchor arm {DEMAND_ANCHOR_ARM!r} "
                    f"(the floor table's shape, {DEMAND_CLASS_ADR})"
                )
        if not isinstance(self.corpus_trunc_demand_seq_tokens, Mapping):
            problems.append(
                f"corpus_trunc_demand_seq_tokens={self.corpus_trunc_demand_seq_tokens!r} "
                "must be a mapping rung -> served sequence tokens"
            )
        else:
            for rung, value in self.corpus_trunc_demand_seq_tokens.items():
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    problems.append(
                        f"corpus_trunc_demand_seq_tokens[{rung!r}]={value!r} must be an "
                        "integer >= 1"
                    )
            if isinstance(self.corpus_trunc_budgets, tuple):
                missing_rungs = [
                    r for r in self.corpus_trunc_budgets
                    if r not in self.corpus_trunc_demand_seq_tokens
                ]
                if missing_rungs:
                    problems.append(
                        f"corpus_trunc_demand_seq_tokens lacks the registered rung(s) "
                        f"{missing_rungs} ({DEMAND_CLASS_ADR}: the rung is the served "
                        "block, so each rung is its own demand class)"
                    )
        # ADR-0158: every registered class has a served maximum (an int >= 1).
        if isinstance(self.demand_seq_tokens, Mapping) and isinstance(self.corpus_trunc_demand_seq_tokens, Mapping):
            classes = set(self.demand_seq_tokens.values()) | set(self.corpus_trunc_demand_seq_tokens.values())
            table = self.demand_class_max_served_tokens
            if not isinstance(table, Mapping):
                problems.append(f"demand_class_max_served_tokens={table!r} must be a mapping class tokens -> served maximum")
            else:
                for seq in sorted(classes, reverse=True):
                    value = table.get(seq)
                    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                        problems.append(
                            f"demand_class_max_served_tokens lacks an integer >= 1 for the class of "
                            f"{seq} served tokens ({REQUEST_CAP_ADR}: the per-class request cap needs it)"
                        )
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
    # - HF oracle: the same reduced 8-cell slice as session a (batch-1
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
# Pool capacity vs the request-length cap (gap triage 2026-10-09, C1)
# ---------------------------------------------------------------------------

#: vLLM refuses to start when the KV pool cannot hold ONE request of
#: max_model_len tokens: 0.19.1 on the S0 pod, "To serve at least one request
#: with the models's max seq len (32768), (4.5 GiB KV cache is needed, which
#: is larger than the available KV cache memory" (results/s0/vm_logs/
#: 2f54e73fb2dc/cluster/vllm_Qwen_Qwen3-8B_replica-1_8101.log:81, S0.env
#: 2026-10-08 note). ADR-0155 sizes a budgeted relaunch at floor(r x D_class)
#: with D_class = c x s_class x bytes/token, so a SMALL class at a TIGHT r
#: plans a pool of r x c x s_class tokens: retr-trunc (348 tokens) at c = 50
#: holds 17,400 x r tokens, below 32,768 at EVERY registered rung; every such
#: relaunch fails on the pod after the setup, validate and calibrate stages
#: billed, and the dry window then skips the configuration's cells (the
#: anchor-only sizing before ADR-0155 never hit this: 238,950 x r tokens).
#: SGLang's behavior with --max-total-tokens below --context-length was not
#: read [?]; the design fact is engine-independent (the registered cap is a
#: one-request guarantee the pool cannot honor), so the plan refuses on both.
#: The remedy (a per-class request cap, a class budget floor, or a larger c)
#: is the owner's registration decision; see the 2026-10-09 report.
MAX_MODEL_LEN_POOL_RULE = (
    "every budgeted relaunch's KV pool (budget_plan.budget_tokens_total, each P/D pool "
    "on a pd stack) holds at least one request of the relaunch's request cap (its class "
    f"cap, {REQUEST_CAP_ADR}); vLLM refuses a smaller pool at engine start (S0 2026-10-08 log)"
)


def _pool_tokens_of(record: Mapping[str, Any]) -> List[Tuple[str, int]]:
    """(pool label, tokens) per pool of one budget_plan record: the total on a
    single or tp stack, each role's pool on a pd stack (bytes to tokens at the
    plan's own kv dtype through MODEL_KV, the planner's arithmetic)."""
    pools = record.get("pools_bytes")
    if not pools:
        return [("pool", int(record["budget_tokens_total"]))]
    model = str(record["model"])
    kv_dtype = str(record.get("kv_dtype") or "bf16")
    eff = math.floor(MODEL_KV[model].kv_bytes_per_token * KV_DTYPE_FACTOR[kv_dtype])
    return [(role, int(b) // eff) for role, b in zip(("prefill", "decode"), pools)]


def pool_shortfalls(
    relaunch_steps: Sequence[Mapping[str, Any]], max_model_len: int
) -> List[Dict[str, Any]]:
    """The budgeted relaunches whose pool cannot hold one max_model_len
    request (MAX_MODEL_LEN_POOL_RULE), one record each: engine, class tokens,
    r, kv dtype, topology, pool label, pool tokens, shortfall tokens. Pure."""
    out: List[Dict[str, Any]] = []
    for step in relaunch_steps:
        record = step.get("budget_plan")
        if not isinstance(record, Mapping):
            continue
        # ADR-0158: the relaunch's own cap (its class); the session cap when absent
        cap = step.get("max_model_len")
        cap = int(max_model_len) if not isinstance(cap, int) or isinstance(cap, bool) else cap
        for label, tokens in _pool_tokens_of(record):
            if tokens < cap:
                out.append({
                    "engine": step.get("engine"),
                    "seq_tokens": (step.get("demand_class") or {}).get("seq_tokens"),
                    # a pd relaunch carries budget_r None (its r is the registered
                    # dist_budget_r); the BudgetPlan's own r is the one planned
                    "budget_r": record.get("r") if step.get("budget_r") is None else step.get("budget_r"),
                    "kv_dtype": record.get("kv_dtype"),
                    "topology": step.get("topology"),
                    "pool": label,
                    "pool_tokens": tokens,
                    "max_model_len": cap,
                    "shortfall_tokens": cap - tokens,
                })
    return out


def _pool_shortfall_text(shortfalls: Sequence[Mapping[str, Any]], where: str) -> str:
    rows = sorted(shortfalls, key=lambda s: (str(s["engine"]), int(s["seq_tokens"] or 0), float(s["budget_r"] or 0)))
    listed = "; ".join(
        f"{s['engine']} class {s['seq_tokens']} tokens r={float(s['budget_r']):g} {s['topology']}/{s['pool']}: "
        f"{s['pool_tokens']} tokens vs cap {s['max_model_len']} (short {s['shortfall_tokens']})"
        for s in rows[:12]
    )
    more = "" if len(rows) <= 12 else f"; and {len(rows) - 12} more"
    return (
        f"{where}: {len(rows)} budgeted relaunch pool(s) cannot hold one request of their class's "
        f"request cap ({MAX_MODEL_LEN_POOL_RULE}): {listed}{more}. The pod "
        "would fail each relaunch at engine start after the setup, validate and calibrate stages "
        "billed. Remedies (owner, registration): a per-class request cap below the pool, a class "
        "budget floor of one request, or a concurrency c large enough at the tightest rung; none "
        "is chosen here"
    )


def mac_pool_shortfalls(
    grid: SessionGrid, cells: Sequence[PlannedCell], floor_concurrency: int
) -> List[Dict[str, Any]]:
    """The Mac-side (stage 0) image of ``pool_shortfalls``: no floor table
    exists yet, so the demand is cache_budget.demand_bytes at the profile's
    FLOOR_CONCURRENCY and the registered anchor shape, exactly what
    build_floor_table.py writes (demand_bytes(model, concurrency,
    avg_seq_tokens); build_floor_table.py:210), scaled per class
    (_class_demand_bytes) and planned through plan_budget with each executable
    config's own r, kv dtype, tp and topology as _relaunch_step does. One
    record per distinct serving configuration."""
    anchor = int(grid.demand_seq_tokens[DEMAND_ANCHOR_ARM])
    seen: set = set()
    steps: List[Dict[str, Any]] = []
    for cell in cells:
        if cell.blocked_on is not None:
            continue
        config = _serving_config(cell, grid)
        if config is None or config[4] is None or config in seen:
            continue
        seen.add(config)
        spec = cell.spec
        seq_tokens = int(config[4])
        d_floor = demand_bytes(spec.model, concurrency=int(floor_concurrency), avg_seq_tokens=anchor)
        d_class = _class_demand_bytes(d_floor, seq_tokens, anchor)
        kv_dtype = ARM_KV_DTYPE.get(spec.arm) or "bf16"
        if spec.topology == "tp":
            r, tp, topology, split = grid.dist_budget_r, int(grid.dist_tp_size or 1), "tp", None
        elif spec.topology == "pd":
            r, tp, topology, split = grid.dist_budget_r, 1, "pd", grid.dist_pd_split
        else:
            r, tp, topology, split = float(spec.budget_r), int(grid.serving_tp), "single", None
            if tp != 1:
                topology = "tp"
        plan = plan_budget(
            model=spec.model, engine=spec.engine, r=float(r), demand=d_class,
            kv_dtype=kv_dtype, tp=tp, topology=topology, pd_split=split,
        )
        steps.append({
            "engine": spec.engine,
            "budget_r": float(r),
            "topology": spec.topology,
            "max_model_len": request_cap_tokens(grid, seq_tokens),
            "demand_class": {"seq_tokens": seq_tokens},
            "budget_plan": json.loads(json.dumps(asdict(plan))),
        })
    return pool_shortfalls(steps, grid.max_model_len)


# ---------------------------------------------------------------------------
# Stage 0 registration preflight (gap triage 2026-10-09): what the plan
# would refuse at stage 6, checked on the Mac before anything bills
# ---------------------------------------------------------------------------


def preflight_registration_check(
    session: str,
    *,
    floor_avg_seq_tokens: Any,
    rehearsal_n: Optional[int] = None,
    query_manifests: Optional[Mapping[str, Any]] = None,
    charter_datasets: Sequence[str] = (),
    floor_concurrency: Optional[int] = None,
) -> Dict[str, Any]:
    """Purpose: run, on the Mac at stage 0, every registration check the pod
    plan (stage 6) would refuse on after the setup, validate and calibrate
    stages have billed. The gap triage of 2026-10-09 found the registered S1
    profile refused at stage 6 twice over: its 50-id manifests cannot supply
    the registered per-row n (shortfall), and its F1 grid carries HotpotQA
    while CHARTER_DATASETS stages three datasets.

    Args:
        session: the section 1 session id (``SESSION`` in the profile).
        floor_avg_seq_tokens: the profile's FLOOR_AVG_SEQ_TOKENS (string or
            int); must equal the registered anchor shape (ADR-0155).
        rehearsal_n: REHEARSAL_N when set (ADR-0144); the rehearsal grid is
            then the one checked, exactly as stage 6 derives it.
        query_manifests: dataset -> manifest path (QUERY_MANIFESTS), resolved
            paths; validated and coverage-checked like ``plan`` does.
        charter_datasets: CHARTER_DATASETS (the datasets stage 3 stages);
            empty skips the staging check. RULER is generated, never staged.
        floor_concurrency: FLOOR_CONCURRENCY (the c the floor table is built
            at); when given, every executable budgeted configuration's pool
            is checked against max_model_len (MAX_MODEL_LEN_POOL_RULE) and
            the shortfalls are returned under ``pool_shortfalls``.

    Returns: {"session", "anchor_seq_tokens", "cells", "executable_cells",
        "needs_lmcache" (an executable vLLM cell carries the LMCache
        connector, so stage 4 must prove the module imports: S0F-57),
        "pool_shortfalls": [record] (empty without floor_concurrency),
        "problems": [str]} where an empty problems list is a pass.

    Raises: nothing on a registration problem (recorded in ``problems``); a
        programming error propagates.
    """
    problems: List[str] = []
    try:
        grid = get_session_grid(session)
    except PlanError as exc:
        return {
            "session": session, "anchor_seq_tokens": None, "cells": 0,
            "executable_cells": 0, "needs_lmcache": False, "problems": [str(exc)],
        }
    anchor = int(grid.demand_seq_tokens[DEMAND_ANCHOR_ARM])
    if str(floor_avg_seq_tokens).strip() != str(anchor):
        problems.append(
            f"FLOOR_AVG_SEQ_TOKENS={floor_avg_seq_tokens} but session {session!r} registers "
            f"{anchor} served tokens for the anchor arm {DEMAND_ANCHOR_ARM!r} ({DEMAND_CLASS_ADR}): "
            f"the plan refuses a floor table sized on another shape; set FLOOR_AVG_SEQ_TOKENS={anchor}"
        )
    paths = {str(k): Path(v) for k, v in (query_manifests or {}).items()}
    if rehearsal_n is not None:
        try:
            grid = rehearsal_grid(grid, n=rehearsal_n, datasets=frozenset(paths))
        except PlanError as exc:
            problems.append(str(exc))
    if charter_datasets:
        staged = set(charter_datasets) | {"ruler"}
        unstaged = sorted(_grid_datasets(grid) - staged)
        if unstaged:
            problems.append(
                f"session {session!r} carries dataset(s) {unstaged} that CHARTER_DATASETS "
                f"{sorted(charter_datasets)} does not stage: stage 3 would never download them "
                "and every cell of theirs would fail at stage 7"
            )
    cells = enumerate_cells(grid)
    manifests: Dict[str, Dict[str, Any]] = {}
    try:
        manifests = _register_query_manifests(grid, paths)
    except PlanError as exc:
        problems.append(str(exc))
    cells = [
        replace(c, blocked_on=trunc_manifest_blocked_on(c.dataset))
        if c.spec.arm == CORPUS_TRUNC_ARM and c.dataset not in manifests and c.blocked_on is None
        else c
        for c in cells
    ]
    try:
        _check_manifest_coverage(grid, manifests, cells)
    except PlanError as exc:
        problems.append(str(exc))
    executable = [c for c in cells if c.blocked_on is None]
    needs_lmcache = any(
        c.spec.engine == "vllm" and ARM_CONNECTOR.get(c.spec.arm) == "lmcache" for c in executable
    )
    shortfalls: List[Dict[str, Any]] = []
    if floor_concurrency is not None:
        shortfalls = mac_pool_shortfalls(grid, cells, int(floor_concurrency))
        if shortfalls:
            problems.append(_pool_shortfall_text(
                shortfalls, f"session {session!r} at FLOOR_CONCURRENCY={int(floor_concurrency)}"
            ))
    return {
        "session": session,
        "anchor_seq_tokens": anchor,
        "cells": len(cells),
        "executable_cells": len(executable),
        "needs_lmcache": needs_lmcache,
        "pool_shortfalls": shortfalls,
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# Dress rehearsal of a registered session (2026-10-07, ADR-0144)
# ---------------------------------------------------------------------------

REHEARSAL_ADR: str = "ADR-0144"
#: The ONE derivation rule, recorded verbatim in the plan header so the
#: operator reviews what was collapsed, never a hand-picked cell list.
REHEARSAL_RULE: str = (
    "every baseline, engine, HF-oracle cell, RULER task and B12 rung of the "
    "registered session; datasets restricted to those with a registered query "
    "manifest; F2 at the lower-median (budget, rate) of its factorial plus the "
    "first fine-only coordinate when a fine grid is registered; F3 at its lower "
    "medians; one window per cell; every row class at n; no achievable_n override"
)


def _lower_median(values: Sequence[float]) -> float:
    """The lower median of a non-empty sequence (sorted ascending, index
    (len - 1) // 2): deterministic, always a registered member."""
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def rehearsal_grid(
    base: SessionGrid, *, n: int, datasets: FrozenSet[str]
) -> SessionGrid:
    """Purpose: derive the dress-rehearsal grid of a registered session.

    Args:
        base: the registered SessionGrid (never modified; a new instance is
            returned through dataclasses.replace, which re-runs the
            registration's own validation).
        n: the per-cell query count every row class is set to; integer >= 1.
        datasets: the datasets with a registered query manifest; F1 and the
            HF-oracle cells keep only these, and the F2/F3 datasets must be
            among them when the session registers F2/F3 cells.

    Returns: a SessionGrid with the same session, group, model, baselines,
        engines, RULER registration, B12 rungs and behavior knobs, the axes
        collapsed by REHEARSAL_RULE, replications 1, every N class = n and
        achievable_n empty.

    Raises: PlanError when n < 1, when no F1 dataset of the session has a
        manifest, or when the F2 or F3 dataset has none (the plan would
        refuse the pressure cells' manifest coverage anyway; refusing here
        names the fix).

    Invariants: every (baseline, engine, family) of ``base`` enumerates at
        least once in the result (pinned by tests); the registered grids are
        untouched.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise PlanError(f"rehearsal n={n!r} must be an integer >= 1 ({REHEARSAL_ADR})")
    if not datasets:
        raise PlanError(
            f"a rehearsal needs at least one registered --query-manifest: it keeps "
            f"only datasets with one ({REHEARSAL_ADR})"
        )
    f1_datasets = tuple(d for d in base.f1_datasets if d in datasets)
    if not f1_datasets:
        raise PlanError(
            f"rehearsal of session {base.session!r}: none of its F1 datasets "
            f"{list(base.f1_datasets)} has a registered query manifest "
            f"(registered: {sorted(datasets)}) ({REHEARSAL_ADR})"
        )
    for family, baselines, dataset in (
        ("F2", base.f2_baselines, base.f2_dataset),
        ("F3", base.f3_baselines, base.f3_dataset),
    ):
        if baselines and dataset not in datasets:
            raise PlanError(
                f"rehearsal of session {base.session!r}: the {family} dataset "
                f"{dataset!r} has no registered query manifest (registered: "
                f"{sorted(datasets)}); plan with --query-manifest {dataset}=<path> "
                f"({REHEARSAL_ADR})"
            )
    hf_cells = tuple(
        (bid, kept)
        for bid, kept in (
            (bid, tuple(d for d in ds if d in datasets)) for bid, ds in base.hf_oracle_cells
        )
        if kept
    )
    fine_budgets = tuple(
        r for r in base.f2_fine_budgets if r not in base.f2_budgets
    )[:1]
    fine_rates = base.f2_fine_rates[:1] if fine_budgets else ()
    return replace(
        base,
        f1_datasets=f1_datasets,
        hf_oracle_cells=hf_cells,
        f2_budgets=(_lower_median(base.f2_budgets),) if base.f2_budgets else (),
        f2_rates=(_lower_median(base.f2_rates),) if base.f2_rates else (),
        f2_fine_budgets=fine_budgets,
        f2_fine_rates=fine_rates,
        f3_budgets=(_lower_median(base.f3_budgets),) if base.f3_budgets else (),
        f3_rates=(_lower_median(base.f3_rates),) if base.f3_rates else (),
        replications=1,
        n_primary=n,
        n_secondary=n,
        n_identity=n,
        window_requests=n,
        achievable_n={},
    )


# ---------------------------------------------------------------------------
# Floor table (T2.4 floor-table-v1) — REQUIRED demand/λ* source
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FloorTable:
    """Validated floor-table-v1 content: demand + λ* rows keyed by r.

    ``avg_seq_tokens`` and ``concurrency_target`` are the shape the demand D
    was sized on (``generated_inputs``; None when the artifact predates the
    fields). ADR-0155: the plan refuses a budgeted cell unless
    ``avg_seq_tokens`` equals the registered anchor arm's served tokens, and
    scales D per demand class from it.
    """

    path: Path
    sha256: str
    model: str
    engine: str
    kv_dtype: str
    grid: str
    rows: Dict[float, Dict[str, Any]]
    avg_seq_tokens: Optional[int] = None
    concurrency_target: Optional[int] = None

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
            pred = float(row["lambda_star_pred_rps"])
            if not (math.isfinite(pred) and pred > 0):
                problems.append(
                    f"rows[{i}].lambda_star_pred_rps={row['lambda_star_pred_rps']!r} must be a "
                    "positive finite rate (the P6 prediction divides every pressure cell's span)"
                )
                continue
            if isinstance(row["demand_bytes"], float) or int(row["demand_bytes"]) < 1:
                problems.append(f"rows[{i}].demand_bytes={row['demand_bytes']!r} must be an integer >= 1")
                continue
            rows[float(row["r"])] = row
    shape: Dict[str, Optional[int]] = {}
    for key in ("avg_seq_tokens", "concurrency_target"):
        value = inputs.get(key)
        if value is None:
            shape[key] = None
        elif isinstance(value, bool) or not isinstance(value, int) or value < 1:
            problems.append(f"generated_inputs.{key}={value!r} must be an integer >= 1")
        else:
            shape[key] = value
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
        avg_seq_tokens=shape.get("avg_seq_tokens"),
        concurrency_target=shape.get("concurrency_target"),
    )


# ---------------------------------------------------------------------------
# Calibration artifacts (cal-v1, calibrate_cell.py): the §6.1 floors source
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationFloor:
    """Validated cal-v1 artifact (calibrate_cell.py --output): ONE model x
    engine's §6.1 single-stream floor pair plus the provenance the plan
    header records. ``engine`` is the §7.3 axis value (the adapter id the
    artifact carries, normalized); ``model`` is as the artifact spells it."""

    path: Path
    sha256: str
    engine: str
    model: str
    budget_fraction: float
    ttft_s: float
    tpot_s: float
    n_requests: int
    statistic: str
    procedure_version: str
    lambda_star_label: Optional[str]


def load_calibration(
    path: Path, budget_fraction: float = FLOOR_BUDGET_FRACTION
) -> CalibrationFloor:
    """Load + validate ONE cal-v1 calibration artifact (fail closed).

    The artifact is ``CellCalibration.to_manifest`` as calibrate_cell.py
    writes it: the registered procedure version, the served model, the
    ADAPTER engine id (``lmdeploy-turbomind`` for LMDeploy; normalized here
    through campaign_session.ENGINE_OF_BACKEND to the axis value the analysis
    keys floors by), the budget ratio the floor was measured at,
    ``confirmatory: false`` and the floor pair in SECONDS over
    FLOOR_N_REQUESTS FLOOR_STATISTIC requests. A floor over fewer requests,
    another statistic, another procedure version, a non-positive or
    non-finite value, an unknown engine or the oracle refuses: a plan never
    pins a floor the registered procedure did not produce. ``budget_fraction``
    is the rung the plan registers, named in the not-found fix only (the
    value itself is checked by the registration, review F7).
    """
    path = Path(path)
    if not path.is_file():
        raise PlanError(
            f"calibration artifact not found: {path} (run "
            f"scripts/3_run/calibrate_cell.py --budget-fraction "
            f"{budget_fraction:g} --output <path> per engine; "
            f"{SLO_FLOORS_FINDING})"
        )
    raw = path.read_bytes()
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PlanError(f"calibration artifact {path} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise PlanError(f"calibration artifact {path} is not a JSON object")
    problems: List[str] = []
    version = doc.get("procedure_version")
    if version != PROCEDURE_VERSION:
        problems.append(
            f"procedure_version is {version!r}, the registered procedure is "
            f"{PROCEDURE_VERSION!r} (src/orchestration/calibration.py)"
        )
    if doc.get("confirmatory") is not False:
        problems.append(
            f"confirmatory is {doc.get('confirmatory')!r}, must be the literal "
            "false (calibration data never enters confirmatory analysis)"
        )
    raw_engine = doc.get("engine")
    engine = ENGINE_OF_BACKEND.get(raw_engine) if isinstance(raw_engine, str) else None
    if engine is None:
        problems.append(
            f"engine {raw_engine!r} is not a runner backend / adapter id "
            f"({sorted(ENGINE_OF_BACKEND)})"
        )
    elif engine == "hf":
        problems.append(
            f"engine {raw_engine!r} is the in-process oracle, which serves no "
            "floor (§7.3: hf is excluded from pressure)"
        )
    model = doc.get("model")
    if not isinstance(model, str) or not model.strip():
        problems.append(f"model {model!r} must be a non-empty string")
    fraction = doc.get("budget_fraction")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or fraction <= 0
    ):
        problems.append(f"budget_fraction {fraction!r} must be finite and > 0")
    floor = doc.get("floor")
    ttft_s = tpot_s = n_requests = statistic = None
    if not isinstance(floor, dict):
        problems.append("floor block missing or not an object")
    else:
        for key in ("ttft_s", "tpot_s"):
            value = floor.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                problems.append(
                    f"floor.{key} is {value!r}, must be a finite number > 0 (seconds)"
                )
        ttft_s, tpot_s = floor.get("ttft_s"), floor.get("tpot_s")
        n_requests = floor.get("n_requests")
        if (
            isinstance(n_requests, bool)
            or not isinstance(n_requests, int)
            or n_requests < FLOOR_N_REQUESTS
        ):
            problems.append(
                f"floor.n_requests is {n_requests!r}, the registered floor is "
                f">= {FLOOR_N_REQUESTS} sequential single-stream requests"
            )
        statistic = floor.get("statistic")
        if statistic != FLOOR_STATISTIC:
            problems.append(
                f"floor.statistic is {statistic!r}, the registered statistic is "
                f"{FLOOR_STATISTIC!r}"
            )
    lambda_star = doc.get("lambda_star")
    label = lambda_star.get("label") if isinstance(lambda_star, dict) else None
    if problems:
        raise PlanError(f"calibration artifact {path}: " + "; ".join(problems))
    assert engine is not None and isinstance(model, str)
    return CalibrationFloor(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        engine=engine,
        model=model,
        budget_fraction=float(fraction),  # type: ignore[arg-type]
        ttft_s=float(ttft_s),  # type: ignore[arg-type]
        tpot_s=float(tpot_s),  # type: ignore[arg-type]
        n_requests=int(n_requests),  # type: ignore[arg-type]
        statistic=str(statistic),
        procedure_version=str(version),
        lambda_star_label=label if isinstance(label, str) else None,
    )


def slo_floors_env_value(floors: Mapping[str, Any]) -> str:
    """The ONE spelling of the floors pin: compact JSON, sorted keys, so the
    plan builder, load_plan and the session compare byte-identical strings."""
    return json.dumps(floors, sort_keys=True, separators=(",", ":"))


def budget_plan_record(
    plan: BudgetPlan,
    floor: FloorTable,
    pd_roles: Optional[Mapping[str, Any]] = None,
    seq_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """The cell.json ``budget_plan`` record: the full BudgetPlan (asdict) plus
    the floor table's sha256 (the demand source), the demand class's served
    tokens (``avg_seq_tokens``, ADR-0155: the shape the record's
    ``demand_bytes`` was scaled to) and, on the pd stack, the relaunch's role
    record (``pd_roles``: the §6.5 pools, the per-rank slices the launcher was
    given and the expected realized total, review F6; the BudgetPlan itself is
    per POOL, tp=1), JSON-normalized ONCE so the relaunch step, the cell step
    and the env pin are the same object. The consumer reads
    ``budget_bytes_total`` and ``kv_dtype``; the launched knobs are
    ``engine_args`` (single/tp) or ``pd_roles`` (pd)."""
    record = asdict(plan)
    record["floor_table_sha256"] = floor.sha256
    if seq_tokens is not None:
        record["avg_seq_tokens"] = int(seq_tokens)
        record["demand_class_adr"] = DEMAND_CLASS_ADR
    if pd_roles is not None:
        record["pd_roles"] = dict(pd_roles)
    return json.loads(json.dumps(record))


def _json_norm(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def window_span_summary(cell_steps: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """ADR-0156 / S0F-68: the distribution of the executable pressure cells'
    expected window spans (W / offered rate) against WINDOW_SPAN_FLOOR_S,
    overall and per demand class; the header record the operator and the
    master read before the GO. Pure: reads the steps' recorded spans."""
    spans: List[Tuple[float, int]] = []
    for s in cell_steps:
        span = s.get("window_span_s_expected")
        if s.get("family") in _PRESSURE_FAMILIES and s.get("blocked_on") is None and isinstance(span, (int, float)):
            seq = (s.get("demand_class") or {}).get("seq_tokens")
            spans.append((float(span), int(seq) if isinstance(seq, int) else 0))
    values = sorted(v for v, _seq in spans)
    by_class: Dict[str, Dict[str, Any]] = {}
    for span, seq in spans:
        rec = by_class.setdefault(str(seq), {"cells": 0, "below_floor": 0, "min_s": None})
        rec["cells"] += 1
        rec["below_floor"] += int(span < WINDOW_SPAN_FLOOR_S)
        rec["min_s"] = span if rec["min_s"] is None else min(rec["min_s"], span)
    median = None if not values else float(statistics.median(values))
    return {
        "finding": WINDOW_SPAN_FINDING,
        "adr": TWO_ANCHOR_ADR,
        "rule": "span_s = num_queries / offered_rate_rps (V3: W arrivals, one per prepared request)",
        "floor_s": WINDOW_SPAN_FLOOR_S,
        "floor_basis": "PROBE_WARMUP_S, the registered warm-up transient: a shorter window is all ramp",
        "sampler_interval_s": TELEMETRY_SAMPLE_INTERVAL_S,
        "executable_pressure_cells": len(values),
        "below_floor": sum(1 for v in values if v < WINDOW_SPAN_FLOOR_S),
        "min_s": values[0] if values else None,
        "median_s": median,
        "max_s": values[-1] if values else None,
        "by_class": dict(sorted(by_class.items(), key=lambda kv: -int(kv[0]))),
        "note": (
            "a count, never a refusal: W is the registered per-row N (DECISION.md A1); "
            "the dry window (75 s) is the live regime proof per serving configuration; "
            "raising W, or a duration bound with replay, is the owner's decision"
        ),
    }


#: _stale_plan_problems sentinel: "the caller did not pass the header floors"
#: (None means "the plan registers no floor, pins must be absent").
_UNCHECKED: Any = object()


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
    in PREFIX_OFF_ARMS (ADR-0103 and ADR-0150: the charter's reuse-off arms,
    in every family); every other cell serves ON.

    ADR-0103 closed the mislabeled-duplicate exposure for B4 and ADR-0150
    closes it for B1, B5/B6, B9 and B11 (the failure class the module
    docstring names): the runner tokens only label telemetry, so without
    this clause each fresh arm shared one prefix-ON server with its reuse
    twin and the pair were byte-identical serving twins under different
    names. Making the prefix mode part of the arm's serving config gives the
    OFF arms their own relaunch group (the grouping key already carries
    prefix_off) while their family carriage is untouched.
    """
    return spec.family == "F2" or spec.arm in PREFIX_OFF_ARMS


def _coord_key(value: Optional[float]) -> Tuple[int, float]:
    # ADR-0153 pressure-first order: budgets sort ASCENDING (the tightest r
    # first, where each demand class's dry window runs) and None (sub-pressure,
    # the budget-free server config) sorts LAST, so an engine's pressure groups
    # and their dry windows run before its F1 block. Before Batch B the F1
    # block ran first and the first pressure window came hours in.
    return (1, 0.0) if value is None else (0, float(value))


def _is_budgeted(spec: CellSpec) -> bool:
    """A cell whose serving stack launches under a byte budget: a pressure
    coordinate (F2/F3) or the DIST overlay at the registered dist_budget_r."""
    return spec.budget_r is not None or spec.topology in ("tp", "pd")


def demand_seq_tokens(grid: SessionGrid, spec: CellSpec) -> int:
    """ADR-0155: the served sequence tokens of a cell's DEMAND CLASS, from the
    session's registration: the arm's entry, or the rung's entry for a
    corpus-trunc cell (the rung is the served block). An arm with no entry
    refuses: a budget sized on another arm's shape is the S0F-63 defect.
    RULER cells are gold-fresh cells (the instrument rides the twin's class)."""
    if spec.arm == CORPUS_TRUNC_ARM:
        rung = spec.corpus_budget_tokens
        value = (
            grid.corpus_trunc_demand_seq_tokens.get(rung) if rung is not None else None
        )
        if value is None:
            raise PlanError(
                f"corpus-trunc rung {rung!r} has no registered demand class "
                f"(SessionGrid.corpus_trunc_demand_seq_tokens, {DEMAND_CLASS_ADR})"
            )
        return int(value)
    value = grid.demand_seq_tokens.get(spec.arm)
    if value is None:
        raise PlanError(
            f"arm {spec.arm!r} has no registered demand class "
            f"(SessionGrid.demand_seq_tokens, {DEMAND_CLASS_ADR}, {DEMAND_CLASS_FINDING}): "
            "a budget sized on another arm's shape is the mislabeled-pressure "
            "defect of the 2026-10-08 landing"
        )
    return int(value)


def _class_demand_bytes(demand_floor: int, seq_tokens: int, anchor_tokens: int) -> int:
    """D_class = floor(D_floor x s_class / s_anchor): the floor table's demand
    scaled by the served-token ratio (D is linear in the sequence length,
    cache_budget.demand_bytes). Exact for the anchor class (ratio 1)."""
    return (int(demand_floor) * int(seq_tokens)) // int(anchor_tokens)


def _check_floor_shape(grid: SessionGrid, floor: FloorTable) -> int:
    """ADR-0155: the P6 artifact and the registration must describe the same
    anchor sequence; returns the anchor's served tokens."""
    anchor = int(grid.demand_seq_tokens[DEMAND_ANCHOR_ARM])
    if floor.avg_seq_tokens is None:
        raise PlanError(
            f"floor table {floor.path} carries no generated_inputs.avg_seq_tokens: "
            f"the demand classes ({DEMAND_CLASS_ADR}) scale D from the shape the "
            "table was sized on; rebuild it with build_floor_table.py "
            f"--avg-seq-tokens {anchor}"
        )
    if floor.avg_seq_tokens != anchor:
        raise PlanError(
            f"floor table {floor.path} was sized on avg_seq_tokens="
            f"{floor.avg_seq_tokens} but session {grid.session!r} registers "
            f"{anchor} served tokens for the anchor arm {DEMAND_ANCHOR_ARM!r} "
            f"({DEMAND_CLASS_ADR}: {DEMAND_SEQ_TOKENS_SOURCE}); rebuild the table "
            f"with --avg-seq-tokens {anchor} (profile FLOOR_AVG_SEQ_TOKENS) or "
            "re-register the shape"
        )
    return anchor


def _lever_key(value: Optional[str]) -> str:
    # None (no lever) sorts before any lever value; a plain string keeps the
    # order total without inventing a numeric encoding.
    return "" if value is None else value


def _demand_class_key(grid: SessionGrid, spec: CellSpec) -> Tuple[int, int]:
    # A budget-free config has no demand class; budgeted configs group by
    # their class's served tokens (ADR-0155: the class is a serving-config
    # dimension because the byte budget differs per class at one r), the
    # LARGEST class first: the anchor (gold) leads each budget and the B12
    # corpus-trunc rungs keep their ADR-0106 descending ladder order.
    return (1, -demand_seq_tokens(grid, spec)) if _is_budgeted(spec) else (0, 0)


def _sort_key(cell: PlannedCell, grid: SessionGrid) -> Tuple[Any, ...]:
    """Relaunch-minimizing order:
    (engine, topology, prefix_mode, model, budget_r, demand class, kv_dtype,
    connector, rate).

    The launch levers (kv_dtype, connector) sort BEFORE rate: they are part
    of the serving config while rate is client-side, so lever-bearing cells
    must group contiguously across rate levels or every rate change would
    straddle a lever boundary and force extra relaunches. The demand class
    (ADR-0155) sits beside the budget: at one r each class launches its own
    byte budget. Trailing components (family, baseline number, dataset) only
    make the order total/deterministic — they never split a serving config.
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
        _demand_class_key(grid, spec),
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


#: (engine, prefix_off, model, budget_r, demand-class seq_tokens, kv_dtype,
#: connector, topology)
_ServingConfig = Tuple[
    str, bool, str, Optional[float], Optional[int], Optional[str], Optional[str], str
]


def _serving_config(cell: PlannedCell, grid: SessionGrid) -> Optional[_ServingConfig]:
    """The relaunch-boundary identity; None for the in-process hf oracle.

    Rate is CLIENT-side load (never a server dial), so it is deliberately
    absent — rate changes must not force a relaunch. The launch levers
    (kv_dtype for corpus-comp, connector for retr-store) ARE server dials
    (run_compression.sh / run_kv_store.sh apply them at launch), so they are
    relaunch-boundary dimensions — and so is topology (T3.2: a pd stack is a
    different process set than a single server; sharing a boundary would run
    one topology's cells against the other's serving stack). ADR-0155: the
    demand class (served sequence tokens) is one too on budgeted configs: the
    byte budget at one r differs per class, so two classes cannot share a
    server; a budget-free config carries None.
    """
    spec = cell.spec
    if spec.engine == "hf":
        return None
    return (
        spec.engine,
        _prefix_off(spec),
        spec.model,
        spec.budget_r,
        demand_seq_tokens(grid, spec) if _is_budgeted(spec) else None,
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
    *,
    seq_tokens: int,
    anchor_tokens: int,
) -> Tuple[Dict[str, str], int]:
    """Launcher budget env for one serving config, via cache_budget.plan_budget.

    Demand comes from the floor table row at this r (T2.4 is the ONE demand
    source), scaled to the relaunch's DEMAND CLASS (ADR-0155: D_class =
    floor(D_floor x seq_tokens / anchor_tokens); the anchor class is
    byte-identical to the table); the primary knob of the resulting
    BudgetPlan maps onto the launcher env (frozen contract). The BYTE budget
    stays anchored to floor(r × D_class) for every config of one class at the
    same r (the iso-bytes comparison anchor that makes B10's double-saving
    measurable: B10 and B3 share the corpus class), but token-denominated
    dials (SGLang --max-total-tokens) must convert bytes at the SERVED KV
    dtype — an fp8 server planned with bf16 arithmetic would realize only
    HALF the byte budget (§6.5 violation). ``tp`` > 1 plans the TP-sharded
    launch (plan_budget topology='tp': GQA shards → the primary knob carries
    the PER-RANK slice; MLA replicates, #20) — the TOTAL byte budget is
    still floor(r × D_class). Returns (env, plan): the BudgetPlan the env was
    derived from rides the relaunch step and every budgeted cell under it
    (Batch 2 W4, cell.json["budget_plan"]).
    """
    row = floor.row(r)
    demand_class = _class_demand_bytes(int(row["demand_bytes"]), seq_tokens, anchor_tokens)
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
            demand=demand_class,
            kv_dtype=served_kv_dtype or floor.kv_dtype,
            tp=tp,
            topology="single" if tp == 1 else "tp",
        )
    except CacheBudgetError as exc:
        raise PlanError(
            f"budget planning refused for engine={engine} r={r:g} "
            f"demand class {seq_tokens} tokens: {exc}"
        ) from exc
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
    return env, plan


def _pd_budget_env(
    grid: SessionGrid,
    engine: str,
    model: str,
    floor: FloorTable,
    role_tp: int,
    *,
    seq_tokens: int,
    anchor_tokens: int,
) -> Tuple[Dict[str, str], Dict[str, Any], BudgetPlan]:
    """PD launcher env: the §6.5 split of floor(dist_budget_r × D) into the
    two REQUIRED per-role byte budgets, plus the role-tagged telemetry
    endpoints. ``role_tp`` is the TP degree BOTH role instances launch with
    (the frozen one-env launcher contract): the launcher passes each budget
    env VERBATIM as ``--kv-cache-memory-bytes`` on that instance, and the
    flag's registered convention is PER-RANK (the same convention the paired
    tp leg's _budget_env emits; live per-rank-vs-whole-pool semantics stays
    the plan's verify_live entry, closed by gate (j) at S0) — so a TP-sharded
    role's env carries the per-rank SLICE of its §6.5 pool, never the pool
    total. Returns (env, pd-record-for-the-plan, the BudgetPlan)."""
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
            # ADR-0155: the DIST leg's demand is its arm's class (B1 gold, B3
            # corpus), the same scaling as the single-instance relaunches.
            demand=_class_demand_bytes(int(row["demand_bytes"]), seq_tokens, anchor_tokens),
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
        PD_TELEMETRY_ENDPOINTS_ENV: PD_TELEMETRY_ENDPOINTS,
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
    return env, record, plan


def _relaunch_step(
    config: _ServingConfig,
    grid: SessionGrid,
    floor: FloorTable,
    launcher_cmds: Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    engine, prefix_off, model, budget_r, seq_tokens, kv_dtype, connector, topology = config
    anchor_tokens = int(grid.demand_seq_tokens[DEMAND_ANCHOR_ARM])
    demand_class: Optional[Dict[str, Any]] = None
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
        # V2: the stop of THIS stack, run at the boundaries stop_boundaries
        # names (the pd launcher's stop verb dismantles proxy + both roles).
        stop_argv = list(launcher) + [LAUNCHER_STOP_VERB]
        # Per-role TP (W4.6): the frozen pd launcher applies ONE
        # CAGE_VLLM_TENSOR_PARALLEL to BOTH role instances (validated equal
        # at registration); degree 1 omits the env — the launcher omits the
        # flag, keeping (1,1) plans byte-identical to pre-W4.6 ones. The
        # degree also shapes the budget env (per-rank slices — see
        # _pd_budget_env), so it is derived BEFORE the env is built.
        role_tp = grid.dist_pd_role_gpus[0]
        assert seq_tokens is not None  # a pd config is budgeted by identity
        env, pd_record, plan = _pd_budget_env(
            grid, engine, model, floor, role_tp,
            seq_tokens=seq_tokens, anchor_tokens=anchor_tokens,
        )
        demand_class = {
            "seq_tokens": seq_tokens,
            "anchor_seq_tokens": anchor_tokens,
            "demand_bytes": plan.demand_bytes,
            "adr": DEMAND_CLASS_ADR,
        }
        if role_tp >= 2:
            env["CAGE_VLLM_TENSOR_PARALLEL"] = str(role_tp)
        # Backlog A10 / ADR-0158: the class request cap, applied by the pd
        # launcher to BOTH role instances (one env, frozen contract).
        request_cap = request_cap_tokens(grid, seq_tokens)
        env[MAX_MODEL_LEN_ENV] = str(request_cap)
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
            "max_model_len": request_cap,
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
            # ADR-0155: the demand class this stack's budget was sized on.
            "demand_class": demand_class,
            # V2: the pd stack is its own launcher family (never the
            # single-instance vllm script), stopped by its own verb.
            "launcher_key": PD_LAUNCHER_KEY,
            "stop_argv": stop_argv,
            "api_base": engine_api_base(engine, topology),  # W2: the proxy
            # Batch 2 W4: the BudgetPlan the role budgets were split from
            # (per POOL, tp=1) plus the pd role record (the per-rank slices
            # the launcher was actually given, review F6).
            "budget_plan": budget_plan_record(
                plan, floor, pd_roles=pd_record, seq_tokens=seq_tokens
            ),
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
    # V2: the stop of this engine family (restart is self-cleaning WITHIN the
    # family; the stop runs only at a launcher change and at the end).
    stop_argv = list(launcher) + [LAUNCHER_STOP_VERB]
    env: Dict[str, str] = {}
    budget_bytes: Optional[int] = None
    budget_plan: Optional[Dict[str, Any]] = None
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
        assert seq_tokens is not None  # a tp config is budgeted by identity
        env, plan = _budget_env(
            engine, model, grid.dist_budget_r, floor,
            served_kv_dtype=kv_dtype, tp=tp_size,
            seq_tokens=seq_tokens, anchor_tokens=anchor_tokens,
        )
        budget_bytes = plan.budget_bytes_total
        budget_plan = budget_plan_record(plan, floor, seq_tokens=seq_tokens)
        demand_class = {
            "seq_tokens": seq_tokens,
            "anchor_seq_tokens": anchor_tokens,
            "demand_bytes": plan.demand_bytes,
            "adr": DEMAND_CLASS_ADR,
        }
        launched_tp = tp_size
    else:
        if budget_r is not None:
            assert seq_tokens is not None  # a budgeted config carries its class
            env, plan = _budget_env(
                engine, model, budget_r, floor,
                served_kv_dtype=kv_dtype, tp=grid.serving_tp,
                seq_tokens=seq_tokens, anchor_tokens=anchor_tokens,
            )
            budget_bytes = plan.budget_bytes_total
            budget_plan = budget_plan_record(plan, floor, seq_tokens=seq_tokens)
            demand_class = {
                "seq_tokens": seq_tokens,
                "anchor_seq_tokens": anchor_tokens,
                "demand_bytes": plan.demand_bytes,
                "adr": DEMAND_CLASS_ADR,
            }
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
    # Backlog A10 / ADR-0158: the request-length cap rides EVERY relaunch of
    # BOTH engines (vLLM --max-model-len, SGLang --context-length), budget-
    # free F1 relaunches included (they take the session cap): a server
    # launched without it would fall back to the pilot shell default 4096 and
    # refuse every RULER request. A budgeted relaunch takes its CLASS cap.
    request_cap = request_cap_tokens(grid, seq_tokens)
    env[MAX_MODEL_LEN_ENV] = str(request_cap)
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
        "max_model_len": request_cap,
        "budget_r": budget_r,
        "budget_bytes": budget_bytes,
        "kv_dtype": kv_dtype,
        "connector": connector,
        "topology": topology,
        "tp": launched_tp,  # the T3.1 degree this serving stack launches with
        "pd": None,  # single/tp relaunch: no §6.5 role split
        # ADR-0155: the demand class this budget was sized on (None on a
        # budget-free relaunch: absence stays absence).
        "demand_class": demand_class,
        # V2: the launcher this boundary belongs to and how to stop it (the
        # tp overlay rides the same single-instance launcher as 'single').
        "launcher_key": engine,
        "stop_argv": stop_argv,
        "api_base": api_base,  # W2: what the cells under this relaunch dial
        # Batch 2 W4: the BudgetPlan the budget env was derived from (null on
        # a budget-free relaunch); every budgeted cell under it pins it.
        "budget_plan": budget_plan,
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


def _budget_plan_problems(
    step: Mapping[str, Any], expected: Optional[Mapping[str, Any]], label: str
) -> List[str]:
    """Batch 2 W4 per-cell clause: ``expected`` is the budget record of the
    relaunch the cell runs under (None for hf, blocked, or a budget-free
    boundary). A budgeted cell carries EXACTLY that record and the env pin;
    every other cell carries neither."""
    row = step.get("row_key")
    env: Mapping[str, Any] = step.get("env") or {}
    record = step.get("budget_plan")
    got_env = env.get(BUDGET_PLAN_ENV)
    problems: List[str] = []
    stale = ": stale plan, re-plan"
    if expected is None:
        if record is not None:
            problems.append(
                f"{label}: cell {row!r} carries a budget_plan record but runs "
                "under no byte budget (hf, blocked, or a budget-free relaunch) "
                f"({SLO_FLOORS_FINDING})" + stale
            )
        if got_env is not None:
            problems.append(
                f"{label}: cell {row!r} env {BUDGET_PLAN_ENV} is set but the cell "
                f"runs under no byte budget ({SLO_FLOORS_FINDING})" + stale
            )
        return problems
    want = _json_norm(expected)
    if _json_norm(record) != want:
        problems.append(
            f"{label}: cell {row!r} budget_plan record differs from the record "
            "of the relaunch it runs under (budget_bytes_total "
            f"{want.get('budget_bytes_total')!r}) ({SLO_FLOORS_FINDING}: a cell "
            "served under a budget it does not record is mislabeled data)"
            + stale
        )
    if got_env is None:
        problems.append(
            f"{label}: cell {row!r} lacks env {BUDGET_PLAN_ENV} "
            f"({SLO_FLOORS_FINDING}: the campaign session persists it into "
            f"cell.json[{BUDGET_PLAN_CELL_KEY!r}], the rho_own basis)" + stale
        )
    else:
        try:
            parsed: Any = json.loads(got_env)
        except (TypeError, json.JSONDecodeError):
            parsed = None
        if not isinstance(parsed, dict) or _json_norm(parsed) != want:
            problems.append(
                f"{label}: cell {row!r} env {BUDGET_PLAN_ENV} differs from the "
                f"relaunch record ({SLO_FLOORS_FINDING})" + stale
            )
    return problems


def _stale_plan_problems(
    step: Mapping[str, Any],
    spec: CellSpec,
    preceding_relaunch: Optional[Mapping[str, Any]],
    label: str,
    per_row_n: Optional[Mapping[str, Any]] = None,
    retrieval: Optional[Mapping[str, Any]] = None,
    slo_floors_env: Any = _UNCHECKED,
) -> List[str]:
    """load_plan's fail-closed per-cell check against EVERY stale plan shape
    today's ADRs fail-close against (v4; one clause per ADR). A stale plan
    that passes the schema literal but carries pre-ADR argv would run
    mislabeled duplicates, so 'run' refuses it and the operator re-plans:

    - ADR-0102: a server-engine cell carries --reset-cache-between-trials
      and --warmup-pool-queries == WARMUP_POOL_QUERIES (the VALUE, not just
      the flag).
    - ADR-0103 and ADR-0150: the cell's serving record and the relaunch it
      runs under agree with ``_prefix_off`` (the one prefix rule): prefix_mode
      and the launcher's --no-prefix-cache.
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
    - Batch 2 W4 (ADR-0117), given the plan header's ``calibration.floors``
      (``slo_floors_env``): EVERY cell env carries SLO_FLOORS_ENV equal to
      the header's canonical spelling (a pre-W4 plan, or a hand-edited
      floor, would create a manifest whose SLO pair the analysis cannot
      trust); an executable server cell carries the ``budget_plan`` record
      of the relaunch it runs under plus BUDGET_PLAN_ENV when that relaunch
      is budgeted, and neither when it is budget-free (a cell moved under
      another budget boundary of the same engine passes the endpoint and
      prefix clauses and is caught here); hf and blocked cells carry neither.
    - S0F-26 (ADR-0136): every server-engine cell (blocked ones included)
      carries TELEMETRY_FLAG exactly once and an hf cell never does; a pd
      cell additionally carries the role endpoint pair (S0F-22 Batch 1) and
      no other cell does.
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

    if slo_floors_env is not _UNCHECKED:
        got_floors = env.get(SLO_FLOORS_ENV)
        if slo_floors_env is None:
            if got_floors is not None:
                problems.append(
                    f"{label}: cell {row!r} env {SLO_FLOORS_ENV} is set but the "
                    f"plan registers no floor (no executable server cell) "
                    f"({SLO_FLOORS_FINDING})" + stale
                )
        elif got_floors != slo_floors_env:
            problems.append(
                f"{label}: cell {row!r} env {SLO_FLOORS_ENV} is "
                + ("absent" if got_floors is None else "not the header's calibration.floors")
                + f" ({SLO_FLOORS_FINDING}: the campaign session writes this pin "
                f"into manifest.json[{SLO_FLOORS_MANIFEST_KEY!r}], the #14 SLO "
                "source; the manifest is amended never)" + stale
            )
    # Review F3: the serving record is what every relaunch-agreement clause
    # below keys on; a server cell without one escaped them all.
    serving_record = step.get("serving")
    if spec.engine == "hf":
        if serving_record is not None:
            problems.append(
                f"{label}: hf cell {row!r} carries a serving record (the "
                "in-process oracle has no relaunch boundary)" + stale
            )
    elif not isinstance(serving_record, dict):
        problems.append(
            f"{label}: server cell {row!r} has no serving record "
            f"({serving_record!r}); the relaunch-agreement clauses (ADR-0103, "
            f"{ENGINE_PORTS_FINDING}, {SLO_FLOORS_FINDING}) key on it" + stale
        )
    if spec.engine == "hf" or step.get("blocked_on") is not None:
        problems.extend(_budget_plan_problems(step, None, label))

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
    # ADR-0154/0155: a pressure cell's offered rate is rate_frac x a lambda*
    # measured on the rung (gold class) or derived from it by the served-token
    # ratio (every other class); a blocked cell carries the labeled P6
    # prediction it never runs at. The floor table's pending basis on an
    # executable cell is the 2026-10-08 landing's shape and refuses.
    lambda_star = step.get("lambda_star_rps")
    rate_basis = step.get("rate_basis")
    demand_class = step.get("demand_class")
    if spec.family in _PRESSURE_FAMILIES:
        if step.get("blocked_on") is None:
            if rate_basis not in (LAMBDA_BASIS_RUNG, LAMBDA_BASIS_SMALL, LAMBDA_BASIS_INTERPOLATED):
                problems.append(
                    f"{label}: pressure cell {row!r} rate_basis is {rate_basis!r}; an "
                    "executable cell offers a rung-calibrated lambda* (measured on an "
                    f"anchor or interpolated between the two, {RUNG_CALIBRATION_ADR}, "
                    f"{TWO_ANCHOR_ADR}: the KV-bound prediction ran 3 to 6 times below "
                    "capacity on the landing)" + stale
                )
            if not isinstance(step.get("lambda_star_source"), dict):
                problems.append(
                    f"{label}: pressure cell {row!r} carries no lambda_star_source "
                    f"({RUNG_CALIBRATION_ADR}: the rung artifact the rate rests on)" + stale
                )
        if (
            isinstance(lambda_star, bool)
            or not isinstance(lambda_star, (int, float))
            or not math.isfinite(lambda_star)
            or lambda_star <= 0
        ):
            problems.append(
                f"{label}: pressure cell {row!r} lambda_star_rps={lambda_star!r} must be a "
                f"finite number > 0 ({RUNG_CALIBRATION_ADR})" + stale
            )
        else:
            want_offered = float(spec.rate_frac or 0.0) * float(lambda_star)
            offered = step.get("offered_rate_rps")
            got_rate = _argv_flag_value(argv, "--rate")
            if not isinstance(offered, (int, float)) or isinstance(offered, bool) or not math.isclose(float(offered), want_offered, rel_tol=1e-9, abs_tol=1e-12):
                problems.append(
                    f"{label}: pressure cell {row!r} offered_rate_rps={offered!r} != "
                    f"rate_frac x lambda_star_rps = {want_offered:.6g} ({RUNG_CALIBRATION_ADR})" + stale
                )
            try:
                rate_ok = got_rate is not None and math.isclose(float(got_rate), want_offered, rel_tol=1e-5)
            except ValueError:
                rate_ok = False
            if not rate_ok:
                problems.append(
                    f"{label}: pressure cell {row!r} carries --rate {got_rate!r}, the offered "
                    f"rate is {want_offered:.6g} ({RUNG_CALIBRATION_ADR})" + stale
                )
            # ADR-0156 / S0F-68: every pressure cell records W / offered rate
            # and its floor verdict (a blocked cell on its labeled rate too),
            # so both are re-derived here.
            got_span = step.get("window_span_s_expected")
            got_n = step.get("num_queries")
            if isinstance(got_n, int) and not isinstance(got_n, bool) and want_offered > 0:
                want_span = got_n / want_offered
                if not isinstance(got_span, (int, float)) or isinstance(got_span, bool) or not math.isclose(float(got_span), want_span, rel_tol=1e-6):
                    problems.append(
                        f"{label}: pressure cell {row!r} window_span_s_expected={got_span!r} != "
                        f"num_queries / offered rate = {want_span:.4g} s ({WINDOW_SPAN_FINDING})" + stale
                    )
                if step.get("window_span_below_floor") is not bool(want_span < WINDOW_SPAN_FLOOR_S):
                    problems.append(
                        f"{label}: pressure cell {row!r} window_span_below_floor="
                        f"{step.get('window_span_below_floor')!r} disagrees with the registered "
                        f"floor {WINDOW_SPAN_FLOOR_S:g} s ({WINDOW_SPAN_FINDING})" + stale
                    )
        seq = demand_class.get("seq_tokens") if isinstance(demand_class, dict) else None
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            problems.append(
                f"{label}: pressure cell {row!r} carries demand_class={demand_class!r}, must "
                f"name its served-token class ({DEMAND_CLASS_ADR})" + stale
            )
    else:
        if lambda_star is not None or rate_basis is not None or step.get("lambda_star_source") is not None:
            problems.append(
                f"{label}: cell {row!r} (family {spec.family!r}) carries a rate basis "
                f"(lambda_star_rps={lambda_star!r}, rate_basis={rate_basis!r}); only "
                "pressure cells offer a rate" + stale
            )
        if step.get("window_span_s_expected") is not None or step.get("window_span_below_floor") is not None:
            problems.append(
                f"{label}: cell {row!r} (family {spec.family!r}) carries a window span "
                f"({WINDOW_SPAN_FINDING}: only pressure cells have one)" + stale
            )
        budgeted_cell = spec.budget_r is not None or spec.topology in ("tp", "pd")
        if budgeted_cell and not isinstance(demand_class, dict):
            problems.append(
                f"{label}: budgeted cell {row!r} carries no demand_class ({DEMAND_CLASS_ADR})" + stale
            )
        if not budgeted_cell and demand_class is not None:
            problems.append(
                f"{label}: budget-free cell {row!r} carries demand_class={demand_class!r} "
                f"({DEMAND_CLASS_ADR}: absence stays absence)" + stale
            )

    # V3 (window bound): a window cell (pressure family, or the RULER
    # instrument) is bounded by --arrival-count == its --num-queries and
    # never by --duration-s; every other cell carries neither flag.
    got_arrivals = _argv_flag_value(argv, "--arrival-count")
    if spec.family in _PRESSURE_FAMILIES or step.get("dataset") == "ruler":
        if "--duration-s" in argv:
            problems.append(
                f"{label}: window cell {row!r} carries --duration-s "
                f"({WINDOW_BOUND_FINDING}: a duration-bound window refuses at the "
                "runner's replay guard once rate x duration exceeds the prepared "
                "pool; the window is W arrivals)" + stale
            )
        if got_arrivals is None:
            problems.append(
                f"{label}: window cell {row!r} lacks --arrival-count "
                f"({WINDOW_BOUND_FINDING}: W requests issued per window, one per "
                "prepared request)" + stale
            )
        elif got_arrivals != str(step.get("num_queries")):
            problems.append(
                f"{label}: window cell {row!r} carries --arrival-count "
                f"{got_arrivals!r} but its pool (--num-queries) is "
                f"{step.get('num_queries')!r} ({WINDOW_BOUND_FINDING}: arrivals "
                "beyond the pool replay; fewer under-measure the window)" + stale
            )
    else:
        for flag in ("--arrival-count", "--duration-s"):
            if flag in argv:
                problems.append(
                    f"{label}: cell {row!r} (family {spec.family!r}) carries {flag} "
                    f"({WINDOW_BOUND_FINDING}: the window bound rides window cells "
                    "only)" + stale
                )
    # V1 (RULER sizing): a RULER step carries the registered haystack target
    # and the rendered input cap; no other cell carries the cap.
    got_cap = _argv_flag_value(argv, "--ruler-rendered-input-cap")
    if step.get("dataset") == "ruler":
        got_haystack = _argv_flag_value(argv, "--ruler-context-tokens")
        if got_haystack != str(RULER_HAYSTACK_TOKENS):
            problems.append(
                f"{label}: RULER cell {row!r} carries --ruler-context-tokens "
                f"{got_haystack!r}, the registered haystack is {RULER_HAYSTACK_TOKENS} "
                f"= {RULER_CONTEXT_TOKENS} - {RULER_WRAPPER_ALLOWANCE} "
                f"({RULER_SIZING_FINDING}: the rendered request must fit the input "
                "shape)" + stale
            )
        if got_cap is None:
            problems.append(
                f"{label}: RULER cell {row!r} lacks --ruler-rendered-input-cap "
                f"({RULER_SIZING_FINDING}: the runner refuses the cell when a rendered "
                "prompt exceeds it)" + stale
            )
        elif got_cap != str(RULER_CONTEXT_TOKENS):
            problems.append(
                f"{label}: RULER cell {row!r} carries --ruler-rendered-input-cap "
                f"{got_cap!r}, the registered input shape is {RULER_CONTEXT_TOKENS} "
                f"({RULER_SIZING_FINDING})" + stale
            )
    elif got_cap is not None:
        problems.append(
            f"{label}: cell {row!r} (dataset {step.get('dataset')!r}) carries "
            f"--ruler-rendered-input-cap ({RULER_SIZING_FINDING}: the cap rides RULER "
            "cells only)" + stale
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

    # S0F-26 (ADR-0136): the telemetry flag rides every server-engine cell
    # exactly once and never an hf cell.
    n_flag = list(argv).count(TELEMETRY_FLAG)
    if spec.engine in SERVER_ENGINES:
        if n_flag != 1:
            problems.append(
                f"{label}: server-engine cell {row!r} carries {TELEMETRY_FLAG} "
                f"{n_flag} time(s), must be exactly once ({TELEMETRY_FINDING}, "
                f"{TELEMETRY_ADR}: without the sampler its windows read "
                "UNKNOWN_TELEMETRY, and on a pd cell the decode role could not "
                "be scraped)" + stale
            )
    elif n_flag:
        problems.append(
            f"{label}: cell {row!r} (engine {spec.engine!r}) carries "
            f"{TELEMETRY_FLAG} ({TELEMETRY_FINDING}, {TELEMETRY_ADR}: the "
            "in-process oracle serves no /metrics; a sampler would record "
            "whatever engine listens on the runner's default port)" + stale
        )
    # S0F-22 Batch 1: a pd cell carries the role endpoint pair (the registered
    # value); every other cell carries the env NOT at all (the runner would
    # sample whatever it named under a single server).
    got_endpoints = env.get(PD_TELEMETRY_ENDPOINTS_ENV)
    if spec.topology == "pd":
        if got_endpoints != PD_TELEMETRY_ENDPOINTS:
            problems.append(
                f"{label}: pd cell {row!r} env {PD_TELEMETRY_ENDPOINTS_ENV} is "
                f"{got_endpoints!r}, the registered role pair is "
                f"{PD_TELEMETRY_ENDPOINTS!r} ({PD_TELEMETRY_FINDING})" + stale
            )
    elif got_endpoints is not None:
        problems.append(
            f"{label}: cell {row!r} (topology {spec.topology!r}) carries env "
            f"{PD_TELEMETRY_ENDPOINTS_ENV} ({PD_TELEMETRY_FINDING}: the role pair "
            "rides pd cells only)" + stale
        )

    expected_mode = "OFF" if _prefix_off(spec) else "ON"
    serving = step.get("serving")
    if isinstance(serving, dict):
        if serving.get("prefix_mode") != expected_mode:
            problems.append(
                f"{label}: cell {row!r} serving.prefix_mode is "
                f"{serving.get('prefix_mode')!r} but the one prefix rule "
                f"(_prefix_off, ADR-0103 and ADR-0150) says {expected_mode}" + stale
            )
        if step.get("blocked_on") is None:
            if preceding_relaunch is None:
                problems.append(
                    f"{label}: executable server cell {row!r} has no relaunch "
                    "step before it (ADR-0103 and ADR-0150: the serving config is a relaunch "
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
                        + " --no-prefix-cache (ADR-0103 and ADR-0150)" + stale
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
                # Batch 2 W4: the budget record IS the relaunch's.
                problems.extend(
                    _budget_plan_problems(
                        step, preceding_relaunch.get("budget_plan"), label
                    )
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
    pins: RetrievalPins,
    query_manifest: Optional[str] = None,
    *,
    slo_floors_env: Optional[str],
    budget_plan: Optional[Dict[str, Any]] = None,
    rungs: Optional[Mapping[str, "RungCalibration"]] = None,
    ladder: bool = False,
) -> Dict[str, Any]:
    """One cell step. ``rungs`` maps engine -> its RungCalibration (ADR-0154):
    an executable pressure cell offers rate_frac x lambda*(engine, r) from it,
    scaled to its demand class (ADR-0155); ``ladder=True`` builds the
    calibrate-rungs ladder cell instead (rate and arrivals are placeholders
    the ladder sets per step, so no lambda* is needed)."""
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
    if spec.engine in SERVER_ENGINES:
        # S0F-26 (ADR-0136): every server-engine cell samples its engine (the
        # single-endpoint sampler dials --api-base); the hf oracle never
        # does. A pd cell samples BOTH role instances through the endpoint
        # pair its env carries below (S0F-22: the regime pd lane, and the
        # Batch 2 transfer proof reads the decode role).
        argv.append(TELEMETRY_FLAG)
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
            # V1: the haystack target is the input shape minus the registered
            # wrapper allowance; the cap is what the RENDERED request must fit
            # (the runner renders every prompt and refuses the cell otherwise).
            "--ruler-context-tokens",
            str(RULER_HAYSTACK_TOKENS),
            "--ruler-rendered-input-cap",
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
    window_span: Optional[float] = None
    lambda_star: Optional[float] = None
    lambda_source: Optional[Dict[str, Any]] = None
    rate_basis: Optional[str] = None
    lambda_kv_pred: Optional[float] = None
    lambda_kv_pred_basis: Optional[str] = None
    # ADR-0155: the demand class the cell's serving budget was sized on, the
    # SAME record its relaunch carries (None on budget-free cells: absence
    # stays absence). A blocked cell on a rung the floor table lacks (it
    # never runs) records the class with no byte figure.
    demand_class: Optional[Dict[str, Any]] = None
    if _is_budgeted(spec):
        class_tokens = demand_seq_tokens(grid, spec)
        class_anchor = int(grid.demand_seq_tokens[DEMAND_ANCHOR_ARM])
        class_r = spec.budget_r if spec.budget_r is not None else grid.dist_budget_r
        try:
            class_demand: Optional[int] = _class_demand_bytes(
                int(floor.row(class_r)["demand_bytes"]), class_tokens, class_anchor
            )
        except PlanError:
            if cell.blocked_on is None:
                raise
            class_demand = None
        demand_class = {
            "seq_tokens": class_tokens,
            "anchor_seq_tokens": class_anchor,
            "demand_bytes": class_demand,
            "adr": DEMAND_CLASS_ADR,
        }
    if spec.family in _PRESSURE_FAMILIES:
        assert spec.budget_r is not None and spec.rate_frac is not None
        assert demand_class is not None
        row = floor.row(spec.budget_r)
        # The floor table's KV-bound rate is the P6 PREDICTION, recorded
        # beside the offered rate for the ±15 percent falsification test and
        # never offered (ADR-0154: the landing offered it, 3 to 6 times below
        # the measured capacity).
        lambda_kv_pred = float(row["lambda_star_pred_rps"])
        lambda_kv_pred_basis = str(row.get("lambda_star_basis") or _LAMBDA_PENDING_BASIS)
        seq_tokens = int(demand_class["seq_tokens"])
        anchor_tokens = int(grid.demand_seq_tokens[DEMAND_ANCHOR_ARM])
        if ladder:
            # ADR-0154: calibrate-rungs sets --rate per ladder step. The window
            # is the registered PROBE_WINDOW_S in DURATION mode with the
            # runner's Jain trim at PROBE_WARMUP_S, the cal-v2 probe's own
            # shape: the V3 pool guard refuses an arrival count above the
            # prepared pool, and a 75 s window at the rates a rung reaches
            # needs more arrivals than the pool holds, so the ladder replays
            # the pool (LADDER_REPLAY_ENV; non-confirmatory by registration,
            # exactly as calibrate_cell.probe_rate wraps its prepared set).
            rate_basis = "ladder step (calibrate-rungs sets the rate)"
            # One window per ladder step (the probe is one rate, one window).
            # No --open-loop-warmup-s: the runner REFUSES it in campaign mode
            # (ADR-0055 amendment 2026-09-19, run_experiment.py); the Jain
            # trim at PROBE_WARMUP_S is applied by _read_ladder_window on the
            # intended arrival instead (review CRITICAL 1, 2026-10-09).
            argv[argv.index("--num-trials") + 1] = "1"
            argv += [
                "--workload-mode", "open_loop", "--rate", "0",
                "--duration-s", f"{PROBE_WINDOW_S:g}",
            ]
        else:
            cal = (rungs or {}).get(spec.engine)
            try:
                resolved = (
                    rung_lambda_for_class(cal, spec.budget_r, seq_tokens, anchor_tokens)
                    if cal is not None
                    else None
                )
            except PlanError:
                # Fable review 2026-10-09 LOW 3: a class outside the two anchors
                # refuses an EXECUTABLE cell (never extrapolate); a blocked one
                # never runs and takes the P6 line below like any blocked cell.
                if cell.blocked_on is None:
                    raise
                resolved = None
            if resolved is None:
                if cell.blocked_on is None:
                    raise PlanError(
                        f"cell {spec.to_row_key()} is executable but no rung "
                        f"calibration covers ({spec.engine}, r={spec.budget_r:g}, class "
                        f"{seq_tokens} tokens) (_register_rung_calibrations should have "
                        "refused; driver invariant violated)"
                    )
                # A BLOCKED cell never runs: its rate is the P6 prediction,
                # labeled as such, so the plan stays reviewable (debt visible).
                lambda_star = lambda_kv_pred
                rate_basis = f"blocked cell, never run: {lambda_kv_pred_basis}"
            else:
                lambda_star, rate_basis, lambda_source = resolved
            offered_rate = spec.rate_frac * lambda_star
            # V3: the window is W arrivals, one per prepared request (the
            # --num-queries above), never a duration: the open-loop generator's
            # replay guard refuses a schedule longer than the pool, and a
            # duration-bound window at a real rate always was. The pre-costed
            # window_duration_s stays in the header as the cost estimate only.
            argv += [
                "--workload-mode",
                "open_loop",
                "--rate",
                f"{offered_rate:.6g}",
                "--arrival-count",
                str(num_queries),
            ]
            # ADR-0156 / S0F-68: the span this W-arrival window is expected to
            # cover at its offered rate, recorded beside the floor verdict.
            window_span = num_queries / offered_rate
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
    if spec.topology == "pd":
        # S0F-22 Batch 1: the SAME value the pd relaunch carries (one table);
        # the runner refuses the env without PD_TELEMETRY_FLAG, so both ride
        # the cell together. Behavior, never identity.
        env[PD_TELEMETRY_ENDPOINTS_ENV] = PD_TELEMETRY_ENDPOINTS
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
    # Batch 2 W4 (ADR-0117): the §6.1 floors ride EVERY cell (hf and blocked
    # included: the manifest is created by whichever cell emits first and is
    # amended never); the relaunch's BudgetPlan record rides BUDGETED cells
    # only (absence stays absence for F1, hf and blocked cells). Provenance,
    # never identity: derive_cell_spec ignores both.
    if slo_floors_env is not None:
        env[SLO_FLOORS_ENV] = slo_floors_env
    if budget_plan is not None:
        env[BUDGET_PLAN_ENV] = json.dumps(
            budget_plan, sort_keys=True, separators=(",", ":")
        )
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
        # ADR-0154: the lambda* the offered rate is a fraction of (measured on
        # the gold rung, scaled to the class), its provenance, and the floor
        # table's P6 prediction recorded beside it (never offered).
        "lambda_star_rps": lambda_star,
        "lambda_star_source": lambda_source,
        "lambda_kv_pred_rps": lambda_kv_pred,
        "lambda_kv_pred_basis": lambda_kv_pred_basis,
        "rate_basis": rate_basis,
        # ADR-0156 / S0F-68: W / offered rate on a pressure cell (null
        # elsewhere, the ladder cell included: its rate is set per step) and
        # whether it sits below WINDOW_SPAN_FLOOR_S (null when no span).
        "window_span_s_expected": window_span,
        "window_span_below_floor": None if window_span is None else bool(window_span < WINDOW_SPAN_FLOOR_S),
        # ADR-0155: the demand class (served sequence tokens) of a budgeted cell.
        "demand_class": demand_class,
        # Batch 2 W4: the record of the relaunch this cell runs under (the
        # same object; null on hf, blocked and budget-free cells).
        "budget_plan": budget_plan,
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


def _session_engines(grid: SessionGrid) -> FrozenSet[str]:
    """Every server engine a grid registers cells on (hf serves no floor)."""
    engines = set(grid.f1_engines) | set(grid.f2_engines) | set(grid.f3_engines)
    engines |= {engine for _bid, engine, _topology in grid.dist_cells}
    return frozenset(engines - {"hf"})


def _register_calibrations(
    grid: SessionGrid,
    calibrations: Optional[Mapping[str, Path]],
    cells: Sequence[PlannedCell],
    budget_fraction: float,
) -> Dict[str, Any]:
    """Validate the operator's per-engine cal-v1 registration (Batch 2 W4)
    and return the plan-header ``calibration`` record.

    Every EXECUTABLE server cell's engine needs a floor: the manifest the
    first emitting cell creates must carry every engine the run serves, and
    it is amended never. An engine no cell of the session registers refuses;
    an engine whose cells are all blocked may carry one (recorded, unused).
    Each artifact must describe the engine it is registered under, the
    session's model (HF id or charter slug), and the ONE floor rung this plan
    registers (``budget_fraction``: the charter's r = 1.5 unless the
    operator passed --calibration-budget-fraction).
    """
    if (
        isinstance(budget_fraction, bool)
        or not isinstance(budget_fraction, (int, float))
        or not math.isfinite(budget_fraction)
        or budget_fraction <= 0
    ):
        raise PlanError(
            f"--calibration-budget-fraction {budget_fraction!r} must be finite and > 0"
        )
    registered = {str(k): Path(v) for k, v in (calibrations or {}).items()}
    session_engines = _session_engines(grid)
    required = sorted(
        {c.spec.engine for c in cells if c.blocked_on is None and c.spec.engine != "hf"}
    )
    fix = (
        "run scripts/3_run/calibrate_cell.py --budget-fraction "
        f"{budget_fraction:g} --output <path> against each engine's server, then "
        f"plan --calibration <engine>=<path> per engine ({SLO_FLOORS_FINDING}, "
        f"{SLO_FLOORS_ADR})"
    )
    unknown = sorted(set(registered) - session_engines)
    if unknown:
        raise PlanError(
            f"--calibration {unknown}: not a server engine of session "
            f"{grid.session!r} (registered engines: {sorted(session_engines)})"
        )
    missing = sorted(set(required) - set(registered))
    if missing:
        raise PlanError(
            f"calibration missing for engine(s) {missing}: every executable "
            "server cell's engine needs its §6.1 single-stream floor before the "
            "plan is built (the first emitting cell writes them into "
            f"manifest.json[{SLO_FLOORS_MANIFEST_KEY!r}], amended never); {fix}"
        )
    accepted_models = (grid.model, HF_ID_OF_SLUG[grid.model])
    floors: Dict[str, Dict[str, Any]] = {}
    artifacts: Dict[str, Dict[str, Any]] = {}
    for engine in sorted(registered):
        cal = load_calibration(registered[engine], budget_fraction)
        if cal.engine != engine:
            raise PlanError(
                f"--calibration {engine}={cal.path}: the artifact describes engine "
                f"{cal.engine!r} (its engine field normalizes to that), not "
                f"{engine!r}; refusing a swapped floor"
            )
        if cal.model not in accepted_models:
            raise PlanError(
                f"--calibration {engine}={cal.path}: the artifact is for model "
                f"{cal.model!r} but session {grid.session!r} runs {grid.model!r} "
                f"({HF_ID_OF_SLUG[grid.model]}); a floor from another model would "
                "mis-set every SLO"
            )
        if not math.isclose(cal.budget_fraction, budget_fraction, rel_tol=0.0, abs_tol=1e-9):
            raise PlanError(
                f"--calibration {engine}={cal.path}: the artifact's budget_fraction "
                f"is {cal.budget_fraction:g} but this plan registers its floors at "
                f"r = {budget_fraction:g} (charter §6.1: the floor is measured at "
                f"r = {FLOOR_BUDGET_FRACTION:g}, concurrency 1; pass "
                f"--calibration-budget-fraction {cal.budget_fraction:g} to register "
                "another rung explicitly, and the same rung for every engine)"
            )
        floors[engine] = {
            "ttft_s": cal.ttft_s,
            "tpot_s": cal.tpot_s,
            "n_requests": cal.n_requests,
            "statistic": cal.statistic,
            "budget_fraction": cal.budget_fraction,
            "source_sha256": cal.sha256,
        }
        artifacts[engine] = {
            "path": str(cal.path.resolve()),
            "sha256": cal.sha256,
            "model": cal.model,
            "procedure_version": cal.procedure_version,
            "lambda_star_label": cal.lambda_star_label,
        }
    return {
        "procedure_version": PROCEDURE_VERSION,
        "budget_fraction": float(budget_fraction),
        "registered_budget_fraction": FLOOR_BUDGET_FRACTION,
        "floors": floors,
        "artifacts": artifacts,
        "env": SLO_FLOORS_ENV,
        "manifest_key": SLO_FLOORS_MANIFEST_KEY,
        "finding": SLO_FLOORS_FINDING,
        "adr": SLO_FLOORS_ADR,
        "charter": (
            "6.1: the primary SLO pair is TTFT <= 10x and TPOT <= 5x the "
            "single-stream floor of the same model x engine, measured at "
            "r = 1.5 and concurrency 1 (calibration.summarize_floor)"
        ),
    }


# ---------------------------------------------------------------------------
# Rung calibration artifacts (ADR-0154): lambda* per budget rung, measured by
# ``calibrate-rungs`` on the gold-fresh F2 workload under the plan's own
# relaunch for that rung
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RungCalibration:
    """Validated ``cage-rung-calibration-v2`` artifact of ONE engine: the
    measured lambda* per budget rung of the ANCHOR class (gold-fresh; every
    rung's label is kept so a refusal can name LADDER_EXHAUSTED or
    NONE_SUSTAINABLE) and, when the engine serves a class below the anchor,
    the same table for its SMALLEST executable class (ADR-0156: the second
    anchor of the log-log interpolation), plus the provenance the plan header
    records. ``small_*`` stay empty on a one-class artifact."""

    path: Path
    sha256: str
    engine: str
    model: str
    session: str
    lambdas: Dict[float, float]
    labels: Dict[float, str]
    anchor_seq_tokens: Optional[int] = None
    small_seq_tokens: Optional[int] = None
    small_arm: Optional[str] = None
    small_prefix_mode: Optional[str] = None
    small_lambdas: Dict[float, float] = field(default_factory=dict)
    small_labels: Dict[float, str] = field(default_factory=dict)

    @staticmethod
    def _at(table: Mapping[float, Any], r: float) -> Optional[Any]:
        for key, value in table.items():
            if math.isclose(key, r, rel_tol=0.0, abs_tol=1e-9):
                return value
        return None

    def lambda_at(self, r: float) -> Optional[float]:
        return self._at(self.lambdas, r)

    def label_at(self, r: float) -> Optional[str]:
        return self._at(self.labels, r)

    def small_lambda_at(self, r: float) -> Optional[float]:
        return self._at(self.small_lambdas, r)

    def small_label_at(self, r: float) -> Optional[str]:
        return self._at(self.small_labels, r)


def _parse_rung_table(
    rungs: Any, where: str, problems: List[str]
) -> Tuple[Dict[float, float], Dict[float, str]]:
    """One ``rungs`` table (r -> rung record) of the artifact: keys parse as
    the rung r, ``r`` agrees with its key, every label is a non-empty string
    and ONLY an ESTIMATED rung carries a finite lambda_star_qps > 0."""
    lambdas: Dict[float, float] = {}
    labels: Dict[float, str] = {}
    if not isinstance(rungs, dict) or not rungs:
        problems.append(f"{where} must be a non-empty mapping r -> rung record")
        return lambdas, labels
    for key, rec in rungs.items():
        try:
            r = float(key)
        except (TypeError, ValueError):
            problems.append(f"{where} key {key!r} does not parse as a budget ratio")
            continue
        if not isinstance(rec, dict):
            problems.append(f"{where}[{key!r}] is not an object")
            continue
        if not isinstance(rec.get("r"), (int, float)) or isinstance(rec.get("r"), bool) or not math.isclose(float(rec["r"]), r, rel_tol=0.0, abs_tol=1e-9):
            problems.append(f"{where}[{key!r}].r={rec.get('r')!r} disagrees with its key")
        label = rec.get("label")
        if not isinstance(label, str) or not label:
            problems.append(f"{where}[{key!r}].label={label!r} must be a non-empty string")
            continue
        labels[r] = label
        value = rec.get("lambda_star_qps")
        if label == "ESTIMATED":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                problems.append(
                    f"{where}[{key!r}] is ESTIMATED but lambda_star_qps={value!r} is not a finite number > 0"
                )
            else:
                lambdas[r] = float(value)
        elif value is not None:
            problems.append(
                f"{where}[{key!r}] has label {label!r} but carries lambda_star_qps={value!r} "
                "(labels are honest: only ESTIMATED carries a value)"
            )
    return lambdas, labels


def load_rung_calibration(path: Path) -> RungCalibration:
    """Load + validate ONE rung-calibration artifact (fail closed): the
    schema (v2; a v1 artifact predates the second anchor and refuses), the
    cal-v2 procedure version, ``confirmatory: false``, a runner engine id
    (normalized through ENGINE_OF_BACKEND, the oracle refused), the model and
    session, the anchor ``rungs`` table, and the optional ``classes`` block
    (ADR-0156): one ``anchor`` entry whose table equals ``rungs`` and whose
    ``seq_tokens`` is the anchor shape, and at most one ``smallest`` entry
    (arm, prefix mode, a shape below the anchor, its own rungs table).
    """
    path = Path(path)
    if not path.is_file():
        raise PlanError(
            f"rung calibration artifact not found: {path} (run scripts/3_run/"
            "run_campaign.py calibrate-rungs --session <s> --engine <e> ... per "
            f"engine; {RUNG_CALIBRATION_FINDING}, {RUNG_CALIBRATION_ADR})"
        )
    raw = path.read_bytes()
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PlanError(f"rung calibration artifact {path} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != RUNG_CALIBRATION_SCHEMA:
        raise PlanError(
            f"rung calibration artifact {path} schema is "
            f"{doc.get('schema') if isinstance(doc, dict) else type(doc).__name__!r}, "
            f"expected {RUNG_CALIBRATION_SCHEMA!r} (a v1 artifact predates the second "
            f"anchor of {TWO_ANCHOR_ADR}: re-run calibrate-rungs)"
        )
    problems: List[str] = []
    if doc.get("procedure_version") != PROCEDURE_VERSION:
        problems.append(
            f"procedure_version is {doc.get('procedure_version')!r}, the registered "
            f"procedure is {PROCEDURE_VERSION!r}"
        )
    if doc.get("confirmatory") is not False:
        problems.append("confirmatory must be the literal false")
    if doc.get("complete") is False:
        # ADR-0162: a per-rung partial write of a calibrate-rungs job that never
        # finished (killed at its bound); its rungs are forensics, never a plan input
        problems.append(
            "the artifact is INCOMPLETE (complete: false): calibrate-rungs did not finish; "
            "re-run it (a killed job leaves this partial record for forensics)"
        )
    raw_engine = doc.get("engine")
    engine = ENGINE_OF_BACKEND.get(raw_engine) if isinstance(raw_engine, str) else None
    if engine is None:
        problems.append(f"engine {raw_engine!r} is not a runner backend ({sorted(ENGINE_OF_BACKEND)})")
    elif engine == "hf":
        problems.append("engine is the in-process oracle, which serves no pressure rung")
    model = doc.get("model")
    if not isinstance(model, str) or not model.strip():
        problems.append(f"model {model!r} must be a non-empty string")
    session = doc.get("session")
    if not isinstance(session, str) or session not in SESSIONS:
        problems.append(f"session {session!r} is not a §1 session ({sorted(SESSIONS)})")
    lambdas, labels = _parse_rung_table(doc.get("rungs"), "rungs", problems)
    anchor_tokens = doc.get("anchor_seq_tokens")
    if anchor_tokens is not None and (
        isinstance(anchor_tokens, bool) or not isinstance(anchor_tokens, int) or anchor_tokens < 1
    ):
        problems.append(f"anchor_seq_tokens={anchor_tokens!r} must be an integer >= 1")
        anchor_tokens = None
    small_seq: Optional[int] = None
    small_arm: Optional[str] = None
    small_prefix: Optional[str] = None
    small_lambdas: Dict[float, float] = {}
    small_labels: Dict[float, str] = {}
    classes = doc.get("classes")
    if classes is not None:
        if not isinstance(classes, dict):
            problems.append("classes must be a mapping seq_tokens -> class record")
        else:
            roles: Dict[str, str] = {}
            for key, rec in classes.items():
                where = f"classes[{key!r}]"
                if not isinstance(rec, dict):
                    problems.append(f"{where} is not an object")
                    continue
                role = rec.get("role")
                if role not in ("anchor", "smallest"):
                    problems.append(f"{where}.role={role!r} must be 'anchor' or 'smallest'")
                    continue
                if role in roles:
                    problems.append(f"classes carries two {role!r} entries ({roles[role]!r}, {key!r})")
                    continue
                roles[role] = str(key)
                seq = rec.get("seq_tokens")
                if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1 or str(seq) != str(key):
                    problems.append(f"{where}.seq_tokens={seq!r} must be the integer >= 1 the key names")
                    seq = None
                c_lambdas, c_labels = _parse_rung_table(rec.get("rungs"), f"{where}.rungs", problems)
                if role == "anchor":
                    if (c_lambdas, c_labels) != (lambdas, labels):
                        problems.append(f"{where} (the anchor) differs from the top-level rungs table")
                    if seq is not None:
                        if anchor_tokens is None:
                            anchor_tokens = seq
                        elif seq != anchor_tokens:
                            problems.append(
                                f"{where}.seq_tokens={seq} disagrees with anchor_seq_tokens={anchor_tokens}"
                            )
                else:
                    arm = rec.get("arm")
                    if not isinstance(arm, str) or not arm.strip():
                        problems.append(f"{where}.arm={arm!r} must name the smallest class's arm")
                    prefix = rec.get("prefix_mode")
                    if prefix not in ("ON", "OFF"):
                        problems.append(f"{where}.prefix_mode={prefix!r} must be 'ON' or 'OFF'")
                    small_seq, small_arm, small_prefix = seq, (arm if isinstance(arm, str) else None), (prefix if isinstance(prefix, str) else None)
                    small_lambdas, small_labels = c_lambdas, c_labels
    if small_seq is not None and anchor_tokens is not None and small_seq >= anchor_tokens:
        problems.append(
            f"the smallest class ({small_seq} tokens) is not below the anchor ({anchor_tokens} tokens) "
            f"({TWO_ANCHOR_ADR}: the anchors are the largest and the smallest class)"
        )
    if problems:
        raise PlanError(f"rung calibration artifact {path}: " + "; ".join(problems))
    assert engine is not None and isinstance(model, str) and isinstance(session, str)
    return RungCalibration(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        engine=engine,
        model=model,
        session=session,
        lambdas=lambdas,
        labels=labels,
        anchor_seq_tokens=anchor_tokens,
        small_seq_tokens=small_seq,
        small_arm=small_arm,
        small_prefix_mode=small_prefix,
        small_lambdas=small_lambdas,
        small_labels=small_labels,
    )


def interpolation_alpha(lambda_gold: float, s_gold: int, lambda_small: float, s_small: int) -> float:
    """ADR-0156: the log-log slope through the two anchors,
    alpha = ln(lambda_m / lambda_g) / ln(s_g / s_m); 1 is the KV-bound limit.
    Refuses (PlanError) non-positive rates or a small class not below the anchor."""
    if not (lambda_gold > 0 and lambda_small > 0 and math.isfinite(lambda_gold) and math.isfinite(lambda_small)):
        raise PlanError(f"interpolation needs two positive finite rates, got {lambda_gold!r} and {lambda_small!r}")
    if not (0 < s_small < s_gold):
        raise PlanError(f"interpolation needs 0 < s_small={s_small} < s_gold={s_gold} ({TWO_ANCHOR_ADR})")
    return math.log(lambda_small / lambda_gold) / math.log(s_gold / s_small)


def alpha_tolerance(s_gold: int, s_small: int) -> float:
    """Fable review 2026-10-09 HIGH 1: the two ladders resolve lambda* to one
    bisected step, (PROBE_LADDER_FACTOR - 1) / 2**PROBE_BISECT_STEPS of the
    rate (7.5 percent at 1.3 and 2 steps). Two classes with the SAME capacity
    (the scheduler-bound regime at loose r, S0F-70) can therefore read a
    small-class lambda* one step BELOW the anchor's by measurement
    resolution alone, a slope of ln(1 / (1 + step)) / ln(s_g / s_m). This is
    the magnitude of that slope: alpha in [-tol, 0) is measurement noise
    around a flat line and clamps to 0; below -tol one ladder is defective."""
    step = (PROBE_LADDER_FACTOR - 1.0) / (2 ** PROBE_BISECT_STEPS)
    return math.log(1.0 + step) / math.log(s_gold / s_small)


def resolve_alpha(
    lambda_gold: float, s_gold: int, lambda_small: float, s_small: int, *, where: str = ""
) -> Tuple[float, float, bool]:
    """(alpha used, alpha raw, clamped): the ADR-0156 slope with the
    resolution rule of ``alpha_tolerance``. Refuses (PlanError) a raw slope
    below -tol: a smaller sequence cannot sustain LESS under the KV, compute
    or request-cap bound beyond the ladders' own resolution, so one ladder is
    defective (the SGLang thinking rows of S0F-59 are one cause)."""
    raw = interpolation_alpha(lambda_gold, s_gold, lambda_small, s_small)
    tol = alpha_tolerance(s_gold, s_small)
    if raw < -tol:
        raise PlanError(
            f"{where}the smallest class ({s_small} tokens) sustained {lambda_small:.4g} rps, "
            f"below the anchor's {lambda_gold:.4g} rps (alpha {raw:.3f} < 0 beyond the ladders' "
            f"resolution, tolerance {tol:.3f}): a smaller sequence cannot sustain less under the "
            "KV, compute or request-cap bound, so one ladder is defective (the SGLang thinking "
            f"rows of S0F-59 are one cause); re-calibrate ({TWO_ANCHOR_ADR})"
        )
    if raw < 0.0:
        return 0.0, raw, True
    return raw, raw, False


def rung_lambda_for_class(
    cal: RungCalibration, r: float, seq_tokens: int, anchor_tokens: int
) -> Optional[Tuple[float, str, Dict[str, Any]]]:
    """The lambda* a pressure cell of ``seq_tokens`` offers a fraction of at
    rung ``r`` (ADR-0154 and ADR-0156): the anchor's measured value on the
    anchor class, the smallest class's measured value on that class, and the
    log-log interpolation between the two anchors for every class in between
    (never outside: PlanError). Returns (lambda*, basis label, source record)
    or None when the artifact lacks the rung the class needs (the caller
    decides: an executable cell refuses, a blocked one records the P6 line)."""
    gold = cal.lambda_at(r)
    if gold is None:
        return None
    source: Dict[str, Any] = {
        "engine": cal.engine,
        "r": r,
        "gold_lambda_star_qps": gold,
        "artifact_sha256": cal.sha256,
        "seq_tokens_gold": anchor_tokens,
        "seq_tokens_class": seq_tokens,
        "adr": RUNG_CALIBRATION_ADR,
    }
    if seq_tokens == anchor_tokens:
        source["alpha"] = None
        return gold, LAMBDA_BASIS_RUNG, source
    small = cal.small_lambda_at(r)
    s_small = cal.small_seq_tokens
    if small is None or s_small is None:
        return None
    alpha, alpha_raw, clamped = resolve_alpha(
        gold, anchor_tokens, small, s_small, where=f"engine {cal.engine!r} at r={r:g}: "
    )
    source.update({
        "small_lambda_star_qps": small,
        "seq_tokens_small": s_small,
        "small_arm": cal.small_arm,
        "alpha": alpha,
        "alpha_raw": alpha_raw,
        "alpha_clamped": clamped,
        "alpha_tolerance": alpha_tolerance(anchor_tokens, s_small),
        "kv_bound_alpha": 1.0,
        "two_anchor_adr": TWO_ANCHOR_ADR,
    })
    if seq_tokens == s_small:
        return small, LAMBDA_BASIS_SMALL, source
    if not (s_small < seq_tokens < anchor_tokens):
        raise PlanError(
            f"class {seq_tokens} tokens lies outside the anchors [{s_small}, {anchor_tokens}] of "
            f"engine {cal.engine!r} at r={r:g}: the interpolation never extrapolates ({TWO_ANCHOR_ADR})"
        )
    return gold * (anchor_tokens / seq_tokens) ** alpha, LAMBDA_BASIS_INTERPOLATED, source


def _register_rung_calibrations(
    grid: SessionGrid,
    cells: Sequence[PlannedCell],
    rung_calibrations: Optional[Mapping[str, Path]],
    anchor_tokens: int,
) -> Tuple[Dict[str, Any], Dict[str, RungCalibration]]:
    """ADR-0154: every EXECUTABLE pressure cell's (engine, r) needs a measured
    anchor lambda* (an ESTIMATED rung in that engine's artifact) before the
    plan is built; a plan whose rates rest on the floor table's assumed service
    time is the 2026-10-08 landing. ADR-0156: an engine serving a demand class
    below the anchor also needs the SMALLEST executable class's ladder (the
    artifact's ``smallest`` entry must be that class), ESTIMATED at every rung
    a non-anchor class is carried at, and the slope alpha through the two
    anchors must be >= 0 at each (a smaller sequence sustaining less is a
    measurement defect, refused). Returns (header record, engine -> artifact)."""
    required: Dict[str, set] = {}
    below_anchor: Dict[str, Dict[int, set]] = {}
    for cell in cells:
        spec = cell.spec
        if cell.blocked_on is None and spec.engine != "hf" and spec.family in _PRESSURE_FAMILIES:
            assert spec.budget_r is not None
            r = float(spec.budget_r)
            required.setdefault(spec.engine, set()).add(r)
            seq = demand_seq_tokens(grid, spec)
            if seq != anchor_tokens:
                below_anchor.setdefault(spec.engine, {}).setdefault(seq, set()).add(r)
    registered = {str(k): Path(v) for k, v in (rung_calibrations or {}).items()}
    fix = (
        f"run scripts/3_run/run_campaign.py calibrate-rungs --session {grid.session} "
        "--engine <engine> --floor-table <table> --calibration <engine cal-v2 floor> "
        "--out <path> on the pod (every engine with executable pressure cells), then "
        f"plan --rung-calibration <engine>=<path> ({RUNG_CALIBRATION_FINDING}, "
        f"{RUNG_CALIBRATION_ADR}, {TWO_ANCHOR_ADR})"
    )
    unknown = sorted(set(registered) - _session_engines(grid))
    if unknown:
        raise PlanError(
            f"--rung-calibration {unknown}: not a server engine of session "
            f"{grid.session!r} (registered engines: {sorted(_session_engines(grid))})"
        )
    missing_engines = sorted(set(required) - set(registered))
    if missing_engines:
        raise PlanError(
            f"rung calibration missing for engine(s) {missing_engines}: the offered "
            "rates of their pressure cells need lambda* measured per budget rung on "
            "the gold-fresh workload (the floor table's KV-bound rate is a prediction, "
            f"never an offered rate since {RUNG_CALIBRATION_ADR}); {fix}"
        )
    accepted_models = (grid.model, HF_ID_OF_SLUG[grid.model])
    artifacts: Dict[str, RungCalibration] = {}
    header_artifacts: Dict[str, Dict[str, Any]] = {}
    for engine in sorted(registered):
        cal = load_rung_calibration(registered[engine])
        where = f"--rung-calibration {engine}={cal.path}"
        if cal.engine != engine:
            raise PlanError(f"{where}: the artifact describes engine {cal.engine!r}; refusing a swapped rung table")
        if cal.model not in accepted_models:
            raise PlanError(
                f"{where}: the artifact is for model {cal.model!r} but session {grid.session!r} runs {grid.model!r}"
            )
        if cal.session != grid.session:
            raise PlanError(
                f"{where}: the artifact was measured for session {cal.session!r}, not "
                f"{grid.session!r} (another grid, another workload)"
            )
        if cal.anchor_seq_tokens is not None and cal.anchor_seq_tokens != anchor_tokens:
            raise PlanError(
                f"{where}: the artifact's anchor class is {cal.anchor_seq_tokens} tokens but the "
                f"session registers {anchor_tokens} for {DEMAND_ANCHOR_ARM!r} ({DEMAND_CLASS_ADR})"
            )
        missing_rungs = [
            r for r in sorted(required.get(engine, ()), reverse=True) if cal.lambda_at(r) is None
        ]
        if missing_rungs:
            named = ", ".join(f"r={r:g} ({cal.label_at(r) or 'absent'})" for r in missing_rungs)
            raise PlanError(
                f"{where}: no ESTIMATED lambda* for {named}; every executable pressure rung "
                "needs one (LADDER_EXHAUSTED: extend the ladder; NONE_SUSTAINABLE: lower the "
                f"start rate; absent: calibrate the rung); {fix}"
            )
        classes = below_anchor.get(engine, {})
        alpha: Dict[float, float] = {}
        alpha_clamped: List[float] = []
        if classes:
            s_min = min(classes)
            if cal.small_seq_tokens is None:
                raise PlanError(
                    f"{where}: the artifact carries no smallest-class ladder but engine {engine!r} "
                    f"serves {len(classes)} demand class(es) below the anchor ({sorted(classes)} tokens); "
                    f"the second anchor is its smallest executable class, {s_min} tokens "
                    f"({TWO_ANCHOR_ADR}); {fix}"
                )
            if cal.small_seq_tokens != s_min:
                raise PlanError(
                    f"{where}: the artifact's smallest class is {cal.small_seq_tokens} tokens "
                    f"({cal.small_arm}) but the smallest executable class of engine {engine!r} on "
                    f"this grid is {s_min} tokens ({TWO_ANCHOR_ADR}: re-calibrate on this grid)"
                )
            rungs_small = sorted(set().union(*classes.values()), reverse=True)
            missing_small = [r for r in rungs_small if cal.small_lambda_at(r) is None]
            if missing_small:
                named = ", ".join(f"r={r:g} ({cal.small_label_at(r) or 'absent'})" for r in missing_small)
                raise PlanError(
                    f"{where}: the smallest class ({s_min} tokens, {cal.small_arm}) has no ESTIMATED "
                    f"lambda* for {named}; every rung a class below the anchor is carried at needs "
                    f"both anchors ({TWO_ANCHOR_ADR}); {fix}"
                )
            # Fable review 2026-10-09 LOW 6: the slope is checked at EVERY rung
            # both anchors are ESTIMATED at (a blocked cell interpolates at any
            # of them), not only where an executable class is carried.
            rungs_check = sorted(
                set(rungs_small) | {r for r in cal.small_lambdas if cal.lambda_at(r) is not None},
                reverse=True,
            )
            for r in rungs_check:
                lam_g = cal.lambda_at(r)
                lam_m = cal.small_lambda_at(r)
                assert lam_g is not None and lam_m is not None
                a, _raw, clamped = resolve_alpha(
                    lam_g, anchor_tokens, lam_m, s_min, where=f"{where}: at r={r:g} "
                )
                alpha[r] = a
                if clamped:
                    alpha_clamped.append(r)
        artifacts[engine] = cal
        header_artifacts[engine] = {
            "path": str(cal.path.resolve()),
            "sha256": cal.sha256,
            "model": cal.model,
            "session": cal.session,
            "anchor_seq_tokens": cal.anchor_seq_tokens,
            "rungs": {f"{r:g}": lam for r, lam in sorted(cal.lambdas.items(), reverse=True)},
            "labels": {f"{r:g}": label for r, label in sorted(cal.labels.items(), reverse=True)},
            "small_class": (
                None
                if cal.small_seq_tokens is None
                else {
                    "seq_tokens": cal.small_seq_tokens,
                    "arm": cal.small_arm,
                    "prefix_mode": cal.small_prefix_mode,
                    "rungs": {f"{r:g}": lam for r, lam in sorted(cal.small_lambdas.items(), reverse=True)},
                    "labels": {f"{r:g}": label for r, label in sorted(cal.small_labels.items(), reverse=True)},
                }
            ),
            "alpha": {f"{r:g}": a for r, a in sorted(alpha.items(), reverse=True)},
            # rungs whose raw slope sat in [-tolerance, 0) and reads as 0 (flat)
            "alpha_clamped": [f"{r:g}" for r in sorted(alpha_clamped, reverse=True)],
            "alpha_tolerance": (
                None if cal.small_seq_tokens is None
                else alpha_tolerance(anchor_tokens, cal.small_seq_tokens)
            ),
        }
    header = {
        "schema": RUNG_CALIBRATION_SCHEMA,
        "procedure_version": PROCEDURE_VERSION,
        "adr": RUNG_CALIBRATION_ADR,
        "two_anchor_adr": TWO_ANCHOR_ADR,
        "finding": RUNG_CALIBRATION_FINDING,
        "basis_gold": LAMBDA_BASIS_RUNG,
        "basis_small": LAMBDA_BASIS_SMALL,
        "basis_interpolated": LAMBDA_BASIS_INTERPOLATED,
        "interpolation": (
            "lambda(s) = lambda_g x (s_g / s)^alpha with alpha = ln(lambda_m / lambda_g) / "
            "ln(s_g / s_m) per (engine, r); the anchors are the largest (gold-fresh) and the "
            "smallest executable class, so no class is extrapolated; alpha = 1 is the KV-bound limit"
        ),
        "anchor_arm": DEMAND_ANCHOR_ARM,
        "anchor_seq_tokens": anchor_tokens,
        "required_rungs": {e: sorted(rs, reverse=True) for e, rs in sorted(required.items())},
        "classes_below_anchor": {
            e: {str(seq): sorted(rs, reverse=True) for seq, rs in sorted(by_class.items())}
            for e, by_class in sorted(below_anchor.items())
        },
        "artifacts": header_artifacts,
    }
    return header, artifacts


def dry_window_key(serving: Mapping[str, Any], demand_class: Mapping[str, Any]) -> Tuple[Any, ...]:
    """ADR-0153: the pair a dry window gates is the budgeted SERVING
    CONFIGURATION minus r: (engine, prefix mode, demand class tokens, kv dtype,
    connector, topology). A prefix-ON arm and a prefix-OFF arm at the same
    class run on different servers, so each gets its own dry window."""
    return (
        str(serving["engine"]),
        str(serving["prefix_mode"]),
        int(demand_class["seq_tokens"]),
        serving.get("kv_dtype"),
        serving.get("connector"),
        str(serving.get("topology")),
    )


def _dry_window_step(cell_step: Mapping[str, Any], spec: CellSpec) -> Dict[str, Any]:
    """ADR-0153: the dry window of one budgeted serving configuration (see
    dry_window_key): its first pressure cell at the tightest r it is carried at, re-minted at
    DRY_WINDOW_RATE_FRAC x lambda* (its own row key, so the dry root's cell
    directory names the rate it ran at), one window. The driver writes it to
    the sibling run root ``<run_id>-dry`` and reads its regime.json."""
    lam = cell_step["lambda_star_rps"]
    assert isinstance(lam, (int, float)) and lam > 0, "a dry window needs a measured lambda*"
    dry_spec = replace(spec, rate_frac=DRY_WINDOW_RATE_FRAC)
    offered = DRY_WINDOW_RATE_FRAC * float(lam)
    argv = list(cell_step["argv"])

    def _set(flag: str, value: str) -> None:
        at = argv.index(flag)
        argv[at + 1] = value

    _set("--rate", f"{offered:.6g}")
    _set("--num-trials", "1")
    # Duration mode with replay (DRY_WINDOW_DURATION_S): the V3 arrival-count
    # bound is a WINDOW CELL contract; the dry window is a gate, never data.
    at = argv.index("--arrival-count")
    del argv[at:at + 2]
    argv += ["--duration-s", f"{DRY_WINDOW_DURATION_S:g}"]
    env = dict(cell_step["env"])
    env.update(_cell_identity_env(dry_spec))
    env.pop("CAGE_WINDOW_ORDINAL_BASE", None)
    env[LADDER_REPLAY_ENV] = "1"
    return {
        "kind": "dry_window",
        "engine": spec.engine,
        "demand_class": cell_step["demand_class"],
        "budget_r": spec.budget_r,
        "row_key": dry_spec.to_row_key(),
        "cellspec": dry_spec.to_flat_dict(),
        "dataset": cell_step["dataset"],
        "windows": 1,
        "num_queries": cell_step["num_queries"],
        "rate_frac": DRY_WINDOW_RATE_FRAC,
        "offered_rate_rps": offered,
        "duration_s": DRY_WINDOW_DURATION_S,
        "replay": True,
        "lambda_star_rps": float(lam),
        "lambda_star_source": cell_step["lambda_star_source"],
        "rate_basis": cell_step["rate_basis"],
        "expected_label": DRY_WINDOW_EXPECTED_LABEL,
        "gate": DRY_WINDOW_GATE,
        "adr": DRY_WINDOW_ADR,
        "serving": cell_step["serving"],
        "budget_plan": cell_step["budget_plan"],
        "argv": argv,
        "env": env,
    }


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
    calibrations: Optional[Mapping[str, Path]] = None,
    calibration_budget_fraction: float = FLOOR_BUDGET_FRACTION,
    rehearsal_n: Optional[int] = None,
    rung_calibrations: Optional[Mapping[str, Path]] = None,
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

    ``calibrations`` maps engine -> cal-v1 artifact path (``plan
    --calibration ENGINE=PATH``, Batch 2 W4): one per executable server
    engine, validated and recorded in the header ``calibration``; every cell
    step pins the floors as CAGE_SLO_FLOORS_JSON. ``calibration_budget_fraction``
    is the floor rung every artifact must carry (the charter's r = 1.5).

    ``rung_calibrations`` maps engine -> rung-calibration artifact path
    (``plan --rung-calibration ENGINE=PATH``, ADR-0154): one per engine with
    executable pressure cells, every (engine, r) of those cells ESTIMATED, or
    the plan refuses. ADR-0155 sizes every budgeted relaunch on its arm's
    demand class and ADR-0153 plans one dry window per (engine, class).
    """
    grid = get_session_grid(session)
    rehearsal: Optional[Dict[str, Any]] = None
    if rehearsal_n is not None:
        # ADR-0144: the rehearsal is derived from the registered grid by the
        # one rule and recorded in the header; the registry is untouched.
        grid = rehearsal_grid(
            grid, n=rehearsal_n, datasets=frozenset(query_manifests or {})
        )
        rehearsal = {
            "of": session,
            "n": rehearsal_n,
            "windows": grid.replications,
            "datasets": sorted(query_manifests or {}),
            "f2_coordinates": [[r, f] for r in grid.f2_budgets for f in grid.f2_rates],
            "f2_fine_coordinates": [
                [r, f] for r in grid.f2_fine_budgets for f in grid.f2_fine_rates
            ],
            "f3_coordinates": [[r, f] for r in grid.f3_budgets for f in grid.f3_rates],
            "rule": REHEARSAL_RULE,
            "adr": REHEARSAL_ADR,
        }
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

    cells = sorted(enumerate_cells(grid), key=lambda c: _sort_key(c, grid))
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
    # Batch 2 W4: one §6.1 floor per executable server engine, pinned on
    # every cell (the manifest is created by the first emitting cell). A plan
    # with no executable server cell (an hf-only shakedown) registers no
    # floor and pins nothing (review F4: an empty pin is not a pin).
    calibration = _register_calibrations(
        grid, calibrations, cells, calibration_budget_fraction
    )
    slo_floors_env: Optional[str] = (
        slo_floors_env_value(calibration["floors"]) if calibration["floors"] else None
    )
    # ADR-0155: the P6 artifact and the registration describe one anchor
    # sequence; every budgeted relaunch scales D from it to its class.
    anchor_tokens = (
        _check_floor_shape(grid, floor)
        if any(_is_budgeted(c.spec) and c.blocked_on is None and c.spec.engine != "hf" for c in cells)
        else int(grid.demand_seq_tokens[DEMAND_ANCHOR_ARM])
    )
    # ADR-0154: lambda* per (engine, rung) for every executable pressure cell.
    rung_header, rung_table = _register_rung_calibrations(
        grid, cells, rung_calibrations, anchor_tokens
    )

    steps: List[Dict[str, Any]] = []
    current: Optional[_ServingConfig] = None
    current_budget_plan: Optional[Dict[str, Any]] = None
    relaunches = 0
    dried: set = set()
    for cell in cells:
        config = _serving_config(cell, grid)
        # Blocked cells launch NOTHING: they never run, so their (possibly
        # unrealizable) serving config must not emit a relaunch step nor
        # disturb the boundary the surrounding executable cells run under.
        if cell.blocked_on is None and config is not None and config != current:
            relaunch = _relaunch_step(config, grid, floor, launcher_cmds)
            steps.append(relaunch)
            current = config
            current_budget_plan = relaunch["budget_plan"]
            relaunches += 1
        manifest_rec = manifests.get(cell.dataset)
        # W4: an executable server cell pins the record of the relaunch it
        # runs under (the same object, by construction; None under a
        # budget-free boundary); hf and blocked cells carry none.
        budget_plan = (
            current_budget_plan
            if cell.blocked_on is None and config is not None
            else None
        )
        cell_step = _cell_step(
            cell,
            grid,
            floor,
            runner_cmd,
            seed,
            pins,
            query_manifest=None if manifest_rec is None else manifest_rec["path"],
            slo_floors_env=slo_floors_env,
            budget_plan=budget_plan,
            rungs=rung_table,
        )
        # ADR-0153: the first executable pressure cell of each (engine,
        # demand class) is preceded by that pair's dry window. Budgets sort
        # ascending, so this is the tightest r the class is carried at.
        if (
            cell.blocked_on is None
            and cell.spec.family in _PRESSURE_FAMILIES
            and cell_step["demand_class"] is not None
        ):
            key = dry_window_key(cell_step["serving"], cell_step["demand_class"])
            if key not in dried:
                dried.add(key)
                steps.append(_dry_window_step(cell_step, cell.spec))
        steps.append(cell_step)
    for i, step in enumerate(steps):
        step["index"] = i

    cell_steps = [s for s in steps if s["kind"] == "cell"]
    dry_steps = [s for s in steps if s["kind"] == "dry_window"]
    window_spans = window_span_summary(cell_steps)
    # Gap triage 2026-10-09 C1: a pool below one max_model_len request fails
    # at engine start on the pod; refused HERE, with every offender listed.
    shortfalls = pool_shortfalls([s for s in steps if s["kind"] == "relaunch"], grid.max_model_len)
    if shortfalls:
        raise PlanError(_pool_shortfall_text(shortfalls, f"plan of session {session!r}"))
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
        # ADR-0144: null on a registered plan; the derivation record of a
        # dress rehearsal otherwise (the per_row_n and counts below are the
        # rehearsal's own).
        "rehearsal": rehearsal,
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
            # ADR-0103 and ADR-0150: the arms served prefix OFF by relaunch
            # in every family (the charter's reuse-off arms); F2 stays the
            # prefix-OFF family.
            "prefix_off_arms": sorted(PREFIX_OFF_ARMS),
            "prefix_off_adr": PREFIX_OFF_ADRS,
            # ADR-0055 (Batch 2 W1): every cell env pins decoupled scoring;
            # re-checked per cell by load_plan.
            "quality_scoring": "decoupled",
            "quality_scoring_env": SKIP_QUALITY_ENV,
            "quality_scoring_adr": DECOUPLED_SCORING_ADR,
            # V3: every window cell is bounded by --arrival-count equal to
            # its --num-queries (per_row_n.window_requests, or the dataset's
            # achievable n); window_duration_s above is the cost estimate.
            "window_bound": WINDOW_BOUND,
            "window_bound_finding": WINDOW_BOUND_FINDING,
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
            # ADR-0158: the per-class request cap every budgeted relaunch of
            # this session carries (a budget-free one carries max_model_len);
            # re-checked per relaunch by load_plan.
            "request_caps": {
                str(seq): request_cap_tokens(grid, seq)
                for seq in sorted(set(grid.demand_seq_tokens.values()) | set(grid.corpus_trunc_demand_seq_tokens.values()), reverse=True)
            },
            "request_cap_rule": REQUEST_CAP_RULE,
            "request_cap_adr": REQUEST_CAP_ADR,
            "request_cap_served_max_tokens": {
                str(k): v for k, v in sorted(grid.demand_class_max_served_tokens.items(), reverse=True)
            },
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
            # Batch 2 W4 (ADR-0117): the relaunch's cache_budget.BudgetPlan
            # record rides every budgeted cell step as this env and lands in
            # cell.json under this key (the rho_own consumer's read);
            # re-checked per cell and per relaunch by load_plan.
            "budget_plan_env": BUDGET_PLAN_ENV,
            "budget_plan_cell_key": BUDGET_PLAN_CELL_KEY,
            "budget_plan_finding": SLO_FLOORS_FINDING,
        },
        # Batch 2 W4 (ADR-0117): the per-engine §6.1 floors this plan pins on
        # every cell (path + sha256 + what was validated); re-checked per cell
        # by load_plan; written into manifest.json["slo_floors"] by the
        # campaign session at manifest creation.
        "calibration": calibration,
        # ADR-0154: the per-(engine, rung) lambda* artifacts the offered rates
        # rest on (path + sha256 + the values), re-checked per cell by load_plan.
        "rung_calibration": rung_header,
        # ADR-0155: the demand classes every budgeted relaunch was sized on.
        "demand_classes": {
            "adr": DEMAND_CLASS_ADR,
            "finding": DEMAND_CLASS_FINDING,
            "source": DEMAND_SEQ_TOKENS_SOURCE,
            "anchor_arm": DEMAND_ANCHOR_ARM,
            "anchor_seq_tokens": anchor_tokens,
            "floor_avg_seq_tokens": floor.avg_seq_tokens,
            "floor_concurrency_target": floor.concurrency_target,
            "seq_tokens": dict(sorted(grid.demand_seq_tokens.items())),
            "corpus_trunc_seq_tokens": {
                str(k): v for k, v in sorted(grid.corpus_trunc_demand_seq_tokens.items())
            },
            "rule": "D_class = floor(D_floor x s_class / s_anchor); the budget at r is floor(r x D_class)",
        },
        # ADR-0153: one dry window per (engine, demand class) with pressure
        # cells, at the tightest r, offered at rate_frac x lambda*; the run
        # fails the class unless its regime.json reads the expected label.
        "dry_window": {
            "adr": DRY_WINDOW_ADR,
            "finding": DRY_WINDOW_FINDING,
            "rate_frac": DRY_WINDOW_RATE_FRAC,
            "expected_label": DRY_WINDOW_EXPECTED_LABEL,
            "root_suffix": DRY_WINDOW_ROOT_SUFFIX,
            "exit_code": EXIT_DRY_WINDOW_FAILED,
            "count": len(dry_steps),
            "key": ["engine", "prefix_mode", "seq_tokens", "kv_dtype", "connector", "topology", "budget_r"],
            "pairs": [
                [*dry_window_key(s["serving"], s["demand_class"]), s["budget_r"]] for s in dry_steps
            ],
        },
        # ADR-0156 / S0F-68: the expected spans of the executable pressure
        # windows (W / offered rate) against the registered floor; a count the
        # operator reads before the GO, never a refusal (W is registered).
        "window_spans": window_spans,
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
                # V1: the registered input shape is the RENDERED request; the
                # haystack carries the shape minus the wrapper allowance and
                # the runner refuses a prompt above the cap.
                "haystack_tokens": RULER_HAYSTACK_TOKENS,
                "wrapper_allowance_tokens": RULER_WRAPPER_ALLOWANCE,
                "rendered_input_cap": RULER_CONTEXT_TOKENS,
                "sizing_finding": RULER_SIZING_FINDING,
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
            # ADR-0153: the dry windows 'run' executes before the pressure
            # cells of each (engine, demand class); not campaign windows.
            "dry_windows": len(dry_steps),
            # V2: the family stops 'run' executes (boundaries + the end of
            # run; the clean-room stops before the first step are not counted
            # here, they are one per launcher the plan uses).
            "engine_stops": len(stop_boundaries(steps)),
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
    # Batch 2 W4: the relaunch's BudgetPlan record on budgeted cells (null
    # elsewhere); persisted into cell.json by the campaign session. The
    # serving record every relaunch-agreement clause keys on is required
    # too (review F3: a server cell with serving null escaped every clause).
    "budget_plan",
    "serving",
    # v6 (Batch B): the demand class (ADR-0155) and the offered-rate basis
    # records (ADR-0154); a v5 plan predates them and is refused.
    "demand_class",
    "lambda_star_rps",
    "lambda_star_source",
    "lambda_kv_pred_rps",
    "rate_basis",
    # ADR-0156 / S0F-68: the window span audit on pressure cells (null elsewhere).
    "window_span_s_expected",
    "window_span_below_floor",
)
#: ADR-0153: the dry window step (one per engine and demand class).
_DRY_STEP_KEYS = (
    "engine",
    "demand_class",
    "budget_r",
    "row_key",
    "cellspec",
    "dataset",
    "windows",
    "num_queries",
    "rate_frac",
    "offered_rate_rps",
    "lambda_star_rps",
    "rate_basis",
    "expected_label",
    "serving",
    "budget_plan",
    "argv",
    "env",
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
    # Batch 2 W4: the BudgetPlan record the budget env was derived from
    # (null on a budget-free relaunch); the cells under it pin it. The
    # byte total is required beside it (review F2: budgeted-ness is derived
    # from the relaunch identity, never inferred from an optional key).
    "budget_plan",
    "budget_bytes",
    # V2 (engine handoff): the launcher this boundary belongs to and the argv
    # that stops its family; both re-checked against the relaunch argv below.
    "launcher_key",
    "stop_argv",
    # v6 (ADR-0155): the demand class a budgeted relaunch was sized on.
    "demand_class",
)


def _stale_relaunch_problems(
    step: Mapping[str, Any], label: str, header_max_model_len: int,
    header_request_caps: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """load_plan's fail-closed per-relaunch check (backlog A10, ADR-0158): the
    ``max_model_len`` record is a positive integer equal to the header's cap
    for the relaunch's demand class (``serving_shapes.request_caps``; the
    session cap on a budget-free relaunch) and the env carries
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
    # V2 (engine handoff): the stop argv must aim at the SAME launcher the
    # relaunch argv starts (its prefix before the start/restart verb), and the
    # launcher key must be the one the topology implies; a drifted record
    # would stop the wrong family, or none, at a boundary.
    argv_raw = step.get("argv")
    if not isinstance(argv_raw, list):
        # load_plan already recorded the argv type problem (review F5: a
        # string argv must refuse, never crash the check below).
        return problems
    argv: List[Any] = argv_raw
    verb = "start" if step.get("topology") == "pd" else "restart"
    want_key = PD_LAUNCHER_KEY if step.get("topology") == "pd" else engine
    if step.get("launcher_key") != want_key:
        problems.append(
            f"{label}: relaunch launcher_key is {step.get('launcher_key')!r}, the "
            f"{step.get('topology')!r} topology of engine {engine!r} is launched "
            f"by {want_key!r} ({ENGINE_STOP_FINDING}: the stop at a family boundary "
            "would target the wrong launcher)" + stale
        )
    # The stop argv is the launcher PREFIX plus the stop verb, and the relaunch
    # argv is that same prefix followed by its start/restart verb (review F6:
    # positional, never a search for the verb, so a prefix token spelled like
    # a verb cannot refuse a fresh plan).
    stop_argv = step.get("stop_argv")
    if (
        not isinstance(stop_argv, list)
        or len(stop_argv) < 2
        or stop_argv[-1] != LAUNCHER_STOP_VERB
    ):
        problems.append(
            f"{label}: relaunch stop_argv is {stop_argv!r}, must be the launcher "
            f"prefix plus {LAUNCHER_STOP_VERB!r} ({ENGINE_STOP_FINDING})" + stale
        )
    else:
        prefix = stop_argv[:-1]
        if argv[: len(prefix)] != prefix or len(argv) <= len(prefix) or argv[len(prefix)] != verb:
            problems.append(
                f"{label}: relaunch stop_argv {stop_argv!r} does not stop the launcher "
                f"the relaunch argv starts ({argv[: len(prefix) + 1]!r} should be the "
                f"same prefix followed by {verb!r}) ({ENGINE_STOP_FINDING}: a stop "
                "aimed at another launcher leaves this family resident at the "
                "boundary)" + stale
            )
    value = step.get("max_model_len")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        problems.append(
            f"{label}: relaunch max_model_len is {value!r}, must be an integer "
            ">= 1 (backlog A10)" + stale
        )
        return problems
    seq = (step.get("demand_class") or {}).get("seq_tokens")
    expected: Any = header_max_model_len
    if seq is not None:
        expected = (header_request_caps or {}).get(str(seq))
        if expected is None:
            problems.append(
                f"{label}: relaunch of demand class {seq} tokens but the header's "
                f"serving_shapes.request_caps has no cap for it ({REQUEST_CAP_ADR})" + stale
            )
            return problems
    if value != expected:
        problems.append(
            f"{label}: relaunch max_model_len {value} differs from the header's cap "
            f"{expected} for its class ({'budget-free: the session cap' if seq is None else f'{seq} tokens'}; "
            f"{REQUEST_CAP_ADR}; backlog A10)" + stale
        )
    got = env.get(MAX_MODEL_LEN_ENV)
    if got != str(value):
        problems.append(
            f"{label}: relaunch env {MAX_MODEL_LEN_ENV} is {got!r} but its "
            f"max_model_len record is {value} (backlog A10: the launcher reads "
            "the env; without it the pilot shell default 4096 would serve the "
            "cells under this relaunch)" + stale
        )
    # Batch 2 W4 (ADR-0117): a budgeted relaunch carries the BudgetPlan
    # record its launcher env was derived from (same total, engine and, off
    # the pd stack, the same TP degree; a 'single' relaunch may be planned
    # TP-sharded); a budget-free one carries null.
    record = step.get("budget_plan")
    budget_bytes = step.get("budget_bytes")
    topology = str(step.get("topology"))
    # Review F2: budgeted-ness follows the relaunch IDENTITY (a pressure
    # coordinate, or the DIST overlay at dist_budget_r), never an optional
    # key: a nulled budget_bytes must not read as budget-free.
    budgeted = step.get("budget_r") is not None or topology in ("tp", "pd")
    budget_env_names = sorted(_BUDGET_FLAG_ENV.values()) + [
        "CAGE_KV_BUDGET_BYTES_PREFILL", "CAGE_KV_BUDGET_BYTES_DECODE",
    ]
    # ADR-0155: a budgeted relaunch names the demand class its budget was
    # sized on, and the record agrees; a budget-free one carries null.
    demand_class = step.get("demand_class")
    if budgeted:
        seq = demand_class.get("seq_tokens") if isinstance(demand_class, dict) else None
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            problems.append(
                f"{label}: budgeted relaunch carries demand_class={demand_class!r}, "
                f"must name its served-token class ({DEMAND_CLASS_ADR}: a budget "
                "sized on no stated shape is the S0F-63 defect)" + stale
            )
        elif isinstance(record, dict) and record.get("avg_seq_tokens") != seq:
            problems.append(
                f"{label}: budget_plan.avg_seq_tokens {record.get('avg_seq_tokens')!r} "
                f"!= the relaunch demand class {seq} ({DEMAND_CLASS_ADR})" + stale
            )
    elif demand_class is not None:
        problems.append(
            f"{label}: budget-free relaunch carries demand_class={demand_class!r} "
            f"({DEMAND_CLASS_ADR}: absence stays absence)" + stale
        )
    if not budgeted:
        if budget_bytes is not None or record is not None:
            problems.append(
                f"{label}: budget-free relaunch (no budget_r, single topology) "
                f"carries budget_bytes={budget_bytes!r} / budget_plan "
                f"{'record' if record is not None else 'null'} "
                f"({SLO_FLOORS_FINDING})" + stale
            )
        carried = [name for name in budget_env_names if name in env]
        if carried:
            problems.append(
                f"{label}: budget-free relaunch env carries {carried} "
                f"({SLO_FLOORS_FINDING}: a server launched under a budget the "
                "plan does not record)" + stale
            )
    elif (
        isinstance(budget_bytes, bool)
        or not isinstance(budget_bytes, int)
        or budget_bytes < 1
    ):
        problems.append(
            f"{label}: budgeted relaunch (budget_r={step.get('budget_r')!r}, "
            f"topology {topology!r}) has budget_bytes={budget_bytes!r}, must be "
            f"an integer >= 1 ({SLO_FLOORS_FINDING})" + stale
        )
    elif not isinstance(record, dict):
        problems.append(
            f"{label}: budgeted relaunch (budget_bytes={budget_bytes}) has no "
            f"budget_plan record ({SLO_FLOORS_FINDING}: the cells under it pin "
            "this record into cell.json)" + stale
        )
    else:
        # Review F1: the served dtype the record carries is the relaunch's
        # lever, else bf16 (the only floor-table dtype a plain launch
        # accepts); rho_own applies the bytes-per-token factor from it.
        want_dtype = step.get("kv_dtype") or "bf16"
        if record.get("kv_dtype") != want_dtype:
            problems.append(
                f"{label}: budget_plan.kv_dtype {record.get('kv_dtype')!r} != the "
                f"served dtype {want_dtype!r} (the relaunch lever, else bf16) "
                f"({SLO_FLOORS_FINDING}: rho_own would apply the wrong "
                "bytes-per-token factor)" + stale
            )
        if topology == "pd":
            pd = step.get("pd") if isinstance(step.get("pd"), dict) else {}
            pools = [pd.get("prefill_bytes"), pd.get("decode_bytes")]
            if record.get("pools_bytes") != pools or _json_norm(
                record.get("pd_roles")
            ) != _json_norm(pd):
                problems.append(
                    f"{label}: budget_plan pools/pd_roles differ from the "
                    f"relaunch pd record ({SLO_FLOORS_FINDING}: the per-rank "
                    "slices the launcher was given are the record's pd_roles)"
                    + stale
                )
        else:
            # Review F1: the launched knob IS the record's primary engine
            # arg, verbatim (vLLM bytes, SGLang tokens).
            primary = [
                a for a in (record.get("engine_args") or [])
                if isinstance(a, dict) and a.get("kind") == "primary"
            ]
            args = primary[0].get("args") if len(primary) == 1 else None
            if not isinstance(args, list) or len(args) != 2:
                problems.append(
                    f"{label}: budget_plan carries no single primary knob "
                    f"({SLO_FLOORS_FINDING})" + stale
                )
            else:
                flag, value = args
                env_name = _BUDGET_FLAG_ENV.get(str(flag))
                if env_name is None or env.get(env_name) != value:
                    problems.append(
                        f"{label}: budget_plan primary knob {flag} {value!r} != "
                        f"relaunch env {env_name}={env.get(env_name) if env_name else None!r} "
                        f"({SLO_FLOORS_FINDING}: the record must be the plan the "
                        "launcher env was derived from)" + stale
                    )
        if record.get("budget_bytes_total") != budget_bytes:
            problems.append(
                f"{label}: budget_plan.budget_bytes_total "
                f"{record.get('budget_bytes_total')!r} != relaunch budget_bytes "
                f"{budget_bytes!r} ({SLO_FLOORS_FINDING})" + stale
            )
        if step.get("budget_r") is not None and record.get("r") != step.get("budget_r"):
            # Review T1: the DIST legs carry budget_r None (their r is the
            # registered dist_budget_r, not a plan-visible coordinate).
            problems.append(
                f"{label}: budget_plan.r {record.get('r')!r} != relaunch budget_r "
                f"{step.get('budget_r')!r} ({SLO_FLOORS_FINDING})" + stale
            )
        if record.get("engine") != engine:
            problems.append(
                f"{label}: budget_plan.engine {record.get('engine')!r} != relaunch "
                f"engine {engine!r} ({SLO_FLOORS_FINDING})" + stale
            )
        allowed = {"single": ("single", "tp"), "tp": ("tp",), "pd": ("pd",)}
        if record.get("topology") not in allowed.get(topology, ()):
            problems.append(
                f"{label}: budget_plan.topology {record.get('topology')!r} cannot "
                f"plan a {topology!r} relaunch ({SLO_FLOORS_FINDING})" + stale
            )
        if topology != "pd" and record.get("tp") != step.get("tp"):
            problems.append(
                f"{label}: budget_plan.tp {record.get('tp')!r} != relaunch tp "
                f"{step.get('tp')!r} ({SLO_FLOORS_FINDING})" + stale
            )
        if not isinstance(record.get("kv_dtype"), str):
            problems.append(
                f"{label}: budget_plan.kv_dtype {record.get('kv_dtype')!r} is not "
                f"a string ({SLO_FLOORS_FINDING})" + stale
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
    calibration = plan.get("calibration")
    header_floors = calibration.get("floors") if isinstance(calibration, dict) else None
    if not isinstance(header_floors, dict):
        raise RunError(
            f"plan {path} has no calibration.floors header ({SLO_FLOORS_FINDING}: "
            "the §6.1 single-stream floors every cell pins as "
            f"{SLO_FLOORS_ENV} and the campaign session writes into "
            f"manifest.json[{SLO_FLOORS_MANIFEST_KEY!r}]; a pre-W4 plan would "
            "produce a manifest contrast #14 refuses) - stale plan, re-plan"
        )
    # Review F5: the header floors must cover every engine an executable
    # server cell serves (build_plan enforces it; a hand-edited header would
    # spend the GPU time before contrast #14 refuses the missing engine).
    # Review F4: a plan with no executable server cell registers no floor
    # and pins nothing (an empty floors object is legal only then).
    executable_engines = sorted({
        str(s["cellspec"].get("engine"))
        for s in steps
        if isinstance(s, dict) and s.get("kind") == "cell"
        and s.get("blocked_on") is None and isinstance(s.get("cellspec"), dict)
        and s["cellspec"].get("engine") != "hf"
    })
    uncovered = sorted(set(executable_engines) - set(header_floors))
    if uncovered:
        raise RunError(
            f"plan {path} calibration.floors has no floor for executable "
            f"engine(s) {uncovered} ({SLO_FLOORS_FINDING}: contrast #14 refuses "
            "an in-regime window whose engine has no floor; re-plan with "
            "--calibration for every engine) - stale plan, re-plan"
        )
    slo_floors_env: Optional[str] = (
        slo_floors_env_value(header_floors) if header_floors else None
    )
    # ADR-0154/0155/0153 headers: a pre-Batch-B plan (v5) is refused by the
    # schema literal; a hand-edited v6 plan without them is refused here.
    executable_pressure = any(
        isinstance(s, dict) and s.get("kind") == "cell" and s.get("blocked_on") is None
        and s.get("family") in _PRESSURE_FAMILIES
        for s in steps
    )
    if executable_pressure and not isinstance(plan.get("rung_calibration"), dict):
        raise RunError(
            f"plan {path} has no rung_calibration header ({RUNG_CALIBRATION_ADR}: the "
            "measured lambda* every pressure cell's rate rests on) - stale plan, re-plan"
        )
    if not isinstance(plan.get("demand_classes"), dict):
        raise RunError(
            f"plan {path} has no demand_classes header ({DEMAND_CLASS_ADR}: the "
            "served-token classes every budgeted relaunch was sized on) - stale plan, "
            "re-plan"
        )
    preceding_relaunch: Optional[Dict[str, Any]] = None
    step_kinds = ("cell", "relaunch", "dry_window")
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("kind") not in step_kinds:
            problems.append(f"steps[{i}]: kind must be one of {list(step_kinds)}")
            continue
        if step["kind"] == "relaunch":
            preceding_relaunch = step
        keys = {
            "cell": _CELL_STEP_KEYS, "relaunch": _RELAUNCH_STEP_KEYS, "dry_window": _DRY_STEP_KEYS,
        }[step["kind"]]
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
                _stale_relaunch_problems(
                    step, f"steps[{i}]", header_max_model_len,
                    shapes.get("request_caps") if isinstance(shapes, dict) else None,
                )
            )
        if step["kind"] == "dry_window":
            # ADR-0153: the dry window runs under the budgeted relaunch of its
            # engine and class at the expected rate; its label is the gate.
            label = f"steps[{i}]"
            if preceding_relaunch is None or preceding_relaunch.get("engine") != step.get("engine") or preceding_relaunch.get("budget_r") is None:
                problems.append(
                    f"{label}: dry window for {step.get('engine')!r} has no budgeted relaunch of "
                    f"its engine before it ({DRY_WINDOW_ADR})"
                )
            elif (preceding_relaunch.get("demand_class") or {}).get("seq_tokens") != (step.get("demand_class") or {}).get("seq_tokens"):
                problems.append(
                    f"{label}: dry window demand class {step.get('demand_class')!r} differs from "
                    f"its relaunch's {preceding_relaunch.get('demand_class')!r} ({DRY_WINDOW_ADR})"
                )
            elif isinstance(step.get("serving"), dict) and any(
                preceding_relaunch.get(k) != step["serving"].get(k)
                for k in ("prefix_mode", "kv_dtype", "connector", "topology")
            ):
                problems.append(
                    f"{label}: dry window serving {step.get('serving')!r} is not the serving "
                    f"configuration of the relaunch before it ({DRY_WINDOW_ADR})"
                )
            if step.get("expected_label") != DRY_WINDOW_EXPECTED_LABEL or step.get("rate_frac") != DRY_WINDOW_RATE_FRAC:
                problems.append(
                    f"{label}: dry window expects {step.get('expected_label')!r} at rate_frac "
                    f"{step.get('rate_frac')!r}; the registered gate is {DRY_WINDOW_EXPECTED_LABEL} "
                    f"at {DRY_WINDOW_RATE_FRAC} ({DRY_WINDOW_ADR})"
                )
            lam = step.get("lambda_star_rps")
            got_rate = _argv_flag_value(step["argv"], "--rate") if isinstance(step["argv"], list) else None
            try:
                want = DRY_WINDOW_RATE_FRAC * float(lam)
                rate_ok = got_rate is not None and math.isclose(float(got_rate), want, rel_tol=1e-5)
            except (TypeError, ValueError):
                rate_ok = False
            if not rate_ok:
                problems.append(
                    f"{label}: dry window --rate {got_rate!r} is not {DRY_WINDOW_RATE_FRAC} x "
                    f"lambda_star_rps {lam!r} ({DRY_WINDOW_ADR})"
                )
            if _argv_flag_value(step["argv"], "--num-trials") != "1" if isinstance(step["argv"], list) else True:
                problems.append(f"{label}: dry window must run exactly one window (--num-trials 1)")
            if isinstance(step["argv"], list) and (
                _argv_flag_value(step["argv"], "--duration-s") != f"{DRY_WINDOW_DURATION_S:g}"
                or "--arrival-count" in step["argv"]
                or "--open-loop-warmup-s" in step["argv"]
            ):
                problems.append(
                    f"{label}: dry window runs in duration mode for {DRY_WINDOW_DURATION_S:g} s "
                    f"(no --arrival-count, no warm-up flag) ({DRY_WINDOW_ADR})"
                )
            if isinstance(step["env"], dict) and step["env"].get(LADDER_REPLAY_ENV) != "1":
                problems.append(
                    f"{label}: dry window carries no {LADDER_REPLAY_ENV}=1 (its duration window "
                    f"replays the pool; non-confirmatory by registration) ({DRY_WINDOW_ADR})"
                )
            try:
                if CellSpec.from_flat_dict(step["cellspec"]).to_row_key() != step["row_key"]:
                    problems.append(f"{label}: dry window row_key is not minted from its cellspec")
            except Exception as exc:  # cellspec is the one legality gate
                problems.append(f"{label}: dry window cellspec is charter-illegal: {exc}")
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
                    slo_floors_env=slo_floors_env,
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


def _stop_record(step: Mapping[str, Any], before_index: Optional[int]) -> Dict[str, Any]:
    return {
        "before_index": before_index,
        "launcher_key": step["launcher_key"],
        "stop_argv": list(step["stop_argv"]),
        "env": dict(step.get("env") or {}),
    }


def stop_boundaries(steps: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """V2 (engine handoff): WHERE a running engine family must be stopped,
    derived from the relaunch sequence alone (pure; the plan's own content).

    One record per relaunch step whose ``launcher_key`` differs from the
    previous relaunch's: ``before_index`` is that relaunch's position in
    ``steps`` and the stop is the PREVIOUS family's ``stop_argv`` under the
    env of its last relaunch; plus, when any relaunch exists, one final record
    with ``before_index`` None: the last family, stopped when the run ends.
    Relaunches of the same launcher need no stop between them (the launcher's
    restart, or the pd stack's start, is self-cleaning within its family). A
    plan without relaunch steps (hf oracle only) stops nothing.
    """
    out: List[Dict[str, Any]] = []
    previous: Optional[Mapping[str, Any]] = None
    for index, step in enumerate(steps):
        if step.get("kind") != "relaunch":
            continue
        if previous is not None and previous["launcher_key"] != step["launcher_key"]:
            out.append(_stop_record(previous, index))
        previous = step
    if previous is not None:
        out.append(_stop_record(previous, None))
    return out


def plan_launchers(steps: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """V2: every launcher the plan uses, once each, in order of first
    appearance, with the stop argv and env of its FIRST relaunch (the clean
    room 'run' performs before its first step: a resident engine of an
    aborted run must not meet the first relaunch of this one). A launcher
    the plan never uses is never named here, so nothing outside the plan is
    ever executed."""
    seen: Dict[str, Dict[str, Any]] = {}
    for step in steps:
        if step.get("kind") == "relaunch" and step["launcher_key"] not in seen:
            seen[step["launcher_key"]] = _stop_record(step, None)
    return list(seen.values())


@dataclass
class _Outcome:
    row_key: str
    dataset: str
    outcome: str  # ok | skipped-complete | skipped-blocked | failed | skipped-launch-failed | skipped-dry-window | dry-window-ok | dry-window-failed


def dry_window_root(campaign_root: Path, plan: Optional[Mapping[str, Any]] = None) -> Path:
    """ADR-0153: the sibling run root the dry windows are written to. The
    name is the run id's first 30 characters, ``-dry-`` and six hex digits of
    the plan's rung artifacts (review HIGH 2, 2026-10-09: a re-plan with new
    lambda* values gets a fresh root, so a stale verdict is never reused
    across plans), kept inside the §1 run-id grammar for every session (a
    cd-act run id plus ``-dry`` alone would exceed it). The campaign tree
    never carries a dry window."""
    campaign_root = Path(campaign_root)
    artifacts = ((plan or {}).get("rung_calibration") or {}).get("artifacts") or {}
    shas = sorted(str((a or {}).get("sha256")) for a in artifacts.values())
    digest = hashlib.sha256(",".join(shas).encode("utf-8")).hexdigest()[:6]
    name = f"{campaign_root.name[:30]}{DRY_WINDOW_ROOT_SUFFIX}-{digest}"
    if not RUN_ID_RE.match(name):
        raise RunError(f"dry window root {name!r} violates the §1 grammar {RUN_ID_RE.pattern}")
    return campaign_root.parent / name


def _dry_window_dir(dry_root: Path, step: Mapping[str, Any], ordinal: int) -> Path:
    return Path(dry_root) / "cells" / step["row_key"] / f"window_{step['dataset']}-{ordinal:02d}"


def _dry_attempt_ordinals(dry_root: Path, step: Mapping[str, Any]) -> List[int]:
    """The window ordinals already written for this dry window (one per
    attempt), ascending; the next attempt runs at the next ordinal through
    CAGE_WINDOW_ORDINAL_BASE so no verdict is ever overwritten."""
    cell_dir = Path(dry_root) / "cells" / step["row_key"]
    prefix = f"window_{step['dataset']}-"
    out: List[int] = []
    if cell_dir.is_dir():
        for entry in cell_dir.iterdir():
            if entry.is_dir() and entry.name.startswith(prefix) and entry.name[len(prefix):].isdigit():
                out.append(int(entry.name[len(prefix):]))
    return sorted(out)


def _read_dry_window_regime(
    dry_root: Path, step: Mapping[str, Any], ordinal: int = 1
) -> Tuple[Optional[str], Dict[str, Any]]:
    """The dry window attempt's regime.json (label, document) or (None, {})
    when the runner wrote none (a refusal before the window, or a crash)."""
    path = _dry_window_dir(dry_root, step, ordinal) / "regime.json"
    if not path.is_file():
        return None, {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, {}
    label = doc.get("label") if isinstance(doc, dict) else None
    return (label if isinstance(label, str) else None), (doc if isinstance(doc, dict) else {})


def _dry_sidecar_path(dry_root: Path, step: Mapping[str, Any], ordinal: int) -> Path:
    return Path(dry_root) / "dry_attempts" / f"{step['row_key']}.{ordinal:02d}.json"


def dry_window_identity(step: Mapping[str, Any]) -> Dict[str, Any]:
    """What a dry window attempt ran at: a later run reuses an IN_REGIME
    verdict only when this identity is unchanged (review HIGH 2)."""
    source = step.get("lambda_star_source") or {}
    return {
        "offered_rate_rps": step["offered_rate_rps"],
        "lambda_star_rps": step["lambda_star_rps"],
        "rate_frac": step["rate_frac"],
        "duration_s": step.get("duration_s"),
        "artifact_sha256": source.get("artifact_sha256") if isinstance(source, dict) else None,
        "seq_tokens": (step.get("demand_class") or {}).get("seq_tokens"),
        "serving": step.get("serving"),
        "budget_plan_total": (step.get("budget_plan") or {}).get("budget_bytes_total"),
    }


def _read_dry_sidecar(dry_root: Path, step: Mapping[str, Any], ordinal: int) -> Dict[str, Any]:
    path = _dry_sidecar_path(dry_root, step, ordinal)
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _write_dry_sidecar(
    dry_root: Path, step: Mapping[str, Any], ordinal: int, label: Optional[str], runner_exit: int
) -> None:
    path = _dry_sidecar_path(dry_root, step, ordinal)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "cage-dry-window-attempt-v1",
                "adr": DRY_WINDOW_ADR,
                "row_key": step["row_key"],
                "dataset": step["dataset"],
                "ordinal": ordinal,
                "identity": dry_window_identity(step),
                "label": label,
                "runner_exit": runner_exit,
                "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


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
    # Batch 2 W4: the floors/budget pins are plan facts the campaign session
    # reads from the step env only; an exported shell value is an operator
    # expecting it to matter (it never does: the step env wins, and a
    # non-driver cell would carry a pin the plan never reviewed). Refused on
    # PRESENCE before the first step, the STALE_INDEX_OPT_IN_ENV rule.
    for name in CELL_PIN_ENVS:
        if name in os.environ:
            raise RunError(
                f"{name} is set in the environment ({os.environ[name]!r}); the "
                "plan pins it on every cell step and the campaign session reads "
                f"only the plan's value ({SLO_FLOORS_FINDING}); unset it before "
                "'run'"
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
    # ADR-0153: (engine, demand-class seq_tokens) -> the dry window's label
    # when it did not read IN_REGIME; every pressure cell of the pair is then
    # skipped with a named sentinel and the exit code is EXIT_DRY_WINDOW_FAILED.
    dry_failed: Dict[Tuple[Any, ...], str] = {}
    dry_root = dry_window_root(campaign_root, plan)
    # V2 (engine handoff): the stops this run performs. Clean room first: every
    # launcher the plan uses is stopped once before the first step (a resident
    # engine of an aborted run); then the previous family before a relaunch of
    # another launcher; then the last family when the run ends. A failed stop
    # is printed and counted, never hidden; the cells still run (the next
    # relaunch, if any, is the recorded symptom when the GPU stayed held).
    stop_failures = 0

    def _stop(record: Mapping[str, Any], why: str) -> None:
        nonlocal stop_failures
        rc_stop = _exec(record["stop_argv"], record["env"])
        if rc_stop == 0:
            print(f"[run_campaign] engine stop ({why}): launcher={record['launcher_key']} ok")
        else:
            stop_failures += 1
            print(
                f"[run_campaign] STOP FAILED (exit {rc_stop}, {why}): launcher="
                f"{record['launcher_key']} argv={record['stop_argv']} "
                f"({ENGINE_STOP_FINDING}: the family may still hold the GPU)"
            )

    launchers = plan_launchers(steps)
    for record in launchers:
        _stop(record, "clean room before the first step")
    boundaries = stop_boundaries(steps)
    stops_before: Dict[int, Dict[str, Any]] = {
        b["before_index"]: b for b in boundaries if b["before_index"] is not None
    }
    # The family running NOW: set at every executed relaunch, stopped once at
    # the end of the run, on the happy path AND on an abort (review F1 and
    # F-2 of the V1 review: a Ctrl-C during the SGLang half of a plan must stop
    # SGLang, the resident family, not the plan's last family).
    resident: Optional[Dict[str, Any]] = None
    final_stop_done = False

    def _end_of_run_stops() -> None:
        nonlocal final_stop_done
        if final_stop_done or resident is None:
            return
        final_stop_done = True
        _stop(resident, "end of run")

    try:
        for index, step in enumerate(steps):
            if step["kind"] == "relaunch":
                if index in stops_before:
                    _stop(stops_before[index], "family change before the next relaunch")
                rc = _exec(step["argv"], step["env"])
                resident = _stop_record(step, None)
                server_ok = rc == 0
                if not server_ok:
                    print(
                        f"[run_campaign] RELAUNCH FAILED (exit {rc}): engine="
                        f"{step['engine']} prefix={step['prefix_mode']} "
                        f"budget_r={step.get('budget_r')}, failing its cells "
                        "until the next relaunch boundary"
                    )
                continue
            row_key, dataset = step["row_key"], step["dataset"]
            if step["kind"] == "dry_window":
                # ADR-0153: one window per (engine, demand class) at the
                # tightest r, DRY_WINDOW_RATE_FRAC x lambda*, in the sibling
                # dry root; its regime label gates the pair's pressure cells.
                key = dry_window_key(step["serving"], step["demand_class"])
                pair = f"engine={key[0]} prefix={key[1]} class={key[2]} tokens kv={key[3]} connector={key[4]} topology={key[5]}"
                if not server_ok:
                    dry_failed[key] = "relaunch-failed"
                    print(
                        f"[run_campaign] DRY WINDOW SKIPPED (relaunch failed): {pair} "
                        f"r={step['budget_r']}; its pressure cells fail ({DRY_WINDOW_ADR})"
                    )
                    outcomes.append(_Outcome(row_key, dataset, "dry-window-failed"))
                    continue
                # Reuse rule (review HIGH 2): an earlier attempt is reused only
                # when its label is IN_REGIME and its sidecar identity (rate,
                # lambda*, artifact, class, serving, budget) equals this
                # step's; anything else runs a FRESH attempt at the next
                # window ordinal, so no verdict is overwritten or inherited.
                attempts = _dry_attempt_ordinals(dry_root, step)
                label: Optional[str] = None
                doc: Dict[str, Any] = {}
                reused = False
                if attempts and not force_rerun:
                    last = attempts[-1]
                    label, doc = _read_dry_window_regime(dry_root, step, last)
                    side = _read_dry_sidecar(dry_root, step, last)
                    if label == DRY_WINDOW_EXPECTED_LABEL and side.get("identity") == dry_window_identity(step):
                        reused = True
                        print(
                            f"[run_campaign] dry window reused: {pair} attempt {last:02d} "
                            f"(same plan identity, label {label})"
                        )
                    else:
                        print(
                            f"[run_campaign] dry window attempt {last:02d} not reused "
                            f"(label={label!r}, identity_match={side.get('identity') == dry_window_identity(step)}): "
                            "running a fresh attempt"
                        )
                if not reused:
                    base = attempts[-1] if attempts else 0
                    env = dict(step["env"])
                    if base:
                        env["CAGE_WINDOW_ORDINAL_BASE"] = str(base)
                    argv = list(step["argv"]) + ["--campaign-root", str(dry_root)]
                    rc = _exec(argv, env)
                    label, doc = _read_dry_window_regime(dry_root, step, base + 1)
                    _write_dry_sidecar(dry_root, step, base + 1, label, rc)
                    if rc != 0:
                        print(f"[run_campaign] dry window runner exited {rc} ({row_key})")
                inputs = doc.get("inputs") if isinstance(doc.get("inputs"), dict) else {}
                summary = (
                    f"label={label!r} rho_kv_time_avg={inputs.get('rho_kv_time_avg')!r} "
                    f"queue_waiting_share={inputs.get('queue_waiting_share')!r} "
                    f"waiting_max={inputs.get('waiting_max')!r} attainment={doc.get('attainment')!r} "
                    f"refusal={doc.get('refusal_reason')!r}"
                )
                if label == DRY_WINDOW_EXPECTED_LABEL:
                    print(
                        f"[run_campaign] dry window ok: {pair} r={step['budget_r']} "
                        f"rate={step['offered_rate_rps']:.4g} rps {summary}"
                    )
                    outcomes.append(_Outcome(row_key, dataset, "dry-window-ok"))
                else:
                    dry_failed[key] = str(label or "no-regime")
                    print(
                        f"[run_campaign] DRY WINDOW FAILED: {pair} r={step['budget_r']} "
                        f"rate={step['offered_rate_rps']:.4g} rps expected "
                        f"{DRY_WINDOW_EXPECTED_LABEL}, {summary}; every pressure cell of this "
                        f"serving configuration is skipped ({DRY_WINDOW_ADR}, {DRY_WINDOW_FINDING}); "
                        "the budget-free cells still run"
                    )
                    outcomes.append(_Outcome(row_key, dataset, "dry-window-failed"))
                continue
            # Per-task RULER steps claim disjoint window-ordinal ranges within
            # a shared (row_key, dataset) space; resume counting and the
            # failure sentinel are both scoped to THIS step's range (base 0 =
            # legacy).
            base = int(step.get("window_ordinal_base") or 0)
            if step.get("blocked_on"):
                # Reaches here only under the operator's explicit
                # --skip-blocked: reported per-cell, executes nothing, gates a
                # plain --seal below.
                outcomes.append(_Outcome(row_key, dataset, "skipped-blocked"))
                continue
            if step.get("serving") is not None and not server_ok:
                _write_failed_sentinel(
                    campaign_root, row_key, dataset, "relaunch-failed", base
                )
                outcomes.append(_Outcome(row_key, dataset, "skipped-launch-failed"))
                continue
            if step.get("family") in _PRESSURE_FAMILIES and isinstance(step.get("demand_class"), dict):
                key = dry_window_key(step["serving"], step["demand_class"])
                if key in dry_failed:
                    _write_failed_sentinel(
                        campaign_root, row_key, dataset, f"dry-window-{dry_failed[key]}", base
                    )
                    outcomes.append(_Outcome(row_key, dataset, "skipped-dry-window"))
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
    finally:
        _end_of_run_stops()

    # ---- summary matrix (per-cell outcomes; the operator's at-a-glance) ----
    counts: Dict[str, int] = {}
    print("\n[run_campaign] ===== summary matrix =====")
    for o in outcomes:
        counts[o.outcome] = counts.get(o.outcome, 0) + 1
        print(f"  [{o.outcome:>21}] {o.row_key} ({o.dataset})")
    print(f"[run_campaign] totals: {counts}")
    # Review F8: the two components are printed apart because the header's
    # counts.engine_stops counts the boundaries only.
    print(
        f"[run_campaign] engine stops: clean room {len(launchers)}, boundaries "
        f"{len(boundaries)}, {stop_failures} failed ({ENGINE_STOP_FINDING})"
    )

    any_failed = any(
        o.outcome in ("failed", "skipped-launch-failed") for o in outcomes
    )
    any_blocked_skipped = any(o.outcome == "skipped-blocked" for o in outcomes)
    any_dry_failed = bool(dry_failed)
    if any_dry_failed:
        print(
            f"[run_campaign] dry windows failed for (engine, prefix, class tokens, kv, connector, "
            f"topology): {sorted(dry_failed.items(), key=str)} ({DRY_WINDOW_ADR}): their pressure cells were "
            "skipped with .STATUS sentinels; the plan's lambda* or demand class for those "
            "pairs does not reach the regime live"
        )
    if seal:
        # Blocked skips gate the seal exactly like failures: the tree is NOT
        # the full registered session, and a plain seal marks it done.
        if (any_failed or any_blocked_skipped or any_dry_failed) and not seal_partial:
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
            print(
                "[run_campaign] sealed."
                + (
                    f" {stop_failures} engine stop(s) failed; the pod may still hold "
                    "an engine (the data tree is complete)"
                    if stop_failures
                    else ""
                )
            )
    # V2: a failed stop never gates the seal (the data tree is complete) but
    # the exit code says the run did not leave the pod clean, with its own
    # value so a re-run operator can tell it from a failed cell (review F4).
    # ADR-0153: a failed dry window has its own value too (3): the executed
    # cells passed, but a whole (engine, class) of pressure cells never ran.
    if any_failed:
        return 1
    if any_dry_failed:
        if stop_failures:
            print(
                f"[run_campaign] exit {EXIT_DRY_WINDOW_FAILED} (dry window) also carries "
                f"{stop_failures} failed engine stop(s) ({ENGINE_STOP_FINDING})"
            )
        return EXIT_DRY_WINDOW_FAILED
    if stop_failures:
        return EXIT_STOP_FAILED
    return 0


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


def parse_calibration_args(
    items: Sequence[str], flag: str = "--calibration"
) -> Dict[str, Path]:
    """``ENGINE=PATH`` registrations -> {engine: path}; malformed or
    duplicate entries refuse (PlanError). Batch 2 W4; ADR-0154 reuses it for
    ``--rung-calibration``."""
    out: Dict[str, Path] = {}
    for item in items:
        engine, sep, raw_path = item.partition("=")
        if not sep or not engine.strip() or not raw_path.strip():
            raise PlanError(f"{flag} {item!r}: expected ENGINE=PATH")
        engine = engine.strip()
        if engine in out:
            raise PlanError(f"{flag} {engine}: registered twice")
        out[engine] = Path(raw_path.strip())
    return out


def parse_rungs_arg(text: Optional[str]) -> Optional[Tuple[float, ...]]:
    """``--rungs 1.5,1,0.75`` -> (1.5, 1.0, 0.75); None (every budgeted rung)
    when absent; a non-numeric, non-positive or repeated entry refuses."""
    if text is None or not text.strip():
        return None
    out: List[float] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            r = float(raw)
        except ValueError as exc:
            raise PlanError(f"--rungs {raw!r} is not a budget ratio") from exc
        if not math.isfinite(r) or r <= 0:
            raise PlanError(f"--rungs {raw!r} must be finite and > 0")
        if any(math.isclose(r, seen, rel_tol=0.0, abs_tol=1e-9) for seen in out):
            raise PlanError(f"--rungs {raw!r} is listed twice")
        out.append(r)
    if not out:
        raise PlanError(f"--rungs {text!r} names no rung")
    return tuple(out)


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
        calibrations=parse_calibration_args(args.calibration),
        calibration_budget_fraction=args.calibration_budget_fraction,
        rehearsal_n=args.rehearsal_n,
        rung_calibrations=parse_calibration_args(
            args.rung_calibration, flag="--rung-calibration"
        ),
    )
    text = json.dumps(plan, indent=2, sort_keys=False) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        c = plan["counts"]
        print(
            f"[run_campaign] plan written: {args.out} — {c['cells']} cells, "
            f"{c['windows']} windows, {c['relaunches']} relaunches, "
            f"{c['engine_stops']} engine stops, {c['dry_windows']} dry windows, "
            f"{c['blocked']} blocked"
        )
        ws = plan["window_spans"]
        if ws["executable_pressure_cells"]:
            print(
                f"[run_campaign] window spans ({WINDOW_SPAN_FINDING}): {ws['executable_pressure_cells']} "
                f"executable pressure cells, min {ws['min_s']:.3g} s, median {ws['median_s']:.3g} s, "
                f"max {ws['max_s']:.3g} s; {ws['below_floor']} below the {ws['floor_s']:g} s floor"
                + (" (WARNING: those windows are shorter than the registered warm-up transient)" if ws["below_floor"] else "")
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


# ---------------------------------------------------------------------------
# Rung calibration (ADR-0154, S0F-62): lambda* per budget rung, measured by the
# registered cal-v2 ladder on the gold-fresh F2 workload under the plan's OWN
# relaunch for that rung (prefix OFF, the gold class budget). The 2026-10-08
# landing offered rates from the floor table's assumed service time and ran 3
# to 6 times below the capacity its own telemetry showed; the ladder measures
# the capacity the cells are then driven at.
# ---------------------------------------------------------------------------

#: Default run root the ladder windows are written under (one §1 run root
#: per window: ``<out_root>/<session>/cal-<engine>-r<r>-s<k>``); never a
#: campaign tree, never read by the analysis (calibration data is
#: non-confirmatory by registration).
DEFAULT_RUNG_OUT_ROOT = "results/rung-calibration"
#: Rung labels the ladder mints beside decide_lambda_star's three: the rung's
#: relaunch failed, or a probe window left no readable requests.jsonl.
RUNG_LABEL_RELAUNCH_FAILED = "RELAUNCH_FAILED"
RUNG_LABEL_PROBE_FAILED = "PROBE_FAILED"


def budgeted_rungs(grid: SessionGrid) -> Tuple[float, ...]:
    """Every budget ratio r a pressure cell of the session is served at (the
    F2 factorial, the anchor fine overlay, F3), DESCENDING: the order the
    ladder calibrates them in, so each rung's start rate chains from the
    looser rung's lambda* (capacity cannot rise as the budget shrinks)."""
    return tuple(
        sorted(
            {float(r) for r in (*grid.f2_budgets, *grid.f2_fine_budgets, *grid.f3_budgets)},
            reverse=True,
        )
    )


def _gold_fresh_cell(grid: SessionGrid, engine: str, r: float) -> PlannedCell:
    """The gold-fresh F2 cell of (engine, r): the ladder's workload. The rate
    fraction is the first registered one (identity only: the ladder sets the
    rate per step); a session without the anchor arm on F2, or an engine
    whose gold cell is blocked, refuses."""
    bids = [bid for bid in grid.f2_baselines if BASELINES[bid].arm == DEMAND_ANCHOR_ARM]  # type: ignore[index]
    if not bids:
        raise PlanError(
            f"session {grid.session!r} registers no F2 baseline on the {DEMAND_ANCHOR_ARM} "
            f"arm; the rung ladder measures lambda* on it ({RUNG_CALIBRATION_ADR})"
        )
    if not grid.f2_rates:
        raise PlanError(f"session {grid.session!r} registers no F2 rate fraction")
    spec = CellSpec.from_baseline(
        bids[0],
        engine=engine,  # type: ignore[arg-type]
        model=grid.model,  # type: ignore[arg-type]
        family="F2",
        budget_r=r,
        rate_frac=grid.f2_rates[0],
    )
    cell = PlannedCell(
        spec=spec,
        baseline_id=bids[0],
        dataset=grid.f2_dataset,
        blocked_on=_launch_blocked_on(spec),
        grids=(GRID_D6_FACTORIAL,),
    )
    if cell.blocked_on is not None:
        raise PlanError(
            f"the {DEMAND_ANCHOR_ARM} F2 cell of engine {engine!r} is blocked "
            f"({cell.blocked_on}); no ladder can run on it"
        )
    return cell


def smallest_pressure_class(
    grid: SessionGrid, engine: str, manifests: Optional[Mapping[str, Any]] = None
) -> Optional[PlannedCell]:
    """ADR-0156: the engine's smallest executable demand class below the
    anchor, as ONE template cell (its arm, family, prefix mode and, for a B12
    rung, the rung); None when the engine serves the anchor class only.
    Executable = not launch-blocked and, for a corpus-trunc rung, a manifest
    registered for its dataset (build_plan's own rule). Ties on the token
    count break toward prefix OFF (a replayed ladder window is clean there)
    and the lower baseline number; the template's own budget_r is replaced
    per rung by ``_class_cell_at``."""
    anchor = int(grid.demand_seq_tokens[DEMAND_ANCHOR_ARM])
    best: Optional[Tuple[Tuple[int, int, int], PlannedCell]] = None
    for cell in enumerate_cells(grid):
        spec = cell.spec
        if spec.engine != engine or spec.family not in _PRESSURE_FAMILIES or cell.blocked_on is not None:
            continue
        if spec.arm == CORPUS_TRUNC_ARM and cell.dataset not in (manifests or {}):
            continue
        seq = demand_seq_tokens(grid, spec)
        if seq >= anchor:
            continue
        key = (seq, 0 if _prefix_off(spec) else 1, _baseline_num(cell.baseline_id))
        if best is None or key < best[0]:
            best = (key, cell)
    return None if best is None else best[1]


def _class_cell_at(template: PlannedCell, r: float) -> PlannedCell:
    """The template class cell re-minted at rung ``r`` (its rate fraction is
    identity only: the ladder sets the rate per step)."""
    spec = replace(template.spec, budget_r=r)
    return replace(
        template, spec=spec, blocked_on=_launch_blocked_on(spec), grids=None,
        ruler_task=None, window_ordinal_base=0,
    )


def ladder_root(out_root: Path, session: str, engine: str, r: float, k: int, role: str = "anchor") -> Path:
    """The §1 run root of ladder window ``k`` of rung ``r``:
    ``<out_root>/<session>/cal-<engine>-r<r with '.' as 'p'>-s<k:02d>`` for
    the anchor class and ``cal-<engine>-smallest-r<r>-s<k>`` for the smallest
    class (ADR-0156; the runner requires a RUN_ID_RE basename under a session
    directory)."""
    tag = "" if role == "anchor" else f"-{role}"
    name = f"cal-{engine}{tag}-r{f'{r:g}'.replace('.', 'p')}-s{k:02d}"
    if not RUN_ID_RE.match(name):
        raise PlanError(f"ladder root basename {name!r} violates the §1 grammar {RUN_ID_RE.pattern}")
    return Path(out_root) / session / name


def _read_ladder_window(
    root: Path, row_key: str, dataset: str, rate_qps: float, phase: str
) -> Tuple[ProbeStep, Optional[str]]:
    """One probe window's ProbeStep from the runner's requests.jsonl (the
    rows after the runner's Jain trim; trimmed again at PROBE_WARMUP_S on the
    intended arrival, idempotent) plus the window's regime label when the
    runner wrote regime.json (provenance; the decision is attainment's alone).
    A window with no requests.jsonl, a row without an intended arrival, or no
    post-warmup arrival raises RunError: the step is refused, never guessed."""
    window_dir = Path(root) / "cells" / row_key / f"window_{dataset}-01"
    path = window_dir / "requests.jsonl"
    if not path.is_file():
        raise RunError(f"ladder window at {rate_qps:.4g} qps wrote no {path}")
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    for i, row in enumerate(rows):
        arrival = row.get("arrival_s")
        if isinstance(arrival, bool) or not isinstance(arrival, (int, float)) or not math.isfinite(arrival):
            raise RunError(
                f"{path} row {i} carries arrival_s={arrival!r}; a ladder window is open-loop "
                "and every row names its intended arrival"
            )
    kept = [row for row in rows if float(row["arrival_s"]) >= PROBE_WARMUP_S]
    if not kept:
        raise RunError(
            f"no arrivals in the post-warmup window at {rate_qps:.4g} qps ({path}): the "
            f"{PROBE_WINDOW_S:g} s window is too short for this rate"
        )
    n_completed = sum(1 for row in kept if row.get("ok") is True)
    step = ProbeStep(
        rate_qps=float(rate_qps),
        n_scheduled=len(kept),
        n_completed=n_completed,
        throughput_rps=n_completed / (PROBE_WINDOW_S - PROBE_WARMUP_S),
        phase=phase,
    )
    label: Optional[str] = None
    regime_path = window_dir / "regime.json"
    if regime_path.is_file():
        try:
            doc = json.loads(regime_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            doc = None
        if isinstance(doc, dict) and isinstance(doc.get("label"), str):
            label = doc["label"]
    return step, label


def _set_flag(argv: List[str], flag: str, value: str) -> None:
    at = argv.index(flag)
    argv[at + 1] = value


def calibrate_rungs(
    session: str,
    engine: str,
    floor: FloorTable,
    *,
    calibration: Path,
    out_root: Path,
    rungs: Optional[Sequence[float]] = None,
    query_manifests: Optional[Mapping[str, Path]] = None,
    freeze_file: Optional[Path] = None,
    rehearsal_n: Optional[int] = None,
    seed: int = 42,
    runner_cmd: Sequence[str] = DEFAULT_RUNNER_CMD,
    launcher_cmds: Optional[Mapping[str, Sequence[str]]] = None,
    start_qps: Optional[float] = None,
    anchor_only: bool = False,
    exec_fn: Callable[[Sequence[str], Mapping[str, str]], int] = _exec,
    log: Callable[[str], None] = print,
    partial_out: Optional[Path] = None,
) -> Dict[str, Any]:
    """Measure lambda* per budget rung for ONE engine and return the
    ``cage-rung-calibration-v2`` artifact document (the caller writes it).
    With ``partial_out`` the document is also written after EVERY rung with
    ``complete: false`` (ADR-0162: a rung job killed at its bound after hours
    of probing left nothing behind on 2026-10-10); the loader refuses an
    incomplete artifact, so the partial file is forensics, never a plan input.

    Two classes, the ANCHOR first (ADR-0154) and then the engine's SMALLEST
    executable class below it (ADR-0156; skipped when the engine serves the
    anchor class only, or under ``anchor_only``). Per class and rung, loosest
    first: the plan's own budgeted relaunch of that class's serving config
    (``_relaunch_step``: prefix mode per the arm, the budget sized on the
    class's shape), then the registered cal-v2 ladder (``geometric_rate_ladder``
    from the start rate, climb until the first unsustainable window or
    PROBE_MAX_STEPS, then PROBE_BISECT_STEPS arithmetic midpoints), every
    window one runner invocation of the class cell (``_cell_step(ladder=True)``)
    at that rate into its own §1 run root under ``out_root``; the decision is
    ``decide_lambda_star`` over the steps sorted by rate. The anchor's first
    rung starts at ``start_qps`` (default: ``floor_start_qps`` of the engine's
    cal-v2 floor at LADDER_START_DECODE_TOKENS); each later rung starts at the
    previous ESTIMATED lambda* / LADDER_CHAIN_DIVISOR, else at the floor start
    again. The smallest class starts each rung at the HIGHER of the anchor's
    lambda* at that rung (a shorter sequence sustains at least the anchor's
    rate under the KV, compute and request-cap bounds; a NONE_SUSTAINABLE
    there is the loud symptom of a defective ladder) and the chained start.
    A failed relaunch or an unreadable window labels the rung (RELAUNCH_FAILED
    / PROBE_FAILED) and the ladder moves on; the engine is stopped at the end,
    success or abort. Pure of RunPod: ``exec_fn`` is the one seam (tests stub
    it with a runner that writes requests.jsonl).
    """
    grid = get_session_grid(session)
    if rehearsal_n is not None:
        grid = rehearsal_grid(grid, n=rehearsal_n, datasets=frozenset(query_manifests or {}))
    if engine not in _session_engines(grid):
        raise PlanError(
            f"--engine {engine!r} is not a server engine of session {session!r} "
            f"(registered: {sorted(_session_engines(grid))})"
        )
    if floor.model != grid.model:
        raise PlanError(
            f"floor table {floor.path} is for model {floor.model!r} but session "
            f"{session!r} runs {grid.model!r}"
        )
    if start_qps is not None and (
        isinstance(start_qps, bool) or not isinstance(start_qps, (int, float))
        or not math.isfinite(start_qps) or start_qps <= 0
    ):
        raise PlanError(f"--start-qps {start_qps!r} must be finite and > 0")
    registered = budgeted_rungs(grid)
    if rungs is None:
        rung_list: Tuple[float, ...] = registered
    else:
        for r in rungs:
            if not any(math.isclose(r, reg, rel_tol=0.0, abs_tol=1e-9) for reg in registered):
                raise PlanError(
                    f"--rungs {r:g} is not a budgeted rung of session {session!r} "
                    f"(registered: {[f'{x:g}' for x in registered]})"
                )
        rung_list = tuple(sorted({float(r) for r in rungs}, reverse=True))
    for r in rung_list:
        floor.row(r)
    pins = resolve_retrieval_pins(freeze_file)
    manifests = _register_query_manifests(grid, query_manifests)
    launcher_cmds = dict(launcher_cmds or DEFAULT_LAUNCHER_CMDS)
    anchor_tokens = _check_floor_shape(grid, floor)
    anchor_cells = [_gold_fresh_cell(grid, engine, r) for r in rung_list]
    small_template = None if anchor_only else smallest_pressure_class(grid, engine, manifests)
    small_cells = [] if small_template is None else [_class_cell_at(small_template, r) for r in rung_list]
    all_cells = anchor_cells + small_cells
    _check_manifest_coverage(grid, manifests, all_cells)
    # Gap triage 2026-10-09 C1: every ladder relaunch's pool is checked against
    # max_model_len BEFORE the first rung runs (a failure at the smallest
    # class's first relaunch would otherwise come after the anchor ladder).
    shortfalls = pool_shortfalls(
        [_relaunch_step(_serving_config(c, grid), grid, floor, launcher_cmds) for c in all_cells],  # type: ignore[arg-type]
        grid.max_model_len,
    )
    if shortfalls:
        raise PlanError(_pool_shortfall_text(shortfalls, f"calibrate-rungs {engine} session {session!r}"))
    cal_header = _register_calibrations(grid, {engine: Path(calibration)}, all_cells, FLOOR_BUDGET_FRACTION)
    floors = cal_header["floors"][engine]
    slo_floors_env = slo_floors_env_value(cal_header["floors"])
    floor_meas = FloorMeasurement(
        ttft_s=float(floors["ttft_s"]),
        tpot_s=float(floors["tpot_s"]),
        n_requests=int(floors["n_requests"]),
        statistic=str(floors["statistic"]),
    )
    floor_start = floor_start_qps(floor_meas, max_tokens=LADDER_START_DECODE_TOKENS)
    first_start = float(start_qps) if start_qps is not None else floor_start
    out_root = Path(out_root)

    anchor_records: Dict[str, Dict[str, Any]] = {}
    small_records: Dict[str, Dict[str, Any]] = {}
    resident: Optional[Dict[str, Any]] = None
    stop_failures = 0

    def _stop(record: Mapping[str, Any], why: str) -> None:
        nonlocal stop_failures
        rc_stop = exec_fn(record["stop_argv"], record["env"])
        if rc_stop == 0:
            log(f"[calibrate-rungs] engine stop ({why}): launcher={record['launcher_key']} ok")
        else:
            stop_failures += 1
            log(
                f"[calibrate-rungs] STOP FAILED (exit {rc_stop}, {why}): launcher="
                f"{record['launcher_key']} argv={record['stop_argv']}"
            )

    def _ladder(role: str, cells: Sequence[PlannedCell], records: Dict[str, Dict[str, Any]]) -> None:
        nonlocal resident
        previous_lambda: Optional[float] = None
        for cell in cells:
            r = float(cell.spec.budget_r)  # type: ignore[arg-type]
            key = f"{r:g}"
            seq_tokens = demand_seq_tokens(grid, cell.spec)
            config = _serving_config(cell, grid)
            assert config is not None
            relaunch = _relaunch_step(config, grid, floor, launcher_cmds)
            if resident is None:
                _stop(_stop_record(relaunch, None), "clean room before the first rung")
            elif resident["launcher_key"] != relaunch["launcher_key"]:
                _stop(resident, "family change before the next rung")
            log(
                f"[calibrate-rungs] {role} class {cell.spec.arm} ({seq_tokens} tokens) rung r={key}: "
                f"relaunch engine={engine} prefix={relaunch['prefix_mode']} "
                f"budget_bytes={relaunch.get('budget_bytes')}"
            )
            rc = exec_fn(relaunch["argv"], relaunch["env"])
            resident = _stop_record(relaunch, None)
            base_record: Dict[str, Any] = {
                "r": r,
                "class": {
                    "role": role,
                    "arm": cell.spec.arm,
                    "baseline_id": cell.baseline_id,
                    "family": cell.spec.family,
                    "prefix_mode": relaunch["prefix_mode"],
                    "seq_tokens": seq_tokens,
                    "corpus_budget_tokens": cell.spec.corpus_budget_tokens,
                },
                "relaunch": {
                    "argv": list(relaunch["argv"]),
                    "env": dict(relaunch["env"]),
                    "budget_bytes": relaunch.get("budget_bytes"),
                    "budget_plan": relaunch.get("budget_plan"),
                    "demand_class": relaunch.get("demand_class"),
                    "exit": rc,
                },
            }
            if rc != 0:
                log(f"[calibrate-rungs] RELAUNCH FAILED (exit {rc}) at r={key} ({role}); rung labeled")
                records[key] = {
                    **base_record,
                    "label": RUNG_LABEL_RELAUNCH_FAILED,
                    "lambda_star_qps": None,
                    "sustained_rate_qps": None,
                    "first_unsustainable_qps": None,
                    "start_qps": None,
                    "steps": [],
                }
                _flush_partial()
                previous_lambda = None
                continue
            manifest_rec = manifests.get(cell.dataset)
            cell_step = _cell_step(
                cell, grid, floor, runner_cmd, seed, pins,
                query_manifest=None if manifest_rec is None else manifest_rec["path"],
                slo_floors_env=slo_floors_env,
                budget_plan=relaunch["budget_plan"],
                ladder=True,
            )
            candidates: List[Tuple[float, str]] = []
            if previous_lambda is not None:
                candidates.append((
                    previous_lambda / LADDER_CHAIN_DIVISOR,
                    f"previous rung lambda* {previous_lambda:.4g} / {LADDER_CHAIN_DIVISOR:g}",
                ))
            if role != "anchor":
                # Fable review 2026-10-09 HIGH 1: the anchor's ESTIMATED lambda*
                # is the LAST sustainable step of ITS ladder (attainment just
                # above 0.9); a class with the same capacity (scheduler-bound
                # at loose r, S0F-70) started exactly there reads 0.88 by
                # Poisson variance and ends NONE_SUSTAINABLE. The small ladder
                # starts one chain step below, the rule tighter rungs use.
                anchor_rec = anchor_records.get(key) or {}
                anchor_lambda = anchor_rec.get("lambda_star_qps")
                if isinstance(anchor_lambda, (int, float)) and anchor_lambda > 0:
                    candidates.append((
                        float(anchor_lambda) / LADDER_CHAIN_DIVISOR,
                        f"anchor lambda* {anchor_lambda:.4g} at r={key} / {LADDER_CHAIN_DIVISOR:g} "
                        "(a shorter sequence sustains at least the anchor's rate; one chain "
                        "step below it so a same-capacity class is not read unsustainable)",
                    ))
            if candidates:
                start, start_basis = max(candidates)
            else:
                start = first_start
                start_basis = (
                    f"--start-qps {first_start:.4g}" if start_qps is not None
                    else f"{START_QPS_RULE}: floor_start_qps at {LADDER_START_DECODE_TOKENS} decode tokens"
                )
            steps: List[ProbeStep] = []
            step_records: List[Dict[str, Any]] = []
            k = 0

            def _probe(rate: float, phase: str) -> ProbeStep:
                nonlocal k
                root = ladder_root(out_root, session, engine, r, k, role)
                k += 1
                if root.exists():
                    raise RunError(
                        f"ladder root {root} already exists; calibration roots are never "
                        "reused (pick another --out-root)"
                    )
                argv = list(cell_step["argv"])
                _set_flag(argv, "--rate", f"{rate:.6g}")
                argv += ["--campaign-root", str(root)]
                env = dict(cell_step["env"])
                env[LADDER_REPLAY_ENV] = "1"
                log(f"[calibrate-rungs]   {phase} window: {rate:.4g} qps for {PROBE_WINDOW_S:g} s -> {root}")
                rc_win = exec_fn(argv, env)
                step, regime_label = _read_ladder_window(
                    root, cell_step["row_key"], cell_step["dataset"], rate, phase
                )
                log(
                    f"[calibrate-rungs]   attainment={step.attainment:.3f} "
                    f"({step.n_completed}/{step.n_scheduled}) throughput={step.throughput_rps:.4g} rps "
                    f"regime={regime_label} runner_exit={rc_win}"
                )
                step_records.append({
                    **step.to_manifest(),
                    "regime_label": regime_label,
                    "runner_exit": rc_win,
                    "root": str(root),
                    "trim_warmup_s": PROBE_WARMUP_S,
                    "replay_env": f"{LADDER_REPLAY_ENV}=1",
                })
                return step

            try:
                for rate in geometric_rate_ladder(start):
                    step = _probe(rate, "ladder")
                    steps.append(step)
                    if step.attainment < PROBE_ATTAINMENT_MIN:
                        break
                if len(steps) >= 2 and steps[-1].attainment < PROBE_ATTAINMENT_MIN:
                    lo, hi = steps[-2].rate_qps, steps[-1].rate_qps
                    for _ in range(PROBE_BISECT_STEPS):
                        mid = (lo + hi) / 2.0
                        step = _probe(mid, "bisect")
                        steps.append(step)
                        if step.attainment >= PROBE_ATTAINMENT_MIN:
                            lo = mid
                        else:
                            hi = mid
                estimate = decide_lambda_star(sorted(steps, key=lambda s: s.rate_qps))
            except (RunError, CalibrationError) as exc:
                log(f"[calibrate-rungs] PROBE FAILED at r={key} ({role}): {exc}; rung labeled")
                records[key] = {
                    **base_record,
                    "label": RUNG_LABEL_PROBE_FAILED,
                    "lambda_star_qps": None,
                    "sustained_rate_qps": None,
                    "first_unsustainable_qps": None,
                    "start_qps": start,
                    "start_basis": start_basis,
                    "error": str(exc),
                    "steps": step_records,
                }
                _flush_partial()
                previous_lambda = None
                continue
            log(
                f"[calibrate-rungs] {role} rung r={key}: {estimate.label} lambda*="
                f"{estimate.lambda_star_qps} sustained={estimate.sustained_rate_qps} "
                f"first_unsustainable={estimate.first_unsustainable_qps} ({len(steps)} windows)"
            )
            records[key] = {
                **base_record,
                "label": estimate.label,
                "lambda_star_qps": estimate.lambda_star_qps,
                "sustained_rate_qps": estimate.sustained_rate_qps,
                "first_unsustainable_qps": estimate.first_unsustainable_qps,
                "start_qps": start,
                "start_basis": start_basis,
                "steps": step_records,
                "cell_row_key": cell_step["row_key"],
                "cell_argv_sha256": hashlib.sha256(
                    json.dumps(cell_step["argv"], separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            }
            previous_lambda = estimate.lambda_star_qps
            _flush_partial()

    def _build_doc(complete: bool) -> Dict[str, Any]:
        gold_cell = anchor_cells[0]
        classes: Dict[str, Dict[str, Any]] = {
            str(anchor_tokens): {
                "role": "anchor",
                "arm": DEMAND_ANCHOR_ARM,
                "baseline_id": gold_cell.baseline_id,
                "family": "F2",
                "prefix_mode": "OFF",
                "seq_tokens": anchor_tokens,
                "dataset": gold_cell.dataset,
                "rungs": anchor_records,
            }
        }
        if small_cells:
            small = small_cells[0]
            small_seq = demand_seq_tokens(grid, small.spec)
            classes[str(small_seq)] = {
                "role": "smallest",
                "arm": small.spec.arm,
                "baseline_id": small.baseline_id,
                "family": small.spec.family,
                "prefix_mode": "OFF" if _prefix_off(small.spec) else "ON",
                "seq_tokens": small_seq,
                "dataset": small.dataset,
                "corpus_budget_tokens": small.spec.corpus_budget_tokens,
                "rungs": small_records,
                "note": (
                    None if _prefix_off(small.spec) else
                    "a prefix-ON class replays its pool with the cache warm, so its ladder "
                    "over-reads the sustainable rate [D]"
                ),
            }
        return {
            "schema": RUNG_CALIBRATION_SCHEMA,
            "procedure_version": PROCEDURE_VERSION,
            "confirmatory": False,
            # ADR-0162: false on the per-rung partial writes; the loader refuses them
            "complete": complete,
            "adr": RUNG_CALIBRATION_ADR,
            "two_anchor_adr": TWO_ANCHOR_ADR,
            "finding": RUNG_CALIBRATION_FINDING,
            "engine": BACKEND_OF_ENGINE[engine],
            "model": HF_ID_OF_SLUG[grid.model],
            "session": session,
            "rehearsal_n": rehearsal_n,
            "seed": seed,
            "anchor_seq_tokens": anchor_tokens,
            "workload": {
                "arm": DEMAND_ANCHOR_ARM,
                "baseline_id": gold_cell.baseline_id,
                "dataset": gold_cell.dataset,
                "num_queries": cell_num_queries(grid, gold_cell)[1],
                "seq_tokens": anchor_tokens,
                "prefix_mode": "OFF",
                "query_manifest": None if manifests.get(gold_cell.dataset) is None else manifests[gold_cell.dataset]["path"],
                "window_mode": "duration",
                "replay": True,
                "replay_note": (
                    "the pool is replayed modulo under CAGE_ALLOW_REPLAY=1, the cal-v2 probe's "
                    "own convention (calibrate_cell.probe_rate); the V3 pool guard refuses an "
                    "arrival count above the prepared pool, so calibration windows run in "
                    "duration mode and never enter confirmatory analysis"
                ),
            },
            "ladder": {
                "window_s": PROBE_WINDOW_S,
                "warmup_s": PROBE_WARMUP_S,
                "factor": PROBE_LADDER_FACTOR,
                "max_steps": PROBE_MAX_STEPS,
                "bisect_steps": PROBE_BISECT_STEPS,
                "attainment_min": PROBE_ATTAINMENT_MIN,
                "start_rule": START_QPS_RULE,
                "start_decode_tokens": LADDER_START_DECODE_TOKENS,
                "chain_divisor": LADDER_CHAIN_DIVISOR,
                "floor_start_qps": floor_start,
                "first_start_qps": first_start,
                "floor": dict(floors),
                "floor_artifact": cal_header["artifacts"][engine],
            },
            "interpolation": {
                "adr": TWO_ANCHOR_ADR,
                "rule": (
                    "lambda(s) = lambda_g x (s_g / s)^alpha, alpha = ln(lambda_m / lambda_g) / "
                    "ln(s_g / s_m) per rung; the plan computes alpha from the two anchors above"
                ),
                "anchor_only": anchor_only,
            },
            "stop_failures": stop_failures,
            "rungs": anchor_records,
            "classes": classes,
        }

    def _flush_partial() -> None:
        # ADR-0162: every finished rung lands on disk at once; a job killed at
        # its bound leaves the rungs it measured, marked incomplete.
        if partial_out is None:
            return
        partial_out.parent.mkdir(parents=True, exist_ok=True)
        partial_out.write_text(
            json.dumps(_build_doc(False), indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )

    try:
        _ladder("anchor", anchor_cells, anchor_records)
        if small_cells:
            _ladder("smallest", small_cells, small_records)
    finally:
        if resident is not None:
            _stop(resident, "end of calibration")

    return _build_doc(True)


def _cmd_rungs(args: argparse.Namespace) -> int:
    grid = get_session_grid(args.session)
    if args.rehearsal_n is not None:
        grid = rehearsal_grid(
            grid, n=args.rehearsal_n,
            datasets=frozenset(parse_query_manifest_args(args.query_manifest)),
        )
    print(" ".join(f"{r:g}" for r in budgeted_rungs(grid)))
    return 0


def _cmd_calibrate_rungs(args: argparse.Namespace) -> int:
    floor = load_floor_table(Path(args.floor_table))
    launcher_cmds: Optional[Dict[str, Tuple[str, ...]]] = None
    if args.launcher_cmd:
        override = tuple(shlex.split(args.launcher_cmd))
        launcher_cmds = {engine: override for engine in DEFAULT_LAUNCHER_CMDS}
    out = Path(args.out)
    doc = calibrate_rungs(
        args.session,
        args.engine,
        floor,
        calibration=Path(args.calibration),
        out_root=Path(args.out_root),
        rungs=parse_rungs_arg(args.rungs),
        query_manifests=parse_query_manifest_args(args.query_manifest),
        freeze_file=Path(args.freeze_file) if args.freeze_file else None,
        rehearsal_n=args.rehearsal_n,
        seed=args.seed,
        runner_cmd=tuple(shlex.split(args.runner_cmd)),
        launcher_cmds=launcher_cmds,
        start_qps=args.start_qps,
        anchor_only=args.anchor_only,
        partial_out=out,  # ADR-0162: the same file, incomplete until this final write
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    not_estimated: Dict[str, str] = {}
    for seq, cls in doc["classes"].items():
        for key, rec in cls["rungs"].items():
            if rec["label"] != "ESTIMATED":
                not_estimated[f"{cls['role']} {seq} tokens r={key}"] = rec["label"]
        print(
            f"[calibrate-rungs] artifact written: {out} engine={args.engine} {cls['role']} class "
            f"{cls['arm']} ({seq} tokens) rungs="
            + ", ".join(
                f"r={key}: {rec['label']}"
                + (f" lambda*={rec['lambda_star_qps']:.4g} rps" if rec["lambda_star_qps"] else "")
                for key, rec in cls["rungs"].items()
            )
        )
    if not_estimated:
        print(
            f"[calibrate-rungs] NOT ESTIMATED: {not_estimated}; plan --rung-calibration "
            f"refuses these rungs (LADDER_EXHAUSTED: raise --start-qps or the rung is "
            "beyond PROBE_MAX_STEPS windows; NONE_SUSTAINABLE: lower --start-qps; "
            "RELAUNCH_FAILED / PROBE_FAILED: read the runner and launcher output)",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_campaign",
        description=(
            "THE campaign sweep driver: 'plan' enumerates a session's "
            "registered D6 grid into a reviewable execution plan (pure, "
            "offline); 'run' executes a plan against a campaign root; "
            "'rungs' lists the session's budgeted rungs; 'calibrate-rungs' "
            "measures lambda* per rung on the pod for one engine (ADR-0154)."
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
        help="pre-costed measurement-window duration ESTIMATE for pressure "
        "cells, recorded in the plan header for the cost model (§6.1; REQUIRED, "
        "never defaulted). It bounds nothing: since Batch 1 V3 every window "
        "cell is bounded by --arrival-count equal to its --num-queries",
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
    p_plan.add_argument(
        "--calibration",
        action="append",
        default=[],
        metavar="ENGINE=PATH",
        help="register the cal-v1 floor artifact (scripts/3_run/calibrate_cell.py "
        "--output) of one server engine; repeatable, REQUIRED for every engine "
        "with an executable cell. The floors are validated, recorded in the "
        "header 'calibration' with their sha256, pinned on every cell step as "
        f"{SLO_FLOORS_ENV} and written into manifest.json['slo_floors'] by the "
        f"campaign session ({SLO_FLOORS_FINDING}, {SLO_FLOORS_ADR})",
    )
    p_plan.add_argument(
        "--calibration-budget-fraction",
        type=float,
        default=FLOOR_BUDGET_FRACTION,
        metavar="R",
        help="the KV budget ratio every registered floor artifact must have been "
        f"measured at (charter §6.1: r = {FLOOR_BUDGET_FRACTION:g}, concurrency 1); "
        "pass another value ONLY to register a shakedown rung explicitly (recorded "
        "in the header beside the registered value)",
    )
    p_plan.add_argument(
        "--rehearsal-n",
        type=int,
        default=None,
        metavar="N",
        help="derive the DRESS REHEARSAL of --session (ADR-0144): every "
        "baseline, engine, HF-oracle cell, RULER task and B12 rung of the "
        "registered grid, datasets restricted to the registered manifests, F2 "
        "and F3 collapsed to one coordinate each (plus one fine-only F2 "
        "coordinate on the anchor), one window per cell, every row class at "
        "n = N; recorded in the header 'rehearsal'. Never for a registered run",
    )
    p_plan.add_argument(
        "--rung-calibration",
        action="append",
        default=[],
        metavar="ENGINE=PATH",
        help="register the cage-rung-calibration-v1 artifact (calibrate-rungs "
        "--out) of one engine; repeatable, REQUIRED for every engine with an "
        "executable pressure cell: every (engine, r) of those cells must be "
        "ESTIMATED, else the plan refuses. The offered rate of every pressure "
        "cell is rate_frac x that lambda* (scaled to the cell's demand class); "
        f"the floor table's KV-bound rate is recorded, never offered ({RUNG_CALIBRATION_FINDING}, "
        f"{RUNG_CALIBRATION_ADR})",
    )
    p_plan.add_argument("--out", default=None, help="write the plan JSON here (else stdout)")
    p_plan.set_defaults(func=_cmd_plan)

    p_rungs = sub.add_parser(
        "rungs", help="print the session's budgeted rungs, descending (pure)"
    )
    p_rungs.add_argument("--session", required=True, help=f"one of {sorted(SESSIONS)}")
    p_rungs.add_argument("--rehearsal-n", type=int, default=None, metavar="N")
    p_rungs.add_argument(
        "--query-manifest", action="append", default=[], metavar="DATASET=PATH",
        help="the registered manifests (only the rehearsal derivation reads them)",
    )
    p_rungs.set_defaults(func=_cmd_rungs)

    p_cal = sub.add_parser(
        "calibrate-rungs",
        help="measure lambda* per budget rung for one engine on the pod (ADR-0154)",
    )
    p_cal.add_argument("--session", required=True, help=f"one of {sorted(SESSIONS)}")
    p_cal.add_argument("--engine", required=True, help="the server engine to calibrate")
    p_cal.add_argument("--floor-table", required=True, help="the T2.4 floor table (demand per r)")
    p_cal.add_argument(
        "--calibration", required=True, metavar="PATH",
        help="this engine's cal-v2 floor artifact (calibrate_cell.py --output): the §6.1 "
        "floors pinned on every ladder window and the ladder's start rate",
    )
    p_cal.add_argument(
        "--rungs", default=None, metavar="R,R,...",
        help="the budget rungs to calibrate (default: every budgeted rung of the session, "
        "see 'rungs'); calibrated loosest first",
    )
    p_cal.add_argument("--query-manifest", action="append", default=[], metavar="DATASET=PATH")
    p_cal.add_argument("--freeze-file", default=None)
    p_cal.add_argument("--rehearsal-n", type=int, default=None, metavar="N")
    p_cal.add_argument("--seed", type=int, default=42)
    p_cal.add_argument(
        "--start-qps", type=float, default=None,
        help="override the first rung's start rate (default: floor_start_qps of the "
        f"cal-v2 floor at {LADDER_START_DECODE_TOKENS} decode tokens); later rungs chain "
        f"from the previous lambda* / {LADDER_CHAIN_DIVISOR:g}",
    )
    p_cal.add_argument(
        "--anchor-only", action="store_true",
        help=f"skip the smallest-class ladder ({TWO_ANCHOR_ADR}); a plan whose engine serves a "
        "class below the anchor then REFUSES the artifact (diagnostics only)",
    )
    p_cal.add_argument("--runner-cmd", default=" ".join(DEFAULT_RUNNER_CMD))
    p_cal.add_argument("--launcher-cmd", default=None)
    p_cal.add_argument(
        "--out-root", default=DEFAULT_RUNG_OUT_ROOT,
        help="the ladder windows' run roots go under <out-root>/<session>/ (never a "
        "campaign tree; roots are never reused)",
    )
    p_cal.add_argument("--out", required=True, help="write the rung-calibration artifact here")
    p_cal.set_defaults(func=_cmd_calibrate_rungs)

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
