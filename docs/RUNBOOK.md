# CAGE — Campaign Runbook (cloud execution, RunPod-first)

> **Authority.** `MyDocs/PUBLICATION.md` is THE design authority (groups, arms, engines,
> matrices — verify section numbers by grepping it). This file is the *execution*
> authority: how a provisioning session is actually brought up, validated, run, drained,
> and torn down. **RunPod is the PRIMARY campaign provider** (owner directive
> 2026-08-18; FINAL SCOPE v2 in `MyDocs/COST_NEBIUS_RUNPOD_2026-08-16.md`). GCP is a
> **retained port**, not the current path — see Appendix A.
>
> ⚠️ **APPROVAL GATE — READ FIRST. Nothing provisions without an explicit user "go".**
> No pod launch, no `terraform apply`, no bucket/network-volume creation — ever — as a
> side effect of preparation. Prepare plans, print the cost+ETA report, then STOP and
> wait for the user. This is a standing, binding discipline, not a preference.

---

## 0. Standing disciplines (binding on every session)

1. **Approval gate** — provisioning (any resource that bills or persists) happens only
   after an explicit user go, per session AND per act. Plans/dry-runs are fine.
2. **Clean-room infra per run** — every run gets fresh infrastructure: a fresh backup
   destination named for the run, no reuse of past-run pods, volumes, buckets, or
   leftover state. A surviving bucket/volume from a previous run counts as a violation.
3. **Pull local BEFORE teardown, fail-closed** — results must exist off-box AND locally,
   ledger-verified, before anything is deleted (§5). Pulling after teardown is one
   failed sync away from data loss; that ordering is forbidden.
4. **Teardown to TRUE $0** — pods, network volumes, buckets. Prove it with a read-only
   listing (teardown_pod.sh step [5/5]; `gpu_vm.sh sweep` on the GCP port).
5. **Cost + ETA on every cloud action** — §7 format. No cloud command in a report
   without its price and its clock.
6. **Validate infra before every run** — the live preflight (§3) on the actual
   provisioned box, on every session and after every engine restart. No mock, no
   cached green.

## 1. Lifecycle at a glance

```
[ship]      scripts/ops/package_repo.sh  ->  tarball + BUILD_INFO  ->  pod ~/CAGE
[setup]     bash scripts/runpod/setup_runpod.sh              (container-shaped, B1 interpreter)
[preflight] bash scripts/checks/preflight_check.sh <MODEL> <API_BASE>   (gates (a)-(q))
[run]       nohup bash scripts/3_run/run_full_sweep.sh <model> <N> <T> > sweep.log 2>&1 &
            (or cloud_run.sh for the core tree alone — see the honesty note in §4)
[sync]      scripts/5_observability/sync_results.sh + gcs_backup_daemon.sh + collect_logs.sh
            (all through scripts/lib/transport.sh: gs:// | s3:// | ssh:// | file://)
[pull+$0]   scripts/runpod/teardown_pod.sh <pod_id> <backup_target> <local_run_dir>
            (ledger-gated pull_run.sh FIRST, delete LAST, read-only $0 listing)
```

Sessions and cell carriage are `MyDocs/PUBLICATION.md` §7.6/§7.6.1; FINAL SCOPE v2
(the RunPod plan: which runs, which pods, the L40S S0 gate) is
`MyDocs/COST_NEBIUS_RUNPOD_2026-08-16.md`. Engine pins and the engine×model
VERIFY-LIVE matrix: `docs/VLLM_COMPATIBILITY.md` (§7; act-2 RDMA preflight §8).

### 1.1 Session vocabulary — the four session ids (tracked definition)

The campaign runs as **three provisioning sessions**; C and D share one session in
two acts. The session id is a path level of every results tree
(`results/<campaign>/<session>/<run_id>/`, `docs/RESULTS_LAYOUT.md` §1) and is
pinned in code as `SESSIONS` (`scripts/4_analysis/organize_results.py` /
`src/orchestration/campaign_layout.py`). The **only legal values**:

| Session id | Group / model | Engines | Mission (full detail: PUBLICATION.md §7.6) |
|---|---|---|---|
| `a` | A — Qwen3-14B (anchor) | all 4 (vLLM, SGLang, LMDeploy-TurboMind, HF oracle) | full controlled grid: every arm, every engine, all datasets; FULL D6 factorial + fine r-grid |
| `b` | B — Llama-3.3-70B | all 4 | pressure at scale (TP=4): FRESH → F2 grid, REUSE → F3; + SCBench slice |
| `cd-act1` | C — Qwen3-Next-80B | vLLM + SGLang (+ HF oracle; LMDeploy absent per P7) | disaggregation PROTOCOL cost isolated from the network (single node, intra-node TP + PD) |
| `cd-act2` | D — DeepSeek-V3-0324 | vLLM + SGLang (HF EXEMPT per D4) | transfer cost + dedup-over-the-wire, cross-node TCP rung → RDMA/RoCE rung; MLA×TP |

Each session: approval → provision → preflight → run cells → pull-verify →
teardown-$0. Act-1 → act-2 transition: preflight ONCE on the first node (act 1),
then scale out for act 2 — act 2 has its OWN additional gate (the RDMA preflight,
`docs/VLLM_COMPATIBILITY.md` §8) and its own user approval. Hardware shapes and
pod choices are FINAL SCOPE v2 (`MyDocs/COST_NEBIUS_RUNPOD_2026-08-16.md`); the
GCP-fallback shapes live in Appendix A.

## 2. Ship + setup (on the pod)

The pod tree is a **tarball, not a git clone** (provenance via `BUILD_INFO`; a clone
would record `sha=null` when git is absent):

```bash
# workstation
scripts/ops/package_repo.sh                    # -> /tmp/cage_<sha8>.tar.gz, warns if tree dirty
scp ${CAGE_SSH_OPTS:-} /tmp/cage_<sha8>.tar.gz <user@pod>:~   # RunPod SSH, often non-standard port
# pod
[ ! -d ~/CAGE ] || [ -L ~/CAGE ] || { echo '~/CAGE is a real directory: remove it before linking'; false; }
mkdir -p /workspace/CAGE && ln -sfn /workspace/CAGE ~/CAGE   # the network volume; ~/CAGE stays the path every script cites
tar xzf cage_*.tar.gz -C ~/CAGE
head -3 ~/CAGE/BUILD_INFO                      # verify sha/dirty/packaged_at
cd /workspace/CAGE && bash scripts/runpod/setup_runpod.sh   # every pod shell: the PHYSICAL path
source cage-env/bin/activate
```

One path spelling on every pod shell: `cd /workspace/CAGE`. Bash keeps the `~/CAGE`
symlink spelling while Python resolves it physically, and two prefix strips in the
tree assume one spelling (integration audit 2026-09-26, pod-10); the symlink stays
for the scripts that cite `~/CAGE`.

Pod shape (S0F-18, live 2026-09-30): the image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
runs its own nginx on 3001, 7270, 7861, 8001, 8081 and 9091, forwarding to 3000, 7271,
7860, 8000, 8080 and 9090; its 502 page answers HTTP 200, so a health probe on one of
those ports can read 200 with nothing of ours behind it. The cluster manager's default
`--base-port` is therefore 8101 (replicas take 8101 to 8100+N; router 9000), it refuses
a bound port before any launch and names the holder, and a replica or router that exits
during its wait fails the start at once with its log tail. Knobs: `CAGE_CLUSTER_BASE_PORT`
and `CAGE_CLUSTER_ROUTER_PORT` for `run_tests.sh --with-cluster`, `VLLM_START_TIMEOUT`
and `ROUTER_START_TIMEOUT` for the readiness budgets (the same knob the engine launchers
read). Two or more replicas on one GPU need `CAGE_KV_BUDGET_BYTES_REPLICA` (ADR-0124).

Venv siting (ADR-0125, S0F-6): the bootstrap builds the three venvs at their real
paths under `CAGE_VENV_ROOT` (default `/root/cage-venvs`, the container disk) and links
`cage-env`, `sglang-env` and `lmdeploy-env` at the repo root to them, so every command
above keeps its spelling. On a volume-backed pod the repo is on the MooseFS network
volume, and a venv there cost S0 20 s per `import vllm`, 4 to 7.5 min per engine start
and 79 min per bootstrap (1.8 to 2.4 h of pod time per day). `logs/` and `results/` stay
on the volume. A pod restart wipes the container disk; rerun the bootstrap and it rebuilds
the venvs and re-points the links. `provision_pod.sh` sizes the container disk at 120 GB.

`setup_runpod.sh` is container-shaped (root, no sudo/systemd/PPA — finding J7): it
installs the pinned vLLM + `requirements.txt` into `cage-env` built from the
**canonical interpreter** (`CAGE_CANONICAL_PYTHON`, finding B1 — it fails closed rather
than fall back to bare `python3`; never hand-build the venv with `python3 -m venv`),
exports `HF_HUB_DOWNLOAD_TIMEOUT` BEFORE dataset staging and model prefetch, stages the
full charter dataset roster (D5), and prefetches the FINAL-SCOPE model roster
(override per pod role: `PREFETCH_MODELS="Qwen/Qwen3-14B" bash scripts/runpod/setup_runpod.sh`).

Long-job discipline (kept from the pilots — still true): never run a long command
through a blocking SSH. `scripts/gcp/remote_job.sh` gives submit/status/tail/wait/kill
with a durable remote PID + state file; pair every run with `nohup ... &`.

## 3. Preflight — Gate 2, gates (a)–(q) (a failing gate = do NOT launch)

Start the serving engine (`scripts/2_serving/manage_vllm_server.sh`, or the
`manage_sglang_server.sh` / `manage_lmdeploy_server.sh` launchers for those backends),
then:

```bash
bash scripts/checks/preflight_check.sh <MODEL> <API_BASE>    # exit 0 = safe to launch
```

The script codifies the user-mandated live-infra validation as gates **(a)–(q)** — no
mock, no cached green. Abbreviated (the script header is the authority): (a) serving
health + model listed + `/reset_prefix_cache` 200, (b) quality layer scores a REAL
pair, (c) cage-stats importable, (d) FAISS + embedding + reranker, (e) no
mock/disable/unrecorded-deviation env var set (incl. `CAGE_ALLOW_NO_BACKUP`,
`CAGE_QUALITY_STRICT` poison values, `CAGE_CLAIM_CHECKER` state), (f) disk space,
(g) vllm CLI importable at the venv level, (h) D2 telemetry parity, (i)
environment-vs-registration pins (a `pip check` line listed in
`scripts/checks/pip_check_allowlist.txt` with its reason and ADR id is printed as an
accepted deviation; today the one vLLM 0.19.1 / numba 0.67.0 line, ADR-0126; any
other line fails), (j) charter §6.5 realized-KV **iso-BYTES parity**
across engine startup logs (`CAGE_ISO_BYTES_TOL`/`CAGE_ISO_BYTES_LOGS`), (k)
per-backend endpoint liveness (`CAGE_PREFLIGHT_BACKENDS`), (l) campaign-layout
round-trip, (m) open-loop schedule + measured-replay guard, (n) calibration artifact,
(o) regime-inputs bridge on live telemetry, (p) dataset staleness refusal.

Qasper (ADR-0127, S0F-5): `datasets` 4.x refuses the repo's loading script, so the
loader and `download_datasets.py` read the Hub's parquet export at one pinned commit
(`QASPER_REVISION`, `src/data/loader.py`; `DATASET_REVISIONS`,
`scripts/1_setup/download_datasets.py`). The bootstrap proves the route on every pod
after staging (`CAGE_HF_LIVE=1 python -m pytest tests/test_qasper_revision_s0f5.py -m
integration`: the rebuilt 50x3 manifest must hash to the tracked
`data/manifests/qasper_50x3_seed42.json`). The cache directory stays `allenai___qasper`,
so gate (p) is unchanged. In offline mode `datasets` ignores `revision`, which is why
both call sites carry the same commit.

Prefill/decode pair (ADR-0128, S0F-13): `manage_vllm_pd.sh start` refuses, before its
self-cleaning stop, when the serving interpreter cannot import `nixl._api` /
`nixl._bindings` or a top-level `nixl_ep` is present but broken; each role gets its own
`VLLM_NIXL_SIDE_CHANNEL_PORT` (5600 / 5601); both roles are awaited in one loop that
fails at once with the log tail when a role process is gone. Operating rule:
never run a pd `stop` while another chain's `start` is live. The pidfiles are shared
and the stop sweeps by command pattern, so a stop from one shell kills the newest
stack, whichever shell started it (S0: attempt 3 died at 16:45:10 to attempt 2's stop). The Run-C-prime
pd test runs with `UCX_LOG_LEVEL=info` so the selected transports are in the log; the
launcher records the operator's `UCX_*` values per role and sets none.

Version pins: record the actually-served engine versions into the run manifest; the
engine×model VERIFY-LIVE matrix is `docs/VLLM_COMPATIBILITY.md` §7. Re-run gate (a)
after **every** engine relaunch (prefix ON/OFF, policy knobs, and topology are
launch-time levers — relaunches between cells are normal).

### 3.1 Budget calibration — bytes → knobs → gate (j) (per engine × budget level)

Charter P2 makes the pressure axis a BYTE quantity, but the engines take dialect
knobs. The planner emits them; gate (j) verifies them; this loop binds the two.
Run it once per (model, engine, budget level) BEFORE the level's first cell:

```bash
# 1. PLAN — bytes + per-engine knobs + the expected realized bytes
.venv/bin/python -m src.orchestration.cache_budget \
  --model qwen3-14b --engine vllm --r 0.5 \
  --concurrency <target-c> --avg-seq-tokens <shape>   # → JSON plan (engine_args, gate_j)

# 2. LAUNCH the engine with the plan's primary knob
#    vllm:    --kv-cache-memory-bytes <B>     (fallback: --num-gpu-blocks-override)
#    sglang:  --max-total-tokens <B // kv_per_token>
#    lmdeploy: cache_max_entry_count <B / free-after-weights>  (config key)
#    P/D (§6.5): TWO instances, one knob per pool; pd_split is EXPLICIT, never defaulted

# 3. VERIFY — gate (j) against the startup logs; the plan's gate_j.expected_bytes_total
#    must match realized bytes within CAGE_ISO_BYTES_TOL (default 0.05)
CAGE_ISO_BYTES_LOGS="vllm=<log>,sglang=<log>" bash scripts/checks/preflight_check.sh <MODEL> <API_BASE>

# 4. RECORD — append {model, engine, r, plan_bytes, realized_bytes, knob} to
#    results/<run>/calibration/budget_knob_map.jsonl (operator-recorded; the
#    manifest references it). Off-tolerance → adjust the knob, relaunch, re-verify —
#    NEVER proceed on an unverified budget (the cell would mis-state its r).
```

On the campaign path the driver does steps 1 and 2 itself: every relaunch step
carries the `cache_budget.BudgetPlan` record its launcher env was derived from
(`budget_plan`), every budgeted cell step pins the same record as
`CAGE_BUDGET_PLAN_JSON`, and the campaign session persists it into
`cell.json["budget_plan"]` (Batch 2 W4, ADR-0117; `docs/RESULTS_LAYOUT.md` §3.1).
Gate (j) stays the live verification of the realized bytes.

### 3.2 SLO floor calibration: one cal-v2 artifact per engine, BEFORE `plan`

The §6.1 primary SLO pair is relative to the measured single-stream floor of the
same model x engine (TTFT <= 10x, TPOT <= 5x). The floor comes from the registered
procedure in `src/orchestration/calibration.py` (30 sequential streamed requests
at concurrency 1, median), driven by `scripts/3_run/calibrate_cell.py` against a
live server at the r = 1.5 control rung, once per engine of the session:

```bash
# one artifact per server engine of the session, at the charter floor rung
.venv/bin/python scripts/3_run/calibrate_cell.py \
    --backend vllm --model Qwen/Qwen3-14B --api-base http://localhost:8000 \
    --manifest data/manifests/qasper_2000x3_seed42.json \
    --budget-fraction 1.5 \
    --output results/calibration/vllm.json
# then the same for sglang on its own server (--api-base http://localhost:30000)
```

The lambda* probe (cal-v2, 2026-09-30, ADR-0121 and ADR-0122) starts at the floor's
single-stream service rate (1 / (TTFT floor + 255 x TPOT floor); `--start-qps` overrides
it and the artifact records which one was used), climbs a 1.3x ladder of 75 s windows
until the first rung whose attainment falls under 0.9, then probes two midpoints of
the bracket so lambda* carries a 7.5% resolution. Attainment is the only
sustainability test; per-rung throughput is recorded, never judged. The ladder stops
at 30 rungs with `LADDER_EXHAUSTED` (about 2,000 times the start rate; never
extrapolate). Budget about 20 windows (25 min) per engine from a floor-derived start;
gate (n) accepts cal-v2 artifacts only. On S0 (cal-v1: fixed 12 rungs from a
hand-picked start, plus a retrograde-throughput clause) vLLM needed three passes and
SGLang two; see `MyDocs/RunPod/S0_RUN_2026-09-30.md` and backlog S0F-11/12.

`plan --calibration vllm=results/calibration/vllm.json --calibration sglang=...`
registers them (Batch 2 W4, ADR-0117): the plan REFUSES without one per engine
that has an executable cell, refuses an artifact whose engine, model,
procedure version, request count or statistic is not the registered one, and
refuses a `budget_fraction` other than 1.5 unless the operator passes
`--calibration-budget-fraction <r>` explicitly (an S0 shakedown at 0.5 is
registered that way, and the header records both values). The floors ride every
cell step as `CAGE_SLO_FLOORS_JSON`, and the campaign session writes them into
`manifest.json["slo_floors"]` when the first cell creates the manifest (amended
never; a later cell pinning other floors refuses). `plan` therefore runs after
calibration, on the pod or locally on the pulled artifacts, never before.

The planner refuses HF (the oracle is excluded from pressure sweeps, P2), refuses
P/D without an explicit split, and carries every live-only knob semantic as a
`verify_live` entry — S0-19/S0-20 are where those close.

## 4. Run

The J4 refusal gate applies at launch: a run with NO off-box backup target **refuses to
start** (`require_backup_target`, `scripts/lib/transport.sh`). Export
`CAGE_BACKUP_TARGET` first — on RunPod normally the network-volume S3 API
(`s3://<volume-id>[/prefix]` + `CAGE_S3_ENDPOINT` + `AWS_DEFAULT_REGION` + the S3 API
key, §6 table) or `ssh://[user@]host/path`.

```bash
export CAGE_BACKUP_TARGET=s3://<network-volume>[/prefix]     # see the §6 env table
nohup bash scripts/3_run/run_full_sweep.sh <MODEL> <N> <T> > sweep.log 2>&1 &
# or the core tree alone:
nohup bash scripts/3_run/cloud_run.sh <MODEL> <N> <T> > run.log 2>&1 &
```

- One run-id for the whole matrix: `results/<phase>/<run-id>/...`, minted by
  `mint_run_id` and exported as `CAGE_RUN_ROOT`/`CAGE_RUN_ID`/`CAGE_PHASE`. Resume
  after a crash with `export CAGE_RUN_ID=<printed-id>` + the same command — completed
  cells are skipped.
- **Honesty note (what these scripts ARE):** `cloud_run.sh` / `run_full_sweep.sh` are
  the **pilot harness** (their headers say so) — they drive the retired 9-name
  taxonomy and write the pilot trial layout, NOT the sealed RESULTS_LAYOUT-v2
  campaign tree. The v2 producer library (`src/orchestration/campaign_layout.py`:
  manifest, `cells/<row_key>/window_<k>/`, run-end `seal_run`) is built and tested;
  the CellSpec-native campaign driver that wires it in is pending (#116). Until it
  lands, a harness run root carries **no `ledger.json`**, and the §5 pull gate will
  refuse it — that refusal is the gate working, not a bug. Plan teardown accordingly
  (§5 note).
- Quality scoring is decoupled on every path (`CAGE_SKIP_QUALITY=1`, a *declared*
  regime, not a mock; ADR-0055 "serving writes, scoring reads"): `run_full_sweep.sh`
  exports it on the pilot path, every campaign branch of the sweep scripts
  (`run_full_sweep.sh`, `run_baselines.sh`, `run_compression.sh`) pins it, the
  campaign driver pins it on every cell step (Batch 2 W1, 2026-09-18; `load_plan`
  refuses a plan without the pin), and `run_experiment.py` refuses a campaign cell
  that lacks it (exit 2, before any dataset or engine work). The box's job is serving
  measurements + raw outputs + evidence; model-based quality is scored after the
  serving trees (`rescore_quality.py --full --scoring-run-id <id>` for a v2 tree;
  `--apply` is the legacy layout only), and every campaign window records the regime
  in `metrics.json["quality_scoring"]`.
- Engine endpoints are pinned by the campaign driver (Batch 2 W2, 2026-09-18): every
  server-engine cell step carries `--api-base http://localhost:<port>` and every
  relaunch exports the launcher port env (`VLLM_PORT=8000`, `SGLANG_PORT=30000`; the pd
  relaunch `CAGE_PD_PROXY_PORT=8000`), both from the driver's one port table
  (`ENGINE_PORTS`, mirroring the launchers' defaults), so the server a relaunch starts
  and the endpoint its cells dial agree by construction; the plan header records the
  table under `serving_shapes`. The runner's own `--api-base` default is the vLLM port,
  so a hand-run SGLang row must pass `--api-base http://localhost:30000` (with
  `CAGE_SGLANG_API_BASE` unset, the adapter, the cache flush and the telemetry sampler
  all dial that flag). `load_plan` refuses a plan whose cell or relaunch lacks the pin
  or whose cell dials an endpoint other than the one its relaunch serves; `run` refuses
  while `CAGE_SGLANG_API_BASE` or `CAGE_LMDEPLOY_API_BASE` is exported (the runner
  resolves them before the pin) and while a shell `VLLM_PORT` / `SGLANG_PORT` / pd
  port value differs from the table (the preflight dials the shell value).

- Floors and budget records are pinned by the campaign driver (Batch 2 W4,
  ADR-0117, §3.2): every cell step carries `CAGE_SLO_FLOORS_JSON` and every
  budgeted cell step `CAGE_BUDGET_PLAN_JSON`; `load_plan` refuses a plan without
  the `calibration` header, a cell whose floors pin differs from it, or a cell
  whose budget record differs from its relaunch's; `run` refuses while either
  env is exported in the shell. S0 runs its cells on the HAND PATH (close-out
  sheet decision 0, 2026-09-24: the driver plans only sessions a and b of the D4
  roster, so no S0 cell can be driver-run): `run_experiment.py --campaign-root
  results/s0/a/<run_id>` with `CAGE_SLO_FLOORS_JSON` built by hand from the S0-6
  cal-v1 artifacts and the `CAGE_CELL_*` coordinates exported per cell (the
  runner's session reads both; only `run_campaign.py run` refuses shell-exported
  pins), recorded as a deviation in the S0 session report
  (`MyDocs/RunPod/S0_CHECKLIST.md`, "Hand path"); the rho_own leg is a labeled
  skip at S0 because no budget-plan record is pinned there. The pilot shell
  drivers stay out of S0: a shell-driven tree carries neither pin, so contrast
  #14 refuses on it. Stage 1 runs through the driver and pulls the plan file and
  the calibration artifacts together with the run (the plan header records each
  artifact's path and sha256; the sealed tree carries only the floors).

- One-token completions (Batch 2 W5, ADR-0118): a completion with fewer than two
  output tokens has no decode phase. The runner writes `tpot_ms: null` for it, the
  analysis judges it on TTFT alone and counts it per window (`n_no_decode`), and
  `verify_results` check (j) refuses a completed row whose `num_tokens` and
  `tpot_ms` disagree (a null or zero TPOT beside two or more output tokens, a TPOT
  value beside at most one) and warns when `num_tokens_source` reads `whitespace`
  (the engine returned no `usage.completion_tokens`, so the count is words). S0 row S0-24 records what
  each engine's `usage.completion_tokens` returns for a bare "Yes", so the exempt set
  is known per engine before Stage 1.

### 4.0 Query manifests: build, register, and the blocked B12 rung cells

Every QA dataset of a session measures ONE pre-drawn query manifest (the uniform
yardstick, `scripts/1_setup/build_query_manifest.py`): build it at the grid's
`corpus_prefix_budget_tokens` (2,800) with `--trunc-budgets 1400,700` so the file
carries the B12 corpus-truncation ladder (ADR-0106), with `--num-trials` equal to the
grid's `replications` and `--num-queries` at least the largest per-row-class n the
session grid registers for that dataset (A9: the n per row class lives in
`SessionGrid`, `n_primary`/`n_secondary`/`n_identity`/`window_requests`, lowered only
via `achievable_n`; the runner measures the first n ids of each trial, so a shorter
trial refuses at plan time, before any GPU spends). A dataset whose split cannot
supply the class n (Qasper dev is about 1,005 questions; MuSiQue likely too) MUST have
its lowered n registered in the session grid's `achievable_n` first: `plan
--query-manifest qasper=...` REFUSES until `achievable_n["qasper"]` is registered
(sessions a and b register `achievable_n={}` today, so that refusal is the expected
state, not a bug). Register each manifest with
`plan --query-manifest DATASET=PATH` (repeatable, one per dataset); the plan validates
that the file exists, parses, is for that dataset, carries `block_budget` equal to the
grid budget and every registered rung, and records `path`, `sha256`, `block_budget`
and `trunc_rungs` under the header's `query_manifests`. A B12 rung cell whose dataset
has no registered manifest is still enumerated, but BLOCKED (`blocked_on:
query-manifest:<dataset>`), and `run` refuses the whole plan unless the operator
passes `--skip-blocked` (which runs the executable subset loudly and gates a plain
`--seal`).

```bash
# one manifest per QA dataset, at the grid's block budget, with the B12 ladder
python3 scripts/1_setup/build_query_manifest.py --dataset squad_v2 \
    --num-queries 2000 --num-trials 3 --seed 42 \
    --block-budget 2800 --trunc-budgets 1400,700
# register each one on the plan (repeatable); unregistered datasets keep their B12 rung cells BLOCKED
python3 scripts/3_run/run_campaign.py plan --session a \
    --floor-table results/preflight/floor_table_a.json --window-duration-s 300 \
    --calibration vllm=results/calibration/vllm.json \
    --calibration sglang=results/calibration/sglang.json \
    --query-manifest squad_v2=data/manifests/squad_v2_2000x3_seed42.json \
    --query-manifest hotpotqa=data/manifests/hotpotqa_2000x3_seed42_ov0.33.json \
    --out plan.json
```

### 4.1 G1 confirmatory look from the registered SHA (added 2026-09-16, ADR-0112)

The G1 whole-repo binding (HEAD equals the registered SHA, clean tree; ADR-0089) is
satisfied procedurally: the confirmatory analysis runs from a detached git worktree
checked out at the registered SHA, with the pulled results tree supplied read-only.
Docs-only commits may continue on `main`; any post-freeze CODE change goes through the
§9.11 amendment log and a re-registration (new SHA + OSF timestamp) before the
confirmatory look. `--registered-sha` must match the executing checkout, so run the
script from inside the worktree, never from `main`. The script writes its output to
`<run>/analysis/<timestamp>/`, so only the data subtrees are made read-only.

```bash
# ADR-0112: confirmatory look from the registered SHA (one look, G1)
CAGE_MAIN=/path/to/CAGE   # the worktree has no .venv; the interpreter comes from the main checkout, the code and HEAD from the worktree
RUN=/path/to/results/<campaign>/<session>/<run_id>
git worktree add /path/to/cage-registered <registered-sha>
cd /path/to/cage-registered
mkdir -p "$RUN"/analysis
find "$RUN" -mindepth 1 -maxdepth 1 ! -name analysis -exec chmod -R a-w {} +   # sealed data read-only; analysis/ stays writable for stats.json
"$CAGE_MAIN"/.venv/bin/python scripts/4_analysis/run_campaign_analysis.py \
  "$RUN" \
  --confirmatory --i-understand-one-look --registered-sha <registered-sha>
```

## 5. Sync during the run, then the FAIL-CLOSED end sequence

**During the run** (provider-neutral; every off-box byte goes through
`scripts/lib/transport.sh`):

```bash
# continuous mirror of the whole results/<phase>/ tree (run_full_sweep.sh starts this itself)
bash scripts/5_observability/gcs_backup_daemon.sh start results/<phase>
# one-shot mirror (also the final-sync building block; markers in .agent/last_sync_ok_<backend>)
bash scripts/5_observability/sync_results.sh <dir> [target] [remote_subpath]
# logs + forensics (vLLM logs, dmesg/OOM, pip freeze) with a per-run COLLECT_OK sentinel
bash scripts/5_observability/collect_logs.sh
```

The daemon mirrors on an interval (default 300 s, `CAGE_BACKUP_INTERVAL`), survives
SSH drops (setsid), never uses `--delete`, and `stop` does one final authoritative
sync. The remote layout mirrors the local tree exactly (`docs/RESULTS_LAYOUT.md` §4).

**End sequence — teardown is irreversible; never reorder these steps:**

```
[1] final sync + stop daemon    gcs_backup_daemon.sh stop   (final authoritative sync)
[2] VERIFIED PULL + TEARDOWN    scripts/runpod/teardown_pod.sh <pod_id> <backup_target> <local_run_dir>
                                  [1/5] final on-pod sync (needs CAGE_POD_SSH; else skipped loudly)
                                  [2/5] ledger-gated pull_run.sh -> ONLY its literal
                                        "SAFE TO TEARDOWN" line authorizes destruction
                                  [3/5] confirm ceremony (CAGE_ASSUME_YES=1 for non-interactive)
                                  [4/5] pod delete — the ONLY destructive step, strictly last
                                  [5/5] read-only $0 listing (deletes nothing; re-run until clean)
[3] volumes/buckets             delete the run's network volume / bucket ONLY after [2]'s
                                pull verified; RunPod network volumes and templates bill
                                separately and are NOT touched by teardown_pod.sh
```

`pull_run.sh <target> <local_run_dir>` mirrors the backup target locally and re-hashes
the sha256 ledger (`src.analysis.stats.ledger.verify_ledger`); ANY failure — transfer
error, missing/tampered ledger, hash mismatch — exits nonzero and prints
DO-NOT-TEARDOWN. There is deliberately NO env-var bypass in `teardown_pod.sh`;
`--force` is the single, loud, user-only override (finding J10).

**Pilot-harness trees** (§4 honesty note) carry no ledger, so `pull_run.sh` refuses
them with LEDGER-MISSING. For such a tree the operator pulls with
`sync_results.sh`-style transport (or `transport_pull`), verifies completeness
manually, and only then uses `--force` — a user decision, reported as such.

## 6. ENV-CONTRACT TABLE (the tracked reference for #137/#138 era variables)

| Variable | Consumed by | Contract |
|---|---|---|
| `CAGE_BACKUP_TARGET` | `transport.sh` (all sync/pull/teardown callers) | Off-box target: `gs://bucket[/prefix]` \| `s3://bucket[/prefix]` \| `ssh://[user@]host/abs/path` \| `file:///abs/path`. Anything else dies loud. Takes precedence over `CAGE_RESULTS_BUCKET`. |
| `CAGE_S3_ENDPOINT` | s3 backend | Endpoint URL for `aws s3`: points it at the RunPod network-volume S3 API (`https://s3api-<dc>.runpod.io`, the 15 datacenters listed at docs.runpod.io/storage/s3-api). Pair with `AWS_DEFAULT_REGION=<datacenter id>` (REQUIRED: the endpoint rejects a request signed for any other region, verified 2026-09-25) and the account-level S3 API key created in the console (`AWS_ACCESS_KEY_ID` = its access key, `AWS_SECRET_ACCESS_KEY` = its secret; separate from `RUNPOD_API_KEY`); the bucket name is the network volume id. Both the pod (`setup_runpod.sh` installs the AWS CLI v2 by the official installer; Ubuntu 24.04 carries no `awscli` apt package) and the workstation (`brew install awscli`) need the CLI. |
| `CAGE_SSH_OPTS` | ssh backend, `teardown_pod.sh` | Extra ssh options (e.g. `-p 2222` — RunPod pods expose SSH on non-standard ports). |
| `CAGE_RESULTS_BUCKET` | legacy callers | Legacy GCS spelling (bare name or `gs://`); bare names are normalized to `gs://`. On a GCP box only, the metadata-derived `gs://<project>-cage-results` default still applies (Appendix A). |
| `CAGE_TRANSPORT_DRYRUN=1` | `transport.sh` | Echo `DRYRUN: <exact command>` instead of executing — how gcs/s3/ssh argument construction is unit-tested offline (`tests/test_topic10_transport_runpod.py`). |
| `CAGE_ALLOW_NO_BACKUP=1` | `require_backup_target`, `sync_results.sh` | The ONLY way to start a run with no off-box target (J4). Recorded durably to `<run-root>/NO_BACKUP_OVERRIDE` and echoed into the run manifest. Preflight gate (e) treats it as poison: confirmatory runs refuse to launch while it is set. |
| `CAGE_SKIP_LOCAL_PULL=1` **and** `CAGE_SKIP_LOCAL_PULL_CONFIRM=I-ACCEPT-DATA-LOSS` | `teardown_vm.sh` (GCP port only) | The J10 **double ceremony** to skip the pre-delete local pull; a bypass marker is recorded under `results/` first. One var alone aborts. `teardown_pod.sh` has no equivalent — `--force` only. |
| `CAGE_RUN_ROOT` / `CAGE_RUN_ID` / `CAGE_PHASE` | run scripts, observability | Minted by `cloud_run.sh`/`run_full_sweep.sh` (`mint_run_id`: `<YYYY-MM-DD_HHMMSS>_<model-slug>_<Q>x<T>_<4hex>_<dataset>`) and exported so every child writes the SAME `results/<phase>/<run-id>/` tree. Export `CAGE_RUN_ID` to resume into an existing tree. |
| `CAGE_PREFLIGHT_BACKENDS` | `preflight_check.sh` gates (j)/(k) | Comma-separated adapter list to check (default `vllm,sglang,lmdeploy`). Scope down for single-engine pods. |
| `CAGE_ISO_BYTES_TOL` | gate (j) | Relative tolerance for §6.5 realized-KV iso-bytes parity (default `0.05`; must be a float in (0,1) or the gate FAILS). |
| `CAGE_ISO_BYTES_LOGS` | gate (j) | Pin exact engine startup logs: `vllm=/path/a.log,sglang=/path/b.log` (e.g. one budget point of a pressure sweep). |
| `CAGE_QUALITY_STRICT` | `src/evaluation/quality.py`, gate (e) | Unset/`1` = strict fail-closed quality layer (default). An explicit falsy (`0`/`false`/`no`) downgrades instrument failures to `score=None` for the whole run — preflight FAILS on it; forbidden for confirmatory runs. |
| `CAGE_CLAIM_CHECKER` | `src/evaluation/quality.py` | Claim-check instrument selection. Default `nli` (owner decision #120/F8, 2026-08-19; in-process-safe). `alignscore` is Instrument B and is requested explicitly by `scripts/4_analysis/score_instrument_b.py` — never as the run default. Preflight prints the state either way. |
| `CAGE_SKIP_QUALITY=1` | run scripts, `run_campaign.py` (cell-step env pin), `run_experiment.py` (campaign-mode gate) | Decoupled-scoring regime (default in `run_full_sweep.sh`; pinned on every campaign cell step by the driver, W1 / ADR-0055): inline model-based quality is skipped and scored after the serving trees. A *declared* regime, not a mock. A campaign cell without it refuses before serving; the regime is recorded per window in `metrics.json["quality_scoring"]`. |
| `VLLM_PORT` / `SGLANG_PORT` / `CAGE_PD_PROXY_PORT` / `CAGE_PD_PREFILL_PORT` / `CAGE_PD_DECODE_PORT` | launchers (`manage_vllm_server.sh`, `manage_sglang_server.sh`, `manage_vllm_pd.sh`), `run_campaign.py` (relaunch env) | Listening ports of the launchers (defaults 8000 / 30000 / 8000 / 8100 / 8200). The campaign driver exports them on every relaunch from its port table and pins the matching `--api-base` on every server-engine cell (W2); the operator's shell value never reaches a campaign relaunch (the step env wins). The preflight's own gate URL reads the shell `SGLANG_PORT`, so `run` refuses a shell value that differs from the table (an equal value is fine). |
| `CAGE_VENV_ROOT` | `setup_runpod.sh` | ADR-0125 (S0F-6): the directory the three venvs are REALLY created in (default `/root/cage-venvs`, the container disk); the repo-root names are links to them. Never the network volume. |
| `CAGE_KV_BUDGET_BYTES_REPLICA` | `manage_vllm_cluster.py` | ADR-0124 (S0F-19): the KV pool of EVERY cluster replica in bytes, passed as `--kv-cache-memory-bytes` so vLLM skips its device-wide memory profiler (two replicas profiling at once on one GPU collapse each other's pool). REQUIRED when the replicas share a GPU, optional with distinct `--replica-gpus`; refused beside `CAGE_KV_BUDGET_BYTES` or `CAGE_VLLM_GPU_BLOCKS_OVERRIDE`. Recorded in `cluster_state.json`. Shared-GPU replicas start one at a time. |
| `CAGE_CLUSTER_BASE_PORT` / `CAGE_CLUSTER_ROUTER_PORT` | `run_tests.sh --with-cluster` | S0F-18: forwarded to the cluster manager as `--base-port` / `--router-port` when set (the manager's defaults are 8101 and 9000; 8001 is the image's nginx); the router port also sets `ROUTER_TEST_API_BASE` for the router tests. |
| `ROUTER_START_TIMEOUT` | `manage_vllm_cluster.py` | S0F-18: default `--router-timeout` in seconds (60). The replica budget reads `VLLM_START_TIMEOUT` (300), the engine launchers' knob. |
| `CAGE_PD_NIXL_PORT_PREFILL` / `CAGE_PD_NIXL_PORT_DECODE` | `manage_vllm_pd.sh` | ADR-0128 (S0F-13): the NIXL handshake side-channel port of each role (defaults 5600 / 5601), passed to the role as `VLLM_NIXL_SIDE_CHANNEL_PORT` in its child env and recorded in the per-role serving-config capture. Equal values, a value equal to an HTTP port, or a shell `VLLM_NIXL_SIDE_CHANNEL_PORT` refuse the start. |
| `CAGE_PD_PYTHON` | `manage_vllm_pd.sh` | ADR-0128: the interpreter the nixl import gate probes before the self-cleaning stop (default `python3`, the activated cage-env that runs `vllm serve`). Test seam; never set it on a pod. |
| `CAGE_HF_LIVE=1` | `tests/test_qasper_revision_s0f5.py` (`setup_runpod.sh` step 4a-live) | ADR-0127 (S0F-5): opt-in for the live qasper check (loads the pinned Hub route with the real `datasets` library and compares the rebuilt 50x3 manifest digest). Unset, the test skips; the bootstrap sets it on every pod. |
| `CAGE_SGLANG_API_BASE` / `CAGE_LMDEPLOY_API_BASE` | `run_experiment.py` (adapter + cache flush) | Per-engine endpoint override, resolved BEFORE `--api-base`. Pilot convenience only: `run_campaign.py run` refuses while either is set (W2), because it would beat the plan's pin. |
| `CAGE_SLO_FLOORS_JSON` | `run_campaign.py` (cell-step env pin), `campaign_session.py` | Batch 2 W4 (ADR-0117): the §6.1 single-stream floors of every registered engine as compact JSON, pinned on EVERY cell step from the plan header `calibration` (one cal-v2 artifact per engine, §3.2); the session writes it into `manifest.json["slo_floors"]` at manifest creation and refuses a reopened manifest whose floors differ. `run` refuses while it is exported in the shell. Never set it by hand. |
| `CAGE_BUDGET_PLAN_JSON` | `run_campaign.py` (budgeted cell-step env pin), `campaign_session.py`, `campaign_layout.CellWriter` | Batch 2 W4 (ADR-0117): the `cache_budget.BudgetPlan` record of the relaunch the cell runs under (`asdict` plus `floor_table_sha256`), pinned on budgeted cell steps only; cross-checked against the cell tuple by the session and persisted into `cell.json["budget_plan"]` (the rho_own basis). `run` refuses while it is exported in the shell. Never set it by hand. |
| `CAGE_QUERY_MANIFEST` | `run_experiment.py` (loader), `campaign_session.py` | Path to the dataset's pre-drawn query manifest (`build_query_manifest.py`); the runner's `--query-manifest` sets it (the campaign driver passes that flag on every cell of a registered dataset, §4.0). The loader refuses a manifest built for another dataset, and the corpus-budget guard (A4, ADR-0106) refuses a served budget that is neither its `block_budget` nor one of its `trunc_rungs`. `campaign_session.py` resolves `dataset_manifests_sha256` from it when `CAGE_DATASET_MANIFESTS_SHA256` is unset. |
| `CAGE_MODEL_WEIGHTS_GIB` | `manage_lmdeploy_server.sh` | REQUIRED for an LMDeploy start: the served checkpoint's weight footprint in GiB, measured on the pod (`du -sh` of its safetensors), the section 6.5 input that maps the byte budget to TurboMind's post-weights `cache_max_entry_count`; the launcher refuses to start without it (or an explicit `LMDEPLOY_CACHE_MAX_ENTRY_COUNT`, a recorded deviation). Integration audit 2026-09-26, models-8. |
| `CAGE_KV_BUDGET_BYTES` | `manage_vllm_server.sh` | Positive integer; when set the launch adds `--kv-cache-memory-bytes <B>` (the planner's vLLM knob, S0-19) and the running-server check requires the same value on the live command line. |
| `VLLM_MAX_MODEL_LEN` | `manage_vllm_server.sh`, `manage_sglang_server.sh` (`--context-length`), `_serving_config.sh` | Served context length; the launcher default is 4,096 (`manage_vllm_server.sh:288`). The campaign relaunches and every S0 cell export 32,768 (backlog A10). |
| `CAGE_SGLANG_PYTHON` / `CAGE_LMDEPLOY_BIN` | `manage_sglang_server.sh`, `manage_lmdeploy_server.sh` | The engine interpreter and entry point; default to `sglang-env/bin/python3` and `lmdeploy-env/bin/lmdeploy` (setup step 3c), accept an absolute path or a bare name resolved through `command -v`, and fail closed in the start gate on a missing or non-importable engine before any teardown (pre-GO item 10, `docs/VLLM_COMPATIBILITY.md` section 7). |
| `CAGE_PROMPT_MODE` | `run_experiment.py` | `chat` (default: the model's chat template through `/v1/chat/completions`, Qwen3 thinking pinned off on vLLM) or `raw` (the legacy raw-completions path). The prefix-aware router serves no chat route, so a runner smoke through it uses `raw` (S0-9; audit distributed-7). |
| `CAGE_<ENGINE>_CHAT_TEMPLATE_KWARGS` | `run_experiment.py` (`_adapter_env_extras`) | JSON object forwarded as the adapter's `chat_template_kwargs` for `SGLANG` and `LMDEPLOY`; the `VLLM` variant is REFUSED because the vLLM adapter pins its own verified kwargs (ADR-0007). Malformed values refuse. S0-5 is the only live verification. |
| `CAGE_MODEL_SLUG` | `campaign_session.py` | The design-input model label of a campaign-mode cell when the served model is outside the D4 roster (S0 stand-in: `qwen3-14b` for the BF16 Qwen3-8B); the session refuses a non-roster model without it. Never set it for a roster model. |
| `CAGE_CELL_FAMILY` / `CAGE_CELL_BUDGET_R` / `CAGE_CELL_RATE_FRAC` (also `CAGE_CELL_POLICY`, `CAGE_CELL_TOPOLOGY`, `CAGE_CELL_ARM`, `CAGE_CELL_RETRIEVER`, `CAGE_CELL_CORPUS_BUDGET`) | `campaign_session.py` (`from_cli`) | Per-cell coordinate overrides on the hand path (the driver sets them on the campaign path): family and the F2/F3 pressure coordinates the window records. A hand-run F2 pressure cell exports family `F2`, `budget_r` and `rate_frac` beside the launcher's byte knob. |
| `CAGE_PROVIDER` / `CAGE_HARDWARE` / `CAGE_GPU_COUNT` | `campaign_session.py` | Provenance the run manifest requires (`CAGE_PROVIDER`, `CAGE_HARDWARE`: refused when unset) and the serving stack's GPU count (optional, W4.2; the driver threads it). S0: `runpod`, `L40S x1`, `1`. |
| `HF_HUB_DOWNLOAD_TIMEOUT` | `setup_runpod.sh`, HF downloads | Stalled-read timeout in seconds (default 30). Exported BEFORE dataset staging AND model prefetch (J7 — a stalled socket must raise, then resume, not hang for an hour). |
| `CAGE_BACKUP_INTERVAL` | `gcs_backup_daemon.sh` | Seconds between mirror passes (default 300). |
| `CAGE_POD_SSH` / `CAGE_ASSUME_YES` | `teardown_pod.sh` | `user@host` of the pod for the final on-pod sync (unset = that step skipped loudly); `CAGE_ASSUME_YES=1` answers the confirm ceremony for non-interactive teardowns. |

## 7. Cost + ETA reporting duty

Every cloud action (provision, resize, long job, teardown) is reported in this format,
BEFORE performing it (and provisioning additionally waits for the §0.1 approval):

```
ACTION:   <what, exactly — pod type × count, region, secure/community, volume size>
COST:     <$/h> × <est. hours> = <$ estimate>   (list price, provider, date checked)
ETA:      <wall-clock estimate for the step and for the session>
TEARDOWN: <what returns this to $0 and when>
```

Rates are volatile — quote the provider's current price at action time, never a
remembered number. Session-level dollar totals go into `MyDocs/LEDGER.md` at EOD.

## 8. Quick reference

| Goal | Command |
|---|---|
| Package repo for ship | `scripts/ops/package_repo.sh` |
| Bootstrap the pod | `bash scripts/runpod/setup_runpod.sh` |
| Live preflight (gates a–p) | `bash scripts/checks/preflight_check.sh <MODEL> <API_BASE>` |
| Full sweep (one run-id) | `nohup bash scripts/3_run/run_full_sweep.sh <MODEL> <N> <T> > sweep.log 2>&1 &` |
| Submit long remote job | `scripts/gcp/remote_job.sh submit <name> '<cmd>' [deadline_s]` |
| One-shot sync | `bash scripts/5_observability/sync_results.sh <dir> [target]` |
| Start/stop backup daemon | `bash scripts/5_observability/gcs_backup_daemon.sh start\|stop [phase_dir]` |
| Collect logs + forensics | `bash scripts/5_observability/collect_logs.sh` |
| Verified pull (ledger gate) | `bash scripts/5_observability/pull_run.sh <target> <local_run_dir>` |
| Fail-closed teardown + $0 | `scripts/runpod/teardown_pod.sh <pod_id> <target> <local_run_dir>` |

Compatibility gates and pins: `docs/VLLM_COMPATIBILITY.md`.
Results tree + ledger spec: `docs/RESULTS_LAYOUT.md`.
Design authority: `MyDocs/PUBLICATION.md` (§7.6 groups, §7.6.1 matrix, §7.7f lifecycle).

---

## Appendix A — GCP port (retained, not current)

GCP is kept as a **portability backend**: everything below works, none of it is the
primary path, and nothing here weakens the §0 disciplines.

- **Provision**: `terraform/gcp/` with one tfvars file per session
  (`terraform/gcp/sessions/group-a.tfvars`, `group-b.tfvars`, `group-cd.tfvars` — each sets
  the terraform `session` variable, which is also the `session` label stamped on every
  resource). `terraform plan` is always allowed; `terraform apply` is GATED by the
  §0.1 user approval. Label everything `agent-run=<run_id>` so the orphan sweep can
  find strays.
- **Setup**: `bash scripts/gcp/setup_gpu_cloud.sh` (DLVM-shaped: sudo/systemd; the
  RunPod script is the container-shaped primary). Same canonical-interpreter rule (B1).
- **Ops**: `scripts/gcp/gpu_vm.sh create` (pilot-era L4 zone-hunt) and
  `scripts/gcp/remote_job.sh` (provider-agnostic over SSH). SSH flags that keep agents
  sane: `-o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes`, plus
  `CLOUDSDK_CORE_DISABLE_PROMPTS=1` (a TTY-less prompt hangs forever). Kill by the
  RECORDED PID, never `pkill -f <script>`. A non-login `ssh --command` does NOT
  inherit the run's env — forward `CAGE_BACKUP_TARGET` (etc.) explicitly.
- **Backup default**: on a GCP box (and only there) the metadata server derives
  `gs://<project>-cage-results` when no target is set (`transport_default_target`).
- **Teardown**: `scripts/gcp/teardown_vm.sh <vm> <zone>` — same fail-closed
  ordering (COLLECT_OK sentinel + complete local pull before delete). Skipping the
  pull needs the J10 double ceremony (`CAGE_SKIP_LOCAL_PULL=1` AND
  `CAGE_SKIP_LOCAL_PULL_CONFIRM=I-ACCEPT-DATA-LOSS`, bypass marker recorded).
  Prove $0 by label: instances, disks, buckets all empty for
  `labels.agent-run=<run_id>`; `scripts/gcp/gpu_vm.sh sweep` is the universal check.
- **RDMA path (C/D act 2, [Extension])**: H200 capacity via `a3-ultragpu-8g`
  (typically DWS Flex-start / calendar reservation) needs an RDMA-network-profile VPC;
  the act-2 gate is the `docs/VLLM_COMPATIBILITY.md` §8 RDMA preflight. No
  RDMA-capable fabric → the RDMA rung cannot run there (the TCP rung still can).
