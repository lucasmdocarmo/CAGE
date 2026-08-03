# CAGE — Campaign Runbook (cloud execution)

> **Authority.** `MyDocs/PUBLICATION.md` is THE design authority (groups, arms, engines,
> matrices — verify section numbers by grepping it). This file is the *execution*
> authority: how a provisioning session is actually brought up, validated, run, drained,
> and torn down. Supersedes the pilot-era runbook (Phase-1/2 CPU/L4 content removed;
> pilot data stays read-only under `results/phase2/` — see `cloud/RESULTS_LAYOUT.md`).
>
> ⚠️ **APPROVAL GATE — READ FIRST. Nothing provisions without an explicit user "go".**
> No `terraform apply`, no `gcloud compute instances create`, no neocloud instance
> launch, no bucket creation — ever — as a side effect of preparation. Prepare plans,
> print the cost+ETA report, then STOP and wait for the user. This is a standing,
> binding discipline, not a preference.

---

## 0. Standing disciplines (binding on every session)

1. **Approval gate** — provisioning (any resource that bills or persists) happens only
   after an explicit user go, per session AND per act. Plans/tfvars/dry-runs are fine.
2. **Clean-room infra per run** — every run gets fresh infrastructure: a fresh bucket
   named for the run-id (`gs://cage-<run_id>`), no reuse of past-run VMs, disks,
   buckets, or leftover state. A surviving bucket from a previous run counts as a
   violation.
3. **Labels on everything** — every resource (VM, disk, bucket, reservation) carries
   `agent-run=<run_id>`, `session=<a|b|cd-act1|cd-act2>`, `model=<charter-slug>` so the
   orphan sweep can find strays: `gcloud compute instances list --filter='labels.agent-run:*'`.
4. **Pull local BEFORE teardown, fail-closed** — results must exist in THREE places
   (node + GCS + local) and the ledger must verify before anything is deleted (§5).
5. **Teardown to TRUE $0** — including the bucket, after the local pull + ledger verify.
   Prove it: instance list, disk list, bucket list all empty for the run's labels.
6. **Cost + ETA on every cloud action** — §6 format. No cloud command in a report
   without its price and its clock.
7. **Validate infra before every run** — the live preflight (§3) with a real smoke, on
   every session and after every engine restart. No mock, no cached green.

---

## 1. The three provisioning sessions (PUBLICATION.md §7.6)

The campaign runs as **three provisioning sessions**; C and D share one session in two
acts. Each session: approval → provision → preflight → run cells → pull-verify →
teardown-$0. Cell carriage per family × group is §7.6.1 — that matrix is the single
answer to "who carries what at which density".

| Session | Group / model | Engines | Baselines | Hardware (GCP-fallback shape) | Mission |
|---|---|---|---|---|---|
| **A** | A — Qwen3-14B (anchor) | all 4 (vLLM, SGLang, LMDeploy-TurboMind, HF oracle) | B1–B12 + BM25 offline gate | 1× A100-80 GB (`a2-ultragpu-1g` class) | full controlled grid: every arm, every engine, all datasets; FULL D6 factorial (5×6) + fine r-grid |
| **B** | B — Llama-3.3-70B | all 4 (TurboMind ✓, HF oracle ✓) | B1–B12 | one 4× A100-80 GB node (`a2-ultragpu-4g`), TP=4 (TP=2 = one labeled sensitivity point) | pressure at scale: FRESH set → F2 grid, REUSE set → F3; + SCBench slice |
| **C/D act 1** | C — Qwen3-Next-80B | vLLM + SGLang (HF oracle; LMDeploy absent per P7) | B1–B10 | **ONE** `a3-ultragpu-8g` (8× H200) — intra-node TP + PD (PD needs 2× 160 GB weight copies) | disaggregation PROTOCOL cost isolated from the network; reduced F2 3×3 (pressure = concurrency) |
| **C/D act 2** | D — DeepSeek-V3-0324 | vLLM + SGLang (HF **EXEMPT** per D4) | B1–B10 | **TWO** `a3-ultragpu-8g` nodes — TP=8 + PD **cross-node: TCP rung → RDMA/RoCE rung** (FP8 weights, 671 GB) | transfer cost + dedup-over-the-wire; MLA×TP; + SCBench ≤128k slice |

Datasets across the campaign: SQuAD v2, HotpotQA, MuSiQue, Qasper, RULER, SCBench
slice, ShareGPT (load donor). Per-cell lifecycle is §7.7(f): provision (gated) →
preflight → launch engine with the cell tuple as config → drive manifest → collect
three streams (our clock · engine `/metrics` · policy events) → **quality scored LATER,
offline** → pull, verify, teardown.

Act-1 → act-2 transition: preflight ONCE on the first node (act 1), then scale to two
nodes for act 2 — but act 2 has its OWN additional gate (the RDMA preflight,
`cloud/VLLM_COMPATIBILITY.md` §8) and its own user approval.

---

## 2. Provider decision — neocloud primary, GCP fallback

- **Primary: neoclouds** (grant/credit providers per `MyDocs/GRANTS.md`). Cheapest
  path for A/B; for C/D any provider that offers 8× H200 nodes with a real RoCE/IB
  fabric between them qualifies for act 2. Act 2's fabric requirement is hard:
  **no RDMA-capable inter-node fabric → the RDMA rung cannot run there** (TCP rung
  still can).
- **Fallback: GCP**, provisioned through `terraform/` with **one tfvars file per
  session** (`terraform/sessions/group-a.tfvars`, `group-b.tfvars`, `group-cd.tfvars` —
  each sets the terraform `session` variable to its short value `a` / `b` / `cd`, which
  is also the `session` label stamped on every resource;
  `session_cd.tfvars`). GCP is the *expected* fallback for C/D (H200 capacity via
  `a3-ultragpu-8g`, typically DWS Flex-start / calendar reservation, and an
  **RDMA-network-profile VPC** — an ordinary VPC will not carry RoCE).
- ⚠️ **`terraform plan` is always allowed; `terraform apply` is GATED by the user
  approval in §0.1.** Write the plan output into the session's cost report first.
- Either provider: the node must expose the same contract to the run scripts — Linux +
  NVIDIA driver, `$HOME/CAGE` repo tree, `cage-env` venv, outbound HTTPS to GCS (the
  results bucket is on GCS even when compute is a neocloud).

---

## 3. Pre-flight gates (run per session; a failing gate = do NOT launch)

The user-mandated live-infra validation rule: every component proven live with a real
smoke, on the actual provisioned node, before any paid cell runs.

**Gate list (all sessions):**

1. **Engines up.** For each engine in the session's roster: server `/health` 200, the
   target model listed at `/v1/models`, and one real completion at T=0.
   vLLM extra: `POST /reset_prefix_cache` returns 200 (needs `VLLM_SERVER_DEV_MODE=1`)
   — else cold-start-per-trial silently no-ops. `bash scripts/checks/preflight_check.sh
   <MODEL> <API_BASE>` codifies (a)–(e) below for the vLLM path.
2. **Version pins verified.** The engine×model VERIFY-LIVE matrix in
   `cloud/VLLM_COMPATIBILITY.md` §7, plus the §0 0.19.1 migration gate items. Record
   actual versions into the run manifest.
3. **Metric models live.** Quality layer loads and scores a REAL pair: LettuceDetect
   grounding returns a real number (not None), NLI entailment loads. (Scoring itself is
   offline/decoupled, but the models must be proven loadable before we commit to a
   data format.)
4. **cage-stats live.** Importable, scrapes the engine's `/metrics`, dashboard renders
   (`cage-stats --once --json`).
5. **Retrieval live.** FAISS + embedding model load; one real top-k query against the
   built index; reranker loads for B6+ cells.
6. **No-mock check.** No disable/mock escape-hatch env var set
   (`CAGE_DISABLE_LETTUCEDETECT`, `CAGE_SKIP_QUALITY`, etc.) unless the run design
   explicitly declares it (quality-decoupled runs declare `CAGE_SKIP_QUALITY=1` —
   that is a *declared* regime, not a mock).
7. **T=0 identity smoke.** Single-stream, controlled: (i) HF-oracle match per
   model×engine where the oracle exists (P2/P3; V3 substitutes cross-engine vLLM↔SGLang
   agreement); (ii) B1↔B2 token-identity (reuse changes no tokens — quality equality by
   construction must actually hold); (iii) session C: the §5.2 mandatory vLLM×Qwen3-Next
   prefix-ON identity smoke — if it FAILS the cell is excluded and the failure IS the
   result.
8. **P6 floor table.** Compute per-model λ* and byte floors on the provisioned
   hardware (one number per §7.6.1 cell family) before any pressure cell.
9. **Session C/D act 2 only:** the **RDMA preflight** —
   `cloud/VLLM_COMPATIBILITY.md` §8 (five-check RoCE-not-TCP smoke, UCX hardening,
   version-triple pin). TCP rung cells may run before it; **no RDMA-rung cell runs
   until it passes.**

---

## 4. Run flow (per session)

### 4.1 Package → ship

The node tree is a **tarball, not a git clone** (provenance via `BUILD_INFO`; a clone
would record `sha=null` when git is absent):

```bash
# workstation
scripts/ops/package_repo.sh                      # -> /tmp/cage_<sha8>.tar.gz, warns if tree dirty
gcloud compute scp /tmp/cage_<sha8>.tar.gz <vm>:~ --zone=<zone>   # or plain scp on a neocloud
# node
mkdir -p ~/CAGE && tar xzf cage_*.tar.gz -C ~/CAGE
head -3 ~/CAGE/BUILD_INFO                        # verify sha/dirty/packaged_at
```

Node env (the run scripts hard-source it under `set -e` — do NOT use `.venv` here):

```bash
cd ~/CAGE && python3 -m venv cage-env && source cage-env/bin/activate
pip install -r requirements.txt   # pulls cage-stats (git dep); auth git first if private
pip install "vllm==0.19.1"        # the campaign pin — see cloud/VLLM_COMPATIBILITY.md
export HF_TOKEN=hf_xxx            # gated models
```

### 4.2 SSH + long-job discipline (kept from the pilots — still true)

- **Never run a long command through a blocking SSH.** Use
  `scripts/ops/remote_job.sh` — submit/status/tail/grep/wait/kill/fetch with a durable
  remote PID + status file, base64-shipped commands (quoting-proof), and local state
  JSON under `.agent/tasks/` so a later shell can resume knowing only the job name:
  ```bash
  CAGE_VM=<vm> CAGE_ZONE=<zone> scripts/ops/remote_job.sh submit run_a_cells \
    'cd ~/CAGE && source cage-env/bin/activate && bash scripts/3_run/cloud_run.sh ...'
  scripts/ops/remote_job.sh status run_a_cells    # RUNNING | DONE(0) | FAILED(n) | ...
  scripts/ops/remote_job.sh tail   run_a_cells 40 # bounded read — never firehose
  ```
- SSH flags that keep agents sane: `-o StrictHostKeyChecking=no -o ConnectTimeout=25
  -o BatchMode=yes`, and `CLOUDSDK_CORE_DISABLE_PROMPTS=1` (a TTY-less prompt hangs
  forever — the #1 cause of a "stuck" task).
- Kill by the RECORDED PID, never `pkill -f <script>` (that matches the SSH command's
  own shell and kills the session with exit 255).
- A non-login `ssh --command` does NOT inherit the run's env — forward
  `CAGE_RESULTS_BUCKET` (etc.) explicitly on every remote command.

### 4.3 Preflight, then run cells

1. Run §3 gates. Fix or abort; never "run through" a red gate.
2. Run the session's cells **per the §7.6.1 family × group matrix**, one cell tuple at
   a time (cell identity = `CellSpec`, `src/analysis/cellspec.py`; results tree =
   `cloud/RESULTS_LAYOUT.md`). Quality scoring is decoupled (offline, later) — the
   node's job is serving measurements + raw outputs + evidence.
3. Engine relaunches between cells are normal (prefix ON/OFF, policy knobs, TP/PD
   topology are launch-time levers). After any relaunch, re-run gate 1 (health +
   `/v1/models` + reset endpoint) before the next cell.

### 4.4 Continuous GCS sync (runs the whole session)

```bash
# node, once, right after preflight passes (REQUIRED env, fails LOUDLY if unset):
export CAGE_RESULTS_BUCKET=gs://cage-<run_id>
bash scripts/5_observability/gcs_backup_daemon.sh start results/<campaign>
```

The daemon mirrors the ENTIRE results tree on an interval (default 300 s,
`CAGE_BACKUP_INTERVAL`), is `setsid`-detached (survives its parent shell / SSH drops),
never uses `--delete`, and `stop` does one final authoritative sync. One-shot mirror:
`bash scripts/5_observability/sync_results_to_gcs.sh <dir> [bucket] [remote_subpath]`.
The bucket layout mirrors the local tree exactly (`cloud/RESULTS_LAYOUT.md` §4).

---

## 5. The FAIL-CLOSED end sequence (order is the point)

Teardown is irreversible; the run must exist in three places and be ledger-proven
before anything is destroyed. **Never reorder these steps.**

```
[1] final sync + stop daemon      gcs_backup_daemon.sh stop  (final authoritative sync)
[2] PULL LOCAL                    scripts/5_observability/pull_run.sh cage-<run_id> results/<campaign>/<session>/<run_id>
                                  (canonical entry point — until it lands in scripts/,
                                  teardown_vm.sh step [4/6] performs the pull and
                                  refuses to delete on an incomplete pull)
[3] LEDGER VERIFY (local)         verify the pulled tree against ledger.json:
                                  .venv/bin/python -c "from src.analysis.stats.ledger import verify_ledger; \
                                    print(verify_ledger('<run_root>/ledger.json','<run_root>') or 'INTACT')"
                                  any MISSING/HASH-MISMATCH line -> STOP. Re-pull. Do not tear down.
[4] TEARDOWN                      scripts/6_teardown/teardown_vm.sh <vm> <zone>
                                  (per node; fail-closed: unique COLLECT_OK sentinel in
                                  GCS + complete local pull required before delete;
                                  --force exists and is a user-only decision)
[5] DELETE THE BUCKET             gcloud storage rm -r gs://cage-<run_id>   # only after [3] passed
[6] PROVE TRUE $0                 orphan sweep by label:
                                  gcloud compute instances list --filter='labels.agent-run=<run_id>'
                                  gcloud compute disks list     --filter='labels.agent-run=<run_id>'
                                  gcloud storage buckets list   --filter='labels.agent-run=<run_id>'
                                  all three empty -> report "$0 confirmed" with timestamps
```

Notes:
- The ledger (`ledger.json`, sha256 of every raw artifact) is written on the node at
  run end, BEFORE any analysis — step [3] verifies the *local* copy against it, so a
  truncated pull or a corrupted object cannot pass silently.
- Pulling after teardown is one failed sync away from data loss — that ordering is
  forbidden.
- On a neocloud, steps [4]/[6] use the provider's CLI/console; the bucket steps are
  unchanged (results live on GCS regardless of where compute ran).

---

## 6. Cost + ETA reporting duty

Every cloud action (provision, resize, long job, teardown) is reported in this format,
BEFORE performing it (and provisioning additionally waits for the §0.1 approval):

```
ACTION:   <what, exactly — instance type × count, region, spot/on-demand/DWS>
COST:     <$/h> × <est. hours> = <$ estimate>   (list price, provider, date checked)
ETA:      <wall-clock estimate for the step and for the session>
TEARDOWN: <what returns this to $0 and when>
```

Rates are volatile — quote the provider's current price at action time (for GCP:
pricing calculator / `gcloud` SKU lookup), never a remembered number. Session-level
dollar totals go into the session's ledger entry in `MyDocs/LEDGER.md` at EOD.

---

## 7. Quick reference

| Goal | Command |
|---|---|
| Package repo for ship | `scripts/ops/package_repo.sh` |
| Submit long remote job | `scripts/ops/remote_job.sh submit <name> '<cmd>' [deadline_s]` |
| Poll / bounded tail | `scripts/ops/remote_job.sh status\|tail <name>` |
| Live preflight (vLLM path) | `bash scripts/checks/preflight_check.sh <MODEL> <API_BASE>` |
| Start GCS sync daemon | `CAGE_RESULTS_BUCKET=gs://cage-<run_id> scripts/5_observability/gcs_backup_daemon.sh start` |
| One-shot sync | `scripts/5_observability/sync_results_to_gcs.sh <dir>` |
| Pull run local | `scripts/5_observability/pull_run.sh cage-<run_id> results/<campaign>/<session>/<run_id>` (or teardown step [4/6]) |
| Fail-closed teardown | `scripts/6_teardown/teardown_vm.sh <vm> <zone>` |
| Orphan sweep | `gcloud compute instances list --filter='labels.agent-run:*'` |

Compatibility gates and pins: `cloud/VLLM_COMPATIBILITY.md`.
Results tree + ledger spec: `cloud/RESULTS_LAYOUT.md`.
Design authority: `MyDocs/PUBLICATION.md` (§7.6 groups, §7.6.1 matrix, §7.7f lifecycle).
