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
[setup]     bash scripts/1_setup/setup_runpod.sh              (container-shaped, B1 interpreter)
[preflight] bash scripts/checks/preflight_check.sh <MODEL> <API_BASE>   (gates (a)-(q))
[run]       nohup bash scripts/3_run/run_full_sweep.sh <model> <N> <T> > sweep.log 2>&1 &
            (or cloud_run.sh for the core tree alone — see the honesty note in §4)
[sync]      scripts/5_observability/sync_results.sh + gcs_backup_daemon.sh + collect_logs.sh
            (all through scripts/lib/transport.sh: gs:// | s3:// | ssh:// | file://)
[pull+$0]   scripts/6_teardown/teardown_pod.sh <pod_id> <backup_target> <local_run_dir>
            (ledger-gated pull_run.sh FIRST, delete LAST, read-only $0 listing)
```

Sessions and cell carriage are `MyDocs/PUBLICATION.md` §7.6/§7.6.1; FINAL SCOPE v2
(the RunPod plan: which runs, which pods, the L40S S0 gate) is
`MyDocs/COST_NEBIUS_RUNPOD_2026-08-16.md`. Engine pins and the engine×model
VERIFY-LIVE matrix: `cloud/VLLM_COMPATIBILITY.md` (§7; act-2 RDMA preflight §8).

### 1.1 Session vocabulary — the four session ids (tracked definition)

The campaign runs as **three provisioning sessions**; C and D share one session in
two acts. The session id is a path level of every results tree
(`results/<campaign>/<session>/<run_id>/`, `cloud/RESULTS_LAYOUT.md` §1) and is
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
`cloud/VLLM_COMPATIBILITY.md` §8) and its own user approval. Hardware shapes and
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
mkdir -p ~/CAGE && tar xzf cage_*.tar.gz -C ~/CAGE
head -3 ~/CAGE/BUILD_INFO                      # verify sha/dirty/packaged_at
cd ~/CAGE && bash scripts/1_setup/setup_runpod.sh
source cage-env/bin/activate
```

`setup_runpod.sh` is container-shaped (root, no sudo/systemd/PPA — finding J7): it
installs the pinned vLLM + `requirements.txt` into `cage-env` built from the
**canonical interpreter** (`CAGE_CANONICAL_PYTHON`, finding B1 — it fails closed rather
than fall back to bare `python3`; never hand-build the venv with `python3 -m venv`),
exports `HF_HUB_DOWNLOAD_TIMEOUT` BEFORE dataset staging and model prefetch, stages the
full charter dataset roster (D5), and prefetches the FINAL-SCOPE model roster
(override per pod role: `PREFETCH_MODELS="Qwen/Qwen3-14B" bash scripts/1_setup/setup_runpod.sh`).

Long-job discipline (kept from the pilots — still true): never run a long command
through a blocking SSH. `scripts/ops/remote_job.sh` gives submit/status/tail/wait/kill
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
environment-vs-registration pins, (j) charter §6.5 realized-KV **iso-BYTES parity**
across engine startup logs (`CAGE_ISO_BYTES_TOL`/`CAGE_ISO_BYTES_LOGS`), (k)
per-backend endpoint liveness (`CAGE_PREFLIGHT_BACKENDS`), (l) campaign-layout
round-trip, (m) open-loop schedule + measured-replay guard, (n) calibration artifact,
(o) regime-inputs bridge on live telemetry, (p) dataset staleness refusal.

Version pins: record the actually-served engine versions into the run manifest; the
engine×model VERIFY-LIVE matrix is `cloud/VLLM_COMPATIBILITY.md` §7. Re-run gate (a)
after **every** engine relaunch (prefix ON/OFF, policy knobs, and topology are
launch-time levers — relaunches between cells are normal).

## 4. Run

The J4 refusal gate applies at launch: a run with NO off-box backup target **refuses to
start** (`require_backup_target`, `scripts/lib/transport.sh`). Export
`CAGE_BACKUP_TARGET` first — on RunPod normally the network-volume S3 API
(`s3://<volume>[/prefix]` + `CAGE_S3_ENDPOINT` + AWS creds) or `ssh://[user@]host/path`.

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
- Quality scoring is decoupled by default (`CAGE_SKIP_QUALITY=1` — a *declared*
  regime, not a mock): the box's job is serving measurements + raw outputs + evidence;
  model-based quality is scored after the serving trees.

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
sync. The remote layout mirrors the local tree exactly (`cloud/RESULTS_LAYOUT.md` §4).

**End sequence — teardown is irreversible; never reorder these steps:**

```
[1] final sync + stop daemon    gcs_backup_daemon.sh stop   (final authoritative sync)
[2] VERIFIED PULL + TEARDOWN    scripts/6_teardown/teardown_pod.sh <pod_id> <backup_target> <local_run_dir>
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
| `CAGE_S3_ENDPOINT` | s3 backend | Endpoint URL for `aws s3` — points it at the RunPod network-volume S3 API. Pair with `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` from the volume's credentials. |
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
| `CAGE_SKIP_QUALITY=1` | run scripts | Decoupled-scoring regime (default in `run_full_sweep.sh`): inline model-based quality is skipped and scored after the serving trees. A *declared* regime, not a mock. |
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
| Bootstrap the pod | `bash scripts/1_setup/setup_runpod.sh` |
| Live preflight (gates a–p) | `bash scripts/checks/preflight_check.sh <MODEL> <API_BASE>` |
| Full sweep (one run-id) | `nohup bash scripts/3_run/run_full_sweep.sh <MODEL> <N> <T> > sweep.log 2>&1 &` |
| Submit long remote job | `scripts/ops/remote_job.sh submit <name> '<cmd>' [deadline_s]` |
| One-shot sync | `bash scripts/5_observability/sync_results.sh <dir> [target]` |
| Start/stop backup daemon | `bash scripts/5_observability/gcs_backup_daemon.sh start\|stop [phase_dir]` |
| Collect logs + forensics | `bash scripts/5_observability/collect_logs.sh` |
| Verified pull (ledger gate) | `bash scripts/5_observability/pull_run.sh <target> <local_run_dir>` |
| Fail-closed teardown + $0 | `scripts/6_teardown/teardown_pod.sh <pod_id> <target> <local_run_dir>` |

Compatibility gates and pins: `cloud/VLLM_COMPATIBILITY.md`.
Results tree + ledger spec: `cloud/RESULTS_LAYOUT.md`.
Design authority: `MyDocs/PUBLICATION.md` (§7.6 groups, §7.6.1 matrix, §7.7f lifecycle).

---

## Appendix A — GCP port (retained, not current)

GCP is kept as a **portability backend**: everything below works, none of it is the
primary path, and nothing here weakens the §0 disciplines.

- **Provision**: `terraform/` with one tfvars file per session
  (`terraform/sessions/group-a.tfvars`, `group-b.tfvars`, `group-cd.tfvars` — each sets
  the terraform `session` variable, which is also the `session` label stamped on every
  resource). `terraform plan` is always allowed; `terraform apply` is GATED by the
  §0.1 user approval. Label everything `agent-run=<run_id>` so the orphan sweep can
  find strays.
- **Setup**: `bash scripts/1_setup/setup_gpu_cloud.sh` (DLVM-shaped: sudo/systemd; the
  RunPod script is the container-shaped primary). Same canonical-interpreter rule (B1).
- **Ops**: `scripts/ops/gpu_vm.sh create` (pilot-era L4 zone-hunt) and
  `scripts/ops/remote_job.sh` (provider-agnostic over SSH). SSH flags that keep agents
  sane: `-o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes`, plus
  `CLOUDSDK_CORE_DISABLE_PROMPTS=1` (a TTY-less prompt hangs forever). Kill by the
  RECORDED PID, never `pkill -f <script>`. A non-login `ssh --command` does NOT
  inherit the run's env — forward `CAGE_BACKUP_TARGET` (etc.) explicitly.
- **Backup default**: on a GCP box (and only there) the metadata server derives
  `gs://<project>-cage-results` when no target is set (`transport_default_target`).
- **Teardown**: `scripts/6_teardown/teardown_vm.sh <vm> <zone>` — same fail-closed
  ordering (COLLECT_OK sentinel + complete local pull before delete). Skipping the
  pull needs the J10 double ceremony (`CAGE_SKIP_LOCAL_PULL=1` AND
  `CAGE_SKIP_LOCAL_PULL_CONFIRM=I-ACCEPT-DATA-LOSS`, bypass marker recorded).
  Prove $0 by label: instances, disks, buckets all empty for
  `labels.agent-run=<run_id>`; `scripts/ops/gpu_vm.sh sweep` is the universal check.
- **RDMA path (C/D act 2, [Extension])**: H200 capacity via `a3-ultragpu-8g`
  (typically DWS Flex-start / calendar reservation) needs an RDMA-network-profile VPC;
  the act-2 gate is the `cloud/VLLM_COMPATIBILITY.md` §8 RDMA preflight. No
  RDMA-capable fabric → the RDMA rung cannot run there (the TCP rung still can).
