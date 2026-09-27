# `scripts/runpod/` — RunPod ops suite (PRIMARY provider)

RunPod-ONLY scripts (hard `runpodctl`/RunPod-REST dependence; ADR-0090: RunPod is
provisioned via `runpodctl`, no terraform). They **bracket** the numbered lifecycle
stages (`scripts/1_setup/` … `scripts/5_observability/`): provisioning before stage 1,
teardown after stage 5, monitoring alongside every stage. All local scripts speak the
**runpodctl 2.x (v2) command tree** — `pod list/create/delete`, `network-volume list` —
never the retired v1 verbs (`get pod`/`remove pod`); the REST fallback targets
`https://api.runpod.io/v2` (see `MyDocs/RunPod/runpod-cli-reference.md`).

| Script | Order (vs numbered stages) | Objective | Cloud |
|---|---|---|---|
| `provision_pod.sh` | **before 1** (opens the lifecycle) | Approval-gate-aware pod creation: DEFAULT = PLAN mode (prints shape/image/seatbelt/$-per-h/estimated total, creates NOTHING, exit 0); `--yes` = the recorded owner GO → real `pod create`, then the **12h client-side seatbelt is armed** (`pod_watchdog.sh`) and a `create` event (with `watchdog_pid`) is appended to the pod ledger. | runpod |
| `pod_watchdog.sh` | **before 1, armed right after the create; runs until the deadline or `teardown_pod.sh`** | The cost seatbelt that exists: RunPod has NO server-side auto-terminate (`pod create --terminate-after` is `usage_error` on 2.11.0 and 2.14.0; no v2 API field; live 2026-09-26), so `arm <pod_id> <RFC3339Z>` spawns a detached loop on the operator's Mac that runs `pod delete` at the instant, re-checks the listing, appends a `delete` event with `"by":"watchdog"`, and exits 1 loudly if the pod survives. `status`, `disarm`. Dies with the Mac (off/asleep = fires at next wake). | runpod |
| `setup_runpod.sh` | **stage 1, ON the pod** | Container-shaped bootstrap (root, no systemd): canonical CPython env, charter dataset roster, model prefetch. Run it from the shipped repo tarball (`scripts/ops/package_repo.sh`). | runpod |
| `pod_status.sh` | **alongside 2–5** (watch loop) | READ-ONLY monitor: `pod list --all` + per-pod `pod get`, joined with the ledger → uptime + estimated spend; **exits nonzero when any pod's known age exceeds `--max-age-hours` (default 24)** — the runaway-cost alarm. | runpod |
| `cost_report.sh` | **alongside / after any stage** | OFFLINE cost table from the ledger's create/delete pairs (open pods flagged LIVE/BILLING; malformed lines refused loudly by line number). `--billing` adds `runpodctl billing pods` + `runpodctl billing network-volume`: the account **authority** (the bare `billing` verb is a command group that only prints help). | runpod |
| `teardown_pod.sh` | **after 5** (closes the lifecycle) | Fail-closed $0 teardown: final on-pod sync → **verified pull gate** (`pull_run.sh` must print `SAFE TO TEARDOWN`) → confirm ceremony → `pod delete` (appends the ledger `delete` event) → read-only $0 proof: `pod list --all` pod-free AND `network-volume list` empty. | runpod |

## Observability story (what watches a run, and from where)

- **On the pod:** `scripts/5_observability/collect_logs.sh` gathers logs/forensics and
  `scripts/5_observability/sync_results.sh` mirrors the run tree to the backup target —
  both through the provider-neutral `scripts/lib/transport.sh`
  (`s3://` network-volume endpoint, `ssh://`, `gs://`, `file://`).
- **Locally (during the run):** `scripts/5_observability/watch_campaign.sh` follows the
  synced markers/progress; `pod_status.sh` (this dir) is the cost/uptime alarm:
  loop it with `--max-age-hours`; it also prints each live pod's seatbelt state
  (watchdog ALIVE / DEAD / NONE), and `pod_watchdog.sh status` lists the armed loops.
- **Locally (end of run):** `scripts/5_observability/pull_run.sh` mirrors the backup
  target and re-hashes the sha256 ledger — only its literal `SAFE TO TEARDOWN` line
  authorizes `teardown_pod.sh`. `cost_report.sh` prints the final spend
  (`--billing` for the account authority).

## The pod ledger

`results/ops/pod_ledger.jsonl` (override: `CAGE_POD_LEDGER`; gitignored under
`results/`). Append-only JSONL, one event per line:

```json
{"ts_utc": "…Z", "pod_id": "…", "name": "…", "gpu_id": "…", "gpu_count": 1,
 "price_per_hour_usd": 0.86, "terminate_after": "2026-08-26T22:00:00Z", "watchdog_pid": 48213,
 "data_center_ids": "US-IL-1,EU-RO-1", "network_volume_id": null, "purpose": "…", "event": "create"}
{"ts_utc": "…Z", "pod_id": "…", "event": "delete"}
{"ts_utc": "…Z", "pod_id": "…", "event": "delete", "by": "watchdog"}
```

`price_per_hour_usd` is a number or `null` (pass `--price-per-hour` at provision to make
spend estimates work); `terminate_after` is the RESOLVED absolute RFC3339 deadline
(`…Z`) the client-side watchdog guards, or `null` (seatbelt disabled): an instant, so
readers can compare it to a wall clock, which a duration string could not support;
`watchdog_pid` is the armed loop's PID (int) or `null` (seatbelt disabled, or arming
failed, which `provision_pod.sh` announces on stderr as `SEATBELT NOT ARMED` right after
the ledger line). Delete events have three writers: `teardown_pod.sh` (no `by`),
`pod_watchdog.sh` when the seatbelt fired (`"by": "watchdog"`), and the operator by hand
after a raw `runpodctl pod delete` (the runbook's bad-machine fallback). `data_center_ids` is the verbatim
`--data-center-ids` CSV or `null` (no siting pin); `network_volume_id` is the
`--network-volume-id` value or `null` — recorded because a network volume attaches ONLY
at create time and ONLY in its own datacenter, so the ledger is the audit trail of
where the pod (and its volume) were sited. Writers: `provision_pod.sh` (create), `teardown_pod.sh` (delete), `pod_watchdog.sh` (delete with `"by": "watchdog"` when the seatbelt fired). Readers:
`pod_status.sh`, `cost_report.sh` — the latter REFUSES malformed lines loudly with the
line number rather than skipping them.

## Seatbelt + approval policy (standing)

- **No pod without the owner GO**: `provision_pod.sh` is plan-only until `--yes`.
- **Every pod gets a seatbelt** (`--terminate-after`, default `12h`), and it is CLIENT-SIDE:
  RunPod has no server-side auto-terminate (`runpodctl pod create --terminate-after` is
  `usage_error` on 2.11.0 and 2.14.0, the v2 API has no field, and RunPod's docs schedule
  a stop with a client-side `sleep`; live 2026-09-26). `provision_pod.sh` resolves the
  duration to an RFC3339 instant and arms `pod_watchdog.sh`, which deletes the pod at
  that instant from the operator's Mac and is disarmed by `teardown_pod.sh`. It dies with
  the Mac (off or asleep at the deadline = fires at the next wake), so the Mac stays on
  and awake for the run, `pod_status.sh --max-age-hours` loops as the alarm, and the
  console is the last stop. `--no-terminate-after` exists but warns LOUDLY; then
  `teardown_pod.sh` is the only stop.
- **$0 means proven $0**: `pod delete` + `pod list --all` pod-free + `network-volume list`
  empty (a surviving volume is a clean-room violation), after the ledger-verified pull.
