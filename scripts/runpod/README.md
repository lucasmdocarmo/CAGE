# `scripts/runpod/` — RunPod ops suite (PRIMARY provider)

RunPod-ONLY scripts (hard `runpodctl`/RunPod-REST dependence; ADR-0090: RunPod is
provisioned via `runpodctl`, no terraform). They **bracket** the numbered lifecycle
stages (`scripts/1_setup/` … `scripts/5_observability/`): provisioning before stage 1,
teardown after stage 5, monitoring alongside every stage. All local scripts speak the
**runpodctl 2.x (v2) command tree** — `pod list/create/delete`, `network-volume list` —
never the retired v1 verbs (`get pod`/`remove pod`); the REST fallback targets
`https://api.runpod.io/v2` (see `MyDocs/runpod-cli-reference.md`).

| Script | Order (vs numbered stages) | Objective | Cloud |
|---|---|---|---|
| `provision_pod.sh` | **before 1** (opens the lifecycle) | Approval-gate-aware pod creation: DEFAULT = PLAN mode (prints shape/image/seatbelt/$-per-h/estimated total, creates NOTHING, exit 0); `--yes` = the recorded owner GO → real `pod create` with the **12h `--terminate-after` seatbelt** + a `create` event appended to the pod ledger. | runpod |
| `setup_runpod.sh` | **stage 1, ON the pod** | Container-shaped bootstrap (root, no systemd): canonical CPython env, charter dataset roster, model prefetch. Run it from the shipped repo tarball (`scripts/ops/package_repo.sh`). | runpod |
| `pod_status.sh` | **alongside 2–5** (watch loop) | READ-ONLY monitor: `pod list --all` + per-pod `pod get`, joined with the ledger → uptime + estimated spend; **exits nonzero when any pod's known age exceeds `--max-age-hours` (default 24)** — the runaway-cost alarm. | runpod |
| `cost_report.sh` | **alongside / after any stage** | OFFLINE cost table from the ledger's create/delete pairs (open pods flagged LIVE/BILLING; malformed lines refused loudly by line number). `--billing` adds `runpodctl billing` — the account **authority**. | runpod |
| `teardown_pod.sh` | **after 5** (closes the lifecycle) | Fail-closed $0 teardown: final on-pod sync → **verified pull gate** (`pull_run.sh` must print `SAFE TO TEARDOWN`) → confirm ceremony → `pod delete` (appends the ledger `delete` event) → read-only $0 proof: `pod list --all` pod-free AND `network-volume list` empty. | runpod |

## Observability story (what watches a run, and from where)

- **On the pod:** `scripts/5_observability/collect_logs.sh` gathers logs/forensics and
  `scripts/5_observability/sync_results.sh` mirrors the run tree to the backup target —
  both through the provider-neutral `scripts/lib/transport.sh`
  (`s3://` network-volume endpoint, `ssh://`, `gs://`, `file://`).
- **Locally (during the run):** `scripts/5_observability/watch_campaign.sh` follows the
  synced markers/progress; `pod_status.sh` (this dir) is the cost/uptime watchdog —
  loop it with `--max-age-hours` as the runaway alarm.
- **Locally (end of run):** `scripts/5_observability/pull_run.sh` mirrors the backup
  target and re-hashes the sha256 ledger — only its literal `SAFE TO TEARDOWN` line
  authorizes `teardown_pod.sh`. `cost_report.sh` prints the final spend
  (`--billing` for the account authority).

## The pod ledger

`results/ops/pod_ledger.jsonl` (override: `CAGE_POD_LEDGER`; gitignored under
`results/`). Append-only JSONL, one event per line:

```json
{"ts_utc": "…Z", "pod_id": "…", "name": "…", "gpu_id": "…", "gpu_count": 1,
 "price_per_hour_usd": 0.86, "terminate_after": "12h", "purpose": "…", "event": "create"}
{"ts_utc": "…Z", "pod_id": "…", "event": "delete"}
```

`price_per_hour_usd` is a number or `null` (pass `--price-per-hour` at provision to make
spend estimates work); `terminate_after` is a duration string or `null` (seatbelt
disabled). Writers: `provision_pod.sh` (create), `teardown_pod.sh` (delete). Readers:
`pod_status.sh`, `cost_report.sh` — the latter REFUSES malformed lines loudly with the
line number rather than skipping them.

## Seatbelt + approval policy (standing)

- **No pod without the owner GO**: `provision_pod.sh` is plan-only until `--yes`.
- **Every pod gets `--terminate-after`** (default `12h`) — a server-side deadline that
  survives a dead laptop. `--no-terminate-after` exists but warns LOUDLY; then
  `teardown_pod.sh` is the only stop.
- **$0 means proven $0**: `pod delete` + `pod list --all` pod-free + `network-volume list`
  empty (a surviving volume is a clean-room violation), after the ledger-verified pull.
