# CAGE scripts

Organized by **lifecycle stage**, numbered to show execution order. Anything off the live run
path lives in [`deprecated/`](deprecated/README.md) (untouched by the ordering).

```
scripts/
  1_setup/          pull data onto the box             download_datasets.py  build_query_manifest.py
  2_serving/        start / manage the engines         manage_vllm_server.sh  manage_sglang_server.sh  manage_lmdeploy_server.sh  manage_vllm_cluster.py  deploy_cluster.sh
  3_run/            run the experiments                cloud_run.sh  run_full_sweep.sh  run_baselines.sh  run_compression.sh  run_kv_store.sh  run_experiment.py  seal_campaign_run.py
  4_analysis/       verify + index + stats             verify_results.py  organize_results.py  run_campaign_analysis.py
                    (campaign, D9)                     run_calibration.py  run_power_sim.py  score_instrument_b.py  rescore_quality.py
                    (pilot archive, 2026-07)           run_phase2_stats.sh  statistical_tests.py  token_divergence.py  generate_plots.py
  5_observability/  live monitor + durable off-box mirror  observe_run.py  watch_campaign.sh  sync_results.sh (provider-neutral)  gcs_backup_daemon.sh  pull_run.sh  log_sync_daemon.sh  collect_logs.sh  run_status_logger.sh
  runpod/           RunPod-ONLY (PRIMARY provider)     provision_pod.sh (plan -> owner GO --yes; 12h seatbelt + pod ledger)  setup_runpod.sh (container-shaped bootstrap)  pod_status.sh (uptime/spend + runaway-cost alarm)  cost_report.sh (offline ledger cost table)  teardown_pod.sh (pull-verified $0 teardown)  README.md (ops-suite map)
  gcp/              GCP-ONLY (retained port)           setup_gpu_cloud.sh  teardown_vm.sh  gpu_vm.sh  remote_job.sh  watch_run.sh  gcp_shutdown_hook.sh  sync_results_to_gcs.sh (compat shim)
  checks/           gates & tests (run as needed)      preflight_check.sh  check_fp8_prefix_cache.sh  smoke_staleness.sh  run_tests.sh
  lib/              sourced by drivers (not run)        _common.sh  _serving_config.sh  _log_guard.sh  transport.sh (gs://|s3://|ssh://|file:// backends)
  deprecated/       off the live path (see README.md)   run_speculative_matrix.sh  check_mtp_spec_decode.sh  (speculative arms retired, charter §7.5)
```

The numbered folders are the **happy-path order**; `checks/` and `5_observability/` run
*alongside* the numbered stages (a gate before, a monitor during), not at a fixed position —
which is why they aren't numbered. `lib/` is sourced, never executed directly.

**Provider dirs** (`runpod/`, `gcp/`): a script lives there iff it is provider-ONLY — it
hard-depends on that provider's tooling/surface (`runpodctl`/RunPod REST/pod container shape
vs `gcloud`/`gsutil`/GCE metadata/`terraform/gcp/` state). Provisioning and teardown bracket
the numbered stages, so both providers' setup/teardown pairs live here, not in numbered dirs.
A script that merely *supports* `gs://` as one transport scheme via `scripts/lib/transport.sh`
is provider-neutral and stays in its lifecycle dir. GCP IaC lives in `terraform/gcp/`
(RunPod has none — `runpodctl` provisions; see `terraform/README.md` and `docs/RUNBOOK.md`).

## Master catalog (Script | Order | Objective | Cloud)

Every non-deprecated script declares three header lines at its top (`# `-comments right
after the shebang for `.sh`; the first lines of the module docstring for `.py`):

```
Order:     <where it sits: stage number / gate / bracket, and what runs before/after>
Objective: <one line — what the script actually does>
Cloud:     gcp | runpod | both | local
```

`Cloud:` semantics — `gcp` / `runpod`: runs only against that provider's surfaces;
`both`: used in cloud runs on either provider (transport-neutral observability, run
drivers executed on any pod/VM); `local`: operator's machine / analysis only.
**`tests/test_script_headers.py` enforces the header contract** (labels present, Cloud
value legal, provider dirs self-consistent) **and that this catalog matches the
headers** — the table below is generated FROM the headers, so regenerate the row when
you change a header, and add a row (plus headers) for every new script.

| Script | Order | Objective | Cloud |
|---|---|---|---|
| `1_setup/build_query_manifest.py` | stage 1 — once per (dataset, N, T, seed), before any runner; consumed by every cell via CAGE_QUERY_MANIFEST | Build the uniform-yardstick query manifest so all cells/engines/models measure the SAME query set | both |
| `1_setup/download_datasets.py` | stage 1 — run by the provider bootstrap (runpod/setup_runpod.sh, gcp/setup_gpu_cloud.sh) or manually, before any serving | Stage the charter HF datasets (+ RAGTruth/TRUE calibration anchors) with fail-closed non-empty-split checks | both |
| `2_serving/deploy_cluster.sh` | stage 2 — multi-node cluster bring-up (side path), before 3_run | Deploy a multi-node vLLM cluster: local Docker Compose / generic-k8s side paths + the confirm-gated terraform/gcp stack (its only cloud surface; unused on RunPod) | gcp |
| `2_serving/manage_lmdeploy_server.sh` | stage 2 — after 1_setup, before any 3_run driver; charter engine #3 launcher | Start/stop/status the LMDeploy TurboMind api_server under the uniform regime (fail-closed iso-bytes mapping; TurboMind asserted from the launch log) | both |
| `2_serving/manage_sglang_server.sh` | stage 2 — after 1_setup, before any 3_run driver; charter engine #2 launcher | Start/stop/status the SGLang server under the SAME uniform regime as vLLM (iso-bytes dial mapping) | both |
| `2_serving/manage_vllm_cluster.py` | stage 2 — after 1_setup, before the distributed 3_run arm | Start/validate/stop a local multi-replica vLLM cluster plus the CAGE router as one unit | both |
| `2_serving/manage_vllm_server.sh` | stage 2 — after 1_setup, before any 3_run driver; re-run per engine relaunch | Start/stop/restart/status the vLLM server under the uniform serving regime (lib/_serving_config.sh) | both |
| `3_run/calibrate_cell.py` | stage 3 — before the D6 campaign cells it calibrates; needs a live engine ([VERIFY-LIVE at S0]) | Per-cell lambda*/SLO-floor calibration (D6 §6.1): sequential floor + Poisson probe ladder -> calibration JSON for the campaign driver | both |
| `3_run/cloud_run.sh` | stage 3 — core tree of run_full_sweep.sh (or standalone); starts its own vLLM via run_baselines.sh | Pilot-harness core-suite driver for ONE GPU box with continuous off-box result mirroring (J4 backup-target gate at start); CAGE_CAMPAIGN/CAGE_CAMPAIGN_ROOT routes the run into the v2 campaign tree (task #116) | both |
| `3_run/run_baselines.sh` | stage 3 — invoked by cloud_run.sh after 2_serving concepts are in place (manages the vLLM server itself) | Run the core pilot baselines (no_cache/rag/redis/prefix_cache/hybrid[/distributed]) fault-tolerantly for one model; campaign mode writes v2 cells (resume gates on window_<dataset>-NN/metrics.json, task #116) | both |
| `3_run/run_cag_reference.py` | stage 3 — HF-oracle arm, standalone (no serving stack needed); after 1_setup | True-CAG reference runner on HF transformers (greedy; corpus-KV precomputed once, cropped after every query) for the idea-gain/engine-gain decomposition | both |
| `3_run/run_compression.sh` | stage 3 — lever tree after the core suite; driven by run_full_sweep.sh or standalone | Run the ratio-matched compression 2x2 (CAG/RAG x full/compressed: client-side LLMLingua-2 vs server FP8 KV); campaign mode writes v2 cells (task #116) | both |
| `3_run/run_experiment.py` | stage 3 — the per-cell workhorse every tree driver invokes; after 2_serving | Run ONE experiment cell (model x dataset x baseline config) against the serving stack and persist per-trial results; --campaign-root/CAGE_CAMPAIGN_ROOT emits each trial as a RESULTS_LAYOUT-v2 window via campaign_layout (task #116) | both |
| `3_run/run_full_sweep.sh` | stage 3 — THE top-level sweep entry; drives cloud_run.sh + lever trees + run_phase2_stats.sh under ONE run-id | Pilot-harness full-sweep orchestrator (core suite -> compression 2x2 -> opt-in trees -> consolidated stats) with skip-completed resume; campaign mode runs core+compression as v2 cells then seals + verifies the run (task #116) | both |
| `3_run/run_kv_store.sh` | stage 3 — lever tree after the core suite; driven by run_full_sweep.sh or standalone | Run the LMCache KV-block-store arm (lmcache_rag via vLLM's KV-connector) behind live import/health pairing gates | both |
| `3_run/run_memory_sweep.sh` | stage 3 — standalone sweep with its own server lifecycle; after 1_setup | Sweep gpu_memory_utilization so KV capacity brackets the CAG corpus block (prefix-eviction mechanism readout + telemetry deltas) | both |
| `3_run/run_prefix_envelope.sh` | stage 3 — envelope cells on a served GPU box; after 2_serving | Run the prefix-cache workload-envelope cells + the true-CAG baseline (cag_true on/off, grouped/multiturn/repeat) | both |
| `3_run/seal_campaign_run.py` | stage 3 — after the LAST tree of a campaign run, before verify_results.py / any pull | Seal ONE campaign run root (write-time-hash cross-check -> §5 ledger.json via campaign_layout.seal_run) | both |
| `4_analysis/_plot_style.py` | stage 4 helper — imported by the plotting/table tools, never run directly | Shared figure style, canonical display names, family ordering and save helpers | local |
| `4_analysis/_pub_tables.py` | stage 4 helper — imported by generate_plots.py, never run directly | Publication tables (Markdown + booktabs LaTeX) for the pilot figure set, PILOT-stamped | local |
| `4_analysis/_results_loader.py` | stage 4 helper — imported by the pilot analysis tools, never run directly | Canonical pilot-layout results loader: ONE parser, ONE validity rule, ONE estimand (pooled per-example) | local |
| `4_analysis/assemble_preregistration.py` | stage 4 — the #112 freeze step, after every calibration/resolution input exists | Assemble the ONE tracked PRE_REGISTRATION.md from the draft skeleton + resolutions JSON (embed-at-freeze doctrine) | local |
| `4_analysis/build_legacy_index.py` | stage 4 — bridge step; feeds run_campaign_analysis.py design-input dry-runs | Re-key the read-only pilot 100x3 archive into a v2-shaped bridge index (cells_index.csv + provenance + skip report) | local |
| `4_analysis/build_predicate_table.py` | stage 4 — after a sealed run AND a sealed scoring pass; before stats consume predicates | Join the scoring sidecar onto evidence rows and compute the §8.5 per-query veridicality predicate table (seal-verified, fail-closed) | local |
| `4_analysis/calibrate_instrument_a_tau.py` | stage 4 — registration-time calibration, before the #112 freeze embeds its τ | Calibrate the REGISTERED Instrument-A (LettuceDetect) τ on the public RAGTruth/TRUE anchor pool (design input, not findings) | local |
| `4_analysis/figure_pipeline.py` | stage 4 — after run_campaign_analysis.py (consumes its registered stats rows + index) | Tuple-keyed campaign figure pipeline (registered forest, pressure/goodput and coverage figures) | local |
| `4_analysis/generate_plots.py` | stage 4 — pilot re-renders only; after run_phase2_stats.sh aggregation (campaign figures = figure_pipeline.py) | Render the PILOT figure/table set from the canonical loader + pilot_stats.json | local |
| `4_analysis/organize_results.py` | stage 4 — after verify_results.py, before run_campaign_analysis.py | Validate the layout and parse every cell of ONE campaign run -> index/cells_index.csv + coverage report | local |
| `4_analysis/rescore_quality.py` | stage 4 — decoupled scoring pass over a pulled tree; before build_predicate_table.py on campaign runs | Offline quality re-scorer over saved qa_evidence.jsonl (model-free --fast default; --full loads the metric stack) | local |
| `4_analysis/run_calibration.py` | stage 4 — §9.7 design-input pass over the pilot archive, before the registration freeze | Pipeline calibration: A/A split-half + effect-injection operating characteristics of the measurement machinery | local |
| `4_analysis/run_campaign_analysis.py` | stage 4 — after organize_results.py; the final analysis step | D9 stats driver over the run index (§9.11 one-look policy: design-input default, doubly-gated confirmatory mode + analysis lock) | local |
| `4_analysis/run_phase2_stats.sh` | stage 4 — final tree of run_full_sweep.sh (on-box) or a standalone pilot re-render | Aggregate a pilot run root and drive the pilot stats -> divergence -> plots chain, then sync the run off-box | both |
| `4_analysis/run_power_sim.py` | stage 4 — §9.6 design-input pass, before the registration freeze; sets the registered N | Simulation-based power analysis against pilot-calibrated noise over the exact registered test paths | local |
| `4_analysis/score_instrument_b.py` | stage 4 — decoupled scoring pass (isolated AlignScore env), after the serving trees | Instrument-B (AlignScore-large) out-of-process scorer -> per-item scores + provenance sidecar / §6 scoring tree | local |
| `4_analysis/statistical_tests.py` | stage 4 — pilot re-renders only; driven by run_phase2_stats.sh (campaign stats = run_campaign_analysis.py) | Pilot-era per-query Wilcoxon/Holm stats engine -> pilot_stats.json (deprecated; refuses the registered stats.json name) | local |
| `4_analysis/token_divergence.py` | stage 4 — §8.9 divergence pass over a pilot run root; driven by run_phase2_stats.sh or standalone | Quantify T=0 output divergence vs the reference arm (agreement rate, first-divergence token position, answer-changing split) | local |
| `4_analysis/verify_results.py` | stage 4 — FIRST gate on a pulled campaign run, before organize_results.py (pilot mode: --pilot) | Fail-closed campaign-tree verification (schema, reconciliation, duplicates, coverage, ledger, contamination) -> report OUTSIDE the tree | local |
| `5_observability/collect_logs.sh` | alongside/after stage 3 — final full run BEFORE teardown (its COLLECT_OK sentinel gates the teardown scripts) | Gather every run/system log + forensics into logs/, mirror off-box host-namespaced, and write a per-run success sentinel | both |
| `5_observability/gcs_backup_daemon.sh` | alongside stage 3 — started by run_full_sweep.sh at launch; stop does the final authoritative sync | Interval-mirror a whole results tree to the backup target for the run's duration (legacy GCS name; provider-neutral transports) | both |
| `5_observability/log_sync_daemon.sh` | alongside stage 3 — auto-started via lib/_log_guard.sh for run scripts with no sync loop of their own | Continuously mirror logs (and results) to the off-box backup target so a dying box never loses them | both |
| `5_observability/observe_run.py` | alongside stage 3 — background sidecar launched with the run drivers | Observability sidecar: run manifest + periodic GPU/serving/progress snapshots + exit-time artifact hashing, decoupled from the orchestrator | both |
| `5_observability/pull_run.sh` | after stage 3, before teardown — THE fail-closed pre-teardown gate (called by runpod/teardown_pod.sh and gcp/teardown_vm.sh) | Pull ONE campaign run from its backup target and ledger-verify it locally; only its literal SAFE-TO-TEARDOWN line authorizes destruction | both |
| `5_observability/run_status_logger.sh` | alongside stage 3 — lightweight manual logger (observe_run.py is the superseding sidecar) | Append a compact one-line run-status snapshot to a timeline log every INTERVAL seconds | both |
| `5_observability/sync_results.sh` | alongside stages 3-5 — the one-shot mirror building block every sync caller routes through | Mirror a local dir to the off-box backup target (gs://|s3://|ssh://|file:// via lib/transport.sh) with per-backend markers; loud on failure | both |
| `5_observability/watch_campaign.sh` | alongside stage 3 — single-shot (or --loop) status read of a live layout-v2 campaign run dir | Bounded local-disk campaign status: cells/windows vs expected, heartbeat age, off-box sync lag, ONE verdict line | both |
| `checks/check_fp8_prefix_cache.sh` | gate — before any compressed_cag run (3_run/run_compression.sh) | Verify FP8 KV cache still coexists with prefix caching on the pulled vLLM (else compressed_cag is confounded) | both |
| `checks/preflight_check.sh` | gate — after 2_serving is up, before EVERY 3_run launch (Gate 2; re-run after every engine relaunch) | Live infra preflight gates (a)-(q): serving health, quality stack, telemetry, retrieval, env poison, iso-bytes parity, layout round-trip | both |
| `checks/run_tests.sh` | gate — anytime; the offline suite before shipping/launching (--with-cluster is the live opt-in) | Run the CAGE test suite (plain local pytest by default; --with-cluster starts+stops a live vLLM cluster) | local |
| `checks/smoke_staleness.sh` | gate — before committing a staleness sweep (validate-before-run) | 5-query staleness smoke: assert the stale-serving injection fires and its metrics are computable | both |
| `lib/_common.sh` | sourced library, never executed | Shared helpers for every CAGE script: die/log/warn/require_cmd/confirm, CAGE_ROOT, mint_run_id, run locks, identity-checked pidfiles | both |
| `lib/_log_guard.sh` | sourced library, never executed | Auto-start log_sync_daemon.sh + EXIT-trap final collect for run scripts that have no sync loop of their own | both |
| `lib/_serving_config.sh` | sourced library, never executed | Single source of truth for the uniform serving regime (non-eager, max_len, mem-util + per-engine iso-bytes dial mappings) | both |
| `lib/transport.sh` | sourced library, never executed | Provider-neutral off-box transport verbs (resolve/join/push/pull/ls/exists/ensure) over gs://|s3://|ssh://|file:// + the J4 require_backup_target gate | both |
| `ops/package_repo.sh` | provisioning bracket — before 1_setup (ship step; workstation side) | Package the repo as a provenance-stamped tarball (BUILD_INFO) for deploy onto a box without git | local |
| `gcp/gcp_shutdown_hook.sh` | teardown bracket — runs automatically at ACPI shutdown/spot preemption (installed as instance shutdown-script metadata) | Best-effort final results+logs mirror inside the ~30s preemption budget, as the run user, never exiting early | gcp |
| `gcp/gpu_vm.sh` | provisioning bracket — before 1_setup (create/ip/zone); after teardown, `sweep` proves $0 | Create/locate/sweep the labeled CAGE L4 GPU VM with zone-hunt + shape fallback (pilot-era path; terraform/gcp is the campaign path) | gcp |
| `gcp/remote_job.sh` | alongside stages 1-4 — drives long-running commands on an EXISTING VM over SSH | Submit/poll/stream/reap long remote jobs with durable handles (remote PID, status file, resumable local state JSON) | gcp |
| `gcp/setup_gpu_cloud.sh` | provisioning bracket — on the fresh GCP VM, before/as stage 1 (it stages datasets itself) | One-shot GCP DLVM bootstrap (sudo/systemd-shaped): venv from the canonical interpreter, pinned vLLM wheel, datasets, telemetry deps | gcp |
| `gcp/sync_results_to_gcs.sh` | alongside stages 3-5 — DEPRECATED forwarding shim; canonical = 5_observability/sync_results.sh | Forward legacy GCS-era callers verbatim (same args/env/exit code) to the provider-neutral sync_results.sh | gcp |
| `gcp/teardown_vm.sh` | teardown bracket — after 5_observability's final collect; LAST step of a GCP session | Fail-closed VM teardown: sentinel-verified log collect + complete local results pull BEFORE the delete (step [4/6] enforced) | gcp |
| `gcp/watch_run.sh` | alongside stage 3 — laptop-side interval puller during a GCS-mirrored run | Pull the run's mirrored artifacts from the GCS bucket every INTERVAL seconds and print the sidecar's progress line | gcp |
| `runpod/cost_report.sh` | alongside/after any stage — offline ledger analysis; --billing adds the authoritative account view | Pair create/delete pod-ledger events into a per-pod runtime+cost table (offline-first; malformed lines refused loudly) | runpod |
| `runpod/pod_status.sh` | alongside stages 1-5 — read-only monitoring loop over live pods + the pod ledger | Join `runpodctl pod list` with the pod ledger for per-pod uptime/spend + runaway-cost alarm (exit 1 past --max-age-hours) | runpod |
| `runpod/provision_pod.sh` | provisioning bracket — before 1_setup; PLAN by default, creates only with --yes (the owner's recorded GO) | Approval-gated RunPod pod provisioning with a server-side --terminate-after cost seatbelt + pod-ledger create event | runpod |
| `runpod/setup_runpod.sh` | provisioning bracket — on the fresh pod, before/as stage 1 (it stages datasets + prefetches models itself) | Container-shaped RunPod bootstrap (root, no sudo/systemd/PPA): canonical-CPython venv, pinned vLLM, charter datasets, model prefetch | runpod |
| `runpod/teardown_pod.sh` | teardown bracket — after 5_observability's final sync; LAST step of a RunPod session | Fail-closed pod teardown: ledger-gated pull_run.sh FIRST, confirm ceremony, pod delete strictly last, read-only $0 listing | runpod |

## Live analysis path (campaign, RESULTS_LAYOUT v2)

Campaign runs land as `results/<campaign>/<session>/<run_id>/` trees carrying `manifest.json`,
`cells/<row_key>/window_<dataset>-<k>/`, and the §5 sha256 ledger (`docs/RESULTS_LAYOUT.md` is
the layout authority). After the fail-closed pull (`pull_run.sh`), the analysis chain is:

```
# 1. gate the pulled tree (schema, reconciliation, dup detection, ledger + EXTRA sweep;
#    report written OUTSIDE the tree; exit 0 only on PASS)
python3 scripts/4_analysis/verify_results.py results/<campaign>/<session>/<run_id>

# 2. validate the layout + parse every cell -> index/cells_index.csv + coverage report
python3 scripts/4_analysis/organize_results.py results/<campaign>/<session>/<run_id>

# 3. the D9 stats engine (src.analysis.stats) over the index — design-input by default;
#    the ONE confirmatory look needs the §9.11 flags + the frozen registration SHA
python3 scripts/4_analysis/run_campaign_analysis.py results/<campaign>/<session>/<run_id>
```

`run_campaign_analysis.py` writes the registered `stats.json` (per-dataset contrast rows,
gatekeeping, equivalence, exploratory BH-FDR) and, with `figure_pipeline.py`, the campaign
figures. No pilot-era tool below may touch these trees — each one refuses a root that carries
`manifest.json`/`cells/`.

## Pilot archive (2026-07) tools — design input only

These aggregate the retired PILOT (Phase-2) layout (`results/<phase>/<run-id>/{baselines,
compression,speculative,envelope,kv_store}/<cell>/trial_*/results.csv`). They are kept runnable
so the 2026-07 pilot reports can be regenerated, and for nothing else: their numbers are design
input, never citable as campaign results (charter: pilots inform design only). Their stats
artifact is `pilot_stats.json`, stamped `"engine": "pilot-era statistical_tests.py — NOT the
registered D9 artifact"` — the registered `stats.json` name is refused.

```
# aggregate a pilot run root + per-query Wilcoxon/Holm stats + figures
bash scripts/4_analysis/run_phase2_stats.sh results/<phase>/<run-id>
# pieces it drives (all pilot-layout-bound, all deprecation-bannered):
#   statistical_tests.py   pilot Wilcoxon engine  -> pilot_stats.json / pilot_stats.tex
#   token_divergence.py    §8.9 T=0 divergence over the pilot layout
#   generate_plots.py      pilot figure/table set (via _results_loader/_pub_tables)
# pilot-mode verification: verify_results.py --pilot --results-dir <dir>
```

The historical pilot run flow (provision → `cloud_run.sh` → `run_compression.sh` →
`run_phase2_stats.sh` → `teardown_vm.sh`) is retired with the pilot era; see git history for
the full recipe. Pilot outputs live under `results/<phase>/<run-id>/` (`run-id =
<YYYY-MM-DD_HHMM>_<model-slug>_<Q>x<T>`), minted by `cloud_run.sh` and exported as
`CAGE_RUN_ROOT` / `CAGE_RUN_ID` / `CAGE_PHASE`. Never write to the legacy `analysis/`.

Env knobs: `PHASE` (default `phase2`), `CAGE_RUN_ID` (override the auto run-id), `CAGE_AUTO_PLOTS=0`
(skip end-of-run plotting), `ENABLE_DISTRIBUTED=1` (opt into the local 3-replica arm), `VLLM_TELEMETRY=0`.

### Path convention (for maintainers)
Scripts live two levels deep now (`scripts/<stage>/<name>`), so each resolves the repo root as
`PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"` (bash) / `Path(__file__).resolve().parents[2]`
(python), and calls a sibling in another stage via `$SCRIPT_DIR/../<stage>/<name>` or
`scripts/<stage>/<name>`.
