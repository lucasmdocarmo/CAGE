# CAGE scripts

Organized by **lifecycle stage**, numbered to show execution order. Anything off the live run
path lives in [`deprecated/`](deprecated/README.md) (untouched by the ordering).

```
scripts/
  1_setup/          provision the box + pull data      setup_runpod.sh (RunPod PRIMARY)  setup_gpu_cloud.sh (GCP port)  download_datasets.py
  2_serving/        start / manage the engines         manage_vllm_server.sh  manage_sglang_server.sh  manage_lmdeploy_server.sh  manage_vllm_cluster.py  deploy_cluster.sh
  3_run/            run the experiments                cloud_run.sh  run_full_sweep.sh  run_baselines.sh  run_compression.sh  run_kv_store.sh  run_experiment.py
  4_analysis/       verify + index + stats             verify_results.py  organize_results.py  run_campaign_analysis.py
                    (campaign, D9)                     run_calibration.py  run_power_sim.py  score_instrument_b.py  rescore_quality.py
                    (pilot archive, 2026-07)           run_phase2_stats.sh  statistical_tests.py  token_divergence.py  generate_plots.py
  5_observability/  live monitor + durable off-box mirror  observe_run.py  watch_run.sh  watch_campaign.sh  sync_results.sh (provider-neutral; sync_results_to_gcs.sh = compat shim)  gcs_backup_daemon.sh  pull_run.sh  log_sync_daemon.sh  collect_logs.sh  gcp_shutdown_hook.sh  run_status_logger.sh
  6_teardown/       pull-verified $0 teardown           teardown_pod.sh (RunPod PRIMARY)  teardown_vm.sh (GCP port)
  checks/           gates & tests (run as needed)      preflight_check.sh  check_fp8_prefix_cache.sh  smoke_staleness.sh  run_tests.sh
  lib/              sourced by drivers (not run)        _common.sh  _serving_config.sh  _log_guard.sh  transport.sh (gs://|s3://|ssh://|file:// backends)
  deprecated/       off the live path (see README.md)   run_speculative_matrix.sh  check_mtp_spec_decode.sh  (speculative arms retired, charter §7.5)
```

The numbered folders are the **happy-path order**; `checks/` and `5_observability/` run
*alongside* the numbered stages (a gate before, a monitor during), not at a fixed position —
which is why they aren't numbered. `lib/` is sourced, never executed directly.

## Live analysis path (campaign, RESULTS_LAYOUT v2)

Campaign runs land as `results/<campaign>/<session>/<run_id>/` trees carrying `manifest.json`,
`cells/<row_key>/window_<dataset>-<k>/`, and the §5 sha256 ledger (`cloud/RESULTS_LAYOUT.md` is
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
