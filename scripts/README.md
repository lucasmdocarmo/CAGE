# CAGE scripts

Organized by **lifecycle stage**, numbered to show execution order. Anything off the live run
path lives in [`deprecated/`](deprecated/DEPRECATED.md) (untouched by the ordering).

```
scripts/
  1_setup/          provision the box + pull data      setup_gpu_cloud.sh  download_datasets.py
  2_serving/        start / manage vLLM                manage_vllm_server.sh  manage_vllm_cluster.py  deploy_cluster.sh
  3_run/            run the experiments                cloud_run.sh  run_baselines.sh  run_compression.sh  run_speculative_matrix.sh  run_experiment.py
  4_analysis/       stats + plots + verify             run_phase2_stats.sh  statistical_tests.py  token_divergence.py  generate_plots.py  verify_results.py
  5_observability/  live monitor + durable GCS mirror  observe_run.py  watch_run.sh  sync_results_to_gcs.sh  log_sync_daemon.sh  collect_logs.sh  gcp_shutdown_hook.sh  run_status_logger.sh
  6_teardown/       sentinel-verified $0 teardown       teardown_vm.sh
  checks/           gates & tests (run as needed)      preflight_check.sh  check_fp8_prefix_cache.sh  check_mtp_spec_decode.sh  smoke_staleness.sh  run_tests.sh  simulate_network.sh
  lib/              sourced by drivers (not run)        _serving_config.sh  _log_guard.sh
  deprecated/       off the live path (see DEPRECATED.md)
```

The numbered folders are the **happy-path order**; `checks/` and `5_observability/` run
*alongside* the numbered stages (a gate before, a monitor during), not at a fixed position —
which is why they aren't numbered. `lib/` is sourced, never executed directly.

## Live path (phase2, single-node GPU)

```
# 1. provision (once per VM)
bash scripts/1_setup/setup_gpu_cloud.sh
# (checks/preflight_check.sh gates before you spend the sweep)

# 2-3. run — cloud_run.sh mints the run-id, starts vLLM, runs the core suite, auto-plots
nohup bash scripts/3_run/cloud_run.sh Qwen/Qwen3-8B 500 3 > run.log 2>&1 &
bash scripts/3_run/run_compression.sh        Qwen/Qwen3-8B        # 2x2, shares the run-id
bash scripts/3_run/run_speculative_matrix.sh Qwen/Qwen3-8B        # 2x2 (repeat for MiMo-7B-RL)

# 4. aggregate + stats (reads the shared run root)
bash scripts/4_analysis/run_phase2_stats.sh

# 6. teardown to $0
bash scripts/6_teardown/teardown_vm.sh <instance> <zone>
```

**All run outputs go to `results/<phase>/<run-id>/`** (`run-id = <YYYY-MM-DD_HHMM>_<model-slug>_<Q>x<T>`),
minted once by `cloud_run.sh` and exported as `CAGE_RUN_ROOT` / `CAGE_RUN_ID` / `CAGE_PHASE`. Every
tree (core / compression / speculative), the stats, the plots, and observability nest under that one
root, so runs never mix and the GCS bucket mirrors it verbatim. To keep the three run trees in ONE
root, export `CAGE_RUN_ID` once and reuse it across the invocations (or let `run_phase2_stats.sh`
default to the newest run under `results/<phase>/`). Never write to the legacy `analysis/`.

Env knobs: `PHASE` (default `phase2`), `CAGE_RUN_ID` (override the auto run-id), `CAGE_AUTO_PLOTS=0`
(skip end-of-run plotting), `ENABLE_DISTRIBUTED=1` (opt into the local 3-replica arm), `VLLM_TELEMETRY=0`.

### Path convention (for maintainers)
Scripts live two levels deep now (`scripts/<stage>/<name>`), so each resolves the repo root as
`PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"` (bash) / `Path(__file__).resolve().parents[2]`
(python), and calls a sibling in another stage via `$SCRIPT_DIR/../<stage>/<name>` or
`scripts/<stage>/<name>`.
