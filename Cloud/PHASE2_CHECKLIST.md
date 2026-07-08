# Phase 2 — Single-GPU Cloud Run: Checklist & Definition of Done

> The one authoritative, ordered procedure for Phase 2. Phase 2 runs the CAGE suite on a single
> real NVIDIA **L4** GPU on GCP: the nine baselines (seven on a single node + the two compression
> arms), FP8 KV compression and speculative decoding as launch-time levers, GPU + serving
> telemetry, and durable result sync. Phase 1 (local CPU) is done; Phase 3 (multi-node, real
> cross-node transfer) is [`PHASE3_PLAN.md`](PHASE3_PLAN.md).
>
> Cross-refs: serving levers + version pin → [`VLLM_COMPATIBILITY.md`](VLLM_COMPATIBILITY.md);
> deeper ops → [`RUNBOOK.md`](RUNBOOK.md); console click-through → `CLOUD_CONSOLE_GUIDE.md`.

## What Phase 2 delivers (maps to the dissertation)

| Dissertation component | How it runs in Phase 2 |
|---|---|
| Nine baselines | `cloud_run.sh` runs the 7 non-distributed baselines; `run_compression.sh` adds `compressed_rag`/`compressed_cag`. (`distributed` is Phase 3.) |
| vLLM integration | Local `vllm serve` via `manage_vllm_server.sh`, pinned to v0.11.0; streaming TTFT, `cached_tokens`, headers. |
| cage-stats | Installed from `requirements.txt`; `--vllm-telemetry` scrapes `/metrics` (spec-decode acceptance, KV dtype, token source). |
| Analytical components | `compression.analytical_kv_footprint` writes `compression_analytical` into each run's metrics JSON; empirical footprint from `GPUMetricsTracker`. |
| Orchestrator + features | `run_experiment.py` (warmup split, context-source control, per-trial seeds, metric suite). |

## Prerequisites

- [ ] GCP project with billing; **L4 quota** in the target region (`NVIDIA_L4_GPUS >= 1`).
- [ ] `HF_TOKEN` for the model (Qwen3 is gated on some mirrors).
- [ ] vLLM pinned version confirmed in [`VLLM_COMPATIBILITY.md`](VLLM_COMPATIBILITY.md) (currently `v0.11.0`).

## Steps

**0. Provision a single L4 VM.**
- [ ] `terraform -chdir=terraform/gcp apply -var num_replicas=1 -var preemptible=true` (Spot for the cheap sweep), or create a `g2-standard-8` + 1× `nvidia-l4` Deep Learning VM in the console.
- [ ] Wait for the NVIDIA driver: `nvidia-smi` returns a table.

**1. Bootstrap the environment (makes everything run on GCP).**
- [ ] `git clone <repo>` (or it is already at `/opt/cage`), then `bash scripts/setup/setup_gpu_cloud.sh`.
- [ ] Confirm the verify step prints `pynvml OK` and `cage_stats import OK`.

**2. Validate GPU fit BEFORE the long sweep (the one real unknown).**
- [ ] Start the server once and confirm the model loads: `./scripts/manage_vllm_server.sh restart Qwen/Qwen3-8B`.
- [ ] On a 24 GB L4, **Qwen3-8B + FP8 + speculative can OOM at load.** If it does:
  - lower `--gpu-memory-utilization` (terraform default is now `0.85`; try `0.80`), or
  - use **Qwen3-4B** for the FP8 / speculative arms (`MODEL=Qwen/Qwen3-4B`).

**3. Run the core suite (7 baselines, telemetry, continuous GCS sync).**
- [ ] `nohup bash scripts/cloud_run.sh Qwen/Qwen3-8B 500 3 > run.log 2>&1 &`
- [ ] Confirm `analysis/phase1/...` fills with per-baseline `metrics.json` + `vllm_telemetry.json`, and that results mirror to `gs://<project>-cage-results`.

**4. FP8 × prefix-cache gate (guards the compression confound).**
- [ ] `bash scripts/check_fp8_prefix_cache.sh` must print **PASS** (`cached_tokens > 0` under FP8). A FAIL means `compressed_cag` is "no-reuse + compression" and the axis is confounded — do not trust those numbers.

**5. Compression 2×2 axis.**
- [ ] `bash scripts/run_compression.sh Qwen/Qwen3-8B` (runs the gate first, then `cag_full`/`rag_full`/`compressed_rag`/`compressed_cag`). Confirm `compression_analytical` and the empirical GPU footprint are both present.

**6. Speculative decoding.**
- [ ] `bash scripts/run_phase5.sh` — confirm `vllm_telemetry.json` shows non-zero `spec_decode` acceptance.

**7. Collect + analyze.**
- [ ] `bash scripts/sync_results_to_gcs.sh analysis` (final flush).
- [ ] `python scripts/statistical_tests.py` over the result files (per-query Wilcoxon + Holm + bootstrap).

**8. Teardown (bounds cost).**
- [ ] `terraform -chdir=terraform/gcp destroy` — compute is removed; the versioned results bucket (`force_destroy=false`) is retained.

## Definition of Done

Phase 2 is complete when **all** of the following hold:

- [ ] All **nine** baseline families have result files (7 from the suite + `compressed_rag`/`compressed_cag`; `distributed` deferred to Phase 3 and labeled as such).
- [ ] `gpu` is **non-null** in every GPU-run `metrics.json` (memory-pressure telemetry captured — the point of Phase 2).
- [ ] The FP8 × prefix-cache gate **passed** (recorded), so the compression axis is unconfounded.
- [ ] `compression_analytical` (analytical KV footprint) and an empirical footprint are both recorded for the compression arms.
- [ ] `vllm_telemetry.json` shows speculative acceptance for the `speculative` arm.
- [ ] All results are in the durable GCS bucket and survive teardown.
- [ ] `statistical_tests.py` produced significance + effect sizes for the headline comparisons.

## Troubleshooting

| Symptom | Fix |
|---|---|
| vLLM OOM at load | Lower `--gpu-memory-utilization` or use Qwen3-4B for FP8/spec arms (Step 2). |
| `gpu` is null in metrics | `pynvml` not installed on the host — re-run the bootstrap (it is now in `requirements.txt`). |
| `vllm_telemetry` skipped | cage-stats not importable — `pip install -r requirements.txt`, or set `CAGE_STATS_HOME`. |
| FP8 gate FAILS | Record it; FP8 disabled prefix caching on this build — treat `compressed_cag` as confounded for this version. |
| Datasets download mid-run | Pre-stage with `scripts/download_datasets.py --dataset {squad_v2,natural_questions,musique}`. |
