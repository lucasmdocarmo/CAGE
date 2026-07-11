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
- [ ] Canonical single-VM path: create a `g2-standard-8` + 1× `nvidia-l4` Deep Learning VM per `CLOUD_CONSOLE_GUIDE.md` (on-demand recommended). The create command attaches `shutdown-script=scripts/gcp_shutdown_hook.sh`, so a preemption/delete still flushes to GCS — that makes Spot safe if you want the discount. The `terraform/gcp` module provisions the multi-node CLUSTER (Phase 3), not this single VM.
- [ ] Wait for the NVIDIA driver: `nvidia-smi` returns a table.

**1. Bootstrap the environment (makes everything run on GCP).**
- [ ] `git clone <repo>` (or it is already at `/opt/cage`), then `bash scripts/setup/setup_gpu_cloud.sh`.
- [ ] Confirm the verify step prints `pynvml OK` and `cage_stats import OK`.

**2. Validate GPU fit + live infra BEFORE the long sweep (the one real unknown).**
- [ ] Start the server once and confirm the model loads: `./scripts/manage_vllm_server.sh restart Qwen/Qwen3-8B`.
- [ ] **Gate 2 live-infra preflight (mandatory before EVERY sweep):** `bash scripts/preflight_check.sh Qwen/Qwen3-8B` must print `PREFLIGHT PASS` — it checks vLLM `/health` + model, LettuceDetect + NLI scoring a real pair, cage-stats import, FAISS + the retrieval embedding model, and that no mock/disable env var is set. Do NOT launch on any `[FAIL]`.
- [ ] (Optional, validates the staleness arm) `bash scripts/smoke_staleness.sh Qwen/Qwen3-8B` before a full staleness sweep.
- [ ] On a 24 GB L4, **Qwen3-8B + FP8 + speculative can OOM at load.** If it does:
  - lower `--gpu-memory-utilization` (terraform default is now `0.85`; try `0.80`), or
  - use **Qwen3-4B** for the FP8 / speculative arms (`MODEL=Qwen/Qwen3-4B`).

**3. Run the core suite (7 baselines, telemetry, continuous GCS sync).**
- [ ] `nohup bash scripts/cloud_run.sh Qwen/Qwen3-8B 500 3 > run.log 2>&1 &`
- [ ] **Repeat the core suite for MiMo** (produces the `_mimo7b` core arms including `no_cache_mimo7b`, which the within-MiMo stats pass requires): after the Qwen core finishes, `nohup bash scripts/cloud_run.sh XiaomiMiMo/MiMo-7B-RL 500 3 > run_mimo.log 2>&1 &`. The per-baseline model-scoped clean in `run_phase1.sh` means this does NOT wipe Qwen's arms.
- [ ] Confirm `analysis/phase1/...` fills with per-baseline `metrics.json` + `vllm_telemetry.json`, and that results mirror to `gs://<project>-cage-results`.

**4. FP8 × prefix-cache gate (guards the compression confound).**
- [ ] `bash scripts/check_fp8_prefix_cache.sh` must print **PASS** (`cached_tokens > 0` under FP8). A FAIL means `compressed_cag` is "no-reuse + compression" and the axis is confounded — do not trust those numbers.

**5. Compression 2×2 axis.**
- [ ] `bash scripts/run_compression.sh Qwen/Qwen3-8B` (runs the gate first, then `cag_full`/`rag_full`/`compressed_rag`/`compressed_cag`). Confirm `compression_analytical` and the empirical GPU footprint are both present.
- [ ] **Repeat compression for MiMo:** `bash scripts/run_compression.sh XiaomiMiMo/MiMo-7B-RL` (produces the `_mimo7b` compression cells). Confirm MiMo-7B + fp8 KV fits the 24 GB L4 during the Step-2 smoke first.

**6. Speculative decoding (per model).**
- [ ] `bash scripts/run_speculative_matrix.sh Qwen/Qwen3-8B` and `bash scripts/run_speculative_matrix.sh XiaomiMiMo/MiMo-7B-RL`. Each runs the ngram + native-draft (eagle3 for Qwen, mimo_mtp for MiMo) 2×2. A pre-flight gate (`check_mtp_spec_decode.sh`) skips the native-draft cells and writes `STATUS=failed` if speculation does not actually engage, so a silent no-op cannot masquerade as a completed cell. Confirm `vllm_telemetry.json` has a non-null `spec_decode` acceptance for the cells that ran.

**7. Collect + analyze.**
- [ ] `bash scripts/sync_results_to_gcs.sh analysis` (final flush).
- [ ] `bash scripts/run_phase2_stats.sh` — consolidates `analysis/phase1` (core) + `analysis/compression` + `analysis/speculative_matrix`, runs the per-query Wilcoxon + Holm + bootstrap vs `no_cache`, excludes cross-model MiMo arms from the Qwen reference, and writes `spec_acceptance_summary.csv`. Do NOT call `statistical_tests.py` bare — it needs the consolidated dir and `--reference no_cache`.

**8. Teardown (bounds cost).**
- [ ] Single-VM (console/gcloud) path: `bash scripts/teardown_vm.sh cage-gpu us-central1-a` — it does a final sync, verifies THIS run's log sentinel is in GCS, and only then deletes the VM (fail-closed). Never pass `--force`.
- [ ] Terraform cluster path (Phase 3 only): `terraform -chdir=terraform/gcp destroy`. Either way the versioned results bucket (`force_destroy=false`) is retained.

## Definition of Done

Phase 2 is complete when **all** of the following hold:

- [ ] The headline result sets exist for **both models** (Qwen3-8B and MiMo-7B-RL): per model, 6 core (`no_cache`/`rag`/`redis`/`prefix_cache`/`hybrid` cold+warm) + 4 compression (`cag_full`/`rag_full`/`compressed_rag`/`compressed_cag`) + 4 speculative ≈ **28 total**. MiMo arms carry the `_mimo7b` label tag and are analyzed **within-model** (MiMo arms vs `no_cache_mimo7b`, never vs the Qwen reference). `distributed` is deferred to Phase 3.
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
