> <!-- CAGE-DOC-STATUS -->
> **⚠️ STATUS: SUPERSEDED / HISTORICAL (2026-06-09).** This document predates the
> June 2026 reorganization and metric fixes. It is kept for history only and
> contains stale paths, pre-fix metric numbers, and/or invalid CLI flags.
> **Authoritative docs:** [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md) (project reference),
> [`RUNBOOK.md`](RUNBOOK.md) (setup/deploy/run), [`VALIDATION_AND_SOTA_REVIEW.md`](VALIDATION_AND_SOTA_REVIEW.md) (limitations).
> See [`README.md`](README.md) for the doc index.

# CAGE Project Continuation Guide

**Last Updated:** 2026-04-08

---

## Current State

- **Phase 1:** COMPLETE — 7 baselines, SQuADv2, Qwen3-4B, CPU, 3 trials × 50 queries
- **Results:** `analysis/phase1/results/*/aggregated_metrics.json`
- **Paper:** `analysis/Articles/main-12page.tex` (SBC format, ~12 pages, all claims verified)
- **Next:** Phase 2 (GPU validation + model scaling)

---

## How to Resume

### 1. Start vLLM Server

```bash
# CPU (local development)
export VLLM_CPU_KVCACHE_SPACE=10
export VLLM_CPU_OMP_THREADS_BIND=auto
vllm serve Qwen/Qwen3-4B --port 8000 --enable-prefix-caching --enable-prompt-tokens-details

# GPU (production)
vllm serve Qwen/Qwen3-4B --port 8000 --enable-prefix-caching --enable-prompt-tokens-details --gpu-memory-utilization 0.9
```

### 2. Verify Server Health

```bash
curl http://localhost:8000/health
```

### 3. Run Experiments

```bash
# Single baseline
python scripts/run_experiment.py --baseline prefix_cache --model Qwen/Qwen3-4B --num-trials 3 --num-queries 50

# Run each baseline individually (there is no --all-baselines flag)
for baseline in no_cache prefix_cache rag redis hybrid distributed; do
  python scripts/run_experiment.py --baseline $baseline --model Qwen/Qwen3-4B --num-trials 3 --num-queries 50
done
```

### 4. Generate Plots

```bash
python scripts/generate_publication_plots.py \
  --results-dir analysis/phase1/results \
  --output-dir analysis/phase1/images

python scripts/generate_compact_figures.py
```

### 5. Verify Results

```bash
python scripts/verify_results.py
```

---

## Phase 2 Execution Plan

### Prerequisites
- GPU VM provisioned (A100 recommended, L4 acceptable)
- `docker-compose.gpu.yml` configured with correct GPU type
- Model weights downloaded (Qwen3-4B, Qwen3-8B, optionally Qwen3-14B)

### Steps
1. Deploy vLLM with GPU using `docker-compose.gpu.yml`
2. Run all 7 baselines with `--trials 10 --queries 100`
3. Run with Qwen3-8B and Qwen3-14B for scaling curves
4. Add second dataset (TriviaQA or Natural Questions)
5. Run `scripts/statistical_tests.py` (to be created) for significance
6. Generate Phase 2 plots
7. Update paper with GPU results and scaling analysis

### Expected Outcomes
- Relative baseline rankings preserved (Prefix Cache > Hybrid Warm > No Cache)
- Absolute latency 10–100× lower than CPU
- Scaling curves showing prefix caching benefit vs model size
- Statistical significance for key comparisons

---

## Phase 3 Execution Plan

### Prerequisites
- GCP account with GPU quota
- Terraform configured (`terraform/gcp/terraform.tfvars`)
- 2–4 GPU VMs provisioned

### Steps
1. Provision multi-node cluster via `terraform/gcp/main.tf`
2. Deploy vLLM replicas with real inter-node networking
3. Implement KV tensor transfer in router (gRPC/NCCL)
4. Run distributed experiments with real transfer measurement
5. Test disaggregated prefilling and speculative decoding
6. Evaluate eviction policies under memory pressure

---

## Stopping Services

```bash
# Check running processes
ps aux | grep -E "vllm|redis|python.*experiment"

# Stop vLLM
pkill -f "vllm serve"

# Stop Redis
redis-cli shutdown

# Stop cluster replicas
python scripts/manage_vllm_cluster.py stop
```

---

## Quick Start for AI Assistant

When resuming, tell the assistant:

> "Resume CAGE project. Phase 1 is complete. Read PROJECT_OVERVIEW.md for full context, then start Phase 2 execution."

The assistant should:
1. Read `PROJECT_OVERVIEW.md` at the repo root
2. Read `TECHNICAL_ARCHITECTURE.md` for module details
3. Check `analysis/phase1/results/` for existing data
4. Follow Phase 2 execution plan above
