# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Current repository state

This repository contains a working implementation of the CAGE framework core components:

### Implemented
- ✅ `src/data/loader.py` - Dataset loaders for HotpotQA, QASPER, SQuAD v2, TriviaQA
- ✅ `src/inference/engine.py` + `vllm_adapter.py` - Inference abstraction with vLLM integration
- ✅ `src/evaluation/quality.py` - Quality metrics (faithfulness, relevance, completeness)
- ✅ `src/evaluation/performance.py` - Performance metrics (throughput, TTFT, latency, resources)
- ✅ `src/orchestration/baselines.py` - Baseline configurations (no-cache, prefix-cache, rag, redis, hybrid, distributed, speculative)
- ✅ `src/orchestration/ir.py` - Local IR (SentenceTransformers + FAISS) with on-disk persistence
- ✅ `src/orchestration/redis_cache.py` - Redis cache baseline for caching retrieval artifacts
- ✅ `src/orchestration/router.py` - FastAPI router with prefix-aware hashing + streaming passthrough
- ✅ `src/utils/prompting.py` - Shared prompt formatting (consistent across baselines)
- ✅ `scripts/download_datasets.py` - Dataset download script
- ✅ `scripts/run_experiment.py` - Main experiment runner (supports RAG/Redis/Hybrid)
- ✅ `tests/` - Basic test suite with pytest
- ✅ `configs/` - Hydra config files for models, datasets, experiments

### Partially implemented / future work
- ⚠️ `experiments/notebooks/analysis.ipynb` - Analysis notebook (not committed)
- ⚠️ `terraform/` - Infrastructure configs (scaffolding only)
- ⚠️ `src/monitoring/` - Telemetry beyond Prometheus (e.g., tracing/OpenTelemetry)
- ⚠️ Deeper KV telemetry (remote KV transfer / disagg-prefill internals) may require patching vLLM

Source-of-truth references:

- `README.md` (intended architecture + example commands)
- `docs/IMPLEMENTATION_GUIDE.md` (detailed step-by-step setup)
- `requirements.txt` (Python dependencies)

## Intended architecture (from docs)

High-level data flow (as described in `README.md`):

Workload Generator → Orchestrator/Router → vLLM replicas → Monitoring → Analysis
                                           ↘ Quality evaluator

When implementation lands, responsibilities are intended to map roughly to:

- `src/data/`: dataset loading / prompt formatting
- `src/inference/`: vLLM adapter / inference abstraction
- `src/orchestration/`: workload generation, routing, baselines
- `src/evaluation/`: quality + performance metrics
- `src/monitoring/`: telemetry/metrics hooks
- `src/utils/`: config/logging/helpers

## Commands (verified in this repo)

## Telemetry (Prometheus)
Runner (`scripts/run_experiment.py`) exposes metrics on `PROM_PORT` (default: 9400):
- `cage_runner_ttft_seconds`, `cage_runner_latency_seconds`, `cage_runner_tokens`
- `cage_runner_prompt_tokens`, `cage_runner_cached_prompt_tokens`, `cage_runner_cached_prompt_ratio`
- `cage_runner_cached_prompt_requests_total`

Router (`src/orchestration/router.py`) exposes metrics at `GET /metrics` and proxies streaming SSE.
It adds `x-router-replica` to both streaming and non-streaming responses.

To enable prompt-cache telemetry (cached prompt tokens) from vLLM:
- Start vLLM with `--enable-prompt-tokens-details`.
- Use streaming requests with `stream_options: {"include_usage": true}` (VLLMAdapter sets this when `stream=True`).

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Download datasets:

```bash
python3 scripts/download_datasets.py
# Or download specific dataset:
python3 scripts/download_datasets.py --dataset squad_v2
```

Run experiments:

```bash
# CAG baselines (require starting vLLM with/without --enable-prefix-caching)
python3 scripts/run_experiment.py --baseline no_cache --model Qwen/Qwen3-4B --dataset squad_v2 --num-queries 10
python3 scripts/run_experiment.py --baseline prefix_cache --model Qwen/Qwen3-4B --dataset squad_v2 --num-queries 10

# RAG baseline (builds/loads a FAISS index under ./experiments/ir_index)
python3 scripts/run_experiment.py --baseline rag --model Qwen/Qwen3-4B --dataset squad_v2 --num-queries 10 --top-k 3

# Redis baseline (caches retrieval results in Redis; use repeat-queries to see cache hit rate)
python3 scripts/run_experiment.py --baseline redis --model Qwen/Qwen3-4B --dataset squad_v2 --num-queries 10 --repeat-queries 2

# Hybrid baseline (first pass = RAG, later repeats = CAG-style; use repeat-queries)
python3 scripts/run_experiment.py --baseline hybrid --model Qwen/Qwen3-4B --dataset squad_v2 --num-queries 10 --repeat-queries 2

# Use router for local distributed simulation: point api-base to router (http://localhost:9000)
# (start 3 vLLM replicas on 8001-8003 and the router on 9000)
python3 scripts/run_experiment.py --baseline distributed --model Qwen/Qwen3-4B --dataset squad_v2 --num-queries 10 --api-base http://localhost:9000

# Speculative decoding baseline (ngram method - no draft model needed)
# First start vLLM with speculative config:
# vllm serve Qwen/Qwen3-4B --port 8000 --enable-prefix-caching --speculative_config '{"method": "ngram", "num_speculative_tokens": 5}'
python3 scripts/run_experiment.py --baseline speculative --model Qwen/Qwen3-4B --dataset squad_v2 --num-queries 10 --speculative-method ngram --num-speculative-tokens 5
```

Run tests:

```bash
# Run all tests
python3 -m pytest

# Run with coverage
python3 -m pytest --cov=src

# Run specific test file
python3 -m pytest tests/test_data.py
```

Start multi-replica router:

```bash
# Start router (expects replicas on ports 8001, 8002, 8003)
python3 -m src.orchestration.router

# Or with uvicorn directly
uvicorn src.orchestration.router:app --host 0.0.0.0 --port 9000
```

There is currently no committed:

- lint/format configuration
- Jupyter analysis notebook at `experiments/notebooks/analysis.ipynb`

## External dependencies (not in this repo)

vLLM must be built from source for CPU inference (see `docs/IMPLEMENTATION_GUIDE.md`):

```bash
cd ~/projects
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
pip install -e .
```

Start vLLM server before running experiments:

```bash
# Single server
vllm serve Qwen/Qwen3-4B --port 8000

# With prefix caching enabled
vllm serve Qwen/Qwen3-4B --port 8000 --enable-prefix-caching

# For prompt-cache telemetry (cached_tokens), also enable prompt token details
vllm serve Qwen/Qwen3-4B --port 8000 --enable-prefix-caching --enable-prompt-tokens-details

# With speculative decoding (ngram method)
vllm serve Qwen/Qwen3-4B --port 8000 --enable-prefix-caching --speculative_config '{"method": "ngram", "num_speculative_tokens": 5}'

# With draft model speculative decoding (requires compatible model pair)
# vllm serve Qwen/Qwen3-4B --port 8000 --speculative_config '{"model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 5}'
```

### GPU / Omni / Diffusion notes
- GPU Compose: use `docker-compose.gpu.yml` (CUDA image, gpus: all) and switch models (e.g., `mistralai/Mistral-7B-Instruct` or better).
- K8s: add `nvidia.com/gpu` limits and set `VLLM_TENSOR_PARALLEL_SIZE`, `VLLM_GPU_MEMORY_UTILIZATION`.
- Omni/diffusion: swap image to `ghcr.io/vllm-project/vllm-omni:latest` and pick omni-capable models (e.g., Qwen-Omni). Multimodal requests need an adapter extension (not yet implemented here).
