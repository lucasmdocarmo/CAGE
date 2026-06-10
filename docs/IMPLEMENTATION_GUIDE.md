> <!-- CAGE-DOC-STATUS -->
> **⚠️ STATUS: PARTIALLY STALE (2026-06-09).** The local setup here uses the old
> vLLM source-build path and old repo name. For current setup/deploy/run see
> [`RUNBOOK.md`](RUNBOOK.md).

# CAGE Framework — Implementation Guide

**Last Updated:** 2026-04-08

Step-by-step setup for running CAGE experiments locally or on GPU.

---

## Prerequisites

- macOS with Apple Silicon (or Linux with CUDA GPUs)
- Python 3.12+
- 16 GB+ RAM (24 GB+ recommended)
- 20 GB+ free disk space

---

## 1. Environment Setup

```bash
# Create isolated environment
conda create -n cage-vllm python=3.12 -y
conda activate cage-vllm

# Install project dependencies
cd /path/to/cag-llm-kvcache
pip install -r requirements.txt
```

## 2. Build vLLM (macOS CPU)

```bash
cd ~/projects
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
pip install -e .
```

Verify: `python -c "from vllm import LLM; print('OK')"`

## 3. Configure Environment

```bash
# Add to ~/.zshrc
export VLLM_CPU_KVCACHE_SPACE=10     # 10 GB KV cache (adjust for RAM)
export VLLM_CPU_OMP_THREADS_BIND=auto

source ~/.zshrc
```

## 4. Download Datasets

```bash
python scripts/download_datasets.py
# Downloads SQuAD v2, HotpotQA, TriviaQA to ~/.cache/huggingface/datasets/
```

## 5. Start vLLM Server

```bash
# CPU mode
vllm serve Qwen/Qwen3-4B --port 8000 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details

# GPU mode (on GPU machine)
vllm serve Qwen/Qwen3-4B --port 8000 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --gpu-memory-utilization 0.9
```

Verify: `curl http://localhost:8000/health`

## 6. Run First Experiment

```bash
python scripts/run_experiment.py \
  --baseline no_cache \
  --model Qwen/Qwen3-4B \
  --num-trials 1 \
  --num-queries 10

python scripts/run_experiment.py \
  --baseline prefix_cache \
  --model Qwen/Qwen3-4B \
  --num-trials 1 \
  --num-queries 10
```

## 7. Run Full Phase 1

```bash
# All single-instance baselines, 3 trials, 50 queries each
for baseline in no_cache prefix_cache rag redis hybrid; do
  python scripts/run_experiment.py --baseline $baseline --model Qwen/Qwen3-4B --num-trials 3 --num-queries 50
done

# Distributed (requires multi-replica setup first)
python scripts/manage_vllm_cluster.py start --model Qwen/Qwen3-4B --replicas 3
python scripts/run_experiment.py --baseline distributed --model Qwen/Qwen3-4B --num-trials 3 --num-queries 50
```

## 8. Generate Plots

```bash
python scripts/generate_publication_plots.py \
  --results-dir analysis/phase1/results \
  --output-dir analysis/phase1/images
```

## 9. Run Tests

```bash
pytest tests/ -v
```

---

## Troubleshooting

**vLLM build fails on macOS:**
- Ensure XCode Command Line Tools: `xcode-select --install`
- If headers missing: `sudo rm -rf /Library/Developer/CommandLineTools && xcode-select --install`

**Out of memory with multi-replica:**
- Reduce `VLLM_CPU_KVCACHE_SPACE` to 2–3 GB per replica
- Use smaller model for testing (3 × Qwen3-4B needs ~24 GB)

**Slow inference on CPU:**
- Expected. Phase 1 CPU latencies are 10–27 s per request.
- GPU will be 10–100× faster.
