# CAGE: Cache-Augmented Generation Evaluation Framework

A comprehensive benchmarking framework for evaluating Cache-Augmented Generation (CAG) systems with distributed KV cache management.

## Overview

CAGE is the first framework dedicated to holistically benchmarking CAG systems, measuring trade-offs between system performance and semantic quality in distributed environments. Unlike RAG evaluation tools, CAGE specifically evaluates how cache distribution policies, network placement, and memory tiering impact both retrieval correctness and generated output quality.

## Key Features

- **Multi-Dimensional Evaluation**: System performance (throughput, latency) + Quality metrics (faithfulness, relevance, completeness)
- **Distributed Testing**: Multi-replica orchestration with prefix-aware routing
- **Multiple Baselines**: No-cache, prefix-cache, Redis, RAG, hybrid CAG↔RAG
- **Cloud-Ready**: Terraform configs for GCP GPU deployment
- **Reproducible**: Docker, pinned dependencies, experiment manifests

## Architecture

```
Workload Generator → CAGE Orchestrator → [vLLM Replicas] → Monitoring → Analysis
                                              ↓
                                       Quality Evaluator
```

## Quick Start

### Prerequisites
- macOS with Apple Silicon (or Linux with CUDA GPUs for full features)
- Python 3.12
- 16GB+ RAM (24GB+ recommended)
- 20GB+ free disk space

### Installation

```bash
# 1. Clone repository
git clone <repo-url>
cd cag-llm-kvcache

# 2. Create environment
conda create -n cage-vllm python=3.12 -y
conda activate cage-vllm

# 3. Install dependencies
pip install -r requirements.txt

# 4. Build vLLM (macOS)
cd ~/projects
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
pip install -e .

# 5. Configure environment
export VLLM_CPU_KVCACHE_SPACE=10
export VLLM_CPU_OMP_THREADS_BIND=auto
```

### Run First Experiment

```bash
# Download dataset
python3 scripts/download_datasets.py

# Run baseline comparison
python3 scripts/run_experiment.py --baseline no_cache --model Qwen/Qwen3-4B
python3 scripts/run_experiment.py --baseline prefix_cache --model Qwen/Qwen3-4B

# Analyze results
jupyter notebook experiments/notebooks/analysis.ipynb
```

## Project Structure

```
cag-llm-kvcache/
├── src/                      # Core framework code
│   ├── data/                 # Dataset loaders
│   ├── inference/            # Inference engines (vLLM adapter)
│   ├── evaluation/           # Quality & performance metrics
│   ├── orchestration/        # Router, workload gen, baselines
│   ├── monitoring/           # Telemetry & metrics collection
│   └── utils/                # Config, logging, helpers
├── configs/                  # Hydra configs (model, dataset, experiment)
├── experiments/              # Jupyter notebooks, results
├── scripts/                  # Setup and execution scripts
├── docker/                   # Dockerfiles (CPU + CUDA)
├── terraform/                # Infrastructure-as-code (GCP)
├── docs/                     # Implementation guides
└── tests/                    # Unit tests
```

## Documentation

- [Implementation Guide](docs/IMPLEMENTATION_GUIDE.md) - Step-by-step setup with feasibility analysis
- [Dataset Guide](docs/datasets.md) - Dataset preparation and formatting
- [API Reference](docs/api.md) - Framework API documentation

## Supported Baselines

1. **No Caching**: Worst-case, full context reprocessing
2. **Single-Node CAG**: Prefix caching on single instance
3. **Centralized Cache**: Redis-backed cache server
4. **Standard RAG**: FAISS vector store + retrieval
5. **Distributed Cache**: Tensor parallelism (GPU only)
6. **Hybrid CAG↔RAG**: Fallback strategy
7. **Speculative Decoding**: Draft model acceleration with KV cache management

## Datasets

Recommended HuggingFace datasets for CAG evaluation:
- `hotpotqa` - Multi-hop reasoning (primary)
- `allenai/qasper` - Scientific papers (HPC-relevant)
- `squad_v2` - Reading comprehension
- `trivia_qa` - Multi-evidence questions

## Metrics

### System Performance
- Throughput (QPS, tokens/sec)
- Time-To-First-Token (TTFT)
- End-to-end latency
- Resource utilization

### Cache-Specific
- Prompt cached token ratio (via vLLM `usage.prompt_tokens_details.cached_tokens` when enabled)
- Local/remote hit ratios
- Inter-node data transfer
- Remote fetch latency

### Quality
- Faithfulness (NLI-based)
- Relevance (embedding similarity)
- Completeness (BERTScore, ROUGE)

### Speculative Decoding
- Acceptance rate (accepted/proposed draft tokens)
- Draft tokens per step
- Rollback overhead
- Speedup ratio vs non-speculative

## Speculative Decoding

CAGE supports evaluating speculative decoding's impact on CAG systems:

```bash
# N-gram speculation (no draft model needed)
vllm serve Qwen/Qwen3-4B --port 8000 --enable-prefix-caching \
  --speculative_config '{"method": "ngram", "num_speculative_tokens": 5}'

python3 scripts/run_experiment.py \
  --model Qwen/Qwen3-4B \
  --baseline speculative \
  --speculative-method ngram \
  --num-speculative-tokens 5

# Draft model speculation (requires compatible model pair)
# vllm serve Qwen/Qwen3-4B --speculative_config '{"model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 5}'
```

Supported methods: `draft_model`, `ngram`, `suffix`, `medusa`, `eagle`

## Cloud Deployment

```bash
# Provision GCP GPU instance
cd terraform/gcp
terraform init
terraform apply

# Deploy vLLM with tensor parallelism
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 4 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details
```

## Requirements

See [requirements.txt](requirements.txt) for complete list:
- torch, transformers, datasets
- sentence-transformers, bert-score, ragas
- mlflow, hydra-core, fastapi
- redis, ray, prometheus-client

## Hardware Requirements

### Local Development (macOS)
- M1/M2/M3/M4 chip
- 16GB+ unified memory
- CPU-only inference

### Production (Cloud)
- NVIDIA A100/H100 GPUs
- 40GB+ VRAM per GPU
- NVLink interconnect recommended

## Contributing

[Contributing guidelines to be added]

## License

[License to be determined]

## Citation

```bibtex
@article{carmo2025cage,
  title={CAGE: A Framework for Holistic Evaluation of Cache-Augmented Generation Models},
  author={Carmo, Lucas Mariano do},
  year={2025},
  institution={Pontifícia Universidade Católica de Minas Gerais}
}
```

## Contact

Lucas Mariano do Carmo - lucas.mariano.carmo@gmail.com

## Acknowledgments

- vLLM team for the inference engine
- HuggingFace for datasets and models
- Research based on [original paper](docs/Artigo-Lucas-Mariano.pdf)
