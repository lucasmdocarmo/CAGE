# CAGE Technical Architecture

**Last updated:** 2026-06-28 · **Status:** CURRENT (reflects Phase 1 + Phase 2 code and results)

> **Purpose:** deep technical reference for how the codebase works, how results are generated,
> how vLLM is integrated, and how every component connects. Written for an AI or developer
> picking up the project cold. Higher-level overview: [`SOLUTION_DESCRIPTION.md`](SOLUTION_DESCRIPTION.md).
> Commands: [`RUNBOOK.md`](RUNBOOK.md). Current metrics/status: [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md).
> Portuguese version: [`TECHNICAL_ARCHITECTURE.pt-BR.md`](TECHNICAL_ARCHITECTURE.pt-BR.md).

---

## How an experiment runs end-to-end

A single baseline is run with (real, current flags):

```bash
python scripts/run_experiment.py \
  --baseline prefix_cache --baseline-label cag_full \
  --model Qwen/Qwen3-8B --dataset squad_v2 \
  --num-queries 100 --num-trials 1 --seed 42 \
  --context-source auto --vllm-telemetry \
  --output-dir analysis/phase1/results/prefix_cache
```

> Note: the CLI is `--baseline`, `--num-trials`, `--num-queries`, `--context-source`
> (`auto|gold|retrieved`). The old `--phase`/`--all-baselines`/`--trials`/`--queries` flags do
> not exist. The vLLM server is started separately via `scripts/manage_vllm_server.sh`.

```
1. CLI parsed -> baseline config from src/orchestration/baselines.py
2. Dataset loaded -> src/data/loader.py -> CAGExample objects
3. IR index built if retrieval-backed OR --context-source retrieved -> src/orchestration/ir.py (FAISS)
4. Compressor created if baseline_config.compress_method set -> src/orchestration/compression.py
5. vLLM telemetry sampler started (if --vllm-telemetry) -> src/monitoring/vllm_telemetry.py
6. For each trial, for each query:
   a. Context selected: gold (example.context) OR retrieved (ir_index.search) per --context-source
   b. Optional prompt compression (compressed_rag) -> ContextCompressor.compress()
   c. Prompt built -> src/utils/prompting.py -> format_qa_prompt()
   d. Request sent to vLLM (streaming) -> src/inference/vllm_adapter.py -> TTFT from first SSE chunk
   e. Response + usage telemetry (prompt_tokens, cached_prompt_tokens) collected
   f. Quality scored -> src/evaluation/quality.py -> grounding (primary), faithfulness, F1/EM, ROUGE-L
   g. Per-query row appended (serving + quality + retrieval + compression fields)
7. Telemetry sampler stopped -> aggregate() folded in
8. Aggregates computed (mean_or_none skips None) + per-query CSV + metrics.json written
9. Results synced to GCS (cloud_run.sh / sync_results_to_gcs.sh)
```

### Output structure (per baseline)
```
analysis/<phase>/results/<baseline>/
├── results.csv                              # canonical: one row per query (serving + quality + retrieval + compression)
├── metrics.json                             # canonical: aggregate metrics + run metadata + telemetry
├── vllm_telemetry.json                      # GPU/KV/serving telemetry snapshot + aggregates
├── commands.log                             # the exact invocation line(s)
└── <baseline>_<dataset>_<timestamp>_results.csv / _metrics.json   # raw per-run copies (provenance)
```
The unsuffixed `results.csv` / `metrics.json` are authoritative (the latest valid run); timestamped
copies are kept for provenance. Phase-2 data lives locally in `phase2_archive/` (GCS bucket deleted).

---

## Source code modules

### `src/data/loader.py` - dataset loading
Loads HuggingFace datasets into `CAGExample` objects:
```python
@dataclass
class CAGExample:
    id: str
    question: str
    context: List[str]   # gold passages
    answer: str          # reference answer
    metadata: Dict[str, Any]
```
Loaders: SQuAD v2, HotpotQA, TriviaQA, Natural Questions, MuSiQue (+ code datasets for `code_evaluator`).
Gold-context baselines use `example.context`; retrieval baselines (and `--context-source retrieved`)
ignore it and pull passages from the IR index.

### `src/inference/engine.py` - abstract inference interface
Defines `InferenceEngine` and the `InferenceRequest`/`InferenceResponse` dataclasses. `InferenceResponse`
carries `generated_text`, `ttft_ms`, `total_time_ms`, `num_tokens`, `finish_reason`, `prompt_tokens`,
`cached_prompt_tokens`, and `kv_transfer_params`. `prompt_tokens`/`cached_prompt_tokens` are populated
only when vLLM returns usage telemetry (`--enable-prompt-tokens-details`).

### `src/inference/vllm_adapter.py` - vLLM HTTP client (+ `gemini_adapter.py`, `ollama_adapter.py`)
Talks to vLLM via the OpenAI-compatible `/v1/completions` API.
- **TTFT** is measured in streaming mode: `perf_counter()` at send vs at the first SSE `data:` chunk.
- **Usage telemetry** comes from the final chunk when `stream_options={"include_usage": true}` (gives `prompt_tokens` and `prompt_tokens_details.cached_tokens`, the prompt-cached ratio).
- **Generation parameters (validity fix):** `temperature=0.0` + `stop=["\n"]` + `max_tokens=100`. Greedy decoding makes runs comparable; the `stop=["\n"]` is required because Qwen3-8B otherwise appends chain-of-thought after a single newline.
- For the distributed baseline the adapter targets the router (`:9000`) instead of a single server (`:8000`).

### `src/orchestration/baselines.py` - baseline configuration
`BaselineType` enum and `get_baseline_config(name)` returning a `BaselineConfig` (API base, prefix-cache
flag, IR config, Redis config, `compress_method`, `compress_target_ratio`, `top_k_retrieval`, embedding/
reranker models). What changes between baselines is the context source, whether Redis is consulted,
whether compression/speculation is on, and which endpoint receives the request, not the application code.
The `compressed_rag` config sets `compress_method="llmlingua2"`, `compress_target_ratio=0.5`.

### `src/orchestration/ir.py` - information retrieval
- `build_corpus_from_contexts()` dedupes all dataset passages by `stable_text_id()` (SHA1 of the text).
- `FaissIRIndex` embeds passages with `intfloat/e5-large-v2` into an `IndexFlatIP` (inner product), persisted to `experiments/ir_index/` for reuse.
- Retrieval: embed the query, FAISS top-k (k=3), optional `BAAI/bge-reranker-large` CrossEncoder rerank.
- **`retrieval_hit_rate()` (fixed):** primary check is exact `doc_id` match; a normalized-text fallback now also counts a hit when the gold passage text is among the retrieved texts. This fixed a Phase-2 bug where the metric read a false 0.0 for every row (a doc-id hash divergence) even when top-1 similarity was ~0.99. Report `retrieval_top1_score`, not the raw hit flag, for retrieval quality.

### `src/orchestration/compression.py` - context/prompt compression
`ContextCompressor` (method `llmlingua2`, lazy-loads `llmlingua.PromptCompressor`). `compress()` returns
`(compressed_docs, CompressionStats)` with `compression_ratio`, `compression_applied`, token counts.
**Strict mode (validity fix):** with `CAGE_REQUIRE_COMPRESSION=1`, a missing/failed compressor RAISES
instead of silently passing through. In Phase 2 `llmlingua` was not installed, so this arm silently
no-opped (ratio 1.0); `llmlingua` is now in `requirements.txt` and the compression scripts pre-flight
`import llmlingua` and run strict.

### `src/evaluation/quality.py` - quality metrics
- **Grounding (LettuceDetect, PRIMARY):** ModernBERT token/span hallucination detector; `grounding_score = 1 − hallucinated_span_ratio`. Disable with `CAGE_DISABLE_LETTUCEDETECT=1` (falls back to NLI).
- **Faithfulness (secondary):** answer split into claims; each checked for NLI entailment against the context (DeBERTa-mnli / BART-mnli fallback); score = fraction entailed.
- **F1 / Exact Match, ROUGE-L, context relevance** (relevance is diagnostic, embedding cosine).
- **`evaluate_completeness()` (fixed):** returns `None` (not a sentinel) when the reference is empty, so SQuAD-v2 unanswerable items are excluded from the aggregate. **BERTScore is deprecated** (non-discriminative).
- Aggregation uses `mean_or_none()` which skips `None`, so undefined per-row metrics never pollute the mean.

### `src/evaluation/performance.py` - performance metrics
`PerformanceEvaluator` aggregates per-request timing into QPS, tokens/sec, and avg/p50/p95/p99 for TTFT,
TPOT (`(total_time_ms − ttft_ms)/(num_tokens − 1)`), and end-to-end latency. `CacheMetricsTracker`
computes the prompt-cached ratio from `prompt_tokens`/`cached_prompt_tokens`.

### `src/evaluation/compression.py` - analytical KV footprint
`analytical_kv_footprint()` bridges model architecture to a KV-cache byte estimate so FP8 (compressed_cag)
can be compared against a full-precision baseline analytically (the compression-axis KV model).

### `src/monitoring/vllm_telemetry.py` - serving/GPU telemetry sampler
`VllmTelemetrySampler` is a threaded poller of cage-stats `capture_snapshot()` that runs for the duration
of a baseline. `.aggregate()` returns peak/avg gauges (GPU util, memory, power, temperature, KV-cache
utilization, prefix-hit) plus final counters, written to `vllm_telemetry.json`. Enabled with `--vllm-telemetry`.

### `src/orchestration/router.py` - distributed router (FastAPI)
Receives `/v1/completions`, hashes the prompt prefix (`sha256(prompt[:prefix_length])`), maps it to a
replica (`hash % num_replicas`), forwards the request (blocking or streamed), and reports the serving
replica. Prefix affinity keeps a replica's prefix cache warm; a cold miss on the wrong replica is what
produced Phase-1's heavy distributed tail.

### `src/orchestration/redis_cache.py` - Redis retrieval cache
Caches retrieval results (query -> doc ids), NOT KV tensors. `get/set` keyed by SHA1 of the query.
Flushed before cold baselines; warm baselines pre-populate it.

### `src/orchestration/cache_manager.py` - KV-cache distribution policies
Defines the policy interface: `REPLICATED` (used now), `SHARDED_TENSOR`, `SHARDED_CONTEXT`, `OFFLOAD_CPU`,
`OFFLOAD_NVME`. `SimulatedKVCacheManager` derives cross-node transfer bytes/latency analytically from the
cache footprint and interconnect bandwidth (no tensors move). **Phase 3 replaces this with a real vLLM
KV connector (LMCache/NIXL) under a sharded policy.** Until then, all transfer numbers are simulated.

### `src/utils/prompting.py` - prompt templates
`format_qa_prompt(question, contexts, system_prefix=...)`. The system prefix is identical across requests
(so prefix caching can reuse its KV) and now instructs a SHORT, direct answer to suppress reasoning leakage.

---

## vLLM integration

### `scripts/manage_vllm_server.sh` - the server lifecycle
Starts/stops/restarts a single vLLM server (paths anchored to the repo root; logs to `logs/vllm/`).
The argv is built as a **bash array** so values with whitespace (the JSON speculative config) are never
word-split. Levers (all env-driven):
- `--enable-prefix-caching` (or `--no-enable-prefix-caching`).
- `--kv-cache-dtype fp8` via `VLLM_KV_CACHE_DTYPE` (compressed_cag).
- `--speculative-config '<json>'` via `VLLM_SPECULATIVE_CONFIG` (current API; the old `--speculative-model` is deprecated).
- `--max-model-len ${VLLM_MAX_MODEL_LEN:-8192}` and `--gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION:-0.92}`.
- `--enforce-eager` via `VLLM_ENFORCE_EAGER=1` (skips torch.compile/CUDA-graph; faster, reliable startup on the L4).
- `--enable-prompt-tokens-details` (required for the cached-token telemetry).

`stop_server()` kills `vllm serve`, the separate `VLLM::EngineCore` worker, and any process still holding
the GPU. This fixes a real leak: vLLM v1 spawns EngineCore as its own process; killing only `vllm serve`
orphaned it, it kept the VRAM, and the next start failed. `get_vllm_pid()` uses `head -n1` so a multi-PID
match cannot break the prefix-cache-mode check.

### Speculative decoding
Configured via `--speculative-config` JSON. Verified methods: ngram, draft_model, eagle, eagle3, medusa,
mlp_speculator, mtp (+ deepseek_mtp/ernie_mtp/mimo_mtp). On a single L4 only **ngram** and **EAGLE-3**
(`AngelSlim/Qwen3-8B_eagle3`, `num_speculative_tokens=5`) are viable; the speculative matrix caps
`--max-model-len 4096` so the EAGLE head fits beside the 8B target. Speculative is output-lossless.

### FP8 x prefix-cache gate
`scripts/check_fp8_prefix_cache.sh` verifies FP8 KV does NOT disable prefix caching before running
compressed_cag (otherwise that arm would be confounded as "no-reuse + compression").

---

## Compression and speculative axes

**Compression 2×2** is produced by `scripts/run_compression.sh` (cag_full, rag_full, compressed_cag,
compressed_rag), gated by the FP8 check and (for compressed_rag) the llmlingua pre-flight. Two mechanisms:
FP8 KV is a server launch lever (does not change prompt tokens); LLMLingua-2 is client-side prompt
compression (reduces prompt tokens). `scripts/rerun_compressed_rag.sh` reruns the RAG arm with forced
retrieval and strict compression.

**Speculative 2×2** is produced by `scripts/run_speculative_matrix.sh`: {ngram, eagle3} × {CAG gold,
RAG retrieved}. It isolates the serving effect (acceptance/TTFT/throughput) of each speculative method
under each context strategy, a cross other frameworks do not measure.

---

## Statistical layer

`scripts/statistical_tests.py` reads per-baseline `results.csv`, runs **per-query Wilcoxon** signed-rank
tests vs a `--reference` baseline (paired by `example_id`; falls back to unpaired Mann-Whitney when shared
ids are insufficient, e.g. hybrid_warm), applies **Holm** correction, and reports **Cliff's delta** and
**bootstrap** CIs. Emits a JSON summary (`--output`) and a paper-ready LaTeX table (`--latex-out`).
Requires scipy. Phase-2 output: `phase2_archive/analysis/all_results/phase2_stats.{json,tex}`.

---

## Telemetry, logging, and safe teardown

- **`scripts/sync_results_to_gcs.sh`** mirrors a local dir to the GCS bucket; optional 3rd arg sets the
  remote subpath (used to namespace logs by host); `-c` checksum compare guards against partial uploads.
- **`scripts/collect_logs.sh`** gathers ALL logs (vLLM server, run stdout, status timeline) plus system
  forensics (nvidia-smi, dmesg OOM/Xid, journalctl, pip freeze, env, docker logs) to `vm_logs/<host>/`,
  then writes a per-run **success sentinel** as the last upload.
- **`scripts/teardown_vm.sh`** runs the collection, **verifies the run's sentinel is in GCS, and refuses
  to delete the VM if it is missing** (fail-closed; `--force` overrides). This exists because Phase-2's
  first teardown lost VM-only logs.
- **`scripts/log_sync_daemon.sh`** / **`scripts/_log_guard.sh`** continuously mirror logs+results during a
  run; **`scripts/gcp_shutdown_hook.sh`** collects on spot-preemption / ACPI shutdown, but ONLY if it is
  explicitly wired at VM creation (`gcloud ... --metadata-from-file shutdown-script=scripts/gcp_shutdown_hook.sh`);
  no setup script installs it automatically. On an **on-demand** L4 torn down via `teardown_vm.sh` it is not
  needed (the EXIT-trap collect + fail-closed teardown cover it); wire it only for **spot** instances.
- **`scripts/cloud_run.sh`** orchestrates the core suite and syncs results + logs every interval and on
  exit (EXIT + SIGTERM traps).

---

## Infrastructure

- **Primary path (Phase 2):** a single GCP `g2-standard-8` + L4 VM; vLLM run directly via
  `manage_vllm_server.sh`; orchestration + telemetry via `cloud_run.sh`; results to a durable GCS bucket.
- **Terraform (`terraform/gcp/`):** provisions the cluster (router + N GPU replicas + Redis + GCS), with
  GVNIC + MTU 8896 for the Phase-3 high-bandwidth interconnect (`num_replicas`, `nic_type`, `network_mtu`,
  `vllm_extra_args` are tfvars).
- **Docker Compose / Kubernetes (`docker/`, `k8s/`):** legacy/local and cluster manifests (router +
  replicas + Redis). Phase 2 used the direct-`vllm serve` path, not compose.

---

## Configuration files
- `configs/experiment/*.yaml` - baseline, num_queries, evaluation flags, output dir.
- `configs/model/*.yaml` - qwen3-4b/8b/14b/30b-a3b, qwen2.5-7b-instruct (name, max_tokens, dtype, hardware).
- `configs/dataset/*.yaml` - squad_v2, hotpotqa, and the additional loaders.

---

## How specific baselines work (current)
- **no_cache:** server without prefix caching; gold context; full prefill every request.
- **prefix_cache / cag_full:** server with prefix caching; gold context; shared prefix KV reused.
- **rag / rag_full:** retrieval (FAISS + rerank); gold not used; full recompute.
- **redis (cold/warm):** RAG with retrieval results cached in Redis.
- **hybrid (cold/warm):** retrieval + prefix caching (+ Redis); warm pre-populates both caches.
- **compressed_cag:** prefix cache + `--kv-cache-dtype fp8` (KV halved; prompt tokens unchanged by design).
- **compressed_rag:** RAG + LLMLingua-2 prompt compression (run strict so it cannot silently no-op).
- **speculative (ngram | eagle3) × (CAG | RAG):** the underlying context strategy plus a draft method; output-lossless, varies serving speed only.
- **distributed:** N replicas behind the prefix-hash router; replicated policy now (transfer = 0); sharded + real KV transfer in Phase 3.

---

## Plot generation and verification
- `scripts/generate_publication_plots.py` / `generate_additional_plots.py` / `generate_compact_figures.py` - figures from `metrics.json` (latency, throughput, Pareto, radar, heatmap, tail latency, ranking).
- `scripts/run_status.py` - per-baseline status (started / running / finished / errors); `scripts/extract_qa_evidence.py` - begin/middle/end Q/A evidence per baseline; `scripts/verify_results.py` - result sanity checks.

## Test suite (`tests/`)
`test_inference.py`, `test_ir.py`, `test_baselines.py`, `test_router_integration.py`,
`test_vllm_integration.py`, `test_data.py`. Run with `pytest tests/ -v` (or `scripts/run_tests.sh`).
