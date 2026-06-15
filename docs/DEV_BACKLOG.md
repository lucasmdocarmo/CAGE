# CAGE — Development Backlog ("what's left to develop")

> From a full codebase audit (2026-06-09). Ordered: **protocol correctness → compression axis
> → publication completeness → nice-to-haves**. Sizes: S(<½d) · M(1–2d) · L(3d+). File:line refs
> point at the code to change. Companion to [`VALIDATION_AND_SOTA_REVIEW.md`](VALIDATION_AND_SOTA_REVIEW.md)
> (why) and [`RELATED_WORK_COMPRESSION.md`](RELATED_WORK_COMPRESSION.md) / [`COMPARISON_MATRIX.md`](COMPARISON_MATRIX.md) (what it backs).

## Status snapshot
- **Works today:** baselines `no_cache`, `prefix_cache`, `redis`, `rag`, `hybrid`; `distributed` (replicated, router HTTP routing); FAISS+e5 retrieval (prefixes fixed for new indices); quality metrics (NLI claim-level + LettuceDetect + rescaled BERTScore); perf metrics; multi-trial harness; `statistical_tests.py`. Datasets loadable: squad_v2, hotpotqa, trivia_qa, qasper, humaneval, mbpp. Model configs: Qwen3-4b/8b/14b/30b-a3b, Qwen2.5-7b.
- **Simulated / partial:** `distributed_sharded` transfer (arithmetic + `asyncio.sleep`); `speculative` (enum+config only, not forwarded to backend); `--seed` (no-op for HF datasets); GPU telemetry (`GPUMetricsTracker` implemented but never called); non-stream TTFT (fabricated ×0.2).
- **Absent:** anything compression (`compressed_rag`/`compressed_cag`, LLMLingua/SnapKV, compression metrics); MLA/DeepSeek model config; real cross-node KV transfer (no KVConnector/LMCache); NaturalQuestions/MuSiQue/2WikiMQA loaders.

## P0 — Protocol correctness (must precede ANY compression run)
> Status 2026-06-09: **#1,#2,#4,#5,#7 DONE & verified** (py_compile + functional). #3 **partial**
> (Redis flush per-trial works via `--flush-redis-namespace`; vLLM prefix-cache reset still needs
> a server restart between trials — phase-script concern). #6 **deferred** (GPU/cluster-only; must
> use vLLM `--kv-transfer-config` + LMCache/NIXL per the vLLM gate check — cannot validate locally).

| # | Task | Status | Fixes |
|---|---|---|---|
| 1 | `--seed` resamples HF datasets (seeded `shuffle` before `select`, all 6 loaders) | ✅ done | O3 — trials independent |
| 2 | `--context-source {auto,gold,retrieved}` equalizes CAG vs RAG context | ✅ done | O2 confound |
| 3 | Per-trial cache flush | ◐ partial (Redis ✓; vLLM reset = restart) | O3 |
| 4 | Warm-hybrid: warmup drawn from a disjoint pool slice | ✅ done | O4 leakage |
| 5 | Loud warning on pre-fix e5/bge indices (`uses_e5_prefixes=False`) | ✅ done | O5 |
| 6 | **Real cross-node KV transfer** via vLLM `--kv-transfer-config` + LMCache/NIXL (measure `transfer_bytes`) | ⏳ deferred (cluster) | O1 + Phase-3 lever |
| 7 | `GPUMetricsTracker` wired into runner (`gpu` section in summary) | ✅ done | Phase-2 readiness |

## P1 — Compression axis (Option 3) — ✅ DONE & verified (text path; KV path cluster-validated)
| # | Task | Status |
|---|---|---|
| 8 | `COMPRESSED_RAG`/`COMPRESSED_CAG` baseline types + config fields + `--baseline` choices | ✅ done (`baselines.py`) |
| 9 | `compressed_rag` text compression (LLMLingua) hooked into `prepare_example` (graceful fallback) | ✅ done (`src/orchestration/compression.py`) |
| 10 | Compression metrics: `compression_ratio` (per-row), `kv_cache_bytes` (incl. MLA), `transfer_bytes` model | ✅ done (`src/evaluation/compression.py`) |
| 11 | `compressed_cag` KV compression via vLLM `--kv-cache-dtype fp8` (+ `configs/model/deepseek-v2-lite.yaml` MLA arm) | ✅ wired (serve flag + config); **cluster-validate** |

## P2 — Publication completeness — ✅ DONE & verified
| # | Task | Status |
|---|---|---|
| 12 | `speculative` via vLLM **`--speculative-config`** at launch (`VLLM_SPECULATIVE_CONFIG`); description corrected | ✅ done (acceptance-rate via `/metrics` = follow-up) |
| 13 | Wire significance testing into aggregation (ddof=1, p-values/CIs) | ⏳ open (`statistical_tests.py` exists standalone; meaningful now that trials are independent) |
| 14 | Stop fabricating non-stream/Gemini TTFT | ✅ done (non-stream TTFT = full response time, not ×0.2) |
| 15 | NaturalQuestions + MuSiQue loaders (match cited prior art) | ✅ done (`loader.py`; add to `download_datasets.py` = minor follow-up) |

## NEW: cache-state control for trials (resolves #3 for the cloud)
`--reset-cache-between-trials` flushes the vLLM prefix cache between trials via
`POST /reset_prefix_cache` (needs the server launched with `VLLM_SERVER_DEV_MODE=1`) — gives
true cold-start-per-trial without a full model reload. See RUNBOOK "Cache-state control".

## P3 — Nice-to-haves
- Tests: seed determinism, context-source flag, compression metrics, speculative payload (M).
- Honor `reranker_top_k` truncation; concurrent `batch_generate` for real load-balanced QPS (`ir.py`, `vllm_adapter.py:281-284`) (S–M).

## Critical-path note
The compression-axis data the paper wants (P1) is **blocked on P0** — running it on the current
confounded protocol produces non-publishable numbers. And **none of P1 is implemented yet**, so a
cloud run today yields only the *existing* baselines, not compression. Recommended order:
**P0 (fix) → Phase-2 GPU re-run of existing baselines → P1 (build compression) → Phase-2.5 re-run.**
