# CAGE Rewrite Knowledge Pack — Cited, Ground-Truthed, Single-Source

**Purpose.** This is the SOLE input for a documentation author rewriting CAGE's solution / architecture / presentation docs. It merges six verified research briefs (codebase ground-truth, dissertation claims, SCBench, KV-cache-DB baseline, staleness/freshness baseline, and per-doc doc-state analysis) into one canonical pack. Every external claim carries its source URL; every codebase claim carries `file:line`. Where an adversarial Verify step corrected or refuted a claim, the corrected version is used and tagged `[VERIFIED]`, `[CORRECTED]`, or `[REFUTED]`.

**Compiled:** 2026-07-02. **Path conventions:** `RE` = repo `/Users/lucasmariano/CAGE`; `CS` = `/Users/lucasmariano/cage-stats`. All line numbers absolute.

**How to use this pack.** Sections 1–5 are the source of truth. Section 6 is the actionable per-doc checklist. When code and a docstring disagree, the code (Section 1) wins — three stale docstrings are called out explicitly. When a citation is load-bearing, mirror the exact bib key from Section 2 (do NOT "fix" the deliberately-mismatched keys — they are load-bearing for `\cite` compatibility).

---

## Table of Contents
1. [Codebase ground-truth](#1-codebase-ground-truth) — baselines, metrics, telemetry, audit fixes, datasets, phase state (with `file:line`)
2. [Dissertation claims + bib keys to mirror](#2-dissertation-claims--bib-keys-to-mirror)
3. [SCBench: what it is + Phase-3 comparison plan](#3-scbench-what-it-is--a-phase-3-comparison-plan-for-cage)
4. [KV-cache-DB baseline: reality check, survey, go/no-go](#4-kv-cache-db-baseline-reality-check-survey-and-gono-go)
5. [Staleness / freshness baseline: design + citations](#5-stalenessfreshness-baseline-implementable-design--citations)
6. [Per-doc update checklist](#6-per-doc-update-checklist)
7. [Master citation index (all URLs preserved)](#7-master-citation-index-all-urls-preserved)

---

# 1. Codebase Ground-Truth

*Verified against code 2026-07-02. All numbered flags feed directly into the per-doc checklist (§6).*

## 1.1 Baseline taxonomy — 9 families (`RE/src/orchestration/baselines.py`)

`BaselineType` enum (`baselines.py:20-32`) defines **9 families**. The module docstring (`baselines.py:4-11`) is **STALE** — it lists only 7, omitting `compressed_rag`/`compressed_cag`. Trust the enum, not the docstring. Config defaults come from `get_baseline_config` (`baselines.py:119-201`). CLI `--baseline` choices list all 9 (`run_experiment.py:1802`). `check_baseline_requirements` (`baselines.py:221-261`) probes vllm/redis/faiss/gpu.

| Enum | value | Config default | What it REALLY does — the honesty tell |
|---|---|---|---|
| `NO_CACHE` | `no_cache` | `enable_prefix_caching=False` (`120-124`) | Full reprocessing; worst-case control. **Honest.** |
| `PREFIX_CACHE` | `prefix_cache` | `enable_prefix_caching=True` (`126-130`) | Relies on vLLM native prefix cache. Honest, **but** requires the *server* launched with `--enable-prefix-caching`; the CLI does not set it (see §1.7). |
| `REDIS_CACHE` | `redis` | `use_faiss=True`, `enable_prefix_caching=False`, `top_k=3` (`132-140`) | **MISLABEL RISK.** Redis stores **retrieval artifacts** (query→doc-ids), NOT KV blocks. `redis_cache.py:5-8` states this explicitly; `RetrievalCache` (`redis_cache.py:79-110`) keys on `(dataset, embedding_model, top_k, query)` and stores hit lists. It is a RAG run with a doc-id cache in front. **Never describe it as a KV cache.** |
| `RAG` | `rag` | `use_faiss=True`, `top_k=3` (`142-148`) | Real FAISS + SentenceTransformers retrieval (`ir.py`). **Honest.** |
| `DISTRIBUTED_CACHE` | `distributed` | `enable_prefix_caching=True`, metadata flags (`150-159`) | **PARTIALLY SIMULATED.** `replicated` policy = real router fan-out to replicas (`router.py:175-206`), transfer cost forced to 0. `sharded_context` policy = **simulated** KV transfer: bytes/latency computed analytically from `bytes_per_token = hidden_size(2048)*layers(16)*2*2` hard-coded for Llama-3.2-1B (`cache_manager.py:85-88, 122-146`), then `asyncio.sleep()`-ed in the router (`router.py:530-532`). No real KV movement. Honest tell: metadata key `supports_sharded_context_simulation=True` (`155-157`). |
| `HYBRID` | `hybrid` | `enable_prefix_caching=True`, `use_faiss=True`, `cache_threshold=0.8` (`161-167`) | Retrieval-artifact cache + native prefix caching. **`cache_threshold=0.8` is DEAD config** — never read in `run_experiment.py`; there is no cache-vs-RAG confidence gate; hybrid always retrieves. |
| `SPECULATIVE` | `speculative` | `enable_prefix_caching=True`, `num_speculative_tokens=5`, `speculative_method="draft_model"` (`169-183`) | **SCAFFOLD / record-only.** No per-request spec-decode wiring. Speculation is a **vLLM launch-time** setting (`VLLM_SPECULATIVE_CONFIG`); the experiment only measures resulting TTFT/TPOT and scrapes acceptance from `/metrics` (`run_experiment.py:757-765, 1444-1449`). Docstring line 11 ("backend wiring incomplete") matches. |
| `COMPRESSED_RAG` | `compressed_rag` | `use_faiss=True`, `compress_method="llmlingua2"`, `compress_target_ratio=0.5` (`186-194`) | Real text compression of retrieved docs via LLMLingua-2 before prompting (`compression.py`). Honest when the package is present (now strict — see §1.4). |
| `COMPRESSED_CAG` | `compressed_cag` | `enable_prefix_caching=True`, `kv_cache_dtype="fp8"` (`195-201`) | **RECORD-ONLY in the runner.** `kv_cache_dtype` is NOT applied to the server by `setup_inference_engine` (`run_experiment.py:512-595` never passes it); it must be set when *launching* vLLM. CLI help says so verbatim: "record-only here" (`run_experiment.py:1890-1892`). Only an *analytical* fp8-vs-bf16 footprint is computed post-hoc (`run_experiment.py:1708-1717` → `analytical_kv_footprint`; `evaluation/compression.py:80-124`, "never raises"). |

**Docs must say (baseline honesty flags):**
- (a) `baselines.py:4-11` docstring is stale (7 vs 9). Do not cite it.
- (b) `redis` = retrieval-artifact cache, NOT a KV cache.
- (c) `distributed` `sharded_context` transfer is simulated with a hard-coded Llama-3.2-1B tensor shape; `replicated` is real routing with zero transfer cost. Invariant enforced by `validate_distributed_artifacts` (`run_experiment.py:394-478`): replicated ⇒ no positive transfer; sharded ⇒ positive transfer.
- (d) `speculative` and `compressed_cag` are **server-launch-configured**; the runner *records* the flag (and, for cag, an analytical footprint) — it does not enable them. Never claim CAGE "enables fp8 KV compression" or "runs speculative decoding"; it *measures a server launched with them*.
- (e) `hybrid`'s `cache_threshold` is dead config.

## 1.2 Metric hierarchy (`RE/src/evaluation/quality.py` + `performance.py`)

**PRIMARY — LettuceDetect span grounding** (`quality.py:466-508`). `HallucinationDetector(method="transformer", model=KRLabsOrg/lettucedect-base-modernbert-en-v1)` (default `quality.py:164-170`). Predicts unsupported char spans; `hallucinated_span_ratio = flagged_chars/len(answer)`, `grounding_score = 1 - ratio` (`502-504`). Returns all-`None` if detector unavailable (`474-483`). Force-disable via `CAGE_DISABLE_LETTUCEDETECT` (`130-132`).

**SECONDARY — NLI faithfulness** (`quality.py:427-464`). Claim-split (`363-376`) → per-claim entailment = MAX over context docs → mean (RAGAS-style). Entailment index resolved from `id2label`, never hard-coded (`378-395`); premise/hypothesis passed as a proper `{"text","text_pair"}` pair (`400-407`). Default `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, fallback `facebook/bart-large-mnli` (`141-147`). Also emits `supported_claim_ratio` (claims with entailment ≥ 0.5).

**DIAGNOSTIC — F1/EM/ROUGE-L** (`quality.py:606-675` F1/precision/recall/EM; ROUGE-L `596-603`). SQuAD-style normalization. These are the only quality fields defaulting to `0.0` not `None` (dataclass `68-71`).

**NEGATIVE CONTROL — BERTScore** (`quality.py:272-313, 573-594`). `rescale_with_baseline=True` REQUIRED for dynamic range (`281-291`); multi-model fallback chain (`156-163`); auto-disables run-wide on unrecoverable failure (`263-270`). Empty reference → `None` (excluded); empty generation → genuine `0.0` (`561-571`). **Deprecated as a "completeness" metric — it is a control, not a quality metric.**

**RETRIEVER DIAGNOSTIC — context relevance** (`quality.py:510-543`). question↔context max cosine; exposed as `context_relevance` with `relevance` alias (`86-87`). Explicitly NOT an answer-quality metric.

**PERFORMANCE** (`performance.py`): QPS/TPS (`214-215`); TTFT avg/p50/p95/p99 (`220-223`); **TPOT** avg/p50/p95/p99; latency percentiles (`245-248`); CPU/mem (`251-253`). Errors filtered before aggregation (`174-177`). Separate `GPUMetricsTracker` via pynvml (`511+`), `SpeculativeMetricsTracker` (`323+`), `CacheMetricsTracker` (`385+`).

**Generation params (`run_experiment.py:1272-1309`):** `temperature=0.0, top_p=0.95, stop=["\n"]` — **greedy / deterministic.** This corrects any doc still claiming temperature 0.7.

### Verified recent audit fixes (mirror these as "recent fixes")
- **TPOT fix [VERIFIED]** (`performance.py:225-243`): `tpot = (total_time_ms - ttft_ms) / (num_tokens - 1)`; **single-token outputs excluded** (`if req.num_tokens > 1`). Comment `225-235` matches code.
- **Error/empty nulling [VERIFIED]** (`run_experiment.py:1173-1181`): `_empty_gen = (not error) and not generated_text.strip()`; when `error or _empty_gen`, `_quality_row = {k: None for k}` and `code_metrics=None` — they never enter means. Belt-and-suspenders in `mean_or_none` also excludes `error`/`empty_generation` rows (`run_experiment.py:1505-1513`). `empty_generation` flag stored per row (`1220`); run-wide warning if > 5% empty (`1498-1501`).
- **BERTScore run-wide null [VERIFIED]** only when a row with a NON-empty reference returned `None` (model genuinely unavailable), not merely from unanswerable rows (`run_experiment.py:1488-1494`).
- **Warm pairing [VERIFIED]:** the runner now primes warm baselines on the same query set (`warmup_queries`), fixing the Phase-2 unpaired `hybrid_warm` issue. Cold/warm are **runtime labels** derived from `warmup_queries`, not enum members.

## 1.3 Telemetry (`RE/src/monitoring/vllm_telemetry.py` + `CS/cage_stats`)

Resolution order (`vllm_telemetry.py:32-46, 61-81`): in-process `cage_stats.api` → `cage-stats --once --json` CLI → dependency-free stdlib `/metrics` scraper.

- **Mock path REMOVED [VERIFIED].** `capture_snapshot` docstring (`58-60`): "There is no synthetic/mock path… an unavailable server yields None, never fake data." No mock/synthetic branch exists. Confirmed by commit `318d0e8 update - remove mocks, update stats`. Remaining "mock/synthetic" hits in `src` are negations or the HPC-code synthetic dataset. **This is now a hard code guarantee — surface it as an explicit invariant.**
- **Spec-decode exact-name match [VERIFIED]** (`vllm_telemetry.py:266-285`). `_sum()` computes `name = line.split("{")[0].split(" ")[0]` and requires `name == metric` (`274-275`); a prefix match would wrongly fold `_total/_bucket/_sum` siblings (comment `267-270`). Scrapes `vllm:spec_decode_num_accepted_tokens_total`, `vllm:spec_decode_num_draft_tokens_total`, `vllm:spec_decode_num_drafts_total` (`283-285`).
- `VllmTelemetrySampler` (`84-181`) threads polling during the workload; aggregates gauges (peak/avg) / counters (final) / last; canonicalizes `spec_decode_acceptance_rate` (`174-179`). `_LAST` whitelist now includes BOTH flat (`spec_active/spec_acceptance/spec_accepted_per_draft`) and nested (`spec_decode`) schemas (`111-114`) — fixes the Phase-2 "None for every speculative cell" bug.
- **cage-stats raises on no vLLM series [VERIFIED]** (`CS/cage_stats/api.py:69-73`): `if "vllm:" not in (r1.text or ""): raise RuntimeError(...)` — prevents fabricated all-zero snapshots (comment `63-68`). Note: package path is `cage_stats/api.py`, not `src/cage_stats/api.py`.

## 1.4 Compression strictness (`RE/src/orchestration/compression.py`)

- **Strict-by-default [VERIFIED]** (`compression.py:57-94`). `_strict = (not _allow_noop) and (_disabled_reason is None)`, where `_allow_noop = CAGE_ALLOW_NO_COMPRESSION ∈ {1,true,yes}` (`71-72`). When llmlingua import/model load fails and `_strict`, it **RAISES** (`91-92`); also raises if compressor unavailable at compress time (`117-121`) or compression throws (`142-143`). LLMLingua-2 → `microsoft/llmlingua-2-xlm-roberta-large-meetingbank`; `llmlingua` → `NousResearch/Llama-2-7b-hf` (`52-55`). `rate=target_ratio` = fraction to KEEP (`129-134`).
- **Env-var naming is inconsistent in-file [CORRECTED]:** the **live opt-out is `CAGE_ALLOW_NO_COMPRESSION`** (`71`). The RaiseError MESSAGES still say `"CAGE_REQUIRE_COMPRESSION=1 but …"` (`92, 119, 143`) and the module docstring (`9`) references `CAGE_DISABLE_COMPRESSION` for pass-through — three different var names in one file. `CAGE_DISABLE_COMPRESSION` (`62-63`) is a *separate* switch that intentionally disables compression (pass-through; strictness moot). `CAGE_REQUIRE_COMPRESSION` is called back-compat/redundant in the comment (`69-70`) but is **NOT actually read anywhere** (only in error strings + `scripts/rerun_compressed_rag.sh:45`). **Do NOT document `CAGE_REQUIRE_COMPRESSION` as the live control.** Live controls: `CAGE_ALLOW_NO_COMPRESSION` (opt out of strictness) / `CAGE_DISABLE_COMPRESSION` (disable compression entirely).

## 1.5 Datasets (`RE/src/data/loader.py` `get_loader`, `loader.py:802-821`)

Registry — **11 loaders**: `hotpotqa, qasper, squad_v2, trivia_qa, natural_questions, musique, crag, sharegpt, humaneval, mbpp, hpc_code`. (8 QA/serving + 3 code.)
- **crag** (`677-734`): configurable HF path (`CAGE_CRAG_HF_PATH`, default `"crag"`); defensive field mapping (query/question, answer, search_results/contexts). Gold answer + retrieved candidate docs — RAG-fair.
- **sharegpt** (`737-799`): configurable (`CAGE_SHAREGPT_HF_PATH`, default `RyokoAI/ShareGPT52K`); **serving-workload trace, NO gold answer** (`no_gold_answer=True`, `795`); first assistant turn is a similarity-only reference, not extractive gold (`743-745`).
- All QA loaders now **shuffle(seed) BEFORE select** for trial-independence (e.g. `72-74, 118-119, 166-167`). `hpc_code` is a **synthetic** in-file 12-prompt set (`337-592`).
- `CRAGLoader` and `ShareGPTLoader` are wired end-to-end: registry + `run_experiment.py --dataset` choices + `scripts/download_datasets.py --dataset` choices (`45-48`; env overrides `61-62`).
- **STALE docstrings:** `scripts/download_datasets.py:5-11` lists only 6 datasets; `loader.py:4-12` header also stale. Trust the registry, not these comments.

## 1.6 Phase state

- **Phase 1 (CPU/small): DONE.** `run_phase1.sh` → Qwen3-4B, squad_v2, NUM_QUERIES=50, 3 trials. **Used stochastic decoding (temperature 0.7)** — a distinguishing property of Phase 1 only.
- **Phase 2 (single-L4, clean re-run): DONE.** `run_phase2.sh` → Qwen3-8B (arg default line 13), squad_v2, NUM_QUERIES=**500** (line 15), 3 trials. Baselines: `no_cache, rag, redis(cold), prefix_cache, hybrid(cold), hybrid(warm)` (`128-149`). **`distributed` is OFF by default** on a 24 GB L4 (comment line 22; gated block at `153+`). Greedy T=0. **`phase2_archive/` exists.** ⚠️ **Query-count conflict:** the script default is 500; the plan-of-record memo says 300×3. **Flag before quoting a number** — reconcile against the actual run before publishing.
- **Phase 3 (multi-node): DEFERRED / present but not the active target.** `run_phase3.sh` exists (Qwen2.5-7B-Instruct, `ENABLE_DISTRIBUTED=1` default line 21) — the distributed/multi-node arm. Per project memory, deferred. `analysis/phase3` and `phase4` dirs are empty (datasets are an axis, not phases).

## 1.7 Cross-cutting truthfulness flags (apply everywhere)

1. `compressed_cag` fp8 and `speculative` are configured at **vLLM launch**, not by the CAGE runner. The runner records the flag (and, for cag, an analytical footprint).
2. `distributed` `sharded_context` transfer is **simulated** (hard-coded Llama-3.2-1B KV geometry + `asyncio.sleep`); `replicated` is real routing with zero transfer cost.
3. `redis` is a **retrieval-artifact cache**, not a KV cache.
4. `prefix_cache`/`hybrid`/`compressed_cag` need `--enable-prefix-caching` **on the server**; `setup_inference_engine` only validates the server, never launches/configures it.
5. Three stale docstrings to fix or ignore as authority: `baselines.py:4-11` (7 vs 9 families), `download_datasets.py:5-11` and `loader.py:4-12` (dataset lists). Trust the enum/registry.
6. Compression env-var naming is inconsistent in-file: live opt-out is `CAGE_ALLOW_NO_COMPRESSION`; `CAGE_REQUIRE_COMPRESSION` survives only in error strings and is not read.

**Key files:** `RE/src/orchestration/{baselines,router,cache_manager,redis_cache,compression,ir}.py`; `RE/src/evaluation/{quality,performance,compression}.py`; `RE/src/monitoring/vllm_telemetry.py`; `RE/src/data/loader.py`; `RE/scripts/{run_experiment,download_datasets,run_phase1,run_phase2,run_phase3}.{py,sh}`; `CS/cage_stats/api.py`.

---

# 2. Dissertation Claims + Bib Keys to Mirror

*Sources: `RE/my-article/CAGE___Dissertation_Mestrado_2026/TEXT/{1_INTRODUCTION,3_RELATED WORK,4_METHODOLOGY,6_RESULTS}.tex` and `.../Main.bib`. Quotes are verbatim.*

## 2.1 Central Research Question (verbatim, `1_INTRODUCTION.tex:16`)

> "How should a cache-aware LLM serving be evaluated when its KV-cache efficiency (cache-reuse rate, latency, and TTFT), must be analyzed together against response quality metrics (faithfulness, grounding, relevance)? and, as cache grows and it needs to be distributed across nodes under a heavier memory pressure, does that quality uphold, or fall as efficiency is pushed to its limit?"

Title-level framing (`:26`): "the *efficiency* gained by reusing and distributing the KV cache must be tested against the answer *quality* obtainable from information retrieval."

## 2.2 RQ1–RQ4 (verbatim, `1_INTRODUCTION.tex:36-39`)

- **RQ1:** "What metric suite can jointly capture serving efficiency, cache-reuse behavior, and answer faithfulness, so that trade-offs are measured rather than assumed?"
- **RQ2:** "Under controlled conditions, what is the measurable effect of cache reuse on serving performance and on answer faithfulness, relative to retrieval-backed and hybrid baselines where RAG is used as a fallback?"
- **RQ3:** "As serving scales from CPU to GPU and multi-node high-performance-computing (HPC) deployments under heavy KV-cache memory pressure, how does the distributed KV-cache design itself hold up against information-retrieval quality metrics?"
- **RQ4:** "Which telemetry signals are necessary and sufficient to attribute serving behavior to local reuse, retrieval success, or cross-node transfer?"

**RQ-to-phase mapping (verbatim, `:42-50`):** "RQ1, together with the local form of RQ2, is addressed by the controlled CPU validation reported in this dissertation; RQ2 at real serving scale, and the compression question, are addressed by the single-GPU phase; and RQ3 and RQ4 … are posed and operationalised here but answered only once those regimes are realised in the GPU and high-performance-computing phases."

## 2.3 The GAP (verbatim, `1_INTRODUCTION.tex:24-28`, `:94-95`)

> "The gap this work presents comes from the recent necessity to evaluate exactly where a cache-aware serving mechanism is most valuable: at scale."
> "To our knowledge, no existing framework quantifies both sides of this trade-off under a common, reproducible protocol that spans local, single-GPU, and multi-node high-performance-computing (HPC) regimes."

Justification restatement (`:94-95`): "frameworks such as RAGAS, ARES and RAGBench measuring faithfulness and relevance but treating the serving system as a black box… and on the efficiency side, high-performance serving systems such as vLLM, DistServe and Mooncake optimize KV-cache placement and cross-node transfer but never ask whether the answer is still correct… recent work demonstrates that aggressive KV-cache compression can cause a model to ignore parts of its prompt, a quality cost no throughput benchmark reveals currently."

Related-Work positioning (`3_RELATED WORK.tex:166`): "CAGE's difference is therefore structural across the four tables, as is the only entry positive on every capability."

## 2.4 Metric hierarchy as the thesis frames it (support-strength order)

1. **Grounding = PRIMARY** (`4_METHODOLOGY.tex:104-106`): "Grounding is the primary signal, computed by a span-level detector (`LettuceDetect`) as *grounding* = 1 − r, where r is the fraction of answer characters flagged as unsupported by the context."
2. **Faithfulness = SECONDARY / corroborating** (`:110-114`): "a strict, claim-level Natural Language Inference (NLI) metric … a secondary, corroborating check on the primary span-level grounding signal, not the primary measure of support." (Empirically the *most discriminating* metric in `6_RESULTS.tex`: 0.570 gold vs 0.504 RAG, ≈11.6% drop.)
3. **BERTScore = NEGATIVE CONTROL** (`:115-118`): "a token-level soft-F1 similarity … used here deliberately as a negative control." Confirmed `6_RESULTS.tex:32`: "BERTScore was intentionally included not as a primary metric, but as a negative empirical control."
4. **Completeness / ROUGE-L / F1 / EM = standard references** (`:119`): "token-level F1 and exact match are retained as standard references."
5. **Serving metrics (co-measured, not ranked under quality):** Latency, TTFT ("most directly improved by cache reuse"), TPOT, Throughput (QPS + tokens/s) — `4_METHODOLOGY.tex:92-102`. Plus **prompt-cache ratio** = "cached prompt tokens divided by the total prompt tokens" (`:122`).

Joint co-measurement thesis (`1_INTRODUCTION.tex:74`): "the same request produces a TTFT, a prompt-cache ratio, and a grounding score, which is what makes cross-axis trade-offs visible."

## 2.5 Phase design (verbatim, `4_METHODOLOGY.tex:167-178`)

- **Phase 1 (local validation):** "CPU execution of the nine-baseline suite on Qwen3-4B and SQuAD v2 on a local environment. The objective was to validate orchestration and metric integrity; the absolute latencies are not generalizable, and the quality numbers are preliminary." (Phase 1 used stochastic decoding, temperature 0.7.)
- **Phase 2 (single-GPU scaling):** "Execution on L4 and A100 accelerators with larger models, under authentic KV-cache memory pressure. This phase exercises the compression axis and the speculative arm … and re-establishes the quality numbers under realistic load." (Greedy, T=0.)
- **Phase 3 (distributed / HPC):** "Execution across multiple nodes, where the analytical transfer model of the distributed baseline is replaced by a real disaggregated-prefill transfer path, exposing the frontier between distributed KV-cache efficiency and information-retrieval quality."

**Baseline taxonomy the thesis names (9 families, 2 core axes = context source × reuse policy):** No Cache (gold/full recompute — control), Prefix Cache (gold/vLLM prefix reuse), RAG (retrieved/dense), Redis cold (retrieved/centralised cache), Hybrid cold & warm (retrieved + prefix), Distributed (gold/prefix-aware routing across 3 replicas), Speculative (gold/spec decoding), Compressed RAG (retrieved/LLMLingua prompt compression), Compressed CAG (gold/FP8 KV compression). Extended into a "2×2 compression axis" plus a matched speculative arm. **This matches the code's 9 enum families (§1.1); keep the count at 9.**

## 2.6 Bib keys to mirror (all confirmed present in `Main.bib`)

**The 6 highest-priority keys** the docs should mirror to credit prior work correctly: `kwon2023efficient` (vLLM), `lettucedetect2025` (primary metric), `espejel2023ragas` + `ares2024` (quality-eval lineage), `li2025scbench` + `chen2025pitfalls` (the "compression has a measured quality cost" spine), and `yu2024dontdorag` (the CAG-vs-RAG decision the whole thesis operationalizes).

### A. Frameworks CAGE positions AGAINST / builds metrics on (quality axis)
| bib key | Credits |
|---|---|
| `espejel2023ragas` | RAGAS — reference-free RAG eval; motivates CAGE's strict NLI faithfulness. |
| `ares2024` | ARES — fine-tuned LLM/NLI judges; similarity ≠ factual answering. |
| `bergen2024` | BERGEN — reproducible RAG benchmarking library. |
| `li2024ragbench` | RAGBench — explainable TRACe suite. |
| `yang2024crag` | CRAG — comprehensive factual-QA benchmark. |

### B. Cache-aware serving / KV-cache systems (efficiency axis)
| bib key | Credits |
|---|---|
| `kwon2023efficient` | **vLLM / PagedAttention** (SOSP'23) — the engine CAGE instruments. |
| `zhong2024distserve` | DistServe (OSDI'24) — prefill/decode disaggregation. |
| `patel2024splitwise` | Splitwise (ISCA'24) — phase-splitting inference. |
| `agrawal2024sarathi` | Sarathi-Serve (OSDI'24) — chunked-prefill tradeoff. |
| `qin2024mooncake` | **Mooncake** — KVCache-centric disaggregated architecture. |
| `cachedattention2024` | CachedAttention (USENIX ATC'24) — multi-turn KV reuse. |
| `gim2024promptcache` | Prompt Cache (MLSys'24) — modular attention reuse. |
| `zheng2024sglang` | **SGLang / RadixAttention** (NeurIPS'24) — prefix-tree KV reuse. |
| `ye2024chunkattention` | ChunkAttention (ACL'24) — prefix-aware chunk KV cache. |
| `chen2024turborag` | TurboRAG (EMNLP'25) — precomputed chunk KV for RAG. |
| `cacheblend2025` | CacheBlend (EuroSys'25) — non-prefix chunk-KV fusion for RAG. |
| `lmcache2024` | **LMCache** — shared/hierarchical KV pool (key says 2024, paper is 2025). |
| `infiniteLLM2024` | Infinite-LLM / DistAttention + distributed KVCache. |
| `megatron-lm` | Tensor/model parallelism basis. |
| `cachegen2024` | CacheGen — KV compression + streaming (SIGCOMM'24). |
| `hicache2025` | SGLang HiCache — hierarchical GPU/host/disk KV tiers. |
| `preserve2025`, `lee2024infinigen` | PRESERVE (KV/weight prefetch); InfiniGen (dynamic KV mgmt) — HPC supports. |

### C. Compression axis — text-side + KV-side + quality-cost benchmarks
| bib key | Credits |
|---|---|
| `llmlingua2` | **LLMLingua-2** — text-side prompt compression (Compressed-RAG arm). |
| `deepseekv2` | DeepSeek-V2 **MLA** — low-rank latent attention (KV-side future arm). |
| `li2024snapkv` | **SnapKV** — KV eviction; accuracy cost of compression. |
| `ge2023h2o` | H2O — heavy-hitter KV eviction. |
| `xiao2023streamingllm` | StreamingLLM — attention-sink eviction. |
| `kvtuner` | KVTuner — FP8 mixed-precision "nearly lossless" (pro-compression side). |
| `kvcompresssurvey` | KV-compression survey (Gao 2025). |
| `paul2024kvreview` | KV-cache optimization review. |
| `javidnia2025kvccompress` | Systematic KV compression taxonomy. |
| `chen2025pitfalls` | **The Pitfalls of KV Cache Compression** — "selective amnesia"; the CENTRAL "quality cost no throughput benchmark reveals" citation. |
| `yuan2024kvreturn` | "KV Cache Compression, But What Must We Give in Return?" — quality-cost benchmark. |
| `li2025scbench` | **SCBench** — KV-cache-centric analysis across cache life-cycle (see §3). |
| `hsieh2024ruler` | **RULER** — effective context far below nominal window. |

### D. CAG / routing decision (closest in intent)
| bib key | Credits |
|---|---|
| `yu2024dontdorag` | **"Don't Do RAG" (CAG)** — cache-augmented generation can replace retrieval when KB is stable/small. |
| `li2024selfroute` | Self-Route — RAG-vs-long-context cost/quality trade-off. |

### E. Primary metric model + methods CAGE runs
| bib key | Credits |
|---|---|
| `lettucedetect2025` | **LettuceDetect** — span-level hallucination detector = PRIMARY grounding signal. |
| `leviathan2023fast` | Speculative decoding — output-preservation basis. |
| `holtzman2020curious` | Neural text degeneration — motivates greedy decoding. |
| `yuan2025nondeterminism`, `vllmblog` | Numerical nondeterminism / batch-size determinism caveat. |
| `renze2024temperature` | Temperature-sensitivity sub-study basis. |
| `holm1979`, `wilcoxon1945`, `mannwhitney1947`, `cliff1993`, `efron1993` | Stats: Holm, Wilcoxon signed-rank, Mann-Whitney fallback, Cliff's delta, bootstrap CI. |

### F. Motivation / economics / hallucination / attention foundations
| bib key | Credits |
|---|---|
| `lewis2020retrieval` | RAG (NeurIPS'20) — foundational RAG. |
| `zhang2024siren`, `phillips2024seven` | Hallucination survey; "Seven Failure Points" of RAG. |
| `li2024kvsurvey` | KV-cache management survey — KV grows linearly, exceeds weights. |
| `samsi2023words`, `wu2022sustainable`, `sardana2024inference` | Inference cost/energy dominance; inference-aware scaling. |
| `openaicache`, `anthropiccache`, `geminicache` | Provider cache pricing (50% / ~90% / up to 90% discounts) = market evidence for CAG. |
| `vaswani2017attention` | Transformer / attention foundation. |
| `shazeer2019mqa`, `ainslie2023gqa`, `child2019generating`, `beltagy2020longformer` | MQA, GQA, sparse/sliding-window attention. |
| `dettmers2022llmint8`, `frantar2022gptq` | Weight quantization (LLM.int8, GPTQ). |
| `hashevict2024`, `kvzip2025` | HashEvict; KVzip — additional KV eviction/compression. |
| `liu2023lost`, `lazaridou2023freshllms` | Lost-in-the-Middle; FreshLLMs (long-context / freshness). |
| `qwen3report`, `mimo2025`, `glm45report` | Model cards: Qwen3, MiMo, GLM-4.5. |
| `redisdocumentation`, `vllmdocs`, `vllm2024disagg` | Redis / vLLM docs / vLLM disaggregated-prefill. |
| `wendersonIJCAE` | Author's prior RAG-for-computer-architecture work (self-cite). |

### Bib-key hygiene flags — DO NOT "correct" these keys; they are load-bearing for `\cite`
- `ge2023h2o` → first author is **Zhang** (H2O), not Ge. [CORRECTED author, key retained]
- `paul2024kvreview` → first author is **Shi**, not Paul. [CORRECTED author, key retained]
- `lazaridou2023freshllms` → first author is **Vu**, not Lazaridou. [CORRECTED author, key retained]
- `shazeer2019mqa` → real title is "Fast Transformer Decoding: One Write-Head is All You Need" (introduces MQA).
- `child2019generating` → is Sparse Transformers, **not** sliding-window; `beltagy2020longformer` carries the SWA claim.
- `lmcache2024` → citable paper is **2025** (key retained for `\cite` compatibility).
- `yu2024dontdorag` → first author is **Chan** ("Don't Do RAG" CAG paper). [CORRECTED author, key retained]

---

# 3. SCBench: What It Is + a Phase-3 Comparison Plan for CAGE

*Every identity/descriptive claim below is `[VERIFIED]` against arXiv abstract + HTML (v1/v2), OpenReview/ICLR, the MInference repo, the HF dataset, and the project page, unless tagged otherwise.*

## 3.1 Paper identity [VERIFIED]

- **Title:** *SCBench: A KV Cache-Centric Analysis of Long-Context Methods* (a.k.a. "SharedContextBench"). [arXiv abstract](https://arxiv.org/abs/2412.10319)
- **Authors:** Yucheng Li, Huiqiang Jiang, Qianhui Wu, Xufang Luo, Surin Ahn, Chengruidong Zhang, Amir H. Abdi, Dongsheng Li, Jianfeng Gao, Yuqing Yang, Lili Qiu. [arXiv abstract](https://arxiv.org/abs/2412.10319)
- **Affiliations [CORRECTED]:** **Microsoft Corporation and University of Surrey** — Yucheng Li is the University of Surrey author; the rest are Microsoft. (Earlier "all Microsoft + academic collaborators" was vague; name University of Surrey explicitly.) Confirmed on project page and arXiv v2.
- **arXiv id:** 2412.10319; v1 submitted **13 Dec 2024** (17:59:52 UTC), v2 revised **11 Mar 2025** (14:02:04 UTC). [arXiv abstract](https://arxiv.org/abs/2412.10319)
- **Venue:** **ICLR 2025** ("Published as a conference paper at ICLR 2025"). [OpenReview forum gkUyYcY1W9](https://openreview.net/forum?id=gkUyYcY1W9), [ICLR 2025 poster 28803](https://iclr.cc/virtual/2025/poster/28803), [MSR publication page](https://www.microsoft.com/en-us/research/publication/scbench-a-kv-cache-centric-analysis-of-long-context-methods/)
- **Repo:** `microsoft/MInference`, SCBench code under `scbench/` (contains `run_scbench.py`, `compute_scores.py`, `args.py`, `eval_utils.py`, `repo_qa_utils.py`, `cache_blend.yaml`, `scripts/`, `setup/`, `readme.md`). [MInference repo](https://github.com/microsoft/MInference/tree/main/scbench)
- **Dataset:** `microsoft/SCBench` on HF — **922 rows** across 12 subsets. [HF dataset](https://huggingface.co/datasets/microsoft/SCBench)
  - **[CORRECTED] Row count vs paper totals:** the HF artifact reports **922 rows**; the *paper* describes **931 multi-turn sessions with 4,853 queries** (project page + arXiv). These are different counts — do NOT conflate the HF "922 rows" with the paper's "931 sessions / 4,853 queries."
- **Project page:** https://hqjiang.com/scbench.html

## 3.2 What SCBench measures [VERIFIED]

SCBench evaluates efficient long-context methods **from a KV-cache-centric perspective, across the full KV-cache lifecycle**, in scenarios where the context (KV cache) is **shared and reused across multiple requests/turns** — a setting the authors argue prior long-context benchmarks miss by testing only single requests. It notes KV-cache reuse is now standard in vLLM/SGLang and among OpenAI/Microsoft/Google/Anthropic. [arXiv abstract](https://arxiv.org/abs/2412.10319), [arXiv HTML v1](https://arxiv.org/html/2412.10319v1)

**Four KV-cache lifecycle stages [VERIFIED]:**
1. **Generation** — efficient KV production during prefill (sparse attention, SSM/hybrids, prompt compression).
2. **Compression** — post-generation size reduction before storage (KV dropping, quantization).
3. **Retrieval** — fetching relevant cached KV blocks from a pool by request prefix to cut TTFT.
4. **Loading** — dynamically moving KV from storage (VRAM/DRAM/SSD/RDMA) to on-chip SRAM during decode.

## 3.3 Shared-context / KV-reuse modes [VERIFIED]

First long-context benchmark covering **single-turn, multi-turn, and multi-request** scenarios, with **two shared-context modes**:
- **Multi-turn** — context cached **within a single session**; the same long context persists across follow-up turns. Uses **ground-truth answers (not model output) as context for follow-up turns** to isolate context handling. [arXiv HTML v1](https://arxiv.org/html/2412.10319v1)
- **Multi-request** — the same context is encoded **once and reused across separate sessions/users**; tests whether a sparse-encoding strategy generalizes when future queries can't inform it.

Each example = shared `context` string + `multi_turns` list of sequential Q/A pairs; **2–4 turns per example** ([VERIFIED] against HF dataset viewer, `listlengths: 2 4`). [HF dataset](https://huggingface.co/datasets/microsoft/SCBench)

## 3.4 Tasks, metrics, models, methods [VERIFIED]

**12 tasks in 4 capability categories:**
- **String retrieval:** Retr.KV, Retr.Prefix-Suffix, Retr.MultiHop — *Accuracy*
- **Semantic retrieval (4 tasks):** Code.RepoQA (*Pass@1*), En.QA, Zh.QA, En.MultiChoice (*Accuracy*)
- **Global information:** Math.Find (*Accuracy*), ICL.ManyShot (*Accuracy*), En.Sum (*ROUGE*)
- **Multi-tasking:** Mix.Sum+NIAH (*ROUGE + Accuracy*), Mix.RepoQA+KV (*Pass@1 + Accuracy*)

**Models (8 total):** 6 Transformer — Llama-3.1-8B/70B, Qwen2.5-72B/32B, Llama-3-8B-262K, GLM-4-9B-1M — plus 2 non-Transformer: Codestral Mamba, Jamba-1.5-Mini. (The abstract's "six Transformer-based LLMs" counts only the Transformer subset.) [VERIFIED]

**Eight method categories (13 concrete methods):** Gated Linear RNNs (Codestral-Mamba); SSM-Attention hybrids (Jamba-1.5); sparse attention (A-shape, Tri-shape, MInference); prompt compression (LLMLingua-2); KV dropping (StreamingLLM, SnapKV, PyramidKV); KV quantization (KIVI); KV retrieval (CacheBlend); KV loading (Quest, RetrievalAttention). [arXiv HTML v1](https://arxiv.org/html/2412.10319v1)

**Headline finding [VERIFIED]:** sub-O(n)-memory methods degrade in multi-turn shared-context settings ("Sub-O(n) memory is almost infeasible in multi-turn decoding"), while sparse encoding with O(n) memory and sub-O(n²) prefill stays robust. [arXiv abstract](https://arxiv.org/abs/2412.10319)

## 3.5 How SCBench is KV-cache-CENTRIC — and how it differs from CAGE

- **SCBench's axis is the KV cache / method.** Unit of analysis = a *long-context/compression method* scored on how well its cached representation survives reuse across generation→compression→retrieval→loading. Dependent variable ≈ **task quality (accuracy/Pass@1/ROUGE) under a memory/compute budget**; it stresses multi-turn/multi-request reuse to expose where compressed KV loses information.
- **CAGE's axis is the serving policy, co-measured on both systems and grounding at once.** CAGE pairs **serving metrics (latency, TTFT, TPOT, throughput, p50/p95 tails, cache/retrieval telemetry)** with **span-level answer grounding (LettuceDetect) plus NLI faithfulness** on the *same requests*, across a **9-family baseline taxonomy** (§1.1).

**The precise differences (all SCBench-side claims [VERIFIED]):**
- **Same-request co-measurement:** SCBench reports quality vs. a method's memory/compute class; it does **not** score serving latency/TTFT/throughput and answer-grounding jointly on the same request. CAGE's defining move is exactly that.
- **Grounding instrumentation:** SCBench uses task-level correctness (accuracy/Pass@1/ROUGE); it has **no span-level hallucination/grounding metric**. CAGE wires `grounding_score` / `hallucinated_span_ratio` next to TTFT/throughput.
- **Taxonomy axis:** SCBench's rows are **8 long-context/compression *methods*** over 4 KV lifecycle stages; CAGE's rows are **9 serving *policies***.
- **Reuse framing:** SCBench treats multi-turn/multi-request as a *robustness stressor for compressed KV*; CAGE treats reuse as a *serving-policy design choice whose efficiency-vs-grounding trade-off is the measured object*.
- **Latency reporting [VERIFIED — safe to claim]:** SCBench reports task-quality metrics only, with **no per-method end-to-end serving-latency / TTFT tables**. It is safe to state SCBench does not publish per-method serving latency (this is exactly the gap CAGE fills).

**Net one-liner:** SCBench asks "how much task quality does each KV/long-context method retain when the cache is reused?"; CAGE asks "for each serving policy, what is the joint (serving-efficiency, answer-grounding) operating point on the same requests?" They are complementary.

## 3.6 Concrete Phase-3 comparison plan for CAGE

**Goal:** show CAGE's joint serving+grounding lens surfaces a trade-off SCBench's quality-only, method-centric lens cannot — specifically in the **multi-request shared-context** regime that motivates Phase-3 (`compressed_cag`, `distributed`). (This is a *design*, not an executed result — no SCBench code was run in the source session.)

**What to run:**
1. **Adopt SCBench's shared-context workload as an external, reuse-native input.** Load `microsoft/SCBench` from HF and drive CAGE's harness with the shared `context` + `multi_turns` structure so the SAME long context is reused across turns/requests inside CAGE's serving pipeline. [HF dataset](https://huggingface.co/datasets/microsoft/SCBench)
2. **Select a grounding-clean task subset.** Use the **semantic-retrieval group** (Code.RepoQA, En.QA, Zh.QA) and **En.Sum** — free-text answers over a bounded gold context, which is what LettuceDetect/NLI grounding needs. Avoid pure string-retrieval tasks (Retr.KV etc.), whose exact-match accuracy carries little grounding signal.
3. **Map SCBench methods onto CAGE baselines:** SCBench KV-dropping/quantization ↔ CAGE `compressed_cag` (FP8/MLA); SCBench KV-loading/retrieval (Quest/CacheBlend) ↔ CAGE `hybrid`/`distributed`; SCBench "full-cache" ↔ CAGE `prefix_cache`; add CAGE-only `rag`/`compressed_rag` rows SCBench has no equivalent for.
4. **Run each cell under CAGE's protocol** (T=0, repeated trials, same-request logging) so every SCBench-derived request emits BOTH CAGE serving metrics AND grounding, in single-turn, multi-turn, and multi-request modes.

**Mapping SCBench signals onto CAGE's metric suite:**
- SCBench **accuracy / Pass@1 / ROUGE** → CAGE **task-quality column**, alongside (not replacing) CAGE's **span-level grounding** (the axis SCBench lacks).
- SCBench **lifecycle stage** → CAGE **serving telemetry** at that stage: Generation→TTFT/prefill; Compression→KV bytes/token; Retrieval→TTFT reduction from cache hits; Loading→transfer/cold-start cost in the `distributed` baseline.
- SCBench **multi-turn degradation of sub-O(n) methods** → tested against CAGE's KV-eviction-hurts-faithfulness hypothesis, now measured as a *grounding* drop, not just an accuracy drop.

**Exact claim the comparison supports:**
> "On Microsoft's SCBench shared-context, KV-cache-reuse workload (ICLR 2025), CAGE reproduces SCBench's method-level quality ranking AND additionally quantifies the paired serving-efficiency-vs-answer-grounding operating point of each cache policy on the same requests — revealing, for KV-compression and multi-request reuse, a grounding-cost that SCBench's quality-only, method-centric lifecycle analysis does not expose."

**Doc guidance:** mirror bib key `li2025scbench`. When citing SCBench in prose, use it as the *closest cache+quality co-measurement prior work* that nonetheless (a) has no grounding metric and (b) publishes no per-method serving latency — the two gaps CAGE fills.

---

# 4. KV-Cache-DB Baseline: Reality Check, Survey, and Go/No-Go

## 4.1 "ConDb" / "TableCache" reality check [VERIFIED as REFUTED-for-purpose]

Both names resolve to real systems, but **NEITHER is a mainstream KV-cache-store** usable as a serving-cache baseline. The "red herring" verdict is CONFIRMED.

- **ConDB (VectifyAI)** — REAL but wrong category. A tree-structured, reasoning-based **context/retrieval database** (SQLite-backed) that replaces vector search with LLM tree search and "caches intermediate results during tree search" (up to ~70% token reduction). Does **NOT** store vLLM KV blocks; does **NOT** integrate with vLLM. Brand-new/immature: v1.0 released 2026-05-27, ~40 GitHub stars; depends on Anthropic/OpenAI APIs at runtime. This is a retrieval-layer artifact (closer to what CAGE's Redis already does). https://github.com/VectifyAI/ConDB
- **TableCache** — REAL but narrow. Paper "TableCache: Primary-Foreign-Key-Guided KV Cache Precomputation for Low-Latency Text-to-SQL" (submitted 2026-01-13). Precomputes per-table KV caches offline, matched via a "Table Trie"; up to 3.62× TTFT speedup. Domain-specific Text-to-SQL KV-precompute technique, not a general serving-cache system or off-the-shelf DB. https://arxiv.org/abs/2601.08743

**What was most likely meant:** for "a caching-database system that stores KV blocks," the intended systems are **Mooncake Store** (a distributed KVCache storage engine — closest to a "KV-cache database") and/or **LMCache** (the KV-cache storage/offload layer). A secondary candidate for "ConDb" is **CacheGen** (encodes KV caches to a bitstream on disk).

## 4.2 The real options and what each actually caches [VERIFIED]

| System | What it caches | vLLM integration | Effort (master's) | URL |
|---|---|---|---|---|
| **LMCache** | **KV blocks** extracted from GPU mem; tiered across CPU DRAM / SSD / Redis-Valkey / Mooncake / S3 / NIXL (+InfiniStore/GDS) | First-class; `LMCacheConnector` via `--kv-transfer-config`; in vLLM production-stack | **Low** — de-facto standard, config-driven | https://github.com/LMCache/LMCache |
| **Mooncake (Store + Transfer Engine)** | **KV blocks** in a distributed store over DRAM/SSD; RDMA/TCP/NVMe-oF transfer | Official since 2024-12-16; `MooncakeConnector` / tier under LMCache | **Medium** — closest to a real "KV-cache DB"; RDMA shines multi-node | https://github.com/kvcache-ai/Mooncake |
| **CacheGen** | **KV blocks compressed to a bitstream on disk**, shareable across vLLM instances (SIGCOMM'24) | Ships inside LMCache | Low-Medium (LMCache mode) | **[CORRECTED URL]** https://arxiv.org/abs/2310.07240 |
| **CacheBlend** | **KV blocks for non-prefix RAG chunks** + selective cross-attention recompute (EuroSys'25 best paper) | Ships inside LMCache + vLLM production-stack | Low-Medium (LMCache mode) | https://blog.lmcache.ai/en/2025/03/31/cacheblend/ |
| **GPTCache** | **Full LLM responses**, keyed by query **embedding** (semantic cache); SQLite/Postgres + vector store | App-layer wrapper (LangChain/llama_index); NOT a vLLM KV backend | Low, but **different axis** — response cache, not KV | https://github.com/zilliztech/gptcache |
| **Redis / Valkey** | Generic KV; as a KV-block backend it is a **tier under LMCache**. In CAGE today it caches **retrieval doc-ids only** | Only via LMCache as a remote tier | Trivial (already running) | https://github.com/LMCache/LMCache |
| **Memcached** | Generic bytes; no first-class LLM-KV integration | None official | Not worth it | (no LLM-KV integration) |
| **vLLM prefix cache + KVConnector (NixlConnector)** | **KV blocks**: in-GPU automatic prefix cache + `KVConnectorBase_V1` for cross-process/RDMA transfer (NIXL over UCX/RoCE/IB) | **Native, built-in** | **Lowest** — already in the engine; NIXL is the Phase-3 disaggregation path | https://docs.vllm.ai/en/stable/features/nixl_connector_usage/ |
| **CachedAttention / Pcache** | KV reuse across **multi-turn** conversations (ATC'24 research) | Research prototype, not a drop-in DB | High | https://www.usenix.org/conference/atc24/presentation/gao-bin-cost |
| **ChunkAttention** | Prefix-aware **in-memory** KV sharing via prefix tree + custom kernel | Research kernel, not a store | High | https://arxiv.org/abs/2402.15220 |

**Key distinction for the thesis:** GPTCache caches **responses/embeddings** (semantic hit → skip inference entirely); LMCache/Mooncake/CacheGen/CacheBlend/NIXL cache **KV blocks** (skip prefill recompute). CAGE's current Redis caches **neither** — only retrieval doc-ids (`RE/src/orchestration/redis_cache.py`; its docstring: "This repo does NOT store raw vLLM KV-cache blocks in Redis").

## 4.3 Go/No-Go recommendation

**Does CAGE need a KV-cache-DB baseline beyond Redis? — YES, exactly ONE.** CAGE's current Redis baseline exercises only the *retrieval* axis. A reviewer will note CAGE claims a serving-cache angle while never caching the thing that dominates serving cost — the KV blocks / prefill. One KV-block-store baseline closes that gap and makes the "serving" half of the joint axis real.

**Which single system: LMCache.**
- **Defensible / standard.** ~10k stars, de-facto KV-cache layer, first-class vLLM connector, MLSys visibility, CoreWeave adoption. https://github.com/LMCache/LMCache
- **Distinct from Redis on the right axis.** Caches **KV blocks** (prefill reuse), genuinely different from CAGE's retrieval-doc-id Redis — and it can *use the existing Redis as its remote tier*, giving a clean narrative ("Redis-as-retrieval-cache" vs "Redis-as-KV-tier-under-LMCache").
- **Feasible for a master's.** Config-driven via `--kv-transfer-config` / `LMCacheConnector`; no kernel work.
- **Subsumes the compression/quality question.** CacheBlend and CacheGen ship *inside* LMCache → a compressed-KV and a RAG-chunk-fusion variant for free, feeding the quality-vs-serving trade-off with published (self-reported) numbers to reproduce or contest.

**Do NOT add** Mooncake as a *separate* baseline unless Phase 3 goes multi-node with real RDMA — it overlaps LMCache (becomes a tier under it); its value is cross-node bandwidth, relevant only with the A3-Ultra RoCE setup. If Phase 3 lands, Mooncake/NixlConnector become the *disaggregation transport*, not a second cache DB. **GPTCache** is optional/orthogonal: add only for an explicit **response-cache** contrast (semantic hit skips inference); it does not substitute for a KV-block baseline.

**Net:** one addition — **LMCache (with CacheBlend as the quality-preserving KV-reuse variant)** — gives a standard, distinct, low-effort KV-cache-store baseline that makes the serving half of the joint thesis defensible without scope creep.

**Caveats to carry into the docs:**
- "ConDb"/"TableCache" as *KV-cache-store baselines* are **not verifiable** — the real systems by those names are a context-retrieval DB and a Text-to-SQL precompute paper.
- Vendor speedup figures (LMCache "up to 15×"; CacheBlend "3× TTFT / 3× throughput, F1 preserved" on 2WikiMQA / Llama-70B / A40; CacheGen "3.5–4.3× size, 3.2–3.7× delay") are **self-reported** on specific datasets — treat as claims to reproduce, not established fact. [VERIFIED as self-reported]
- LMCache's "de-facto standard" and star counts come from the project's own README — directionally reliable but self-described.

---

# 5. Staleness/Freshness Baseline: Implementable Design + Citations

## 5.1 Prior art (what exists, and the gap it leaves)

### A. Semantic-cache staleness and false-hit risk
Semantic caching reuses a prior answer when a new query is embedding-similar to a cached one, trading a cosine threshold for cost. Core hazard: the **false hit** — a semantically-adjacent-but-not-equivalent query gets served the wrong cached answer. Lowering the threshold raises hit rate but injects false positives; raising it protects accuracy but collapses hit rate — a direct cost-vs-correctness dial. Per-entry TTL bounds staleness, but TTL "is insufficient for rapidly changing data."
- GPTCache (Bang, NLP-OSS 2023): https://aclanthology.org/2023.nlposs-1.24/
  - **[CORRECTED] threshold attribution:** the "0.6–0.9 sweep with 0.8 optimum" is from **GPT Semantic Cache** (Regmi & Pun), NOT GPTCache. GPTCache's own **0.8 default** comes from its *library docs/implementation*, not the ACL paper — attribute accordingly and keep the two claims separate.
- GPT Semantic Cache (threshold swept 0.6–0.9; below 0.8 raises hit rate but "introduces irrelevant matches decreasing accuracy" — [VERIFIED verbatim]): https://arxiv.org/html/2411.05276v2
- Threshold/false-hit tradeoff + TTL insufficiency: https://blog.premai.io/semantic-caching-for-llms-how-to-cut-api-bills-by-60-without-hurting-quality/ , https://www.buildmvpfast.com/blog/semantic-caching-ai-agents-cost-optimization

### B. Bounding the false-hit / error rate (correctness-constrained view)
- **vCache** [VERIFIED] — frames a user-specified maximum error rate δ and guarantees `Pr(vCache(x)=r(x)) ≥ 1−δ` (Theorem 4.1), learning per-embedding thresholds. Reported: **up to 26× lower error at matched latency, up to 12.5× higher hit rate at matched error**; **~57% hit rate while keeping error below 0.5%** (on SemCacheLMArena). **[CORRECTED]** drop the over-precise "at δ=0.005" attribution — the paper ties the ~57%/<0.5% result to a general low-error target, not specifically δ=0.005. https://arxiv.org/html/2502.03771
- **Closing the Calibration Gap in Semantic Caching** [VERIFIED] — defines the **P-CHR curve** (precision vs cache-hit-ratio as the score threshold varies) and P-CHR-AUC: "how well a model maintains precision as cache utilization grows." As τ falls, CHR rises but precision falls. Exact serving-vs-grounding curve shape, applied to caching: https://arxiv.org/html/2606.19719v1 , PDF https://arxiv.org/pdf/2606.19719

### C. Reuse of stale retrieved context / stale KV (closest prior art)
- **GroundedCache — "Grounded Cache Routing for RAG: When Is It Safe to Reuse an Answer?"** [VERIFIED] (S.H. Shah, Duke, 26 May 2026). THE primary anchor. Caches *answers* and gates reuse with four checks including **G3 Version Match** and **G4 Evidence Support**. Defines exactly the metrics CAGE needs:
  - **USR (Unsafe-Served Rate)** = fraction of *all* queries served a wrong cached answer;
  - **aHR (Answer-cache Hit Rate)**;
  - **FH (False-Hit Rate)** = error rate *conditional on a cache hit* (**FH = USR/aHR = Pr[dis|ac]**, [VERIFIED]);
  - **Stale Hit (SH)** = cached answer whose evidence version no longer matches; **Unsupported Hit (UH)** = cached answer not lexically supported by current evidence.
  - Reported numbers [VERIFIED verbatim]: naive semantic caching **USR 15.5% / 22.0% / 35.0% (HotpotQA)** and **26.0% / 26.0% / 51.5% (mtRAG)**; gating drove USR toward 0 at only **1.04–1.07× no-cache latency**. Proves stale reuse is a *measurable, large* grounding cost. https://arxiv.org/html/2605.27494 , PDF https://arxiv.org/pdf/2605.27494
- **KV-reuse quality cost:** reusing precomputed KV for non-prefix / changed chunks "sacrifices quality due to missing inter-chunk attention"; CacheBlend must *selectively recompute* to recover it [VERIFIED verbatim]. Even at the KV layer, stale/approximate reuse has a documented quality penalty. CacheBlend: https://arxiv.org/html/2405.16444v3 ; CacheClip: https://arxiv.org/abs/2510.10129

### D. RAG index freshness (why entries go stale)
Staleness arises when source updates don't propagate to the index; a pipeline can score 0.95 faithfulness yet be wrong because the index is stale. Freshness is measured via timestamp/version verification and is "the metric most often missing from RAG panels."
- https://www.amicited.com/faq/how-do-rag-systems-handle-outdated-information/ , https://atlan.com/know/how-to-evaluate-rag-systems-explained/
- Fast-changing-fact benchmarks: **FreshQA / FreshLLMs** (never/slow/fast-changing/false-premise categories [VERIFIED verbatim]; venue = **Findings of ACL 2024**, `2024.findings-acl.813`): https://arxiv.org/abs/2310.03214 ; UnSeenTimeQA (ACL 2025 long 94): https://aclanthology.org/2025.acl-long.94.pdf

### E. TTL / eviction / invalidation policy background
TTL, event-based invalidation, and version keys trade freshness vs performance; production systems "accept a defined maximum staleness … and design cache TTLs to keep staleness within that bound." Provider prompt caches use **5-min–1-hr** time-based eviction.
- https://mbrenndoerfer.com/writing/caching-prompt-semantic-invalidation-hit-rates-llm , https://bugfree.ai/knowledge-hub/ttl-eviction-policies-cache-invalidation
- Learned semantic-aware eviction: "Not All Tokens Are Worth Caching: Learning Semantic-Aware Eviction for LLM Prefix Caches" (Fang et al.): https://arxiv.org/pdf/2605.18825
  - **[CORRECTED]** This paper is a *learned token-importance eviction* method for prefix caches. It does **NOT** introduce a "SAECache" name, does **NOT** discuss "v-LRU," and does **NOT** make the "immediately invalidate expired-TTL entries" claim. Cite it only for "learned semantic-aware eviction"; move the "v-LRU / immediate TTL-invalidation" narrative to a different source (e.g. the TTL/eviction pages above) or drop it.

### The gap CAGE fills
Every serving-efficiency paper treats reuse as *free* once conditions match; every correctness paper fixes false hits but reports the tradeoff on *its own axis*. **No one places a single controlled "how stale is the reused entry" knob on CAGE's joint axis (serving win vs grounding loss) alongside a family of cache-warmth baselines measured on identical footing** — CAGE's declared differentiator (unified taxonomy comparing cache *policies* on equal footing).

## 5.2 The "Staleness / Freshness" baseline design

**Design principle.** CAGE's arms vary cache *presence/warmth* (no_cache → prefix_cache → redis → hybrid cold/warm → distributed). None varies cache *age*. The new arm holds warmth fixed at "warm" and sweeps a single **staleness** knob, so the paper can plot the **serving-win vs grounding-loss** curve the whole thesis is about.

**Independent variable (the knob).** `stale_fraction ∈ {0.0, 0.25, 0.5, 0.75, 1.0}` — the fraction of served cache hits that are deliberately **stale** (the retrieved-artifact/context bound to the cached entry is an *outdated version* of the gold evidence). Expressible alternatively as a **TTL sweep** (`cache_ttl_seconds`) for a temporal narrative, but `stale_fraction` is the cleaner primary IV (deterministic, no wall-clock dependence). This operationalizes GroundedCache's "Version Match (G3)" failure axis as a continuous dial (https://arxiv.org/html/2605.27494) and reframes the calibration-gap/P-CHR threshold sweep (https://arxiv.org/html/2606.19719v1) as *age* rather than *similarity*.

**Making an entry stale deterministically** (reusing the existing SQuAD-style corpus):
- **v1 (fresh)** = gold passage / current top-k chunks for the query.
- **v0 (stale)** = a prior version: passage with the answer span perturbed/redacted, an older paragraph from the same document, or a sibling passage that no longer contains the answer. Tag each cached entry `evidence_version ∈ {v0, v1}` + `version_ts`. For `stale_fraction=p`, `p` of served hits carry `v0`.
This reuses `RetrievalCache` keying (`RE/src/orchestration/redis_cache.py`); staleness is just serving the `v0` hits for the chosen fraction. **`set()` already accepts `ttl_seconds`, so the TTL variant needs no new cache API.**

**Held constant (clean axis):** model, decoding (T=0), dataset, query set/order, top_k, embedding + reranker models. **Cache is warm in every cell** (population identical to `hybrid_warm`) — only *age/validity* of served entries changes. **Hit rate held ~constant** across the sweep (all cells serve from cache at the same rate; only the fresh/stale mix changes). This is what makes it *distinct* from vCache-style curves, which vary the *threshold* and thus co-vary hit rate with error. CAGE holds hit rate fixed and varies *age* — a genuinely new cut.

**Measured (the joint axis):**
- **Serving side (win):** TTFT, TPOT, end-to-end latency, QPS, prompt-cached ratio, retrieval-cache rate (already in `METRICS_SPECIFICATION.md`). Expectation: ~flat across `stale_fraction` (a stale hit is served as cheaply as a fresh hit — the whole trap). That flatness *is* the finding.
- **Grounding side (loss):** CAGE primary metrics (LettuceDetect faithfulness/hallucination; answer correctness/EM/F1), **plus three staleness-specific metrics ported from GroundedCache** (https://arxiv.org/html/2605.27494):
  - **USR (Unsafe-Served Rate)** = fraction of *all* queries served a wrong-because-stale answer.
  - **FH (False-Hit Rate)** = error rate *conditional on a cache hit*.
  - **SHR (Stale-Hit Rate)** = fraction of served `v0` hits producing an ungrounded answer (CAGE's controlled analogue of GroundedCache's SH).
- **Headline artifact:** a **Staleness-Cost curve** — grounding/faithfulness (and USR) on Y vs `stale_fraction`/TTL on X — overlaid with the flat serving-win line. CAGE's own P-CHR/vCache-style tradeoff plot, on the *freshness* axis it uniquely owns.

**Distinctness from existing arms:**
| Existing arm | Varies | New staleness arm |
|---|---|---|
| `prefix_cache` | KV present vs absent | age of reused entry, KV held on |
| `redis` (cold) | retrieval-cache empty vs full | retrieval-cache full but *outdated* |
| `hybrid_warm` | cache warmed vs cold | cache warm but a controlled *fraction is stale* |
| `distributed` | routing/placement | orthogonal — staleness composes with any |

Existing arms answer "does caching help?"; the staleness arm answers "**what does a cheap cache hit cost when the cached thing is out of date?**" — the missing quadrant.

## 5.3 Implementable spec against the current harness
1. **`BaselineType.STALE` / `"staleness"`** in `RE/src/orchestration/baselines.py`.
2. **New `BaselineConfig` fields:** `stale_fraction: float = 0.0`, `cache_ttl_seconds: Optional[int] = None`, `stale_evidence_mode: str = "version"` (`"version"|"ttl"`), `evidence_version_field: str = "evidence_version"`. Add to `to_dict()`.
3. **Preset:** clone `hybrid` (warm, `use_faiss=True`, `enable_prefix_caching=True`) and set the staleness fields — identical serving path, only age varies.
4. **Cache layer:** extend `RetrievalCache` entries with `{evidence_version, version_ts}` (or use the existing `ttl_seconds` for TTL mode). A `StaleServingPolicy` picks `v0`/`v1` per query to hit target `stale_fraction` deterministically (seeded).
5. **Corpus prep:** offline step in `scripts/extract_qa_evidence.py` to emit `v0` (stale) alongside `v1` (fresh) evidence per query.
6. **Metrics:** add USR / FH / SHR to `RE/src/evaluation` next to the LettuceDetect scorer; they need only (served-from-cache?, evidence_version, grounded?) which the run loop already has.
7. **Sweep:** `for f in 0 0.25 0.5 0.75 1.0: run_experiment.py --baseline staleness --stale-fraction $f --trials 3 --queries 300` (matching the 300×3 plan of record — reconcile with the actual query count per §1.6).

**Threats to validity to pre-empt:**
- **Confound with context-source quality** — here it is *controlled* (v0/v1 are the same document family), a strength not a leak.
- **Synthetic staleness ≠ real drift** — optionally validate one cell against a FreshQA-style fast-changing-fact set (https://arxiv.org/abs/2310.03214) to show the synthetic knob tracks real temporal drift.

**Note:** the internal-harness claims (`RetrievalCache.set()` taking `ttl_seconds`, `BaselineType`/`BaselineConfig` surface, `COMPARISON_MATRIX.md:87`, `PROJECT_OVERVIEW.md:340`) are code/doc assertions about the local repo, not web citations — confirm by reading the files directly before relying on exact line numbers.

---

# 6. Per-Doc Update Checklist

*Priority order: **#6 CAGE_PRESENTATION_GUIDE (major rewrite) > #8 COMPARISON_MATRIX (baseline-count + Cmp status) > #1/#2/#4/#5 (add datasets, refresh) > #7 PHASE3_PLAN (dataset status) > #3 SOLUTION_DESCRIPTION.txt (leave frozen)**.*

## Cross-cutting themes (apply consistently to all live files)
1. **Datasets:** add **Qasper, CRAG, ShareGPT** everywhere loaders are enumerated (#1, #2, #4, #5, #7) — wired end-to-end (§1.5).
2. **Baseline count = 9** (not 7) — fix in #8 §4; already correct in #1/#2/#4/#5.
3. **Telemetry no-mock invariant** — surface as an explicit guarantee (§1.3) in #1, #2, #4, #5.
4. **Temperature 0.0 / greedy** — #6 is the only file still on 0.7; pervasive and load-bearing (Q4/Q6 premises break).
5. **Phase 2 complete on L4** — #6 is the only file still framing CPU-only-Phase-1 as the delivered state.
6. **BERTScore = negative control**, not a completeness/quality metric — fix in #6.
7. **New optional axes to note where relevant:** LMCache KV-store baseline (§4), Staleness/Freshness arm (§5), SCBench Phase-3 comparison (§3).

## 1. `SOLUTION_DESCRIPTION.md` — CURRENT, minor updates
High-level EN overview (§1 What is CAGE, §2 Architecture, §3 Baseline taxonomy 9+2×2, §4 Metrics, §5 Results Ph1/2, §6 Tech stack, §7 Status+fixes). Largely accurate (already 9 baselines, LettuceDetect-primary, mock-free telemetry).
- [ ] **§2 & §6 datasets:** loader list says "SQuAD v2, HotpotQA, TriviaQA, NQ, MuSiQue" — **add Qasper, CRAG, ShareGPT** (biggest content gap).
- [ ] **Header date** `2026-06-28` → refresh; note "remove mocks / update stats" commit landed after.
- [ ] **§4/§6 telemetry:** state explicitly telemetry is **live-only, no mock path** (now a hard code guarantee, §1.3).
- [ ] **§7 Phase 3 datasets:** note CRAG/ShareGPT now available as RAG-favorable / serving-trace datasets (the "RAG-favorable dataset is a Phase-3 need" is now partially met).
- [ ] Verify — no staleness — 9-baseline list, FP8/EAGLE-3, temperature 0.0. All correct. (Do NOT reintroduce `CAGE_REQUIRE_COMPRESSION` as a live control — §1.4.)

## 2. `SOLUTION_DESCRIPTION.pt-BR.md` — CURRENT, mirror of #1
Exact PT-BR translation (same 7 sections).
- [ ] Apply **identical edits as #1**: add Qasper/CRAG/ShareGPT (§2 ~line 47, §6 ~line 136); refresh date; add no-mock telemetry note; note CRAG/ShareGPT availability in §7.
- [ ] Keep EN/PT in lockstep — every #1 content change mirrored here.

## 3. `SOLUTION_DESCRIPTION.txt` — SUPERSEDED / do NOT update
Header (lines 1-4) declares "STATUS: SUPERSEDED / HISTORICAL (2026-06-09), kept for history only."
- [ ] **Do NOT update** — intentionally frozen; stale on nearly everything (7 baselines, Qwen3-4B/CPU only, no LettuceDetect, NLI-only quality, no compression axis, no crag/sharegpt, non-deterministic temperature).
- [ ] **Optional cleanup:** if the `.md` versions are canonical, consider deleting this `.txt` or confirm the README doc-index no longer points to it as authoritative. No edit effort here.

## 4. `TECHNICAL_ARCHITECTURE.md` — CURRENT, minor updates
Module-by-module deep reference. Highly accurate (matches CLI flags, module names, fixes).
- [ ] **`src/data/loader.py` section (~line 76):** loader list is "SQuAD v2, HotpotQA, TriviaQA, NQ, MuSiQue (+ code)" — **add Qasper, CRAG, ShareGPT**. One line each: CRAG (gold answer + retrieved candidate docs); ShareGPT (serving-workload trace, reference-only answers, `no_gold_answer=True`).
- [ ] **Configs section (~line 246):** `configs/dataset/*.yaml` says "squad_v2, hotpotqa, and the additional loaders" — only `squad_v2.yaml` + `hotpotqa.yaml` exist on disk; crag/sharegpt/qasper are loader-only (no YAML). Clarify dataset-YAML lags the loader registry.
- [ ] **Configs model (~line 245):** says "qwen3-4b/8b/14b/30b-a3b, qwen2.5-7b-instruct" — disk also has **`deepseek-v2-lite.yaml`** (MTP candidate). Add it.
- [ ] **Telemetry section:** reinforce the **no-mock invariant** (`capture_snapshot` returns None, never fabricated).
- [ ] **Statistical layer:** still correctly standalone script. Optionally add that the runner now **primes warm baselines on the same query set** so `hybrid_warm` is paired (fixes the Phase-2 unpaired fallback).
- [ ] Verify — accurate — `--baseline`/`--num-trials`/`--context-source`, EngineCore kill, FP8×prefix gate, `retrieval_hit_rate` fix, `evaluate_completeness` None fix, `mean_or_none`. No change.
- [ ] (Optional new material) reference §4 LMCache as the recommended KV-store baseline and §5 staleness arm as future baselines.

## 5. `TECHNICAL_ARCHITECTURE.pt-BR.md` — CURRENT, mirror of #4
PT-BR translation, section-for-section identical.
- [ ] Apply **identical edits as #4**: add Qasper/CRAG/ShareGPT (~line 78); add deepseek-v2-lite to model configs (~line 253); clarify dataset-YAML vs loader gap (~line 254); reinforce no-mock telemetry; optional warm-pairing note.
- [ ] Keep in lockstep with #4.

## 6. `CAGE_PRESENTATION_GUIDE.md` — STALE (largest divergence; TOP-PRIORITY rewrite)
Bilingual defense guide (elevator pitch, 15 slides, narrative arc, 13 Q&A, timed pitches, checklist). Pinned to a Phase-1-only, pre-audit reality; contradicts current code in load-bearing ways.
- [ ] **Temperature 0.7 → 0.0.** Pervasive: top banner, Slide 11, Slide 4 Q&A, **Q4 ("Why temperature 0.7?")**, Q6, Slide 13, both timed pitches. Code is `temperature=0.0, stop=["\n"]` (greedy). **Q4 is now a wrong premise** — rewrite as "why greedy/deterministic decoding" (comparability + output-preservation for the losslessness gate; cite `holtzman2020curious`, `leviathan2023fast`). Q6's "0.636 is sampling noise" rationale collapses under greedy — re-derive or drop.
- [ ] **"Phase 1, single CPU node" → Phase 1 + Phase 2 (L4 GPU) complete.** Slides 10, 14, 15, Q5, Q7, both pitches need Phase-2 results folded in (prefix TTFT −3.3%, EAGLE-3 TPOT −41%, FP8 lossless, RAG faithfulness −24.7%). ⚠️ Verify these Phase-2 numbers against the actual run before publishing (§1.6 query-count conflict applies).
- [ ] **Phase-1 numbers (37.4% latency / 65.7% TTFT / faithfulness 0.570 / distributed 0.636) are CPU-relative** — keep only if explicitly labeled Phase-1-CPU; the headline is now the L4 numbers.
- [ ] **"Grounding primary but tables show NLI / no grounding column" (Q10, Slide 8):** stale — `grounding_score` is computed and is the Phase-2 primary reported metric. Q10 as "a real gap I own" is no longer true.
- [ ] **"Compression axis has no results / no runtime caller" (Slide 6, 7, Q8):** stale — compression baselines run (`compressed_cag` FP8 Phase-2 result exists; `compressed_rag` strict-enforced). "7 of 9 reported" → 8 of 9 in Phase 2. (Describe fp8/spec correctly per §1.1(d): server-launch-configured, runner records.)
- [ ] **"Significance script not integrated / runner reports mean±sd" (Slide 9, Q9):** partially stale — still standalone, but Phase 2 ran Holm-corrected Wilcoxon vs no_cache and warm baselines are now paired. Reframe.
- [ ] **"BERTScore baseline-rescaled completeness metric" (Slide 8, Q2):** now **negative control** — update Q2 and Slide 8.
- [ ] **Datasets:** guide never mentions Qasper/CRAG/ShareGPT; the "SQuAD gold-vs-retrieved confound / equalized arms not run" thread (Q3, Slide 13) is now addressable with CRAG/HotpotQA and `--context-source gold|retrieved`.
- [ ] **Simulated transfer / replicated-only distributed:** still TRUE (Phase 3 not run) — keep, align with PHASE3_PLAN Option-2 framing.
- [ ] (Optional Q&A additions) prepare answers using §3 (SCBench as closest prior work with no grounding + no per-method latency), §4 (LMCache as the planned KV-store baseline that makes the serving half real), §5 (staleness arm as the missing quadrant).

## 7. `PHASE3_PLAN.md` — CURRENT, small dataset-status refresh
Phase-3 architecture (4 KV-distribution options + fork + recommendation), dataset axis, what's-real-now, steps, DoD. Architecturally current (Option 2, simulated-transfer honesty, LMCache/NIXL).
- [ ] **Dataset table "Code status":** CRAG and ShareGPT are **not listed** but are now fully wired (loaders + CLI + download script). Add CRAG (RAG-fair/serving) and ShareGPT. Update the "genuine future phases (a) production-realism on ShareGPT/Azure trace" note (~line 57) — ShareGPT loading now **exists**.
- [ ] **Qasper/HotpotQA/MuSiQue "NOT yet validated end-to-end":** re-verify against current smoke-test state before publishing.
- [ ] **RULER/SCBench "loader NOT wired":** re-confirm (still appears unwired — no loader found — likely accurate). Note: SCBench is now covered as a *comparison plan* (§3), even if no loader is wired.
- [ ] Verify — accurate — `SimulatedKVCacheManager` still analytic, replicated = zero transfer, `--baseline distributed` against router, Terraform GVNIC/MTU 8896. No change.
- [ ] File already has uncommitted edits (`M cloud_docs/PHASE3_PLAN.md`) — reconcile with those.
- [ ] (Optional) fold in §4's go/no-go: LMCache is the recommended real KV-store baseline for Phase 3; Mooncake/NixlConnector become the disaggregation transport only if multi-node RDMA lands. Fix the CacheGen URL to https://arxiv.org/abs/2310.07240 if cited.

## 8. `COMPARISON_MATRIX.md` — CURRENT with one internal inconsistency
Related-work/novelty positioning (8-axis matrix, new refs, novelty statement, BibTeX). Positioning sound and largely code-independent, but has a stale baseline count internal to the doc.
- [ ] **§4 item 2 says "A unified 7-baseline taxonomy (no-cache → prefix-cache → RAG → Redis → hybrid cold/warm → distributed)"** — inconsistent with the code's **9 families** and with SOLUTION_DESCRIPTION §3. Update to 9 (add speculative + two compression arms) or clarify "7 core reuse policies + 2 compression arms."
- [ ] **CAGE row "Cmp" = ✗ and CAGE✦ = target:** now understates reality — `compressed_cag` (FP8) produced a Phase-2 result and `compressed_rag` is strict-enforced. Move compression from pure-target (✦) toward ◐/partial-delivered, with the honesty guardrail (§1.1(d): runner records; server launches).
- [ ] **"Dist = ◐ (simulated)":** still accurate — keep (§1.1(c)).
- [ ] Verify — accurate — LettuceDetect-as-metric, Wilcoxon+Holm+bootstrap via `statistical_tests.py`, LMCache as Phase-3 path.
- [ ] (Optional) strengthen the novelty note with §3 (SCBench: closest cache+quality prior work, but quality-only, no grounding, no per-method latency) and §4 (LMCache KV-store baseline closes the "serving half" gap). Mirror bib keys `li2025scbench`, `lmcache2024`, `cacheblend2025`, `cachegen2024`.
- [ ] Verify venues/arXiv IDs before camera-ready (per the doc's own ⚠️). Use the corrected SCBench identity (§3.1) and CacheGen URL (§4.2).

---

# 7. Master Citation Index (all URLs preserved)

## SCBench (§3) — all [VERIFIED] unless noted
- arXiv abstract: https://arxiv.org/abs/2412.10319
- arXiv HTML v1: https://arxiv.org/html/2412.10319v1
- arXiv HTML v2: https://arxiv.org/html/2412.10319v2
- arXiv PDF: https://arxiv.org/pdf/2412.10319
- OpenReview (forum gkUyYcY1W9): https://openreview.net/forum?id=gkUyYcY1W9
- ICLR 2025 poster 28803: https://iclr.cc/virtual/2025/poster/28803
- MSR publication page: https://www.microsoft.com/en-us/research/publication/scbench-a-kv-cache-centric-analysis-of-long-context-methods/
- MInference repo (scbench/): https://github.com/microsoft/MInference/tree/main/scbench
- HF dataset (922 rows; paper = 931 sessions / 4,853 queries): https://huggingface.co/datasets/microsoft/SCBench
- Project page: https://hqjiang.com/scbench.html

## KV-cache-DB survey (§4)
- ConDB (VectifyAI — retrieval DB, NOT KV-store): https://github.com/VectifyAI/ConDB
- TableCache (Text-to-SQL KV precompute paper): https://arxiv.org/abs/2601.08743
- LMCache: https://github.com/LMCache/LMCache
- Mooncake: https://github.com/kvcache-ai/Mooncake
- CacheGen **[CORRECTED URL]**: https://arxiv.org/abs/2310.07240 (SIGCOMM'24, DOI 10.1145/3651890.3672274) — the brief's `2405.16444` is CacheBlend, not CacheGen
- CacheBlend: https://blog.lmcache.ai/en/2025/03/31/cacheblend/ (arXiv: https://arxiv.org/html/2405.16444v3)
- GPTCache: https://github.com/zilliztech/gptcache
- vLLM NixlConnector: https://docs.vllm.ai/en/stable/features/nixl_connector_usage/
- CachedAttention (ATC'24): https://www.usenix.org/conference/atc24/presentation/gao-bin-cost
- ChunkAttention: https://arxiv.org/abs/2402.15220
- CAGE Redis (local): `RE/src/orchestration/redis_cache.py`

## Staleness/Freshness (§5)
- GroundedCache (primary anchor; USR/aHR/FH/SH): https://arxiv.org/html/2605.27494 · PDF https://arxiv.org/pdf/2605.27494
- vCache (error-rate-bounded caching): https://arxiv.org/html/2502.03771
- Closing the Calibration Gap / P-CHR: https://arxiv.org/html/2606.19719v1 · PDF https://arxiv.org/pdf/2606.19719
- GPTCache (Bang, NLP-OSS 2023): https://aclanthology.org/2023.nlposs-1.24/
- GPT Semantic Cache (threshold-accuracy sweep 0.6–0.9): https://arxiv.org/html/2411.05276v2
- Semantic-cache false hits + TTL insufficiency: https://blog.premai.io/semantic-caching-for-llms-how-to-cut-api-bills-by-60-without-hurting-quality/ · https://www.buildmvpfast.com/blog/semantic-caching-ai-agents-cost-optimization
- CacheBlend (stale-KV quality cost): https://arxiv.org/html/2405.16444v3 · CacheClip: https://arxiv.org/abs/2510.10129
- RAG index freshness: https://www.amicited.com/faq/how-do-rag-systems-handle-outdated-information/ · https://atlan.com/know/how-to-evaluate-rag-systems-explained/
- FreshLLMs/FreshQA (Findings of ACL 2024): https://arxiv.org/abs/2310.03214 · UnSeenTimeQA: https://aclanthology.org/2025.acl-long.94.pdf
- TTL/eviction/invalidation: https://mbrenndoerfer.com/writing/caching-prompt-semantic-invalidation-hit-rates-llm · https://bugfree.ai/knowledge-hub/ttl-eviction-policies-cache-invalidation
- Learned semantic-aware eviction **[CORRECTED scope]** (only "learned eviction," no SAECache/v-LRU/TTL-invalidation claim): https://arxiv.org/pdf/2605.18825

## Corrections summary (what Verify changed)
- **[CORRECTED]** CacheGen URL: use `2310.07240`, not `2405.16444` (latter is CacheBlend).
- **[CORRECTED]** SCBench affiliation: Microsoft **+ University of Surrey** (Yucheng Li).
- **[CORRECTED]** SCBench counts: HF **922 rows** ≠ paper **931 sessions / 4,853 queries** — keep distinct.
- **[CORRECTED]** vCache: drop "at δ=0.005"; the ~57%/<0.5% result is a general low-error target.
- **[CORRECTED]** arXiv 2605.18825 = "learned semantic-aware eviction" only; the "SAECache / v-LRU / immediate TTL-invalidation" narrative is NOT in it — move or drop.
- **[CORRECTED]** GPTCache vs GPT-Semantic-Cache: the 0.6–0.9 sweep is GPT-Semantic-Cache (Regmi & Pun); GPTCache's 0.8 default is from library docs, not the ACL paper.
- **[CORRECTED]** Compression env var: live opt-out is `CAGE_ALLOW_NO_COMPRESSION`; `CAGE_REQUIRE_COMPRESSION` is dead (error-strings only) — do NOT document it as live.
- **[REFUTED-for-purpose]** "ConDb"/"TableCache" as KV-cache-store baselines: real systems, wrong category — red herrings.
- **[CORRECTED-in-place, keys retained]** bib author mismatches: `ge2023h2o` (Zhang), `paul2024kvreview` (Shi), `lazaridou2023freshllms` (Vu), `yu2024dontdorag` (Chan) — do NOT change the keys.
- **[CORRECTED]** codebase docstrings stale (do not cite as authority): `baselines.py:4-11` (7 vs 9), `download_datasets.py:5-11`, `loader.py:4-12` (dataset lists).

## Codebase files (source of §1)
`RE/src/orchestration/{baselines,router,cache_manager,redis_cache,compression,ir}.py` · `RE/src/evaluation/{quality,performance,compression}.py` · `RE/src/monitoring/vllm_telemetry.py` · `RE/src/data/loader.py` · `RE/scripts/{run_experiment,download_datasets,run_phase1,run_phase2,run_phase3}.{py,sh}` · `CS/cage_stats/api.py`

## Dissertation files (source of §2)
`RE/my-article/CAGE___Dissertation_Mestrado_2026/TEXT/{1_INTRODUCTION,3_RELATED WORK,4_METHODOLOGY,6_RESULTS}.tex` · `.../Main.bib`
