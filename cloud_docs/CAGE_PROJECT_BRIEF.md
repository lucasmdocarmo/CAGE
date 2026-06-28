# CAGE — Project Brief & Research Context (portable knowledge file)

> **Purpose.** A self-contained context primer for the CAGE project, written to be dropped into a
> Claude Project (or any LLM context) so future research conversations start fully informed.
> **Author/owner:** Lucas Mariano do Carmo (PUC Minas), advisors Prof. Wladmir Cardoso Brandão &
> Prof. Henrique Cota de Freitas. Building toward a master's dissertation + article.
> **Integrity rule (carry it forward):** every project claim must be grounded in the codebase/data;
> every external claim must cite a verified reference; Phase‑1 numbers are *preliminary*; anything
> from Phases 2–3 is *proposed*, never asserted as a result. Do not fabricate citations.

---

## 1. What CAGE is (one paragraph)

**CAGE (Cache‑Augmented Generation Evaluation)** is a modular framework that **jointly** evaluates
LLM serving **efficiency** (latency, TTFT, TPOT, throughput, tail percentiles), **cache/retrieval
telemetry**, and **semantic answer quality** (faithfulness, grounding) across cache‑backed,
retrieval‑backed, hybrid, and distributed baselines — under one protocol. It exists because no
existing tool measures both the *systems* and the *quality* sides of cache‑aware serving at once.

**Dissertation title:** *Quantifying Distributed KV‑Cache Efficiency against Information‑Retrieval
Quality for LLM Cache‑Augmented Generation Models under HPC Workloads.*

**The central question.** How should cache‑aware LLM serving be evaluated when efficiency (latency,
TTFT, throughput) must be weighed jointly against answer quality (faithfulness, grounding,
relevance) — and, once that evaluation moves to distributed nodes under heavy KV‑cache memory
pressure, does quality hold or fall as efficiency is pushed to its limit?

---

## 2. Core concepts (the mental model)

- **KV cache** = the Transformer's "short‑term notes": stored key/value tensors for processed tokens
  so they aren't recomputed. It is both the accelerator of inference and its dominant memory cost.
  *Verified anchor:* Llama‑3.1‑70B at 128K context ≈ **40 GB** KV cache — about equal to the Q4
  model weights.
- **Two inference phases:** **prefill** (read the prompt; compute‑bound; measured by **TTFT**) and
  **decode** (generate one token at a time; memory‑bandwidth‑bound; measured by **TPOT**).
- **RAG (Retrieval‑Augmented Generation):** fetch fresh context per query (embed → search → rerank →
  augment → generate). Fresh but pays retrieval cost/error every request.
- **CAG (Cache‑Augmented Generation):** preload stable context once, reuse its KV cache for every
  query. Fast but bounded by context window + KV memory, and stale until recomputed.
- **The trade‑off (the thesis):** caching buys efficiency but costs memory/freshness; retrieval buys
  freshness but costs latency and can hurt grounding. At distributed/HPC scale the KV cache must be
  compressed, evicted, or transferred — and the open question is what that costs in answer quality.

---

## 3. The framework (architecture, baselines, metrics)

**Layered design:** workload → orchestration → telemetry → quality → analysis (separation prevents
confounds and keeps it modular).

**Baselines — 9 types** (`src/orchestration/baselines.py`); the standard suite is **7 labeled runs**:
| Baseline | Context source | Reuse policy |
|---|---|---|
| `no_cache` | gold passage | none (recompute) — control |
| `prefix_cache` | gold | native vLLM prefix cache |
| `rag` | retrieved | none |
| `redis` (cold) | retrieved | centralised retrieval cache |
| `hybrid` (cold/warm) | retrieved | prefix cache + retrieval cache |
| `distributed` | gold | multi‑replica prefix‑aware routing |
| `speculative` | — | speculative decoding (own script) |
| `compressed_rag` | retrieved | **LLMLingua‑2 text compression** |
| `compressed_cag` | gold/cached | **FP8 KV / MLA compression** |

**Metrics:**
- *Serving:* latency, **TTFT**, **TPOT**, throughput (QPS, tokens/s), tail **p50/p95**.
- *Quality:* **LettuceDetect** grounding (primary; `KRLabsOrg/lettucedect-base-modernbert-en-v1`,
  trained on RAGTruth) → `grounding_score`, `hallucinated_span_ratio`; **NLI faithfulness**
  (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, fallback `facebook/bart-large-mnli`); **BERTScore**
  (baseline‑rescaled) deliberately retained as a **negative control**; context relevance
  (retriever diagnostic); SQuAD‑style F1/EM. *Null (not a sentinel) when a model is unavailable.*
- *Cache telemetry:* retrieval‑hit rate, prompt‑cached ratio, cache‑hit %, prompt‑token source
  (recomputed / local cache / remote transfer), KV bytes/token, speculative acceptance.

**Datasets:** SQuADv2 (Phase 1), Natural Questions, MuSiQue, HotpotQA, TriviaQA, Qasper, HumanEval,
MBPP, custom HPC‑code (loaders in `src/data/loader.py`, seeded resampling).

**Tech stack:** vLLM (V1 engine; flags `--enable-prefix-caching`, `--enable-prompt-tokens-details`,
`--kv-cache-dtype fp8`, `--speculative-config`, `/metrics`; disaggregated prefill + LMCache planned);
models Qwen3‑4B (Phase 1) → Qwen3‑8B/14B (GPU); retrieval `intfloat/e5-large-v2` + `BAAI/bge-reranker-large`,
top‑k=3, FAISS exact; Redis (retrieval‑artifact cache only); GCP (L4 `g2-standard-8`, A100
`a2-highgpu-1g`, Terraform).

---

## 4. The compression axis (a distinguishing contribution)

A **2×2 of context source × compression**, because the two paradigms compress **different objects**:
| | **CAG (cached)** | **RAG (retrieved)** |
|---|---|---|
| **Full** | `prefix_cache` | `rag` |
| **Compressed** | `compressed_cag` — FP8 (`--kv-cache-dtype fp8`) / MLA | `compressed_rag` — LLMLingua‑2 (keep 0.5 ≈ 2×) |

**Fairness rule:** *match the ratio, not the algorithm* — pin FP8 (~2×) against LLMLingua‑2 keep‑0.5
(~2×); read the 2×2 down (CAG vs RAG) or across (full vs compressed), never on the diagonal
(LLMLingua vs FP8). Metrics: `compression_ratio`, `kv_bytes_per_token`, `transfer_bytes` (Phase 3).
**The gap:** prior CAG‑vs‑RAG compares full context only; prior compression work optimizes one object
and reports systems efficiency, **never crossed with the cache‑vs‑retrieve choice while measuring
grounding** — and compression can silently degrade quality ("selective amnesia", see
`chen2025pitfalls`). Hypothesis: compression is *largely orthogonal* to the CAG>RAG quality ordering.

**`cage-stats`** is a companion "nvtop‑for‑vLLM" telemetry tool (separate repo
`lucasmdocarmo/cage-stats`, pip git dependency) that polls `/metrics` and captures spec‑decode
acceptance, KV‑compression ratio/dtype, prompt‑token source, GPU stats; integrated via
`src/monitoring/vllm_telemetry.py` (`--vllm-telemetry` → `vllm_telemetry.json`).

---

## 5. Three‑phase roadmap

| Phase | Status | Hardware | What it tests |
|---|---|---|---|
| **1 — Local validation** | ✅ done | Apple M4 Pro (vLLM CPU), Qwen3‑4B, SQuADv2, 50q×3 | metric + orchestration integrity (NOT generalisable absolute numbers) |
| **2 — GPU scaling** | proposed | L4 / A100, Qwen3‑8B/14B, 100q×10 | real KV‑memory pressure, eviction, the compression axis, re‑establish quality numbers |
| **3 — Distributed/HPC** | proposed | multi‑node A100 + router, gVNIC/NCCL | **real cross‑node KV transfer** (LMCache/disagg prefill), the efficiency‑vs‑IR‑quality frontier |

---

## 6. Phase‑1 results (PRELIMINARY — frame carefully)

From the article tables (CPU, Qwen3‑4B, SQuADv2). **Systems numbers are robust; quality constants
are preliminary** (metric pipeline since corrected; gold‑vs‑retrieved confound; distributed is
simulated).
- Prefix cache: **−37.4% latency**, **−65.7% TTFT**, identical faithfulness (0.570) vs No Cache.
- RAG: **+70.4% latency**, **−11.6% faithfulness** (0.570→0.504).
- Distributed baseline: **7.6× p95/p50 TTFT spread** (means are insufficient).
- BERTScore flat (0.324–0.328) across all baselines → validated as a non‑discriminative control.

---

## 7. Research questions & hypotheses (final, forward‑looking)

**RQ1** joint metric suite · **RQ2** reuse vs retrieval (serving+faithfulness) · **RQ3** distributed KV
efficiency vs IR quality under memory pressure · **RQ4** telemetry to attribute behaviour (reuse /
retrieval / transfer) · **RQ5** does compressing each paradigm's native object cut cost without
re‑ordering the CAG‑vs‑RAG quality ranking, and where does aggressive compression start costing
faithfulness?

**Hypotheses** (each grounded in a CAGE component; *Evaluated in* a phase — none asserted as proven):
- **H1** local prompt‑cache reuse (not retrieval success) drives TTFT — *prompt‑cache telemetry* — Phases 1–2.
- **H2** retrieval lowers faithfulness when gold context exists; strict NLI catches it, soft‑matching doesn't — *NLI + BERTScore control* — Phases 1–2.
- **H3** distributed reuse efficiency is bounded by transfer + cold‑start → quantifiable efficiency‑quality frontier — *distributed baseline + transfer telemetry* — Phase 3.
- **H4** near‑lossless KV compression (FP8/MLA) holds faithfulness + delays saturation; aggressive eviction degrades it — *`compressed_cag` + KV bytes/token* — Phases 2–3.
- **H5** prompt compression cuts RAG cost but not its grounding gap; compression is orthogonal to the CAG>RAG ordering — *`compressed_rag`* — Phase 2.
- **H6** tail (p95), not mean, governs cache‑aware UX; cold replicas dominate the tail — *percentile reporting* — Phases 1–3.

---

## 8. Threats to validity (own these in any write‑up)

1. **CPU‑only Phase 1** → absolute latency not generalisable; read orderings only.
2. **"Distributed" baseline is simulated** (HTTP routing + modelled transfer cost); real cross‑node KV
   transfer is a Phase‑3 target.
3. **Gold‑vs‑retrieved confound** (CAG arms get the oracle passage; RAG gets retrieved) — context
   source is confounded with reuse policy; to be isolated with `--context-source`.
4. **Preliminary quality metrics** — faithfulness/BERTScore implementations were since corrected;
   LettuceDetect adopted as primary; numbers re‑established in Phase 2.

---

## 9. Verified citation set (use for further research; full BibTeX in `paper/article-references.bib`)

> All web‑verified (title + arXiv/venue + authors). Misnomer keys preserved for `\cite` compatibility
> are flagged. ⚠️ = key name ≠ first author.

**RAG / CAG core & eval frameworks:** `lewis2020retrieval` (RAG, NeurIPS'20) · `yu2024dontdorag`
(Don't Do RAG — Chan et al., WWW'25, 2412.15605) · `chen2024turborag` (TurboRAG — Lu et al.,
EMNLP'25) · `li2024selfroute` (RAG vs long‑context — Z. Li et al., EMNLP'24) · `espejel2023ragas`
(RAGAS, EACL'24) · `ares2024` (ARES, NAACL'24) · `li2024ragbench` (RAGBench, 2407.11005) · `bergen2024`
(BERGEN, Findings EMNLP'24) · `lazaridou2023freshllms` ⚠️ (FreshLLMs — first author **Vu**, 2310.03214).

**KV cache, serving & surveys:** `kwon2023efficient` (vLLM/PagedAttention, SOSP'23) · `li2024kvsurvey`
(KV mgmt survey, 2412.19442) · `paul2024kvreview` ⚠️ (KV review — first author **Shi**, 2407.18003) ·
`zhong2024distserve` (DistServe, OSDI'24, 2401.09670) · `qin2024mooncake` (Mooncake, 2407.00079).

**KV compression (the axis):** `kvcompresssurvey` (Gao et al., 2503.24000) · `javidnia2025kvccompress`
(Key,Value,Compress — IEEE CICC'25, 2503.11816) · `deepseekv2` (MLA, 2405.04434) · `kvtuner` (FP8
mixed‑precision, 2502.04420) · `chen2025pitfalls` (**Pitfalls of KV Cache Compression** — the
quality‑gap anchor, 2510.00231) · `ge2023h2o` ⚠️ (H2O eviction — first author **Zhang**, 2306.14048) ·
`xiao2023streamingllm` (StreamingLLM, 2309.17453) · `dettmers2022llmint8` (LLM.int8(), 2208.07339) ·
`frantar2022gptq` (GPTQ, 2210.17323) · `hashevict2024` (2412.16187) · `kvzip2025` (2505.23416).

**Prompt compression:** `llmlingua2` (LLMLingua‑2, Findings ACL'24, 2403.12968).

**Attention / architecture:** `shazeer2019mqa` ⚠️ (MQA — titled *Fast Transformer Decoding*, 1911.02150) ·
`ainslie2023gqa` (GQA, EMNLP'23, 2305.13245) · `beltagy2020longformer` (Longformer/SWA, 2004.05150) ·
`child2019generating` (Sparse Transformers, 1904.10509 — *not* SWA) · `megatron-lm` (tensor parallelism,
1909.08053).

**Distributed / offloading / reuse:** `infiniteLLM2024` (DistAttention/DistKV‑LLM, 2401.02669) ·
`cachedattention2024` (multi‑turn KV reuse, ATC'24) · `lee2024infinigen` (InfiniGen offloading, OSDI'24) ·
`lmcache2024` ⚠️ (LMCache, year is **2025**, 2510.09665) · `cacheblend2025` (CacheBlend, EuroSys'25,
2405.16444) · `preserve2025` (HBM→L2 prefetch, 2501.08192) · `hicache2025` (SGLang HiCache feature, docs) ·
`vllm2024disagg` (vLLM disaggregated‑prefill docs).

**Hallucination / quality:** `zhang2024siren` (hallucination survey, Findings EMNLP'24) ·
`phillips2024seven` (Seven Failure Points of RAG, CAIN'24) · RAGTruth & LettuceDetect & ModernBERT
(detection stack — see ANNOTATIONS.md).

**Datasets:** SQuADv2 (Rajpurkar 2018), Natural Questions (Kwiatkowski 2019), MuSiQue (Trivedi 2022).

---

## 10. Repository & paper artifacts (where everything lives)

**Code (`/Users/lucasmariano/CAGE`):** `src/orchestration/baselines.py` (9 baselines),
`src/orchestration/compression.py` (LLMLingua), `src/evaluation/quality.py` (metrics),
`src/monitoring/vllm_telemetry.py` (cage‑stats bridge), `src/data/loader.py`, `scripts/run_experiment.py`
(the runner), `scripts/run_phase{1..5}.sh`, `scripts/cloud_run.sh`, `scripts/manage_vllm_server.sh`,
`terraform/gcp/`. Docs live in `docs/` and authoritative cloud docs in `cloud_docs/`
(RUNBOOK, CLOUD_CONSOLE_GUIDE, KNOWLEDGE_BASE, FEATURE_MAP, VALIDATION_AND_SOTA_REVIEW).

**Paper (`paper/`, git‑ignored):**
- `updated-my-article.tex` — the published SBC article (Phase‑1).
- `dissertation/en/` (Dissertation.tex, 7‑ch `texto/`; **Dissertation‑extended.tex**, 9‑ch `texto-2/`),
  `dissertation/pt-br/` (Dissertacao.tex) — two full proposal‑stage dissertations.
- `my-cap1.tex` — the user's current intro chapter (detailed, with the distributed‑KV literature).
- `CAGE_Technical_Companion.tex` — the deep technical companion (inference, vLLM, RAG‑vs‑CAG, compression).
- `INTRODUCTION_OPTIONS.{md,tex}` — pattern‑matched intro options + RQ1‑5/H1‑6 + mapping table.
- `article-references.bib` — **the verified BibTeX (43 entries)**, ABNT/PUC format.
- `docs/ANNOTATIONS.md` — research notebook (KV‑reduction landscape, verified primary refs).

**Formatting law:** `/Users/lucasmariano/CAGE-Formats` = the PUC Minas ABNT template
(`abnt_pucmg_utf8.cls`). Match it for any `.tex`: `\cite{}` at sentence end / `\citeonline{}` in prose;
figures/tables need the `\captionfont{...\\Fonte: ...}` line; `compactitem` with `\item[a)]`;
caption ABOVE tables; BibTeX `.bib` flow (not manual `thebibliography`).

---

## 11. Open items / next steps for research

- **Run Phase 2** (GPU): re‑establish quality numbers under corrected metrics; exercise the
  compression 2×2 and `speculative` under real memory pressure.
- **Run Phase 3** (distributed): real cross‑node KV transfer (LMCache/disagg prefill); measure the
  efficiency‑vs‑IR‑quality frontier and the transfer break‑even point.
- **Decisions pending:** booktabs vs the template's `\hline\hline`+`|` table style; whether to fold
  RQ5 + the compression paragraph into the dissertation `cap1` (EN+PT) and add the 5 compression
  citations to its `referencias.tex`; confirm `wendersonIJCAE` exact title/DOI (self‑citation).
- **Citation hygiene:** misnomer keys (⚠️ above) are correct entries with mislabeled keys — rename if
  desired. `child2019generating` is now an uncited (valid) Sparse‑Transformers entry.

---

## 12. How to use this file for further research

Good next research prompts inside a Claude Project seeded with this file:
- "Design the Phase‑2 GPU experiment matrix that tests H4 and H5 with statistical power."
- "Survey 2024–2026 work on KV‑cache compression *quality* (not just throughput) and position CAGE."
- "Draft the §Methodology chapter from the baselines/metrics/compression‑axis above (ABNT style)."
- "Stress‑test the threats to validity (§8) as an adversarial examiner would."
Always keep the integrity rule: cite verified sources, mark Phase‑2/3 as proposed, never fabricate.
