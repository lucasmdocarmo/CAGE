# KV-Cache-Store Baseline - Reality Check, Survey, and Go/No-Go

**Question answered:** does CAGE need a KV-cache-database baseline *in addition to* its current Redis baseline, and if so which one? The user named "ConDb" and "TableCache" as candidates.
**Verdict:** yes, add exactly **one** real KV-block store: **LMCache (with CacheBlend)**. "ConDb"/"TableCache" are the wrong category (see 1).
**Status:** research + recommendation only (no code added). Compiled 2026-07-02; every system carries a source URL.

---

## 0. Why this matters for CAGE's thesis

CAGE claims a joint *serving-efficiency + answer-grounding* lens. Its current `redis` baseline, however, caches only **retrieval artifacts** (query -> doc-ids); its own module docstring states "This repo does NOT store raw vLLM KV-cache blocks in Redis" (`src/orchestration/redis_cache.py`). So CAGE never caches the thing that actually dominates serving cost: the **KV blocks / prefill**. A reviewer will note that CAGE asserts a serving-cache angle while never caching the KV state. One KV-block-store baseline closes that gap and makes the "serving" half of the joint axis real.

---

## 1. "ConDb" / "TableCache" reality check (REFUTED for purpose)

Both names resolve to real systems, but **neither is a mainstream KV-cache store** usable as a serving-cache baseline.

- **ConDB (VectifyAI)** - real, wrong category. A tree-structured, reasoning-based **context/retrieval database** (SQLite-backed) that replaces vector search with LLM tree search and caches intermediate tree-search results (up to ~70% token reduction). It does **not** store vLLM KV blocks and does **not** integrate with vLLM. It is brand-new/immature (v1.0 released 2026-05-27, ~40 GitHub stars) and depends on Anthropic/OpenAI APIs at runtime. This is a retrieval-layer artifact, close to what CAGE's Redis already does. https://github.com/VectifyAI/ConDB
- **TableCache** - real, narrow. Paper "TableCache: Primary-Foreign-Key-Guided KV Cache Precomputation for Low-Latency Text-to-SQL" (submitted 2026-01-13). Precomputes per-table KV caches offline, matched via a "Table Trie" (up to 3.62x TTFT speedup). A domain-specific Text-to-SQL KV-precompute technique, not a general serving-cache system or off-the-shelf DB. https://arxiv.org/abs/2601.08743

**What was most likely meant:** for "a caching-database that stores KV blocks," the intended systems are **Mooncake Store** (a distributed KVCache storage engine, the closest thing to a "KV-cache database") and/or **LMCache** (the KV-cache storage/offload layer). A secondary read of "ConDb" is **CacheGen** (encodes KV caches to a bitstream on disk).

---

## 2. The real options and what each actually caches

| System | What it caches | vLLM integration | Effort (master's) | URL |
|---|---|---|---|---|
| **LMCache** | **KV blocks** from GPU memory; tiered across CPU DRAM / SSD / Redis-Valkey / Mooncake / S3 / NIXL | First-class; `LMCacheConnector` via `--kv-transfer-config`; in the vLLM production stack | **Low** - config-driven | https://github.com/LMCache/LMCache |
| **Mooncake (Store + Transfer Engine)** | **KV blocks** in a distributed store over DRAM/SSD; RDMA/TCP/NVMe-oF transfer | Official since 2024-12-16; connector / tier under LMCache | **Medium** - closest to a real "KV-cache DB"; RDMA shines multi-node | https://github.com/kvcache-ai/Mooncake |
| **CacheGen** | **KV blocks compressed to a bitstream on disk**, shareable across vLLM instances (SIGCOMM'24) | Ships inside LMCache | Low-Medium | https://arxiv.org/abs/2310.07240 |
| **CacheBlend** | **KV blocks for non-prefix RAG chunks** + selective cross-attention recompute (EuroSys'25) | Ships inside LMCache + vLLM production stack | Low-Medium | https://arxiv.org/html/2405.16444v3 |
| **GPTCache** | **Full LLM responses**, keyed by query embedding (semantic cache) | App-layer wrapper; NOT a vLLM KV backend | Low, but different axis | https://github.com/zilliztech/gptcache |
| **Redis / Valkey** | Generic KV; as a KV-block backend it is a **tier under LMCache**. In CAGE today it caches retrieval doc-ids only | Only via LMCache as a remote tier | Trivial (already running) | https://github.com/LMCache/LMCache |
| **vLLM prefix cache + KVConnector (NixlConnector)** | **KV blocks**: in-GPU automatic prefix cache + cross-process/RDMA transfer (NIXL over UCX/RoCE/IB) | **Native, built-in** | **Lowest** - the Phase-3 disaggregation path | https://docs.vllm.ai/en/stable/features/nixl_connector_usage/ |
| **CachedAttention / Pcache** | KV reuse across multi-turn conversations (ATC'24 research) | Research prototype, not a drop-in DB | High | https://www.usenix.org/conference/atc24/presentation/gao-bin-cost |
| **ChunkAttention** | Prefix-aware in-memory KV sharing via prefix tree + custom kernel | Research kernel, not a store | High | https://arxiv.org/abs/2402.15220 |

**Key distinction for the thesis:** GPTCache caches **responses/embeddings** (a semantic hit skips inference entirely); LMCache/Mooncake/CacheGen/CacheBlend/NIXL cache **KV blocks** (skip prefill recompute). CAGE's current Redis caches **neither** - only retrieval doc-ids.

---

## 3. Go/No-Go recommendation

**Add exactly one KV-block store: LMCache.**
- **Defensible / standard.** ~10k stars, de-facto KV-cache layer, first-class vLLM connector, MLSys visibility, CoreWeave adoption. (Star counts and "de-facto standard" are the project's own self-description; directionally reliable.) https://github.com/LMCache/LMCache
- **Distinct from Redis on the right axis.** It caches **KV blocks** (prefill reuse), genuinely different from CAGE's retrieval-doc-id Redis, and it can *use the existing Redis as its remote tier* - a clean narrative: "Redis-as-retrieval-cache" (current `redis` baseline) vs "Redis-as-KV-tier-under-LMCache" (new baseline).
- **Feasible for a master's.** Config-driven via `--kv-transfer-config` / `LMCacheConnector`; no kernel work.
- **Subsumes the compression/quality question.** CacheBlend and CacheGen ship *inside* LMCache, giving a compressed-KV variant and a RAG-chunk-fusion variant for free, each feeding the serving-vs-grounding trade-off with published (self-reported) numbers to reproduce or contest.

**Do NOT add Mooncake as a separate baseline** unless Phase 3 goes multi-node with real RDMA. It overlaps LMCache (it becomes a tier under it); its value is cross-node bandwidth, relevant only with the A3-Ultra RoCE setup. If Phase 3 lands, Mooncake / NixlConnector become the *disaggregation transport*, not a second cache DB.

**GPTCache is optional / orthogonal:** add only for an explicit **response-cache** contrast (a semantic hit skips inference). It does not substitute for a KV-block baseline, and it is where the staleness / false-hit risk (see `STALENESS_BASELINE_DESIGN.md`) is sharpest.

### Suggested CAGE integration (when scheduled)
- New family `lmcache` (or `kv_store`): a warm-KV baseline where prefill KV is served from LMCache instead of recomputed. Measure the same CAGE joint suite; the expected story is a TTFT/prefill win at (ideally) preserved grounding, with CacheBlend as the quality-preserving KV-reuse variant and CacheGen as the compressed-KV variant.
- This pairs naturally with `compressed_cag` (server-side fp8 KV) and the `distributed` arm (NIXL transport) already in the taxonomy.

---

## 4. Caveats to carry into the docs and dissertation
- "ConDb"/"TableCache" as KV-cache-store baselines are not verifiable; the real systems by those names are a context-retrieval DB and a Text-to-SQL precompute paper. Do not cite them as KV-store baselines.
- Vendor speedup figures (LMCache "up to 15x"; CacheBlend "3x TTFT / 3x throughput, F1 preserved" on 2WikiMQA / Llama-70B / A40; CacheGen "3.5-4.3x size, 3.2-3.7x delay") are **self-reported** on specific datasets. Treat as claims to reproduce, not established fact.
- Bib keys to mirror when this lands in the manuscript: `lmcache2024` (LMCache; citable paper is 2025), `cacheblend2025` (CacheBlend), `cachegen2024` (CacheGen), `qin2024mooncake` (Mooncake), `kwon2023efficient` (vLLM prefix cache).

## 5. References (all URLs preserved)
- ConDB (retrieval DB, NOT KV-store): https://github.com/VectifyAI/ConDB
- TableCache (Text-to-SQL KV precompute): https://arxiv.org/abs/2601.08743
- LMCache: https://github.com/LMCache/LMCache
- Mooncake: https://github.com/kvcache-ai/Mooncake
- CacheGen (SIGCOMM'24, DOI 10.1145/3651890.3672274): https://arxiv.org/abs/2310.07240
- CacheBlend (EuroSys'25): https://arxiv.org/html/2405.16444v3 ; blog: https://blog.lmcache.ai/en/2025/03/31/cacheblend/
- GPTCache: https://github.com/zilliztech/gptcache
- vLLM NixlConnector: https://docs.vllm.ai/en/stable/features/nixl_connector_usage/
- CachedAttention (ATC'24): https://www.usenix.org/conference/atc24/presentation/gao-bin-cost
- ChunkAttention: https://arxiv.org/abs/2402.15220
- CAGE Redis (local, retrieval-artifact cache): `src/orchestration/redis_cache.py`
