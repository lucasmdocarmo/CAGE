# Staleness / Freshness Baseline - Design and Implementation Plan

**Status:** WIRED (serving path implemented via on-the-fly gold-evidence redaction; run a live 5-query smoke pass before a full sweep).
**Scope:** a new CAGE baseline family, `staleness`, for Phase 2 onwards.
**Compiled:** 2026-07-02. Grounded in the current code and in cited prior art (see references).

---

## 1. Why this baseline exists (the missing quadrant)

CAGE's nine families vary cache *presence* and *warmth*: `no_cache` -> `prefix_cache` -> `redis` -> `hybrid` (cold/warm) -> `distributed`. **None of them varies cache *age*.** Every serving-efficiency system treats a cache hit as free once its match condition is satisfied; every correctness paper fixes false hits but reports the trade-off on its own axis. No existing framework places a single controlled "how stale is the reused entry" knob on CAGE's joint axis (serving win vs grounding loss) alongside a family of cache-warmth baselines measured on identical footing.

That is exactly CAGE's declared differentiator: a unified taxonomy comparing cache *policies* on equal footing, co-measuring serving efficiency and answer grounding on the same requests. The staleness baseline adds the axis the taxonomy is missing and directly serves the dissertation's central question ("does that quality uphold, or fall as efficiency is pushed to its limit?") and RQ2 (effect of cache reuse on serving and faithfulness).

**One-line framing:** existing arms answer "does caching help?"; the staleness arm answers "**what does a cheap cache hit cost when the cached thing is out of date?**"

---

## 2. Prior art (what exists, and the gap it leaves)

### 2.1 Semantic-cache staleness and the false-hit dial
Semantic caching reuses a prior answer when a new query is embedding-similar to a cached one, trading a cosine threshold for cost. The core hazard is the **false hit**: a semantically-adjacent but not equivalent query is served the wrong cached answer. Lowering the threshold raises hit rate but injects false positives; raising it protects accuracy but collapses hit rate. Per-entry TTL bounds staleness but "is insufficient for rapidly changing data."
- GPTCache (Bang, NLP-OSS 2023): https://aclanthology.org/2023.nlposs-1.24/ . Note: GPTCache's 0.8 similarity default comes from its library docs/implementation, not the ACL paper.
- GPT Semantic Cache (Regmi and Pun): threshold swept 0.6-0.9; below 0.8 raises hit rate but "introduces irrelevant matches decreasing accuracy": https://arxiv.org/html/2411.05276v2
- Threshold/false-hit trade-off and TTL insufficiency: https://blog.premai.io/semantic-caching-for-llms-how-to-cut-api-bills-by-60-without-hurting-quality/

### 2.2 Bounding the error rate (correctness-constrained view)
- **vCache** frames a user-specified maximum error rate delta and guarantees `Pr(vCache(x) = r(x)) >= 1 - delta`, learning per-embedding thresholds. Reported: up to 26x lower error at matched latency, up to 12.5x higher hit rate at matched error, and roughly 57% hit rate while keeping error below 0.5% on SemCacheLMArena: https://arxiv.org/html/2502.03771
- **Closing the Calibration Gap in Semantic Caching** defines the **P-CHR curve** (precision vs cache-hit-ratio as the score threshold varies) and P-CHR-AUC. As the threshold falls, hit ratio rises but precision falls: https://arxiv.org/html/2606.19719v1

### 2.3 Reuse of stale retrieved context / stale KV (the closest anchor)
- **GroundedCache - "Grounded Cache Routing for RAG: When Is It Safe to Reuse an Answer?"** (S. H. Shah, Duke, 26 May 2026) is the primary anchor. It caches answers and gates reuse with four checks, including **G3 Version Match** and **G4 Evidence Support**, and defines the exact metrics CAGE needs (USR, aHR, FH, Stale-Hit). Reported: naive semantic caching reaches **USR 15.5-35.0% on HotpotQA and 26.0-51.5% on mtRAG**; gating drove USR toward 0 at only 1.04-1.07x no-cache latency. This proves stale reuse is a measurable, large grounding cost: https://arxiv.org/abs/2605.27494
- **KV-reuse quality cost:** reusing precomputed KV for non-prefix or changed chunks "sacrifices quality due to missing inter-chunk attention"; CacheBlend must selectively recompute to recover it. Even at the KV layer, stale/approximate reuse has a documented quality penalty: https://arxiv.org/html/2405.16444v3

### 2.4 RAG index freshness and TTL background
Staleness arises when source updates do not propagate to the index; a pipeline can score high faithfulness yet be wrong because the index is stale. Fast-changing-fact benchmarks (FreshQA / FreshLLMs, Findings of ACL 2024) categorise questions as never/slow/fast-changing/false-premise: https://arxiv.org/abs/2310.03214 . Provider prompt caches use 5-min to 1-hr time-based eviction; learned semantic-aware eviction is explored by Fang et al. (learned token-importance eviction for prefix caches): https://arxiv.org/pdf/2605.18825

---

## 3. Design

### 3.1 Independent variable (the knob)
`stale_fraction ∈ {0.0, 0.25, 0.5, 0.75, 1.0}` - the fraction of served cache hits deliberately bound to an **outdated version** of the gold evidence. This operationalises GroundedCache's Version Match (G3) failure axis as a continuous dial, and reframes the calibration-gap / P-CHR threshold sweep as *age* rather than *similarity*. A **TTL sweep** (`cache_ttl_seconds`) is available as a temporal narrative, but `stale_fraction` is the cleaner primary IV (deterministic, no wall-clock dependence).

### 3.2 Making an entry stale deterministically (reusing the existing corpus)
- **v1 (fresh)** = the gold passage / current top-k chunks for the query.
- **v0 (stale)** = a prior version: the passage with the answer span perturbed or redacted, an older paragraph from the same document, or a sibling passage that no longer contains the answer.
- Tag each cached entry `evidence_version ∈ {v0, v1}` and `version_ts`. For `stale_fraction = p`, a seeded policy serves `v0` for exactly `p` of the served hits.

This reuses `RetrievalCache` keying (`src/orchestration/redis_cache.py`); staleness is just serving the `v0` hits for the chosen fraction. `RetrievalCache.set()` already accepts `ttl_seconds`, so the TTL variant needs no new cache API.

### 3.3 Held constant (clean axis)
Model, decoding (T=0 greedy), dataset, query set and order, `top_k`, embedding and reranker models. **The cache is warm in every cell** (population identical to `hybrid_warm`); only the age/validity of served entries changes. **Hit rate is held roughly constant** across the sweep (all cells serve from cache at the same rate; only the fresh/stale mix changes). This is what makes the arm distinct from vCache-style curves, which vary the *threshold* and thus co-vary hit rate with error. CAGE holds hit rate fixed and varies *age* - a genuinely new cut.

### 3.4 Measured (the joint axis)
- **Serving side (win):** TTFT, TPOT, end-to-end latency, QPS, prompt-cached ratio, retrieval-cache rate. Expectation: roughly flat across `stale_fraction` (a stale hit is served as cheaply as a fresh hit - the whole trap). That flatness *is* the finding.
- **Grounding side (loss):** CAGE primary metrics (LettuceDetect grounding / hallucination; NLI faithfulness; EM/F1), plus three staleness-specific metrics ported from GroundedCache and implemented in `src/evaluation/staleness.py`:
  - **USR (Unsafe-Served Rate)** = fraction of *all* queries served a wrong-because-stale answer.
  - **FH (False-Hit Rate)** = error rate *conditional on a cache hit* (FH = USR / aHR).
  - **SHR (Stale-Hit Rate)** = fraction of served `v0` hits producing an ungrounded answer (CAGE's controlled analogue of GroundedCache's Stale-Hit).
- **Headline artifact:** a **Staleness-Cost curve** - grounding / faithfulness (and USR) on Y vs `stale_fraction` (or TTL) on X - overlaid with the flat serving-win line. This is CAGE's own P-CHR / vCache-style trade-off plot, on the freshness axis it uniquely owns.

### 3.5 Distinctness from existing arms
| Existing arm | Varies | New staleness arm |
|---|---|---|
| `prefix_cache` | KV present vs absent | age of reused entry, KV held on |
| `redis` (cold) | retrieval-cache empty vs full | retrieval-cache full but *outdated* |
| `hybrid_warm` | cache warmed vs cold | cache warm but a controlled *fraction is stale* |
| `distributed` | routing / placement | orthogonal - staleness composes with any |

---

## 4. What has landed (this scaffold)

1. **`BaselineType.STALE = "staleness"`** in `src/orchestration/baselines.py`.
2. **New `BaselineConfig` fields** (serialised in `to_dict()`): `stale_fraction: float = 0.0`, `cache_ttl_seconds: Optional[int] = None`, `stale_evidence_mode: str = "version"`, `evidence_version_field: str = "evidence_version"`.
3. **Preset `"staleness"`** in `get_baseline_config` - clones the warm hybrid serving path (warm, `use_faiss=True`, `enable_prefix_caching=True`) and sets the staleness fields; `metadata={"scaffold": True, "serving_path_wired": False}`.
4. **Metrics module** `src/evaluation/staleness.py` - `unsafe_served_rate`, `answer_hit_rate`, `false_hit_rate`, `stale_hit_rate`, `staleness_metrics` (pure stdlib, unit-tested).
5. **CLI registration + loud guard** in `scripts/run_experiment.py` - `staleness` is a `--baseline` choice, and the runner raises a clear `NotImplementedError` pointing here (so it cannot silently run as a plain warm hybrid and misreport).

## 5. Implementation (WIRED) and remaining validation

The serving path is now implemented directly in `scripts/run_experiment.py` (gated to the staleness baseline): v0 (stale) evidence is generated ON THE FLY from the gold context by redacting the answer span (`src/evaluation/staleness.make_stale_context`), `select_stale` picks v0/v1 deterministically per query to hit `stale_fraction`, and USR/FH/SHR are aggregated into the results JSON. `stale_fraction` is swept via the `CAGE_STALE_FRACTION` env var, e.g. `for f in 0 0.25 0.5 0.75 1.0; do CAGE_STALE_FRACTION=$f python3 scripts/run_experiment.py --baseline staleness --num-queries 500 --num-trials 3 ...; done`. This replaces the originally-planned separate-corpus approach below; a live 5-query smoke run is the remaining validation step.

### Originally-planned steps (superseded by the on-the-fly approach above)
1. **`StaleServingPolicy`** - a seeded selector that, per query, serves `v0` vs `v1` to hit the target `stale_fraction` deterministically. Slot it into the retrieval/serving step in `run_experiment.py` where `used_contexts` is chosen.
2. **Cache-layer fields** - extend `RetrievalCache` entries with `{evidence_version, version_ts}` (or use the existing `ttl_seconds` for TTL mode).
3. **Corpus prep** - an offline step (`scripts/extract_qa_evidence.py --emit-stale`) to emit `v0` (stale) alongside `v1` (fresh) evidence per query.
4. **Per-row fields** - record `served_from_cache`, `grounded`, `evidence_version` per query (the run loop already has served/grounded signals) and feed `staleness_metrics` in the aggregation step.
5. **Sweep + remove the guard** - `for f in 0 0.25 0.5 0.75 1.0: run_experiment.py --baseline staleness --stale-fraction $f --num-trials 3 --num-queries <N>` (the locked Phase-2 count is `<N>` = 500, three trials).

## 6. Threats to validity to pre-empt
- **Confound with context-source quality** - here it is *controlled* (v0/v1 are the same document family), a strength not a leak.
- **Synthetic staleness is not real drift** - optionally validate one cell against a FreshQA-style fast-changing-fact set (https://arxiv.org/abs/2310.03214) to show the synthetic knob tracks real temporal drift.

## 7. References
- GroundedCache (USR/aHR/FH/SH; primary anchor): https://arxiv.org/abs/2605.27494
- vCache (error-rate-bounded caching): https://arxiv.org/html/2502.03771
- Closing the Calibration Gap / P-CHR: https://arxiv.org/html/2606.19719v1
- GPTCache: https://aclanthology.org/2023.nlposs-1.24/ ; GPT Semantic Cache: https://arxiv.org/html/2411.05276v2
- CacheBlend (stale-KV quality cost): https://arxiv.org/html/2405.16444v3
- FreshLLMs / FreshQA (Findings of ACL 2024): https://arxiv.org/abs/2310.03214
- Learned semantic-aware eviction (Fang et al.): https://arxiv.org/pdf/2605.18825
- Semantic-cache false hits + TTL insufficiency: https://blog.premai.io/semantic-caching-for-llms-how-to-cut-api-bills-by-60-without-hurting-quality/

*Dissertation crediting: cite alongside `chen2025pitfalls` (KV-compression quality cost), `li2025scbench` (KV-cache lifecycle), and the RAG-eval lineage (`espejel2023ragas`, `ares2024`). The staleness arm is the freshness complement to those compression and reuse axes.*
