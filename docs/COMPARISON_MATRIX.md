# CAGE — Expanded Comparison Matrix & Novelty Positioning

> Builds out the 5-row table in `paper/my-article.tex` (lines 104–111) into a defensible,
> multi-dimensional comparison against the broader 2023–2025 literature, and states
> explicitly **what CAGE delivers that is new**. Use for Related Work + the comparison table.
> Pairs with [`RELATED_WORK_COMPRESSION.md`](RELATED_WORK_COMPRESSION.md) (compression depth)
> and [`DEV_BACKLOG.md`](DEV_BACKLOG.md) (what must be built to back the claims).
>
> ⚠️ Verify venues/arXiv IDs before camera-ready. "CAGE✦" = planned/target capability not yet
> in code (compression axis, real cross-node transfer) — keep that distinction honest in the paper.

---

## 1. Your paper's current matrix (baseline)

`my-article.tex` compares **5** systems on **4** axes (Semantic Quality · Systems Metrics ·
Cache Reuse · Distributed): RAGAS, RAGBench, TurboRAG, DistServe, Self-Route. CAGE = Yes×4.

**Gaps a reviewer will see:** (a) no compression work cited at all; (b) missing the closest
KV-reuse *systems* (RAGCache, CacheBlend, CacheGen, Mooncake is cited in text but not in the
table); (c) missing newer *eval* frameworks (RAGChecker, RAGTruth) that already do fine-grained
/ span-level quality — your "we measure quality" claim must distinguish from them; (d) only 4
axes, so CAGE's true differentiator (joint quality+systems+reuse+distributed+compression under
one workload, with significance testing) isn't visible.

## 2. Expanded matrix (8 axes)

Axes: **Q**=semantic quality eval · **Sys**=serving metrics (TTFT/latency/thru) · **Reuse**=KV/cache
reuse · **Dist**=multi-node/distributed · **Cmp**=compression evaluated · **Hall**=span-level
hallucination/grounding · **Unif**=unified multi-baseline framework (not a point method) ·
**Stat**=statistical significance reported. (✓ yes · ◐ partial · ✗ no)

| System (cite) | Type | Q | Sys | Reuse | Dist | Cmp | Hall | Unif | Stat |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| RAGAS `espejel2023ragas` | eval | ✓ | ✗ | ✗ | ✗ | ✗ | ◐ | ◐ | ✗ |
| ARES `ares2024` | eval | ✓ | ✗ | ✗ | ✗ | ✗ | ◐ | ◐ | ✗ |
| BERGEN `bergen2024` | eval | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ◐ |
| RAGBench `li2024ragbench` | eval | ✓ | ✗ | ✗ | ✗ | ✗ | ◐ | ✓ | ✗ |
| **RAGChecker** `ragchecker2024` | eval | ✓ | ✗ | ✗ | ✗ | ✗ | ◐ | ✓ | ✓ |
| **RAGTruth** `ragtruth2024` | eval/data | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| **CRUD-RAG** `crudrag2024` | eval | ✓ | ◐ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| vLLM/PagedAttn `kwon2023efficient` | system | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| DistServe `zhong2024distserve` | system | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Mooncake `qin2024mooncake` | system | ✗ | ✓ | ✓ | ✓ | ◐ | ✗ | ✗ | ✗ |
| TurboRAG `chen2024turborag` | system | ◐ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **RAGCache** `ragcache2024` | system | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| **CacheBlend** `cacheblend2025` | system | ◐ | ✓ | ✓ | ✗ | ◐ | ✗ | ✗ | ✗ |
| **CacheGen** `cachegen2024` | system | ✗ | ✓ | ✓ | ◐ | ✓ | ✗ | ✗ | ✗ |
| Self-Route `li2024selfroute` | method | ◐ | ◐ | ✗ | ✗ | ✗ | ✗ | ◐ | ✗ |
| **LongLLMLingua** `longllmlingua2024` | compress | ◐ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| **RECOMP** `recomp2024` | compress | ✓ | ◐ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| **SnapKV/H2O** `snapkv2024,h2o2023` | compress | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| CAG `yu2024dontdorag` | method | ◐ | ◐ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **CAGE (this work)** | framework | ✓ | ✓ | ✓ | ◐ | ✗ | ✓ | ✓ | ✓ |
| **CAGE✦ (target)** | framework | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Reading: **eval frameworks** own the Q column but are empty on Sys/Reuse/Dist/Cmp. **Systems**
own Sys/Reuse/Dist but are empty on Q. **Compression** work owns Cmp but is a point method, no
unified eval. **No single row except CAGE spans Q + Sys + Reuse + Hall + Unif + Stat** — and with
the planned axis, + Dist + Cmp. That empty-everywhere-but-CAGE pattern *is* the contribution.

## 3. New references to add (grouped; not currently in your .bib)

**RAG evaluation / hallucination (sharpen "we measure quality" vs. them):**
- **RAGChecker** — NeurIPS 2024 D&B; fine-grained retrieval+generation diagnostics, best human-correlation. *CAGE differs:* adds serving/cache telemetry; RAGChecker is quality-only. [OpenReview](https://openreview.net/forum?id=J9oefdGUuM)
- **RAGTruth** — ACL 2024; ~18k span-level hallucination annotations. *CAGE differs:* uses span-level grounding (LettuceDetect, trained on RAGTruth) **as a metric inside a systems benchmark**. [arXiv 2401.00396](https://arxiv.org/abs/2401.00396)
- **CRUD-RAG** — 2024; CRUD task taxonomy. *CAGE differs:* cache-policy axis, not task taxonomy. [arXiv 2401.17043](https://arxiv.org/abs/2401.17043)
- **LettuceDetect** — 2025; ModernBERT span hallucination detector. *Your primary grounding metric.* [arXiv 2502.17125](https://arxiv.org/abs/2502.17125)
- **Lost in the Middle** (Liu et al.) — TACL 2024; position sensitivity. *Motivates* why compression/ordering matters for both RAG and CAG. [TACL](https://aclanthology.org/2024.tacl-1.9/)

**KV-reuse / cache serving systems (your closest prior art — must engage):**
- **RAGCache** — multilevel KV cache across GPU/host for RAG. [arXiv 2404.12457](https://arxiv.org/pdf/2404.12457)
- **CacheBlend** — EuroSys 2025; cached-KV fusion + selective recompute (fixes cross-attention loss). [arXiv 2405.16444](https://arxiv.org/abs/2405.16444)
- **CacheGen** — SIGCOMM 2024; KV-cache compression+streaming, 3.5–4.3× smaller, 3.2–3.7× faster load. [arXiv 2310.07240](https://arxiv.org/abs/2310.07240)
- **Pensieve** — cross-request conversation-state KV reuse, 1.14–3.0× vLLM throughput. [arXiv 2312.05516](https://arxiv.org/abs/2312.05516)
- **LMCache** — the production KV-transfer layer for vLLM (your Phase-3 real-transfer path). [repo](https://github.com/LMCache/LMCache)

**Compression methods (the new axis — see RELATED_WORK_COMPRESSION.md for full table):**
- Text: **LongLLMLingua** (ACL'24), **RECOMP** (ICLR'24), **LLMLingua-2** (ACL'24 Findings), **xRAG** (NeurIPS'24), **CompAct** (EMNLP'24).
- KV: **H2O** (NeurIPS'23), **SnapKV** (NeurIPS'24), **StreamingLLM** (ICLR'24), **KVQuant** (NeurIPS'24), **Scissorhands** (NeurIPS'23), **MLA/DeepSeek-V2** (2024).

## 4. What CAGE delivers that is new (the explicit novelty statement)

State it as the **intersection**, not any single capability:

1. **The only framework that jointly evaluates serving behavior AND semantic quality across multiple cache-aware baselines under one common workload.** Eval frameworks (RAGAS/ARES/RAGBench/RAGChecker/RAGTruth) measure quality but ignore TTFT/locality/reuse; serving systems (vLLM/DistServe/Mooncake/TurboRAG/RAGCache/CacheBlend/CacheGen) optimize latency but never score faithfulness. CAGE is the bridge.
2. **A unified 7-baseline taxonomy** (no-cache → prefix-cache → RAG → Redis → hybrid cold/warm → distributed) that lets cache *policies* be compared on equal footing — none of the cited point-methods do this.
3. **Span-level grounding as a first-class systems metric** (LettuceDetect/RAGTruth lineage) wired next to TTFT/throughput — quality degradation from retrieval is measured, not assumed.
4. **Statistical rigor** (per-query Wilcoxon + Holm + bootstrap CIs via `statistical_tests.py`) — most cited works report point numbers.
5. **(Planned, CAGE✦) A compression axis** that makes "CAG vs RAG" a 2×2 (cache/retrieve × full/compressed) and connects text compression (LongLLMLingua) and KV compression (SnapKV/MLA/CacheGen) into the *same* evaluation — no prior work compares both layers under one quality+systems lens.
6. **(Planned, CAGE✦) Real cross-node KV transfer cost** (LMCache) measured against recompute — turning the simulated distributed result into an empirical one.

> Honesty guardrail for the paper: axes Cmp and Dist are **target** capabilities (simulated /
> unbuilt today). Either present them as the framework's design + roadmap, or complete the
> DEV_BACKLOG P0/P1 items before claiming them as results.

## 5. BibTeX additions (verify before camera-ready)

```bibtex
@inproceedings{ragchecker2024, title={RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation}, author={Ru, Dongyu and others}, booktitle={NeurIPS Datasets and Benchmarks}, year={2024}}
@inproceedings{ragtruth2024, title={RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models}, author={Niu, Cheng and others}, booktitle={ACL}, year={2024}, note={arXiv:2401.00396}}
@article{crudrag2024, title={CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models}, author={Lyu, Yuanjie and others}, year={2024}, note={arXiv:2401.17043}}
@article{ragcache2024, title={RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation}, author={Jin, Chao and others}, year={2024}, note={arXiv:2404.12457}}
@inproceedings{cacheblend2025, title={CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion}, author={Yao, Jiayi and others}, booktitle={EuroSys}, year={2025}, note={arXiv:2405.16444}}
@inproceedings{cachegen2024, title={CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving}, author={Liu, Yuhan and others}, booktitle={ACM SIGCOMM}, year={2024}, note={arXiv:2310.07240}}
@article{pensieve2023, title={Pensieve: Retrospect-then-Compare Mitigates Visual Hallucination / cross-request KV reuse}, author={Yu, Lingfeng and others}, year={2023}, note={arXiv:2312.05516}}
@article{lettucedetect2025, title={LettuceDetect: A Hallucination Detection Framework for RAG Applications}, author={Kovacs, Adam and others}, year={2025}, note={arXiv:2502.17125}}
@article{liu2024lostmiddle, title={Lost in the Middle: How Language Models Use Long Contexts}, author={Liu, Nelson F. and others}, journal={TACL}, volume={12}, year={2024}}
@inproceedings{snapkv2024, title={SnapKV: LLM Knows What You are Looking for Before Generation}, author={Li, Yuhong and others}, booktitle={NeurIPS}, year={2024}, note={arXiv:2404.14469}}
@inproceedings{h2o2023, title={H2O: Heavy-Hitter Oracle for Efficient Generative Inference}, author={Zhang, Zhenyu and others}, booktitle={NeurIPS}, year={2023}, note={arXiv:2306.14048}}
@inproceedings{streamingllm2024, title={Efficient Streaming Language Models with Attention Sinks}, author={Xiao, Guangxuan and others}, booktitle={ICLR}, year={2024}, note={arXiv:2309.17453}}
@inproceedings{recomp2024, title={RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation}, author={Xu, Fangyuan and Shi, Weijia and Choi, Eunsol}, booktitle={ICLR}, year={2024}, note={arXiv:2310.04408}}
@inproceedings{longllmlingua2024, title={LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression}, author={Jiang, Huiqiang and others}, booktitle={ACL}, year={2024}, note={arXiv:2310.06839}}
```
(See RELATED_WORK_COMPRESSION.md for the remaining compression bibtex.)
