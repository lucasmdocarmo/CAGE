# Related Work & Comparison Basis — Compression for CAG / RAG

> **Purpose.** Citable evidentiary base + experiment-design backbone for adding a
> **compression axis** to CAGE (the "Option 3 / full axis" direction). Use the tables for
> your Related Work section and the §6–§8 design for Methods. Every entry is a real,
> published 2023–2025 work with overlapping datasets/metrics, so the comparison is
> ground-based, not hand-waved.
>
> ⚠️ **Verify arXiv IDs / venues against the official page before camera-ready** — IDs
> below are best-effort. Status: design note (2026-06-09); experiments gated on the protocol
> fixes in [`VALIDATION_AND_SOTA_REVIEW.md`](VALIDATION_AND_SOTA_REVIEW.md) Part C.

---

## 1. The two layers of "compression" (don't conflate them)

| Layer | Compresses | When it acts | Tools | Primary effect |
|---|---|---|---|---|
| **Text / context** | the **words** of the prompt | before tokenization | LLMLingua, LongLLMLingua, RECOMP, CompAct, xRAG | fewer input tokens → cheaper prefill; mainly a **RAG** optimization |
| **KV-cache** | the **KV tensors** | during/after prefill | SnapKV, H2O, ChunkKV, MLA; KV-reuse: TurboRAG, RAGCache, CacheBlend | smaller/cheaper cache → less eviction, less cross-node transfer; the **heart of CAG** |

CAGE already implements a *form* of the KV-reuse idea (its `prefix_cache`/`hybrid` baselines).
The compression axis makes both layers explicit and measurable.

---

## 2. Table A — Text / context compression for RAG

| Key | Paper | Venue | Method (1-line) | Datasets | Metrics reported | CAGE use |
|---|---|---|---|---|---|---|
| RECOMP | Improving Retrieval-Augmented LMs with Compression and Selective Augmentation | ICLR 2024 | extractive + abstractive compressor → 5–10% of tokens | NQ, TriviaQA, HotpotQA | QA acc, compression ratio, perf drop | **Compare + reuse method** (same QA datasets) |
| LongLLMLingua | Accelerating & Enhancing LLMs in Long Context via Prompt Compression | ACL 2024 | question-aware token-level compression | NaturalQuestions, LongBench, MuSiQue, ZeroSCROLLS, LooGLE | acc, **e2e latency 2.1×**, cost, lost-in-middle +21.4%@4× | **Build on** → `compressed_rag` arm |
| LLMLingua / LLMLingua-2 | Compressing Prompts for Accelerated Inference | EMNLP 2023 / ACL 2024 Findings | small-LM token pruning, up to 20× | GSM8K, BBH, LongBench | acc retention, latency | **Build on** (pip, MIT) |
| xRAG | Extreme Context Compression for RAG with One Token | NeurIPS 2024 | project retrieval embedding → 1 soft token | RAG QA suites | acc, FLOPs | Cite (extreme-compression upper bound) |
| CompAct | Compressing Retrieved Documents Actively for QA | EMNLP 2024 | active iterative doc compression | HotpotQA, 2WikiMQA, MuSiQue | EM/F1, compression | Compare (multi-hop) |

## 3. Table B — KV-cache reuse & compression for RAG (the "CAG-for-RAG" family; **closest prior art**)

| Key | Paper | Venue | Method (1-line) | Why central to CAGE |
|---|---|---|---|---|
| TurboRAG | Accelerating RAG with Precomputed KV Caches for Chunked Text | preprint 2024 (arXiv 2410.07590) | precompute & store chunk KV offline; load KV for prefill | **Literally CAG-for-RAG.** Direct competitor to your `prefix_cache`/`hybrid`; reuse its TTFT methodology |
| RAGCache | Efficient Knowledge Caching for RAG | preprint 2024 (arXiv 2404.12457) | multilevel dynamic KV cache across GPU/host | Prior art for your **distributed / memory-tiering** thesis |
| CacheBlend | Fast LLM Serving for RAG with Cached Knowledge Fusion | EuroSys 2025 (arXiv 2405.16444) | fuse cached chunk-KV, **selectively recompute** to restore cross-attention | Peer-reviewed; fixes the concatenated-chunk cross-attention flaw you'd otherwise be criticized for |

## 4. Table C — KV-cache compression primitives + compressed-attention models

| Key | Paper | Venue | Method | CAGE use |
|---|---|---|---|---|
| H2O | Heavy-Hitter Oracle for Efficient Generative Inference | NeurIPS 2023 (2306.14048) | attention-score KV eviction | `compressed_cag` (KV) candidate |
| SnapKV | LLM Knows What You're Looking for Before Generation | NeurIPS 2024 (2404.14469) | cluster-based KV selection | `compressed_cag` (KV) candidate |
| ChunkKV | Semantic-Preserving KV Cache Compression | preprint 2025 (2502.00299) | chunk-granular KV retention | `compressed_cag` (KV) candidate |
| MLA (DeepSeek-V2) | Multi-head Latent Attention (low-rank KV) | preprint 2024 (2405.04434) | architectural low-rank KV (7–14× smaller) | **MLA model arm** — cheapest cross-node KV transfer (Phase 3) |

## 5. Surveys + the CAG anchor (cite these)

| Key | Paper | Venue | Role |
|---|---|---|---|
| CAG | **Don't Do RAG: When Cache-Augmented Generation is All You Need** (Chan, Chen, Cheng, Huang) | WWW 2025 Companion (doi 10.1145/3701716.3715490) | **Foundational citation** — uses **HotpotQA + SQuAD** (your datasets). Your direct lineage |
| Prompt-Compression Survey | Prompt Compression for LLMs: A Survey (Li et al.) | NAACL 2025 (2410.12388) | taxonomy (hard vs soft prompt) |
| RAG-Compression Survey | Contextual Compression in RAG: A Survey | preprint 2024 (2409.13385) | RAG-specific compression taxonomy |

---

## 6. Mapping to CAGE's compression axis (the 2×2)

Add **one orthogonal dimension** — *compression* — on top of your existing *context source*:

```
                       FULL (no compression)        COMPRESSED
   CAG (cached) ───►  prefix_cache (existing)   compressed_cag   ← KV compression (Table C)
   RAG (retrieved) ─► rag         (existing)    compressed_rag   ← text compression (Table A)
```

**New baselines to implement (Option 3):**
1. **`compressed_rag`** — compress retrieved docs with **LongLLMLingua/LLMLingua-2** (Table A) before the prompt. *Compare against:* RECOMP, LongLLMLingua, CompAct.
2. **`compressed_cag`** — compress the cached context's **KV** via SnapKV/H2O (Table C) **or** use an **MLA model**. *Compare against:* TurboRAG, CacheBlend, RAGCache (KV reuse) + the KV-compression primitives.
3. *(optional bridge)* **`compressed_cag_text`** — text-compress the context, *then* cache its KV (compression → cache). Shows the two layers compose.

> Sub-choice to state explicitly in Methods: `compressed_rag` uses **text** compression;
> `compressed_cag` uses **KV** compression. Keep them labeled so the axis is interpretable.

## 7. New metrics for the compression axis

CAGE already logs TTFT, latency, throughput, EM/F1, grounding/faithfulness. Add:
- **`compression_ratio`** — retained tokens (or bytes) ÷ original. The x-axis of every plot.
- **`kv_cache_bytes` / `kv_bytes_per_token`** — cache footprint (drives eviction + transfer).
- **`transfer_bytes`** (Phase 3) — cross-node KV moved; **directly reduced by KV compression** (this is the lever that makes distributed CAG beat recompute).
- **Pareto curves:** quality (faithfulness / F1) **vs** compression_ratio, and TTFT **vs** compression_ratio — the exact framing RECOMP/LongLLMLingua use, so your plots are directly comparable to theirs.

## 8. Why a ground comparison is feasible (methodology alignment)

- **Datasets overlap:** RECOMP & CompAct (HotpotQA, TriviaQA), LongLLMLingua (NaturalQuestions), CAG (HotpotQA, SQuAD) — all intersect CAGE's set, so you can place CAGE on the **same axes** and reproduce comparable numbers.
- **Metrics overlap:** their headline metrics (accuracy/F1, latency/TTFT, compression ratio) are already in your pipeline. Adding compression is "one new x-axis," not a new measurement stack.
- **Reviewer expectation:** TurboRAG / RAGCache / CacheBlend are the precise prior art for "reuse KV instead of recomputing retrieved context." Engaging them is expected; omitting them is a visible gap.

## 9. Sequencing (critical)

Do **not** run the compression axis until the protocol confounds are fixed (simulated
distributed baseline, gold-vs-retrieved confound, per-trial cache flush, warm-hybrid
leakage; see VALIDATION Part C). Order: **fix protocol → Phase-2 GPU re-run → add compression
axis (Phase 2.5)**. The literature here is ready now for the *paper's design*; the *runs*
come after the foundation is sound.

---

## 10. BibTeX (verify before camera-ready)

```bibtex
@inproceedings{chan2025cag,
  title={Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks},
  author={Chan, Brian J and Chen, Chao-Ting and Cheng, Jui-Hung and Huang, Hen-Hsen},
  booktitle={Companion Proc. ACM Web Conference (WWW)}, year={2025}, doi={10.1145/3701716.3715490}}

@inproceedings{xu2024recomp,
  title={RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation},
  author={Xu, Fangyuan and Shi, Weijia and Choi, Eunsol},
  booktitle={ICLR}, year={2024}, note={arXiv:2310.04408}}

@inproceedings{jiang2024longllmlingua,
  title={LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression},
  author={Jiang, Huiqiang and Wu, Qianhui and Luo, Xufang and Li, Dongsheng and Lin, Chin-Yew and Yang, Yuqing and Qiu, Lili},
  booktitle={ACL}, year={2024}, note={arXiv:2310.06839}}

@inproceedings{jiang2023llmlingua,
  title={LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models},
  author={Jiang, Huiqiang and Wu, Qianhui and Lin, Chin-Yew and Yang, Yuqing and Qiu, Lili},
  booktitle={EMNLP}, year={2023}, note={arXiv:2310.05736}}

@inproceedings{pan2024llmlingua2,
  title={LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression},
  author={Pan, Zhuoshi and Wu, Qianhui and Jiang, Huiqiang and others},
  booktitle={Findings of ACL}, year={2024}, note={arXiv:2403.12968}}

@inproceedings{cheng2024xrag,
  title={xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token},
  author={Cheng, Xin and others}, booktitle={NeurIPS}, year={2024}, note={arXiv:2405.13792}}

@inproceedings{yoon2024compact,
  title={CompAct: Compressing Retrieved Documents Actively for Question Answering},
  author={Yoon, Chanwoong and others}, booktitle={EMNLP}, year={2024}, note={arXiv:2407.09014}}

@article{lu2024turborag,
  title={TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text},
  author={Lu, Songshuo and Wang, Hua and others}, year={2024}, note={arXiv:2410.07590}}

@article{jin2024ragcache,
  title={RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation},
  author={Jin, Chao and others}, year={2024}, note={arXiv:2404.12457}}

@inproceedings{yao2025cacheblend,
  title={CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion},
  author={Yao, Jiayi and others}, booktitle={EuroSys}, year={2025}, note={arXiv:2405.16444}}

@inproceedings{zhang2023h2o,
  title={H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models},
  author={Zhang, Zhenyu and others}, booktitle={NeurIPS}, year={2023}, note={arXiv:2306.14048}}

@inproceedings{li2024snapkv,
  title={SnapKV: LLM Knows What You are Looking for Before Generation},
  author={Li, Yuhong and others}, booktitle={NeurIPS}, year={2024}, note={arXiv:2404.14469}}

@article{deepseekv2_2024,
  title={DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (Multi-head Latent Attention)},
  author={DeepSeek-AI}, year={2024}, note={arXiv:2405.04434}}

@inproceedings{li2025promptcompression_survey,
  title={Prompt Compression for Large Language Models: A Survey},
  author={Li, Zongqian and others}, booktitle={NAACL}, year={2025}, note={arXiv:2410.12388}}
```

---

## Sources (live links)
- [RECOMP](https://arxiv.org/abs/2310.04408) · [LongLLMLingua](https://llmlingua.com/longllmlingua.html) · [LLMLingua repo](https://github.com/microsoft/LLMLingua) · [LLMLingua-2](https://arxiv.org/html/2403.12968v2) · [xRAG](https://arxiv.org/pdf/2405.13792) · [CompAct](https://arxiv.org/pdf/2407.09014)
- [TurboRAG](https://arxiv.org/abs/2410.07590) · [RAGCache](https://arxiv.org/pdf/2404.12457) · [CacheBlend (EuroSys'25)](https://arxiv.org/abs/2405.16444)
- [ChunkKV](https://arxiv.org/pdf/2502.00299) · [Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression)
- [CAG paper (WWW'25)](https://dl.acm.org/doi/10.1145/3701716.3715490) · [CAG code](https://github.com/hhhuang/CAG)
- [Prompt-Compression Survey (NAACL'25)](https://github.com/ZongqianLi/Prompt-Compression-Survey) · [Contextual Compression in RAG: Survey](https://arxiv.org/pdf/2409.13385)
