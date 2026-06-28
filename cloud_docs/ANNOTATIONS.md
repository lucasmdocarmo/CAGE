# CAGE — Research Annotations

> **What this file is:** a working notebook of external reading that may feed CAGE
> (the article, the dissertation, or the project), **kept separate from the paper**.
> Nothing here is committed to the paper until verified and deliberately ported.
> Each entry records: the source, what's useful, the *primary* citation to use
> (never cite a blog in the paper — cite the paper the blog points to), and how it
> maps to CAGE. **Verification status is tracked explicitly.**

---

## Source 1 — Raschka, "Recent Developments in LLM Architectures" (blog)

- **Author / date:** Sebastian Raschka, PhD — *"Recent Developments in LLM
  Architectures: KV Sharing, mHC, and Compressed Attention."*
- **URL:** https://magazine.sebastianraschka.com/p/recent-developments-in-llm-architectures
- **Type:** Substack blog post (secondary source). **Use as a map to primary
  papers; do not cite the blog itself in the dissertation/article.**

### Why it's relevant to CAGE
Its entire subject is **KV-cache reduction in 2025–26 models** — the core of CAGE's
thesis. The article's central claim is a near-verbatim restatement of CAGE's
motivation:

> "KV cache size, memory traffic, and attention cost quickly become the main
> constraints [at serving time]."

→ **Useful as motivation**: independent, current (2026) confirmation that the KV
cache is *the* serving bottleneck, which makes CAGE's "distributed KV efficiency"
framing timely rather than niche. (Paraphrase it; cite the primary technique papers
below, not the blog.)

### Useful data / analysis (with caveats)
The article catalogs KV-reduction families and example models. **Caveat:** the
figures below reached us via a page summarizer and concern bleeding-edge mid-2026
models; the summary even contained an internal inconsistency (it stated both
"27% of FLOPs" and "90% FLOPs reduction" for the same model). **Treat all numbers as
directional motivation only — verify against the primary technical report before
quoting any of them.**

| Technique (per article) | Reported effect | Model | Verify? |
|---|---|---|---|
| Cross-layer KV sharing | "~2.7 GB saved (E2B), ~6 GB (E4B) at 128K ctx" | Gemma 4 | ⚠️ verify vs Gemma 4 report |
| Sequence-dim compression (CSA/HCA) | "~10% of KV cache size at 1M tokens vs V3.2" | DeepSeek V4 | ⚠️ verify vs DeepSeek report |
| Compressed conv. attention (CCA) | compresses Q/K/V in latent space | ZAYA1-8B | ⚠️ verify |
| Sliding-window attention | keeps KV + attention cheaper (local only) | Gemma 4, Mistral | ✅ established (see refs) |
| MLA (per-token latent) | per-token KV compression | DeepSeek V2/V3 | ✅ established (see refs) |
| GQA / MQA (head sharing) | fewer KV heads | standard incl. Qwen3 | ✅ established (see refs) |

**Note on mHC ("multi-residual-stream"):** the article also covers residual-stream
widening (mHC, "~6.7% training overhead"). This is **not directly relevant to CAGE**
(it concerns residual streams, not the KV cache). Skip for now.

---

## The KV-reduction landscape — verified primary citations

This is the family tree the article surveys, with the **real papers** to cite. It
situates CAGE's compression arm (FP8 + MLA) within the broader field. **All entries
below web-verified (title + arXiv ID + venue) on 2026-06-17.**

| Family | What it does to the KV cache | Primary citation | arXiv / venue |
|---|---|---|---|
| **Head sharing — MQA** | one KV head shared across all query heads | Shazeer, *Fast Transformer Decoding* | arXiv:1911.02150 (2019) |
| **Head sharing — GQA** | intermediate # of KV heads (MQA↔MHA) | Ainslie et al., *GQA* | arXiv:2305.13245 — EMNLP 2023 |
| **Per-token latent — MLA** | compress each token's KV into a latent vector | DeepSeek-AI, *DeepSeek-V2* | arXiv:2405.04434 (2024) |
| **Cross-layer sharing — CLA** | later layers reuse earlier layers' KV (≈2× further) | Brandon et al., *Reducing… Cross-Layer Attention* | arXiv:2405.12981 — NeurIPS 2024 |
| **Cross-layer (survey)** | systematic study of cross-layer KV sharing | *A Systematic Study of Cross-Layer KV Sharing* | arXiv:2410.14442 (2024) — authors not yet verified |
| **Local / sliding-window** | attend only to a recent window | Beltagy et al., *Longformer* | arXiv:2004.05150 (2020) |
| **Local + GQA (in a real LLM)** | SWA + GQA together | Jiang et al., *Mistral 7B* | arXiv:2310.06825 (2023) |
| **Precision — FP8 / mixed** | fewer bits per cached value | Li et al., *KVTuner* | arXiv:2502.04420 (2025) |
| **Eviction / sequence-dim** | drop low-value cache entries (H2O, SnapKV) | see surveys → | Gao et al. arXiv:2503.24000 (2025); H. Li et al. arXiv:2412.19442 (2024) |
| **Prompt (text) compression** | shrink the *input text*, not the cache | Jiang et al. *LLMLingua* (arXiv:2310.05736); Pan et al. *LLMLingua-2* (arXiv:2403.12968) | EMNLP 2023 / Findings ACL 2024 |

> **Bleeding-edge (no verified formal paper yet):** DeepSeek V4 CSA/HCA, ZAYA1 CCA,
> Gemma 4 cross-layer specifics — these are 2026 model technical reports. Usable as
> "the field is moving toward sequence-level KV compression" framing, but **do not
> cite as peer-reviewed**; confirm a primary report before any formal use.

---

## How this could feed CAGE (ideas — NOT yet in the paper)

1. **Motivation sentence** (intro/background): add a paraphrase of "KV cache size /
   memory traffic / attention cost are the main serving constraints," cited to the
   KV-cache surveys (Li 2412.19442; Gao 2503.24000), not the blog.
2. **Landscape table** (Related Work / §7 of the companion): a "families of KV
   reduction" table like the verified one above, positioning CAGE's `compressed_cag`
   (FP8) and MLA as two sampled points on a larger map. Strengthens currency.
3. **Baseline-precision honesty note** (methodology/defense): Qwen3 already uses
   **GQA**, so `compressed_cag` (FP8) is an *additional* reduction *on top of*
   standard head-sharing — not a reduction from uncompressed MHA. State this
   explicitly (Ainslie 2305.13245 for GQA).
4. **Future-work baselines**: cross-layer KV sharing (CLA, 2405.12981), sliding-window
   (Longformer 2004.05150 / Mistral 2310.06825), and eviction (H2O/SnapKV via the
   surveys) are concrete additional `compressed_cag`-family arms the framework could
   add later.
5. **Distributed-transfer link**: less KV (via any of these) = less to move between
   nodes in Phase 3 — the article reinforces that architectural KV reduction is now
   mainstream, which supports CAGE's Phase-3 transfer-cost study.

---

## Verification log

- **2026-06-17:** Web-verified MQA (1911.02150), GQA (2305.13245, EMNLP'23), CLA
  (2405.12981, NeurIPS'24), Longformer (2004.05150), Mistral 7B (2310.06825). MLA
  (2405.04434), KVTuner (2502.04420), LLMLingua/-2 (2310.05736 / 2403.12968), KV
  surveys (2412.19442 / 2503.24000) were verified earlier in the project.
- **Unverified / pending:** authors of the cross-layer KV-sharing survey (2410.14442);
  all DeepSeek V4 / Gemma 4 / ZAYA1 numeric claims (2026 technical reports).




CAGE STUFF

Paper	Venue	Method	Datasets	Metrics reported	Use for CAGE
RECOMP (arXiv 2310.04408)	ICLR 2024	Extractive + abstractive compressor of retrieved docs (→5–10% of tokens)	NQ, TriviaQA, HotpotQA	QA acc, compression ratio, perf-drop	Reuse methodology + compare. Same QA datasets as you; the canonical "compress retrieved context" baseline.
LongLLMLingua (ACL 2024)	ACL 2024	Question-aware token-level prompt compression	NaturalQuestions, LongBench, MuSiQue, ZeroSCROLLS, LooGLE	accuracy, end-to-end latency (2.1×), cost, "lost-in-middle" (+21.4% @4×)	Strongest fit. Reports latency and quality vs compression — identical framing to CAGE. Drop-in for a compressed_rag baseline.
LLMLingua / LLMLingua-2 (EMNLP'23 / ACL'24)	EMNLP 2023 / ACL 2024	Small-LM token pruning; up to 20×	GSM8K, BBH, LongBench	acc retention, latency	Task-agnostic compressor; MIT; pip-installable.
xRAG (arXiv 2405.13792)	NeurIPS 2024	Extreme compression → 1 token via modality bridge	RAG QA suites	acc, FLOPs	Upper-bound "extreme compression" reference point.
CompAct (arXiv 2407.09014)	EMNLP 2024	Active document compression for QA	HotpotQA, 2WikiMQA, MuSiQue	EM/F1, compression	Multi-hop comparison.
Prompt Compression: A Survey (NAACL 2025) + Contextual Compression in RAG: A Survey (arXiv 2409.13385)	survey	taxonomy (hard vs soft prompt)	—	—	Cite for the taxonomy in your related-work section.

Paper	Venue	What it does	Why it's central to CAGE
TurboRAG (arXiv 2410.07590)	preprint (Moore Threads)	Precomputes & stores KV caches of chunks offline, retrieves the KV for prefill → eliminates online KV compute, cuts TTFT	This is literally CAG applied to RAG. It's the natural baseline/competitor to your prefix_cache/hybrid arms. Reuse its TTFT methodology verbatim.
RAGCache (arXiv 2404.12457)	preprint	Multilevel dynamic KV caching across GPU/host memory	Your distributed/memory-tiering thesis — RAGCache is the prior art you must position against.
CacheBlend (EuroSys 2025)	EuroSys 2025	Fuses cached KV of multiple chunks, selectively recomputes tokens to restore cross-attention	Directly addresses the flaw your validation review noted (concatenated chunks lose cross-attention). Peer-reviewed systems venue.
CAG — "Don't Do RAG" (Chan et al., WWW 2025, code)	ACM WWW 2025	Preload whole KB into context, reuse KV	Your foundational citation — uses HotpotQA + SQuAD, exactly your datasets. Your direct lineage and the anchor for "CAG vs RAG."

PART 1 — The big picture (what this is and why it exists)
One sentence: CAGE is a measuring framework that fairly compares different ways of feeding knowledge to a large language model (LLM) at serving time — and it measures both how fast/cheap each way is and how truthful the answers are, at the same time.
The problem (the gap): When an LLM answers a question, the answer quality depends not just on the model but on how you give it the supporting information. The standard way today is RAG (Retrieval‑Augmented Generation): for every question, go fetch documents from a search index and stuff them into the prompt. A newer way is CAG (Cache‑Augmented Generation): if the knowledge is stable, load it once, remember it, and reuse that memory for every question — no search.
Why it matters: Everyone benchmarks RAG with quality tools (RAGAS, ARES, RAGBench…), but those tools were built for retrieval. Nobody had a framework that measures the cache‑side behavior — cache hit rates, time‑to‑first‑token, memory reuse — together with answer quality. So you couldn't honestly answer the question "is caching actually better, and what does it cost in truthfulness?" CAGE fills that hole.
The thesis title's promise: Quantifying distributed KV‑cache efficiency against IR quality under HPC. In plain terms: when you scale caching across many GPUs, how much speed/memory do you gain, and what do you pay in answer quality versus just retrieving? That's the one number nobody has measured.

PART 2 — The core concepts (so you can teach anyone in 2 minutes)
The KV cache = the model's short‑term notes. As the model reads text, it writes "notes" (key/value vectors) about every word so it doesn't have to re‑read everything to produce the next word. Those notes are the KV cache.
Analogy: reading a book and jotting margin notes. Without notes, you'd re‑read the whole book to write each sentence of your summary. The notes make it fast — but they take desk space (GPU memory).
Two phases of answering:
* Prefill = reading the question + context (fast, parallel). This is what TTFT (time‑to‑first‑token) measures.
* Decode = writing the answer one word at a time (slow, sequential). This is what TPOT (time‑per‑output‑token) measures.
Why we split them: Caching mostly speeds up prefill (you skip re‑reading known context). If you only report total latency, you hide where the win comes from. So CAGE always separates TTFT from TPOT.
RAG vs CAG, said simply:
* RAG = for every question, run to the library, grab pages, read them fresh, answer. Fresh, but slow, and you might grab the wrong book.
* CAG = load the textbook once, keep your notes, answer everything from the notes. Fast, but the notes take memory and go stale if the textbook changes.
The central trade‑off (your whole thesis in one line): notes (cache) are fast but cost memory and can be stale; library trips (retrieval) are fresh but slow and can be wrong. CAGE measures exactly where one beats the other.

PART 3 — What we actually built
1) A layered framework. Five clean layers so nothing contaminates anything: workload (picks the questions) → orchestration (picks the strategy, keeps conditions fair) → telemetry (records speed + cache signals) → quality (scores truthfulness) → analysis (tables, stats, plots).
Why layered: so you can change the model or dataset without changing how it's scored. Fair comparisons require holding everything else constant.
2) Nine "baselines" (strategies we compare). Think of them as 9 contestants:
* no_cache (recompute everything — the control), prefix_cache (reuse the cache), rag (retrieve), redis (cache the retrieved artifacts), hybrid (retrieve + cache), distributed (route across replicas), speculative (a decode speed‑up), and the two newest: compressed_rag and compressed_cag.
* In practice the standard run is 7 (hybrid runs twice — cold and warm); speculative and the two compression ones are extra.
Why so many: the trick is they vary along two independent axes — where the knowledge comes from (gold/retrieved) and how computation is reused (none/prefix/distributed). Prior work mixes these up; separating them is a real methodological contribution.
3) An honest quality stack. Not one score, but a layered judgment:
* LettuceDetect (primary): a hallucination detector that highlights which spans of the answer aren't supported by the context.
* NLI faithfulness: splits the answer into claims and checks each is entailed by the context.
* BERTScore (negative control on purpose): a surface‑similarity score we expect to be useless here — and showing it's flat is the point (it proves you need the strict metrics).
Why a negative control: it's the scientific move that makes the evaluation credible — we demonstrate why simple similarity metrics miss hallucinations.
4) cage‑stats — "nvtop for vLLM." A telemetry tool we built that watches the serving engine and reports the internal signals normal benchmarks can't see: cache‑hit rate, where each prompt token came from (recomputed / cached / transferred), GPU power, KV‑compression ratio, speculative acceptance.
Why it exists: our #1 Phase‑1 finding — "local reuse, not retrieval success, is what speeds things up" — is only provable because we recorded these signals. Without cage‑stats you'd just see "RAG is slow" without knowing why.
5) The compression axis (recent work). Two ways to make things cheaper, and they attack different objects:
* compressed_rag shrinks the retrieved text (library LLMLingua‑2 — keeps ~50% of tokens).
* compressed_cag shrinks the cache tensors (FP8 quantization, or MLA architecturally).
Why both: prompt compression helps RAG (less fresh text to read each time); KV compression helps CAG (smaller notes = more fit in memory, less to move between machines). Same idea — "spend less" — but applied to whichever side it actually helps.

PART 4 — What we've found so far (Phase 1) — and the honest version
What Phase 1 was: a small, controlled run on a laptop‑class CPU (Qwen3‑4B, SQuAD v2, 7 baselines, 50 questions × 3 trials). The goal was never the numbers — it was to prove the plumbing is correct: that the framework orchestrates fairly and the metrics behave.
What it showed (preliminary):
* Prefix caching was the clear winner: ~37% lower latency, ~66% lower TTFT, identical truthfulness vs the control. → caching is a free lunch when context is stable.
* RAG paid ~70% more latency and ~12% lower faithfulness. → retrieval costs time and can hurt grounding when the gold context was already available.
* The "distributed" contestant had a 7.6× gap between typical and worst‑case TTFT. → averages lie; cold replicas create ugly tails. This is why we report p95, not just means.
The honest caveat you must own in a defense (this is a strength, not a weakness):
"The Phase‑1 absolute quality numbers are preliminary. Our own audit found the two quality metrics were mis‑implemented and the retriever was missing its required prefixes — we've since fixed all of that and adopted LettuceDetect. Also, the 'distributed' baseline is currently a simulated cost model, not real cross‑machine cache transfer. So Phase 1 validates the method; Phase 2 re‑establishes the numbers under the corrected protocol."
Why say this proactively: examiners will find it anyway. Naming your own confounds first turns a vulnerability into evidence of rigor. The framework architecture is sound; the early numbers were a shakedown run.

PART 5 — The roadmap: what each phase tests and why it's the gap
Phase 1 — Local validation (DONE).
* Tests: does the pipeline orchestrate fairly and do metrics work?
* Why important: you can't trust a distributed result if the measuring tape itself is wrong. Establish correctness first.
Phase 2 — Single GPU, real scale (NEXT).
* Tests: re‑run the baselines on real GPUs (NVIDIA L4, then A100) with bigger models (8–14B), more trials, and the compression axis, under genuine memory pressure.
* Why important / the gap: a laptop CPU never fills up GPU memory, so the whole reason CAG is interesting (memory pressure, cache eviction) never appears. Phase 2 is where caching's real costs and benefits show up — and where the corrected metrics produce publishable numbers.
Phase 3 — Distributed / HPC (FUTURE — the headline).
* Tests: a multi‑GPU cluster moving real KV cache between machines (disaggregated prefill + LMCache), measuring the actual transfer cost vs. just retrieving locally.
* Why important / the gap: this is the unmeasured number in the field. Everyone knows distributed serving exists (DistServe, Mooncake, LMCache), but nobody has put distributed cache efficiency on the same chart as answer quality. Phase 3 turns the title into a measured frontier: "at this scale, moving the cache beats retrieving when ___."
The throughline to repeat: each phase only adds realism after the previous one's measurements are trusted. That discipline is the project's spine.

PART 6 — The compression story (because it's your newest piece)
The gap we tackled: Prior CAG‑vs‑RAG comparisons use full context only. Prior compression papers optimize one object and report systems efficiency only — never the quality cost, and never crossed with the cache‑vs‑retrieve choice. No one measured {CAG, RAG} × {full, compressed} with quality attached. We did.
The expected result (pre‑registered, so we can't fudge it later):
* Compressing makes everything cheaper but doesn't change who's more truthful. Compression is orthogonal to the CAG‑vs‑RAG quality gap.
* Practical takeaway: the paradigm (cache vs retrieve) decides quality; compression is just the cost/memory knob within each. And KV compression specifically is what makes distributed CAG affordable at scale.
Why "pre‑registered" matters in a defense: stating the expectation before the GPU runs means the result is a real test, not a story we backfilled. That's a credibility move examiners respect.

PART 7 — Likely hard questions & crisp answers (your prep)
Q: "Isn't CAG just RAG with a cache?"
No. RAG decides what context to fetch per query; CAG fixes the context and reuses its computation. They make opposite bets — freshness vs. reuse. We separate "what context" from "how it's served" precisely so they're not conflated.
Q: "Your distributed results — are those real?"
Today the distributed baseline is a simulated cost model (HTTP routing + modeled transfer); no real tensors move. We say so explicitly. Phase 3 wires real cross‑node transfer via LMCache/disaggregated prefill. That honesty is by design.
Q: "Why are the faithfulness numbers so low (0.57)?"
It's a strict, claim‑level metric on a dataset with unanswerable questions, and a small model that adds filler. The absolute value isn't the point — the relative drop when you switch to retrieval is. And those early constants are preliminary; the metrics were since corrected.
Q: "Why should I trust the quality metric at all?"
We use span‑level hallucination detection (LettuceDetect, trained on RAGTruth) as the primary signal, NLI as backup, and we deliberately include BERTScore as a negative control to show why naive similarity fails. The control is the proof.
Q: "Why run on a CPU first — isn't that meaningless?"
The CPU run isn't for performance numbers; it's to validate the orchestration and metric math cheaply and reproducibly before spending on GPUs. Measure the tape before you measure the room.
Q: "What's the actual contribution?"
A framework that, for the first time, scores cache‑aware serving on speed and truthfulness together, across cache/retrieve/hybrid/distributed and compressed variants — and a plan to quantify the distributed cache‑efficiency‑vs‑quality frontier under HPC.
Q: "What could break your thesis?"
If, at GPU scale, compression closed RAG's quality gap, or if cache transfer were so cheap that distribution were always free. Both are exactly what Phase 2/3 are built to test — falsifiable, not assumed.

ART 8 — The 30‑second version (memorize this)
"LLMs answer better when you feed them the right context. The standard way, RAG, fetches documents for every question — fresh but slow and sometimes wrong. The newer way, CAG, loads stable knowledge once and reuses the model's memory — fast but memory‑hungry and stale‑prone. No tool measured both the speed and the truthfulness of these approaches together, especially at distributed scale. CAGE does. We validated the measurement pipeline locally (Phase 1), we're scaling it to real GPUs with a compression study (Phase 2), and the headline is quantifying — for the first time — when distributed caching beats retrieval, and what it costs in answer quality (Phase 3)."
