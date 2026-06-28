# CAGE → CLI: Feature Research and Design Rationale

> Planning + research document for transforming **CAGE** (Cache-Augmented Generation Evaluation)
> from a Python script orchestrator into a polished, pip-installable **`cage` CLI**. Every gap
> comparison is grounded in 2024–2026 primary sources; each citation was verified and each "gap"
> claim adversarially checked against overstatement (see **Method**). Generated 2026-06-26.

---

## 1. Research objective

CAGE's defensible identity — confirmed by this research — is that it is the **only** framework that
**jointly measures, on one workload, both LLM serving efficiency and answer faithfulness**, and
reports their trade-off with per-query statistics. The serving-systems literature measures
efficiency only; the RAG-evaluation literature measures quality only; long-context benchmarks probe
capability; and no tool *enforces* the joint frontier.

The objective of the CLI transformation is to **operationalize that unique joint-evaluation
capability as a usable, reproducible, and gate-able developer tool** — turning a research
orchestrator into a `cage` command that practitioners can install, configure declaratively, run
across baselines, export to publication-ready artifacts, and wire into CI as a regression contract
on *both* axes. This (a) strengthens the practical/engineering contribution of the dissertation and
(b) converts CAGE's measured trade-off curves into a build-blocking guardrail no competitor offers.

**Method.** The backbone was produced by a deep-research run (fan-out web search + 3-vote
adversarial verification, 25/25 claims confirmed) and a per-area citation-verification pass (one
verifier per gap area confirming each paper's title/venue/year and checking each gap claim for
overstatement). Verified citations appear in §8; honest nuances the verifiers surfaced are stated
inline as **Note:** under each comparison, so the arguments are defensible rather than promotional.

---

## 2. Positioning: the verified gap

The single strongest finding (3-0 adversarial vote) is that **every serving/KV-cache system CAGE
benchmarks measures efficiency only and never scores answer quality**, while the RAG-evaluation
frameworks do the reverse. CAGE sits in the empty intersection: it is the framework that holds both
axes on the same queries and the same hardware, and adds a per-query Wilcoxon + Holm + bootstrap
layer so the trade-off is reported with calibrated significance. The CLI should reinforce exactly
this identity at every layer (config, metrics, reports, and CI gates).

---

## 3. Gap-by-gap comparison (why each change benefits CAGE)

### 3.1 Serving / KV-cache systems — efficiency without faithfulness

A mature line of serving-systems research has driven down the cost of LLM inference by attacking the
KV cache: PagedAttention eliminates KV-cache fragmentation to raise throughput (Kwon et al., 2023),
RadixAttention reuses prefix caches across requests to cut prefill and first-token latency (Zheng
et al., 2024), Mooncake disaggregates prefill and decode around a KV-cache-centric scheduler to meet
latency SLOs (Qin et al., 2024), and RAG-specific schemes such as CacheBlend (Yao et al., 2025) and
TurboRAG (Chen/Lu et al., 2025) precompute and fuse chunk-level KV caches to slash time-to-first-token.
These systems are evaluated almost entirely on efficiency (TTFT, TPOT, throughput, tail latency).
Where output quality appears, it is only a parity check that the optimization did not break the
model; none scores whether the served answer is faithful to or grounded in the retrieved evidence,
and none frames its headline speedup as a measured efficiency-versus-faithfulness trade-off. CAGE
closes this gap by running the answer-quality axis (LettuceDetect grounding, claim-level NLI
faithfulness, BERTScore, ROUGE/F1/EM) on the very same workload and hardware that produces the
serving metrics, and couples both axes through a per-query Wilcoxon/Holm/bootstrap layer — converting
an unverified "fast and lossless" assertion into a defensible cost-faithfulness trade-off curve.

**Note (verifier):** the gap is *not* "no quality metric exists." CacheBlend reports F1/ROUGE-L
drift and TurboRAG reports QA accuracy, but **only as lexical-overlap parity checks** vs. full
prefill / naive RAG — neither scores grounding, claim-level entailment, or hallucination, and
neither quantifies the trade-off. (Mooncake: arXiv preprint later in ACM Trans. on Storage, 2025.)

### 3.2 RAG evaluation frameworks — quality with a black-box serving system

A mature line of RAG evaluation frameworks standardized answer-quality measurement without human
gold labels: RAGAS (Es et al., 2024) scores faithfulness, answer relevance, and context
precision/recall via reference-free LLM metrics; ARES (Saad-Falcon et al., 2024) fine-tunes light LM
judges and corrects them with prediction-powered inference; RAGChecker (Ru et al., 2024) decomposes
responses into claims for fine-grained, retriever- and generator-attributable diagnostics. These
frameworks are deliberately scoped to the textual output and **treat the serving system as a black
box**: none reports TTFT, TPOT, throughput, tail latency, or KV-cache reuse, so they cannot say what
a faithfulness gain costs to serve. CAGE closes this gap by instrumenting cache + serving telemetry
on the same workload that yields the quality scores, under one harness and one statistical layer.
Where RAGAS/ARES/RAGChecker answer "is the answer good," CAGE answers "is the answer good, **at what
serving cost, and is that trade-off real**."

**Note (verifier):** the gap holds cleanly. Citation correction — RAGAS's lead author is **Es** (Es
et al.), not "Espejel"; keep the bib key but fix the rendered author. RAGChecker = arXiv:2408.08067,
NeurIPS 2024 Datasets & Benchmarks (not yet in `Main.bib` — add it).

### 3.3 Long-context / KV-cache benchmarks — capability, not the joint frontier

A parallel line of benchmarks stress-tests what long-context and RAG systems *can do* rather than
what they cost. RULER (Hsieh et al., 2024) probes effective context length through multi-hop and
aggregation tasks; SCBench (Li et al., 2025) analyzes the full KV-cache life cycle (generation,
compression, retrieval, loading) across eight method families; CRAG (Yang et al., 2024) measures
factual QA and hallucination over 4,409 questions. Each is a capability/quality probe: RULER and CRAG
report accuracy alone, and SCBench — though explicitly cache-centric — reports task accuracy across
cache strategies and reasons about *asymptotic* memory/compute rather than measured serving cost.
None co-locates empirical efficiency and grounded answer quality on a single frontier. CAGE closes
this by jointly instrumenting serving efficiency and answer quality over the same queries, arranged
along the 9-baseline taxonomy and 2×2 compression axis with a per-query significance layer — turning
"capability vs. cost" into a directly measured trade-off curve rather than two disjoint leaderboards.

**Note (verifier):** SCBench is the **closest prior work** to CAGE's cache axis (it is genuinely
KV-life-cycle-centric), so position CAGE as adding the *measured* efficiency↔faithfulness trade-off
that SCBench's accuracy-and-complexity analysis does not.

### 3.4 Context / KV compression — the unmeasured faithfulness cost

Compression methods reduce long-context serving cost by shrinking the prompt or the cached attention
state: LLMLingua-2 compresses prompts by token classification (Pan et al., 2024); SnapKV evicts
unimportant cache positions (Li et al., 2024); KVTuner searches layer-wise mixed-precision
quantization at "nearly lossless" perplexity (Li et al., 2025); DeepSeek-V2's Multi-head Latent
Attention compresses the KV cache into a latent vector for a ~93% cache reduction (DeepSeek-AI, 2024).
Their shared metric is efficiency, optionally with aggregate accuracy or perplexity, but **none pairs
a swept compression ratio with claim-level NLI faithfulness or retrieval grounding at a matched
operating point** — precisely where the cost of compression is most contested. CAGE resolves this
with its 2×2 compression axis, evaluating each configuration at a matched operating point inside one
harness that jointly records serving efficiency and answer quality, with the per-query layer
certifying whether quality moves significantly as the cache shrinks. This converts compression from
an opaque "nearly lossless" knob into a measured, statistically grounded cost-faithfulness trade-off.

**Note (verifier):** *The Pitfalls of KV Cache Compression* (Chen et al., 2025) is itself a
quality-cost study — it shows compression silently drops instructions / leaks system prompts — but
it measures instruction-following on **IFEval**, not RAG grounding or claim-level faithfulness, and
does not sweep ratio inside a joint efficiency-quality harness. So cite it as **corroborating that
the cost is real and contested**, with CAGE's contribution being the unified, ratio-swept harness.
(KVTuner = ICML 2025.)

### 3.5 Evaluation CI gates — single-axis thresholds, not a joint contract

Modern eval harnesses made regression gating a first-class CI/CD primitive: DeepEval wraps quality
metrics (faithfulness, answer relevancy, hallucination) in pytest-style `assert_test` calls that fail
a build on threshold breach (Confident AI, 2026); promptfoo expresses the same declaratively via
model-graded asserts (Promptfoo, 2026). These are quality-axis gates. CAGE closes the gap by gating
the **joint** frontier in a single run: it co-measures decomposed serving metrics (TTFT, TPOT,
throughput, tail latency, KV-cache reuse) alongside the faithfulness stack across the 9-baseline,
2×2 taxonomy, and fails the build if grounding drops **or** TTFT regresses. Because CAGE wraps each
comparison in a per-query Wilcoxon+Holm+bootstrap layer, its gate is a *statistically defensible*
regression claim, not a bare threshold. This is the dissertation thesis made operational: cost and
faithfulness are a single trade-off to enforce jointly, and `cage` would be the first tool to make
that joint frontier a build-blocking contract.

**Note (verifier):** do **not** claim "no eval tool touches latency" — promptfoo *does* expose a
coarse end-to-end `latency` and `cost` assertion that can co-occur with an `llm-rubric` in one test.
The genuine, defensible gap is **granularity + rigor on both axes at once**: promptfoo's latency is a
single wall-clock number (not decomposed TTFT/TPOT/tail/KV-reuse), neither it nor DeepEval pairs that
with a grounding/claim-NLI stack across a baseline+compression taxonomy, and neither carries a
per-query significance layer. (DeepEval/promptfoo are software, cited by access date — not papers.)

### 3.6 Cost / energy reported in isolation from quality

Inference, not training, dominates the recurring spend of a deployed model: Samsi et al. (2023)
benchmark per-token energy and latency of LLaMA inference across GPUs; Wu et al. (2022) frame AI's
carbon footprint as a first-class constraint. Providers made cost a deployment lever, with
prompt-caching discounts of ~50% (OpenAI, 2024) and up to ~90% (Anthropic, 2024) for reused context
— exactly the KV-cache reuse CAGE already instruments. But these cost/energy accounts are reported
**in isolation from answer quality**: an energy benchmark never says whether the cheaper config still
produces grounded answers, and the grounding/faithfulness benchmarks never say what quality costs in
dollars or watts. CAGE closes this by co-reporting cost + energy per token alongside grounding, NLI
faithfulness, BERTScore, and ROUGE/F1/EM on the same per-query items, with the shared significance
layer answering not just "is it cheaper" but "is it cheaper at a significant loss of faithfulness."

**Note (verifier):** all four citations verified. CAGE's contribution is the **joint co-report**, not
the existence of either metric alone. (Samsi: IEEE HPEC 2023; Wu: MLSys 2022; provider pages are
vendor docs, not papers.)

---

## 4. Competitor feature survey (✓ = adversarially verified)

**RAG / LLM evaluation frameworks**

| Framework | Measures | CLI / UX | Gap for CAGE to exploit |
|---|---|---|---|
| RAGAS | faithfulness, answer relevance, context precision/recall (LLM-judge) | Python lib; no real CLI | no serving/cache axis |
| ARES ✓ | context relevance, faithfulness, answer relevance + PPI confidence intervals | synthetic-data pipeline | quality only |
| RAGChecker ✓ | **claim-level** entailment, retriever/generator-attributable P/R | Python | quality only |
| DeepEval ✓ | 14+ metrics; LLM-judge; hallucination | `deepeval` CLI + GitHub Action + `assert_test` CI gate | no serving/cache |
| promptfoo ✓ | model-graded asserts; side-by-side; coarse latency/cost asserts | declarative YAML + CLI + CI/CD | latency is one wall-clock number; no faithfulness/cache; no stats |
| TruLens / Phoenix | feedback functions / tracing + eval | dashboards | observability, not a benchmark harness |
| BERGEN / RAGBench | reproducible RAG benchmarking / labeled corpus + TRACe | library | quality, no serving |

**Serving / KV-cache systems** — vLLM ✓, SGLang ✓, Mooncake ✓, CacheBlend ✓, TurboRAG ✓, DistServe,
Splitwise, Sarathi-Serve, CacheGen, LMCache, CachedAttention, Prompt Cache, ChunkAttention →
**all efficiency-only** (paged/radix/disaggregated KV, prefix/chunk reuse, KV streaming). None scores
faithfulness or grounding. *This empty space is CAGE's moat.*

**Long-context / cache benchmarks** — RULER ✓ (effective-context probing), SCBench ✓ (KV life-cycle:
gen/compress/retrieve/load), CRAG ✓ (factual QA, varied popularity/dynamism); HELMET, LongBench,
InfiniteBench (long-context suites). All capability/quality; none co-reports measured serving cost.

**Compression** — LLMLingua-2 ✓ (prompt; CAGE's `compressed_rag`), RECOMP, SnapKV ✓ (KV eviction),
KVTuner ✓ (mixed-precision KV quant), DeepSeek MLA ✓ (low-rank latent KV; CAGE's `compressed_cag`).

---

## 5. Feature angles for CAGE (inspiration → CAGE's novel twist)

| Angle | Inspired by | CAGE's novel twist | Effort |
|---|---|---|---|
| **Joint-axis CI regression-gate** | DeepEval / promptfoo (§3.5) | gate on **both** axes — fail if grounding ↓ **or** TTFT ↑, backed by per-query stats | S–M ⭐ |
| Declarative YAML/TOML run config | promptfoo, DeepEval, lm-eval-harness | one config drives the whole 9-baseline + 2×2 sweep | S |
| LLM-as-judge quality lane | RAGAS / ARES (§3.2) | judge as an *extra* quality lane, plotted against latency/cost | M |
| Claim-level precision/recall | RAGChecker (§3.2) | you already compute claim-level NLI — expose per-claim P/R | S |
| PPI confidence intervals | ARES (§3.2) | label-efficient CIs layered on your Wilcoxon/Holm/bootstrap | M |
| Side-by-side multi-model compare | promptfoo | multi-model efficiency+quality Pareto | M |
| Effective-context stress mode | RULER (§3.3) | run the baselines under length stress | M–L |
| KV-life-cycle stress mode | SCBench (§3.3) | the compression axis under gen/compress/retrieve/load | L |
| Cost + energy accounting | CodeCarbon; Samsi et al. (§3.6) | $/1k-tok + energy/token next to grounding | S |

---

## 6. Novel CLI features

**Developer experience.** Pip-installable `cage` entrypoint (Typer/Click); subcommands
`run / sweep / report / compare / gate / dash / repro`; YAML/TOML run configs (replace long flag
strings); a baseline + metric **plugin registry** (entry-points); **multi-format report export**
(HTML + JSON + **LaTeX tables** + Pareto/tail plots → feeds the dissertation directly); run cache +
reproducibility manifest (surface the provenance CAGE already writes via `cage repro <run>`); local
sortable leaderboard; a live/watch dashboard wired to the existing `cage-stats`.

**Evaluation capability.** LLM-as-judge lane; multi-model / multi-backend sweeps; cost ($/1k-tok) +
energy (CodeCarbon) metrics; statistical-power planning (`cage power` — queries needed for a
detectable effect); model-regression testing (gate a new model vs. a baseline run).

---

## 7. Recommended MVP (minimal-but-impactful, in order)

1. **Package as `cage` (Typer) + YAML config.** Turns the scripts into a tool; lowest risk. **[S]**
2. **Multi-format report export (HTML + JSON + LaTeX + Pareto).** Immediate dissertation payoff. **[S–M]**
3. **Joint-axis CI regression-gate.** The standout — no competitor gates efficiency **and** quality
   together, and CAGE's per-query stats make the gate defensible. **[S–M] ⭐**
4. **Cost + energy metrics.** Cheap, and ties straight to the "AI cost" motivation. **[S]**
5. **LLM-judge lane + PPI confidence intervals.** Extra quality signal + tighter intervals. **[M]**

*Defer:* RULER/SCBench stress modes, the plugin registry, the live dashboard.

**Throughline:** every MVP item reinforces CAGE's verified, uncontested identity — the only tool that
benchmarks **and gates** both serving efficiency and answer quality on one workload.

---

## 8. References (verified 2026-06)

Most keys below are already in the dissertation `Main.bib` and reusable; **new** ones to add are
flagged.

- **kwon2023efficient** — Kwon et al. *Efficient Memory Management for LLM Serving with PagedAttention.* SOSP 2023. arXiv:2309.06180.
- **zheng2024sglang** — Zheng et al. *SGLang: Efficient Execution of Structured LM Programs.* NeurIPS 2024. arXiv:2312.07104.
- **qin2024mooncake** — Qin et al. *Mooncake: A KVCache-centric Disaggregated Architecture.* arXiv:2407.00079 (later ACM Trans. Storage 2025).
- **cacheblend2025** — Yao et al. *CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion.* EuroSys 2025. arXiv:2405.16444.
- **chen2024turborag** — Lu et al. *TurboRAG: Accelerating RAG with Precomputed KV Caches for Chunked Text.* EMNLP 2025. arXiv:2410.07590.
- **espejel2023ragas** — Es, James, Espinosa Anke, Schockaert. *RAGAS: Automated Evaluation of RAG.* EACL 2024 (Demos), 150–158. *(fix rendered author to "Es et al.")*
- **ares2024** — Saad-Falcon, Khattab, Potts, Zaharia. *ARES: An Automated Evaluation Framework for RAG.* NAACL 2024, 338–354.
- **ragchecker2024** — Ru, Qiu, Zhang et al. (Amazon AWS AI). *RAGChecker: A Fine-grained Framework for Diagnosing RAG.* NeurIPS 2024 D&B. arXiv:2408.08067. **(NEW — add to Main.bib)**
- **hsieh2024ruler** — Hsieh et al. *RULER: What's the Real Context Size of Your Long-Context LMs?* COLM 2024. arXiv:2404.06654.
- **li2025scbench** — Li et al. (Microsoft). *SCBench: A KV Cache-Centric Analysis of Long-Context Methods.* ICLR 2025. arXiv:2412.10319.
- **yang2024crag** — Yang et al. *CRAG — Comprehensive RAG Benchmark.* NeurIPS 2024 D&B. arXiv:2406.04744.
- **llmlingua2** — Pan et al. *LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression.* Findings of ACL 2024. arXiv:2403.12968.
- **li2024snapkv** — Li et al. *SnapKV: LLM Knows What You Are Looking For Before Generation.* NeurIPS 2024. arXiv:2404.14469.
- **kvtuner** — Li et al. *KVTuner: Sensitivity-Aware Layer-wise Mixed-Precision KV Cache Quantization.* ICML 2025. arXiv:2502.04420.
- **deepseekv2** — DeepSeek-AI. *DeepSeek-V2 (Multi-head Latent Attention).* 2024. arXiv:2405.04434.
- **chen2025pitfalls** — Chen, Geh, Grover, Van den Broeck, Israel. *The Pitfalls of KV Cache Compression.* 2025. arXiv:2510.00231.
- **samsi2023words** — Samsi et al. *From Words to Watts: Benchmarking the Energy Costs of LLM Inference.* IEEE HPEC 2023. arXiv:2310.03003.
- **wu2022sustainable** — Wu et al. *Sustainable AI: Environmental Implications, Challenges and Opportunities.* MLSys 2022. arXiv:2111.00364.
- **openaicache** — OpenAI. *Prompt Caching in the API.* 2024. (vendor docs; ~50% cached-input discount.)
- **anthropiccache** — Anthropic. *Prompt Caching with Claude.* 2024. (vendor docs; up to ~90% on cache reads.)
- **deepeval** — Confident AI. *DeepEval* (open-source software + docs, `assert_test` CI gating). Accessed 2026-06-26. **(NEW — software, cite as @misc)**
- **promptfoo** — Promptfoo. *promptfoo* (open-source software + docs, declarative YAML + model-graded asserts + CI/CD). Accessed 2026-06-26. **(NEW — software, cite as @misc)**
