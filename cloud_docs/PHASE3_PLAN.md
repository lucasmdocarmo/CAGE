# Phase 3 - Multi-Node HPC Cluster: Plan & Definition of Done

> Phase 3 takes CAGE to the regime the title is about: the KV cache as a distributed object across
> several GPU nodes. It builds on Phase 2 ([`PHASE2_CHECKLIST.md`](PHASE2_CHECKLIST.md)). Be precise
> about what is **real now** versus what is the **Phase-3 implementation**, because the dissertation
> states this honestly and the defense depends on it.

## Architecture options and the Phase-3 goal (annotated)

**Validated goal.** Phase 3 takes the CAG KV cache (context preloaded into the KV, no retrieval) and distributes it across nodes, then measures serving AND semantic/answer quality. Two clarifications shape the design:

1. **A correctly transferred KV is lossless, so "quality of the distributed response" alone is a null result.** Moving the exact KV tensors across nodes over RDMA and continuing to decode yields token-for-token identical output to single-node (under greedy). Quality only changes when a system makes a LOSSY coping choice: KV compression (FP8/MLA) or context truncation are lossy; preemption-plus-recompute is lossless but slow. The quality FINDING must therefore be a comparison against a single node UNDER PRESSURE that is forced to degrade. The publishable statement is "distributing the KV preserves the quality that a pressured single node would have had to sacrifice, at a measured transfer cost." Measure the quality distribution SAVES, not the quality it loses.
2. **"Break its memory across nodes" has three concrete meanings** with very different feasibility: transfer the whole KV between nodes; shard one context's KV across nodes; or tier/offload KV to other nodes under pressure. This fork picks the architecture.

**The options.**

| Option | What distributes | Feasibility (vLLM today) | Real memory pressure | Quality result |
|---|---|---|---|---|
| 1. Replicated router + N replicas | nothing (KV replicated per node) | trivial (real now) | no | none (lossless, no transfer) |
| 2. Prefill/Decode disaggregation + RDMA (NIXL) | whole-request KV moves node A to node B | supported (NixlConnector) | partial | lossless vs single-node; finding only vs pressured baseline |
| 3. Sharded-context KV (1/N per node, gather on attention) | one context's KV split across nodes | research-grade, likely infeasible in timeline | yes (max) | lossless if correct; the literal goal |
| 4. KV tiering / offload under pressure (LMCache) | KV spills GPU to CPU to remote when full | supported (LMCache storage tier) | yes (direct) | preserves quality vs recompute/truncate |

- **Option 1 (replicated router).** Best: already real, cheap, real prefix-affinity routing. Worst: does NOT break the KV across nodes (it replicates it), so transfer cost is zero by construction and it cannot answer the Phase-3 question. It is the CONTROL, never the contribution. This is the "What is real now" section below.
- **Option 2 (P/D disaggregation + RDMA).** Best: mainstream, supported (the committed Plan B NixlConnector/UCX recipe in `PHASE3_PLANB_HPC_STUDY.md`), real cross-node KV movement over RoCE, a clean measured transfer cost, and the losslessness gate is the T=0 token-for-token check. Worst: it disaggregates prefill from decode, it does NOT shard one context's KV; relief is modest; it only helps if the CAG context fits one node for prefill; quality is lossless so it still needs the pressured-baseline comparison to yield a finding.
- **Option 3 (true sharded-context KV, the literal goal).** Best: the only option that lets a CAG context LARGER than one node's KV exist at all; the strongest "distribution extends the memory envelope" story. Worst: there is NO off-the-shelf vLLM connector for "each node holds 1/N of the KV", and naive sharding forces every decode step to gather remote KV for attention (bandwidth-murderous per token). High risk, most likely to not finish. OUT OF SCOPE for the timeline; state this honestly in the thesis.
- **Option 4 (KV tiering / offload under pressure).** Best: the truest match to "memory pressure breaks the KV out to other tiers/nodes", feasible via LMCache; measures fetch-evicted-KV vs recompute vs truncate (a real lossy-vs-lossless quality and cost story); lowest engineering risk of the "real" options. Worst: "tier" is often CPU RAM or a remote store, not another GPU node, so the HPC/RDMA framing is weaker; remote-fetch latency can dominate; it reads as a caching contribution more than distributed attention.

(A fifth option, multi-node tensor/pipeline parallelism, distributes the MODEL not the KV; its cross-node traffic is all-reduce activations, not KV transfer. Well-trodden, different contribution, not Phase 3.)

**The decision fork: does the CAG context fit one node's KV?**
- **Fits one node:** Option 2 (P/D disaggregation + RDMA) is the right, feasible, publishable choice, and is exactly what the committed Plan B (2x A3 Ultra) build supports.
- **Exceeds one node:** Option 3/4 territory. Do NOT attempt true sharded attention (Option 3) on a master's timeline; use Option 4 (LMCache tiering/offload) as the feasible approximation of "the CAG KV no longer fits, so it spills across the tier/fabric", and measure the spill cost.

**Recommendation.** Option 2 as the MECHANISM, wrapped in a memory-pressure sweep, with the quality result measured against a PRESSURED single-node baseline (compression / eviction / truncation). The pressured baseline must be demonstrably in-regime (vLLM `num_preemptions_total` > 0 and `gpu_cache_usage_perc` near 1.0) so the quality it sacrifices is real, not an undersizing artifact. Feasible on the committed H200 RDMA build, directly tests CAG KV distributed across nodes, and yields a real quality finding rather than a lossless null. Keep Option 1 as the control; Option 4 as a stretch if the context exceeds one node; Option 3 out of scope.

**Open items before locking:** (a) confirm whether the intended CAG context size fits one A3-Ultra node's KV (decides Option 2 vs Option 4); (b) novelty check that "distributed-KV quality-under-pressure vs a pressured single node" is not already published.

## Dataset suite for Phase 3 (datasets are an AXIS, not new phases)

**A phase is a hardware/scale REGIME (CPU -> single-GPU -> multi-node HPC); a dataset is CONTENT run within a regime.** HotpotQA and Qasper answer the SAME question (RQ3) on different content, so they are the Phase-3 dataset axis, not "Phase 4 / Phase 5." Do NOT promote datasets to phases: it inflates the phase count, implies five sequential hardware deployments (five times the cost/timeline), and deepens the already-flagged "undischarged RQ3" validity risk. Run the same baseline matrix across the dataset suite ON the Phase-3 infra.

| Dataset | Role in Phase 3 | Why | Code status |
|---|---|---|---|
| SQuAD v2 | continuity baseline | standard extractive QA; ties Phase 3 to Phases 1-2; ~50% unanswerable = hallucination probe. BUT short passages -> no KV pressure on its own. | run in Phase 1-2 |
| Qasper (`allenai/qasper`) | long-context / pressure / reuse | full scientific papers (~5-8k+ tokens) -> genuine KV-cache memory pressure; multiple questions per paper -> shared-context CAG reuse. **Best single fit for the Phase-3 goal.** | `QasperLoader` in `src/data/loader.py`; NOT yet validated end-to-end |
| HotpotQA (fullwiki) or MuSiQue | RAG-fair / multi-hop | multi-hop, retrieval genuinely helps -> a FAIR RAG-vs-CAG comparison; fixes the SQuAD "retrieval too weak (top-1 0.113), so RAG lost" confound. | `HotpotQALoader` / MuSiQue loader present; NOT yet validated end-to-end |
| CRAG (`yang2024crag`) | RAG-fair / retrieval-quality | Meta / KDD Cup 2024 benchmark; natural-language query + gold answer + retrieved web `search_results` across simple/conditional/comparison/aggregation/multi-hop/false-premise types -> a strong RAG-fairness input for the `rag` and `compressed_rag` arms. | **fully wired**: `CRAGLoader` (`loader.py:677-734`) + `get_loader` registry + `run_experiment.py --dataset` choices + `scripts/download_datasets.py --dataset` (HF path via `CAGE_CRAG_HF_PATH`, default `crag`); smoke-test the chosen mirror's schema before a full run |
| ShareGPT (serving-trace) | production-realism / serving-load | real user<->assistant conversations with highly variable prompt lengths and turn counts; NO extractive gold answer (`no_gold_answer=True`), so it is a serving-workload / KV-pressure trace (TTFT/TPOT/throughput under heterogeneous prompts), NOT a QA quality benchmark. First assistant turn is a similarity-only reference, never gold. | **fully wired**: `ShareGPTLoader` (`loader.py:737-799`) + `get_loader` registry + `run_experiment.py --dataset` choices + `scripts/download_datasets.py --dataset` (HF path via `CAGE_SHAREGPT_HF_PATH`, default `RyokoAI/ShareGPT52K`); smoke-test before a full run |
| RULER (`hsieh2024ruler`) | pressure knob (optional) | synthetic, controllable length 4k-128k -> memory pressure as a DIALED independent variable (ideal for the sweep). | cited; loader NOT wired in `download_datasets.py`/`loader.py` (re-confirmed 2026-07-02, no loader found) |
| SCBench (`li2025scbench`) | cache-centric peer (optional) | shared-context multi-request KV-cache lifecycle; the closest peer benchmark (see the "SCBench Phase-3 comparison" subsection below). | cited; loader NOT wired (re-confirmed 2026-07-02); covered as a comparison PLAN, not an executed loader |

**Recommended Phase-3 datasets:** SQuAD v2 (continuity) + **Qasper** (long-context/pressure/reuse) + **HotpotQA or MuSiQue** (RAG-fair). Optionally add **RULER** as the explicit pressure-sweep knob if a loader is wired.

**Field-standard sets to acknowledge in Related Work even if not run:** LongBench / LongBench v2 (the standard realistic long-context suite), InfiniteBench, BEIR / KILT (retrieval standards), ShareGPT / Azure LLM inference trace (serving-load realism). Naming them signals field awareness.

**Practical gate (validate-infra rule):** only SQuAD has flowed through the full pipeline. Qasper is the trickiest (answers can be extractive / abstractive / yes-no / unanswerable, and the "context" is a whole paper the retrieval corpus must chunk). SMOKE-TEST each new dataset loader (sensible question / gold-context / answer tuples + retrieval corpus build + metric scoring) before any Phase-3 run.

**Genuine future phases (named, NOT executed here):** a real new phase needs a new QUESTION or REGIME, not a new dataset. Candidates for the conclusion's future-work section: (a) a production-realism phase on real request traces. Note that the ShareGPT half of this is no longer a missing capability: the `ShareGPTLoader` is now fully wired (`loader.py:737-799`, CLI + download script), so a ShareGPT serving-trace run is available today; the remaining future-work piece is the Azure LLM inference trace and the production WORKLOAD framing around it. (b) a cross-model generalization phase (does the trade-off hold across model families); (c) an energy / cost-per-token phase. These are future directions, not additional hardware runs in this dissertation.

## SCBench Phase-3 comparison (`li2025scbench`)

**What SCBench is.** *SCBench: A KV Cache-Centric Analysis of Long-Context Methods* (a.k.a. "SharedContextBench"), published at **ICLR 2025** by **Microsoft Corporation and University of Surrey** (Yucheng Li is the University of Surrey author; the remaining authors are Microsoft), arXiv **2412.10319**. It evaluates efficient long-context methods from a **KV-cache-centric perspective across the full KV-cache lifecycle** (generation, compression, retrieval, loading), in scenarios where the context (KV cache) is **shared and reused across multiple requests/turns**. It is the first long-context benchmark to cover single-turn, multi-turn, and multi-request modes, with two shared-context modes (context cached within a session vs. the same context encoded once and reused across separate sessions). It spans 12 tasks in 4 capability categories (string retrieval, semantic retrieval, global information, multi-tasking) over 8 models and 8 method categories (13 concrete methods). Its headline finding is that sub-O(n)-memory methods degrade in multi-turn shared-context settings while O(n)-memory sparse-encoding methods stay robust.

**How SCBench differs from CAGE.** SCBench's unit of analysis is a **long-context / compression method**, scored on task quality (accuracy / Pass@1 / ROUGE) under a memory/compute budget. CAGE's unit is a **serving policy** across the 9-family baseline taxonomy, and its defining move is to co-measure, on the SAME requests, serving metrics (latency, TTFT, TPOT, throughput, p50/p95 tails, cache/retrieval telemetry) TOGETHER with span-level answer grounding (LettuceDetect) plus NLI faithfulness. Two concrete gaps make SCBench complementary rather than overlapping:
- **No grounding metric.** SCBench uses task-level correctness (accuracy / Pass@1 / ROUGE); it has no span-level hallucination / grounding signal. CAGE wires `grounding_score` / `hallucinated_span_ratio` next to TTFT / throughput.
- **No per-method serving latency.** SCBench reports task-quality metrics only, with no per-method end-to-end serving-latency / TTFT tables. It is safe to state SCBench does not publish per-method serving latency, exactly the gap CAGE fills.

**The concrete comparison plan (a DESIGN, not an executed result; no SCBench code was run).**
1. **Adopt SCBench's shared-context workload as an external, reuse-native input.** Load `microsoft/SCBench` from HF and drive CAGE's harness with the shared `context` + `multi_turns` structure so the SAME long context is reused across turns/requests inside CAGE's serving pipeline. (Note: the HF artifact reports 922 rows; the paper describes 931 multi-turn sessions with 4,853 queries. Do NOT conflate the two counts.)
2. **Select a grounding-clean task subset.** Use the semantic-retrieval group (Code.RepoQA, En.QA, Zh.QA) and En.Sum, which are free-text answers over a bounded gold context, what LettuceDetect / NLI grounding needs. Avoid pure string-retrieval tasks (Retr.KV etc.), whose exact-match accuracy carries little grounding signal.
3. **Map SCBench methods onto CAGE baselines.** SCBench KV-dropping / quantization maps onto CAGE `compressed_cag` (FP8/MLA); SCBench KV-loading / retrieval (Quest, CacheBlend) onto CAGE `hybrid` / `distributed`; SCBench "full-cache" onto CAGE `prefix_cache`; add CAGE-only `rag` / `compressed_rag` rows for which SCBench has no equivalent.
4. **Run each cell under CAGE's protocol** (greedy T=0, repeated trials, same-request logging) so every SCBench-derived request emits BOTH CAGE serving metrics AND grounding, in single-turn, multi-turn, and multi-request modes.

The exact claim this supports: on Microsoft's SCBench shared-context, KV-cache-reuse workload (ICLR 2025), CAGE reproduces SCBench's method-level quality ranking AND additionally quantifies the paired serving-efficiency-vs-answer-grounding operating point of each cache policy on the same requests, revealing, for KV-compression and multi-request reuse, a grounding cost that SCBench's quality-only, method-centric lifecycle analysis does not expose. Mirror bib key `li2025scbench`.

## KV-cache-store baseline (see `cloud_docs/RELATED_WORK_KVCACHE_STORES.md`)

CAGE's current `redis` baseline caches only **retrieval artifacts** (query->doc-ids), NOT KV blocks (`src/orchestration/redis_cache.py`), so the serving half of the joint axis is not yet exercised by a real KV-block store. The recommended single addition is **LMCache** (`lmcache2024`): a de-facto-standard KV-cache layer with a first-class vLLM connector (`--kv-transfer-config` / `LMCacheConnector`), config-driven (no kernel work), that caches actual KV blocks (prefill reuse) and can use the existing Redis as a remote tier, giving a clean "Redis-as-retrieval-cache vs Redis-as-KV-tier-under-LMCache" narrative. CacheBlend (`cacheblend2025`) and CacheGen (`cachegen2024`, arXiv **2310.07240**) ship inside LMCache, providing quality-preserving KV-reuse and compressed-KV variants. **Mooncake** (`qin2024mooncake`) and the vLLM **NixlConnector** are the **disaggregation transport only if Phase 3 goes multi-node with real RDMA** (the committed A3-Ultra RoCE build); do NOT add Mooncake as a separate cache baseline, as it becomes a tier under LMCache. See `cloud_docs/RELATED_WORK_KVCACHE_STORES.md` for the full survey and go/no-go.

## What is real now (runs on GCP today)

- A **prefix-aware router** in front of **N vLLM replicas**, each a real engine with its own local
  prefix cache. Routing hashes the tokenized prompt prefix and sends matching requests to the
  replica that already holds those KV blocks. The router forwards blocking and streamed requests
  and reports the serving replica per request.
- The `distributed` baseline runs end-to-end against this cluster (`--baseline distributed` against
  the router). Under the **replicated** policy every node holds the context, so the modeled
  cross-node transfer cost is **zero** - the arm measures real prefix-affinity routing, not transfer.
- Terraform provisions the cluster (router + N GPU replicas + Redis + durable GCS bucket), with the
  high-bandwidth interconnect (GVNIC, MTU 8896) and tensor-parallel options provisioned but not yet
  exercised for real transfer.

## What is the Phase-3 implementation (the future work the dissertation names)

**Real cross-node KV-tensor transfer.** Today `SimulatedKVCacheManager` derives a transfer size and
latency from the cache footprint and the interconnect bandwidth (analytic model; no tensors move).
Phase 3 replaces this with a real vLLM **KV connector**:

- Launch replicas with `--kv-transfer-config` and a connector such as **LMCache** or **NIXL**.
- Exercise a **sharded** context policy (each node holds 1/N of the context) so transfer cost is
  actually paid and measured, instead of zeroed by the replicated policy.
- Read the measured transfer bytes/latency from serving telemetry (cage-stats token-source
  breakdown: recomputed vs cache-hit vs externally transferred), replacing the analytic
  `transfer_bytes_for` estimate with empirical numbers.

This is tracked as `DEV_BACKLOG #6` and is the explicit object of the HPC phase.

## Steps

**0. Provision the cluster.**
- [ ] `terraform -chdir=terraform/gcp apply -var num_replicas=3` (A100 `a2-highgpu-1g` for headroom,
      or L4 for a cheaper routing-only run; keep `preemptible=false` so all replicas stay up together).
- [ ] For real transfer later: `-var nic_type=GVNIC -var network_mtu=8896`.

**1. Bootstrap the driver/router host** with `scripts/setup/setup_gpu_cloud.sh` so the orchestrator,
cage-stats, and telemetry run there too.

**2. Run the distributed baseline (routing, real replicas).**
- [ ] `python scripts/run_experiment.py --baseline distributed --api-base http://<router>:9000 ...`
      plus `sync_results_to_gcs.sh` (see `RUNBOOK.md` Path B).

**3. (Implementation) Wire the real KV connector.**
- [ ] Add `--kv-transfer-config` + LMCache/NIXL to the replica launch (terraform `vllm_extra_args`).
- [ ] Switch `cache_manager.py` from the simulated path to reading connector telemetry.
- [ ] Run the sharded policy and confirm non-zero measured transfer bytes in `vllm_telemetry.json`.

**4. Analyze + teardown** as in Phase 2.

## Definition of Done

- [ ] The `distributed` baseline runs on a real multi-node GCP cluster with prefix-affinity routing,
      and its results join the other eight baselines.
- [ ] Telemetry attributes prompt tokens to recomputed / cache-hit / transferred on the cluster.
- [ ] **Stretch (the HPC contribution):** real cross-node KV transfer is wired (LMCache/NIXL),
      the sharded policy pays a measured transfer cost, and the analytic `transfer_bytes` model is
      validated against the empirical numbers.

> Until the stretch item lands, every cross-node-transfer number in the dissertation must be labeled
> **simulated/analytic**. Do not present simulated transfer as measured.
