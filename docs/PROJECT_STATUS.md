> <!-- CAGE-DOC-STATUS -->
> **⚠️ STATUS: PARTIALLY STALE (2026-06-09).** Useful background, but some commands,
> CLI flags, paths, and metric numbers here predate the June 2026 fixes. In
> particular, `run_experiment.py` has **no** `--phase`/`--all-baselines`/`--trials`/`--queries`
> flags (use `--baseline`, `--num-trials`, `--num-queries`), and pre-fix metric numbers
> (faithfulness 0.570, BERTScore 0.324) are now obsolete. For runnable commands use
> [`RUNBOOK.md`](RUNBOOK.md); for current metrics see [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md).

# CAGE Framework — Project Status

**Last Updated:** 2026-04-08
**Current Phase:** Phase 1 Complete, Phase 2 Pending (requires GPU)

---

## Executive Summary

CAGE has completed Phase 1: local CPU validation of 7 caching baselines on SQuADv2 using Qwen3-4B. All experiment data is verified and a 12-page paper draft (SBC format) is ready. The next step is Phase 2: GPU validation and model scaling.

---

## Phase 1 — COMPLETE

- **Model:** Qwen/Qwen3-4B (vLLM CPU backend)
- **Dataset:** SQuAD v2 (validation split)
- **Hardware:** Apple M4 Pro, 24 GB unified memory
- **Scope:** 50 queries × 3 trials × 7 baselines = 1,050 measured queries
- **Date:** March 19, 2026
- **Results:** `analysis/phase1/results/*/aggregated_metrics.json`
- **Plots:** `analysis/phase1/images/` (16 numbered + composite)
- **Paper:** `analysis/Articles/main-12page.tex`

**Key findings:**
- Prefix Cache: −37.4% latency, −65.7% TTFT, +58.3% QPS vs No Cache, zero quality loss
- RAG: +70.4% latency, +171.4% TTFT, −11.6% faithfulness (worst baseline)
- Distributed: 7.6× p95/p50 TTFT spread (tail latency from cold replicas)
- BERTScore: non-discriminative (0.324–0.328 range)
- Prompt-cached ratio: Prefix Cache 68.4%, Hybrid Warm 89.2%

## Phase 2 — PENDING (Next)

- **Requires:** GPU access (NVIDIA A100 or L4)
- **Tasks:** GPU rerun, model scaling (8B/14B), 10+ trials, additional datasets, statistical tests
- **Code status:** Ready — `scripts/run_experiment.py` works unchanged with GPU-backed vLLM

## Phase 3 — FUTURE

- **Requires:** Multi-node GCP deployment
- **Tasks:** Real KV transfer, disaggregated prefilling, speculative decoding, eviction policies

---

## Artifact Locations

| Artifact | Path |
|---|---|
| Experiment results | `analysis/phase1/results/` |
| Publication plots | `analysis/phase1/images/` |
| Paper source | `analysis/Articles/main-12page.tex` |
| Paper images | `analysis/Articles/Publish/images/` |
| FAISS indexes | `experiments/ir_index/` |
| Experiment logs | `logs/` |
| Core framework | `src/` |
| Config files | `configs/` |
| Tests | `tests/` |
