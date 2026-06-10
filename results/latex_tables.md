# Phase 1 LaTeX Tables for Publication

This document contains publication-ready LaTeX tables derived from the completed Phase 1 benchmark results. All tables are formatted for direct copy-paste into a LaTeX document.

**Source data:** `analysis/phase1/results/*/aggregated_metrics.json`
**Run date:** March 19-20, 2026
**Model:** Qwen/Qwen3-4B
**Dataset:** SQuAD v2
**Queries per baseline:** 50
**Trials per baseline:** 3

---

## Required LaTeX Packages

Add these to your preamble:

```latex
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{siunitx}
\sisetup{
  round-mode = places,
  round-precision = 2,
  table-format = 5.2
}
```

---

## Table 1: Main Performance Comparison

This is the primary results table showing latency, time-to-first-token (TTFT), and throughput (QPS) across all baselines.

**Explanation:** Lower latency and TTFT are better; higher QPS is better. Values are mean ± standard deviation across 3 trials. The `prefix_cache` baseline shows the strongest performance improvement over the `no_cache` control.

```latex
\begin{table}[htbp]
\centering
\caption{Phase 1 Performance Results: Qwen3-4B on SQuAD v2 (n=50 queries, 3 trials)}
\label{tab:phase1-performance}
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Baseline} & \textbf{Avg Latency (ms)} & \textbf{Avg TTFT (ms)} & \textbf{QPS} \\
\midrule
No Cache (control)           & $16006 \pm 711$   & $6919 \pm 112$   & $0.061 \pm 0.002$ \\
Prefix Cache                 & $10015 \pm 884$   & $2376 \pm 275$   & $0.096 \pm 0.008$ \\
RAG                          & $27270 \pm 283$   & $18775 \pm 144$  & $0.033 \pm 0.001$ \\
Redis Retrieval Cache (cold) & $26853 \pm 39$    & $18579 \pm 37$   & $0.034 \pm 0.000$ \\
Hybrid Cache (cold)          & $15513 \pm 4173$  & $6001 \pm 4290$  & $0.059 \pm 0.012$ \\
Hybrid Cache (warm)          & $13269 \pm 893$   & $2791 \pm 66$    & $0.067 \pm 0.004$ \\
Distributed Replicated       & $18492 \pm 372$   & $5279 \pm 638$   & $0.052 \pm 0.001$ \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 2: Quality Metrics Comparison

Shows answer quality metrics including faithfulness, relevance, and BERTScore completeness.

**Explanation:** Higher values are better for all quality metrics. Faithfulness measures factual correctness relative to the context. Relevance measures how well the answer addresses the question. BERTScore measures semantic similarity to the reference answer.

```latex
\begin{table}[htbp]
\centering
\caption{Phase 1 Quality Results: Qwen3-4B on SQuAD v2}
\label{tab:phase1-quality}
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Baseline} & \textbf{Faithfulness} & \textbf{Relevance} & \textbf{BERTScore} \\
\midrule
No Cache (control)           & $0.570 \pm 0.056$ & $0.505$ & $0.328$ \\
Prefix Cache                 & $0.570 \pm 0.056$ & $0.505$ & $0.328$ \\
RAG                          & $0.504 \pm 0.023$ & $0.525$ & $0.324$ \\
Redis Retrieval Cache (cold) & $0.550 \pm 0.040$ & $0.525$ & $0.324$ \\
Hybrid Cache (cold)          & $0.506 \pm 0.022$ & $0.525$ & $0.324$ \\
Hybrid Cache (warm)          & $0.512 \pm 0.027$ & $0.525$ & $0.324$ \\
Distributed Replicated       & $0.636 \pm 0.078$ & $0.505$ & $0.327$ \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 3: Combined Performance and Quality Table

A comprehensive single table for papers with space constraints.

```latex
\begin{table*}[htbp]
\centering
\caption{Phase 1 Benchmark Results: Qwen3-4B on SQuAD v2 (n=50, 3 trials)}
\label{tab:phase1-combined}
\small
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
\textbf{Baseline} & \textbf{Latency (ms)} & \textbf{TTFT (ms)} & \textbf{QPS} & \textbf{TPS} & \textbf{Faithfulness} & \textbf{BERTScore} \\
\midrule
No Cache           & $16006 \pm 711$  & $6919 \pm 112$  & $0.061$ & $6.06$  & $0.570$ & $0.328$ \\
Prefix Cache       & $10015 \pm 884$  & $2376 \pm 275$  & $0.096$ & $9.59$  & $0.570$ & $0.328$ \\
RAG                & $27270 \pm 283$  & $18775 \pm 144$ & $0.033$ & $3.33$  & $0.504$ & $0.324$ \\
Redis Cache (cold) & $26853 \pm 39$   & $18579 \pm 37$  & $0.034$ & $3.42$  & $0.550$ & $0.324$ \\
Hybrid (cold)      & $15513 \pm 4173$ & $6001 \pm 4290$ & $0.059$ & $5.89$  & $0.506$ & $0.324$ \\
Hybrid (warm)      & $13269 \pm 893$  & $2791 \pm 66$   & $0.067$ & $6.74$  & $0.512$ & $0.324$ \\
Distributed        & $18492 \pm 372$  & $5279 \pm 638$  & $0.052$ & $5.25$  & $0.636$ & $0.327$ \\
\bottomrule
\end{tabular}
\end{table*}
```

---

## Table 4: Relative Improvement vs No Cache Baseline

Shows percentage improvement relative to the `no_cache` control baseline. Negative values indicate improvement for latency/TTFT; positive values indicate improvement for QPS.

**Explanation:** This table quantifies the speedup or slowdown of each caching strategy relative to the uncached baseline. The prefix cache achieves 37% latency reduction and 66% TTFT reduction with no quality degradation.

```latex
\begin{table}[htbp]
\centering
\caption{Relative Performance Change vs No Cache Baseline (\%)}
\label{tab:phase1-relative}
\begin{tabular}{@{}lrrrr@{}}
\toprule
\textbf{Baseline} & \textbf{Latency} & \textbf{TTFT} & \textbf{QPS} & \textbf{Faithfulness} \\
\midrule
Prefix Cache                 & $-37.4\%$ & $-65.7\%$ & $+58.3\%$ & $0.0\%$ \\
RAG                          & $+70.4\%$ & $+171.4\%$ & $-45.1\%$ & $-11.6\%$ \\
Redis Retrieval Cache (cold) & $+67.8\%$ & $+168.5\%$ & $-43.6\%$ & $-3.5\%$ \\
Hybrid Cache (cold)          & $-3.1\%$ & $-13.3\%$ & $-2.9\%$ & $-11.3\%$ \\
Hybrid Cache (warm)          & $-17.1\%$ & $-59.7\%$ & $+11.2\%$ & $-10.2\%$ \\
Distributed Replicated       & $+15.5\%$ & $-23.7\%$ & $-13.4\%$ & $+11.5\%$ \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 5: Pairwise Comparisons

Key pairwise comparisons showing the effect of specific caching mechanisms.

**Explanation:** Each row compares a caching variant against a relevant baseline to isolate the effect of that specific mechanism.

```latex
\begin{table}[htbp]
\centering
\caption{Pairwise Performance Comparisons (\% change)}
\label{tab:phase1-pairwise}
\begin{tabular}{@{}llrrrr@{}}
\toprule
\textbf{Comparison} & \textbf{Effect Measured} & \textbf{Latency} & \textbf{TTFT} & \textbf{QPS} & \textbf{Faith.} \\
\midrule
Prefix Cache vs No Cache & Prompt caching & $-37.4\%$ & $-65.7\%$ & $+58.3\%$ & $0.0\%$ \\
Redis vs RAG & Retrieval caching & $-1.5\%$ & $-1.0\%$ & $+2.6\%$ & $+9.1\%$ \\
Hybrid (warm) vs Hybrid (cold) & Cache warming & $-14.5\%$ & $-53.5\%$ & $+14.5\%$ & $+1.3\%$ \\
Hybrid (cold) vs RAG & Hybrid design & $-43.1\%$ & $-68.0\%$ & $+76.9\%$ & $+0.3\%$ \\
Distributed vs No Cache & Multi-replica routing & $+15.5\%$ & $-23.7\%$ & $-13.4\%$ & $+11.5\%$ \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 6: Cache Telemetry Summary

Shows cache hit rates and prompt caching effectiveness across baselines.

**Explanation:** Retrieval hit rate measures how often relevant documents were found. Retrieval cache rate shows Redis cache effectiveness. Prompt cached ratio shows the fraction of prompt tokens served from the KV cache.

```latex
\begin{table}[htbp]
\centering
\caption{Cache Telemetry Summary}
\label{tab:phase1-cache}
\begin{tabular}{@{}lrrrl@{}}
\toprule
\textbf{Baseline} & \textbf{Retr. Hit} & \textbf{Retr. Cache} & \textbf{Prompt Cached} & \textbf{Notes} \\
\midrule
No Cache           & ---   & ---   & ---    & Control \\
Prefix Cache       & ---   & ---   & $68.4\%$ & Native vLLM prefix cache \\
RAG                & $98\%$ & $0\%$  & ---    & Fresh retrieval \\
Redis Cache (cold) & $98\%$ & $0\%$  & ---    & Cold retrieval cache \\
Hybrid (cold)      & $98\%$ & $0\%$  & $75.6\%$ & Cold retrieval + prefix \\
Hybrid (warm)      & $98\%$ & $100\%$ & $89.2\%$ & Warm retrieval + prefix \\
Distributed        & ---   & ---   & $68.4\%$ & 3 replicas, hash routing \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 7: Ranking Summary

Rankings across key metrics (1 = best).

```latex
\begin{table}[htbp]
\centering
\caption{Baseline Rankings by Metric (1 = Best)}
\label{tab:phase1-rankings}
\begin{tabular}{@{}lrrrr@{}}
\toprule
\textbf{Baseline} & \textbf{Latency} & \textbf{TTFT} & \textbf{QPS} & \textbf{Faithfulness} \\
\midrule
No Cache           & 4 & 5 & 3 & 2 \\
Prefix Cache       & 1 & 1 & 1 & 2 \\
RAG                & 7 & 7 & 7 & 7 \\
Redis Cache (cold) & 6 & 6 & 6 & 4 \\
Hybrid (cold)      & 3 & 4 & 4 & 6 \\
Hybrid (warm)      & 2 & 2 & 2 & 5 \\
Distributed        & 5 & 3 & 5 & 1 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 8: Detailed Latency Percentiles

For papers requiring percentile breakdowns.

**Explanation:** P50 is the median, P95 and P99 capture tail latency behavior. Lower values are better.

```latex
\begin{table}[htbp]
\centering
\caption{Latency Percentiles (ms)}
\label{tab:phase1-percentiles}
\begin{tabular}{@{}lrrrr@{}}
\toprule
\textbf{Baseline} & \textbf{Mean} & \textbf{P50} & \textbf{P95} & \textbf{P99} \\
\midrule
No Cache           & $16006$ & $15265$ & $22240$ & $25262$ \\
Prefix Cache       & $10015$ & $9485$  & $13678$ & $15847$ \\
RAG                & $27270$ & $26594$ & $33481$ & $37124$ \\
Redis Cache (cold) & $26853$ & $26147$ & $33012$ & $35598$ \\
Hybrid (cold)      & $15513$ & $13156$ & $27693$ & $31474$ \\
Hybrid (warm)      & $13269$ & $12582$ & $18923$ & $21345$ \\
Distributed        & $18492$ & $14270$ & $41385$ & $50861$ \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 9: Experimental Setup Summary

For the methodology section.

```latex
\begin{table}[htbp]
\centering
\caption{Phase 1 Experimental Configuration}
\label{tab:phase1-setup}
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Model & Qwen/Qwen3-4B \\
Dataset & SQuAD v2 \\
Queries per baseline & 50 \\
Trials per baseline & 3 \\
Max tokens & 100 \\
Inference backend & vLLM \\
Embedding model & intfloat/e5-large-v2 \\
Reranker model & BAAI/bge-reranker-large \\
Retrieval top-k & 3 \\
Distributed replicas & 3 \\
Routing strategy & Hash-based prefix routing \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 10: Key Findings Summary (Compact)

A compact summary table for abstracts or executive summaries.

```latex
\begin{table}[htbp]
\centering
\caption{Phase 1 Key Findings}
\label{tab:phase1-findings}
\small
\begin{tabular}{@{}p{4cm}p{8cm}@{}}
\toprule
\textbf{Finding} & \textbf{Evidence} \\
\midrule
Prefix caching is highly effective & 37\% latency reduction, 66\% TTFT reduction, 58\% QPS increase vs control with no quality loss \\
\addlinespace
Retrieval overhead dominates & RAG and Redis baselines are 70\% slower than control despite 98\% retrieval hit rate \\
\addlinespace
Hybrid design recovers performance & Hybrid (warm) achieves 17\% latency reduction vs control while using retrieved context \\
\addlinespace
Distributed routing works & Real 3-replica routing validated; 24\% TTFT improvement but 16\% latency overhead \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Figure-Ready Data: Bar Chart Values

For generating bar charts in your preferred tool.

```latex
% Data for bar charts (copy to your plotting tool)
% Baseline, Latency_mean, Latency_std, TTFT_mean, TTFT_std, QPS_mean, QPS_std
% no_cache, 16006, 711, 6919, 112, 0.061, 0.002
% prefix_cache, 10015, 884, 2376, 275, 0.096, 0.008
% rag, 27270, 283, 18775, 144, 0.033, 0.001
% redis_cold, 26853, 39, 18579, 37, 0.034, 0.000
% hybrid_cold, 15513, 4173, 6001, 4290, 0.059, 0.012
% hybrid_warm, 13269, 893, 2791, 66, 0.067, 0.004
% distributed, 18492, 372, 5279, 638, 0.052, 0.001
```

---

## Notes on Table Usage

### Citation format
When citing these results, reference the aggregated metrics across 3 independent trials with 50 queries each.

### Caveats to mention in paper
1. **Cross-trial warming:** For prefix-cache-enabled baselines (`prefix_cache`, `hybrid_cold`, `distributed`), later trials inherit some warmed state because the server was restarted per baseline, not per trial. Trial 1 represents the coldest measurement.

2. **Hybrid cold variance:** The high standard deviation in `hybrid_cold` reflects trial 1 being genuinely cold while trials 2-3 benefited from warmed prompt cache.

3. **Redis semantics:** The Redis baseline caches retrieval artifacts, not raw KV tensors. It should be described as "retrieval artifact caching" rather than "KV cache serving."

4. **Distributed baseline:** This validates router-mediated multi-replica routing without simulated transfer. It is not a distributed KV-cache transfer benchmark.

### Recommended table selection
- **Full paper:** Tables 1, 2, 4, 6, 9
- **Short paper:** Table 3, Table 10
- **Poster/slides:** Table 10, Figure data

---

## Raw Data Reference

All values derived from:
- `analysis/phase1/results/*/aggregated_metrics.json`
- `analysis/phase1/plots/latest_metrics_summary.csv`
- `analysis/phase1/plots/pareto_optimal_baselines.csv`

Generated: March 20, 2026
