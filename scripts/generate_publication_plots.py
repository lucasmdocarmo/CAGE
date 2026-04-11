#!/usr/bin/env python3
"""
Generate publication-quality plots for CAGE Phase 1 results.

This script produces numbered plots (01-15) matching the expected naming convention:
01_latency_comparison.png      - Latency comparison bar chart with error bars
02_throughput_comparison.png   - QPS and TTFT grouped comparison
03_quality_comparison.png      - Quality metrics (faithfulness, etc.)
04_latency_breakdown.png       - Stacked bar: TTFT vs decode time
05_speedup_vs_nocache.png      - Speedup relative to no_cache baseline
06_pareto_latency_vs_quality.png - Pareto frontier plot
07_radar_profile.png           - Multi-dimensional radar/spider chart
08_heatmap.png                 - Normalized performance heatmap
09_boxplots_variance.png       - Trial variance boxplots
10_summary_table.png           - Visual summary table
11_cache_hit_vs_ttft.png       - Cache hit rate vs TTFT scatter
12_quality_performance_matrix.png - Quality vs Performance 2D matrix
13_overhead_decomposition.png  - Latency decomposition waterfall
14_efficiency_ranking.png      - Composite efficiency score ranking
15_context_type_impact.png     - Gold vs Retrieved context comparison

Output: analysis/phase1/images/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
})

# Color palette for baselines
COLORS = {
    'no_cache': '#2ecc71',           # Green - control
    'prefix_cache': '#3498db',       # Blue - native caching
    'rag': '#e74c3c',                # Red - retrieval
    'redis_retrieval_cache_cold': '#e67e22',  # Orange - cached retrieval
    'distributed_router_replicated': '#9b59b6',  # Purple - distributed
    'hybrid_retrieval_cache_cold': '#95a5a6',    # Gray - hybrid cold
    'hybrid_retrieval_cache_warm': '#7f8c8d',    # Darker gray - hybrid warm
}

# Nice display names
NICE_NAMES = {
    'no_cache': 'No Cache (Control)',
    'prefix_cache': 'Prefix Cache',
    'rag': 'RAG',
    'redis_retrieval_cache_cold': 'Redis Cache (Cold)',
    'distributed_router_replicated': 'Distributed Router',
    'hybrid_retrieval_cache_cold': 'Hybrid Cache (Cold)',
    'hybrid_retrieval_cache_warm': 'Hybrid Cache (Warm)',
}

# Context type classification
GOLD_CONTEXT = ['no_cache', 'prefix_cache', 'distributed_router_replicated']
RETRIEVED_CONTEXT = ['rag', 'redis_retrieval_cache_cold', 'hybrid_retrieval_cache_cold', 'hybrid_retrieval_cache_warm']

# Preferred baseline order
BASELINE_ORDER = [
    'no_cache',
    'prefix_cache',
    'rag',
    'redis_retrieval_cache_cold',
    'hybrid_retrieval_cache_cold',
    'hybrid_retrieval_cache_warm',
    'distributed_router_replicated',
]


def get_color(baseline: str) -> str:
    return COLORS.get(baseline, '#bdc3c7')


def get_name(baseline: str) -> str:
    return NICE_NAMES.get(baseline, baseline.replace('_', ' ').title())


def metric_stats(section: dict, key: str) -> tuple[float | None, float]:
    """Extract mean and std from a metric section."""
    value = (section or {}).get(key)
    if isinstance(value, dict):
        return value.get("mean"), value.get("std", 0.0)
    if value is None:
        return None, 0.0
    return value, 0.0


def load_aggregated_data(results_dir: Path) -> pd.DataFrame:
    """Load aggregated metrics from all baselines."""
    rows = []
    for metrics_path in sorted(results_dir.glob("*/aggregated_metrics.json")):
        with open(metrics_path) as f:
            data = json.load(f)

        experiment = data.get("experiment", {}) or {}
        baseline = experiment.get("baseline")
        if not baseline:
            continue

        performance = data.get("performance", {}) or {}
        quality = data.get("quality", {}) or {}
        
        qps_mean, qps_std = metric_stats(performance, "queries_per_second")
        ttft_mean, ttft_std = metric_stats(performance, "avg_ttft_ms")
        tpot_mean, tpot_std = metric_stats(performance, "avg_tpot_ms")
        latency_mean, latency_std = metric_stats(performance, "avg_latency_ms")
        tokens_per_sec_mean, tokens_per_sec_std = metric_stats(performance, "tokens_per_second")
        faith_mean, faith_std = metric_stats(quality, "faithfulness")
        relevance_mean, relevance_std = metric_stats(quality, "relevance")
        bertscore_mean, bertscore_std = metric_stats(quality, "completeness_bertscore")
        rouge_mean, rouge_std = metric_stats(quality, "completeness_rouge_l")

        rows.append({
            "baseline": baseline,
            "qps_mean": qps_mean,
            "qps_std": qps_std,
            "avg_ttft_ms_mean": ttft_mean,
            "avg_ttft_ms_std": ttft_std,
            "avg_tpot_ms_mean": tpot_mean,
            "avg_tpot_ms_std": tpot_std,
            "avg_latency_ms_mean": latency_mean,
            "avg_latency_ms_std": latency_std,
            "tokens_per_sec_mean": tokens_per_sec_mean,
            "tokens_per_sec_std": tokens_per_sec_std,
            "faithfulness_mean": faith_mean,
            "faithfulness_std": faith_std,
            "relevance_mean": relevance_mean,
            "relevance_std": relevance_std,
            "bertscore_mean": bertscore_mean,
            "bertscore_std": bertscore_std,
            "rouge_l_mean": rouge_mean,
            "rouge_l_std": rouge_std,
        })

    if not rows:
        raise FileNotFoundError(f"No aggregated metrics found under {results_dir}")

    df = pd.DataFrame(rows)
    
    # Sort by preferred order
    df['sort_order'] = df['baseline'].map({b: i for i, b in enumerate(BASELINE_ORDER)}).fillna(999)
    df = df.sort_values('sort_order').drop(columns=['sort_order'])
    
    return df


def load_trial_data(results_dir: Path) -> pd.DataFrame:
    """Load per-trial data for variance analysis."""
    rows = []
    for baseline_dir in sorted(results_dir.iterdir()):
        if not baseline_dir.is_dir():
            continue
        baseline = baseline_dir.name
        
        for trial_dir in sorted(baseline_dir.glob("trial_*")):
            for metrics_file in trial_dir.glob("*_metrics.json"):
                with open(metrics_file) as f:
                    data = json.load(f)
                
                perf = data.get("performance", {}) or {}
                qual = data.get("quality", {}) or {}
                cache = data.get("cache_telemetry", {}) or {}
                
                rows.append({
                    "baseline": baseline,
                    "trial": trial_dir.name,
                    "qps": perf.get("queries_per_second"),
                    "avg_ttft_ms": perf.get("avg_ttft_ms"),
                    "avg_latency_ms": perf.get("avg_latency_ms"),
                    "faithfulness": qual.get("faithfulness"),
                    "local_hit_ratio": cache.get("local_hit_ratio", 0.0),
                })
    
    return pd.DataFrame(rows)


def plot_01_latency_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 01: Latency comparison bar chart with error bars."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(df))
    width = 0.35
    
    # Avg Latency bars
    bars1 = ax.bar(x - width/2, df['avg_latency_ms_mean'], width,
                   yerr=df['avg_latency_ms_std'],
                   label='Avg Latency (ms)',
                   color=[get_color(b) for b in df['baseline']],
                   alpha=0.8, edgecolor='black', linewidth=1.5, capsize=5)
    
    # TTFT bars
    bars2 = ax.bar(x + width/2, df['avg_ttft_ms_mean'], width,
                   yerr=df['avg_ttft_ms_std'],
                   label='Avg TTFT (ms)',
                   color=[get_color(b) for b in df['baseline']],
                   alpha=0.5, edgecolor='black', linewidth=1.5, capsize=5,
                   hatch='///')
    
    ax.set_xlabel('Baseline', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Latency Comparison Across Baselines\n(with standard deviation)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([get_name(b) for b in df['baseline']], rotation=20, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, df['avg_latency_ms_mean']):
        ax.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / '01_latency_comparison.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated 01_latency_comparison.png")


def plot_02_throughput_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 02: Throughput comparison (QPS and Tokens/sec)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(df))
    
    # QPS comparison
    bars1 = ax1.bar(x, df['qps_mean'], yerr=df['qps_std'],
                    color=[get_color(b) for b in df['baseline']],
                    alpha=0.8, edgecolor='black', linewidth=1.5, capsize=5)
    ax1.set_xlabel('Baseline', fontsize=12)
    ax1.set_ylabel('Queries per Second (QPS)', fontsize=12)
    ax1.set_title('Throughput: Queries per Second', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([get_name(b) for b in df['baseline']], rotation=25, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, df['qps_mean']):
        ax1.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    # Tokens/sec comparison
    bars2 = ax2.bar(x, df['tokens_per_sec_mean'], yerr=df['tokens_per_sec_std'],
                    color=[get_color(b) for b in df['baseline']],
                    alpha=0.8, edgecolor='black', linewidth=1.5, capsize=5)
    ax2.set_xlabel('Baseline', fontsize=12)
    ax2.set_ylabel('Tokens per Second', fontsize=12)
    ax2.set_title('Throughput: Tokens per Second', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([get_name(b) for b in df['baseline']], rotation=25, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, df['tokens_per_sec_mean']):
        if pd.notna(val):
            ax2.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                         xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    # Panel labels below center
    ax1.text(0.5, -0.32, '(a)', transform=ax1.transAxes,
             fontsize=14, fontweight='normal', ha='center', va='top', fontfamily='serif')
    ax2.text(0.5, -0.32, '(b)', transform=ax2.transAxes,
             fontsize=14, fontweight='normal', ha='center', va='top', fontfamily='serif')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    plt.savefig(output_dir / '02_throughput_comparison.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated 02_throughput_comparison.png")


def plot_03_quality_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 03: Quality metrics comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(df))
    width = 0.2
    
    metrics = [
        ('faithfulness_mean', 'faithfulness_std', 'Faithfulness', -1.5),
        ('relevance_mean', 'relevance_std', 'Relevance', -0.5),
        ('bertscore_mean', 'bertscore_std', 'BERTScore', 0.5),
        ('rouge_l_mean', 'rouge_l_std', 'ROUGE-L', 1.5),
    ]
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22']
    
    for (mean_col, std_col, label, offset), color in zip(metrics, colors):
        if mean_col in df.columns and df[mean_col].notna().any():
            ax.bar(x + offset * width, df[mean_col], width,
                   yerr=df[std_col] if std_col in df.columns else None,
                   label=label, color=color, alpha=0.8, edgecolor='black',
                   linewidth=1, capsize=3)
    
    ax.set_xlabel('Baseline', fontsize=14)
    ax.set_ylabel('Score (0-1)', fontsize=14)
    ax.set_title('Quality Metrics Comparison\n(Higher is Better)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([get_name(b) for b in df['baseline']], rotation=20, ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=True,
              fontsize=12, columnspacing=1.5)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    plt.savefig(output_dir / '03_quality_comparison.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated 03_quality_comparison.png")


def plot_04_latency_breakdown(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 04: Stacked bar showing TTFT vs decode time."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(df))
    width = 0.6
    
    # Generation time = total latency - TTFT
    generation_ms = (df['avg_latency_ms_mean'] - df['avg_ttft_ms_mean']).clip(lower=0)
    
    bars1 = ax.bar(x, df['avg_ttft_ms_mean'], width, 
                   label='TTFT (Prefill)', color='#3498db', alpha=0.85,
                   edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, generation_ms, width, bottom=df['avg_ttft_ms_mean'],
                   label='Generation (Decode)', color='#2ecc71', alpha=0.85,
                   edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Baseline', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Latency Breakdown: Prefill vs Decode\n(Stacked Components)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([get_name(b) for b in df['baseline']], rotation=20, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add total latency labels
    for i, (_, row) in enumerate(df.iterrows()):
        ax.annotate(f'{row["avg_latency_ms_mean"]:.0f}ms total',
                    xy=(i, row['avg_latency_ms_mean']),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '04_latency_breakdown.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated 04_latency_breakdown.png")


def plot_05_speedup_vs_nocache(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 05: Speedup relative to no_cache baseline."""
    if 'no_cache' not in df['baseline'].values:
        print("⚠ Skipping 05_speedup_vs_nocache.png: no_cache baseline not found")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ref_row = df[df['baseline'] == 'no_cache'].iloc[0]
    ref_latency = ref_row['avg_latency_ms_mean']
    ref_ttft = ref_row['avg_ttft_ms_mean']
    
    df_other = df[df['baseline'] != 'no_cache'].copy()
    df_other['latency_speedup'] = ref_latency / df_other['avg_latency_ms_mean']
    df_other['ttft_speedup'] = ref_ttft / df_other['avg_ttft_ms_mean']
    
    x = np.arange(len(df_other))
    
    # Latency speedup
    colors1 = ['#27ae60' if s > 1 else '#e74c3c' for s in df_other['latency_speedup']]
    bars1 = ax1.bar(x, df_other['latency_speedup'], color=colors1, 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=1, color='black', linestyle='--', linewidth=2, label='No Cache (1.0×)')
    ax1.set_xlabel('Baseline', fontsize=12)
    ax1.set_ylabel('Speedup Factor (×)', fontsize=12)
    ax1.set_title('Latency Speedup vs No Cache', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([get_name(b) for b in df_other['baseline']], rotation=25, ha='right')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, df_other['latency_speedup']):
        color = 'darkgreen' if val > 1 else 'darkred'
        ax1.annotate(f'{val:.2f}×', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center', 
                     fontsize=10, fontweight='bold', color=color)
    
    # TTFT speedup
    colors2 = ['#27ae60' if s > 1 else '#e74c3c' for s in df_other['ttft_speedup']]
    bars2 = ax2.bar(x, df_other['ttft_speedup'], color=colors2,
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=1, color='black', linestyle='--', linewidth=2, label='No Cache (1.0×)')
    ax2.set_xlabel('Baseline', fontsize=12)
    ax2.set_ylabel('Speedup Factor (×)', fontsize=12)
    ax2.set_title('TTFT Speedup vs No Cache', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([get_name(b) for b in df_other['baseline']], rotation=25, ha='right')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, df_other['ttft_speedup']):
        color = 'darkgreen' if val > 1 else 'darkred'
        ax2.annotate(f'{val:.2f}×', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center',
                     fontsize=10, fontweight='bold', color=color)
    
    plt.suptitle('Performance Speedup Relative to No Cache Baseline', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / '05_speedup_vs_nocache.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated 05_speedup_vs_nocache.png")


def plot_06_pareto_latency_vs_quality(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 06: Pareto frontier - latency vs quality."""
    fig, ax = plt.subplots(figsize=(11, 8))
    
    # Use faithfulness as quality metric
    x = df['avg_latency_ms_mean']
    y = df['faithfulness_mean']
    
    # Find Pareto-optimal points (lower latency, higher quality)
    is_pareto = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            # j dominates i if j has lower latency AND higher or equal quality
            if x.iloc[j] <= x.iloc[i] and y.iloc[j] >= y.iloc[i]:
                if x.iloc[j] < x.iloc[i] or y.iloc[j] > y.iloc[i]:
                    is_pareto[i] = False
                    break
    
    # Plot all points
    for i, (_, row) in enumerate(df.iterrows()):
        baseline = row['baseline']
        marker = '*' if is_pareto[i] else 'o'
        size = 300 if is_pareto[i] else 150
        alpha = 1.0 if is_pareto[i] else 0.6
        zorder = 10 if is_pareto[i] else 5
        
        ax.scatter(row['avg_latency_ms_mean'], row['faithfulness_mean'],
                   c=get_color(baseline), s=size, marker=marker, alpha=alpha,
                   edgecolors='black', linewidths=2, zorder=zorder,
                   label=get_name(baseline))
        
        # Annotate
        fontweight = 'bold' if is_pareto[i] else 'normal'
        ax.annotate(get_name(baseline),
                    (row['avg_latency_ms_mean'], row['faithfulness_mean']),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=10, fontweight=fontweight)
    
    # Draw Pareto frontier line
    pareto_df = df[is_pareto].sort_values('avg_latency_ms_mean')
    if len(pareto_df) > 1:
        ax.plot(pareto_df['avg_latency_ms_mean'], pareto_df['faithfulness_mean'],
                'r--', linewidth=2, alpha=0.5, zorder=2)
    
    ax.set_xlabel('Average Latency (ms) — Lower is Better →', fontsize=14)
    ax.set_ylabel('Faithfulness — Higher is Better →', fontsize=14)
    ax.set_title('Pareto Frontier: Latency vs Quality Trade-off\n(★ = Pareto-optimal)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add ideal corner annotation
    ax.annotate('← IDEAL', xy=(ax.get_xlim()[0] + 100, ax.get_ylim()[1] - 0.01),
                fontsize=13, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '06_pareto_latency_vs_quality.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated 06_pareto_latency_vs_quality.png")


def plot_07_radar_profile(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 07: Multi-dimensional radar/spider chart."""
    metrics = ['qps_mean', 'avg_ttft_ms_mean', 'avg_latency_ms_mean', 
               'faithfulness_mean', 'relevance_mean', 'bertscore_mean']
    labels = ['QPS', 'TTFT', 'Latency', 'Faithfulness', 'Relevance', 'BERTScore']
    
    # Filter available metrics
    available_metrics = [m for m in metrics if m in df.columns and df[m].notna().any()]
    available_labels = [labels[metrics.index(m)] for m in available_metrics]
    
    if len(available_metrics) < 3:
        print("⚠ Skipping 07_radar_profile.png: not enough metrics")
        return
    
    # Normalize metrics (0-1, higher = better)
    df_norm = df.copy()
    for col in available_metrics:
        min_val, max_val = df[col].min(), df[col].max()
        if max_val > min_val:
            df_norm[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0.5
    
    # Invert latency metrics (lower is better)
    for col in ['avg_ttft_ms_mean', 'avg_latency_ms_mean']:
        if col in df_norm.columns:
            df_norm[col] = 1 - df_norm[col]
    
    # Setup radar
    num_vars = len(available_metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for _, row in df_norm.iterrows():
        baseline = row['baseline']
        values = [row[m] for m in available_metrics]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=get_name(baseline),
                color=get_color(baseline), markersize=8)
        ax.fill(angles, values, alpha=0.15, color=get_color(baseline))
    
    # Adjust labels
    display_labels = []
    for m, label in zip(available_metrics, available_labels):
        if m in ['avg_ttft_ms_mean', 'avg_latency_ms_mean']:
            display_labels.append(f'{label}\n(inverted)')
        else:
            display_labels.append(label)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_labels, size=11)
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Dimensional Performance Profile\n(Outer = Better)', fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_dir / '07_radar_profile.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated 07_radar_profile.png")


def plot_08_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 08: Normalized performance heatmap."""
    metrics = ['qps_mean', 'tokens_per_sec_mean', 'avg_ttft_ms_mean', 
               'avg_latency_ms_mean', 'faithfulness_mean', 'relevance_mean', 'bertscore_mean']
    labels = ['QPS', 'Tokens/s', 'TTFT', 'Latency', 'Faithfulness', 'Relevance', 'BERTScore']
    
    available = [(m, l) for m, l in zip(metrics, labels) if m in df.columns and df[m].notna().any()]
    if len(available) < 3:
        print("⚠ Skipping 08_heatmap.png: not enough metrics")
        return
    
    metrics, labels = zip(*available)
    
    # Create heatmap data
    df_heat = df.set_index('baseline')[list(metrics)].copy()
    
    # Normalize (0-1)
    for col in metrics:
        min_val, max_val = df_heat[col].min(), df_heat[col].max()
        if max_val > min_val:
            df_heat[col] = (df_heat[col] - min_val) / (max_val - min_val)
        else:
            df_heat[col] = 0.5
    
    # Invert latency metrics
    for col in ['avg_ttft_ms_mean', 'avg_latency_ms_mean']:
        if col in df_heat.columns:
            df_heat[col] = 1 - df_heat[col]
    
    df_heat.columns = labels
    df_heat.index = [get_name(b) for b in df_heat.index]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(df_heat, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5,
                ax=ax, cbar_kws={'label': 'Normalized Score (0-1)'}, linewidths=0.5)
    ax.set_title('Performance Heatmap (Normalized)\n(Green = Better, Latency metrics inverted)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Baseline', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / '08_heatmap.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated 08_heatmap.png")


def plot_09_boxplots_variance(df_trials: pd.DataFrame, output_dir: Path) -> None:
    """Plot 09: Trial variance boxplots."""
    if df_trials.empty:
        print("⚠ Skipping 09_boxplots_variance.png: no trial data")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = [
        ('avg_latency_ms', 'Latency (ms)', axes[0, 0]),
        ('avg_ttft_ms', 'TTFT (ms)', axes[0, 1]),
        ('qps', 'QPS', axes[1, 0]),
        ('faithfulness', 'Faithfulness', axes[1, 1]),
    ]
    
    # Order baselines
    order = [b for b in BASELINE_ORDER if b in df_trials['baseline'].unique()]
    
    for col, title, ax in metrics:
        if col not in df_trials.columns or df_trials[col].isna().all():
            ax.text(0.5, 0.5, f'No {title} data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        palette = {b: get_color(b) for b in order}
        sns.boxplot(data=df_trials, x='baseline', y=col, order=order, palette=palette, ax=ax)
        ax.set_title(f'{title} Variance Across Trials', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(title)
        ax.set_xticklabels([get_name(b) for b in order], rotation=25, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Trial-to-Trial Variance Analysis\n(3 trials per baseline)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / '09_boxplots_variance.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated 09_boxplots_variance.png")


def plot_10_summary_table(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 10: Visual summary table."""
    metrics_config = {
        'qps_mean': {'higher_better': True, 'label': 'QPS'},
        'avg_ttft_ms_mean': {'higher_better': False, 'label': 'TTFT (ms)'},
        'avg_latency_ms_mean': {'higher_better': False, 'label': 'Latency (ms)'},
        'tokens_per_sec_mean': {'higher_better': True, 'label': 'Tokens/s'},
        'faithfulness_mean': {'higher_better': True, 'label': 'Faithfulness'},
        'bertscore_mean': {'higher_better': True, 'label': 'BERTScore'},
    }
    
    # Build ranking for each metric
    rankings = {}
    for metric, config in metrics_config.items():
        if metric not in df.columns or df[metric].isna().all():
            continue
        
        if config['higher_better']:
            ranked = df.nlargest(len(df), metric)['baseline'].tolist()
        else:
            ranked = df.nsmallest(len(df), metric)['baseline'].tolist()
        
        rankings[config['label']] = ranked
    
    if not rankings:
        print("⚠ Skipping 10_summary_table.png: no metrics")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    metrics_list = list(rankings.keys())
    n_baselines = len(df)
    
    cell_text = []
    for i in range(n_baselines):
        row = []
        for metric in metrics_list:
            if i < len(rankings[metric]):
                row.append(get_name(rankings[metric][i]))
            else:
                row.append('')
        cell_text.append(row)
    
    row_labels = [f'#{i+1}' for i in range(n_baselines)]
    
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=metrics_list,
        cellLoc='center',
        loc='center',
        colWidths=[0.15] * len(metrics_list)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Color winners
    for j in range(len(metrics_list)):
        table[(1, j)].set_facecolor('#2ecc71')
        table[(1, j)].set_text_props(fontweight='bold', color='white')
    
    ax.set_title('Baseline Rankings by Metric (Best → Worst)', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / '10_summary_table.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated 10_summary_table.png")


def plot_11_cache_hit_vs_ttft(df: pd.DataFrame, df_trials: pd.DataFrame, output_dir: Path) -> None:
    """Plot 11: Cache hit rate vs TTFT reduction scatter."""
    if df_trials.empty or 'local_hit_ratio' not in df_trials.columns:
        print("⚠ Skipping 11_cache_hit_vs_ttft.png: no cache telemetry")
        return
    
    if 'no_cache' not in df['baseline'].values:
        print("⚠ Skipping 11_cache_hit_vs_ttft.png: no_cache baseline not found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate cache hit rates per baseline
    cache_rates = df_trials.groupby('baseline')['local_hit_ratio'].mean() * 100
    
    ref_ttft = df[df['baseline'] == 'no_cache']['avg_ttft_ms_mean'].values[0]
    
    for _, row in df.iterrows():
        baseline = row['baseline']
        cache_hit = cache_rates.get(baseline, 0)
        ttft = row['avg_ttft_ms_mean']
        ttft_reduction = (ref_ttft - ttft) / ref_ttft * 100
        
        size = row['faithfulness_mean'] * 500
        
        ax.scatter(cache_hit, ttft_reduction, s=size, c=get_color(baseline),
                   alpha=0.7, edgecolors='black', linewidths=1.5,
                   label=get_name(baseline))
        
        offset = (5, 5) if not baseline.startswith('hybrid_') else (-60, -15)
        ax.annotate(get_name(baseline), (cache_hit, ttft_reduction),
                    xytext=offset, textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='No Cache baseline')
    ax.set_xlabel('Local Cache Hit Rate (%)', fontsize=12)
    ax.set_ylabel('TTFT Reduction vs No Cache (%)', fontsize=12)
    ax.set_title('Cache Hit Rate vs Time-to-First-Token Improvement\n(bubble size = faithfulness)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(-5, 105)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '11_cache_hit_vs_ttft.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated 11_cache_hit_vs_ttft.png")


def plot_12_quality_performance_matrix(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 12: Quality vs Performance 2D matrix with quadrants."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Normalize metrics
    max_latency = df['avg_latency_ms_mean'].max()
    min_latency = df['avg_latency_ms_mean'].min()
    
    for _, row in df.iterrows():
        baseline = row['baseline']
        quality = row['faithfulness_mean']
        perf = 1 - (row['avg_latency_ms_mean'] - min_latency) / (max_latency - min_latency + 1e-9)
        
        marker = 'o' if baseline in GOLD_CONTEXT else 's'
        ax.scatter(perf, quality, s=300, c=get_color(baseline),
                   alpha=0.8, edgecolors='black', linewidths=2,
                   marker=marker, zorder=5)
        
        offset_x = 0.02 if baseline not in RETRIEVED_CONTEXT else -0.02
        ax.annotate(get_name(baseline), (perf + offset_x, quality + 0.01),
                    fontsize=10, fontweight='bold')
    
    # Quadrant lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Quadrant labels
    ax.text(0.75, 0.55, '✓ BEST\nHigh Quality\nHigh Performance',
            ha='center', va='center', fontsize=10, color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.text(0.25, 0.55, 'High Quality\nLow Performance',
            ha='center', va='center', fontsize=10, color='orange',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    # Legend
    gold_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                              markersize=12, label='Gold Context')
    retrieved_marker = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                                   markersize=12, label='Retrieved Context')
    ax.legend(handles=[gold_marker, retrieved_marker], loc='lower left', fontsize=10)
    
    ax.set_xlabel('Performance Score\n(normalized inverse latency: 1 = fastest)', fontsize=12)
    ax.set_ylabel('Quality Score\n(faithfulness: 1 = perfect)', fontsize=12)
    ax.set_title('Quality vs Performance Trade-off Matrix\n(○ = gold context, □ = retrieved context)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0.35, 0.65)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '12_quality_performance_matrix.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated 12_quality_performance_matrix.png")


def plot_13_overhead_decomposition(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 13: Overhead decomposition waterfall chart."""
    if 'no_cache' not in df['baseline'].values:
        print("⚠ Skipping 13_overhead_decomposition.png: no_cache baseline not found")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ref_ttft = df[df['baseline'] == 'no_cache']['avg_ttft_ms_mean'].values[0]
    
    baselines = [b for b in BASELINE_ORDER if b in df['baseline'].values]
    x = np.arange(len(baselines))
    width = 0.5
    
    bars_data = []
    for baseline in baselines:
        row = df[df['baseline'] == baseline].iloc[0]
        ttft = row['avg_ttft_ms_mean']
        tpot = row.get('avg_tpot_ms_mean', 30)  # default estimate
        
        decode_time = (tpot if pd.notna(tpot) else 30) * 100  # ~100 tokens
        
        if baseline == 'no_cache':
            prefill = ttft
            overhead = 0
        else:
            prefill = min(ttft, ref_ttft)
            overhead = max(0, ttft - ref_ttft)
        
        bars_data.append({
            'baseline': baseline,
            'prefill': prefill,
            'overhead': overhead,
            'decode': decode_time,
        })
    
    prefills = [d['prefill'] for d in bars_data]
    overheads = [d['overhead'] for d in bars_data]
    decodes = [d['decode'] for d in bars_data]
    
    ax.bar(x, prefills, width, label='Base Prefill', color='#3498db', alpha=0.8)
    ax.bar(x, overheads, width, bottom=prefills, label='Additional Overhead', color='#e74c3c', alpha=0.8)
    ax.bar(x, decodes, width, bottom=np.array(prefills) + np.array(overheads),
           label='Decode (est.)', color='#2ecc71', alpha=0.8)
    
    # Total latency line
    latencies = [df[df['baseline'] == b]['avg_latency_ms_mean'].values[0] for b in baselines]
    ax.plot(x, latencies, 'ko-', markersize=8, linewidth=2, label='Actual Total Latency')
    
    ax.set_xlabel('Baseline', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Latency Decomposition: Where Does the Time Go?\n(Prefill + Overhead + Decode)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([get_name(b) for b in baselines], rotation=20, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / '13_overhead_decomposition.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated 13_overhead_decomposition.png")


def plot_14_efficiency_ranking(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 14: Composite efficiency score ranking."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    df = df.copy()
    
    # Normalize metrics
    for col, invert in [('avg_latency_ms_mean', True), ('qps_mean', False), ('faithfulness_mean', False)]:
        if col not in df.columns:
            continue
        min_val, max_val = df[col].min(), df[col].max()
        if max_val > min_val:
            df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
            if invert:
                df[f'{col}_norm'] = 1 - df[f'{col}_norm']
        else:
            df[f'{col}_norm'] = 0.5
    
    # Composite score
    w_quality, w_latency, w_throughput = 0.40, 0.35, 0.25
    
    df['efficiency_score'] = (
        df.get('faithfulness_mean_norm', 0.5) * w_quality +
        df.get('avg_latency_ms_mean_norm', 0.5) * w_latency +
        df.get('qps_mean_norm', 0.5) * w_throughput
    )
    
    df_sorted = df.sort_values('efficiency_score', ascending=True)
    y = np.arange(len(df_sorted))
    
    bars = ax.barh(y, df_sorted['efficiency_score'],
                   color=[get_color(b) for b in df_sorted['baseline']],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(y)
    ax.set_yticklabels([get_name(b) for b in df_sorted['baseline']], fontsize=11)
    ax.set_xlabel(f'Composite Efficiency Score\n(Quality×{w_quality:.0%} + Latency×{w_latency:.0%} + Throughput×{w_throughput:.0%})', fontsize=12)
    ax.set_title('Overall Efficiency Ranking\n(Higher = Better)', fontsize=14, fontweight='bold')
    
    # Score labels
    for i, (bar, score) in enumerate(zip(bars, df_sorted['efficiency_score'])):
        ax.text(0.02, i, f'{score:.3f}', va='center', ha='left',
                fontsize=11, fontweight='bold', color='white')
    
    ax.text(0.95, len(df_sorted)-1, '🏆', fontsize=20, va='center')
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / '14_efficiency_ranking.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated 14_efficiency_ranking.png")


def plot_15_context_type_impact(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 15: Impact of context type (gold vs retrieved) on metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    gold_data = df[df['baseline'].isin(GOLD_CONTEXT)]
    retrieved_data = df[df['baseline'].isin(RETRIEVED_CONTEXT)]
    
    if gold_data.empty or retrieved_data.empty:
        print("⚠ Skipping 15_context_type_impact.png: missing context type data")
        return
    
    metrics = [
        ('faithfulness_mean', 'Faithfulness', axes[0, 0]),
        ('avg_ttft_ms_mean', 'TTFT (ms)', axes[0, 1]),
        ('avg_latency_ms_mean', 'Latency (ms)', axes[1, 0]),
        ('qps_mean', 'QPS', axes[1, 1]),
    ]
    
    for mean_col, title, ax in metrics:
        x = np.arange(2)
        
        gold_mean = gold_data[mean_col].mean()
        gold_std = gold_data[mean_col].std()
        retrieved_mean = retrieved_data[mean_col].mean()
        retrieved_std = retrieved_data[mean_col].std()
        
        bars = ax.bar(x, [gold_mean, retrieved_mean], 0.6,
                      yerr=[gold_std, retrieved_std],
                      color=['#27ae60', '#c0392b'],
                      alpha=0.8, edgecolor='black', linewidth=1.5, capsize=8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Gold Context\n(No Cache, Prefix, Distributed)',
                          'Retrieved Context\n(RAG, Redis, Hybrid)'], fontsize=10)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add percentage difference
        if gold_mean > 0:
            pct_diff = (gold_mean - retrieved_mean) / gold_mean * 100
            better = "Gold" if (title in ['Faithfulness', 'QPS']) == (pct_diff > 0) else "Retrieved"
            if title in ['TTFT (ms)', 'Latency (ms)']:
                better = "Gold" if pct_diff < 0 else "Retrieved"
                pct_diff = -pct_diff
            
            color = 'green' if better == 'Gold' else 'red'
            ax.annotate(f'{better} is\n{abs(pct_diff):.1f}% better',
                        xy=(0.5, 0.85), xycoords='axes fraction',
                        ha='center', fontsize=10, color=color,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('Impact of Context Source on All Metrics\n(Gold Context vs Retrieved Context)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / '15_context_type_impact.png', bbox_inches='tight')
    plt.close()
    print("✓ Generated 15_context_type_impact.png")


def plot_16_ttft_tail_latency(df: pd.DataFrame, results_dir: Path, output_dir: Path) -> None:
    """Plot 16: TTFT tail latency — p50 vs p95 range chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Load p50 and p95 from aggregated metrics
    rows = []
    for baseline in BASELINE_ORDER:
        p = results_dir / baseline / 'aggregated_metrics.json'
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        perf = data.get('performance', {})
        p50 = perf.get('p50_ttft_ms', {}).get('mean')
        p95 = perf.get('p95_ttft_ms', {}).get('mean')
        if p50 is not None and p95 is not None:
            rows.append({'baseline': baseline, 'p50': p50, 'p95': p95, 'ratio': p95 / p50})

    if not rows:
        print("Warning: Skipping 16_ttft_tail_latency.png: no percentile data")
        return

    baselines = [r['baseline'] for r in rows]
    y = np.arange(len(baselines))
    p50s = [r['p50'] for r in rows]
    p95s = [r['p95'] for r in rows]
    ratios = [r['ratio'] for r in rows]

    # Draw range lines (p50 to p95)
    for i, (b, v50, v95, ratio) in enumerate(zip(baselines, p50s, p95s, ratios)):
        color = get_color(b)
        # Horizontal line from p50 to p95
        ax.plot([v50, v95], [i, i], color=color, linewidth=3, alpha=0.7, solid_capstyle='round')
        # p50 marker (circle)
        ax.scatter(v50, i, color=color, s=120, zorder=5, edgecolors='black', linewidths=1.5, label=None)
        # p95 marker (diamond)
        ax.scatter(v95, i, color=color, s=120, zorder=5, edgecolors='black', linewidths=1.5,
                   marker='D', label=None)
        # Ratio annotation
        fontweight = 'bold' if ratio > 3 else 'normal'
        fontcolor = '#c0392b' if ratio > 3 else '#2c3e50'
        ax.annotate(f'{ratio:.1f}x',
                    xy=(v95, i), xytext=(12, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=11, fontweight=fontweight, color=fontcolor)

    ax.set_yticks(y)
    ax.set_yticklabels([get_name(b) for b in baselines], fontsize=11)
    ax.set_xlabel('Time to First Token (ms)', fontsize=12)
    ax.set_title('TTFT Tail Latency: Median (p50) vs 95th Percentile (p95)\n'
                 'Line length = latency spread; number = p95/p50 ratio',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='gray', markerfacecolor='gray',
               markersize=10, linestyle='None', label='p50 (median)'),
        Line2D([0], [0], marker='D', color='gray', markerfacecolor='gray',
               markersize=10, linestyle='None', label='p95 (tail)'),
        Line2D([0], [0], color='gray', linewidth=3, alpha=0.7, label='Latency spread'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_dir / '16_ttft_tail_latency.png', bbox_inches='tight')
    plt.close()
    print("Checkmark Generated 16_ttft_tail_latency.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-quality Phase 1 plots.")
    parser.add_argument(
        "--results-dir",
        default="analysis/phase1/results",
        help="Directory containing baseline result directories with aggregated_metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/phase1/images",
        help="Directory to write plots",
    )
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Publication-Quality Phase 1 Plots")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load data
    print("Loading aggregated metrics...")
    df = load_aggregated_data(results_dir)
    print(f"  Found {len(df)} baselines: {', '.join(df['baseline'].tolist())}")
    
    print("Loading trial data...")
    df_trials = load_trial_data(results_dir)
    print(f"  Found {len(df_trials)} trial records")
    print()
    
    # Generate all plots
    print("Generating plots...")
    print("-" * 40)
    
    plot_01_latency_comparison(df, output_dir)
    plot_02_throughput_comparison(df, output_dir)
    plot_03_quality_comparison(df, output_dir)
    plot_04_latency_breakdown(df, output_dir)
    plot_05_speedup_vs_nocache(df, output_dir)
    plot_06_pareto_latency_vs_quality(df, output_dir)
    plot_07_radar_profile(df, output_dir)
    plot_08_heatmap(df, output_dir)
    plot_09_boxplots_variance(df_trials, output_dir)
    plot_10_summary_table(df, output_dir)
    plot_11_cache_hit_vs_ttft(df, df_trials, output_dir)
    plot_12_quality_performance_matrix(df, output_dir)
    plot_13_overhead_decomposition(df, output_dir)
    plot_14_efficiency_ranking(df, output_dir)
    plot_15_context_type_impact(df, output_dir)
    plot_16_ttft_tail_latency(df, results_dir, output_dir)
    
    print()
    print("=" * 60)
    print(f"All plots saved to: {output_dir}")
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
