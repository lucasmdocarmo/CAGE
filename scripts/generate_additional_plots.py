#!/usr/bin/env python3
"""
Generate 5 additional analysis plots for CAGE Phase 1 results.

Plots:
11. Cache Hit Rate vs TTFT Reduction (scatter with annotations)
12. Quality vs Performance Trade-off Matrix (2D with quadrants)  
13. Overhead Decomposition (waterfall chart)
14. Efficiency Score Ranking (composite metric bar chart)
15. Context Type Impact (grouped comparison showing gold vs retrieved)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PHASE_DIR = PROJECT_DIR / "analysis" / "phase1"
RESULTS_DIR = PHASE_DIR / "results"
PLOTS_DIR = PHASE_DIR / "images"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def metric_stats(section, key):
    value = (section or {}).get(key)
    if isinstance(value, dict):
        return value.get("mean"), value.get("std", 0.0)
    if value is None:
        return None, None
    return value, 0.0


def load_aggregated_data():
    rows = []
    for metrics_path in sorted(RESULTS_DIR.glob("*/aggregated_metrics.json")):
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
        faith_mean, faith_std = metric_stats(quality, "faithfulness")

        rows.append(
            {
                "baseline": baseline,
                "qps_mean": qps_mean,
                "qps_std": qps_std,
                "avg_ttft_ms_mean": ttft_mean,
                "avg_ttft_ms_std": ttft_std,
                "avg_tpot_ms_mean": tpot_mean,
                "avg_tpot_ms_std": tpot_std,
                "avg_latency_ms_mean": latency_mean,
                "avg_latency_ms_std": latency_std,
                "faithfulness_mean": faith_mean,
                "faithfulness_std": faith_std,
            }
        )

    if not rows:
        raise FileNotFoundError(f"No aggregated metrics found under {RESULTS_DIR}")

    return pd.DataFrame(rows)


# Load aggregated data
df = load_aggregated_data()

# Define colors for baselines
COLORS = {
    'no_cache': '#2ecc71',      # Green - baseline
    'prefix_cache': '#3498db',  # Blue - native caching
    'rag': '#e74c3c',           # Red - retrieval
    'redis': '#e67e22',         # Legacy alias
    'redis_retrieval_cache_cold': '#e67e22',   # Orange - cached retrieval
    'distributed': '#9b59b6',   # Legacy alias
    'distributed_router_replicated': '#9b59b6',  # Purple - routed distributed cache
    'hybrid': '#95a5a6',        # Legacy alias
    'hybrid_retrieval_cache_cold': '#95a5a6',  # Gray - hybrid cold
    'hybrid_retrieval_cache_warm': '#7f8c8d',  # Darker gray - hybrid warm
}

# Nice names for display
NICE_NAMES = {
    'no_cache': 'No Cache',
    'prefix_cache': 'Prefix Cache',
    'rag': 'RAG',
    'redis': 'Redis CAG',
    'redis_retrieval_cache_cold': 'Redis Retrieval Cache (Cold)',
    'distributed': 'Distributed CAG',
    'distributed_router_replicated': 'Distributed Router Replicated',
    'hybrid': 'Hybrid',
    'hybrid_retrieval_cache_cold': 'Hybrid Retrieval Cache (Cold)',
    'hybrid_retrieval_cache_warm': 'Hybrid Retrieval Cache (Warm)',
}

# Context type classification
GOLD_CONTEXT = ['no_cache', 'prefix_cache', 'distributed', 'distributed_router_replicated']
RETRIEVED_CONTEXT = ['rag', 'redis', 'redis_retrieval_cache_cold', 'hybrid', 'hybrid_retrieval_cache_cold', 'hybrid_retrieval_cache_warm']


def load_cache_telemetry():
    """Load cache telemetry from all trials."""
    cache_data = {}
    
    for baseline in df['baseline'].values:
        baseline_dir = RESULTS_DIR / baseline
        local_hits = []
        
        for trial_dir in baseline_dir.glob("trial_*"):
            for metrics_file in trial_dir.glob("*_metrics.json"):
                with open(metrics_file) as f:
                    data = json.load(f)
                    cache = data.get('cache_telemetry', {})
                    local_hits.append(cache.get('local_hit_ratio', 0.0))
        
        cache_data[baseline] = {
            'local_hit_mean': np.mean(local_hits) if local_hits else 0.0,
            'local_hit_std': np.std(local_hits) if local_hits else 0.0,
        }
    
    return cache_data


def plot_11_cache_vs_ttft():
    """Plot 11: Cache Hit Rate vs TTFT Reduction scatter."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    cache_data = load_cache_telemetry()
    
    # Reference TTFT (no_cache)
    ref_ttft = df[df['baseline'] == 'no_cache']['avg_ttft_ms_mean'].values[0]
    
    for _, row in df.iterrows():
        baseline = row['baseline']
        cache_hit = cache_data[baseline]['local_hit_mean'] * 100
        ttft = row['avg_ttft_ms_mean']
        ttft_reduction = (ref_ttft - ttft) / ref_ttft * 100
        
        # Size based on faithfulness
        size = row['faithfulness_mean'] * 500
        
        ax.scatter(cache_hit, ttft_reduction, 
                   s=size, c=COLORS[baseline], 
                   alpha=0.7, edgecolors='black', linewidths=1.5,
                   label=NICE_NAMES[baseline])
        
        # Annotate
        offset = (5, 5) if not baseline.startswith('hybrid_') else (-60, -15)
        ax.annotate(NICE_NAMES[baseline], 
                    (cache_hit, ttft_reduction),
                    xytext=offset, textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='No Cache baseline')
    ax.axvline(x=50, color='gray', linestyle=':', alpha=0.3)
    
    ax.set_xlabel('Local Cache Hit Rate (%)', fontsize=12)
    ax.set_ylabel('TTFT Reduction vs No Cache (%)', fontsize=12)
    ax.set_title('Cache Hit Rate vs Time-to-First-Token Improvement\n(bubble size = faithfulness)', 
                 fontsize=14, fontweight='bold')
    
    # Add quadrant labels
    ax.text(75, 8, 'OPTIMAL\nHigh cache hit,\nfast TTFT', ha='center', fontsize=9, 
            style='italic', color='green', alpha=0.7)
    ax.text(25, -150, 'POOR\nLow cache hit,\nslow TTFT', ha='center', fontsize=9,
            style='italic', color='red', alpha=0.7)
    
    ax.set_xlim(-5, 105)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '11_cache_hit_vs_ttft.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved 11_cache_hit_vs_ttft.png")


def plot_12_quality_performance_matrix():
    """Plot 12: Quality vs Performance 2D matrix with quadrants."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Normalize metrics to 0-1 scale for comparison
    # Performance score: higher QPS = better, lower latency = better
    max_qps = df['qps_mean'].max()
    min_latency = df['avg_latency_ms_mean'].min()
    max_latency = df['avg_latency_ms_mean'].max()
    
    for _, row in df.iterrows():
        baseline = row['baseline']
        
        # Quality score (faithfulness)
        quality = row['faithfulness_mean']
        
        # Performance score (normalized inverse latency)
        perf = 1 - (row['avg_latency_ms_mean'] - min_latency) / (max_latency - min_latency)
        
        # Variance as error bars
        quality_err = row['faithfulness_std']
        
        ax.scatter(perf, quality, 
                   s=300, c=COLORS[baseline], 
                   alpha=0.8, edgecolors='black', linewidths=2,
                   marker='o' if baseline in GOLD_CONTEXT else 's',
                   zorder=5)
        
        ax.errorbar(perf, quality, yerr=quality_err,
                    fmt='none', color=COLORS[baseline], alpha=0.5, capsize=5)
        
        # Label
        offset_x = 0.02 if baseline not in ['rag', 'hybrid_retrieval_cache_cold', 'hybrid_retrieval_cache_warm'] else -0.15
        offset_y = 0.01 if baseline != 'distributed' else 0.02
        ax.annotate(NICE_NAMES[baseline], 
                    (perf + offset_x, quality + offset_y),
                    fontsize=10, fontweight='bold')
    
    # Draw quadrant lines at median
    median_perf = 0.5
    median_quality = 0.5
    
    ax.axhline(y=median_quality, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=median_perf, color='gray', linestyle='--', alpha=0.5)
    
    # Quadrant labels
    ax.text(0.75, 0.57, '✓ BEST\nHigh Quality\nHigh Performance', 
            ha='center', va='center', fontsize=10, color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.text(0.25, 0.57, 'High Quality\nLow Performance', 
            ha='center', va='center', fontsize=10, color='orange',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    ax.text(0.75, 0.43, 'Low Quality\nHigh Performance', 
            ha='center', va='center', fontsize=10, color='orange',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    ax.text(0.25, 0.43, '✗ WORST\nLow Quality\nLow Performance', 
            ha='center', va='center', fontsize=10, color='red',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    # Legend for marker shapes
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
    plt.savefig(PLOTS_DIR / '12_quality_performance_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved 12_quality_performance_matrix.png")


def plot_13_overhead_decomposition():
    """Plot 13: Overhead decomposition waterfall chart."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Reference is No Cache
    ref_ttft = df[df['baseline'] == 'no_cache']['avg_ttft_ms_mean'].values[0]
    ref_latency = df[df['baseline'] == 'no_cache']['avg_latency_ms_mean'].values[0]
    
    baselines = [
        'no_cache',
        'prefix_cache',
        'distributed_router_replicated',
        'redis_retrieval_cache_cold',
        'rag',
        'hybrid_retrieval_cache_cold',
        'hybrid_retrieval_cache_warm',
    ]
    x = np.arange(len(baselines))
    width = 0.35
    
    # TTFT values
    ttfts = [df[df['baseline'] == b]['avg_ttft_ms_mean'].values[0] for b in baselines]
    
    # Decompose into: base TTFT + retrieval overhead + cache overhead
    # This is illustrative - showing the components
    base_ttft = ref_ttft  # No Cache baseline
    
    bars_data = []
    for i, baseline in enumerate(baselines):
        row = df[df['baseline'] == baseline].iloc[0]
        ttft = row['avg_ttft_ms_mean']
        tpot = row['avg_tpot_ms_mean']
        latency = row['avg_latency_ms_mean']
        
        # Estimate decode time (100 tokens * TPOT)
        decode_time = tpot * 100
        
        # Everything else is "overhead" compared to baseline
        if baseline == 'no_cache':
            prefill = ttft
            overhead = 0
        else:
            prefill = min(ttft, base_ttft)
            overhead = max(0, ttft - base_ttft)
        
        bars_data.append({
            'baseline': baseline,
            'prefill': prefill,
            'overhead': overhead,
            'decode': decode_time,
        })
    
    # Create stacked bars
    prefills = [d['prefill'] for d in bars_data]
    overheads = [d['overhead'] for d in bars_data]
    decodes = [d['decode'] for d in bars_data]
    
    p1 = ax.bar(x, prefills, width, label='Base Prefill', color='#3498db', alpha=0.8)
    p2 = ax.bar(x, overheads, width, bottom=prefills, label='Additional Overhead', color='#e74c3c', alpha=0.8)
    p3 = ax.bar(x, decodes, width, bottom=np.array(prefills) + np.array(overheads), 
                label='Decode (est.)', color='#2ecc71', alpha=0.8)
    
    # Add total latency line
    latencies = [df[df['baseline'] == b]['avg_latency_ms_mean'].values[0] for b in baselines]
    ax.plot(x, latencies, 'ko-', markersize=8, linewidth=2, label='Actual Total Latency')
    
    ax.set_xlabel('Baseline', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Latency Decomposition: Where Does the Time Go?\n(Prefill + Overhead + Decode)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([NICE_NAMES[b] for b in baselines], rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    
    # Add value labels
    for i, (baseline, ttft, lat) in enumerate(zip(baselines, ttfts, latencies)):
        if baseline != 'no_cache':
            overhead_pct = (ttft - ref_ttft) / ref_ttft * 100
            if overhead_pct > 0:
                ax.annotate(f'+{overhead_pct:.0f}%\nTTFT', (i, ttft + 500), 
                           ha='center', fontsize=9, color='red')
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '13_overhead_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved 13_overhead_decomposition.png")


def plot_14_efficiency_score():
    """Plot 14: Composite efficiency score ranking."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Compute composite efficiency score
    # Score = (Faithfulness * 0.4) + (1/Latency_normalized * 0.3) + (QPS_normalized * 0.3)
    
    # Normalize metrics
    df['latency_norm'] = 1 - (df['avg_latency_ms_mean'] - df['avg_latency_ms_mean'].min()) / \
                         (df['avg_latency_ms_mean'].max() - df['avg_latency_ms_mean'].min())
    df['qps_norm'] = (df['qps_mean'] - df['qps_mean'].min()) / \
                     (df['qps_mean'].max() - df['qps_mean'].min())
    df['faith_norm'] = (df['faithfulness_mean'] - df['faithfulness_mean'].min()) / \
                       (df['faithfulness_mean'].max() - df['faithfulness_mean'].min())
    
    # Weights
    w_quality = 0.40
    w_latency = 0.35
    w_throughput = 0.25
    
    df['efficiency_score'] = (df['faith_norm'] * w_quality + 
                              df['latency_norm'] * w_latency + 
                              df['qps_norm'] * w_throughput)
    
    # Sort by score
    df_sorted = df.sort_values('efficiency_score', ascending=True)
    
    y = np.arange(len(df_sorted))
    
    # Create horizontal bar chart with component breakdown
    bars = ax.barh(y, df_sorted['efficiency_score'], 
                   color=[COLORS[b] for b in df_sorted['baseline']],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add component indicators
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        score = row['efficiency_score']
        # Add text showing component breakdown
        ax.text(score + 0.02, i, 
                f"Q:{row['faith_norm']:.2f} L:{row['latency_norm']:.2f} T:{row['qps_norm']:.2f}",
                va='center', fontsize=9, alpha=0.7)
    
    ax.set_yticks(y)
    ax.set_yticklabels([NICE_NAMES[b] for b in df_sorted['baseline']], fontsize=11)
    ax.set_xlabel('Composite Efficiency Score\n(Quality×0.40 + Latency×0.35 + Throughput×0.25)', fontsize=12)
    ax.set_title('Overall Efficiency Ranking\n(Higher = Better)', fontsize=14, fontweight='bold')
    
    # Add score values on bars
    for i, (bar, score) in enumerate(zip(bars, df_sorted['efficiency_score'])):
        ax.text(0.02, i, f'{score:.3f}', va='center', ha='left', 
                fontsize=11, fontweight='bold', color='white')
    
    # Mark top performer
    ax.axhline(y=len(df_sorted)-0.5, color='gold', linestyle='--', alpha=0.5)
    ax.text(0.95, len(df_sorted)-1, '🏆', fontsize=20, va='center')
    
    ax.set_xlim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend for weights
    weight_text = f"Weights: Quality={w_quality:.0%}, Latency={w_latency:.0%}, Throughput={w_throughput:.0%}"
    ax.text(0.5, -0.8, weight_text, transform=ax.transAxes, ha='center', 
            fontsize=10, style='italic', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '14_efficiency_ranking.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved 14_efficiency_ranking.png")


def plot_15_context_type_impact():
    """Plot 15: Impact of context type (gold vs retrieved) on metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Prepare data by context type
    gold_data = df[df['baseline'].isin(GOLD_CONTEXT)]
    retrieved_data = df[df['baseline'].isin(RETRIEVED_CONTEXT)]
    
    metrics = [
        ('faithfulness_mean', 'faithfulness_std', 'Faithfulness', axes[0, 0]),
        ('avg_ttft_ms_mean', 'avg_ttft_ms_std', 'TTFT (ms)', axes[0, 1]),
        ('avg_latency_ms_mean', 'avg_latency_ms_std', 'Latency (ms)', axes[1, 0]),
        ('qps_mean', 'qps_std', 'QPS', axes[1, 1]),
    ]
    
    for mean_col, std_col, title, ax in metrics:
        x = np.arange(2)
        width = 0.6
        
        gold_mean = gold_data[mean_col].mean()
        gold_std = gold_data[mean_col].std()
        retrieved_mean = retrieved_data[mean_col].mean()
        retrieved_std = retrieved_data[mean_col].std()
        
        bars = ax.bar(x, [gold_mean, retrieved_mean], width,
                      yerr=[gold_std, retrieved_std],
                      color=['#27ae60', '#c0392b'],
                      alpha=0.8, edgecolor='black', linewidth=1.5,
                      capsize=8)
        
        # Add individual baseline points
        for i, (_, row) in enumerate(gold_data.iterrows()):
            ax.scatter(0 + (i-1)*0.1, row[mean_col], 
                       c=COLORS[row['baseline']], s=100, zorder=5,
                       edgecolors='black', linewidths=1)
        
        for i, (_, row) in enumerate(retrieved_data.iterrows()):
            ax.scatter(1 + (i-1)*0.1, row[mean_col], 
                       c=COLORS[row['baseline']], s=100, zorder=5,
                       edgecolors='black', linewidths=1)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Gold Context\n(No Cache, Prefix, Distributed)', 
                           'Retrieved Context\n(RAG, Redis, Hybrid)'], fontsize=10)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Add percentage difference
        if gold_mean > 0:
            pct_diff = (gold_mean - retrieved_mean) / gold_mean * 100
            better = "Gold" if (title == 'Faithfulness' or title == 'QPS') == (pct_diff > 0) else "Retrieved"
            if title in ['TTFT (ms)', 'Latency (ms)']:
                better = "Gold" if pct_diff < 0 else "Retrieved"
                pct_diff = -pct_diff
            
            color = 'green' if better == 'Gold' else 'red'
            ax.annotate(f'{better} is\n{abs(pct_diff):.1f}% better', 
                        xy=(0.5, 0.85), xycoords='axes fraction',
                        ha='center', fontsize=10, color=color,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Impact of Context Source on All Metrics\n(Gold Context vs Retrieved Context)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Add legend
    handles = [mpatches.Patch(color=COLORS[b], label=NICE_NAMES[b]) for b in df['baseline']]
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.12, 0.5), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '15_context_type_impact.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved 15_context_type_impact.png")


def main():
    print("Generating 5 additional analysis plots...")
    print("=" * 50)
    
    plot_11_cache_vs_ttft()
    plot_12_quality_performance_matrix()
    plot_13_overhead_decomposition()
    plot_14_efficiency_score()
    plot_15_context_type_impact()
    
    print("=" * 50)
    print(f"All plots saved to: {PLOTS_DIR}")
    print("\nNew plots:")
    print("  11_cache_hit_vs_ttft.png        - Cache hit rate vs TTFT reduction")
    print("  12_quality_performance_matrix.png - Quality vs Performance quadrants")
    print("  13_overhead_decomposition.png   - Latency breakdown waterfall")
    print("  14_efficiency_ranking.png       - Composite efficiency score")
    print("  15_context_type_impact.png      - Gold vs Retrieved context impact")


if __name__ == "__main__":
    main()
