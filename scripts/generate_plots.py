#!/usr/bin/env python3
"""
Generate summary plots from the latest baseline metrics JSON files.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_timestamp(path: str) -> str | None:
    m = re.search(r"_(\d{8}_\d{6})_metrics\.json$", path)
    return m.group(1) if m else None

def metric_value(section: dict, key: str):
    value = (section or {}).get(key)
    if isinstance(value, dict) and "mean" in value:
        return value.get("mean")
    return value


def discover_metric_files(results_dir: Path) -> list[Path]:
    aggregated = sorted(results_dir.rglob("aggregated_metrics.json"))
    if aggregated:
        return aggregated

    stable = [
        path for path in sorted(results_dir.rglob("metrics.json"))
        if "trial_" not in path.parts
    ]
    if stable:
        return stable

    return sorted(Path(path) for path in glob.glob(str(results_dir / "**" / "*_metrics.json"), recursive=True))


def load_latest_metrics(results_dir: Path) -> pd.DataFrame:
    latest_by_experiment: dict[tuple[str, str | None, str | None], dict] = {}

    for metrics_path in discover_metric_files(results_dir):
        with open(metrics_path, "r") as fh:
            data = json.load(fh)

        experiment = data.get("experiment", {}) or {}
        baseline = experiment.get("baseline")
        dataset = experiment.get("dataset")
        model = experiment.get("model")
        if not baseline:
            continue

        timestamp = experiment.get("timestamp") or parse_timestamp(str(metrics_path)) or ""
        key = (baseline, dataset, model)
        prev = latest_by_experiment.get(key)
        if prev is None or timestamp > prev["timestamp"]:
            latest_by_experiment[key] = {
                "timestamp": timestamp,
                "data": data,
                "file": str(metrics_path),
            }

    rows = []
    for (baseline, dataset, model), entry in latest_by_experiment.items():
        data = entry["data"]
        perf = data.get("performance", {}) or {}
        qual = data.get("quality", {}) or {}
        retr = data.get("retrieval", {}) or {}
        rows.append(
            {
                "baseline": baseline,
                "dataset": dataset,
                "model": model,
                "timestamp": entry["timestamp"],
                "file": entry["file"],
                "qps": metric_value(perf, "queries_per_second"),
                "ttft_ms": metric_value(perf, "avg_ttft_ms"),
                "latency_ms": metric_value(perf, "avg_latency_ms"),
                "tokens_per_sec": metric_value(perf, "tokens_per_second"),
                "faithfulness": metric_value(qual, "faithfulness"),
                "relevance": metric_value(qual, "relevance"),
                "bertscore": metric_value(qual, "completeness_bertscore"),
                "rouge_l": metric_value(qual, "completeness_rouge_l"),
                "retrieval_hit": metric_value(retr, "avg_hit"),
            }
        )
    return pd.DataFrame(rows)
    return df


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, filename: Path) -> None:
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x=x, y=y, order=df[x].tolist())
    plt.title(title)
    plt.xlabel("Baseline")
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_scatter(df: pd.DataFrame, x: str, y: str, title: str, filename: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x=x, y=y)
    for _, row in df.iterrows():
        plt.text(row[x], row[y], row["baseline"], fontsize=8, ha="left", va="bottom")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_pie(df: pd.DataFrame, value_col: str, title: str, filename: Path) -> bool:
    values = df[value_col].fillna(0)
    total = float(values.sum())
    if total <= 0:
        return False
    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=df["baseline"].tolist(), autopct="%1.1f%%", startangle=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    return True


def plot_pareto(df: pd.DataFrame, value_col: str, title: str, filename: Path) -> bool:
    vals = df[value_col].fillna(0)
    if float(vals.sum()) <= 0:
        return False
    df_sorted = df.sort_values(by=value_col, ascending=False).reset_index(drop=True)
    vals_sorted = df_sorted[value_col].fillna(0)
    cum_pct = vals_sorted.cumsum() / max(float(vals_sorted.sum()), 1.0) * 100.0

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(df_sorted["baseline"], vals_sorted)
    ax1.set_ylabel(value_col)
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(df_sorted["baseline"], cum_pct, color="red", marker="o")
    ax2.set_ylabel("Cumulative %")
    ax2.set_ylim(0, 100)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    return True


def compute_pareto_frontier(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_minimize: bool = True,
    y_maximize: bool = True,
) -> pd.DataFrame:
    """
    Compute the Pareto frontier for multi-objective optimization.
    
    Args:
        df: DataFrame with data points
        x_col: Column for X objective (e.g., latency)
        y_col: Column for Y objective (e.g., quality)
        x_minimize: True if lower X is better (e.g., latency)
        y_maximize: True if higher Y is better (e.g., quality)
    
    Returns:
        DataFrame containing only Pareto-optimal points
    """
    df_valid = df.dropna(subset=[x_col, y_col]).copy()
    if df_valid.empty:
        return df_valid
    
    # Convert to numpy for efficiency
    import numpy as np
    points = df_valid[[x_col, y_col]].values
    n_points = len(points)
    
    # Track which points are Pareto-optimal
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if not is_pareto[i]:
            continue
        for j in range(n_points):
            if i == j or not is_pareto[j]:
                continue
            
            # Check if point j dominates point i
            # Domination: j is at least as good in all objectives and strictly better in at least one
            x_i, y_i = points[i]
            x_j, y_j = points[j]
            
            # Determine "better" based on optimization direction
            x_j_better = (x_j < x_i) if x_minimize else (x_j > x_i)
            x_j_equal_or_better = (x_j <= x_i) if x_minimize else (x_j >= x_i)
            y_j_better = (y_j > y_i) if y_maximize else (y_j < y_i)
            y_j_equal_or_better = (y_j >= y_i) if y_maximize else (y_j <= y_i)
            
            # j dominates i if j is at least as good in both and strictly better in at least one
            if x_j_equal_or_better and y_j_equal_or_better and (x_j_better or y_j_better):
                is_pareto[i] = False
                break
    
    return df_valid[is_pareto].copy()


def plot_pareto_frontier(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    filename: Path,
    x_label: str | None = None,
    y_label: str | None = None,
    x_minimize: bool = True,
    y_maximize: bool = True,
) -> bool:
    """
    Plot multi-objective Pareto frontier with all points and frontier highlighted.
    
    Args:
        df: DataFrame with experiment results
        x_col: Column for X axis (e.g., latency_ms)
        y_col: Column for Y axis (e.g., bertscore)
        title: Plot title
        filename: Output file path
        x_label: X axis label (defaults to x_col)
        y_label: Y axis label (defaults to y_col)
        x_minimize: True if lower X is better
        y_maximize: True if higher Y is better
    
    Returns:
        True if plot was generated, False if insufficient data
    """
    df_valid = df.dropna(subset=[x_col, y_col])
    if len(df_valid) < 2:
        return False
    
    # Compute Pareto frontier
    pareto_df = compute_pareto_frontier(df_valid, x_col, y_col, x_minimize, y_maximize)
    
    # Sort Pareto points for line plotting
    pareto_df = pareto_df.sort_values(by=x_col)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot all points
    non_pareto = df_valid[~df_valid.index.isin(pareto_df.index)]
    ax.scatter(
        non_pareto[x_col], non_pareto[y_col],
        c="lightgray", s=100, alpha=0.7, label="Dominated", zorder=1
    )
    
    # Plot Pareto-optimal points
    ax.scatter(
        pareto_df[x_col], pareto_df[y_col],
        c="red", s=150, marker="*", label="Pareto Optimal", zorder=3
    )
    
    # Draw Pareto frontier line (step function for clarity)
    if len(pareto_df) > 1:
        ax.step(
            pareto_df[x_col], pareto_df[y_col],
            where="post", color="red", linestyle="--", alpha=0.5, linewidth=2, zorder=2
        )
    
    # Label all points with baseline names
    for _, row in df_valid.iterrows():
        is_pareto_pt = row.name in pareto_df.index
        fontweight = "bold" if is_pareto_pt else "normal"
        color = "darkred" if is_pareto_pt else "gray"
        ax.annotate(
            row["baseline"],
            (row[x_col], row[y_col]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=9, fontweight=fontweight, color=color
        )
    
    # Add optimization direction arrows
    arrow_props = dict(arrowstyle="->", color="green", lw=2)
    x_dir = "←" if x_minimize else "→"
    y_dir = "↑" if y_maximize else "↓"
    ax.annotate(
        f"Better {x_dir}", xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=10, color="green", ha="left"
    )
    ax.annotate(
        f"Better {y_dir}", xy=(0.02, 0.06), xycoords="axes fraction",
        fontsize=10, color="green", ha="left"
    )
    
    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(y_label or y_col)
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    
    return True


def generate_pareto_analysis(
    df: pd.DataFrame,
    plots_dir: Path,
) -> list[tuple[str, str]]:
    """
    Generate comprehensive Pareto frontier analysis plots.
    
    Returns list of (filename, description) tuples for generated plots.
    """
    generated = []
    
    # Primary tradeoff: Latency vs Quality (BERTScore)
    if plot_pareto_frontier(
        df, "latency_ms", "bertscore",
        "Pareto Frontier: Latency vs Quality",
        plots_dir / "pareto_latency_vs_quality.png",
        x_label="Average Latency (ms) - Lower is Better",
        y_label="BERTScore - Higher is Better",
        x_minimize=True, y_maximize=True,
    ):
        generated.append((
            "pareto_latency_vs_quality.png",
            "Multi-objective Pareto frontier showing optimal tradeoffs between latency and quality. "
            "Red stars are Pareto-optimal baselines (no other baseline is better in both dimensions)."
        ))
    
    # TTFT vs Quality
    if plot_pareto_frontier(
        df, "ttft_ms", "bertscore",
        "Pareto Frontier: TTFT vs Quality",
        plots_dir / "pareto_ttft_vs_quality.png",
        x_label="Time to First Token (ms) - Lower is Better",
        y_label="BERTScore - Higher is Better",
        x_minimize=True, y_maximize=True,
    ):
        generated.append((
            "pareto_ttft_vs_quality.png",
            "Pareto frontier for time-to-first-token vs quality tradeoff."
        ))
    
    # Throughput vs Faithfulness
    if plot_pareto_frontier(
        df, "qps", "faithfulness",
        "Pareto Frontier: Throughput vs Faithfulness",
        plots_dir / "pareto_throughput_vs_faithfulness.png",
        x_label="Queries per Second - Higher is Better",
        y_label="Faithfulness - Higher is Better",
        x_minimize=False, y_maximize=True,  # Higher QPS is better
    ):
        generated.append((
            "pareto_throughput_vs_faithfulness.png",
            "Pareto frontier for throughput vs faithfulness. Note: higher is better for both axes."
        ))
    
    # Latency vs Relevance
    if plot_pareto_frontier(
        df, "latency_ms", "relevance",
        "Pareto Frontier: Latency vs Relevance",
        plots_dir / "pareto_latency_vs_relevance.png",
        x_label="Average Latency (ms) - Lower is Better",
        y_label="Relevance - Higher is Better",
        x_minimize=True, y_maximize=True,
    ):
        generated.append((
            "pareto_latency_vs_relevance.png",
            "Pareto frontier for latency vs relevance tradeoff."
        ))
    
    # Generate Pareto summary CSV
    pareto_summary = []
    for x_col, y_col, x_min, y_max in [
        ("latency_ms", "bertscore", True, True),
        ("ttft_ms", "bertscore", True, True),
        ("qps", "faithfulness", False, True),
        ("latency_ms", "relevance", True, True),
    ]:
        pareto_df = compute_pareto_frontier(df, x_col, y_col, x_min, y_max)
        for _, row in pareto_df.iterrows():
            pareto_summary.append({
                "tradeoff": f"{x_col}_vs_{y_col}",
                "baseline": row["baseline"],
                x_col: row[x_col],
                y_col: row[y_col],
            })
    
    if pareto_summary:
        pareto_csv = pd.DataFrame(pareto_summary)
        pareto_csv.to_csv(plots_dir / "pareto_optimal_baselines.csv", index=False)
        print(f"Saved Pareto-optimal baselines to {plots_dir / 'pareto_optimal_baselines.csv'}")
    
    return generated


def plot_grouped_bar(
    df: pd.DataFrame,
    metrics: list[str],
    title: str,
    filename: Path,
    ylabel: str = "Value",
    normalize: bool = False,
) -> bool:
    """
    Create grouped bar chart comparing multiple metrics across baselines.
    Useful for side-by-side comparison of related metrics.
    """
    df_valid = df.dropna(subset=metrics, how="all")
    if df_valid.empty:
        return False
    
    # Melt for seaborn
    df_melted = df_valid.melt(
        id_vars=["baseline"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Value"
    )
    
    if normalize:
        # Normalize each metric to 0-1 range for comparison
        for metric in metrics:
            vals = df_melted[df_melted["Metric"] == metric]["Value"]
            min_val, max_val = vals.min(), vals.max()
            if max_val > min_val:
                df_melted.loc[df_melted["Metric"] == metric, "Value"] = \
                    (df_melted.loc[df_melted["Metric"] == metric, "Value"] - min_val) / (max_val - min_val)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_melted, x="baseline", y="Value", hue="Metric", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Baseline")
    ax.set_ylabel(ylabel)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_heatmap(
    df: pd.DataFrame,
    metrics: list[str],
    title: str,
    filename: Path,
    annot: bool = True,
) -> bool:
    """
    Create heatmap of metrics across baselines.
    Great for showing overall performance patterns at a glance.
    """
    df_valid = df.dropna(subset=metrics, how="all").set_index("baseline")
    if df_valid.empty:
        return False
    
    # Normalize each column to 0-1 for fair comparison
    df_norm = df_valid[metrics].copy()
    for col in metrics:
        min_val, max_val = df_norm[col].min(), df_norm[col].max()
        if max_val > min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        df_norm,
        annot=annot,
        fmt=".2f",
        cmap="RdYlGn",
        center=0.5,
        ax=ax,
        cbar_kws={"label": "Normalized Score (0-1)"}
    )
    ax.set_title(title)
    ax.set_ylabel("Baseline")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_radar(
    df: pd.DataFrame,
    metrics: list[str],
    title: str,
    filename: Path,
) -> bool:
    """
    Create radar/spider chart comparing baselines across multiple dimensions.
    Excellent for visualizing multi-dimensional performance profiles.
    """
    import numpy as np
    
    df_valid = df.dropna(subset=metrics, how="all")
    if df_valid.empty or len(df_valid) < 2:
        return False
    
    # Normalize metrics to 0-1 (higher = better for visualization)
    df_norm = df_valid.copy()
    for col in metrics:
        min_val, max_val = df_norm[col].min(), df_norm[col].max()
        if max_val > min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0.5
    
    # For latency metrics, invert (lower is better)
    latency_cols = [c for c in metrics if "latency" in c.lower() or "ttft" in c.lower()]
    for col in latency_cols:
        df_norm[col] = 1 - df_norm[col]
    
    # Setup radar chart
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(df_norm)))
    
    for idx, (_, row) in enumerate(df_norm.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, "o-", linewidth=2, label=row["baseline"], color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    # Fix labels - remove latency inversion note, just show metric names
    labels = []
    for m in metrics:
        if "latency" in m.lower() or "ttft" in m.lower():
            labels.append(f"{m}\n(inverted)")
        else:
            labels.append(m)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(0, 1)
    ax.set_title(title, size=14, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_latency_breakdown(
    df: pd.DataFrame,
    title: str,
    filename: Path,
) -> bool:
    """
    Stacked bar chart showing TTFT vs generation time breakdown.
    Shows where time is spent in the inference pipeline.
    """
    df_valid = df.dropna(subset=["ttft_ms", "latency_ms"])
    if df_valid.empty:
        return False
    
    # Generation time = total latency - TTFT
    df_plot = df_valid.copy()
    df_plot["generation_ms"] = df_plot["latency_ms"] - df_plot["ttft_ms"]
    df_plot["generation_ms"] = df_plot["generation_ms"].clip(lower=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(df_plot))
    width = 0.6
    
    bars1 = ax.bar(x, df_plot["ttft_ms"], width, label="TTFT (Prefill)", color="#2ecc71")
    bars2 = ax.bar(x, df_plot["generation_ms"], width, bottom=df_plot["ttft_ms"],
                   label="Generation (Decode)", color="#3498db")
    
    ax.set_xlabel("Baseline")
    ax.set_ylabel("Time (ms)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["baseline"], rotation=45, ha="right")
    ax.legend()
    
    # Add total latency labels on top
    for i, (_, row) in enumerate(df_plot.iterrows()):
        ax.annotate(
            f"{row['latency_ms']:.0f}ms",
            xy=(i, row["latency_ms"]),
            ha="center", va="bottom",
            fontsize=9, fontweight="bold"
        )
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_quality_comparison(
    df: pd.DataFrame,
    title: str,
    filename: Path,
) -> bool:
    """
    Grouped bar chart of all quality metrics side by side.
    Essential for showing quality parity across approaches.
    """
    quality_metrics = ["faithfulness", "relevance", "bertscore", "rouge_l"]
    available = [m for m in quality_metrics if m in df.columns and df[m].notna().any()]
    
    if not available:
        return False
    
    return plot_grouped_bar(
        df, available, title, filename,
        ylabel="Score (0-1)", normalize=False
    )


def plot_efficiency_scatter(
    df: pd.DataFrame,
    title: str,
    filename: Path,
) -> bool:
    """
    Bubble chart: X=latency, Y=quality, size=throughput.
    Shows the efficiency frontier with three dimensions.
    """
    df_valid = df.dropna(subset=["latency_ms", "bertscore", "qps"])
    if len(df_valid) < 2:
        return False
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize QPS for bubble sizes (100-1000 range)
    qps_min, qps_max = df_valid["qps"].min(), df_valid["qps"].max()
    if qps_max > qps_min:
        sizes = 100 + 900 * (df_valid["qps"] - qps_min) / (qps_max - qps_min)
    else:
        sizes = 500
    
    scatter = ax.scatter(
        df_valid["latency_ms"],
        df_valid["bertscore"],
        s=sizes,
        alpha=0.6,
        c=range(len(df_valid)),
        cmap="viridis",
        edgecolors="black",
        linewidth=1
    )
    
    # Label points
    for _, row in df_valid.iterrows():
        ax.annotate(
            f"{row['baseline']}\n({row['qps']:.1f} QPS)",
            (row["latency_ms"], row["bertscore"]),
            xytext=(10, 5), textcoords="offset points",
            fontsize=9, ha="left"
        )
    
    ax.set_xlabel("Latency (ms) - Lower is Better →")
    ax.set_ylabel("BERTScore - Higher is Better →")
    ax.set_title(f"{title}\n(Bubble size = Throughput/QPS)")
    ax.grid(True, alpha=0.3)
    
    # Add ideal corner annotation
    ax.annotate(
        "← IDEAL",
        xy=(ax.get_xlim()[0], ax.get_ylim()[1]),
        fontsize=12, color="green", fontweight="bold",
        ha="left", va="top"
    )
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_speedup_chart(
    df: pd.DataFrame,
    baseline_name: str,
    title: str,
    filename: Path,
) -> bool:
    """
    Bar chart showing speedup relative to a baseline (e.g., no_cache).
    Critical for quantifying improvement claims.
    """
    if baseline_name not in df["baseline"].values:
        return False
    
    baseline_row = df[df["baseline"] == baseline_name].iloc[0]
    baseline_latency = baseline_row["latency_ms"]
    baseline_ttft = baseline_row["ttft_ms"]
    
    if pd.isna(baseline_latency) or baseline_latency <= 0:
        return False
    
    df_plot = df[df["baseline"] != baseline_name].copy()
    df_plot["latency_speedup"] = baseline_latency / df_plot["latency_ms"]
    df_plot["ttft_speedup"] = baseline_ttft / df_plot["ttft_ms"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Latency speedup
    colors = ["#27ae60" if x > 1 else "#e74c3c" for x in df_plot["latency_speedup"]]
    bars1 = ax1.bar(df_plot["baseline"], df_plot["latency_speedup"], color=colors, edgecolor="black")
    ax1.axhline(y=1, color="black", linestyle="--", linewidth=1, label=f"Baseline ({baseline_name})")
    ax1.set_ylabel("Speedup (×)")
    ax1.set_title(f"Latency Speedup vs {baseline_name}")
    ax1.tick_params(axis="x", rotation=45)
    
    # Add value labels
    for bar, val in zip(bars1, df_plot["latency_speedup"]):
        ax1.annotate(
            f"{val:.2f}×",
            xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    
    # TTFT speedup
    colors = ["#27ae60" if x > 1 else "#e74c3c" for x in df_plot["ttft_speedup"]]
    bars2 = ax2.bar(df_plot["baseline"], df_plot["ttft_speedup"], color=colors, edgecolor="black")
    ax2.axhline(y=1, color="black", linestyle="--", linewidth=1, label=f"Baseline ({baseline_name})")
    ax2.set_ylabel("Speedup (×)")
    ax2.set_title(f"TTFT Speedup vs {baseline_name}")
    ax2.tick_params(axis="x", rotation=45)
    
    for bar, val in zip(bars2, df_plot["ttft_speedup"]):
        ax2.annotate(
            f"{val:.2f}×",
            xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_ranking_table(
    df: pd.DataFrame,
    filename: Path,
) -> bool:
    """
    Create a visual ranking table showing which baseline wins each metric.
    Great for executive summary / abstract support.
    """
    metrics_config = {
        "qps": {"higher_better": True, "label": "Throughput (QPS)"},
        "ttft_ms": {"higher_better": False, "label": "TTFT (ms)"},
        "latency_ms": {"higher_better": False, "label": "Latency (ms)"},
        "tokens_per_sec": {"higher_better": True, "label": "Tokens/sec"},
        "faithfulness": {"higher_better": True, "label": "Faithfulness"},
        "relevance": {"higher_better": True, "label": "Relevance"},
        "bertscore": {"higher_better": True, "label": "BERTScore"},
    }
    
    rankings = {}
    for metric, config in metrics_config.items():
        if metric not in df.columns or df[metric].isna().all():
            continue
        
        if config["higher_better"]:
            ranked = df.nlargest(len(df), metric)[["baseline", metric]]
        else:
            ranked = df.nsmallest(len(df), metric)[["baseline", metric]]
        
        rankings[config["label"]] = ranked["baseline"].tolist()
    
    if not rankings:
        return False
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    
    # Build table data
    metrics_list = list(rankings.keys())
    n_baselines = len(df)
    
    cell_text = []
    for i in range(n_baselines):
        row = []
        for metric in metrics_list:
            if i < len(rankings[metric]):
                row.append(rankings[metric][i])
            else:
                row.append("")
        cell_text.append(row)
    
    row_labels = [f"#{i+1}" for i in range(n_baselines)]
    
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=metrics_list,
        cellLoc="center",
        loc="center",
        colWidths=[0.14] * len(metrics_list)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Color the winners (first row)
    for j in range(len(metrics_list)):
        table[(1, j)].set_facecolor("#2ecc71")
        table[(1, j)].set_text_props(fontweight="bold", color="white")
    
    ax.set_title("Baseline Rankings by Metric (Best → Worst)", fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def write_plot_explanations(plots_dir: Path, plots: list[tuple[str, str]]) -> None:
    lines = [
        "Plot explanations",
        "=================",
        "",
    ]
    for fname, desc in plots:
        lines.append(f"- {fname}: {desc}")
    (plots_dir / "plots_explained.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots from metrics JSON files.")
    parser.add_argument(
        "--results-dir",
        default="analysis/phase1/results",
        help="Directory containing *_metrics.json files.",
    )
    parser.add_argument(
        "--plots-dir",
        default="analysis/phase1/images",
        help="Directory to write plots and summary CSV.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_latest_metrics(results_dir)
    if df.empty:
        raise SystemExit("No metrics JSON files found.")

    # Ensure a consistent baseline order if present
    preferred_order = [
        "no_cache",
        "prefix_cache",
        "rag",
        "redis_retrieval_cache_cold",
        "hybrid_retrieval_cache_cold",
        "hybrid_retrieval_cache_warm",
        "distributed_router_replicated",
        "redis",
        "hybrid",
        "distributed_replicated_real",
        "distributed_sharded_sim",
        "distributed",
        "speculative",
    ]
    df = df.sort_values(
        by="baseline",
        key=lambda s: s.map({k: i for i, k in enumerate(preferred_order)}).fillna(999),
    )

    plot_bar(
        df,
        "baseline",
        "qps",
        "Throughput (QPS) by Baseline",
        plots_dir / "qps_by_baseline.png",
    )
    plot_bar(
        df,
        "baseline",
        "ttft_ms",
        "Average TTFT (ms) by Baseline",
        plots_dir / "ttft_by_baseline.png",
    )
    plot_bar(
        df,
        "baseline",
        "latency_ms",
        "Average Latency (ms) by Baseline",
        plots_dir / "latency_by_baseline.png",
    )
    plot_bar(
        df,
        "baseline",
        "tokens_per_sec",
        "Throughput (tokens/sec) by Baseline",
        plots_dir / "tokens_per_sec_by_baseline.png",
    )
    plot_bar(
        df,
        "baseline",
        "bertscore",
        "Completeness (BERTScore) by Baseline",
        plots_dir / "bertscore_by_baseline.png",
    )
    plot_scatter(
        df,
        "latency_ms",
        "bertscore",
        "Performance vs Quality (Latency vs BERTScore)",
        plots_dir / "tradeoff_latency_vs_bertscore.png",
    )
    # Pie (pizza) + Pareto-like plot (tokens/sec share)
    pie_file = plots_dir / "tokens_per_sec_share_pie.png"
    pareto_file = plots_dir / "tokens_per_sec_pareto.png"
    plot_pie(df, "tokens_per_sec", "Tokens/sec Share by Baseline", pie_file)
    plot_pareto(df, "tokens_per_sec", "Tokens/sec Pareto by Baseline", pareto_file)

    # Generate multi-objective Pareto frontier analysis
    print("\nGenerating Pareto frontier analysis...")
    pareto_plots = generate_pareto_analysis(df, plots_dir)
    print(f"Generated {len(pareto_plots)} Pareto frontier plots")

    # === NEW PAPER-QUALITY PLOTS ===
    print("\nGenerating paper-quality plots...")
    
    # 1. Radar chart - multi-dimensional performance profile
    radar_metrics = ["qps", "ttft_ms", "latency_ms", "faithfulness", "relevance", "bertscore"]
    radar_available = [m for m in radar_metrics if m in df.columns and df[m].notna().any()]
    if len(radar_available) >= 3:
        plot_radar(
            df, radar_available,
            "Multi-Dimensional Performance Profile",
            plots_dir / "radar_performance_profile.png"
        )
        print("  - Generated radar chart")
    
    # 2. Heatmap - overall performance at a glance
    heatmap_metrics = ["qps", "tokens_per_sec", "ttft_ms", "latency_ms", "faithfulness", "relevance", "bertscore"]
    heatmap_available = [m for m in heatmap_metrics if m in df.columns and df[m].notna().any()]
    if len(heatmap_available) >= 3:
        plot_heatmap(
            df, heatmap_available,
            "Performance Heatmap (Normalized)",
            plots_dir / "heatmap_all_metrics.png"
        )
        print("  - Generated heatmap")
    
    # 3. Latency breakdown - TTFT vs generation time
    if plot_latency_breakdown(
        df,
        "Latency Breakdown: Prefill vs Decode",
        plots_dir / "latency_breakdown_stacked.png"
    ):
        print("  - Generated latency breakdown")
    
    # 4. Quality comparison - all quality metrics side by side
    if plot_quality_comparison(
        df,
        "Quality Metrics Comparison",
        plots_dir / "quality_metrics_grouped.png"
    ):
        print("  - Generated quality comparison")
    
    # 5. Efficiency bubble chart - 3D view
    if plot_efficiency_scatter(
        df,
        "Efficiency Analysis: Latency vs Quality vs Throughput",
        plots_dir / "efficiency_bubble_chart.png"
    ):
        print("  - Generated efficiency bubble chart")
    
    # 6. Speedup chart vs no_cache baseline
    if plot_speedup_chart(
        df, "no_cache",
        "Performance Speedup vs No-Cache Baseline",
        plots_dir / "speedup_vs_no_cache.png"
    ):
        print("  - Generated speedup chart")
    
    # 7. Ranking table - which baseline wins
    if plot_ranking_table(df, plots_dir / "ranking_table.png"):
        print("  - Generated ranking table")
    
    # 8. Performance metrics grouped bar
    perf_metrics = ["qps", "tokens_per_sec"]
    perf_available = [m for m in perf_metrics if m in df.columns and df[m].notna().any()]
    if len(perf_available) >= 1:
        plot_grouped_bar(
            df, perf_available,
            "Throughput Metrics Comparison",
            plots_dir / "throughput_metrics_grouped.png",
            ylabel="Value"
        )
        print("  - Generated throughput grouped bar")

    # Explanation file for plots
    plots_explained = [
        ("qps_by_baseline.png", "Average queries/sec per baseline (bar chart)."),
        ("ttft_by_baseline.png", "Average time-to-first-token per baseline (bar chart)."),
        ("latency_by_baseline.png", "Average end-to-end latency per baseline (bar chart)."),
        ("tokens_per_sec_by_baseline.png", "Average tokens/sec per baseline (bar chart)."),
        ("bertscore_by_baseline.png", "Average completeness (BERTScore) per baseline (bar chart)."),
        ("tradeoff_latency_vs_bertscore.png", "Scatter of latency vs quality to show tradeoff."),
        ("tokens_per_sec_share_pie.png", "Pie (pizza) chart of tokens/sec share by baseline."),
        ("tokens_per_sec_pareto.png", "Pareto-like chart: tokens/sec by baseline + cumulative share."),
        # New paper-quality plots
        ("radar_performance_profile.png", "Radar/spider chart showing multi-dimensional performance profile for each baseline. Latency metrics are inverted so outer = better for all axes."),
        ("heatmap_all_metrics.png", "Heatmap of normalized metrics (0-1) across baselines. Green = better, red = worse."),
        ("latency_breakdown_stacked.png", "Stacked bar showing TTFT (prefill) vs generation (decode) time breakdown."),
        ("quality_metrics_grouped.png", "Grouped bar chart comparing all quality metrics (faithfulness, relevance, BERTScore, ROUGE-L) side by side."),
        ("efficiency_bubble_chart.png", "Bubble chart: X=latency, Y=quality, bubble size=throughput. Shows 3D efficiency frontier."),
        ("speedup_vs_no_cache.png", "Speedup factor relative to no_cache baseline. Green = faster, red = slower."),
        ("ranking_table.png", "Visual ranking table showing which baseline ranks #1, #2, etc. for each metric."),
        ("throughput_metrics_grouped.png", "Grouped comparison of throughput metrics (QPS, tokens/sec)."),
    ]
    # Add Pareto frontier plot explanations
    plots_explained.extend(pareto_plots)
    write_plot_explanations(plots_dir, plots_explained)

    summary_path = plots_dir / "latest_metrics_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"Saved plots to {plots_dir}")
    print(f"Saved summary CSV to {summary_path}")


if __name__ == "__main__":
    main()
