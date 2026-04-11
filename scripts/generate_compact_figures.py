#!/usr/bin/env python3
"""
Generate compact A4-friendly composite figures from Phase 1 plots.

Creates multiple layouts:
- 3 images (1 row)
- 4 images (2x2)
- 5 images (2 rows)
- 6 images (2x3 or 3x2)
- Various 2-column grouped layouts

Output: analysis/phase1/plots/compact/
"""

import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PLOTS_DIR = PROJECT_DIR / "analysis" / "phase1" / "images"
OUTPUT_DIR = PLOTS_DIR / "compact"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# A4 dimensions at 300 DPI (portrait)
A4_WIDTH_INCHES = 8.27
A4_HEIGHT_INCHES = 11.69
DPI = 300

# Available plots
PLOTS = {
    "latency": "01_latency_comparison.png",
    "throughput": "02_throughput_comparison.png",
    "quality": "03_quality_comparison.png",
    "breakdown": "04_latency_breakdown.png",
    "speedup": "05_speedup_vs_nocache.png",
    "pareto": "06_pareto_latency_vs_quality.png",
    "radar": "07_radar_profile.png",
    "heatmap": "08_heatmap.png",
    "variance": "09_boxplots_variance.png",
    "summary": "10_summary_table.png",
    "cache_ttft": "11_cache_hit_vs_ttft.png",
    "quality_perf": "12_quality_performance_matrix.png",
    "overhead": "13_overhead_decomposition.png",
    "ranking": "14_efficiency_ranking.png",
    "context": "15_context_type_impact.png",
}


def create_composite(plot_keys, layout, output_name, title=None, figsize=None):
    """
    Create a composite figure from multiple plots.
    
    Args:
        plot_keys: List of keys from PLOTS dict
        layout: Tuple (rows, cols)
        output_name: Output filename
        title: Optional figure title
        figsize: Optional (width, height) in inches
    """
    rows, cols = layout
    n_plots = len(plot_keys)
    
    if figsize is None:
        # Default to A4 proportions, scaled for content
        figsize = (A4_WIDTH_INCHES, A4_HEIGHT_INCHES * 0.8)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=DPI)
    
    # Flatten axes for easy iteration
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Load and display each plot
    for idx, key in enumerate(plot_keys):
        if idx >= len(axes):
            break
        
        img_path = PLOTS_DIR / PLOTS[key]
        if img_path.exists():
            img = mpimg.imread(str(img_path))
            axes[idx].imshow(img)
            axes[idx].axis('off')
        else:
            axes[idx].text(0.5, 0.5, f"Missing: {key}", 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.95)
    
    output_path = OUTPUT_DIR / output_name
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Created: {output_path}")
    return output_path


def create_labeled_split(plot_keys, labels, output_name, title=None, figsize=None):
    """
    Create a 1-row composite with panel labels (A, B, C, D, etc.).

    Args:
        plot_keys: List of keys from PLOTS dict
        labels: List of panel labels (e.g. ["A", "B"] or ["C", "D"])
        output_name: Output filename
        title: Optional figure title
        figsize: Optional (width, height) in inches
    """
    n_plots = len(plot_keys)

    if figsize is None:
        figsize = (A4_WIDTH_INCHES * 1.5, A4_HEIGHT_INCHES * 0.45)

    fig, axes = plt.subplots(1, n_plots, figsize=figsize, dpi=DPI)
    if n_plots == 1:
        axes = [axes]

    for idx, (key, label) in enumerate(zip(plot_keys, labels)):
        img_path = PLOTS_DIR / PLOTS[key]
        if img_path.exists():
            img = mpimg.imread(str(img_path))
            axes[idx].imshow(img)
            axes[idx].axis('off')
            # Add panel label below center
            axes[idx].text(
                0.5, -0.04, f'({label.lower()})',
                transform=axes[idx].transAxes,
                fontsize=16, fontweight='normal', va='top', ha='center',
                fontfamily='serif',
            )
        else:
            axes[idx].text(
                0.5, 0.5, f"Missing: {key}",
                ha='center', va='center', transform=axes[idx].transAxes,
            )
            axes[idx].axis('off')

    if title:
        fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)

    plt.tight_layout()
    output_path = OUTPUT_DIR / output_name
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Created: {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("Generating Compact A4-Friendly Composite Figures")
    print("=" * 60)
    
    # =========================================================================
    # 3 IMAGES - Single row (best for key results)
    # =========================================================================
    print("\n--- 3-Image Layouts ---")
    
    # Key performance metrics
    create_composite(
        ["latency", "throughput", "quality"],
        layout=(1, 3),
        output_name="compact_3_performance_overview.png",
        title="Phase 1: Performance Overview",
        figsize=(A4_WIDTH_INCHES, A4_HEIGHT_INCHES * 0.35)
    )
    
    # Analysis trio
    create_composite(
        ["pareto", "heatmap", "speedup"],
        layout=(1, 3),
        output_name="compact_3_analysis.png",
        title="Phase 1: Trade-off Analysis",
        figsize=(A4_WIDTH_INCHES, A4_HEIGHT_INCHES * 0.35)
    )
    
    # =========================================================================
    # 4 IMAGES - 2x2 grid
    # =========================================================================
    print("\n--- 4-Image Layouts (2x2) ---")
    
    # Core results
    create_composite(
        ["latency", "throughput", "quality", "pareto"],
        layout=(2, 2),
        output_name="compact_4_core_results.png",
        title="Phase 1: Core Results",
        figsize=(A4_WIDTH_INCHES, A4_HEIGHT_INCHES * 0.65)
    )
    
    # Detailed analysis
    create_composite(
        ["breakdown", "speedup", "variance", "heatmap"],
        layout=(2, 2),
        output_name="compact_4_detailed_analysis.png",
        title="Phase 1: Detailed Analysis",
        figsize=(A4_WIDTH_INCHES, A4_HEIGHT_INCHES * 0.65)
    )
    
    # New metrics (additional plots)
    create_composite(
        ["cache_ttft", "quality_perf", "overhead", "context"],
        layout=(2, 2),
        output_name="compact_4_additional_metrics.png",
        title="Phase 1: Additional Metrics",
        figsize=(A4_WIDTH_INCHES, A4_HEIGHT_INCHES * 0.65)
    )
    
    # =========================================================================
    # 5 IMAGES - 2 rows (3 top, 2 bottom or 2-3 split)
    # =========================================================================
    print("\n--- 5-Image Layouts ---")
    
    # Using 2x3 grid with one empty
    create_composite(
        ["latency", "throughput", "quality", "pareto", "heatmap"],
        layout=(2, 3),
        output_name="compact_5_main_results.png",
        title="Phase 1: Main Results Summary",
        figsize=(A4_WIDTH_INCHES, A4_HEIGHT_INCHES * 0.6)
    )
    
    # =========================================================================
    # 6 IMAGES - 2x3 grid
    # =========================================================================
    print("\n--- 6-Image Layouts (2x3) ---")
    
    # Complete overview
    create_composite(
        ["latency", "throughput", "quality", "breakdown", "pareto", "heatmap"],
        layout=(2, 3),
        output_name="compact_6_complete_overview.png",
        title="Phase 1: Complete Results Overview",
        figsize=(A4_WIDTH_INCHES, A4_HEIGHT_INCHES * 0.7)
    )
    
    # 3x2 layout (portrait oriented)
    create_composite(
        ["latency", "quality", "throughput", "pareto", "speedup", "heatmap"],
        layout=(3, 2),
        output_name="compact_6_portrait_3x2.png",
        title="Phase 1: Results (Portrait Layout)",
        figsize=(A4_WIDTH_INCHES, A4_HEIGHT_INCHES * 0.85)
    )
    
    # =========================================================================
    # 2-COLUMN GROUPED LAYOUTS
    # =========================================================================
    print("\n--- 2-Column Grouped Layouts ---")
    
    # Performance vs Quality (4 plots, 2 each side conceptually)
    create_composite(
        ["latency", "throughput", "quality", "pareto"],
        layout=(2, 2),
        output_name="compact_2col_perf_vs_quality.png",
        title="Performance (Left) vs Quality Analysis (Right)",
        figsize=(A4_WIDTH_INCHES, A4_HEIGHT_INCHES * 0.6)
    )
    
    # Latency deep dive (2 columns)
    create_composite(
        ["latency", "breakdown", "speedup", "variance"],
        layout=(2, 2),
        output_name="compact_2col_latency_deepdive.png",
        title="Latency Analysis: Comparison & Breakdown",
        figsize=(A4_WIDTH_INCHES, A4_HEIGHT_INCHES * 0.6)
    )
    
    # Cache analysis
    create_composite(
        ["cache_ttft", "quality_perf", "heatmap", "context"],
        layout=(2, 2),
        output_name="compact_2col_cache_analysis.png",
        title="Cache Impact Analysis",
        figsize=(A4_WIDTH_INCHES, A4_HEIGHT_INCHES * 0.6)
    )
    
    # =========================================================================
    # FULL PAGE LAYOUTS (all 15 or key selection)
    # =========================================================================
    print("\n--- Full Page Layouts ---")
    
    # 8 key plots (2x4)
    create_composite(
        ["latency", "throughput", "quality", "breakdown",
         "pareto", "heatmap", "speedup", "variance"],
        layout=(4, 2),
        output_name="compact_8_full_page_portrait.png",
        title="Phase 1: Comprehensive Results",
        figsize=(A4_WIDTH_INCHES, A4_HEIGHT_INCHES * 0.95)
    )
    
    # 9 plots (3x3) - most important
    create_composite(
        ["latency", "throughput", "quality",
         "breakdown", "pareto", "heatmap",
         "speedup", "variance", "context"],
        layout=(3, 3),
        output_name="compact_9_grid.png",
        title="Phase 1: Key Results Grid",
        figsize=(A4_WIDTH_INCHES, A4_HEIGHT_INCHES * 0.9)
    )
    
    # 12 plots (4x3) - comprehensive
    create_composite(
        ["latency", "throughput", "quality",
         "breakdown", "pareto", "heatmap",
         "speedup", "variance", "ranking",
         "cache_ttft", "quality_perf", "context"],
        layout=(4, 3),
        output_name="compact_12_comprehensive.png",
        title="Phase 1: All Key Metrics",
        figsize=(A4_WIDTH_INCHES, A4_HEIGHT_INCHES * 0.95)
    )
    
    # =========================================================================
    # PRESENTATION-READY (specific themes)
    # =========================================================================
    print("\n--- Presentation-Ready Themed Layouts ---")
    
    # Executive summary (3 most important)
    create_composite(
        ["latency", "quality", "pareto"],
        layout=(1, 3),
        output_name="compact_exec_summary.png",
        title="Executive Summary: CAG vs RAG Performance",
        figsize=(A4_WIDTH_INCHES, A4_HEIGHT_INCHES * 0.3)
    )
    
    # Technical deep dive
    create_composite(
        ["breakdown", "speedup", "variance", "overhead", "cache_ttft", "heatmap"],
        layout=(2, 3),
        output_name="compact_technical_deepdive.png",
        title="Technical Analysis: Latency Decomposition",
        figsize=(A4_WIDTH_INCHES, A4_HEIGHT_INCHES * 0.65)
    )
    
    # Quality focus
    create_composite(
        ["quality", "quality_perf", "context"],
        layout=(1, 3),
        output_name="compact_quality_focus.png",
        title="Quality Metrics: Faithfulness & Context Impact",
        figsize=(A4_WIDTH_INCHES, A4_HEIGHT_INCHES * 0.35)
    )
    
    # =========================================================================
    # SPLIT AB / CD LAYOUTS (for paper figures)
    # =========================================================================
    print("\n--- Split AB/CD Layouts ---")
    
    create_labeled_split(
        ["latency", "throughput"],
        labels=["A", "B"],
        output_name="compact_4_core_results_ab.png",
        title="Phase 1: Core Results",
        figsize=(A4_WIDTH_INCHES * 1.5, A4_HEIGHT_INCHES * 0.45)
    )
    
    create_labeled_split(
        ["quality", "pareto"],
        labels=["A", "B"],
        output_name="compact_4_core_results_cd.png",
        title=None,
        figsize=(A4_WIDTH_INCHES * 1.8, A4_HEIGHT_INCHES * 0.55)
    )
    
    print("\n" + "=" * 60)
    print(f"All compact figures saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
