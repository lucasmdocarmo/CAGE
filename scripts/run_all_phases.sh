#!/bin/bash
# =============================================================================
# CAGE Full Experiment Suite
# =============================================================================
# Runs all 5 phases sequentially and generates plots
# Total estimated time: 12-18 hours
#
# Usage:
#   ./scripts/run_all_phases.sh           # Run all phases
#   ./scripts/run_all_phases.sh --quick   # Quick mode (fewer queries)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ANALYSIS_DIR="$PROJECT_DIR/analysis"

# Check for quick mode
QUICK_MODE=false
if [[ "${1:-}" == "--quick" ]]; then
    QUICK_MODE=true
    echo "*** QUICK MODE: Using reduced queries for faster testing ***"
fi

echo "=============================================="
echo "CAGE Full Experiment Suite"
echo "=============================================="
echo "Project: $PROJECT_DIR"
echo "Started: $(date)"
echo "=============================================="

# Log file for the full run
mkdir -p "$PROJECT_DIR/logs"
LOG_FILE="$PROJECT_DIR/logs/full_run_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"

{
    echo "=== Phase 1: Qwen3-4B on SQuAD v2 ==="
    bash "$SCRIPT_DIR/run_phase1.sh"
    
    echo ""
    echo "=== Phase 2: Qwen3-8B on SQuAD v2 ==="
    bash "$SCRIPT_DIR/run_phase2.sh"
    
    echo ""
    echo "=== Phase 3: Qwen2.5-7B-Instruct on SQuAD v2 ==="
    bash "$SCRIPT_DIR/run_phase3.sh"
    
    echo ""
    echo "=== Phase 4: Cross-Dataset (TriviaQA, HotpotQA) ==="
    bash "$SCRIPT_DIR/run_phase4.sh"
    
    echo ""
    echo "=== Phase 5: Speculative Decoding ==="
    bash "$SCRIPT_DIR/run_phase5.sh"
    
    echo ""
    echo "=== Generating Phase 1 Plots ==="
    python3 "$SCRIPT_DIR/generate_plots.py" \
        --results-dir "$ANALYSIS_DIR/phase1/results" \
        --plots-dir "$ANALYSIS_DIR/phase1/plots"

    echo ""
    echo "=== Generating Phase 2 Plots ==="
    python3 "$SCRIPT_DIR/generate_plots.py" \
        --results-dir "$ANALYSIS_DIR/phase2/results" \
        --plots-dir "$ANALYSIS_DIR/phase2/plots"

    echo ""
    echo "=== Generating Phase 3 Plots ==="
    python3 "$SCRIPT_DIR/generate_plots.py" \
        --results-dir "$ANALYSIS_DIR/phase3/results" \
        --plots-dir "$ANALYSIS_DIR/phase3/plots"

    echo ""
    echo "=== Generating Phase 4 TriviaQA Plots ==="
    python3 "$SCRIPT_DIR/generate_plots.py" \
        --results-dir "$ANALYSIS_DIR/phase4/results/trivia_qa" \
        --plots-dir "$ANALYSIS_DIR/phase4/plots/trivia_qa"

    echo ""
    echo "=== Generating Phase 4 HotpotQA Plots ==="
    python3 "$SCRIPT_DIR/generate_plots.py" \
        --results-dir "$ANALYSIS_DIR/phase4/results/hotpotqa" \
        --plots-dir "$ANALYSIS_DIR/phase4/plots/hotpotqa"

    echo ""
    echo "=== Generating Phase 5 Plots ==="
    python3 "$SCRIPT_DIR/generate_plots.py" \
        --results-dir "$ANALYSIS_DIR/phase5/results" \
        --plots-dir "$ANALYSIS_DIR/phase5/plots"
    
    echo ""
    echo "=============================================="
    echo "ALL PHASES COMPLETE!"
    echo "=============================================="
    echo "Artifacts: $ANALYSIS_DIR/"
    echo "Finished: $(date)"
    echo "=============================================="

} 2>&1 | tee "$LOG_FILE"
