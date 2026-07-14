#!/bin/bash
# Continuous status logger for CAGE runs (Phase 2 / Phase 3 live tracking).
# Appends a compact one-line status snapshot every INTERVAL seconds to a timeline
# log, so progress is recorded at fine granularity without a held SSH connection.
#
# Usage:
#   nohup bash scripts/5_observability/run_status_logger.sh [RESULTS_DIR] [RUN_LOG] [OUT] [INTERVAL] &
# Superseded by the observe_run.py sidecar (auto-launched by cloud_run.sh); kept as a lightweight
# manual logger. Defaults: results=$CAGE_RUN_ROOT/baselines, run.log=~/run.log, out=~/status_timeline.log, interval=20
RESULTS="${1:-${CAGE_RUN_ROOT:-$HOME/CAGE/results}/baselines}"
RUNLOG="${2:-$HOME/run.log}"
OUT="${3:-$HOME/status_timeline.log}"
INTERVAL="${4:-20}"

strip_ansi() { sed -E 's/\x1b\[[0-9;]*m//g'; }

while true; do
  finished=$(ls "$RESULTS"/*/aggregated_metrics.json 2>/dev/null | wc -l | tr -d ' ')
  current=$(grep ">>> Running baseline" "$RUNLOG" 2>/dev/null | tail -1 | sed -E 's/.*baseline:[[:space:]]*//' | strip_ansi)
  errors=$(grep -ciE "Traceback|Error running experiment|CUDA out of memory" "$RUNLOG" 2>/dev/null)
  suite_done=$(grep -c "suite complete" "$RUNLOG" 2>/dev/null)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  echo "$(date +%H:%M:%S) finished=${finished} running=${current:-none} errors=${errors} suite_done=${suite_done} gpu=[${gpu}]" >> "$OUT"
  sleep "$INTERVAL"
done
