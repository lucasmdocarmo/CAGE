#!/bin/bash
# Run the CAGE test suite.
#
# DEFAULT (finding J12, Topic-10 walkthrough): plain LOCAL pytest -- no GPU, no vLLM
# cluster, no server. The suite is 1500+ static/unit tests that run anywhere; the old
# version hard-gated ALL of them behind the July-era manage_vllm_cluster start, so on
# any machine without a GPU serving stack ZERO tests ran.
#
# GPU/cluster mode is an EXPLICIT opt-in for the (few) tests that talk to a live
# replica: pass --with-cluster (or export CAGE_TESTS_WITH_CLUSTER=1) to start a fresh
# single-replica vLLM cluster first; it is always stopped on exit (success, failure,
# or Ctrl-C) so a red run cannot leave a replica holding the GPU/port.
#
# Usage:
#   bash scripts/checks/run_tests.sh [pytest args...]                 # local, no GPU
#   bash scripts/checks/run_tests.sh --with-cluster [pytest args...]  # GPU/cluster mode
# Self-locating: works regardless of where it's invoked from.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"

WITH_CLUSTER="${CAGE_TESTS_WITH_CLUSTER:-0}"
PYTEST_ARGS=()
for a in "$@"; do
  case "$a" in
    --with-cluster) WITH_CLUSTER=1 ;;
    *) PYTEST_ARGS+=("$a") ;;
  esac
done

# Prefer the project venv's interpreter when none is active (matches the runner scripts'
# activation discipline; system python lacks the pinned instruments).
PYTHON="python3"
if [ -z "${VIRTUAL_ENV:-}" ]; then
  for _v in .venv cage-env ../cage-env; do
    if [ -x "$_v/bin/python" ]; then PYTHON="$_v/bin/python"; break; fi
  done
fi

if [ "$WITH_CLUSTER" = "1" ]; then
  export VLLM_TEST_MODEL="${VLLM_TEST_MODEL:-Qwen/Qwen2.5-Coder-0.5B-Instruct}"
  log "GPU/cluster mode: starting a fresh single-replica vLLM cluster ($VLLM_TEST_MODEL)"
  # Always stop the test cluster on exit so a red test run cannot hold the GPU/port.
  trap '"$PYTHON" scripts/2_serving/manage_vllm_cluster.py stop >/dev/null 2>&1 || true' EXIT
  "$PYTHON" scripts/2_serving/manage_vllm_cluster.py stop || true
  "$PYTHON" scripts/2_serving/manage_vllm_cluster.py start --model "$VLLM_TEST_MODEL" --replicas 1
else
  log "local mode: running pytest with NO GPU/cluster requirement (--with-cluster opts in)"
fi

"$PYTHON" -m pytest tests/ ${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}
