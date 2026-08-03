#!/bin/bash
# Run the CAGE test suite against a locally-managed single-replica vLLM cluster.
# Self-locating: works regardless of where it's invoked from.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"

export VLLM_TEST_MODEL="${VLLM_TEST_MODEL:-Qwen/Qwen2.5-Coder-0.5B-Instruct}"

# Always stop the test cluster on exit (success, failure, or Ctrl-C) so a red test run
# cannot leave a replica holding the GPU/port.
trap 'python scripts/2_serving/manage_vllm_cluster.py stop >/dev/null 2>&1 || true' EXIT

# Stop any existing cluster, start a fresh single replica, run tests.
python scripts/2_serving/manage_vllm_cluster.py stop || true
python scripts/2_serving/manage_vllm_cluster.py start --model "$VLLM_TEST_MODEL" --replicas 1
python -m pytest tests/
