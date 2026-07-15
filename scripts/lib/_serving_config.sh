#!/bin/bash
# =============================================================================
# CAGE uniform serving configuration  (single source of truth)
# =============================================================================
# Sourced by EVERY single-node baseline-tree driver -- run_baselines.sh (core baselines),
# run_compression.sh (compression 2x2), run_speculative_matrix.sh (speculative 2x2) --
# so all three trees serve under IDENTICAL conditions and cross-mechanism comparisons are
# FAIR. Consumed by scripts/2_serving/manage_vllm_server.sh, which reads these env vars when it launches
# vLLM. The phase3 cluster path (manage_vllm_cluster.py) consumes the SAME VLLM_* env via
# build_serve_args() with fallbacks mirroring this file -- gap closed 2026-07-15 (task #63);
# source this file before cluster bring-up so overrides propagate.
#
# WHY THIS EXISTS (Option A, 2026-07-14): previously the trees diverged --
#   core:        non-eager, max_len 8192, gpu-mem-util 0.92
#   compression: --enforce-eager, max_len 4096, 0.92
#   speculative: --enforce-eager, max_len 4096, 0.90
# so a cross-tree serving delta (esp. TPOT) mixed the MECHANISM with an eager-vs-compiled +
# context-length + memory-util artifact, and cross-tree numbers were only comparable within a
# tree. Holding these three variables identical removes that confound: a cross-mechanism
# serving/quality delta is now attributable to the mechanism, not the serving regime.
#
# CONFOUND-CONTROLLED VARIABLES (held IDENTICAL across all trees):
#   VLLM_ENFORCE_EAGER=0        non-eager (CUDA graphs ON) -- production-realistic decode; the
#                               eager penalty that inflated lever-tree TPOT is removed.
#   VLLM_MAX_MODEL_LEN=4096     ample for SQuAD (contexts are short paragraphs); uniform so the
#                               KV-planning/chunked-prefill regime is identical across trees.
#
# THE SWEPT AXIS (the memory-pressure trade-off distribution objective):
#   VLLM_GPU_MEMORY_UTILIZATION default 0.90 is the uniform OPERATING POINT for a like-for-like
#   baseline comparison. The memory-pressure study OVERRIDES this one variable to trace the
#   trade-off, e.g.:  for p in 0.80 0.85 0.90 0.95; do
#                        VLLM_GPU_MEMORY_UTILIZATION=$p bash scripts/3_run/cloud_run.sh ...; done
#   Holding eager + max_len fixed means the pressure sweep varies ONLY memory, cleanly.
#
# Every value is overridable (:-default), so the pre-flight can fall back to eager for a single
# tree if a cell OOMs non-eager on the 24GB L4 (e.g. the speculative tree):
#   VLLM_ENFORCE_EAGER=1 bash scripts/3_run/run_speculative_matrix.sh <model>
# Such a fallback is a DELIBERATE, RECORDED deviation -- the run manifest captures the actual
# enforce_eager/max_model_len used, so any non-uniform cell is visible in provenance.
# =============================================================================

export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

echo "[cage] serving config: enforce_eager=${VLLM_ENFORCE_EAGER} max_model_len=${VLLM_MAX_MODEL_LEN} gpu_mem_util=${VLLM_GPU_MEMORY_UTILIZATION} (uniform across trees; mem-util is the swept axis)"
