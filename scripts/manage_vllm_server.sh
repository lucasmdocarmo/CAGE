#!/bin/bash
# =============================================================================
# vLLM Server Management Script
# =============================================================================
# Manages the vLLM inference server for CAGE experiments
#
# Usage:
#   ./scripts/manage_vllm_server.sh start <model>
#   ./scripts/manage_vllm_server.sh stop
#   ./scripts/manage_vllm_server.sh restart <model>
#   ./scripts/manage_vllm_server.sh status
# =============================================================================

set -euo pipefail

PORT=${VLLM_PORT:-8000}
LOG_DIR="logs/vllm"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

mkdir -p "$LOG_DIR"

get_vllm_pid() {
    pgrep -f "vllm serve" || true
}

get_loaded_model() {
    curl -s http://localhost:${PORT}/v1/models 2>/dev/null | \
        python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data'][0]['id'] if data.get('data') else '')" 2>/dev/null || echo ""
}
get_server_prefix_cache_mode() {
    local pid=$(get_vllm_pid)
    if [ -z "$pid" ]; then
        echo "unknown"
        return 1
    fi

    local cmd=$(ps -p "$pid" -o command= 2>/dev/null || true)
    if [[ "$cmd" == *"--no-enable-prefix-caching"* ]]; then
        echo "disabled"
        return 0
    fi
    if [[ "$cmd" == *"--enable-prefix-caching"* ]]; then
        echo "enabled"
        return 0
    fi
    echo "unknown"
    return 0
}

server_has_prefix_cache() {
    local mode
    mode=$(get_server_prefix_cache_mode)
    [[ "$mode" == "enabled" ]]
}

start_server() {
    local model="$1"
    local cache_flag="--enable-prefix-caching"
    local want_prefix_cache=true

    if [ "${2:-}" = "--no-prefix-cache" ]; then
        cache_flag="--no-enable-prefix-caching"
        want_prefix_cache=false
    fi
    
    echo -e "${YELLOW}Starting vLLM server with model: $model${NC}"
    
    # Check if already running
    local pid=$(get_vllm_pid)
    if [ -n "$pid" ]; then
        local loaded_model=$(get_loaded_model)
        local prefix_cache_mode
        prefix_cache_mode=$(get_server_prefix_cache_mode)
        local has_prefix_cache="$prefix_cache_mode"
        if [ "$prefix_cache_mode" = "enabled" ]; then
            has_prefix_cache=true
        elif [ "$prefix_cache_mode" = "disabled" ]; then
            has_prefix_cache=false
        fi

        if [ "$loaded_model" = "$model" ] && [ "$has_prefix_cache" = "$want_prefix_cache" ]; then
            echo -e "${GREEN}✓ Server already running with correct model and cache mode ($model)${NC}"
            return 0
        else
            echo -e "${RED}✗ Server state does not match requested model/cache mode${NC}"
            echo -e "${YELLOW}  Loaded model: $loaded_model | prefix cache: $has_prefix_cache${NC}"
            echo -e "${YELLOW}  Requested model: $model | prefix cache: $want_prefix_cache${NC}"
            echo -e "${YELLOW}  Stopping and restarting...${NC}"
            stop_server
            sleep 2
        fi
    fi
    
    # Start server
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="$LOG_DIR/vllm_${model//\//_}_${timestamp}.log"
    
    # Optional server-side KV-cache compression for the compressed_cag baseline:
    #   VLLM_KV_CACHE_DTYPE=fp8 ./scripts/manage_vllm_server.sh restart <model>
    local kv_dtype_flag=""
    if [ -n "${VLLM_KV_CACHE_DTYPE:-}" ]; then
        kv_dtype_flag="--kv-cache-dtype ${VLLM_KV_CACHE_DTYPE}"
        echo "KV-cache compression enabled: --kv-cache-dtype ${VLLM_KV_CACHE_DTYPE}"
    fi
    # Set VLLM_SERVER_DEV_MODE=1 in the environment to enable POST /reset_prefix_cache
    # (used by --reset-cache-between-trials for cold-start-per-trial measurement).

    # Speculative decoding is a LAUNCH-time config in current vLLM (the old
    # --speculative-model flag is deprecated). Pass a JSON via VLLM_SPECULATIVE_CONFIG, e.g.
    #   VLLM_SPECULATIVE_CONFIG='{"method":"ngram","num_speculative_tokens":5}'
    #   VLLM_SPECULATIVE_CONFIG='{"model":"Qwen/Qwen3-0.6B","num_speculative_tokens":5}'
    local spec_flag=""
    if [ -n "${VLLM_SPECULATIVE_CONFIG:-}" ]; then
        spec_flag="--speculative-config ${VLLM_SPECULATIVE_CONFIG}"
        echo "Speculative decoding enabled: --speculative-config ${VLLM_SPECULATIVE_CONFIG}"
    fi

    # A 24GB L4 cannot hold Qwen3-8B's default max_model_len (40960) KV cache after
    # the ~15GB of weights (vLLM aborts: "needed 5.62 GiB > available 3.12 GiB").
    # Cap context length (override via VLLM_MAX_MODEL_LEN) and raise memory
    # utilization (override via VLLM_GPU_MEMORY_UTILIZATION). CAGE prompts are a few
    # thousand tokens, so 8192 is ample headroom.
    local max_len_flag="--max-model-len ${VLLM_MAX_MODEL_LEN:-8192}"
    local gpu_mem_flag="--gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
    # Optional eager mode: skip torch.compile + CUDA-graph capture for much faster,
    # more reliable startup (esp. on smaller GPUs like the L4, where compile takes
    # 2-3 min and recompiles per prefix-cache config). Serving is uniform across all
    # baselines, so CAGE's *comparative* metrics stay valid. Set VLLM_ENFORCE_EAGER=1.
    local eager_flag=""
    if [ "${VLLM_ENFORCE_EAGER:-0}" = "1" ]; then
        eager_flag="--enforce-eager"
        echo "Eager mode ON: --enforce-eager"
    fi
    echo "Context/memory caps: $max_len_flag $gpu_mem_flag $eager_flag"

    echo "Starting vLLM server (logging to $log_file)..."
    nohup vllm serve "$model" \
        --port "$PORT" \
        $cache_flag \
        $kv_dtype_flag \
        $spec_flag \
        $max_len_flag \
        $gpu_mem_flag \
        $eager_flag \
        --enable-prompt-tokens-details \
        > "$log_file" 2>&1 &
    
    local server_pid=$!
    echo "Server PID: $server_pid"
    
    # Wait for server to be ready. vLLM's torch.compile + CUDA-graph capture can take
    # 2-3 min on smaller GPUs (e.g. L4), so 60s is too short and aborts the suite under
    # set -e. Allow 5 min by default; override with VLLM_START_TIMEOUT.
    echo "Waiting for server to start..."
    local max_wait=${VLLM_START_TIMEOUT:-300}
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
            local loaded=$(get_loaded_model)
            if [ "$loaded" = "$model" ]; then
                echo -e "${GREEN}✓ Server ready with model: $model${NC}"
                echo "  View logs: tail -f $log_file"
                return 0
            fi
        fi
        sleep 2
        waited=$((waited + 2))
        echo -n "."
    done
    
    echo -e "\n${RED}✗ Server failed to start within ${max_wait}s${NC}"
    echo "Check logs: $log_file"
    return 1
}

stop_server() {
    echo -e "${YELLOW}Stopping vLLM server...${NC}"

    # vLLM v1 spawns the engine in a SEPARATE process named "VLLM::EngineCore" that
    # is NOT matched by "vllm serve". The old code only killed "vllm serve", so each
    # restart ORPHANED the EngineCore worker, which kept ~all of the GPU memory and
    # made the next start fail ("Engine core initialization failed"). Kill the whole
    # vLLM process group AND anything still holding the GPU.
    pkill -f "vllm serve"          2>/dev/null || true
    pkill -f "VLLM::EngineCore"    2>/dev/null || true
    pkill -f "vllm.v1.engine.core" 2>/dev/null || true
    sleep 2
    pkill -9 -f "vllm serve"          2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore"    2>/dev/null || true
    pkill -9 -f "vllm.v1.engine.core" 2>/dev/null || true

    # Belt-and-suspenders: kill any remaining process still holding the GPU.
    local held
    held=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    for p in $held; do kill -9 "$p" 2>/dev/null || true; done
    sleep 2

    local gpu_mem
    gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)
    echo -e "${GREEN}✓ Server stopped${NC} (GPU mem used: ${gpu_mem:-n/a})"
}

status_server() {
    local pid=$(get_vllm_pid)
    
    if [ -z "$pid" ]; then
        echo -e "${RED}✗ vLLM server is NOT running${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✓ vLLM server is running${NC}"
    echo "  PID: $pid"
    
    local loaded_model=$(get_loaded_model)
    if [ -n "$loaded_model" ]; then
        echo "  Model: $loaded_model"
        echo "  Port: $PORT"
        echo "  Health: http://localhost:${PORT}/health"
    else
        echo -e "${YELLOW}  Warning: Unable to query loaded model${NC}"
    fi
}

case "${1:-}" in
    start)
        if [ -z "${2:-}" ]; then
            echo "Usage: $0 start <model> [--no-prefix-cache]"
            echo "Example: $0 start Qwen/Qwen3-4B"
            exit 1
        fi
        start_server "$2" "${3:-}"
        ;;
    stop)
        stop_server
        ;;
    restart)
        if [ -z "${2:-}" ]; then
            echo "Usage: $0 restart <model> [--no-prefix-cache]"
            exit 1
        fi
        stop_server
        sleep 2
        start_server "$2" "${3:-}"
        ;;
    status)
        status_server
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status} [model]"
        echo ""
        echo "Commands:"
        echo "  start <model>   - Start vLLM server with specified model"
        echo "  stop            - Stop vLLM server"
        echo "  restart <model> - Restart vLLM server with specified model"
        echo "  status          - Check vLLM server status"
        echo ""
        echo "Examples:"
        echo "  $0 start Qwen/Qwen3-4B"
        echo "  $0 status"
        echo "  $0 stop"
        exit 1
        ;;
esac
