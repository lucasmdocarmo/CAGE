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
    
    echo "Starting vLLM server (logging to $log_file)..."
    nohup vllm serve "$model" \
        --port "$PORT" \
        $cache_flag \
        --enable-prompt-tokens-details \
        > "$log_file" 2>&1 &
    
    local server_pid=$!
    echo "Server PID: $server_pid"
    
    # Wait for server to be ready
    echo "Waiting for server to start..."
    local max_wait=60
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
    
    local pid=$(get_vllm_pid)
    if [ -z "$pid" ]; then
        echo -e "${YELLOW}No vLLM server running${NC}"
        return 0
    fi
    
    echo "Killing PID: $pid"
    pkill -f "vllm serve" || true
    
    # Wait for shutdown
    local max_wait=10
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if [ -z "$(get_vllm_pid)" ]; then
            echo -e "${GREEN}✓ Server stopped${NC}"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    
    # Force kill if still running
    echo -e "${YELLOW}Force killing...${NC}"
    pkill -9 -f "vllm serve" || true
    echo -e "${GREEN}✓ Server killed${NC}"
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
