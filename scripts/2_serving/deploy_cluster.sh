#!/bin/bash
# Order:     stage 2 — multi-node cluster bring-up (side path), before 3_run
# Objective: Deploy a multi-node vLLM cluster: local Docker Compose / generic-k8s side paths + the confirm-gated terraform/gcp stack (its only cloud surface; unused on RunPod)
# Cloud:     gcp
# CAGE Framework - Multi-Node Cluster Deployment Script
# Supports: local (Docker), Kubernetes, GCP (Terraform)
#
# PROVISIONING (2026-08-02 charter): the GCP campaign path now provisions via
# terraform/gcp/ (sessions/*.tfvars; `terraform apply` is GATED by
# explicit user approval — see terraform/gcp/main.tf header). The `gcp` command here
# drives that same stack interactively and fail-closed (confirm prompt);
# never wire it into automation. `local`/`k8s` remain the SSH-config +
# neocloud-manual side paths.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_ROOT/scripts/lib/_common.sh"

# GCP terraform stack (terraform/gcp/ since the provider split; sessions tfvars select the shape).
TF_DIR="$PROJECT_ROOT/terraform/gcp"

usage() {
    cat << EOF
CAGE Multi-Node Deployment Script

Usage: $0 <command> [options]

Commands:
  local           REMOVED 2026-10-07 (ADR-0147): refuses
  k8s             REMOVED 2026-10-07 (ADR-0147): refuses
  gcp             Deploy to GCP with Terraform (terraform/gcp/ stack; apply is confirm-gated)
  status          Check deployment status
  destroy [tgt]   Tear down deployment (tgt: local|k8s|gcp|all, default all)

Options:
  --replicas N    Number of vLLM replicas (default: 3)
  --model NAME    Model to serve (default: Qwen/Qwen3-4B)
  --gpu           Enable GPU support (local/k8s)
  --dry-run       Show what would be done without executing

Examples:
  $0 local --replicas 3
  $0 k8s --replicas 3 --gpu
  $0 gcp --replicas 3 --model Qwen/Qwen3-8B
  $0 status
  $0 destroy local

EOF
}

log_info()  { printf '[INFO] %s\n' "$1"; }
log_warn()  { printf '[WARN] %s\n' "$1" >&2; }
log_error() { printf '[ERROR] %s\n' "$1" >&2; }

# Default values
REPLICAS=3
MODEL="Qwen/Qwen3-4B"
GPU_ENABLED=false
DRY_RUN=false
COMMAND=""
TARGET=""

# Parse arguments. The FIRST bare word is the command; a SECOND bare word is the
# target (so `destroy local` works — previously the second word overwrote COMMAND
# and `destroy local` silently ran a local DEPLOY).
while [[ $# -gt 0 ]]; do
    case "$1" in
        local|k8s|gcp|status|destroy)
            if [[ -z "$COMMAND" ]]; then
                COMMAND="$1"
            elif [[ -z "$TARGET" ]]; then
                TARGET="$1"
            else
                log_error "unexpected extra argument: $1"
                usage
                exit 1
            fi
            shift
            ;;
        --replicas)
            REPLICAS="${2:?--replicas needs a value}"
            shift 2
            ;;
        --model)
            MODEL="${2:?--model needs a value}"
            shift 2
            ;;
        --gpu)
            GPU_ENABLED=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$COMMAND" ]]; then
    usage
    exit 1
fi

# REMOVED (2026-10-07, ADR-0147): the local and k8s paths deployed the CAGE
# prefix router container (src/orchestration/router.py), which reported a KV
# transfer it simulated and left src. Both commands refuse; gcp, status and
# destroy are untouched.
ROUTER_REMOVED_MSG="the CAGE prefix router left src on 2026-10-07 (ADR-0147); the local and k8s cluster paths have no serving path without it. Use scripts/2_serving/manage_vllm_server.sh (single instance) or manage_vllm_pd.sh (prefill/decode pair)."

deploy_local() {
    log_error "REFUSING local deploy: $ROUTER_REMOVED_MSG"
    exit 1
}

deploy_k8s() {
    log_error "REFUSING k8s deploy: $ROUTER_REMOVED_MSG"
    exit 1
}

# Deploy to GCP with Terraform (terraform/gcp/ stack; sessions tfvars select the shape).
deploy_gcp() {
    log_info "Deploying CAGE cluster to GCP with $REPLICAS replicas..."

    require_cmd terraform "install Terraform first"
    require_cmd gcloud "install the Google Cloud SDK first"

    cd "$TF_DIR"

    # Check for tfvars
    if [[ ! -f "terraform.tfvars" ]]; then
        log_error "terraform.tfvars not found in terraform/gcp/. Copy terraform.tfvars.example and fill in your values."
        log_error "Campaign sessions additionally use -var-file=sessions/<group>.tfvars (see terraform/gcp/main.tf)."
        exit 1
    fi

    if "$DRY_RUN"; then
        terraform plan -var="num_replicas=$REPLICAS" -var="model_name=$MODEL"
        return
    fi

    # RUN-APPROVAL GATE (standing discipline, 2026-07-16): apply starts billing.
    # confirm() is fail-closed -- with no TTY the answer is NO unless CAGE_ASSUME_YES=1.
    terraform init
    terraform plan -var="num_replicas=$REPLICAS" -var="model_name=$MODEL"
    if ! confirm "terraform APPLY the plan above (BILLING STARTS)?"; then
        log_warn "apply NOT confirmed -- nothing created, still \$0."
        exit 1
    fi
    terraform apply -var="num_replicas=$REPLICAS" -var="model_name=$MODEL" -auto-approve

    # Show outputs
    log_info "Deployment complete!"
    terraform output
}

# Check deployment status
check_status() {
    log_info "Checking deployment status..."

    # Check local Docker
    if [[ -f "$PROJECT_ROOT/docker-compose.generated.yml" ]] \
        && docker compose -f "$PROJECT_ROOT/docker-compose.generated.yml" ps > /dev/null 2>&1; then
        log_info "Local Docker deployment:"
        docker compose -f "$PROJECT_ROOT/docker-compose.generated.yml" ps
    fi

    # Check Kubernetes
    if command -v kubectl > /dev/null 2>&1 && kubectl get pods -l app=cage-router > /dev/null 2>&1; then
        log_info "Kubernetes deployment:"
        kubectl get pods -l app=cage-router -l app=vllm-replica-1 -l app=cage-redis
    fi

    # Check GCP Terraform (root stack)
    if [[ -f "$TF_DIR/terraform.tfstate" ]]; then
        log_info "GCP deployment:"
        cd "$TF_DIR"
        terraform output 2>/dev/null || true
    fi
}

# Destroy deployment. (Explicit if-blocks, not `;;&` fallthrough: `;;&` is bash-4-only
# and choked `bash -n` on macOS bash 3.2.)
destroy_deployment() {
    local target="${1:-all}"

    log_info "Destroying deployment: $target"

    if "$DRY_RUN"; then
        log_info "[DRY RUN] Would destroy $target deployment"
        return
    fi

    if [[ "$target" == "local" || "$target" == "all" ]]; then
        if [[ -f "$PROJECT_ROOT/docker-compose.generated.yml" ]]; then
            docker compose -f "$PROJECT_ROOT/docker-compose.generated.yml" down -v
            rm -f "$PROJECT_ROOT/docker-compose.generated.yml"
            log_info "Local deployment destroyed"
        fi
    fi

    if [[ "$target" == "k8s" || "$target" == "all" ]]; then
        if command -v kubectl > /dev/null 2>&1 && [[ -d "$PROJECT_ROOT/k8s" ]] \
            && kubectl get namespace default > /dev/null 2>&1; then
            kubectl delete -f "$PROJECT_ROOT/k8s/" --ignore-not-found=true
            log_info "Kubernetes deployment destroyed"
        fi
    fi

    if [[ "$target" == "gcp" || "$target" == "all" ]]; then
        if [[ -f "$TF_DIR/terraform.tfstate" ]]; then
            cd "$TF_DIR"
            log_info "Terraform destroy of the root stack. Data flush on destroy relies on each"
            log_info "node's shutdown-script (metadata) syncing to GCS within its shutdown window."
            log_info "For a single VM, prefer 'scripts/gcp/teardown_vm.sh <vm> <zone>', which"
            log_info "verifies this run's GCS log sentinel AND pulls results local BEFORE deleting"
            log_info "(fail-closed). PULL RESULTS LOCAL FIRST -- teardown is irreversible."
            # Fail-closed confirm replaces the old 'sleep 5 then destroy anyway' window.
            if ! confirm "terraform DESTROY the GCP cluster (results pulled local already)?"; then
                log_warn "destroy NOT confirmed -- GCP deployment left as-is (STILL BILLING)."
                return 1
            fi
            terraform destroy -auto-approve
            log_info "GCP deployment destroyed"
        fi
    fi
}

# Main command dispatch
case "$COMMAND" in
    local)
        deploy_local
        ;;
    k8s)
        deploy_k8s
        ;;
    gcp)
        deploy_gcp
        ;;
    status)
        check_status
        ;;
    destroy)
        destroy_deployment "${TARGET:-all}"
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac
