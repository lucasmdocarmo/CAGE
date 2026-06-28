# CAGE Deployment Guide

**Last Updated:** 2026-04-08

---

## Deployment Options

| Option | Best For | GPU Support | Files |
|---|---|---|---|
| Local vLLM (single) | Development, most baselines | Optional | CLI commands |
| Local Docker Compose | Multi-replica + Redis | CPU/GPU | `docker-compose.yml` / `docker-compose.gpu.yml` |
| Kubernetes | Production, auto-scaling | Yes | `k8s/` |
| GCP Terraform | Cloud GPU experiments | Yes | `terraform/gcp/` |

---

## Option 1: Local Single-Instance (Phase 1 Default)

Most baselines only need one vLLM server:

```bash
# CPU mode (macOS)
export VLLM_CPU_KVCACHE_SPACE=10
export VLLM_CPU_OMP_THREADS_BIND=auto
vllm serve Qwen/Qwen3-4B --port 8000 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details

# GPU mode
vllm serve Qwen/Qwen3-4B --port 8000 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --gpu-memory-utilization 0.9
```

Run experiments against `http://localhost:8000`.

## Option 2: Local Multi-Replica (Distributed Baseline)

The Distributed Router Replicated baseline requires 3 vLLM replicas + router.

### Using manage_vllm_cluster.py

```bash
python scripts/manage_vllm_cluster.py start --model Qwen/Qwen3-4B --replicas 3
# Starts replicas on 8001/8002/8003 + router on 9000

python scripts/run_experiment.py \
  --baseline distributed \
  --model Qwen/Qwen3-4B \
  --api-base http://localhost:9000
```

### Using Docker Compose (CPU)

```bash
docker-compose up -d
# Starts: redis:6379, vllm-replica-1:8001, vllm-replica-2:8002, vllm-replica-3:8003, router:9000
```

### Using Docker Compose (GPU)

```bash
docker-compose -f docker-compose.gpu.yml up -d
```

## Option 3: Kubernetes

```bash
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/vllm-replica.yaml
kubectl apply -f k8s/router.yaml

kubectl wait --for=condition=ready pod -l app=cage-router --timeout=300s
```

See `k8s/README.md` for details.

## Option 4: GCP with Terraform

```bash
cd terraform/gcp
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with project ID, region, GPU type

terraform init
terraform plan
terraform apply

# IMPORTANT: Destroy after experiments to stop billing
terraform destroy
```

---

## Health Checks

```bash
# Single instance
curl http://localhost:8000/health

# Router
curl http://localhost:9000/health
curl http://localhost:9000/stats

# Individual replicas
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

---

## Architecture (Distributed Mode)

```
Workload Generator (run_experiment.py)
        │
        ▼
   CAGE Router (port 9000)
   [prefix-hash routing]
        │
   ┌────┼────┐
   ▼    ▼    ▼
vLLM  vLLM  vLLM    Redis
:8001 :8002 :8003    :6379
```

The router hashes `sha256(prompt[:prefix_length]) % num_replicas` to route each request to a consistent replica, maximizing per-replica prefix cache hits.

---

## Troubleshooting

**vLLM won't start:**
- Check memory: Qwen3-4B needs ~8 GB RAM; 3 replicas need ~24 GB
- Check port conflicts: `lsof -i :8000`
- Check logs: `logs/vllm/`

**Router not forwarding:**
- Verify replicas are healthy: `curl http://localhost:800{1,2,3}/health`
- Check router stats: `curl http://localhost:9000/stats`

**Docker out of memory:**
- Increase Docker Desktop memory allocation to 16+ GB
- Reduce `VLLM_CPU_KVCACHE_SPACE` per replica
