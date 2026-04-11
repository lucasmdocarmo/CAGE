# K8s quickstart (local kind/minikube)

Apply in order:

```bash
kubectl apply -f redis.yaml
kubectl apply -f vllm-replica.yaml  # creates 3 Deployments + Services (8001-8003)
kubectl apply -f router.yaml       # exposes router on NodePort 30090
```

Note: vLLM CPU images are heavy; adjust resources as needed.
