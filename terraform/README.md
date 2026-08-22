# terraform/ — provider IaC, split by compatibility

`terraform/gcp/` is the **GCP port's** infrastructure-as-code (root stack + `sessions/*.tfvars` + modules). It is RETAINED, not the current campaign path: per ADR-0090 (2026-08-18) RunPod is the PRIMARY provider and GCP is a port-for-later.
RunPod is provisioned via `runpodctl` / the RunPod console — there is **no terraform for RunPod**; its lifecycle scripts live in `scripts/runpod/` (setup, teardown) with the shared provider-neutral transport in `scripts/lib/transport.sh`.
Full ordered procedure for both providers: `docs/RUNBOOK.md`. GCP-only lifecycle scripts: `scripts/gcp/`.
Any `terraform apply` here starts billing and stays gated by explicit user approval (run-approval gate; see `terraform/gcp/main.tf` header).
