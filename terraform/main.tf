# =============================================================================
# !! APPLY IS GATED BY EXPLICIT USER APPROVAL — NEVER APPLY AUTOMATICALLY !!
#
#   `terraform plan` is always allowed. `terraform apply` / `terraform destroy`
#   (and ANY provisioning side-channel: gcloud create, bulk create, resize
#   requests) happen ONLY after the user has said yes to THIS run, with the
#   cost + ETA quoted first (see estimated_hourly_cost output). This encodes
#   the standing run-approval gate (2026-07-16).
# =============================================================================
#
# CAGE GCP-fallback campaign stack. Neoclouds are primary; this is the GCP
# fallback, especially for session C/D. One state = one run = one blast radius.
#
#   session a  : Qwen3-14B anchor        — 1x a2-ultragpu-1g  (1x A100-80)
#   session b  : Llama-3.3-70B TP=4      — 1x a2-ultragpu-4g  (4x A100-80)
#   session cd : act 1 — Qwen3-Next intra-node PD, ONE a3-ultragpu-8g (8x H200)
#                act 2 — DeepSeek-V3 cross-node PD, TWO a3-ultragpu-8g
#                        (TCP rung first, then RDMA/RoCE rung: enable_rdma=true)
#
# Usage (plan only until the user approves):
#   terraform init -backend=false
#   terraform plan -var-file=sessions/group-a.tfvars -var-file=terraform.tfvars
#
# -----------------------------------------------------------------------------
# TEARDOWN CHECKLIST (fail-closed order — encode of the standing disciplines)
# -----------------------------------------------------------------------------
#  1. PULL RESULTS LOCAL FIRST: rsync/`gcloud storage rsync` gs://cage-<run_id>
#     down to results/ mirroring the bucket structure, and ledger-verify the
#     pull. NO destroy before this passes (teardown_vm.sh step [4/6] semantics).
#  2. terraform destroy -var-file=sessions/<session>.tfvars -var-file=terraform.tfvars
#     (bucket has force_destroy=true — it goes down WITH its objects; that is
#     intentional and is exactly why step 1 is fail-closed).
#  3. ORPHAN SWEEP by label (catches anything created outside terraform):
#       gcloud compute instances list --filter='labels.agent-run=<run_slug>'
#       gcloud compute disks     list --filter='labels.agent-run=<run_slug>'
#       gcloud compute networks  list --filter='name~cage-<run_slug>'
#       gcloud storage ls --project=<project> | grep cage-<run_slug>
#       gcloud compute resize-requests list --zone=<zone>   # DWS leftovers
#  4. VERIFY TRUE $0: `terraform state list` empty; asset inventory / billing
#     shows nothing accruing; the bucket is GONE (surviving buckets count as
#     violations). Report final cost.
# -----------------------------------------------------------------------------

locals {
  # run_id is validated to the RESULTS_LAYOUT §1 bucket-name grammar
  # ([a-z0-9-], variables.tf), so the slug IS the run_id — no re-slugging.
  # This guarantees bucket cage-<run_id> matches the gs://cage-<run_id> the
  # RUNBOOK tells the node to export (a lossy replace() here once let the two
  # names diverge on underscore run_ids). model may carry dots
  # (llama-3.3-70b), so it still gets slugged for label values.
  run_slug   = var.run_id
  model_slug = replace(lower(var.model), "/[^a-z0-9-]/", "-")

  # Every resource carries these — the orphan sweep keys on labels.agent-run.
  common_labels = {
    agent-run = local.run_slug
    session   = var.session
    model     = local.model_slug
  }

  bucket_name = "cage-${local.run_slug}"
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# --- APIs first; everything else depends on this module -----------------------
module "services" {
  source     = "./modules/services"
  project_id = var.project_id
}

# --- Network: mgmt VPC (+ variable-gated RoCE data VPC for C/D act 2) ---------
module "network" {
  source     = "./modules/network"
  depends_on = [module.services]

  project_id                = var.project_id
  region                    = var.region
  run_slug                  = local.run_slug
  mgmt_subnet_cidr          = var.mgmt_subnet_cidr
  enable_rdma               = var.enable_rdma
  rdma_network_profile_name = var.rdma_network_profile_name
  rdma_rail_count           = var.rdma_rail_count
}

# --- Firewall: IAP SSH only + intra-VPC; never 0.0.0.0/0 ingress --------------
module "firewall" {
  source     = "./modules/firewall"
  depends_on = [module.services]

  project_id              = var.project_id
  run_slug                = local.run_slug
  mgmt_network_self_link  = module.network.mgmt_network_self_link
  mgmt_internal_cidrs     = [module.network.mgmt_subnet_cidr]
  data_network_self_links = module.network.data_network_self_links
  data_internal_cidrs     = module.network.data_subnet_cidrs
}

# --- Run bucket: cage-<run_id>, force_destroy INTENTIONAL ---------------------
module "bucket" {
  source     = "./modules/bucket"
  depends_on = [module.services]

  project_id  = var.project_id
  region      = var.region
  bucket_name = local.bucket_name
  labels      = local.common_labels
}

# --- IAM: per-run SA, bucket-scoped objectAdmin, no primitive roles -----------
module "iam" {
  source     = "./modules/iam"
  depends_on = [module.services]

  project_id  = var.project_id
  run_slug    = local.run_slug
  bucket_name = module.bucket.bucket_name
}

# --- GPU nodes ---------------------------------------------------------------
module "gpu_session" {
  source     = "./modules/gpu_session"
  depends_on = [module.services]

  project_id             = var.project_id
  zone                   = var.zone
  run_slug               = local.run_slug
  session                = var.session
  machine_type           = var.machine_type
  node_count             = var.node_count
  boot_disk_size_gb      = var.boot_disk_size_gb
  boot_disk_type         = var.boot_disk_type
  image_family           = var.image_family
  image_project          = var.image_project
  provisioning_model     = var.provisioning_model
  max_run_duration_hours = var.max_run_duration_hours
  startup_script         = var.startup_script
  subnet_self_link       = module.network.mgmt_subnet_self_link
  rdma_subnet_self_links = module.network.data_subnet_self_links
  service_account_email  = module.iam.service_account_email
  labels                 = local.common_labels
}

# --- Cost tripwires: budget (0.5/0.9/1.0) + >18h-uptime alert -----------------
module "monitoring" {
  source     = "./modules/monitoring"
  depends_on = [module.services]

  project_id         = var.project_id
  billing_account_id = var.billing_account_id
  run_slug           = local.run_slug
  budget_amount_usd  = var.budget_amount_usd
  alert_email        = var.alert_email
}
