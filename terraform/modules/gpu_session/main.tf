# modules/gpu_session — the GPU node(s) for one provisioning session.
#
#   session a : 1x a2-ultragpu-1g  (1x A100-80)          — Qwen3-14B anchor
#   session b : 1x a2-ultragpu-4g  (4x A100-80, TP=4)    — Llama-3.3-70B
#   session cd: 1..2x a3-ultragpu-8g (8x H200)           — Qwen3-Next / DeepSeek-V3
#
# GPUs are implicit in a2-*/a3-* machine types (no guest_accelerator block).
# Nodes get NO external IP — SSH is IAP-only (see modules/firewall).
#
# --- Provisioning models -----------------------------------------------------
# SPOT and ON_DEMAND are fully implemented via the provider scheduling block
# (ON_DEMAND maps to the API's STANDARD). GPU VMs always require
# on_host_maintenance = TERMINATE (no live migration with accelerators).
#
# FLEX_START (DWS Flex-start, the intended H200 acquisition path) is a
# [VERIFY-AT-APPLY] rung: provider support for FLEX_START on a plain
# google_compute_instance is moving (historically DWS was reachable only via
# MIG resize-requests / GKE queued provisioning). This module DOES emit
# provisioning_model = "FLEX_START" + max_run_duration + DELETE-on-termination
# so it works where the provider/API accept it; if plan/apply rejects it,
# use the documented gcloud fallback and `terraform import` the instances:
#
#   # DWS Flex-start via bulk create (verify flag names against current gcloud):
#   gcloud compute instances bulk create \
#     --name-pattern='cage-cd-<run_slug>-#' --count=<n> \
#     --project=<project> --zone=<zone> --machine-type=a3-ultragpu-8g \
#     --provisioning-model=FLEX_START --instance-termination-action=DELETE \
#     --max-run-duration=<h>h [VERIFY-AT-APPLY] \
#     ... (image/disk/SA/labels exactly as below, then terraform import)
#   LABEL PARITY (mandatory): instance labels do NOT propagate to boot disks,
#   so after any gcloud-created instance, label its disk too or the TRUE-$0
#   disk sweep (labels.agent-run) can never see it:
#     gcloud compute disks update <disk> --zone=<zone> \
#       --update-labels=agent-run=<run_slug>,session=<session>,model=<model_slug>
#
# --- RDMA NICs (C/D act 2, RDMA rung) ---------------------------------------
# When rail subnets are passed in, each node gets 8 extra NICs with
# nic_type = "MRDMA" on the RoCE data VPC (A3 Ultra CX-7 NICs; mgmt NIC stays
# GVNIC). [VERIFY-AT-APPLY]: the MRDMA nic_type string and per-NIC ordering
# against the current A3 Ultra docs; if the provider rejects MRDMA, create
# the instances with the gcloud fallback:
#
#   gcloud compute instances create ... \
#     --network-interface=nic-type=GVNIC,subnet=<mgmt-subnet> \
#     --network-interface=nic-type=MRDMA,subnet=<rail-0> \
#     ... one MRDMA interface per rail subnet (8 total) ...
# -----------------------------------------------------------------------------

terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
    }
  }
}

locals {
  # ON_DEMAND is CAGE vocabulary; the compute API calls it STANDARD.
  api_provisioning_model = var.provisioning_model == "ON_DEMAND" ? "STANDARD" : var.provisioning_model
  is_spot                = var.provisioning_model == "SPOT"
  is_flex_start          = var.provisioning_model == "FLEX_START"
}

resource "google_compute_instance" "node" {
  count = var.node_count

  project      = var.project_id
  zone         = var.zone
  name         = format("cage-%s-%s-%02d", var.session, var.run_slug, count.index)
  machine_type = var.machine_type

  # Orphan-sweep keys: labels.agent-run / session / model.
  labels = var.labels

  boot_disk {
    initialize_params {
      image = "projects/${var.image_project}/global/images/family/${var.image_family}"
      size  = var.boot_disk_size_gb
      type  = var.boot_disk_type
      # Disks are BILLABLE resources and GCE does NOT propagate instance
      # labels to them — without this, `gcloud compute disks list
      # --filter='labels.agent-run=<run_id>'` (main.tf teardown checklist,
      # RUNBOOK §5 step [6]) can never match a surviving boot disk and the
      # TRUE-$0 orphan sweep reads "clean" while a 2 TB hyperdisk keeps
      # billing. Every disk carries the sweep keys.
      labels = var.labels
    }
  }

  # mgmt NIC — no access_config: NO external IP, SSH via IAP only.
  network_interface {
    subnetwork = var.subnet_self_link
    nic_type   = "GVNIC"
  }

  # RoCE rail NICs (C/D RDMA rung only; empty list otherwise).
  dynamic "network_interface" {
    for_each = var.rdma_subnet_self_links
    content {
      subnetwork = network_interface.value
      nic_type   = "MRDMA" # [VERIFY-AT-APPLY] — see module header for fallback
    }
  }

  scheduling {
    provisioning_model = local.api_provisioning_model
    preemptible        = local.is_spot
    # GPU VMs cannot live-migrate.
    on_host_maintenance = "TERMINATE"
    automatic_restart   = local.api_provisioning_model == "STANDARD"
    # Spot/Flex nodes self-delete on termination — no stopped-but-billed disks.
    instance_termination_action = local.is_spot || local.is_flex_start ? "DELETE" : null

    # DWS Flex-start bounds the run window. [VERIFY-AT-APPLY]
    dynamic "max_run_duration" {
      for_each = local.is_flex_start ? [1] : []
      content {
        seconds = var.max_run_duration_hours * 3600
      }
    }
  }

  # Per-run least-privilege SA; broad scope narrowed by IAM (modules/iam).
  service_account {
    email  = var.service_account_email
    scopes = ["cloud-platform"]
  }

  metadata = merge(
    {
      enable-oslogin = "TRUE"
    },
    # Startup hook stays EMPTY by default — provisioning lives in
    # cloud/RUNBOOK's remote_job flow, not in instance metadata.
    var.startup_script == null ? {} : { startup-script = var.startup_script },
  )

  allow_stopping_for_update = true
}
