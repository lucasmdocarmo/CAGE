# modules/network — mgmt VPC for all sessions; variable-gated RoCE data VPC
# for the C/D act-2 RDMA rung.
#
# Sessions A/B: the mgmt VPC alone (one subnet, private_ip_google_access so
# nodes with NO external IP still reach GCS/Artifact endpoints).
#
# C/D RDMA rung (enable_rdma = true) — facts on record:
#   * A3 Ultra RoCE traffic ONLY flows on a VPC created from a zonal RDMA
#     network profile (an ordinary VPC will not carry RoCE).
#   * 8 rail subnets (one data NIC per GPU) + a separate ordinary gVNIC
#     mgmt VPC for control traffic.
#   * Jumbo MTU 8896 on the data VPC; VPC, NIC, and guest must agree.
#
# Provider support status:
#   google_compute_network.network_profile IS modeled by the google provider
#   (recent 6.x+). [VERIFY-AT-APPLY]: confirm the profile NAME for your zone
#   (`gcloud compute network-profiles list --filter='name~roce'`) and that the
#   provider version in .terraform.lock.hcl accepts the field. If the provider
#   rejects it, fall back to the gcloud block below and `import` the network.
#
# --- gcloud fallback (documented, for provider gaps) -------------------------
#   gcloud compute networks create cage-<run_slug>-data \
#     --project=<project> --subnet-mode=custom --mtu=8896 \
#     --network-profile=$(gcloud compute network-profiles list \
#         --format='value(name)' --filter='name~roce AND name~<region>' | head -1)
#   for i in $(seq 0 7); do
#     gcloud compute networks subnets create cage-<run_slug>-rail-$i \
#       --network=cage-<run_slug>-data --region=<region> \
#       --range=10.$((10+i)).0.0/16
#   done
#   # then: terraform import into google_compute_network.data / .rail subnets
# -----------------------------------------------------------------------------

terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
    }
  }
}

# --- mgmt VPC (all sessions) -------------------------------------------------

resource "google_compute_network" "mgmt" {
  project                 = var.project_id
  name                    = "cage-${var.run_slug}-mgmt"
  auto_create_subnetworks = false
  mtu                     = 1460
}

resource "google_compute_subnetwork" "mgmt" {
  project                  = var.project_id
  name                     = "cage-${var.run_slug}-mgmt"
  region                   = var.region
  network                  = google_compute_network.mgmt.self_link
  ip_cidr_range            = var.mgmt_subnet_cidr
  private_ip_google_access = true # no-external-IP nodes still reach GCS
}

# --- RoCE data VPC + 8 rail subnets (C/D act-2 RDMA rung only) ---------------

resource "google_compute_network" "data" {
  count = var.enable_rdma ? 1 : 0

  project                 = var.project_id
  name                    = "cage-${var.run_slug}-data"
  auto_create_subnetworks = false
  mtu                     = 8896 # jumbo — must match NIC + guest config

  # [VERIFY-AT-APPLY] zonal RDMA profile name, e.g. "<zone>-vpc-roce".
  # (null-safe so the precondition below—not an interpolation error—reports
  # a missing profile name.)
  network_profile = var.rdma_network_profile_name == null ? null : "projects/${var.project_id}/global/networkProfiles/${var.rdma_network_profile_name}"

  lifecycle {
    precondition {
      condition     = var.rdma_network_profile_name != null
      error_message = "enable_rdma=true requires rdma_network_profile_name (gcloud compute network-profiles list --filter='name~roce')."
    }
  }
}

resource "google_compute_subnetwork" "rail" {
  count = var.enable_rdma ? var.rdma_rail_count : 0

  project       = var.project_id
  name          = format("cage-%s-rail-%d", var.run_slug, count.index)
  region        = var.region
  network       = google_compute_network.data[0].self_link
  ip_cidr_range = format("10.%d.0.0/16", 10 + count.index)
}
