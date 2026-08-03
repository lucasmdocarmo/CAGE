# modules/firewall — ingress policy for the run.
#
#   * SSH ONLY via IAP TCP forwarding: source 35.235.240.0/20 (Google's fixed
#     IAP range). Nodes have no external IPs; there is NO 0.0.0.0/0 ingress
#     rule anywhere in this stack — ever.
#   * Intra-VPC allow-all (tcp/udp/icmp) inside the run's own subnets: NCCL
#     rendezvous + engine RPC hang at init if cluster traffic is filtered
#     (networking-rdma doctrine: "allow all within the cluster subnets").

terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
    }
  }
}

# --- mgmt VPC: SSH via IAP only ---------------------------------------------

resource "google_compute_firewall" "iap_ssh" {
  project   = var.project_id
  name      = "cage-${var.run_slug}-allow-iap-ssh"
  network   = var.mgmt_network_self_link
  direction = "INGRESS"

  source_ranges = ["35.235.240.0/20"] # IAP range ONLY — never widen

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

# --- mgmt VPC: intra-cluster traffic ----------------------------------------

resource "google_compute_firewall" "mgmt_internal" {
  project   = var.project_id
  name      = "cage-${var.run_slug}-allow-internal"
  network   = var.mgmt_network_self_link
  direction = "INGRESS"

  source_ranges = var.mgmt_internal_cidrs

  allow {
    protocol = "tcp"
  }
  allow {
    protocol = "udp"
  }
  allow {
    protocol = "icmp"
  }
}

# --- RoCE data VPC(s): intra-rail traffic (C/D RDMA rung) --------------------

resource "google_compute_firewall" "data_internal" {
  count = length(var.data_network_self_links)

  project   = var.project_id
  name      = format("cage-%s-allow-data-internal-%d", var.run_slug, count.index)
  network   = var.data_network_self_links[count.index]
  direction = "INGRESS"

  source_ranges = var.data_internal_cidrs

  allow {
    protocol = "tcp"
  }
  allow {
    protocol = "udp"
  }
  allow {
    protocol = "icmp"
  }
}
