# CAGE Framework - GCP Infrastructure
# Terraform configuration for multi-node distributed CAG experiments.
#
# Phase 2 (defaults): 3x g2-standard-8 (NVIDIA L4) replicas + 1 CPU router.
# Phase 3: set machine_type=a2-highgpu-1g, gpu_type=nvidia-tesla-a100,
#          nic_type=GVNIC, network_mtu=8896 via terraform.tfvars (no file edits).

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone (verify the chosen GPU exists here)"
  type        = string
  default     = "us-central1-a"
}

variable "num_replicas" {
  description = "Number of vLLM replica nodes"
  type        = number
  default     = 3
}

variable "gpu_type" {
  description = "GPU type (nvidia-l4 for Phase 2, nvidia-tesla-a100 for Phase 3)"
  type        = string
  default     = "nvidia-l4"
}

variable "gpu_count" {
  description = "Number of GPUs per node"
  type        = number
  default     = 1
}

variable "machine_type" {
  description = "GCE machine type (g2-standard-8 for L4, a2-highgpu-1g for A100)"
  type        = string
  default     = "g2-standard-8"
}

variable "model_name" {
  description = "HuggingFace model to serve. MUST match the model passed to run_experiment.py."
  type        = string
  default     = "Qwen/Qwen3-8B"
}

variable "vllm_image" {
  description = "vLLM serving image. Pin a concrete tag for reproducibility."
  type        = string
  default     = "vllm/vllm-openai:v0.11.0"
}

variable "vllm_extra_args" {
  description = "Extra flags appended to the vLLM launch, e.g. '--kv-cache-dtype fp8' (compressed_cag) or a --speculative-config JSON. Empty by default. See cloud_docs/VLLM_COMPATIBILITY.md."
  type        = string
  default     = ""
}

variable "gpu_memory_utilization" {
  description = "vLLM --gpu-memory-utilization. 0.85 leaves KV/draft headroom on a 24GB L4; for Qwen3-8B + FP8 + speculative, lower further or use Qwen3-4B for those arms (see cloud_docs/PHASE2_CHECKLIST.md)."
  type        = number
  default     = 0.85
}

variable "disk_size_gb" {
  description = "Replica boot disk size in GB (holds image + model weights)"
  type        = number
  default     = 200
}

variable "repo_url" {
  description = "Git URL of the CAGE repo. If set, the router clones it into /opt/cage automatically. If empty, the router waits for an SCP upload."
  type        = string
  default     = ""
}

variable "ssh_source_ranges" {
  description = "CIDRs allowed to SSH. Lock this to your IP in production."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "nic_type" {
  description = "NIC type. Set to GVNIC for Phase 3 (unlocks ~100 Gbps for KV transfer)."
  type        = string
  default     = "VIRTIO_NET"
}

variable "network_mtu" {
  description = "VPC MTU. Set to 8896 (jumbo frames) for Phase 3 cross-node KV transfer."
  type        = number
  default     = 1460
}

variable "hf_token" {
  description = "HuggingFace API token (for gated models)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "preemptible" {
  description = "Use Spot/preemptible VMs for the GPU replicas. Set true for Phase 2 single-node baseline sweeps (cheap and interruptible; results sync to GCS, so a preemption costs at most one in-flight baseline). Keep false for the Phase 3 coordinated cluster, where all replicas must stay up together."
  type        = bool
  default     = false
}

# ---------------------------------------------------------------------------
# Enable required APIs (a fresh project has these disabled)
# ---------------------------------------------------------------------------
resource "google_project_service" "compute" {
  project            = var.project_id
  service            = "compute.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudresourcemanager" {
  project            = var.project_id
  service            = "cloudresourcemanager.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "storage" {
  project            = var.project_id
  service            = "storage.googleapis.com"
  disable_on_destroy = false
}

# ---------------------------------------------------------------------------
# Durable results bucket (survives `terraform destroy` of the GPU cluster)
# ---------------------------------------------------------------------------
# The VMs run as the project's default Compute service account; grant it object
# access so cloud_run.sh / sync_results_to_gcs.sh can push results to GCS.
data "google_compute_default_service_account" "default" {
  project    = var.project_id
  depends_on = [google_project_service.compute]
}

resource "google_storage_bucket" "results" {
  name                        = "${var.project_id}-cage-results"
  project                     = var.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false # keep results when the cluster is torn down
  versioning {
    enabled = true # protect against accidental overwrite
  }
  depends_on = [google_project_service.storage]
}

resource "google_storage_bucket_iam_member" "results_writer" {
  bucket = google_storage_bucket.results.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${data.google_compute_default_service_account.default.email}"
}

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
resource "google_compute_network" "cage_network" {
  name                    = "cage-network"
  auto_create_subnetworks = false
  mtu                     = var.network_mtu
  depends_on              = [google_project_service.compute]
}

resource "google_compute_subnetwork" "cage_subnet" {
  name          = "cage-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.cage_network.id
}

# Firewall rules
resource "google_compute_firewall" "cage_internal" {
  name    = "cage-internal"
  network = google_compute_network.cage_network.name

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }
  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }
  allow {
    protocol = "icmp"
  }

  source_ranges = ["10.0.0.0/24"]
}

resource "google_compute_firewall" "cage_ssh" {
  name    = "cage-ssh"
  network = google_compute_network.cage_network.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = var.ssh_source_ranges
}

resource "google_compute_firewall" "cage_router" {
  name    = "cage-router-external"
  network = google_compute_network.cage_network.name

  allow {
    protocol = "tcp"
    ports    = ["9000"] # Router API only; vLLM replicas stay internal.
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["cage-router"]
}

# ---------------------------------------------------------------------------
# vLLM Replica Instances
# ---------------------------------------------------------------------------
resource "google_compute_instance" "vllm_replica" {
  count        = var.num_replicas
  name         = "vllm-replica-${count.index + 1}"
  machine_type = var.machine_type
  zone         = var.zone

  tags = ["cage-vllm"]

  boot_disk {
    initialize_params {
      image = "deeplearning-platform-release/common-cu121-debian-11"
      size  = var.disk_size_gb
      type  = "pd-ssd"
    }
  }

  guest_accelerator {
    type  = var.gpu_type
    count = var.gpu_count
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    # Spot/preemptible (var.preemptible=true) requires automatic_restart=false.
    automatic_restart  = var.preemptible ? false : true
    preemptible        = var.preemptible
    provisioning_model = var.preemptible ? "SPOT" : "STANDARD"
  }

  network_interface {
    subnetwork = google_compute_subnetwork.cage_subnet.id
    nic_type   = var.nic_type
    access_config {
      // Ephemeral public IP (needed to pull the image + model weights).
    }
  }

  metadata = {
    # CRITICAL: tells the DLVM image to install the NVIDIA kernel driver on first
    # boot. Without this, `docker run --gpus all` fails and vLLM never starts.
    install-nvidia-driver = "True"

    # Sync results + logs to GCS on ACPI soft-off (SPOT preemption ~30s budget, or a normal
    # instances delete/stop), so data is captured even when no operator is watching and the
    # run-script EXIT trap never fires. See scripts/gcp_shutdown_hook.sh.
    shutdown-script = file("${path.module}/../../scripts/gcp_shutdown_hook.sh")

    startup-script = <<-EOF
      #!/bin/bash
      set -euo pipefail
      exec > >(tee /var/log/cage-startup.log) 2>&1
      echo "[cage] replica ${count.index + 1} startup begin"

      # Wait for the NVIDIA driver to finish installing (DLVM installs it async).
      echo "[cage] waiting for NVIDIA driver..."
      for i in $(seq 1 60); do
        if nvidia-smi >/dev/null 2>&1; then echo "[cage] driver ready"; break; fi
        sleep 10
      done

      # Docker
      if ! command -v docker &>/dev/null; then
        curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh
      fi

      # NVIDIA Container Toolkit (keyring pattern; apt-key is removed on Debian 11+)
      if ! command -v nvidia-ctk &>/dev/null; then
        distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
          | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
          | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
          | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        apt-get update && apt-get install -y nvidia-container-toolkit
        nvidia-ctk runtime configure --runtime=docker
        systemctl restart docker
      fi

      REPLICA_PORT=$((8001 + ${count.index}))
      docker pull ${var.vllm_image}
      docker run -d \
        --name vllm-replica \
        --gpus all \
        --restart unless-stopped \
        -p $REPLICA_PORT:$REPLICA_PORT \
        -e HF_TOKEN="${var.hf_token}" \
        ${var.vllm_image} \
        --model ${var.model_name} \
        --port $REPLICA_PORT \
        --enable-prefix-caching \
        --enable-prompt-tokens-details \
        --gpu-memory-utilization ${var.gpu_memory_utilization} \
        ${var.vllm_extra_args}

      echo "[cage] vLLM replica ${count.index + 1} started on port $REPLICA_PORT"
    EOF
  }

  service_account {
    scopes = ["cloud-platform"]
  }

  depends_on = [google_project_service.compute]
}

# ---------------------------------------------------------------------------
# Router Instance (CPU-only)
# ---------------------------------------------------------------------------
resource "google_compute_instance" "cage_router" {
  name         = "cage-router"
  machine_type = "e2-standard-4" # 4 vCPUs, 16GB RAM
  zone         = var.zone

  tags = ["cage-router"]

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
      size  = 50
      type  = "pd-standard"
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.cage_subnet.id
    nic_type   = var.nic_type
    access_config {
      // Ephemeral public IP
    }
  }

  metadata = {
    startup-script = <<-EOF
      #!/bin/bash
      set -euo pipefail
      exec > >(tee /var/log/cage-startup.log) 2>&1
      echo "[cage] router startup begin"

      if ! command -v docker &>/dev/null; then
        curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh
      fi
      apt-get update && apt-get install -y python3-pip git curl

      # Discover replica internal IPs; build ROUTER_REPLICAS + health-check targets.
      REPLICAS=""
      HEALTH_TARGETS=""
      for i in $(seq 1 ${var.num_replicas}); do
        PORT=$((8000 + i))
        IP=$(gcloud compute instances describe vllm-replica-$i --zone=${var.zone} \
             --format='get(networkInterfaces[0].networkIP)')
        if [ -n "$REPLICAS" ]; then REPLICAS="$REPLICAS,"; fi
        REPLICAS="$REPLICAS replica-$i=http://$IP:$PORT"
        HEALTH_TARGETS="$HEALTH_TARGETS http://$IP:$PORT/health"
      done

      # Redis for retrieval-artifact caching.
      docker run -d -p 6379:6379 --name cage-redis --restart unless-stopped redis:7-alpine

      # Get the code: prefer git clone, else wait (bounded) for an SCP upload.
      mkdir -p /opt/cage
      if [ -n "${var.repo_url}" ]; then
        git clone ${var.repo_url} /opt/cage || echo "[cage] git clone failed; will wait for SCP upload"
      fi
      if [ ! -f /opt/cage/requirements.txt ]; then
        echo "[cage] waiting up to 30 min for /opt/cage upload via SCP..."
        for i in $(seq 1 360); do
          [ -f /opt/cage/requirements.txt ] && break
          sleep 5
        done
      fi
      cd /opt/cage

      # Gate on replica health so the router never points at not-ready vLLM servers.
      echo "[cage] waiting for replicas to become healthy..."
      for url in $HEALTH_TARGETS; do
        for i in $(seq 1 180); do
          if curl -fsS "$url" >/dev/null 2>&1; then echo "[cage] $url healthy"; break; fi
          sleep 10
        done
      done

      # Router needs only the minimal deps (no torch/transformers).
      if [ -f docker/router.requirements.txt ]; then
        pip3 install -r docker/router.requirements.txt
      else
        pip3 install fastapi uvicorn aiohttp prometheus-client
      fi

      ROUTER_REPLICAS="$REPLICAS" nohup python3 -m src.orchestration.router \
        > /var/log/cage-router.log 2>&1 &
      echo "[cage] router started with replicas: $REPLICAS"
    EOF
  }

  depends_on = [google_compute_instance.vllm_replica]

  service_account {
    scopes = ["cloud-platform"]
  }
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
output "router_external_ip" {
  description = "External IP of the CAGE router"
  value       = google_compute_instance.cage_router.network_interface[0].access_config[0].nat_ip
}

output "replica_internal_ips" {
  description = "Internal IPs of vLLM replicas"
  value       = [for instance in google_compute_instance.vllm_replica : instance.network_interface[0].network_ip]
}

output "replica_external_ips" {
  description = "External IPs of vLLM replicas (for debugging)"
  value       = [for instance in google_compute_instance.vllm_replica : instance.network_interface[0].access_config[0].nat_ip]
}

output "ssh_commands" {
  description = "SSH commands to connect to instances"
  value = {
    router   = "gcloud compute ssh cage-router --zone=${var.zone}"
    replicas = [for i in range(var.num_replicas) : "gcloud compute ssh vllm-replica-${i + 1} --zone=${var.zone}"]
  }
}

output "upload_code_command" {
  description = "Run this (if repo_url is unset) to push the code the router is waiting for"
  value       = "gcloud compute scp --recurse . cage-router:/opt/cage --zone=${var.zone}"
}

output "experiment_command" {
  description = "Command to run experiments against the cluster"
  value       = "python3 scripts/run_experiment.py --baseline distributed --model ${var.model_name} --api-base http://${google_compute_instance.cage_router.network_interface[0].access_config[0].nat_ip}:9000"
}

output "results_bucket" {
  description = "Durable GCS bucket for results (survives terraform destroy)"
  value       = "gs://${google_storage_bucket.results.name}"
}
