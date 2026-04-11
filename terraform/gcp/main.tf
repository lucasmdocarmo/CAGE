# CAGE Framework - GCP Infrastructure
# Terraform configuration for multi-node distributed CAG experiments

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

# Variables
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
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

variable "num_replicas" {
  description = "Number of vLLM replica nodes"
  type        = number
  default     = 3
}

variable "gpu_type" {
  description = "GPU type (nvidia-tesla-a100, nvidia-l4, nvidia-tesla-t4)"
  type        = string
  default     = "nvidia-l4"
}

variable "gpu_count" {
  description = "Number of GPUs per node"
  type        = number
  default     = 1
}

variable "machine_type" {
  description = "GCE machine type"
  type        = string
  default     = "g2-standard-8"  # 8 vCPUs, 32GB RAM, optimized for L4 GPUs
}

variable "model_name" {
  description = "HuggingFace model to serve"
  type        = string
  default     = "Qwen/Qwen3-4B"
}

variable "disk_size_gb" {
  description = "Boot disk size in GB"
  type        = number
  default     = 200
}

# Network
resource "google_compute_network" "cage_network" {
  name                    = "cage-network"
  auto_create_subnetworks = false
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

  source_ranges = ["0.0.0.0/0"]
}

resource "google_compute_firewall" "cage_router" {
  name    = "cage-router-external"
  network = google_compute_network.cage_network.name

  allow {
    protocol = "tcp"
    ports    = ["9000", "8000"]  # Router and vLLM ports
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["cage-router"]
}

# vLLM Replica Instances
resource "google_compute_instance" "vllm_replica" {
  count        = var.num_replicas
  name         = "vllm-replica-${count.index + 1}"
  machine_type = var.machine_type
  zone         = var.zone

  tags = ["cage-vllm"]

  boot_disk {
    initialize_params {
      image = "deeplearning-platform-release/common-cu121-v20240128-debian-11"
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
    automatic_restart   = true
  }

  network_interface {
    subnetwork = google_compute_subnetwork.cage_subnet.id
    access_config {
      // Ephemeral public IP
    }
  }

  metadata = {
    startup-script = <<-EOF
      #!/bin/bash
      set -e
      
      # Install Docker
      if ! command -v docker &> /dev/null; then
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
      fi
      
      # Install NVIDIA Container Toolkit
      if ! command -v nvidia-container-toolkit &> /dev/null; then
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add -
        curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        apt-get update
        apt-get install -y nvidia-container-toolkit
        nvidia-ctk runtime configure --runtime=docker
        systemctl restart docker
      fi
      
      # Pull and run vLLM
      REPLICA_PORT=$((8001 + ${count.index}))
      docker pull vllm/vllm-openai:latest
      docker run -d \
        --name vllm-replica \
        --gpus all \
        --restart unless-stopped \
        -p $REPLICA_PORT:$REPLICA_PORT \
        -e HF_TOKEN="${var.hf_token}" \
        vllm/vllm-openai:latest \
        --model ${var.model_name} \
        --port $REPLICA_PORT \
        --enable-prefix-caching \
        --enable-prompt-tokens-details \
        --gpu-memory-utilization 0.9
      
      echo "vLLM replica ${count.index + 1} started on port $REPLICA_PORT"
    EOF
  }

  service_account {
    scopes = ["cloud-platform"]
  }
}

variable "hf_token" {
  description = "HuggingFace API token (for gated models)"
  type        = string
  default     = ""
  sensitive   = true
}

# Router Instance (CPU-only)
resource "google_compute_instance" "cage_router" {
  name         = "cage-router"
  machine_type = "e2-standard-4"  # 4 vCPUs, 16GB RAM
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
    access_config {
      // Ephemeral public IP
    }
  }

  metadata = {
    startup-script = <<-EOF
      #!/bin/bash
      set -e
      
      # Install Docker
      if ! command -v docker &> /dev/null; then
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
      fi
      
      # Build replica URLs
      REPLICAS=""
      for i in $(seq 1 ${var.num_replicas}); do
        PORT=$((8000 + i))
        IP=$(gcloud compute instances describe vllm-replica-$i --zone=${var.zone} --format='get(networkInterfaces[0].networkIP)')
        if [ -n "$REPLICAS" ]; then
          REPLICAS="$REPLICAS,"
        fi
        REPLICAS="$REPLICAS replica-$i=http://$IP:$PORT"
      done
      
      # Clone CAGE repo and start router
      git clone https://github.com/your-repo/cag-llm-kvcache.git /opt/cage
      cd /opt/cage
      
      # Install Python and dependencies
      apt-get update && apt-get install -y python3-pip
      pip3 install -r requirements.txt
      
      # Start router
      ROUTER_REPLICAS="$REPLICAS" python3 -m src.orchestration.router &
      
      echo "CAGE Router started with replicas: $REPLICAS"
    EOF
  }

  depends_on = [google_compute_instance.vllm_replica]

  service_account {
    scopes = ["cloud-platform"]
  }
}

# Outputs
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

output "experiment_command" {
  description = "Command to run experiments against the cluster"
  value       = "python3 scripts/run_experiment.py --baseline distributed --model ${var.model_name} --api-base http://${google_compute_instance.cage_router.network_interface[0].access_config[0].nat_ip}:9000"
}
