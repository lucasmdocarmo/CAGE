# Session A — Qwen3-14B anchor. 1x A100-80 (a2-ultragpu-1g class).
# PUBLICATION.md authority: 14B re-rung to an 80 GB-class GPU (the "~28 GB"
# sizing was wrong).

session      = "a"
model        = "qwen3-14b"
machine_type = "a2-ultragpu-1g"
node_count   = 1

boot_disk_size_gb = 400
boot_disk_type    = "pd-ssd"

# [VERIFY-AT-APPLY] image family list: gcloud compute images list
#   --project=deeplearning-platform-release --filter='family~cu1'
image_family = "common-cu124-ubuntu-2204-nvidia-550"

provisioning_model = "SPOT" # single-node, checkpoint/resume proven -> Spot-safe

enable_rdma = false
