# Session B — Llama-3.3-70B, TP=4 on ONE 4x A100-80 node (a2-ultragpu-4g).
# TP=2 sensitivity point runs on the same node (no shape change).

session      = "b"
model        = "llama-3.3-70b"
machine_type = "a2-ultragpu-4g"
node_count   = 1

boot_disk_size_gb = 400
boot_disk_type    = "pd-ssd"

image_family = "common-cu124-ubuntu-2204-nvidia-550" # [VERIFY-AT-APPLY]

provisioning_model = "SPOT" # single node; preemption costs a resume, not the gang

enable_rdma = false
