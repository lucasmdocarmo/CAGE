# Session C/D — two acts on a3-ultragpu-8g (8x H200, 141 GB each).
#
#   act 1: ONE node  — Qwen3-Next-80B intra-node PD (2x160 GB weight copies;
#          this is why Group C was folded onto Group D's H200 rental).
#          -> node_count = 1, model = "qwen3-next-80b", enable_rdma = false
#   act 2: TWO nodes — DeepSeek-V3-0324 (FP8 weights ~671 GB) cross-node PD.
#          TCP rung first (enable_rdma = false), then RDMA/RoCE rung
#          (enable_rdma = true + rdma_network_profile_name set).
#
# Flip the act-2 lines below between rungs. Each flip is a plan the user must
# approve before apply.

session      = "cd"
model        = "qwen3-next-80b" # act 2: "deepseek-v3-0324"
machine_type = "a3-ultragpu-8g"
node_count   = 1 # act 2: 2

# 2000 GB: V3 weights 671 GB FP8 + HF cache + 2x160 GB Qwen3-Next copies.
boot_disk_size_gb = 2000
# A3 Ultra takes Hyperdisk, not pd-ssd. [VERIFY-AT-APPLY]
boot_disk_type = "hyperdisk-balanced"

# H200 wants a >=570-driver CUDA 12.8-class family. [VERIFY-AT-APPLY]
image_family = "common-cu128-ubuntu-2204-nvidia-570"

# H200 capacity is scarce: DWS Flex-start is the intended acquisition path
# (see modules/gpu_session/main.tf for the [VERIFY-AT-APPLY] notes and the
# gcloud fallback). SPOT on 8x H200 is often unavailable; ON_DEMAND needs
# quota + deep pockets — quote cost before any apply.
provisioning_model     = "FLEX_START"
max_run_duration_hours = 24

# --- act 2 RDMA rung only ---------------------------------------------------
enable_rdma = false
# rdma_network_profile_name = "us-central1-a-vpc-roce" # [VERIFY-AT-APPLY]
#   gcloud compute network-profiles list --filter='name~roce'
