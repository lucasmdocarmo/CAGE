# CAGE GCP-fallback campaign — root variables.
# Session-specific values come from sessions/group-{a,b,cd}.tfvars;
# operator-specific values from terraform.tfvars (gitignored — see .example).

# ---------------------------------------------------------------------------
# Operator / project
# ---------------------------------------------------------------------------

variable "project_id" {
  description = "GCP project id that hosts the run (clean-room: nothing from past runs may survive in it)."
  type        = string
}

variable "region" {
  description = "Region for subnets and the run bucket."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Zone for the GPU nodes. [VERIFY-AT-APPLY] a2-ultragpu and a3-ultragpu-8g availability + quota per zone: `gcloud compute machine-types list --filter='name~a2-ultragpu|a3-ultragpu' --zones=<zone>`."
  type        = string
  default     = "us-central1-a"
}

variable "billing_account_id" {
  description = "Billing account id (XXXXXX-XXXXXX-XXXXXX) for the run budget. Caller needs billing.budgets.create on it. [VERIFY-AT-APPLY]"
  type        = string
}

variable "alert_email" {
  description = "Email for budget + runaway-uptime alerts (monitoring notification channel)."
  type        = string
}

# ---------------------------------------------------------------------------
# Run identity (labels drive the orphan sweep — every resource carries them)
# ---------------------------------------------------------------------------

variable "run_id" {
  description = "Run id minted by the run wrapper (cloud/RESULTS_LAYOUT.md §1: YYYYMMDD-hhmm-<session>-<model-slug>). Names bucket cage-<run_id> VERBATIM, plus the SA id and the agent-run label."
  type        = string

  # RESULTS_LAYOUT.md §1 grammar — lowercase bucket-name-safe [a-z0-9-] ONLY.
  # This makes slug == run_id BY CONSTRUCTION, so the bucket terraform creates
  # is exactly the gs://cage-<run_id> that RUNBOOK §4.4 exports on the node and
  # the sync daemon writes to. (2026-08-02 fix: the old [A-Za-z0-9._-] grammar
  # + slugging let an underscore run_id create cage-a-b-c while the docs
  # pointed the daemon at the nonexistent gs://cage-a_b_c — the pilot
  # "synced to a bucket that didn't exist" failure class.)
  # Kept in lockstep with organize_results.RUN_ID_RE by
  # tests/test_terraform_contract.py.
  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{2,40}$", var.run_id))
    error_message = "run_id must be 3-41 chars of [a-z0-9-], starting alphanumeric (RESULTS_LAYOUT.md §1: it names the GCS bucket cage-<run_id> verbatim — no uppercase, dots, or underscores)."
  }
}

variable "session" {
  description = "Provisioning session on the GCP-fallback path: a (Qwen3-14B anchor, 1x A100-80), b (Llama-3.3-70B TP=4, one 4x A100-80 node), cd (Qwen3-Next intra-node PD then DeepSeek-V3 cross-node PD on 8x H200 nodes)."
  type        = string

  validation {
    condition     = contains(["a", "b", "cd"], var.session)
    error_message = "session must be one of: a, b, cd."
  }
}

variable "model" {
  description = "Model under test for this session (label value, e.g. qwen3-14b, llama-3.3-70b, qwen3-next-80b, deepseek-v3-0324)."
  type        = string
}

# ---------------------------------------------------------------------------
# GPU node shape (per-session tfvars)
# ---------------------------------------------------------------------------

variable "machine_type" {
  description = "a2-ultragpu-1g (session a) | a2-ultragpu-4g (session b) | a3-ultragpu-8g (session cd). GPUs are implicit in these machine types."
  type        = string

  validation {
    condition     = contains(["a2-ultragpu-1g", "a2-ultragpu-4g", "a3-ultragpu-8g"], var.machine_type)
    error_message = "machine_type must be a2-ultragpu-1g, a2-ultragpu-4g, or a3-ultragpu-8g."
  }
}

variable "node_count" {
  description = "GPU nodes: 1 for sessions a/b and c/d act 1; 2 for c/d act 2 (DeepSeek-V3 cross-node PD)."
  type        = number
  default     = 1

  validation {
    condition     = var.node_count >= 1 && var.node_count <= 2
    error_message = "node_count must be 1 or 2 (campaign hardware honesty: max is two 8x H200 nodes)."
  }
}

variable "boot_disk_size_gb" {
  description = "Boot disk GB. A/B: 400. C/D: 2000 (DeepSeek-V3 FP8 weights 671 GB + HF cache + 2x160 GB Qwen3-Next PD weight copies)."
  type        = number
  default     = 400
}

variable "boot_disk_type" {
  description = "pd-ssd for a2-*; hyperdisk-balanced for a3-ultragpu-8g (A3 Ultra does not take pd-ssd). [VERIFY-AT-APPLY] `gcloud compute machine-types describe`."
  type        = string
  default     = "pd-ssd"
}

variable "image_family" {
  description = "Deep Learning VM / accelerator image family. [VERIFY-AT-APPLY] `gcloud compute images list --project=deeplearning-platform-release --filter='family~cu1' --format='value(family)' | sort -u` — H200 nodes need a >=570-driver CUDA 12.8-class family."
  type        = string
  default     = "common-cu124-ubuntu-2204-nvidia-550"
}

variable "image_project" {
  description = "Project hosting the image family."
  type        = string
  default     = "deeplearning-platform-release"
}

variable "provisioning_model" {
  description = "SPOT | ON_DEMAND | FLEX_START. SPOT/ON_DEMAND are implemented via the provider scheduling block. FLEX_START (DWS) is a [VERIFY-AT-APPLY] path — see modules/gpu_session/main.tf for the gcloud fallback."
  type        = string
  default     = "SPOT"

  validation {
    condition     = contains(["SPOT", "ON_DEMAND", "FLEX_START"], var.provisioning_model)
    error_message = "provisioning_model must be SPOT, ON_DEMAND, or FLEX_START."
  }
}

variable "max_run_duration_hours" {
  description = "FLEX_START only: requested run window in hours (DWS Flex-start bounds the instance lifetime)."
  type        = number
  default     = 24
}

variable "startup_script" {
  description = "Optional metadata startup-script hook. Default none — provisioning stays in cloud/RUNBOOK's remote_job flow; this exists only for smoke bootstraps."
  type        = string
  default     = null
}

# ---------------------------------------------------------------------------
# Networking (C/D RoCE rung)
# ---------------------------------------------------------------------------

variable "enable_rdma" {
  description = "C/D act 2 RDMA rung only: create the RoCE data VPC + 8 rail subnets and attach MRDMA NICs. Leave false for sessions a/b, c/d act 1, and the C/D TCP rung."
  type        = bool
  default     = false
}

variable "rdma_network_profile_name" {
  description = "Zonal RDMA network profile name (A3 Ultra RoCE needs a VPC created from it), typically '<zone>-vpc-roce'. [VERIFY-AT-APPLY] `gcloud compute network-profiles list --filter='name~roce'`. Required when enable_rdma=true."
  type        = string
  default     = null
}

variable "rdma_rail_count" {
  description = "RoCE rail subnets (A3 Ultra: one data NIC per GPU = 8)."
  type        = number
  default     = 8
}

variable "mgmt_subnet_cidr" {
  description = "CIDR of the single mgmt subnet (sessions A/B live entirely here)."
  type        = string
  default     = "10.0.0.0/20"
}

# ---------------------------------------------------------------------------
# Cost controls
# ---------------------------------------------------------------------------

variable "budget_amount_usd" {
  description = "Run budget in USD; thresholds fire at 50/90/100 percent."
  type        = number
  default     = 500
}

# PLANNING-GRADE list-price estimates (USD/hour, on-demand, us-central1 class).
# These feed the estimated_hourly_cost output for the mandatory cost-report on
# every cloud action. They are NOT billing data — verify against the pricing
# page / billing export before quoting externally. [VERIFY-AT-APPLY]
variable "hourly_price_estimates" {
  description = "machine_type -> estimated on-demand USD/hour (planning-grade list prices)."
  type        = map(number)
  default = {
    "a2-ultragpu-1g" = 6.50  # 1x A100-80
    "a2-ultragpu-4g" = 26.00 # 4x A100-80
    "a3-ultragpu-8g" = 92.00 # 8x H200
  }
}

variable "spot_discount_factor" {
  description = "Planning-grade multiplier applied to the list estimate when provisioning_model=SPOT (Spot runs ~60-75% off list; 0.35 is a mid estimate)."
  type        = number
  default     = 0.35
}
