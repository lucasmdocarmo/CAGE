variable "project_id" {
  type = string
}

variable "zone" {
  type = string
}

variable "run_slug" {
  type = string
}

variable "session" {
  type = string
}

variable "machine_type" {
  description = "a2-ultragpu-1g | a2-ultragpu-4g | a3-ultragpu-8g (GPUs implicit)."
  type        = string
}

variable "node_count" {
  type    = number
  default = 1
}

variable "boot_disk_size_gb" {
  type    = number
  default = 400
}

variable "boot_disk_type" {
  type    = string
  default = "pd-ssd"
}

variable "image_family" {
  type = string
}

variable "image_project" {
  type    = string
  default = "deeplearning-platform-release"
}

variable "provisioning_model" {
  description = "SPOT | ON_DEMAND | FLEX_START (see module header for FLEX_START status)."
  type        = string
}

variable "max_run_duration_hours" {
  type    = number
  default = 24
}

variable "startup_script" {
  description = "Optional metadata startup-script hook; default none (remote_job flow owns provisioning)."
  type        = string
  default     = null
}

variable "subnet_self_link" {
  description = "mgmt subnet for the primary (GVNIC) NIC."
  type        = string
}

variable "rdma_subnet_self_links" {
  description = "RoCE rail subnets; one MRDMA NIC each. Empty except C/D RDMA rung."
  type        = list(string)
  default     = []
}

variable "service_account_email" {
  type = string
}

variable "labels" {
  type    = map(string)
  default = {}
}
