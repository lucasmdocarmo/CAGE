variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "run_slug" {
  description = "Sanitized run id — prefixes every network resource name."
  type        = string
}

variable "mgmt_subnet_cidr" {
  type    = string
  default = "10.0.0.0/20"
}

variable "enable_rdma" {
  description = "C/D act-2 RDMA rung: create the RoCE data VPC + rail subnets."
  type        = bool
  default     = false
}

variable "rdma_network_profile_name" {
  description = "Zonal RDMA network profile name (e.g. us-central1-a-vpc-roce). Required when enable_rdma=true. [VERIFY-AT-APPLY]"
  type        = string
  default     = null
}

variable "rdma_rail_count" {
  description = "Rail subnets = data NICs per node (A3 Ultra: 8)."
  type        = number
  default     = 8
}
