variable "project_id" {
  type = string
}

variable "run_slug" {
  type = string
}

variable "mgmt_network_self_link" {
  type = string
}

variable "mgmt_internal_cidrs" {
  description = "CIDRs allowed to talk freely inside the mgmt VPC (the run's own subnets)."
  type        = list(string)
}

variable "data_network_self_links" {
  description = "RoCE data VPC self links; empty unless the RDMA rung is enabled."
  type        = list(string)
  default     = []
}

variable "data_internal_cidrs" {
  description = "Rail subnet CIDRs for the data-VPC internal allow."
  type        = list(string)
  default     = []
}
