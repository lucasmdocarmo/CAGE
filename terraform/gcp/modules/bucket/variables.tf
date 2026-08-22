variable "project_id" {
  type = string
}

variable "region" {
  description = "Bucket location — keep it in the run's region (egress + locality)."
  type        = string
}

variable "bucket_name" {
  description = "cage-<run_id> (fresh per run — clean-room discipline)."
  type        = string
}

variable "labels" {
  description = "Common run labels (agent-run/session/model) for the orphan sweep."
  type        = map(string)
  default     = {}
}
