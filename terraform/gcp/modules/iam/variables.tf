variable "project_id" {
  type = string
}

variable "run_slug" {
  type = string
}

variable "bucket_name" {
  description = "Run bucket the SA gets objectAdmin on (bucket-level binding only)."
  type        = string
}
