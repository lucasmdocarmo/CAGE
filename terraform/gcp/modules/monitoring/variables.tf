variable "project_id" {
  type = string
}

variable "billing_account_id" {
  description = "Billing account (XXXXXX-XXXXXX-XXXXXX) the budget attaches to."
  type        = string
}

variable "run_slug" {
  type = string
}

variable "budget_amount_usd" {
  type    = number
  default = 500
}

variable "alert_email" {
  type = string
}
