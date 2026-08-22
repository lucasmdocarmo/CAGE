# modules/monitoring — the two cost tripwires.
#
#   1. google_billing_budget: run budget with 0.5 / 0.9 / 1.0 threshold rules,
#      notifying an email monitoring channel. Caller needs
#      billing.budgets.create on the billing account. [VERIFY-AT-APPLY]
#   2. Runaway-uptime alert: fires when a VM labeled with this run's agent-run
#      label has been up > 18h continuously. Uses the standard agent-free
#      metric compute.googleapis.com/instance/uptime_total (gauge, seconds
#      since boot) — a threshold condition on it is plain-vanilla Cloud
#      Monitoring. [VERIFY-AT-APPLY] only the metadata.user_labels filter key
#      syntax if the policy matches nothing: confirm with
#      `gcloud monitoring metrics-scopes ...` / metrics explorer on a live VM.

terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
    }
  }
}

data "google_project" "this" {
  project_id = var.project_id
}

resource "google_monitoring_notification_channel" "email" {
  project      = var.project_id
  display_name = "cage-${var.run_slug}-alerts"
  type         = "email"

  labels = {
    email_address = var.alert_email
  }
}

resource "google_billing_budget" "run" {
  billing_account = var.billing_account_id
  display_name    = "cage-${var.run_slug}-budget"

  budget_filter {
    # Budgets API wants the project NUMBER path.
    projects               = ["projects/${data.google_project.this.number}"]
    credit_types_treatment = "INCLUDE_ALL_CREDITS"
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(floor(var.budget_amount_usd))
    }
  }

  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 0.9
  }
  threshold_rules {
    threshold_percent = 1.0
  }

  all_updates_rule {
    monitoring_notification_channels = [google_monitoring_notification_channel.email.id]
    # Billing admins/users on the account also get the default emails.
    disable_default_iam_recipients = false
  }
}

# Runaway-cost tripwire: no CAGE session legitimately keeps one node up >18h
# without a human in the loop (the campaign discipline is provision -> run ->
# pull results -> teardown). 18h of continuous uptime means a teardown was
# missed or a run wedged — page the operator.
resource "google_monitoring_alert_policy" "instance_uptime" {
  project      = var.project_id
  display_name = "cage-${var.run_slug}-vm-uptime-over-18h"
  combiner     = "OR"

  conditions {
    display_name = "labeled VM up > 18h continuously"

    condition_threshold {
      filter = "resource.type = \"gce_instance\" AND metric.type = \"compute.googleapis.com/instance/uptime_total\" AND metadata.user_labels.\"agent-run\" = \"${var.run_slug}\""

      comparison      = "COMPARISON_GT"
      threshold_value = 64800 # 18h in seconds
      duration        = "300s"

      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_MAX"
      }

      trigger {
        count = 1
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]

  documentation {
    mime_type = "text/markdown"
    content   = <<-EOT
      **CAGE runaway-cost tripwire** (run `${var.run_slug}`).

      A GPU VM labeled `agent-run=${var.run_slug}` has been running for more
      than 18 hours continuously. Either a teardown was missed or the run is
      wedged. Follow the fail-closed teardown checklist in terraform/main.tf:
      pull results local + ledger-verify FIRST, then `terraform destroy`, then
      the label-keyed orphan sweep, then verify TRUE $0 (bucket gone).
    EOT
  }
}
