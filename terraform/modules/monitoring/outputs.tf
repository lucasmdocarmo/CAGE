output "budget_name" {
  value = google_billing_budget.run.display_name
}

output "notification_channel_id" {
  value = google_monitoring_notification_channel.email.id
}

output "uptime_alert_policy_name" {
  value = google_monitoring_alert_policy.instance_uptime.display_name
}
