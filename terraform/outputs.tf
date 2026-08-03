# Root outputs — everything the RUNBOOK / cost-report needs after plan/apply.

output "bucket_url" {
  description = "Run bucket (clean-room: fresh per run-id, destroyed with the run)."
  value       = module.bucket.bucket_url
}

output "instance_names" {
  description = "GPU node names."
  value       = module.gpu_session.instance_names
}

output "instance_internal_ips" {
  description = "Internal IPs (mgmt NIC). Nodes have NO external IPs — SSH via IAP only."
  value       = module.gpu_session.instance_internal_ips
}

output "service_account_email" {
  description = "Per-run service account the nodes run as."
  value       = module.iam.service_account_email
}

output "ssh_via_iap" {
  description = "SSH helper (IAP tunnel — the only ingress path)."
  value = [
    for name in module.gpu_session.instance_names :
    "gcloud compute ssh ${name} --project ${var.project_id} --zone ${var.zone} --tunnel-through-iap"
  ]
}

# PLANNING-GRADE estimate from var.hourly_price_estimates (list prices), NOT
# billing data. Quote this (x expected wall-clock hours) in the cost+ETA report
# that gates user approval of any apply.
output "estimated_hourly_cost" {
  description = "Planning-grade USD/hour for the requested nodes (list-price map x node_count, Spot factor applied when SPOT)."
  value = format(
    "USD %.2f/hour (PLANNING-GRADE: %d x %s @ %.2f list%s) — verify against pricing page / billing export",
    lookup(var.hourly_price_estimates, var.machine_type, 0) * var.node_count * (var.provisioning_model == "SPOT" ? var.spot_discount_factor : 1.0),
    var.node_count,
    var.machine_type,
    lookup(var.hourly_price_estimates, var.machine_type, 0),
    var.provisioning_model == "SPOT" ? format(", spot factor %.2f", var.spot_discount_factor) : ""
  )
}
