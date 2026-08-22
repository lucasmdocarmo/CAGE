output "enabled_services" {
  description = "APIs enabled for the run."
  value       = [for s in google_project_service.enabled : s.service]
}
