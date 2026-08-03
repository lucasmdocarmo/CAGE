output "instance_names" {
  value = [for i in google_compute_instance.node : i.name]
}

output "instance_internal_ips" {
  description = "mgmt-NIC internal IPs (no external IPs exist)."
  value       = [for i in google_compute_instance.node : i.network_interface[0].network_ip]
}

output "instance_self_links" {
  value = [for i in google_compute_instance.node : i.self_link]
}
