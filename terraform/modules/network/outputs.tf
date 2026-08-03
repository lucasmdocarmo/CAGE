output "mgmt_network_self_link" {
  value = google_compute_network.mgmt.self_link
}

output "mgmt_subnet_self_link" {
  value = google_compute_subnetwork.mgmt.self_link
}

output "mgmt_subnet_cidr" {
  value = google_compute_subnetwork.mgmt.ip_cidr_range
}

output "data_network_self_links" {
  description = "RoCE data VPC self link(s); empty unless enable_rdma."
  value       = [for n in google_compute_network.data : n.self_link]
}

output "data_subnet_self_links" {
  description = "Rail subnet self links (8 for A3 Ultra); empty unless enable_rdma."
  value       = [for s in google_compute_subnetwork.rail : s.self_link]
}

output "data_subnet_cidrs" {
  value = [for s in google_compute_subnetwork.rail : s.ip_cidr_range]
}
