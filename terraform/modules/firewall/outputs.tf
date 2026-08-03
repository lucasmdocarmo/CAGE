output "firewall_rule_names" {
  value = concat(
    [google_compute_firewall.iap_ssh.name, google_compute_firewall.mgmt_internal.name],
    [for f in google_compute_firewall.data_internal : f.name],
  )
}
