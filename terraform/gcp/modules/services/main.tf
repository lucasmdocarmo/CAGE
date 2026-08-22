# modules/services — enable the campaign's APIs.
# disable_on_destroy = false: destroying the run must NOT disable project APIs
# (other tooling in the project may rely on them; API disablement is also slow
# and can wedge a teardown mid-flight).

terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
    }
  }
}

locals {
  services = [
    "compute.googleapis.com",
    "storage.googleapis.com",
    "monitoring.googleapis.com",
    "billingbudgets.googleapis.com",
    "logging.googleapis.com",
  ]
}

resource "google_project_service" "enabled" {
  for_each = toset(local.services)

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}
