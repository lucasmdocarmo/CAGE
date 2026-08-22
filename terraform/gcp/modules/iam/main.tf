# modules/iam — dedicated least-privilege service account per run.
#
#   * storage.objectAdmin ON THE RUN BUCKET ONLY (bucket-level IAM binding —
#     deliberately NOT a project-level grant; the SA cannot touch any other
#     bucket, which is what makes clean-room per-run isolation real).
#   * logging.logWriter + monitoring.metricWriter at project level (these two
#     have no narrower resource scope; they only allow writing telemetry).
#   * NO roles/editor, NO roles/owner, no primitive roles of any kind.
#
# The VM attaches this SA with the broad "cloud-platform" OAuth scope; the
# effective permission set is then narrowed by IAM (scopes cap, IAM grants —
# broad scope + narrow IAM is the recommended modern pattern).

terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
    }
  }
}

locals {
  # SA account_id: 6-30 chars, [a-z][a-z0-9-]*, ends alphanumeric.
  sa_id = replace(substr("cage-${var.run_slug}", 0, 30), "/-+$/", "")
}

resource "google_service_account" "run" {
  project      = var.project_id
  account_id   = local.sa_id
  display_name = "CAGE run ${var.run_slug} (dedicated, least-privilege)"
  description  = "Per-run SA. Destroyed with the run (teardown to TRUE $0)."
}

# Bucket-scoped: read/write objects in cage-<run_id> and nothing else.
resource "google_storage_bucket_iam_member" "bucket_object_admin" {
  bucket = var.bucket_name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.run.email}"
}

# Telemetry-only project grants.
resource "google_project_iam_member" "log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.run.email}"
}

resource "google_project_iam_member" "metric_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.run.email}"
}
