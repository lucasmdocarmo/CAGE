# modules/bucket — the per-run results bucket: cage-<run_id>.
#
# force_destroy = true is INTENTIONAL and load-bearing for the teardown-to-$0
# discipline: `terraform destroy` must be able to take the bucket down WITH
# its objects. The safety is procedural and fail-closed, not technical:
# results are rsynced LOCAL and ledger-verified BEFORE any destroy (standing
# discipline "Pull results local BEFORE teardown"; teardown_vm.sh step [4/6]
# enforces it). A bucket that survives the run is a clean-room violation.

terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
    }
  }
}

resource "google_storage_bucket" "run" {
  project  = var.project_id
  name     = var.bucket_name
  location = var.region

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  # INTENTIONAL — see header. Do not "harden" this to false.
  force_destroy = true

  # Mirrors configs/gcs_lifecycle.json: observability spool is bulky and
  # reproducible — auto-delete after 14 days. Both prefixes covered (the run
  # convention writes observability/ at the run root; the legacy json used
  # analysis/observability/).
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age            = 14
      matches_prefix = ["observability/", "analysis/observability/"]
    }
  }

  labels = var.labels
}
