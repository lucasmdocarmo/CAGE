# CAGE GCP-fallback campaign stack — provider/CLI pins.
#
# Pinning discipline (iac-terraform doctrine): an unpinned provider turns an
# unrelated change into a surprise upgrade. Widen only deliberately.
terraform {
  required_version = ">= 1.9.0, < 2.0.0"

  required_providers {
    google = {
      source = "hashicorp/google"
      # >= 6.30 for network_profile on google_compute_network (RDMA/RoCE VPCs)
      # and FLEX_START-era scheduling fields. [VERIFY-AT-APPLY] if you bump
      # across a major: re-run `terraform plan` and read the diff before apply.
      version = ">= 6.30.0, < 8.0.0"
    }
  }

  # Backend: INTENTIONALLY local. Clean-room discipline = fresh bucket per
  # run-id and TRUE $0 after teardown; a long-lived GCS state bucket would be a
  # surviving resource. Keep the state file next to this run's working tree,
  # verify `terraform state list` is empty after destroy, then archive/delete it.
}
