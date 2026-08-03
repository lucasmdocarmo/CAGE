output "bucket_name" {
  value = google_storage_bucket.run.name
}

output "bucket_url" {
  value = google_storage_bucket.run.url # gs://cage-<run_id>
}
