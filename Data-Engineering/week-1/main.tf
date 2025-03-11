terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "3.5.0"
    }
  }
}

provider "google" {
  credentials = file("/home/malloy/Desktop/workspace/Data-Engineering/week-1/gcp key/data-engineering-453410-aee809a22595.json")  # Ensure correct path
  project     = "data-engineering-453410"  # Add project ID
  region      = "europe-west1"  # Add default region (if applicable)
}

resource "google_storage_bucket" "taxi-bucket" {
  name          = "taxi-bucket-${var.project_id}"
  location      = "EU"
  force_destroy = true

  lifecycle_rule {
    condition {
      age = 1
    }
    action {
      type = "AbortIncompleteMultipartUpload"
    }
  }
}

resource "google_bigquery_dataset" "taxi-dataset" {
  dataset_id = "taxi_dataset"
  location   = "EU"
}
