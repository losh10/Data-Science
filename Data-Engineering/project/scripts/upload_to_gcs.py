from google.cloud import storage
import os

# Configuration
BUCKET_NAME = "your-gcs-bucket-name"
SOURCE_FILE = "processed_data/consolidated_harare_weather.csv"
DEST_BLOB_NAME = "harare/consolidated_harare_weather.csv"

def upload_to_gcs(bucket_name, source_file, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file)
    print(f"✅ Uploaded {source_file} to gs://{bucket_name}/{destination_blob_name}")

if __name__ == "__main__":
    upload_to_gcs(BUCKET_NAME, SOURCE_FILE, DEST_BLOB_NAME)
