from google.cloud import bigquery

# Config
PROJECT_ID = "your-project-id"
DATASET_ID = "your_dataset"
TABLE_ID = "harare_weather"

GCS_URI = "gs://your-gcs-bucket-name/harare/consolidated_harare_weather.csv"

def load_csv_to_bq():
    client = bigquery.Client()

    table_ref = client.dataset(DATASET_ID).table(TABLE_ID)

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
        write_disposition="WRITE_TRUNCATE"
    )

    load_job = client.load_table_from_uri(
        GCS_URI, table_ref, job_config=job_config
    )

    load_job.result()
    print(f"✅ Data loaded into {TABLE_ID} in BigQuery.")

if __name__ == "__main__":
    load_csv_to_bq()
