import os
import csv
import logging
from datetime import datetime

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # One level up from /scripts

# --- Path Definitions ---
DATA_DIR = os.path.join(BASE_DIR, 'weather_data')       # Points to project_root/weather_data
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data') # Points to project_root/processed_data
CONSOLIDATED_FILE = os.path.join(PROCESSED_DIR, 'consolidated_weather_data.csv')
# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Ensure the processed data folder exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

def consolidate_data():
    """Consolidates all downloaded weather data into one CSV file with date as the first column."""
    data_rows = []  # To store all rows before sorting

    # Walk through the data directory and read the downloaded CSV files
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".csv"):
            filepath = os.path.join(DATA_DIR, filename)
            try:
                with open(filepath, 'r', newline='', encoding='utf-8') as infile:
                    reader = csv.reader(infile)
                    next(reader)  # Skip the header row

                    # Extract date from the filename (assumed to be formatted as "City_YYYY-MM-DD.csv")
                    date_str = filename.split('_')[1].split('.')[0]  # Extracting "YYYY-MM-DD"
                    for row in reader:
                        if row:  # Check for empty rows
                            # Insert the date as the first column in the row
                            row.insert(0, date_str)  # Add the date to the beginning of the row
                            data_rows.append(row)
                logging.info(f"Successfully processed file: {filename}")
            except Exception as e:
                logging.error(f"Error processing file {filename}: {e}")

    # Sort the rows by the date column (first column) in descending order (latest to oldest)
    data_rows.sort(key=lambda x: datetime.strptime(x[0], "%Y-%m-%d"), reverse=True)

    # Write the sorted data to the output CSV file
    with open(CONSOLIDATED_FILE, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        # Write the header row with 'date' as the first column
        writer.writerow(["date", "datetime", "temp", "precip", "preciptype", "windspeed"])

        # Write all sorted data rows
        writer.writerows(data_rows)

    logging.info(f"Consolidation complete. Data saved to {CONSOLIDATED_FILE}")

if __name__ == "__main__":
    consolidate_data()
