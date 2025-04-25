import os
import requests
import csv
import logging
from datetime import datetime, timedelta, date
import re

API_KEY = "LC8BG8U9H6JY25FMYPAHZQ4GC"
CITY = "Harare"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
DATA_DIR = os.path.join(BASE_DIR, "weather_data")
os.makedirs(DATA_DIR, exist_ok=True)

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")



def fetch_weather(city, date_str):
    """
    Fetches weather data for a given city and date from Visual Crossing API.
    Returns the JSON response if successful, None if not.
    """
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/{date_str}?unitGroup=metric&include=hours&key={API_KEY}&contentType=json"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"Failed to fetch data for {date_str}: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logging.error(f"Error fetching weather for {date_str}: {e}")
        return None

def save_data_as_csv(data, filename):
    """Saves the hourly weather data to a CSV file."""
    header = ["datetime", "temp", "precip", "preciptype", "windspeed"]

    try:
        if not data or 'days' not in data or not data['days'] or 'hours' not in data['days'][0]:
            logging.warning(f"Hourly data missing or incomplete in response for {filename}")
            return False

        hourly_data = data['days'][0]['hours']

        # Ensure the data directory exists (harmless if it already does)
        os.makedirs(DATA_DIR, exist_ok=True)

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)  # Write header row

            # Write data rows
            for hour_data in hourly_data:
                preciptype_val = hour_data.get('preciptype')
                preciptype_str = ','.join(preciptype_val) if isinstance(preciptype_val, list) else (str(preciptype_val) if preciptype_val is not None else '')

                row = [
                    hour_data.get('datetime', ''),
                    hour_data.get('temp', ''),
                    hour_data.get('precip', ''),
                    preciptype_str,
                    hour_data.get('windspeed', '')
                ]
                writer.writerow(row)
        return True
    except (IOError, OSError, KeyError, IndexError, csv.Error) as e:
        logging.error(f"Failed to write CSV file {filename}: {e}")
        return False

def get_last_downloaded_date():
    """
    Finds the last date for which data was downloaded based on filenames in the DATA_DIR.
    Returns the date object or None.
    """
    if not os.path.exists(DATA_DIR):
        logging.info(f"Data directory '{DATA_DIR}' not found. Starting fresh.")
        return None

    dates = []
    pattern = re.compile(rf"{CITY.replace(' ', '_')}_(\d{{4}}-\d{{2}}-\d{{2}})\.csv")

    for filename in os.listdir(DATA_DIR):
        match = pattern.search(filename)
        if match:
            try:
                file_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
                dates.append(file_date)
            except ValueError:
                logging.warning(f"Found file '{filename}' with unexpected date format. Skipping.")
                continue

    return max(dates) if dates else None

def build_filename(city, date_str):
    """Generates filename for a given city and date."""
    return os.path.join(DATA_DIR, f"{city.replace(' ', '_')}_{date_str}.csv")

def download_range(start_date, end_date):
    """
    Downloads weather data day by day from start_date to end_date.
    Skips days where the file already exists.
    """
    current_date = start_date
    logging.info(f"Attempting download from {start_date} to {end_date}.")

    if start_date > end_date:
        logging.info("Start date is after end date. No download needed.")
        return

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        filename = build_filename(CITY, date_str)

        # Check if file already exists BEFORE attempting fetch
        if os.path.exists(filename):
            logging.info(f"Already downloaded: {filename}")
        else:
            # Only fetch if file doesn't exist
            logging.info(f"Fetching data for {date_str}...")
            data = fetch_weather(CITY, date_str)
            if data:
                if save_data_as_csv(data, filename):
                    logging.info(f"Saved: {filename}")

        # Move to the next date
        current_date += timedelta(days=1)

# --- Main Execution ---

if __name__ == "__main__":
    # Determine end date (today)
    end_date = date.today()

    # Determine start date by finding the last downloaded file
    last_downloaded = get_last_downloaded_date()

    if last_downloaded:
        # If we have previous data, start from the day AFTER the last download
        start_date = last_downloaded + timedelta(days=1)
        logging.info(f"Last downloaded date found: {last_downloaded}. Resuming from: {start_date}")
    else:
        # If no previous data, start from 6 months ago
        logging.info("No previous download found. Starting initial download for the last 6 months.")
        start_date = end_date - timedelta(days=180)
        logging.info(f"Calculated initial start date (6 months ago): {start_date}")

    # Initiate the download process
    download_range(start_date, end_date)

    logging.info("✅ Finished downloading weather data for this run.")
