import requests
import json
import pandas as pd
import logging
import os
import time
from pathlib import Path

# Define the list for known locations
# These match the IESO zones/locations
Known_locations = {
    "Ottawa": {"lat": 45.4112, "lon": -75.6981},
    "Mississauga": {"lat": 43.5789, "lon": -79.6583},
    "Brampton": {"lat": 43.6834, "lon": -79.7663},
    "Hamilton": {"lat": 43.2501, "lon": -79.8496},
    "Toronto": {"lat": 43.7064, "lon": -79.3986},
    "London": {"lat": 42.9834, "lon": -81.233}
}

# Configuration of parameters
# Define the file outputs
# These base names will be prepended with the city name later
BASE_DIR = Path(__file__).resolve().parent
CSV_File = "forecast_daily_weather.csv"
JSON_File = "forecast_daily_weather.json"
Log_File = "forecast_fetch_log.txt"

# Define the variables for forecasting
Variables = [
    'temperature_2m_mean',
    'relative_humidity_2m_mean',
    'wind_speed_10m_mean',
    'cloud_cover_mean',
    'precipitation_sum',
    'shortwave_radiation_sum',
    'apparent_temperature_mean',
    'dew_point_2m_mean'
]

# Logging information to show what is happening and for debugging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG) 
if logger.hasHandlers():
    logger.handlers.clear()
file_handler_fc = logging.FileHandler(BASE_DIR / "Log" / Log_File, mode='w')
file_handler_fc.setLevel(logging.DEBUG)
file_formatter_fc = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler_fc.setFormatter(file_formatter_fc)
logger.addHandler(file_handler_fc)
console_handler_fc = logging.StreamHandler()
console_handler_fc.setLevel(logging.INFO)
console_formatter_fc = logging.Formatter('%(message)s')
console_handler_fc.setFormatter(console_formatter_fc)
logger.addHandler(console_handler_fc)


# Function to get the forecasted weather
def fetch_forecast_weather(lat, lon, city_name):
    logging.info(f"Starting weather data fetch for: {city_name}")

    # Build the URL based on the found coordinates
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&daily={','.join(Variables)}"
        f"&timezone=America%2FNew_York&forecast_days=16"
    )

    # Extracting the forecasted data
    try:
        response = requests.get(url, timeout=30) 
        response.raise_for_status()
        logging.info(f"Forecast data extracted for {city_name}.")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error extracting forecast data for {city_name}: {e}")
        return None
    except json.JSONDecodeError:
        logging.error("API response:")
        try:
            logging.error(response.text[:500])
        except Exception:
             logging.error("Could not get response from API.")
        return None


# Function to process the data
def process_weather_data(data, city_name):
    # Processes the JSON data to be written to JSON/CSV file
    logging.debug(f"Forecasted data type received: {type(data)}")
    if not isinstance(data, dict):
        logging.error(f"API did not return a 'dictionary'. Type received: {type(data)}")
        return None
    
    # This check is to make sure daily weather data is returned
    if 'daily' not in data or 'daily_units' not in data:
        logging.error(f"Missing 'daily' or 'daily_units' from data for {city_name}.")
        return None
    
    # Processing the daily weather data
    daily_data = data['daily']
    daily_units = data.get('daily_units', {})

    # Check for time - hourly/daily
    if 'time' not in daily_data:
        logging.error(f"Missing 'time' data in the forecast data for {city_name}.")
        return None
    dates = daily_data['time']
    logging.info(f"Processing {len(dates)} days of forecast data for {city_name}.")
    processed = {'date': dates}
    new_column_names = ['date']

    # Loop through variables to extract the data and apply new names for the output files
    for var in Variables:
        # Create the new column name according to agreed format
        unit_str = daily_units.get(var)
        if unit_str:
            new_col_name = f"{var} ({unit_str})"
        else:
            new_col_name = var
        new_column_names.append(new_col_name)

        if var not in daily_data:
            logging.warning(f"Variable '{var}' not found in forecast data. Skipping.")
            processed[new_col_name] = [None] * len(dates)
        elif not isinstance(daily_data[var], list) or len(daily_data[var]) != len(dates):
            logging.error(f"Data length error or invalid format for variable '{var}'. Skipping.")
            processed[new_col_name] = [None] * len(dates)
        else:
            processed[new_col_name] = daily_data[var]
            
    logging.info(f"Finished forecast data processing for {city_name}.")

    try:
        # Information about the processed data like size, shape, etc
        final_df = pd.DataFrame(processed)
        for col_name in new_column_names:
            if col_name != 'date':
                final_df[col_name] = pd.to_numeric(final_df[col_name], errors='coerce')
        final_df['date'] = pd.to_datetime(final_df['date'])
        logging.info(f"Final forecast data created for {city_name}.")
        return final_df
    except Exception as e:
        logging.error(f"Error creating final forecast data for {city_name}: {e}", exc_info=True)
        return None


# Function to output the desired files for load forecasting modules
def save_output_files(df, city_name):
    # Saves the processed data to CSV and JSON files in the Data folder
    if df is not None:
        save_directory = os.path.dirname(os.path.abspath(__file__))
        data_folder = os.path.join(save_directory, "Data")
        
        # Create the Data folder if it does not exist
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            logging.info(f"Created new directory: {data_folder}")
        
        # Modify the filename to include the location
        csv_filename = f"{city_name}_{CSV_File}"
        json_filename = f"{city_name}_{JSON_File}"

        csv_full_path = os.path.join(data_folder, csv_filename)
        json_full_path = os.path.join(data_folder, json_filename)
        
        try:
            df.to_csv(csv_full_path, index=False, encoding='utf-8-sig')
            logging.info(f"Forecast data saved to {csv_full_path}")
        except IOError as e:
            logging.error(f"Error saving forecast data to CSV ({csv_full_path}): {e}")
        try:
            df_for_json = df.where(pd.notna(df), None)
            # Format dates to string for JSON output
            df_for_json['date'] = df_for_json['date'].dt.strftime('%Y-%m-%d')
            df_for_json.to_json(json_full_path, orient='records', indent=2)
            logging.info(f"Forecast data saved to {json_full_path}")
        except IOError as e:
            logging.error(f"Error saving forecast data to JSON ({json_full_path}): {e}")
        except Exception as e:
             logging.error(f"Error occurred during JSON saving: {e}")

    else:
        logging.warning(f"No forecast data to save for {city_name}.")


if __name__ == "__main__":
    logging.info("Running batch forecast script for all known locations")
    
    # Do a testing run for all locations in the dictionary
    for city, coords in Known_locations.items():
        print(f"\n--- Extracting data for: {city} ---")
        
        weather_json = fetch_forecast_weather(coords['lat'], coords['lon'], city)
        
        if weather_json:
            final_df = process_weather_data(weather_json, city)
            save_output_files(final_df, city)
        else:
            logging.error(f"Forecast script failed for {city}.")
            
        # Mandatory pause between requests
        time.sleep(1)
        
    logging.info("\nForecast script finished execution.")