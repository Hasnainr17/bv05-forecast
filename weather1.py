# get_weather_forecast.py
import requests
import json
import pandas as pd
import logging
import os # Import os module

# --- Configuration ---
# !!! IMPORTANT: REPLACE THIS with your new API URL for ONE location (e.g., Toronto) !!!
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast?latitude=43.7064&longitude=-79.3986&daily=temperature_2m_mean,apparent_temperature_mean,relative_humidity_2m_mean,wind_speed_10m_mean,cloud_cover_mean,dew_point_2m_mean,precipitation_sum,shortwave_radiation_sum&timezone=America%2FNew_York&forecast_days=16" # <-- REPLACE

# --- Output File Names ---
OUTPUT_CSV_FILE_FORECAST = "forecast_daily_weather.csv"
OUTPUT_JSON_FILE_FORECAST = "forecast_daily_weather.json"
LOG_FILE_FORECAST = "forecast_fetch_log.txt"
# --- End Output File Names ---

# Define the variables you requested in your URL (use the exact API names)
VARIABLES = [
    'temperature_2m_mean',
    'relative_humidity_2m_mean',
    'wind_speed_10m_mean',
    'cloud_cover_mean',
    'precipitation_sum',
    'shortwave_radiation_sum',
    'apparent_temperature_mean',
    'dew_point_2m_mean'
]
# COLUMN_NAME_MAP has been removed

# --- MODIFIED LOGGING SETUP ---
logger = logging.getLogger()
logger.setLevel(logging.DEBUG) 

if logger.hasHandlers():
    logger.handlers.clear()

# File Handler (INFO + DEBUG)
file_handler_fc = logging.FileHandler(LOG_FILE_FORECAST, mode='w')
file_handler_fc.setLevel(logging.DEBUG)
file_formatter_fc = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler_fc.setFormatter(file_formatter_fc)
logger.addHandler(file_handler_fc)

# Console Handler (INFO only)
console_handler_fc = logging.StreamHandler()
console_handler_fc.setLevel(logging.INFO)
console_formatter_fc = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler_fc.setFormatter(console_formatter_fc)
logger.addHandler(console_handler_fc)
# --- END MODIFIED LOGGING SETUP ---


def fetch_forecast_weather():
    """Fetches forecast weather data from the Open-Meteo API."""
    url = FORECAST_API_URL
    logging.info(f"Fetching forecast data from: {url}")
    try:
        response = requests.get(url, timeout=30) 
        response.raise_for_status()
        logging.info("Forecast data fetched successfully.")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching forecast data: {e}")
        return None
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON response from forecast API. Response text was:")
        try:
            logging.error(response.text[:500])
        except Exception:
             logging.error("Could not get response text.")
        return None

def process_weather_data(data):
    """Processes the JSON data and returns a DataFrame."""

    logging.debug(f"Raw forecast data type received: {type(data)}")

    # --- UPDATED CHECK: Expect a dictionary, not a list ---
    if not isinstance(data, dict):
        logging.error(f"API did not return a dictionary as expected. Type received: {type(data)}")
        if isinstance(data, list):
             logging.error("API returned a LIST. This script is for a single location. Did you use the multi-city API URL by mistake?")
        return None

    if 'daily' not in data or 'daily_units' not in data:
        logging.error("Invalid data from API (missing 'daily' or 'daily_units' key).")
        logging.error(f"Content of data (first 500 chars): {str(data)[:500]}")
        return None

    daily_data = data['daily']
    daily_units = data.get('daily_units', {}) # Get units dict safely
    logging.debug(f"Keys in daily forecast data: {list(daily_data.keys())}")
    logging.debug(f"Found daily forecast units: {daily_units}")

    if 'time' not in daily_data:
        logging.error("Missing 'time' data in the forecast response.")
        return None

    dates = daily_data['time']
    logging.info(f"Processing {len(dates)} days of forecast data.")
    logging.debug(f"First few forecast dates: {dates[:5]}")

    processed = {'date': dates}
    new_column_names = ['date'] # Keep track of our new column names

    # Loop through variables to extract them and apply new names
    for var in VARIABLES:
        # --- Dynamically create the new column name ---
        unit_str = daily_units.get(var) # Get the unit, e.g., "°C" or "%"
        if unit_str:
            new_col_name = f"{var} ({unit_str})"
        else:
            new_col_name = var # Fallback if no unit found
        new_column_names.append(new_col_name) # Add to our list
        # --- End dynamic name creation ---

        if var not in daily_data:
            logging.warning(f"Variable '{var}' not found in forecast response. Skipping.")
            processed[new_col_name] = [None] * len(dates)
        elif not isinstance(daily_data[var], list) or len(daily_data[var]) != len(dates):
            logging.error(f"Data length mismatch or invalid format for forecast variable '{var}'. Skipping.")
            processed[new_col_name] = [None] * len(dates)
        else:
            processed[new_col_name] = daily_data[var]
            logging.info(f"Processing forecast variable: {var} (as {new_col_name})")
            logging.debug(f"  Forecast Variable '{new_col_name}' first few values: {processed[new_col_name][:5]}")

    logging.info("--- Finished Forecast Data Extraction ---")

    try:
        final_df = pd.DataFrame(processed)
        
        # --- Robust Numeric Conversion ---
        for col_name in new_column_names:
            if col_name != 'date': # Skip the date column
                final_df[col_name] = pd.to_numeric(final_df[col_name], errors='coerce')
        
        final_df['date'] = pd.to_datetime(final_df['date'])
        logging.info("Final forecast DataFrame created successfully.")
        logging.debug(f"Final forecast DataFrame shape: {final_df.shape}")
        logging.debug(f"Final forecast DataFrame columns: {list(final_df.columns)}")
        logging.debug(f"Final forecast DataFrame head:\n{final_df.head().to_string()}")
        return final_df
    except Exception as e:
        logging.error(f"Error creating final forecast DataFrame: {e}", exc_info=True)
        return None

def save_output_files(df, csv_filename, json_filename):
    """Saves the DataFrame to CSV and JSON files."""
    if df is not None:
        save_directory = os.path.dirname(os.path.abspath(__file__))
        csv_full_path = os.path.join(save_directory, csv_filename)
        json_full_path = os.path.join(save_directory, json_filename)
        logging.info(f"Attempting to save forecast output files in: {save_directory}")

        # Save CSV
        try:
            df.to_csv(csv_full_path, index=False, encoding='utf-8-sig')
            logging.info(f"Forecast data successfully saved to {csv_full_path}")
        except IOError as e:
            logging.error(f"Error saving forecast data to CSV ({csv_full_path}): {e}")

        # Save JSON
        try:
            # Convert NaN to None for JSON compatibility before saving
            df_for_json = df.where(pd.notna(df), None)
            df_for_json.to_json(json_full_path, orient='records', date_format='iso', indent=2)
            logging.info(f"Forecast data successfully saved to {json_full_path}")
        except IOError as e:
            logging.error(f"Error saving forecast data to JSON ({json_full_path}): {e}")
        except Exception as e:
             logging.error(f"An unexpected error occurred during JSON saving: {e}")

    else:
        logging.warning("No forecast DataFrame to save.")

# --- Main function for module use ---
def fetch_and_process_forecast():
    """Fetches and processes forecast data, returning a DataFrame."""
    logging.info("Executing fetch_and_process_forecast function...")
    weather_json = fetch_forecast_weather()
    if weather_json:
        final_df = process_weather_data(weather_json)
        return final_df
    else:
        logging.error("Failed to fetch forecast data in fetch_and_process_forecast.")
        return None # Return None if fetching failed

# --- Allow direct execution for testing ---
if __name__ == "__main__":
    logging.info("Running forecast script directly for testing...")
    
    if "YOUR_TORONTO_ONLY_HISTORICAL_API_URL_GOES_HERE" in FORECAST_API_URL: # Simple check for placeholder
        logging.error("="*50)
        logging.error("SCRIPT HALTED: You have not replaced the placeholder API URL.")
        logging.error("Please replace 'FORECAST_API_URL' at the top of the script with your actual Open-Meteo URL.")
        logging.error("="*50)
    else:
        forecast_df = fetch_and_process_forecast()
        if forecast_df is not None:
            save_output_files(forecast_df, OUTPUT_CSV_FILE_FORECAST, OUTPUT_JSON_FILE_FORECAST)
            logging.info(f"Test forecast files saved. Shape: {forecast_df.shape}")
        else:
            logging.error("Forecast script failed during direct execution. Output files not created.")
    logging.info("Forecast script finished direct execution.")