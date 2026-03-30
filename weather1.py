# weather1.py
import requests
import json
import pandas as pd
import logging
import os

# -----------------------------
# City configuration
# -----------------------------
CITY_COORDS = {
    "Toronto": {"lat": 43.6532, "lon": -79.3832},
    "Ottawa": {"lat": 45.4215, "lon": -75.6972},
    "Hamilton": {"lat": 43.2557, "lon": -79.8711},
    "London": {"lat": 42.9849, "lon": -81.2453},
    "Mississauga": {"lat": 43.5890, "lon": -79.6441},
    "Brampton": {"lat": 43.7315, "lon": -79.7624},
}

# -----------------------------
# Output file names
# -----------------------------
OUTPUT_CSV_FILE_FORECAST = "forecast_daily_weather.csv"
OUTPUT_JSON_FILE_FORECAST = "forecast_daily_weather.json"
LOG_FILE_FORECAST = "forecast_fetch_log.txt"

# Variables requested from Open-Meteo
VARIABLES = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "cloud_cover_mean",
    "precipitation_sum",
    "shortwave_radiation_sum",
    "apparent_temperature_mean",
    "dew_point_2m_mean",
]

# -----------------------------
# Logging setup
# -----------------------------
logger = logging.getLogger("weather_forecast")
logger.setLevel(logging.DEBUG)

if logger.hasHandlers():
    logger.handlers.clear()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
log_full_path = os.path.join(BASE_DIR, LOG_FILE_FORECAST)

file_handler_fc = logging.FileHandler(log_full_path, mode="w")
file_handler_fc.setLevel(logging.DEBUG)
file_formatter_fc = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler_fc.setFormatter(file_formatter_fc)
logger.addHandler(file_handler_fc)

console_handler_fc = logging.StreamHandler()
console_handler_fc.setLevel(logging.INFO)
console_formatter_fc = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler_fc.setFormatter(console_formatter_fc)
logger.addHandler(console_handler_fc)


# -----------------------------
# URL builder
# -----------------------------
def build_forecast_api_url(city: str) -> str:
    if city not in CITY_COORDS:
        raise ValueError(f"Unsupported city: {city}")

    lat = CITY_COORDS[city]["lat"]
    lon = CITY_COORDS[city]["lon"]

    return (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}"
        f"&longitude={lon}"
        "&daily=temperature_2m_mean,apparent_temperature_mean,"
        "relative_humidity_2m_mean,wind_speed_10m_mean,"
        "cloud_cover_mean,dew_point_2m_mean,precipitation_sum,shortwave_radiation_sum"
        "&timezone=America%2FToronto"
        "&forecast_days=16"
    )


# -----------------------------
# Fetch weather
# -----------------------------
def fetch_forecast_weather(city: str = "Toronto"):
    """Fetches forecast weather data from the Open-Meteo API for a selected city."""
    url = build_forecast_api_url(city)
    logger.info(f"Fetching forecast data for {city} from: {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        logger.info(f"Forecast data fetched successfully for {city}.")
        return response.json()

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching forecast data for {city}: {e}")
        return None

    except json.JSONDecodeError:
        logger.error("Failed to decode JSON response from forecast API.")
        try:
            logger.error(response.text[:500])
        except Exception:
            logger.error("Could not retrieve response text.")
        return None


# -----------------------------
# Process JSON -> DataFrame
# -----------------------------
def process_weather_data(data, city: str = "Toronto"):
    """Processes forecast JSON data and returns a DataFrame."""

    logger.debug(f"Raw forecast data type received for {city}: {type(data)}")

    if not isinstance(data, dict):
        logger.error(f"API did not return a dictionary as expected for {city}. Type received: {type(data)}")
        return None

    if "daily" not in data or "daily_units" not in data:
        logger.error(f"Invalid forecast data for {city} (missing 'daily' or 'daily_units').")
        logger.error(f"Content of data (first 500 chars): {str(data)[:500]}")
        return None

    daily_data = data["daily"]
    daily_units = data.get("daily_units", {})
    logger.debug(f"Keys in daily forecast data for {city}: {list(daily_data.keys())}")

    if "time" not in daily_data:
        logger.error(f"Missing 'time' data in forecast response for {city}.")
        return None

    dates = daily_data["time"]
    logger.info(f"Processing {len(dates)} forecast days for {city}.")

    processed = {"date": dates}
    new_column_names = ["date"]

    for var in VARIABLES:
        unit_str = daily_units.get(var)
        new_col_name = f"{var} ({unit_str})" if unit_str else var
        new_column_names.append(new_col_name)

        if var not in daily_data:
            logger.warning(f"Variable '{var}' not found in forecast response for {city}.")
            processed[new_col_name] = [None] * len(dates)
        elif not isinstance(daily_data[var], list) or len(daily_data[var]) != len(dates):
            logger.error(f"Data length mismatch for forecast variable '{var}' in {city}.")
            processed[new_col_name] = [None] * len(dates)
        else:
            processed[new_col_name] = daily_data[var]

    try:
        final_df = pd.DataFrame(processed)

        for col_name in new_column_names:
            if col_name != "date":
                final_df[col_name] = pd.to_numeric(final_df[col_name], errors="coerce")

        final_df["date"] = pd.to_datetime(final_df["date"])
        final_df["city"] = city

        logger.info(f"Final forecast DataFrame created successfully for {city}.")
        logger.debug(f"Final forecast DataFrame shape for {city}: {final_df.shape}")
        return final_df

    except Exception as e:
        logger.error(f"Error creating final forecast DataFrame for {city}: {e}", exc_info=True)
        return None


# -----------------------------
# Save outputs
# -----------------------------
def save_output_files(df, csv_filename, json_filename):
    """Saves the DataFrame to CSV and JSON files."""
    if df is not None:
        save_directory = os.path.dirname(os.path.abspath(__file__))
        csv_full_path = os.path.join(save_directory, csv_filename)
        json_full_path = os.path.join(save_directory, json_filename)

        logger.info(f"Attempting to save forecast output files in: {save_directory}")

        try:
            df.to_csv(csv_full_path, index=False, encoding="utf-8-sig")
            logger.info(f"Forecast data successfully saved to {csv_full_path}")
        except IOError as e:
            logger.error(f"Error saving forecast data to CSV ({csv_full_path}): {e}")

        try:
            df_for_json = df.where(pd.notna(df), None)
            df_for_json.to_json(json_full_path, orient="records", date_format="iso", indent=2)
            logger.info(f"Forecast data successfully saved to {json_full_path}")
        except IOError as e:
            logger.error(f"Error saving forecast data to JSON ({json_full_path}): {e}")
        except Exception as e:
            logger.error(f"Unexpected error during JSON saving: {e}")

    else:
        logger.warning("No forecast DataFrame to save.")


# -----------------------------
# Main function for app use
# -----------------------------
def fetch_and_process_forecast(city: str = "Toronto"):
    """Fetches and processes forecast data for the selected city, returning a DataFrame."""
    logger.info(f"Executing fetch_and_process_forecast for {city}...")
    weather_json = fetch_forecast_weather(city=city)

    if weather_json:
        final_df = process_weather_data(weather_json, city=city)
        return final_df
    else:
        logger.error(f"Failed to fetch forecast data for {city}.")
        return None


# -----------------------------
# Direct execution for testing
# -----------------------------
if __name__ == "__main__":
    logger.info("Running weather1.py directly for testing...")

    test_city = "Toronto"
    forecast_df = fetch_and_process_forecast(city=test_city)

    if forecast_df is not None:
        save_output_files(
            forecast_df,
            OUTPUT_CSV_FILE_FORECAST,
            OUTPUT_JSON_FILE_FORECAST
        )
        logger.info(f"Test forecast files saved for {test_city}. Shape: {forecast_df.shape}")
    else:
        logger.error(f"Forecast script failed during direct execution for {test_city}.")

    logger.info("weather1.py finished direct execution.")