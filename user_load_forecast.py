"""
user_load_forecast.py
=====================
Azure App Service–ready script for:
- Reading user-uploaded CSV/XLSX
- Validating input
- Selecting location-specific model
- Forecasting load (Residential + CI)
- Saving output to Excel

NOTE:
- Assumes this script is in SAME directory as:
    load_forecast_json_and_csv_upgraded.py
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Import functions from your existing module
from load_forecast_json_and_csv_upgraded import (
    train_models_from_historical_csv,
    forecast_daily_load,
    LOCATIONS,
    DATA_DIR
)

# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "Forecasted Output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Expected column names (based on your template)
REQUIRED_COLUMNS = [
    "Date",
    "temperature_2m_mean (°C)",
    "wind_speed_10m_mean (km/h)"
]

# Validation ranges
TEMP_RANGE = (-25, 35)
WIND_RANGE = (0, 35)

TODAY = datetime.today()
MIN_DATE = TODAY - timedelta(days=365)
MAX_DATE = TODAY + timedelta(days=365)


# -----------------------------
# Validation Function
# -----------------------------
def validate_input(df: pd.DataFrame):
    errors = []

    # Check columns
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")

    if errors:
        return False, errors

    # Convert date
    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception:
        errors.append("Invalid data in the date column. Ensure correct format (YYYY-MM-DD).")

    # Check date range
    if "Date" in df.columns:
        invalid_dates = df[
            (df["Date"] < MIN_DATE) | (df["Date"] > MAX_DATE)
        ]
        if not invalid_dates.empty:
            errors.append(
                "Date values must be within 1 year before or after today."
            )

    # Temperature validation
    df["temperature_2m_mean (°C)"] = pd.to_numeric(
        df["temperature_2m_mean (°C)"], errors="coerce"
    )
    if df["temperature_2m_mean (°C)"].isna().any():
        errors.append("Invalid temperature values detected.")

    if not df[
        (df["temperature_2m_mean (°C)"] < TEMP_RANGE[0]) |
        (df["temperature_2m_mean (°C)"] > TEMP_RANGE[1])
    ].empty:
        errors.append("Temperature must be between -25°C and 35°C.")

    # Wind validation
    df["wind_speed_10m_mean (km/h)"] = pd.to_numeric(
        df["wind_speed_10m_mean (km/h)"], errors="coerce"
    )
    if df["wind_speed_10m_mean (km/h)"].isna().any():
        errors.append("Invalid wind speed values detected.")

    if not df[
        (df["wind_speed_10m_mean (km/h)"] < WIND_RANGE[0]) |
        (df["wind_speed_10m_mean (km/h)"] > WIND_RANGE[1])
    ].empty:
        errors.append("Wind speed must be between 0 and 35 km/h.")

    return len(errors) == 0, errors


# -----------------------------
# Main Forecast Function
# -----------------------------
def run_user_forecast(input_file_path: str, selected_location: str):
    """
    Main function to be called by Azure app

    Parameters:
        input_file_path (str): path to uploaded file
        selected_location (str): one of 6 cities
    """

    # -----------------------------
    # 1. Validate location
    # -----------------------------
    if selected_location not in LOCATIONS:
        raise ValueError(
            f"Invalid location. Choose from: {list(LOCATIONS.keys())}"
        )

    # -----------------------------
    # 2. Read input file
    # -----------------------------
    file_path = Path(input_file_path)

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or XLSX.")

    # -----------------------------
    # 3. Validate data
    # -----------------------------
    is_valid, errors = validate_input(df)

    if not is_valid:
        raise ValueError(" | ".join(errors))

    # -----------------------------
    # 4. Train model based on location
    # -----------------------------
    hist_file = DATA_DIR / LOCATIONS[selected_location]

    if not hist_file.exists():
        raise FileNotFoundError(f"Historical data not found for {selected_location}")

    res_model, ci_model = train_models_from_historical_csv(hist_file)

    # -----------------------------
    # 5. Forecast
    # -----------------------------
    forecast_df = forecast_daily_load(res_model, ci_model, df)

    forecast_df = forecast_daily_load(res_model, ci_model, df)

    # Convert Wh → MWh and round
    for col in ['residential_load', 'ci_load']:
        if col in forecast_df.columns:
            forecast_df[col] = (forecast_df[col] / 1_000_000).round(2)
        
    # -----------------------------
    # 6. Save output
    # -----------------------------
    output_filename = f"{selected_location}_user_forecast.xlsx"
    output_path = OUTPUT_DIR / output_filename

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        forecast_df.to_excel(writer, index=False, sheet_name="Forecast")

    return output_path


import sys

if __name__ == "__main__":
    input_file = sys.argv[1]
    location = sys.argv[2]

    output = run_user_forecast(input_file, location)
    print(f"Saved to: {output}")
