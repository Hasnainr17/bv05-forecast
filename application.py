from flask import Flask, render_template, request, session, send_from_directory
import uuid
from datetime import datetime, timedelta
import os
import pandas as pd
from pathlib import Path
from werkzeug.utils import secure_filename

# Import the custom forecasting modules
from user_load_forecast import run_user_forecast
from load_forecast_json_and_csv_upgraded import (
    train_models_from_historical_csv,
    forecast_daily_load,
    load_forecast_weather_from_csv,
    DATA_DIR,
    LOCATIONS,
    perform_validation   # <-- Important: we need this
)

# Import validation display module
from interactive_validation_module import get_validation_section

app = Flask(__name__)
app.secret_key = 'BV_05'

BASE_DIR = Path(__file__).resolve().parent
CITIES = ["Toronto", "Ottawa", "Hamilton", "London", "Mississauga", "Brampton"]
VALIDATION_DIR = BASE_DIR / "Validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


def build_weather(city, load_data):
    if load_data:
        current = load_data[0]
        return {
            "city": city,
            "date": current.get("date", datetime.now().strftime("%Y-%m-%d")),
            "temperature": current.get("temperature", "N/A"),
            "wind_speed": current.get("wind_speed", "N/A"),
        }
    return {"city": city, "date": datetime.now().strftime("%Y-%m-%d"), "temperature": "N/A", "wind_speed": "N/A"}


def build_load_forecast(city):
    try:
        hist_path = DATA_DIR / LOCATIONS[city]
        weather_fc_path = DATA_DIR / f"{city}_forecast_daily_weather.csv"
        if not hist_path.exists() or not weather_fc_path.exists():
            return []
        forecast_df = load_forecast_weather_from_csv(weather_fc_path)
        res_model, ci_model = train_models_from_historical_csv(hist_path)
        out_df = forecast_daily_load(res_model, ci_model, forecast_df)
        return out_df.to_dict(orient='records')
    except Exception as e:
        print(f"Forecast error for {city}: {e}")
        return []


@app.route("/", methods=["GET", "POST"])
def home():
    # ... (keep your existing home() logic for the 16-day forecast and user input - I kept it short here for clarity)
    # Paste your full original home() function here if you want, or use the previous version I gave you.
    # For now, the key part is the validation route below.

    # Default render (you can keep your existing return render_template here)
    return render_template("index.html", ... )  # your existing variables


@app.route("/run_validation", methods=["POST"])
def run_validation():
    city = request.form.get("validation_city", "Toronto")
    start_date = request.form.get("validation_start_date")
    end_date = request.form.get("validation_end_date")

    if not all([city, start_date, end_date]):
        validation_html = Markup('<div class="alert alert-danger">All fields are required.</div>')
    else:
        try:
            # === AUTOMATIC VALIDATION RUN ===
            if city not in LOCATIONS:
                raise ValueError(f"City {city} not supported.")

            hist_path = DATA_DIR / LOCATIONS[city]
            res_model, ci_model = train_models_from_historical_csv(hist_path)

            # Run validation and save to the file
            metrics = perform_validation(
                res_model=res_model,
                ci_model=ci_model,
                output_csv="Interactive_model_validation.xlsx",   # saves to Validation/ folder
                hist_path=hist_path,
                city=city,
                start_date=start_date,
                end_date=end_date,
            )

            # Now display the results
            validation_html = get_validation_section(city, start_date, end_date)

        except Exception as e:
            validation_html = Markup(f'<div class="alert alert-warning">Validation failed: {str(e)}</div>')

    # Re-render the page with validation results
    load_data = build_load_forecast(city)
    weather_data = build_weather(city, load_data)

    default_dates = [(datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]
    user_submitted_data = session.get('user_data', [{"date": d, "temp": "", "wind": ""} for d in default_dates])

    return render_template(
        "index.html",
        selected_city=city,
        cities=CITIES,
        weather_data=weather_data,
        load_data=load_data,
        validation_html=validation_html,
        show_validation=True,
        user_submitted_data=user_submitted_data,
        # add your other variables as needed (latest_load, next_day, etc.)
        min_date=(datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d"),
        max_date=(datetime.today() + timedelta(days=365)).strftime("%Y-%m-%d")
    )


# Keep your existing download routes
@app.route('/download/<filename>')
def download_file(filename):
    directory = BASE_DIR / "Forecasted Output"
    return send_from_directory(directory, filename, as_attachment=True)


@app.route('/download_template')
def download_template():
    try:
        return send_from_directory(BASE_DIR, "user_input_format.xlsx", as_attachment=True)
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    app.run(debug=True)
