from flask import Flask, render_template, jsonify, request

from weather1 import fetch_and_process_forecast
from load_forecast_json_and_csv import run_load_forecast_pipeline

app = Flask(__name__)

ALLOWED_CITIES = [
    "Toronto",
    "Ottawa",
    "Hamilton",
    "London",
    "Mississauga",
    "Brampton"
]


def get_weather_records(city: str):
    weather_df = fetch_and_process_forecast(city=city)
    if weather_df is not None and len(weather_df) > 0:
        return weather_df, weather_df.to_dict(orient="records")
    return None, None


def get_load_records(city: str):
    load_df, csv_path, json_path, metrics = run_load_forecast_pipeline(city=city)
    if load_df is not None and len(load_df) > 0:
        return load_df
    return None


@app.route("/")
def home():
    city = request.args.get("city", "Toronto")

    if city not in ALLOWED_CITIES:
        city = "Toronto"

    weather_data = None
    load_data = []
    latest_load = None
    error_message = None

    try:
        # Weather for selected city
        weather_df, weather_records = get_weather_records(city)
        if weather_records:
            weather_data = weather_records

        # Load forecast for selected city
        load_df = get_load_records(city)
        if load_df is not None and len(load_df) > 0:
            load_data = load_df.to_dict(orient="records")
            latest_load = load_data[0]

    except Exception as e:
        error_message = str(e)

    return render_template(
        "index.html",
        selected_city=city,
        cities=ALLOWED_CITIES,
        weather_data=weather_data,
        load_data=load_data,
        latest_load=latest_load,
        error_message=error_message
    )


@app.route("/api/weather")
def weather():
    try:
        city = request.args.get("city", "Toronto")

        if city not in ALLOWED_CITIES:
            return jsonify({"error": "Invalid city"}), 400

        df, records = get_weather_records(city)
        if df is not None and len(df) > 0:
            return jsonify(df.to_dict(orient="records"))

        return jsonify({"error": "No weather data available"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/load")
def load():
    try:
        city = request.args.get("city", "Toronto")

        if city not in ALLOWED_CITIES:
            return jsonify({"error": "Invalid city"}), 400

        load_df = get_load_records(city)
        if load_df is not None and len(load_df) > 0:
            return jsonify(load_df.to_dict(orient="records"))

        return jsonify({"error": "No load forecast data"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)