from flask import Flask, render_template, jsonify, request, send_file
import os
import pandas as pd
from werkzeug.utils import secure_filename

from weather1 import fetch_and_process_forecast
from load_forecast_json_and_csv import run_load_forecast_pipeline
from user_module.user_load_forecast import run_user_forecast

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
TEMPLATE_FILE = os.path.join(BASE_DIR, "user_module", "user_input_format.xlsx")
DEFAULT_FORECAST_CSV = os.path.join(BASE_DIR, "forecast_daily_load.csv")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_CITIES = {
    "Toronto",
    "Ottawa",
    "Hamilton",
    "London",
    "Mississauga",
    "Brampton"
}

ALLOWED_EXTENSIONS = {".csv", ".xlsx"}


def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def get_weather_records():
    weather_df = fetch_and_process_forecast()
    if weather_df is not None and len(weather_df) > 0:
        return weather_df, weather_df.to_dict(orient="records")
    return None, None


def get_default_load_df():
    """
    First try the existing saved CSV so the dashboard loads quickly.
    If it is missing, fall back to running the pipeline.
    """
    if os.path.exists(DEFAULT_FORECAST_CSV):
        df = pd.read_csv(DEFAULT_FORECAST_CSV)
        if df is not None and len(df) > 0:
            return df

    load_df, csv_path, json_path, metrics = run_load_forecast_pipeline()
    if load_df is not None and len(load_df) > 0:
        return load_df

    return None


@app.route("/")
def home():
    weather_data = None
    load_data = []
    latest_load = None
    error_message = None

    try:
        # Weather
        weather_df, weather_records = get_weather_records()
        if weather_records:
            weather_data = weather_records

        # Default load forecast
        load_df = get_default_load_df()
        if load_df is not None and len(load_df) > 0:
            load_data = load_df.to_dict(orient="records")
            latest_load = load_data[0]

    except Exception as e:
        error_message = str(e)

    return render_template(
        "index.html",
        weather_data=weather_data,
        load_data=load_data,
        latest_load=latest_load,
        error_message=error_message
    )


@app.route("/api/weather")
def weather():
    try:
        df, records = get_weather_records()
        if df is not None and len(df) > 0:
            return jsonify(df.iloc[0].to_dict())
        return jsonify({"error": "No weather data available"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/load")
def load():
    try:
        load_df = get_default_load_df()
        if load_df is not None and len(load_df) > 0:
            return jsonify(load_df.to_dict(orient="records"))
        return jsonify({"error": "No load forecast data"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download-template")
def download_template():
    if not os.path.exists(TEMPLATE_FILE):
        return jsonify({"error": "Template file not found"}), 404
    return send_file(TEMPLATE_FILE, as_attachment=True)


@app.route("/api/user-load", methods=["POST"])
def user_load():
    try:
        city = request.form.get("city")
        file = request.files.get("input_file")

        if not city or city not in ALLOWED_CITIES:
            return jsonify({"error": "Please select a valid city."}), 400

        if file is None or file.filename.strip() == "":
            return jsonify({"error": "Please upload an input file."}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Only .csv and .xlsx files are allowed."}), 400

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        # James' function returns the generated output path
        output_path = run_user_forecast(save_path, city)
        output_path_str = str(output_path)

        if output_path_str.lower().endswith(".xlsx"):
            result_df = pd.read_excel(output_path_str)
        elif output_path_str.lower().endswith(".csv"):
            result_df = pd.read_csv(output_path_str)
        else:
            return jsonify({"error": f"Unsupported output file format: {output_path_str}"}), 500

        return jsonify({
            "message": f"User forecast completed for {city}.",
            "city": city,
            "data": result_df.to_dict(orient="records"),
            "output_file": os.path.basename(output_path_str)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
