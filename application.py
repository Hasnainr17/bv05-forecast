from flask import Flask, render_template, jsonify
from weather1 import fetch_and_process_forecast
from load_forecast_json_and_csv import run_load_forecast_pipeline

app = Flask(__name__)

@app.route("/")
def home():
    weather_data = None
    load_data = []
    latest_load = None
    error_message = None

    try:
        # Weather data
        weather_df = fetch_and_process_forecast()
        if weather_df is not None and len(weather_df) > 0:
            weather_data = weather_df.to_dict(orient="records")

        # Load forecast data
        load_df, csv_path, json_path, metrics = run_load_forecast_pipeline()
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
    df = fetch_and_process_forecast()
    if df is not None and len(df) > 0:
        return jsonify(df.iloc[0].to_dict())
    return jsonify({"error": "No data"}), 500

@app.route("/api/load")
def load():
    try:
        load_df, csv_path, json_path, metrics = run_load_forecast_pipeline()
        if load_df is not None and len(load_df) > 0:
            return jsonify(load_df.to_dict(orient="records"))
        return jsonify({"error": "No load forecast data"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
