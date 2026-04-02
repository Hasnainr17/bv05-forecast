from flask import Flask, render_template, request
from datetime import datetime, timedelta
from load_forecast_json_and_csv import run_load_forecast_pipeline

app = Flask(__name__)

CITIES = ["Toronto", "Ottawa", "Hamilton", "London", "Mississauga", "Brampton"]


def build_weather(city):
    weather_map = {
        "Toronto": {"temp": 10, "feels": 6.8, "humidity": 60, "wind": 12, "condition": "Partly cloudy"},
        "Ottawa": {"temp": 8, "feels": 5.9, "humidity": 57, "wind": 14, "condition": "Cloudy"},
        "Hamilton": {"temp": 9, "feels": 6.4, "humidity": 58, "wind": 11, "condition": "Mostly cloudy"},
        "London": {"temp": 11, "feels": 7.3, "humidity": 55, "wind": 10, "condition": "Partly cloudy"},
        "Mississauga": {"temp": 10, "feels": 6.7, "humidity": 59, "wind": 12, "condition": "Partly cloudy"},
        "Brampton": {"temp": 9, "feels": 6.1, "humidity": 61, "wind": 13, "condition": "Cloudy"},
    }
    w = weather_map.get(city, weather_map["Toronto"])
    return {
        "city": city,
        "date": "2026-03-30",
        "temperature": w["temp"],
        "feels_like": w["feels"],
        "humidity": w["humidity"],
        "wind_speed": w["wind"],
        "condition": w["condition"],
    }


def build_load_forecast(city):
    # Call your actual regression pipeline
    try:
        out_df, csv_path, json_path, metrics = run_load_forecast_pipeline(city=city)
        
        # Student D's HTML expects a list of dictionaries, so we convert your Pandas DataFrame
        return out_df.to_dict(orient='records')
        
    except Exception as e:
        print(f"Error running real forecast for {city}: {e}")
        # If your script fails (e.g., missing data), return an empty list so the site doesn't crash
        return []


def build_user_output(city):
    start = datetime.today()
    rows = []
    for i in range(5):
        rows.append({
            "date": (start + timedelta(days=i)).strftime("%Y-%m-%d"),
            "temperature": 9 + i,
            "wind_speed": 10 + (i % 3),
            "forecast_residential_load": round(13.75 + (i * 0.09), 2),
            "forecast_ci_load": round(4.62 + (i * 0.05), 2),
            "city": city,
        })
    return rows


@app.route("/", methods=["GET", "POST"])
def home():
    city = request.args.get("city", "Toronto")
    if city not in CITIES:
        city = "Toronto"

    weather_data = build_weather(city)
# 1. Get the fake weather (we will overwrite the important parts)
    weather_data = build_weather(city)
    
    # 2. Get the REAL data from your OLS regression pipeline
    load_data = build_load_forecast(city)
    
    # Safety Check: Did the engine actually return data?
    if load_data:
        latest_load = load_data[0]
        next_day = load_data[1]
        
        # 3. OVERRIDE the fake weather with the real CSV data for today
        weather_data["date"] = latest_load["date"]
        weather_data["temperature"] = latest_load["temperature"]
        weather_data["wind_speed"] = latest_load["wind_speed"]
        
    else:
        # Fallback empty state so the server doesn't crash
        latest_load = {"temperature": "N/A", "forecast_residential_load": "Error", "forecast_ci_load": "Error"}
        next_day = {"temperature": "N/A", "forecast_residential_load": "Error", "forecast_ci_load": "Error"}
        weather_data["date"] = "Error"
    latest_load = load_data[0]
    next_day = load_data[1]

    user_output = None
    user_message = None

    if request.method == "POST":
        selected_city = request.form.get("user_city", city)
        if selected_city not in CITIES:
            selected_city = city

        user_output = build_user_output(selected_city)
        user_message = f"Sample forecast output generated for {selected_city}."

    return render_template(
        "index.html",
        selected_city=city,
        cities=CITIES,
        weather_data=weather_data,
        load_data=load_data,
        latest_load=latest_load,
        next_day=next_day,
        user_output=user_output,
        user_message=user_message,
    )


if __name__ == "__main__":
    app.run(debug=True)