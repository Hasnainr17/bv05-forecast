from flask import Flask, render_template, request
from datetime import datetime, timedelta

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
    base_values = {
        "Toronto": (14.32, 4.95),
        "Ottawa": (13.11, 4.51),
        "Hamilton": (12.48, 4.63),
        "London": (11.92, 4.22),
        "Mississauga": (13.88, 4.81),
        "Brampton": (13.44, 4.67),
    }

    res_base, ci_base = base_values.get(city, (14.32, 4.95))
    start = datetime(2026, 3, 30)

    rows = []
    for i in range(16):
        rows.append({
            "date": (start + timedelta(days=i)).strftime("%Y-%m-%d"),
            "forecast_residential_load": round((res_base + (i * 0.07) - ((i % 3) * 0.03)), 2),
            "forecast_ci_load": round((ci_base + (i * 0.04) - ((i % 4) * 0.02)), 2),
            "temperature": 8 + (i % 6),
            "wind_speed": 10 + (i % 4),
            "city": city,
        })
    return rows


def build_user_output(city):
    start = datetime(2026, 3, 30)
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
    load_data = build_load_forecast(city)
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