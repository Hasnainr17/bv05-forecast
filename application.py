from flask import Flask, render_template, request
import uuid
from datetime import datetime, timedelta
import os
import pandas as pd
from datetime import datetime, timedelta
from user_load_forecast import run_user_forecast
from load_forecast_json_and_csv_upgraded import (
    train_models_from_historical_csv, 
    forecast_daily_load, 
    load_forecast_weather_from_csv, 
    DATA_DIR, 
    LOCATIONS
)

app = Flask(__name__)

CITIES = ["Toronto", "Ottawa", "Hamilton", "London", "Mississauga", "Brampton"]


def build_weather(city, load_data):
    # Use the actual data from the modul (Coming from Forecasted Output Folder)
    if load_data:
        current = load_data[0]
        # print(f"Debug: Available keys are: {load_data[0].keys()}")
        return {
            "city": city,
            "date": current.get("date", datetime.now().strftime("%Y-%m-%d")),
            "temperature": current.get("temperature", "N/A"),
            "wind_speed": current.get("wind_speed", "N/A"),
            # May need to implement this if critical to website
            "humidity": current.get("humidity", "N/A"),
            "condition": "Data-Driven",
            "feels_like": round(current.get("temperature", 0) * 0.9, 1)
        }
    return {
        "city": city,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "temperature": "N/A",
        "condition": "No Data Available"
    }


def build_load_forecast(city):
    try:
        # Define paths for the specific city using the new script's logic
        hist_path = DATA_DIR / LOCATIONS[city]
        weather_fc_path = DATA_DIR / f"{city}_forecast_daily_weather.csv"
        
        # Check if the required data files exist
        if not hist_path.exists() or not weather_fc_path.exists():
            print(f"Missing data files for {city}. Run the batch script first.")
            return []

        # Load the pre-fetched weather data
        forecast_df = load_forecast_weather_from_csv(weather_fc_path)

        # Train the OLS model to forecast the load
        res_model, ci_model = train_models_from_historical_csv(hist_path)

        # Predict the 16-day load forecast
        out_df = forecast_daily_load(res_model, ci_model, forecast_df)
        
        return out_df.to_dict(orient='records')
        
    except Exception as e:
        print(f"Error running real forecast for {city}: {e}")
        return []

@app.route("/", methods=["GET", "POST"])
def home():
    city = request.args.get("city", "Toronto")
    if city not in CITIES:
        city = "Toronto"

    # Define the variables needed by the template
    input_dates = [(datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]
    user_output = None
    user_message = None

    # Fetch the load data first
    load_data = build_load_forecast(city)
    
    # Pass the data into build_weather
    weather_data = build_weather(city, load_data)
    
    # Handle the actual data
    if load_data and len(load_data) >= 2:
        latest_load = load_data[0]
        next_day = load_data[1]
        
        # Load the weather data into the CSV data
        weather_data["date"] = latest_load["date"]
        weather_data["temperature"] = latest_load["temperature"]
        weather_data["wind_speed"] = latest_load["wind_speed"]
    else:
        latest_load = {"temperature": "N/A", "forecast_residential_load": 0, "forecast_ci_load": 0}
        next_day = {"temperature": "N/A", "forecast_residential_load": 0, "forecast_ci_load": 0}
  
        if "date" not in weather_data:
            weather_data["date"] = "No Data"

    if request.method == "POST":
        selected_city = request.form.get("user_city", city)
        try:
            temp_list = []
            wind_list = []
            date_list = []

            # Grab the data from the website from user
            for i in range(5):
                d = request.form.get(f"date_{i}")
                t = request.form.get(f"temp_{i}")
                w = request.form.get(f"wind_{i}")

                date_list.append(d)
                temp_list.append(float(t))
                wind_list.append(float(w))

            unique_filename = f"user_input_formatted_from_website.csv"
            
            # Create the data frame
            df_input = pd.DataFrame({
                "Date": date_list, 
                "temperature_2m_mean (°C)": temp_list,
                "wind_speed_10m_mean (km/h)": wind_list
            })
            
            # Run load forecast with user data
            df_input.to_csv(unique_filename, index=False)
            output_excel_path = run_user_forecast(unique_filename, selected_city)
            
            df_out = pd.read_excel(output_excel_path)
            user_output = df_out.to_dict(orient='records')
            user_message = f"Success! 5-day forecast generated for {selected_city}."
            
            # Remove the user file
            if os.path.exists(unique_filename):
                os.remove(unique_filename)


        except Exception as e:
            user_message = f"Error: {e}"

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
        input_dates=input_dates,
    )


if __name__ == "__main__":
    app.run(debug=True)