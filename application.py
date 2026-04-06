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
    LOCATIONS
)

# Import the new validation module
from interactive_validation_module import get_validation_section

# Initialize the flask app
app = Flask(__name__)
app.secret_key = 'BV_05'

BASE_DIR = Path(__file__).resolve().parent
CITIES = ["Toronto", "Ottawa", "Hamilton", "London", "Mississauga", "Brampton"]


def build_weather(city, load_data):
    if load_data:
        current = load_data[0]
        raw_temp = current.get("temperature", "N/A")
        raw_wind = current.get("wind_speed", "N/A")
        return {
            "city": city,
            "date": current.get("date", datetime.now().strftime("%Y-%m-%d")),
            "temperature": raw_temp,
            "wind_speed": raw_wind,
            "condition": "Data-Driven",
        }
   
    return {
        "city": city,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "temperature": "N/A",
        "condition": "No Data Available"
    }


def build_load_forecast(city):
    try:
        hist_path = DATA_DIR / LOCATIONS[city]
        weather_fc_path = DATA_DIR / f"{city}_forecast_daily_weather.csv"
       
        if not hist_path.exists() or not weather_fc_path.exists():
            print(f"Missing data files for {city}. Run the batch script first.")
            return []
       
        forecast_df = load_forecast_weather_from_csv(weather_fc_path)
        res_model, ci_model = train_models_from_historical_csv(hist_path)
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

    input_dates = [(datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]
    user_output = None
    user_message = None
    input_errors = {}

    # Fetch load forecast data
    load_data = build_load_forecast(city)
    weather_data = build_weather(city, load_data)

    if load_data and len(load_data) >= 2:
        latest_load = load_data[0]
        next_day = load_data[1]
        weather_data["date"] = latest_load["date"]
        weather_data["temperature"] = latest_load["temperature"]
        weather_data["wind_speed"] = latest_load["wind_speed"]
    else:
        latest_load = {"temperature": "N/A", "forecast_residential_load": 0, "forecast_ci_load": 0}
        next_day = {"temperature": "N/A", "forecast_residential_load": 0, "forecast_ci_load": 0}
        if "date" not in weather_data:
            weather_data["date"] = "No Data"

    default_dates = [(datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]
    user_submitted_data = session.get('user_data', [{"date": d, "temp": "", "wind": ""} for d in default_dates])

    # Handle POST requests (user forecast)
    if request.method == "POST":
        selected_city = request.form.get("user_city", city)
        form_type = request.form.get("form_type")
        target_filename = None

        try:
            if form_type == "manual_input":
                temp_list, wind_list, date_list, new_data = [], [], [], []
                today = datetime.today().date()
                one_year_ago = today - timedelta(days=365)
                one_year_future = today + timedelta(days=365)

                for i in range(5):
                    d = request.form.get(f"date_{i}")
                    t = request.form.get(f"temp_{i}")
                    w = request.form.get(f"wind_{i}")
                    new_data.append({"date": d, "temp": t, "wind": w})

                    try:
                        d_obj = datetime.strptime(d, "%Y-%m-%d").date()
                        if d_obj < one_year_ago or d_obj > one_year_future:
                            input_errors[f"date_{i}"] = True
                        date_list.append(d)
                    except ValueError:
                        input_errors[f"date_{i}"] = True
                        date_list.append(d)

                    try:
                        t_float = float(t)
                        if t_float < -25.0 or t_float > 35.0:
                            input_errors[f"temp_{i}"] = True
                        temp_list.append(t_float)
                    except (ValueError, TypeError):
                        input_errors[f"temp_{i}"] = True
                        temp_list.append(0.0)

                    try:
                        w_float = float(w)
                        if w_float < 0.0 or w_float > 35.0:
                            input_errors[f"wind_{i}"] = True
                        wind_list.append(w_float)
                    except (ValueError, TypeError):
                        input_errors[f"wind_{i}"] = True
                        wind_list.append(0.0)

                session['user_data'] = new_data
                user_submitted_data = new_data

                unique_id = uuid.uuid4().hex
                target_filename = str(BASE_DIR / f"user_input_{unique_id}.xlsx")

                df_input = pd.DataFrame({
                    "Date": date_list,
                    "temperature_2m_mean (°C)": temp_list,
                    "wind_speed_10m_mean (km/h)": wind_list
                })
                df_input.to_excel(target_filename, index=False)

            elif form_type == "file_upload":
                uploaded_file = request.files.get("upload_file")
                if not uploaded_file or uploaded_file.filename == '':
                    raise ValueError("No file was selected for upload.")

                safe_name = secure_filename(uploaded_file.filename)
                ext = safe_name.rsplit('.', 1)[1].lower() if '.' in safe_name else 'csv'

                if ext not in ['csv', 'xlsx', 'xls']:
                    raise ValueError("Only CSV or Excel files are allowed.")

                unique_id = uuid.uuid4().hex
                target_filename = str(BASE_DIR / f"uploaded_{unique_id}.{ext}")

                uploaded_file.save(target_filename)

                if ext == 'csv':
                    df_check = pd.read_csv(target_filename)
                else:
                    df_check = pd.read_excel(target_filename)

                if df_check.empty:
                    raise ValueError("The uploaded file contains headers but no data.")

                actual_cols = [str(col).strip() for col in df_check.columns]
                expected_cols = ["Date", "temperature_2m_mean (°C)", "wind_speed_10m_mean (km/h)"]
                missing_cols = [col for col in expected_cols if col not in actual_cols]

                if missing_cols:
                    raise ValueError(f"Column Error! Missing: {', '.join(missing_cols)}")

            output_excel_path = run_user_forecast(target_filename, selected_city)
            df_out = pd.read_excel(output_excel_path)
            user_output = df_out.to_dict(orient='records')

            download_dir = BASE_DIR / "Forecasted Output"
            download_dir.mkdir(parents=True, exist_ok=True)
            download_path = download_dir / f"{selected_city}_user_forecast.xlsx"
            df_out.to_excel(download_path, index=False)

            user_message = f"Success! Forecast generated for {selected_city}."

        except Exception as e:
            user_message = f"Error: {e}"
        finally:
            if target_filename and os.path.exists(target_filename):
                os.remove(target_filename)

    # Default values for validation section
    min_date_str = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    max_date_str = (datetime.today() + timedelta(days=365)).strftime("%Y-%m-%d")

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
        user_submitted_data=user_submitted_data,
        input_dates=input_dates,
        input_errors=input_errors,
        min_date=min_date_str,
        max_date=max_date_str,
        # Validation section
        validation_html=None,
        show_validation=False
    )


@app.route("/run_validation", methods=["POST"])
def run_validation():
    city = request.form.get("validation_city", "Toronto")
    start_date = request.form.get("validation_start_date")
    end_date = request.form.get("validation_end_date")

    validation_html = get_validation_section(city, start_date, end_date)

    load_data = build_load_forecast(city)
    weather_data = build_weather(city, load_data)

    if load_data and len(load_data) >= 2:
        latest_load = load_data[0]
        next_day = load_data[1]
    else:
        latest_load = {"temperature": "N/A", "forecast_residential_load": 0, "forecast_ci_load": 0}
        next_day = {"temperature": "N/A", "forecast_residential_load": 0, "forecast_ci_load": 0}

    return render_template(
        "index.html",
        selected_city=city,
        cities=CITIES,
        weather_data=weather_data,
        load_data=load_data,
        latest_load=latest_load,
        next_day=next_day,
        validation_html=validation_html,
        show_validation=True,
        user_output=None,
        user_message=None,
        user_submitted_data=session.get('user_data', []),
        input_dates=[(datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)],
        input_errors={},
        min_date=(datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d"),
        max_date=(datetime.today() + timedelta(days=365)).strftime("%Y-%m-%d")
    )


@app.route('/download/<filename>')
def download_file(filename):
    directory = BASE_DIR / "Forecasted Output"
    return send_from_directory(directory, filename, as_attachment=True)


@app.route('/download_template')
def download_template():
    try:
        return send_from_directory(BASE_DIR, "user_input_format.xlsx", as_attachment=True)
    except Exception as e:
        return f"Error downloading template: {e}"


if __name__ == "__main__":
    app.run(debug=True)
