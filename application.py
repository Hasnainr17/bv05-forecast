from flask import Flask, render_template, request, session, send_from_directory
from markupsafe import Markup
import uuid
from datetime import datetime, timedelta
import os
import pandas as pd
from pathlib import Path
from werkzeug.utils import secure_filename

from user_load_forecast import run_user_forecast
from load_forecast_json_and_csv_upgraded import (
    train_models_from_historical_csv,
    forecast_daily_load,
    load_forecast_weather_from_csv,
    DATA_DIR,
    LOCATIONS,
    perform_validation
)
from interactive_validation_module import get_validation_section

app = Flask(__name__)
app.secret_key = 'BV_05'

BASE_DIR = Path(__file__).resolve().parent
CITIES = ["Toronto", "Ottawa", "Hamilton", "London", "Mississauga", "Brampton"]


def build_weather(city, load_data):
    if load_data:
        current = load_data[0]
        return {
            "city": city,
            "date": current.get("date", datetime.now().strftime("%Y-%m-%d")),
            "temperature": current.get("temperature", "N/A"),
            "wind_speed": current.get("wind_speed", "N/A"),
        }

    return {
        "city": city,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "temperature": "N/A",
        "wind_speed": "N/A"
    }


def build_load_forecast(city):
    try:
        hist_path = DATA_DIR / LOCATIONS.get(city, LOCATIONS.get("Toronto"))
        weather_fc_path = DATA_DIR / f"{city}_forecast_daily_weather.csv"

        if not hist_path.exists() or not weather_fc_path.exists():
            return []

        forecast_df = load_forecast_weather_from_csv(weather_fc_path)
        res_model, ci_model = train_models_from_historical_csv(hist_path)
        out_df = forecast_daily_load(res_model, ci_model, forecast_df)

        # normalize for template safety
        out_df = out_df.rename(columns={
            "Date": "date",
            "temperature_2m_mean (°C)": "temperature",
            "wind_speed_10m_mean (km/h)": "wind_speed",
            "forecast_residential_load": "forecast_residential_load",
            "forecast_ci_load": "forecast_ci_load"
        })

        return out_df.to_dict(orient='records')
    except Exception as e:
        print(f"Forecast error for {city}: {e}")
        return []


def get_forecast_summary_cards(load_data):
    if load_data and len(load_data) >= 1:
        today_day = load_data[0]
    else:
        today_day = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "temperature": "N/A",
            "wind_speed": "N/A",
            "forecast_residential_load": 0,
            "forecast_ci_load": 0
        }

    if load_data and len(load_data) >= 2:
        next_day = load_data[1]
    else:
        next_day = {
            "date": "N/A",
            "temperature": "N/A",
            "wind_speed": "N/A",
            "forecast_residential_load": 0,
            "forecast_ci_load": 0
        }

    latest_load = today_day
    return today_day, next_day, latest_load


def normalize_forecast_output(df):
    df = df.copy()

    rename_map = {
        "Date": "date",
        "date": "date",
        "temperature_2m_mean (°C)": "temperature",
        "Temperature": "temperature",
        "temperature": "temperature",
        "wind_speed_10m_mean (km/h)": "wind_speed",
        "Wind Speed": "wind_speed",
        "wind_speed": "wind_speed",
        "forecast_residential_load": "forecast_residential_load",
        "Residential Load": "forecast_residential_load",
        "forecast_ci_load": "forecast_ci_load",
        "C&I Load": "forecast_ci_load",
        "CI Load": "forecast_ci_load"
    }

    df = df.rename(columns=rename_map)

    for col in ["forecast_residential_load", "forecast_ci_load"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
        
    expected_cols = ["date", "temperature", "wind_speed", "forecast_residential_load", "forecast_ci_load"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""

    return df[expected_cols]


@app.route("/", methods=["GET", "POST"])
def home():
    city = request.args.get("city", "Toronto")
    if city not in CITIES:
        city = "Toronto"

    load_data = build_load_forecast(city)
    weather_data = build_weather(city, load_data)
    today_day, next_day, latest_load = get_forecast_summary_cards(load_data)

    input_dates = [(datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]
    default_dates = [(datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]

    user_output = None
    upload_output = None
    input_errors = {}

    manual_message = None
    upload_message = None
    validation_message = None

    # default all selectors to Toronto on fresh open
    manual_selected_city = "Toronto"
    upload_selected_city = "Toronto"
    validation_selected_city = "Toronto"

    user_submitted_data = session.get(
        'user_data',
        [{"date": d, "temp": "", "wind": ""} for d in default_dates]
    )

    if request.method == "POST":
        form_type = request.form.get("form_type")
        target_filename = None

        try:
            if form_type == "manual_input":
                manual_selected_city = request.form.get("user_city", "Toronto")
                if manual_selected_city not in CITIES:
                    manual_selected_city = "Toronto"

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
                    except (ValueError, TypeError):
                        input_errors[f"date_{i}"] = True
                        date_list.append(d if d else "")

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

                if input_errors:
                    manual_message = f"Error for {manual_selected_city}: Please correct the highlighted input fields and try again."
                    raise ValueError("manual_input_error")

                unique_id = uuid.uuid4().hex
                target_filename = str(BASE_DIR / f"user_input_{unique_id}.xlsx")

                df_input = pd.DataFrame({
                    "Date": date_list,
                    "temperature_2m_mean (°C)": temp_list,
                    "wind_speed_10m_mean (km/h)": wind_list
                })

                df_input.to_excel(target_filename, index=False)

                output_excel_path = run_user_forecast(target_filename, manual_selected_city)
                df_out = pd.read_excel(output_excel_path)
                df_out = normalize_forecast_output(df_out)

                if df_out.empty:
                    manual_message = f"Error for {manual_selected_city}: The forecast completed but returned no rows."
                    raise ValueError("manual_input_error")

                user_output = df_out.to_dict(orient='records')

                download_dir = BASE_DIR / "Forecasted Output"
                download_dir.mkdir(parents=True, exist_ok=True)
                download_path = download_dir / f"{manual_selected_city}_user_forecast.xlsx"
                df_out.to_excel(download_path, index=False)

                manual_message = f"Success! Forecast generated for {manual_selected_city}."

            elif form_type == "file_upload":
                upload_selected_city = request.form.get("user_city", "Toronto")
                if upload_selected_city not in CITIES:
                    upload_selected_city = "Toronto"

                uploaded_file = request.files.get("upload_file")

                if not uploaded_file or uploaded_file.filename == '':
                    upload_message = f"Error for {upload_selected_city}: No file was selected for upload."
                    raise ValueError("upload_error")

                safe_name = secure_filename(uploaded_file.filename)
                ext = safe_name.rsplit('.', 1)[1].lower() if '.' in safe_name else 'csv'

                if ext not in ['csv', 'xlsx', 'xls']:
                    upload_message = f"Error for {upload_selected_city}: Only CSV or Excel files are allowed."
                    raise ValueError("upload_error")

                unique_id = uuid.uuid4().hex
                target_filename = str(BASE_DIR / f"uploaded_{unique_id}.{ext}")
                uploaded_file.save(target_filename)

                if ext == 'csv':
                    df_check = pd.read_csv(target_filename)
                else:
                    df_check = pd.read_excel(target_filename)

                if df_check.empty:
                    upload_message = f"Error for {upload_selected_city}: The uploaded file contains headers but no data."
                    raise ValueError("upload_error")

                actual_cols = [str(col).strip() for col in df_check.columns]
                expected_cols = ["Date", "temperature_2m_mean (°C)", "wind_speed_10m_mean (km/h)"]
                missing_cols = [col for col in expected_cols if col not in actual_cols]

                if missing_cols:
                    upload_message = f"Error for {upload_selected_city}: Missing columns: {', '.join(missing_cols)}"
                    raise ValueError("upload_error")

                output_excel_path = run_user_forecast(target_filename, upload_selected_city)
                df_out = pd.read_excel(output_excel_path)
                df_out = normalize_forecast_output(df_out)

                if df_out.empty:
                    upload_message = f"Error for {upload_selected_city}: The uploaded forecast returned no rows."
                    raise ValueError("upload_error")

                upload_output = df_out.to_dict(orient='records')

                download_dir = BASE_DIR / "Forecasted Output"
                download_dir.mkdir(parents=True, exist_ok=True)
                download_path = download_dir / f"{upload_selected_city}_user_forecast.xlsx"
                df_out.to_excel(download_path, index=False)

                upload_message = f"Success! Forecast generated for {upload_selected_city}."

        except Exception as e:
            if str(e) not in ["manual_input_error", "upload_error"]:
                if form_type == "manual_input":
                    manual_message = f"Error for {manual_selected_city}: {e}"
                elif form_type == "file_upload":
                    upload_message = f"Error for {upload_selected_city}: {e}"

        finally:
            if target_filename and os.path.exists(target_filename):
                os.remove(target_filename)

    min_date_str = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    max_date_str = (datetime.today() + timedelta(days=365)).strftime("%Y-%m-%d")

    return render_template(
        "index.html",
        selected_city=city,
        cities=CITIES,
        weather_data=weather_data,
        load_data=load_data,
        today_day=today_day,
        latest_load=latest_load,
        next_day=next_day,
        user_output=user_output,
        upload_output=upload_output,
        user_submitted_data=user_submitted_data,
        input_dates=input_dates,
        input_errors=input_errors,
        min_date=min_date_str,
        max_date=max_date_str,
        validation_html=None,
        show_validation=False,
        manual_message=manual_message,
        upload_message=upload_message,
        validation_message=validation_message,
        manual_selected_city=manual_selected_city,
        upload_selected_city=upload_selected_city,
        validation_selected_city=validation_selected_city
    )


@app.route("/run_validation", methods=["POST"])
def run_validation():
    city = request.args.get("city", "Toronto")
    if city not in CITIES:
        city = "Toronto"

    validation_selected_city = request.form.get("validation_city", "Toronto")
    if validation_selected_city not in CITIES:
        validation_selected_city = "Toronto"

    start_date = request.form.get("validation_start_date")
    end_date = request.form.get("validation_end_date")

    validation_html = None
    validation_message = None

    try:
        if validation_selected_city not in LOCATIONS:
            raise ValueError(f"City '{validation_selected_city}' not supported.")

        hist_path = DATA_DIR / LOCATIONS[validation_selected_city]
        res_model, ci_model = train_models_from_historical_csv(hist_path)

        perform_validation(
            res_model=res_model,
            ci_model=ci_model,
            output_csv="Interactive_model_validation.xlsx",
            hist_path=hist_path,
            city=validation_selected_city,
            start_date=start_date,
            end_date=end_date,
        )

        validation_html = get_validation_section(validation_selected_city, start_date, end_date)
        validation_message = f"Validation processing complete for {validation_selected_city}."

    except Exception as e:
        validation_message = f"Validation could not be completed for {validation_selected_city}."
        validation_html = Markup(f'''
            <div class="alert alert-warning">
                <strong>Validation Error for {validation_selected_city}:</strong> {str(e)}<br>
                Please try again or check the date range.
            </div>
        ''')

    load_data = build_load_forecast(city)
    weather_data = build_weather(city, load_data)
    today_day, next_day, latest_load = get_forecast_summary_cards(load_data)

    default_dates = [(datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]
    user_submitted_data = session.get(
        'user_data',
        [{"date": d, "temp": "", "wind": ""} for d in default_dates]
    )

    return render_template(
        "index.html",
        selected_city=city,
        cities=CITIES,
        weather_data=weather_data,
        load_data=load_data,
        today_day=today_day,
        latest_load=latest_load,
        next_day=next_day,
        validation_html=validation_html,
        show_validation=True,
        user_output=None,
        upload_output=None,
        user_submitted_data=user_submitted_data,
        input_dates=default_dates,
        input_errors={},
        min_date=(datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d"),
        max_date=(datetime.today() + timedelta(days=365)).strftime("%Y-%m-%d"),
        manual_message=None,
        upload_message=None,
        validation_message=validation_message,
        manual_selected_city="Toronto",
        upload_selected_city="Toronto",
        validation_selected_city=validation_selected_city
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
