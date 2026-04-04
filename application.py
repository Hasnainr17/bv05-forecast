from flask import Flask, render_template, request, session, send_from_directory
import uuid
from datetime import datetime, timedelta
import os
import pandas as pd
from datetime import datetime, timedelta
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

# Initialize the flask app
app = Flask(__name__)

# Important for ensuring each user has thier own secure data
# and no data leaks into other users 
app.secret_key = 'BV_05'

BASE_DIR = Path(__file__).resolve().parent

CITIES = ["Toronto", "Ottawa", "Hamilton", "London", "Mississauga", "Brampton"]


def build_weather(city, load_data):

    # Use the actual data from the module (Coming from "Forecasted Output" Folder)
    if load_data:
        current = load_data[0]
        # print(f"Debug: Available keys are: {load_data[0].keys()}")

        # Get the actual weather data
        raw_temp = current.get("temperature", "N/A")
        raw_wind = current.get("wind_speed", "N/A")

        return {
            "city": city,
            "date": current.get("date", datetime.now().strftime("%Y-%m-%d")),
            "temperature": raw_temp,
            "wind_speed": raw_wind,
            # May need to implement this if critical to website
            # "humidity": current.get("humidity", "N/A"),
            "condition": "Data-Driven",
            # May need to implement this if critical to website
            # "feels_like": round(current.get("temperature", 0) * 0.9, 1)
            
        }
    
    # Default values if error in obtaining actual values
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

# Handles both the loading of the page (GET) and submitting forms (POST)
@app.route("/", methods=["GET", "POST"])
def home():

    # print(f"\nDebug: User request: {request.method}")

    # Grab the chosen city from the URL. Default is Toronto
    city = request.args.get("city", "Toronto")
    if city not in CITIES:
        city = "Toronto"

    # Define the variables needed by the template
    input_dates = [(datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]
    user_output = None
    user_message = None
    input_errors = {}

    # Fetch the load data first
    load_data = build_load_forecast(city)
    
    # Pass the data into build_weather
    weather_data = build_weather(city, load_data)
    
    # Handle the actual data, to display next day data and 16-days data
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

    # Look inside the user's session (cookie). 
    # If they have old 5-day inputs, load them.
    # If not, give them blank boxes with today's dates.
    # Ensures that the user also can save time when necessary when inputting data
    default_dates = [(datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]
    user_submitted_data = session.get('user_data', [{"date": d, "temp": "", "wind": ""} for d in default_dates])

    # This is for the user pressing a button on the website
    # Triggered when the user clicks a button
    if request.method == "POST":
        selected_city = request.form.get("user_city", city)

        # Determines whether the user is uploading their own data file
        # Or they manually entering data into website
        form_type = request.form.get("form_type")

        target_filename = None

        # print(f"Debug: Form submitted. City: {selected_city} | Form Type: {form_type}")

        try:
            # Option 1: Manual input of data (5 days)
            if form_type == "manual_input":
                
                # print("Debug: Entering Manual Input Path...")
                
                temp_list, wind_list, date_list, new_data = [], [], [], []

                # Calculate Date Boundaries (1 year past/future)
                today = datetime.today().date()
                one_year_ago = today - timedelta(days=365)
                one_year_future = today + timedelta(days=365)

                # Loop 5 times to extract the 5 rows of data from HTML
                for i in range(5):
                    d = request.form.get(f"date_{i}")
                    t = request.form.get(f"temp_{i}")
                    w = request.form.get(f"wind_{i}")

                    new_data.append({"date": d, "temp": t, "wind": w})

                    # Check 1: Validate Date (Must be within 1 year)
                    try:
                        d_obj = datetime.strptime(d, "%Y-%m-%d").date()
                        if d_obj < one_year_ago or d_obj > one_year_future:
                            input_errors[f"date_{i}"] = True
                        date_list.append(d)
                    except ValueError:
                        input_errors[f"date_{i}"] = True
                        date_list.append(d)

                    # Check 2: Validate Temperature (Between -25C and 35C)
                    try:
                        t_float = float(t)
                        if t_float < -25.0 or t_float > 35.0:
                            input_errors[f"temp_{i}"] = True
                        temp_list.append(t_float)
                    except (ValueError, TypeError):
                        input_errors[f"temp_{i}"] = True
                        temp_list.append(0.0)

                    # Check 3: Validate Wind Speed (Between 0 and 35 km/h)
                    try:
                        w_float = float(w)
                        if w_float < 0.0 or w_float > 35.0:
                            input_errors[f"wind_{i}"] = True
                        wind_list.append(w_float)
                    except (ValueError, TypeError):
                        input_errors[f"wind_{i}"] = True
                        wind_list.append(0.0)
                        
                # Update the browser's memory so the numbers stay in the boxes
                # Helps save time for the user if there are errors,
                # then they just need to change the incorrect inputs
                session['user_data'] = new_data
                user_submitted_data = new_data

                # Create a unique ID and absolute path
                # so users don't overwrite each other data
                # if multiple users using website
                unique_id = uuid.uuid4().hex
                target_filename = str(BASE_DIR / f"user_input_{unique_id}.xlsx") 
                
                df_input = pd.DataFrame({
                    "Date": date_list, 
                    "temperature_2m_mean (°C)": temp_list,
                    "wind_speed_10m_mean (km/h)": wind_list
                })

                # print(f"Debug: Created DataFrame with shape: {df_input.shape}")

                df_input.to_excel(target_filename, index=False)
                # print(f"Debug: Saved manual input to: {target_filename}")


            # Option 2: User uploads their own file
            elif form_type == "file_upload":

                # print("Debug: Entering File Upload Path...")

                # Grab the file from the request
                uploaded_file = request.files.get("upload_file")
                # print(f"Debug: Received file: {uploaded_file.filename}")
                
                # Check to see did they actually attach a file?
                if not uploaded_file or uploaded_file.filename == '':
                    raise ValueError("No file was selected for upload.")

                # Assign a uinque ID so users don't overwrite each other
                # Ensure CSV/EXCEL
                safe_name = secure_filename(uploaded_file.filename)
                ext = safe_name.rsplit('.', 1)[1].lower() if '.' in safe_name else 'csv'
                
                # Ensure CSV/EXCEL file that is uploaded
                if ext not in ['csv', 'xlsx', 'xls']:
                    raise ValueError("Only CSV or Excel files are allowed.")

                unique_id = uuid.uuid4().hex
                target_filename = str(BASE_DIR / f"uploaded_{unique_id}.{ext}")
                
                # Save the uploaded file temporarily to the server
                uploaded_file.save(target_filename)

                # Any erros in the input data is caught by the script
                # And outputted to the website automatically

                # Validation check for the user upload, to ensure no modification of template
                if ext == 'csv':
                    df_check = pd.read_csv(target_filename)
                else:
                    df_check = pd.read_excel(target_filename)

                # Check 1: Is the file completely empty? Meaning no data and only headers
                if df_check.empty:
                    raise ValueError("The uploaded file contains headers but no data. Please add your daily values starting from row 2.")

                # Check 2: Do the columns match the template exactly?
                # Using .strip()if they accidentally typed a trailing space
                actual_cols = [str(col).strip() for col in df_check.columns]
                expected_cols = ["Date", "temperature_2m_mean (°C)", "wind_speed_10m_mean (km/h)"]
                
                # Find exactly which columns the user forgot or misspelled
                missing_cols = [col for col in expected_cols if col not in actual_cols]
                
                if missing_cols:
                    raise ValueError(f"Column Error! Your file is missing or misspelled these required columns: {', '.join(missing_cols)}. Please download the template to ensure the exact format.")
                
            # Both options create the user forecasting results
            # Downloadable file
            output_excel_path = run_user_forecast(target_filename, selected_city)
            # print(f"Debug: Module returned output path: {output_excel_path}")
            
            df_out = pd.read_excel(output_excel_path)
            # print(f"Debug: Successfully read forecast output. Shape: {df_out.shape}")

            user_output = df_out.to_dict(orient='records')
            
            # Save the file for the user to download
            download_dir = BASE_DIR / "Forecasted Output"
            download_dir.mkdir(parents=True, exist_ok=True)
            download_path = download_dir / f"{selected_city}_user_forecast.xlsx"
            df_out.to_excel(download_path, index=False)
            #  print(f"Debug: Final output saved for download at: {download_path}")

            user_message = f"Success! Forecast generated for {selected_city}."

        except Exception as e:
            user_message = f"Error: {e}"

        finally:
            # Clean up the temporary files before accumulating.
            if target_filename and os.path.exists(target_filename):
                os.remove(target_filename)

    if user_output:
        print(f"Debug: user_output has {len(user_output)} rows.")
    else:
        print(f"Debug: user_output has 0 rows.")

    # Generate the string versions of the date boundaries for the HTML
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
        max_date=max_date_str
    )

# Download option for user
@app.route('/download/<filename>')
def download_file(filename):
    # print(f"\nDebug: User requested download: {filename}")

    # Ensure the directory exists
    directory = BASE_DIR / "Forecasted Output"
    
    # "as_attachment=True" option forces the browser to download the file 
    # instead of trying to open it in a tab
    # print(f"Debug: Fetching from directory: {directory}")
    return send_from_directory(directory, filename, as_attachment=True)

# Downloadable tempalte for the user to use
@app.route('/download_template')
def download_template():
    # Location is in BASE_DIR where user_input_format.xlsx is located
    try:
        return send_from_directory(BASE_DIR, "user_input_format.xlsx", as_attachment=True)
    except Exception as e:
        return f"Error downloading template: {e}"

if __name__ == "__main__":
    app.run(debug=True)