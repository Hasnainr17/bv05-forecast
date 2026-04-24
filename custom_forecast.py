import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from load_forecast_json_and_csv_upgraded import (
    train_models_from_historical_csv,
    forecast_daily_load
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "Forecasted Output"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_custom_forecast(historical_file_path, forecast_file_path):
    historical_file_path = Path(historical_file_path)
    forecast_file_path = Path(forecast_file_path)

    # Read forecast input file
    if forecast_file_path.suffix.lower() == ".csv":
        forecast_df = pd.read_csv(forecast_file_path)
    else:
        forecast_df = pd.read_excel(forecast_file_path)

    # Train model using uploaded historical file
    res_model, ci_model = train_models_from_historical_csv(historical_file_path)

    # Forecast using uploaded forecast file
    output_df = forecast_daily_load(res_model, ci_model, forecast_df)

    # Save Excel output
    output_path = OUTPUT_DIR / "custom_forecast_output.xlsx"
    output_df.to_excel(output_path, index=False)

    # Create Residential plot
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(
        x=output_df["date"],
        y=output_df["forecast_residential_load"],
        mode="lines+markers",
        name="Residential Load"
    ))
    fig_res.update_layout(
        title="Custom Forecast - Residential Load",
        xaxis_title="Date",
        yaxis_title="Residential Load (MWh)",
        height=450,
        template="plotly_white"
    )

    # Create CI plot
    fig_ci = go.Figure()
    fig_ci.add_trace(go.Scatter(
        x=output_df["date"],
        y=output_df["forecast_ci_load"],
        mode="lines+markers",
        name="C&I Load"
    ))
    fig_ci.update_layout(
        title="Custom Forecast - C&I Load",
        xaxis_title="Date",
        yaxis_title="C&I Load (MWh)",
        height=450,
        template="plotly_white"
    )

    return output_path, output_df, fig_res.to_html(full_html=False), fig_ci.to_html(full_html=False)
