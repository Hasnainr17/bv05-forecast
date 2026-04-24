import pandas as pd
import plotly.graph_objects as go
import numpy as np
from markupsafe import Markup
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def calculate_rmsp(actual, predicted):
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    mask = (actual != 0) & (~np.isnan(actual)) & (~np.isnan(predicted))
    if len(actual[mask]) == 0:
        return 0.0
    errors = ((actual[mask] - predicted[mask]) / actual[mask]) ** 2
    return round(np.sqrt(np.mean(errors)) * 100, 4)


def get_validation_section(location: str, start_date: str, end_date: str):
    if not all([location, start_date, end_date]):
        return Markup('<div class="alert alert-danger">Location, Start Date, and End Date are required.</div>')

    file_path = BASE_DIR / "Validation" / "Interactive_model_validation.xlsx"

    try:
        df = pd.read_excel(
            file_path,
            sheet_name="Comparison",
            usecols=["date", "res_actual", "res_predicted", "ci_actual", "ci_predicted"]
        )
    except FileNotFoundError:
        return Markup(
            f'<div class="alert alert-warning">Interactive_model_validation.xlsx not found.<br>'
            f'Please run validation for {location} first.</div>'
        )
    except Exception as e:
        return Markup(f'<div class="alert alert-warning">Error reading validation file: {e}</div>')

    df.columns = [col.strip() for col in df.columns]

    if "date" not in df.columns:
        return Markup('<div class="alert alert-warning">The validation file is missing the "date" column.</div>')

    df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    filtered_df = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()

    if filtered_df.empty:
        return Markup(
            f'<div class="alert alert-warning">'
            f'No data found for {location} between {start_date} and {end_date}.<br>'
            f'Available data range: {df["Date"].min().date()} to {df["Date"].max().date()}'
            f'</div>'
        )

    rmsp_res = calculate_rmsp(filtered_df["res_actual"], filtered_df["res_predicted"])
    rmsp_ci = calculate_rmsp(filtered_df["ci_actual"], filtered_df["ci_predicted"])

    fig_res = go.Figure()
    fig_res.add_trace(
        go.Scatter(
            x=filtered_df["Date"],
            y=filtered_df["res_actual"],
            name="Actual Residential Load",
            mode="lines"
        )
    )
    fig_res.add_trace(
        go.Scatter(
            x=filtered_df["Date"],
            y=filtered_df["res_predicted"],
            name="Predicted Residential Load",
            mode="lines"
        )
    )
    fig_res.update_layout(
        title=f"Residential Load Validation - {location}",
        xaxis_title="Date",
        yaxis_title="Residential Load (MWh)",
        height=520,
        template="plotly_white"
    )
    fig_res.update_xaxes(tickformat="%Y-%m-%d")
    fig_res.update_yaxes(tickformat=",.2f")

    fig_ci = go.Figure()
    fig_ci.add_trace(
        go.Scatter(
            x=filtered_df["Date"],
            y=filtered_df["ci_actual"],
            name="Actual C&I Load",
            mode="lines"
        )
    )
    fig_ci.add_trace(
        go.Scatter(
            x=filtered_df["Date"],
            y=filtered_df["ci_predicted"],
            name="Predicted C&I Load",
            mode="lines"
        )
    )
    fig_ci.update_layout(
        title=f"C&I Load Validation - {location}",
        xaxis_title="Date",
        yaxis_title="C&I Load (MWh)",
        height=520,
        template="plotly_white"
    )
    fig_ci.update_xaxes(tickformat="%Y-%m-%d")
    fig_ci.update_yaxes(tickformat=",.2f")

    return Markup(f"""
    <div class="card">
        <h2>Interactive Model Validation</h2>
        <p>Actual vs Predicted comparison for <strong>{location}</strong> from <strong>{start_date}</strong> to <strong>{end_date}</strong></p>
        <p><strong>Residential RMSPE:</strong> {rmsp_res}%</p>
        <p><strong>C&amp;I RMSPE:</strong> {rmsp_ci}%</p>
        <div>{fig_res.to_html(full_html=False)}</div>
        <div style="margin-top: 24px;">{fig_ci.to_html(full_html=False)}</div>
    </div>
    """)
