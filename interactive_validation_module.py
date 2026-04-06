# interactive_validation_module.py
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from flask import Markup

def calculate_rmsp(actual, predicted):
    """Root Mean Square Percentage Error (RMSPE)"""
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    mask = (actual != 0) & (~np.isnan(actual)) & (~np.isnan(predicted))
    if len(actual[mask]) == 0:
        return 0.0
    errors = ((actual[mask] - predicted[mask]) / actual[mask]) ** 2
    return round(np.sqrt(np.mean(errors)) * 100, 4)


def get_validation_section(location: str, start_date: str, end_date: str):
    """Returns HTML for validation section"""
    
    if not all([location, start_date, end_date]):
        return Markup('<div class="alert alert-danger">Location, Start Date, and End Date are required.</div>')

    file_path = "Validation/Interactive_model_validation.xlsx"

    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        return Markup(f'<div class="alert alert-warning">Interactive_model_validation.xlsx not found.<br>Please run validation for {location} first.</div>')

    df.columns = [col.strip() for col in df.columns]

    if 'date' in df.columns:
        df['Date'] = pd.to_datetime(df['date'])

    # Filter by location
    if 'Location' in df.columns:
        df = df[df['Location'].astype(str).str.lower() == location.lower().strip()]

    # Filter by date range
    filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & 
                     (df['Date'] <= pd.to_datetime(end_date))].copy()

    if filtered_df.empty:
        return Markup(f'<div class="alert alert-warning">No data found for {location} between {start_date} and {end_date}.</div>')

    rmsp_res = calculate_rmsp(filtered_df['res_actual'], filtered_df['res_predicted'])
    rmsp_ci = calculate_rmsp(filtered_df['ci_actual'], filtered_df['ci_predicted'])

    # Residential Plot
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['res_actual'], name='Actual', line=dict(color='#1f77b4', width=3)))
    fig_res.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['res_predicted'], name='Predicted', line=dict(color='#ff7f0e', width=3, dash='dash')))
    fig_res.update_layout(title=f"Residential Load Validation - {location}", height=520, template="plotly_white")

    # C&I Plot
    fig_ci = go.Figure()
    fig_ci.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['ci_actual'], name='Actual', line=dict(color='#1f77b4', width=3)))
    fig_ci.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['ci_predicted'], name='Predicted', line=dict(color='#ff7f0e', width=3, dash='dash')))
    fig_ci.update_layout(title=f"C&I Load Validation - {location}", height=520, template="plotly_white")

    plot_res = fig_res.to_html(full_html=False)
    plot_ci = fig_ci.to_html(full_html=False)

    return Markup(f"""
    <div class="card">
        <h2>Interactive Model Validation</h2>
        <p class="section-text">
            Actual vs Predicted comparison for <strong>{location}</strong> from <strong>{start_date}</strong> to <strong>{end_date}</strong>
        </p>
        
        <div style="display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 280px; background: rgba(255,255,255,0.08); padding: 25px; border-radius: 20px; text-align: center;">
                <h4 style="margin:0 0 10px; color:#93c5fd;">Residential RMSPE</h4>
                <h2 style="margin:0; color:#60a5fa;">{rmsp_res}%</h2>
            </div>
            <div style="flex: 1; min-width: 280px; background: rgba(255,255,255,0.08); padding: 25px; border-radius: 20px; text-align: center;">
                <h4 style="margin:0 0 10px; color:#93c5fd;">C&amp;I RMSPE</h4>
                <h2 style="margin:0; color:#60a5fa;">{rmsp_ci}%</h2>
            </div>
        </div>

        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 300px;">{plot_res}</div>
            <div style="flex: 1; min-width: 300px;">{plot_ci}</div>
        </div>

        <h3 style="margin-top: 40px; text-align: center;">RMSPE Summary</h3>
        <table style="width:100%; margin-top: 15px;">
            <thead>
                <tr style="background: rgba(255,255,255,0.1);">
                    <th style="padding: 14px;">Model</th>
                    <th style="padding: 14px;">RMSPE (%)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="padding: 14px; text-align: center;">Residential</td>
                    <td style="padding: 14px; text-align: center; font-weight: 600;">{rmsp_res}%</td>
                </tr>
                <tr>
                    <td style="padding: 14px; text-align: center;">C&amp;I</td>
                    <td style="padding: 14px; text-align: center; font-weight: 600;">{rmsp_ci}%</td>
                </tr>
            </tbody>
        </table>
    </div>
    """)
