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
    """Returns HTML string ready to be inserted into index.html"""
    
    if not all([location, start_date, end_date]):
        return Markup('<div class="alert alert-danger">Location, Start Date, and End Date are required.</div>')

    file_path = "Validation/Interactive_model_validation.xlsx"

    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        return Markup(f'<div class="alert alert-warning">Interactive_model_validation.xlsx not found.<br>Please run the validation script first for {location}.</div>')

    df.columns = [col.strip() for col in df.columns]

    if 'date' in df.columns:
        df['Date'] = pd.to_datetime(df['date'])

    if 'Location' in df.columns:
        df = df[df['Location'].astype(str).str.lower() == location.lower().strip()]

    filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & 
                     (df['Date'] <= pd.to_datetime(end_date))].copy()

    if filtered_df.empty:
        return Markup(f'<div class="alert alert-warning">No data found for {location} between {start_date} and {end_date}.</div>')

    rmsp_res = calculate_rmsp(filtered_df['res_actual'], filtered_df['res_predicted'])
    rmsp_ci = calculate_rmsp(filtered_df['ci_actual'], filtered_df['ci_predicted'])

    # Create plots
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['res_actual'], 
                                 name='Actual Residential', line=dict(color='#1f77b4', width=3)))
    fig_res.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['res_predicted'], 
                                 name='Predicted Residential', line=dict(color='#ff7f0e', width=3, dash='dash')))
    fig_res.update_layout(title=f"Residential Load Validation - {location}", height=520, template="plotly_white")

    fig_ci = go.Figure()
    fig_ci.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['ci_actual'], 
                                name='Actual C&I', line=dict(color='#1f77b4', width=3)))
    fig_ci.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['ci_predicted'], 
                                name='Predicted C&I', line=dict(color='#ff7f0e', width=3, dash='dash')))
    fig_ci.update_layout(title=f"C&I Load Validation - {location}", height=520, template="plotly_white")

    plot_res_html = fig_res.to_html(full_html=False)
    plot_ci_html = fig_ci.to_html(full_html=False)

    return Markup(f"""
    <div class="mt-5 border-top pt-4">
        <h3>Interactive Model Validation</h3>
        <p><strong>City:</strong> {location} | <strong>Period:</strong> {start_date} to {end_date}</p>
        
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body text-center">
                        <h5>Residential RMSPE</h5>
                        <h2 class="text-primary">{rmsp_res}%</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body text-center">
                        <h5>C&I RMSPE</h5>
                        <h2 class="text-primary">{rmsp_ci}%</h2>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-6">{plot_res_html}</div>
            <div class="col-md-6">{plot_ci_html}</div>
        </div>

        <h5 class="mt-4">RMSPE Summary</h5>
        <table class="table table-striped table-bordered">
            <thead><tr><th>Model</th><th>RMSPE (%)</th></tr></thead>
            <tbody>
                <tr><td>Residential</td><td>{rmsp_res}%</td></tr>
                <tr><td>C&I</td><td>{rmsp_ci}%</td></tr>
            </tbody>
        </table>
    </div>
    """)
