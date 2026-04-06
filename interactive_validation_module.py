# interactive_validation_module.py

import pandas as pd
import plotly.graph_objects as go
import numpy as np
from dash import html, dcc
import dash_bootstrap_components as dbc


def calculate_rmsp(actual, predicted):
    """Root Mean Square Percentage Error (RMSPE)"""
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    mask = (actual != 0) & (~np.isnan(actual)) & (~np.isnan(predicted))
    if len(actual[mask]) == 0:
        return 0.0
    errors = ((actual[mask] - predicted[mask]) / actual[mask]) ** 2
    return round(np.sqrt(np.mean(errors)) * 100, 4)


def create_interactive_validation_layout(location: str, start_date: str, end_date: str):
    """
    This function creates the layout for Page 3 - Interactive Validation
    It reads Validation/Interactive_model_validation.xlsx and shows:
    - Two plots (Residential and C&I)
    - RMSPE values
    """
    if not all([location, start_date, end_date]):
        return dbc.Alert("❌ Please provide Location, Start Date, and End Date.", color="danger")

    file_path = "Validation/Interactive_model_validation.xlsx"

    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        return dbc.Alert(
            "❌ Validation file not found.\n"
            "Please run the validation script (interactive_validation.py) first.", 
            color="warning"
        )

    # Clean columns and parse date
    df.columns = [col.strip() for col in df.columns]
    if 'date' in df.columns:
        df['Date'] = pd.to_datetime(df['date'])

    # Filter by city/location
    if 'Location' in df.columns:
        df = df[df['Location'].astype(str).str.lower() == location.lower().strip()]

    # Filter by date range
    filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & 
                     (df['Date'] <= pd.to_datetime(end_date))].copy()

    if filtered_df.empty:
        return dbc.Alert(f"❌ No data found for {location} between {start_date} and {end_date}.", color="warning")

    # Calculate RMSPE
    rmsp_res = calculate_rmsp(filtered_df['res_actual'], filtered_df['res_predicted'])
    rmsp_ci = calculate_rmsp(filtered_df['ci_actual'], filtered_df['ci_predicted'])

    # Residential Plot
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['res_actual'], 
                                 name='Actual Residential', line=dict(color='#1f77b4', width=3)))
    fig_res.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['res_predicted'], 
                                 name='Predicted Residential', line=dict(color='#ff7f0e', width=3, dash='dash')))
    fig_res.update_layout(title=f"Residential Load Validation - {location}", height=520, template="plotly_white")

    # C&I Plot
    fig_ci = go.Figure()
    fig_ci.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['ci_actual'], 
                                name='Actual C&I', line=dict(color='#1f77b4', width=3)))
    fig_ci.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['ci_predicted'], 
                                name='Predicted C&I', line=dict(color='#ff7f0e', width=3, dash='dash')))
    fig_ci.update_layout(title=f"C&I Load Validation - {location}", height=520, template="plotly_white")

    return html.Div([
        html.H3("Interactive Model Validation", className="mt-4 mb-3"),
        html.P(f"City: {location.title()} | Period: {start_date} to {end_date}", className="text-muted"),

        # RMSPE Cards
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Residential RMSPE"),
                html.H2(f"{rmsp_res}%", className="text-primary")
            ])), width=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("C&I RMSPE"),
                html.H2(f"{rmsp_ci}%", className="text-primary")
            ])), width=6),
        ], className="mb-4"),

        # Plots
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_res), width=6),
            dbc.Col(dcc.Graph(figure=fig_ci), width=6),
        ]),

        # Summary Table
        html.H5("RMSPE Summary", className="mt-4"),
        dbc.Table([
            html.Thead(html.Tr([html.Th("Model"), html.Th("RMSPE (%)")])),
            html.Tbody([
                html.Tr([html.Td("Residential"), html.Td(f"{rmsp_res}%")]),
                html.Tr([html.Td("C&I"), html.Td(f"{rmsp_ci}%")])
            ])
        ], striped=True, bordered=True, hover=True)
    ])
