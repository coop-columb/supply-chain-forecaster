"""Anomaly detection page for the supply chain forecaster dashboard."""

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

from dashboard.components.data_upload import create_upload_component
from dashboard.components.model_selection import create_model_selection
from dashboard.components.charts import create_anomaly_chart


def create_anomaly_layout():
    """
    Create the anomaly detection page layout.
    
    Returns:
        Anomaly detection page layout.
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H1("Anomaly Detection", className="mb-4"),
                html.P(
                    "Train anomaly detection models on your supply chain data to identify "
                    "unusual patterns and outliers that may require attention.",
                    className="lead mb-4",
                ),
            ], width=12),
        ]),
        
        dbc.Tabs([
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        create_upload_component("train-anomaly"),
                    ], width=12),
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Model Configuration")),
                            dbc.CardBody([
                                html.Div(id="train-anomaly-model-selection"),
                                
                                html.Hr(),
                                
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Feature Columns"),
                                        dcc.Dropdown(
                                            id="train-anomaly-feature-columns",
                                            multi=True,
                                            placeholder="Select feature columns",
                                        ),
                                    ], width=12),
                                ], className="mb-3"),
                                
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Target Column (Optional)"),
                                        dcc.Dropdown(
                                            id="train-anomaly-target-column",
                                            placeholder="Select target column (optional)",
                                        ),
                                    ], width=6),
                                    
                                    dbc.Col([
                                        dbc.Button(
                                            "Train Model",
                                            id="train-anomaly-button",
                                            color="primary",
                                            className="w-100",
                                        ),
                                    ], width=6, className="d-flex align-items-end"),
                                ], className="mb-3"),
                            ]),
                        ]),
                    ], width=12),
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        html.Div(id="train-anomaly-results"),
                    ], width=12),
                ]),
            ], label="Train Model", tab_id="train-tab"),
            
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        create_upload_component("detect-anomaly"),
                    ], width=12),
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Detection Configuration")),
                            dbc.CardBody([
                                html.Div(id="detect-anomaly-model-selection"),
                                
                                html.Hr(),
                                
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Feature Columns"),
                                        dcc.Dropdown(
                                            id="detect-anomaly-feature-columns",
                                            multi=True,
                                            placeholder="Select feature columns",
                                        ),
                                    ], width=6),
                                    
                                    dbc.Col([
                                        html.Label("Date Column (for visualization)"),
                                        dcc.Dropdown(
                                            id="detect-anomaly-date-column",
                                            placeholder="Select date column",
                                        ),
                                    ], width=6),
                                ], className="mb-3"),
                                
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Threshold (optional)"),
                                        dcc.Input(
                                            id="detect-anomaly-threshold",
                                            type="number",
                                            placeholder="Default threshold",
                                            className="form-control",
                                        ),
                                    ], width=6),
                                    
                                    dbc.Col([
                                        dbc.Button(
                                            "Detect Anomalies",
                                            id="detect-anomaly-button",
                                            color="primary",
                                            className="w-100",
                                        ),
                                    ], width=6, className="d-flex align-items-end"),
                                ], className="mb-3"),
                            ]),
                        ]),
                    ], width=12),
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        html.Div(id="detect-anomaly-results"),
                    ], width=12),
                ]),
            ], label="Detect Anomalies", tab_id="detect-tab"),
        ], id="anomaly-tabs", active_tab="train-tab"),
    ])