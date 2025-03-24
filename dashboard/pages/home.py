"""Home page for the supply chain forecaster dashboard."""

import dash_bootstrap_components as dbc
from dash import dcc, html


def create_home_layout():
    """
    Create the home page layout.
    
    Returns:
        Home page layout.
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H1("Supply Chain Forecaster", className="text-center mb-4"),
                html.P(
                    "Welcome to the Supply Chain Forecaster dashboard. This application provides tools "
                    "for forecasting supply chain metrics and detecting anomalies in your data.",
                    className="lead text-center mb-5",
                ),
            ], width=12),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.I(className="fas fa-database fa-3x text-primary text-center mt-4"),
                    dbc.CardBody([
                        html.H4("Data Exploration", className="card-title text-center"),
                        html.P(
                            "Explore and analyze your supply chain data to identify patterns and trends. "
                            "Visualize historical data and understand key metrics.",
                            className="card-text",
                        ),
                        dbc.Button("Explore Data", href="/exploration", color="primary", className="mt-2"),
                    ]),
                ], className="h-100 shadow-sm"),
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    html.I(className="fas fa-chart-line fa-3x text-success text-center mt-4"),
                    dbc.CardBody([
                        html.H4("Forecasting", className="card-title text-center"),
                        html.P(
                            "Train forecasting models on your supply chain data and generate "
                            "predictions for future periods. Evaluate model performance.",
                            className="card-text",
                        ),
                        dbc.Button("Forecast", href="/forecasting", color="success", className="mt-2"),
                    ]),
                ], className="h-100 shadow-sm"),
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    html.I(className="fas fa-exclamation-triangle fa-3x text-warning text-center mt-4"),
                    dbc.CardBody([
                        html.H4("Anomaly Detection", className="card-title text-center"),
                        html.P(
                            "Detect anomalies in your supply chain data to identify unusual patterns "
                            "or outliers that may require attention.",
                            className="card-text",
                        ),
                        dbc.Button("Detect Anomalies", href="/anomaly", color="warning", className="mt-2"),
                    ]),
                ], className="h-100 shadow-sm"),
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    html.I(className="fas fa-cogs fa-3x text-info text-center mt-4"),
                    dbc.CardBody([
                        html.H4("Model Management", className="card-title text-center"),
                        html.P(
                            "Manage your trained models. View model details, deploy models, "
                            "and delete models that are no longer needed.",
                            className="card-text",
                        ),
                        dbc.Button("Manage Models", href="/models", color="info", className="mt-2"),
                    ]),
                ], className="h-100 shadow-sm"),
            ], width=3),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Getting Started")),
                    dbc.CardBody([
                        html.P(
                            "Follow these steps to start forecasting your supply chain metrics:",
                            className="card-text",
                        ),
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.Strong("Step 1: "),
                                "Upload your supply chain data in the Data Exploration page.",
                            ]),
                            dbc.ListGroupItem([
                                html.Strong("Step 2: "),
                                "Analyze and visualize your data to understand patterns and trends.",
                            ]),
                            dbc.ListGroupItem([
                                html.Strong("Step 3: "),
                                "Train forecasting models on your historical data in the Forecasting page.",
                            ]),
                            dbc.ListGroupItem([
                                html.Strong("Step 4: "),
                                "Generate predictions for future periods and evaluate model performance.",
                            ]),
                            dbc.ListGroupItem([
                                html.Strong("Step 5: "),
                                "Deploy your best models for production use.",
                            ]),
                        ]),
                    ]),
                ], className="shadow-sm"),
            ], width=12),
        ]),
    ])
