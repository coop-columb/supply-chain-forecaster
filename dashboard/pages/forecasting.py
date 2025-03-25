"""Forecasting page for the supply chain forecaster dashboard."""

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from dashboard.components.charts import create_forecast_chart
from dashboard.components.data_upload import create_upload_component
from dashboard.components.model_selection import create_model_selection


def create_forecasting_layout():
    """
    Create the forecasting page layout.

    Returns:
        Forecasting page layout.
    """
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H1("Forecasting", className="mb-4"),
                            html.P(
                                "Train forecasting models on your supply chain data and generate "
                                "predictions for future periods. Evaluate model performance and deploy models.",
                                className="lead mb-4",
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
            dbc.Tabs(
                [
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            create_upload_component(
                                                "train-forecasting"
                                            ),
                                        ],
                                        width=12,
                                    ),
                                ],
                                className="mb-4",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader(
                                                        html.H5("Model Configuration")
                                                    ),
                                                    dbc.CardBody(
                                                        [
                                                            html.Div(
                                                                id="train-forecasting-model-selection"
                                                            ),
                                                            html.Hr(),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Feature Columns"
                                                                            ),
                                                                            dcc.Dropdown(
                                                                                id="train-forecasting-feature-columns",
                                                                                multi=True,
                                                                                placeholder="Select feature columns",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Target Column"
                                                                            ),
                                                                            dcc.Dropdown(
                                                                                id="train-forecasting-target-column",
                                                                                placeholder="Select target column",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Date Column"
                                                                            ),
                                                                            dcc.Dropdown(
                                                                                id="train-forecasting-date-column",
                                                                                placeholder="Select date column",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Button(
                                                                                "Train Model",
                                                                                id="train-forecasting-button",
                                                                                color="primary",
                                                                                className="w-100",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                        className="d-flex align-items-end",
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            ),
                                        ],
                                        width=12,
                                    ),
                                ],
                                className="mb-4",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(id="train-forecasting-results"),
                                        ],
                                        width=12,
                                    ),
                                ]
                            ),
                        ],
                        label="Train Model",
                        tab_id="train-tab",
                    ),
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            create_upload_component("forecast"),
                                        ],
                                        width=12,
                                    ),
                                ],
                                className="mb-4",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader(
                                                        html.H5(
                                                            "Forecast Configuration"
                                                        )
                                                    ),
                                                    dbc.CardBody(
                                                        [
                                                            html.Div(
                                                                id="forecast-model-selection"
                                                            ),
                                                            html.Hr(),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Feature Columns"
                                                                            ),
                                                                            dcc.Dropdown(
                                                                                id="forecast-feature-columns",
                                                                                multi=True,
                                                                                placeholder="Select feature columns",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Date Column"
                                                                            ),
                                                                            dcc.Dropdown(
                                                                                id="forecast-date-column",
                                                                                placeholder="Select date column",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Forecast Horizon"
                                                                            ),
                                                                            dcc.Input(
                                                                                id="forecast-horizon",
                                                                                type="number",
                                                                                min=1,
                                                                                max=365,
                                                                                value=30,
                                                                                className="form-control",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Button(
                                                                                "Generate Forecast",
                                                                                id="forecast-button",
                                                                                color="primary",
                                                                                className="w-100",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                        className="d-flex align-items-end",
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            ),
                                        ],
                                        width=12,
                                    ),
                                ],
                                className="mb-4",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(id="forecast-results"),
                                        ],
                                        width=12,
                                    ),
                                ]
                            ),
                        ],
                        label="Generate Forecast",
                        tab_id="forecast-tab",
                    ),
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            create_upload_component("cv-forecasting"),
                                        ],
                                        width=12,
                                    ),
                                ],
                                className="mb-4",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader(
                                                        html.H5(
                                                            "Cross-Validation Configuration"
                                                        )
                                                    ),
                                                    dbc.CardBody(
                                                        [
                                                            html.Div(
                                                                id="cv-forecasting-model-selection"
                                                            ),
                                                            html.Hr(),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Feature Columns"
                                                                            ),
                                                                            dcc.Dropdown(
                                                                                id="cv-forecasting-feature-columns",
                                                                                multi=True,
                                                                                placeholder="Select feature columns",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Target Column"
                                                                            ),
                                                                            dcc.Dropdown(
                                                                                id="cv-forecasting-target-column",
                                                                                placeholder="Select target column",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Date Column"
                                                                            ),
                                                                            dcc.Dropdown(
                                                                                id="cv-forecasting-date-column",
                                                                                placeholder="Select date column",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "CV Strategy"
                                                                            ),
                                                                            dcc.Dropdown(
                                                                                id="cv-forecasting-strategy",
                                                                                options=[
                                                                                    {
                                                                                        "label": "Expanding Window",
                                                                                        "value": "expanding",
                                                                                    },
                                                                                    {
                                                                                        "label": "Sliding Window",
                                                                                        "value": "sliding",
                                                                                    },
                                                                                ],
                                                                                value="expanding",
                                                                            ),
                                                                        ],
                                                                        width=6,
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Initial Window"
                                                                            ),
                                                                            dcc.Input(
                                                                                id="cv-forecasting-initial-window",
                                                                                type="number",
                                                                                min=10,
                                                                                value=30,
                                                                                className="form-control",
                                                                            ),
                                                                        ],
                                                                        width=3,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Step Size"
                                                                            ),
                                                                            dcc.Input(
                                                                                id="cv-forecasting-step-size",
                                                                                type="number",
                                                                                min=1,
                                                                                value=7,
                                                                                className="form-control",
                                                                            ),
                                                                        ],
                                                                        width=3,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Label(
                                                                                "Horizon"
                                                                            ),
                                                                            dcc.Input(
                                                                                id="cv-forecasting-horizon",
                                                                                type="number",
                                                                                min=1,
                                                                                value=7,
                                                                                className="form-control",
                                                                            ),
                                                                        ],
                                                                        width=3,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Button(
                                                                                "Run Cross-Validation",
                                                                                id="cv-forecasting-button",
                                                                                color="primary",
                                                                                className="w-100",
                                                                            ),
                                                                        ],
                                                                        width=3,
                                                                        className="d-flex align-items-end",
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            ),
                                        ],
                                        width=12,
                                    ),
                                ],
                                className="mb-4",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(id="cv-forecasting-results"),
                                        ],
                                        width=12,
                                    ),
                                ]
                            ),
                        ],
                        label="Cross-Validation",
                        tab_id="cv-tab",
                    ),
                ],
                id="forecasting-tabs",
                active_tab="train-tab",
            ),
        ]
    )
