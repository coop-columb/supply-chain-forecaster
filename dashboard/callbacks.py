"""Callbacks for the supply chain forecaster dashboard."""

import json
import base64
import io
import pandas as pd
import numpy as np
import requests
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

from dashboard.components.data_upload import parse_contents
from dashboard.components.model_selection import create_model_selection, create_deployed_model_selection, create_model_parameters
from dashboard.components.charts import (
    create_time_series_chart,
    create_forecast_chart,
    create_anomaly_chart,
    create_feature_importance_chart,
)
from dashboard.pages.data_exploration import create_summary_stats, create_correlation_heatmap
from dashboard.pages.model_management import create_model_list_table, create_model_details


def register_callbacks(app: Dash):
    """
    Register callbacks for the dashboard.
    
    Args:
        app: Dash application.
    """
    # URL routing callback
    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname"),
    )
    def render_page_content(pathname):
        from dashboard.pages.home import create_home_layout
        from dashboard.pages.data_exploration import create_data_exploration_layout
        from dashboard.pages.forecasting import create_forecasting_layout
        from dashboard.pages.anomaly_detection import create_anomaly_layout
        from dashboard.pages.model_management import create_model_management_layout
        
        if pathname == "/":
            return create_home_layout()
        elif pathname == "/exploration":
            return create_data_exploration_layout()
        elif pathname == "/forecasting":
            return create_forecasting_layout()
        elif pathname == "/anomaly":
            return create_anomaly_layout()
        elif pathname == "/models":
            return create_model_management_layout()
        else:
            # 404 page
            return html.Div([
                html.H1("404: Page Not Found", className="text-danger"),
                html.P(f"The pathname {pathname} was not recognized..."),
                dbc.Button("Go to Home", href="/", color="primary"),
            ], className="p-3 bg-light rounded-3 text-center")
    
    # Navbar toggle callback
    @app.callback(
        Output("navbar-collapse", "is_open"),
        [Input("navbar-toggler", "n_clicks")],
        [State("navbar-collapse", "is_open")],
    )
    def toggle_navbar_collapse(n, is_open):
        if n:
            return not is_open
        return is_open
    
    # Register data exploration callbacks
    register_data_exploration_callbacks(app)
    
    # Register forecasting callbacks
    register_forecasting_callbacks(app)
    
    # Register anomaly detection callbacks
    register_anomaly_detection_callbacks(app)
    
    # Register model management callbacks
    register_model_management_callbacks(app)


def register_data_exploration_callbacks(app: Dash):
    """
    Register callbacks for the data exploration page.
    
    Args:
        app: Dash application.
    """
    # Data upload callback
    @app.callback(
        [Output("exploration-upload-output", "children"),
         Output("exploration-data-store", "data"),
         Output("exploration-data-preview", "children")],
        [Input("exploration-upload", "contents")],
        [State("exploration-upload", "filename")]
    )
    def update_exploration_output(contents, filename):
        if contents is None:
            return no_update, no_update, no_update
        
        data_json, preview = parse_contents(contents, filename)
        
        return html.Div([
            html.Div(f"Uploaded: {filename}", className="text-success"),
        ]), data_json, preview
    
    # Summary statistics callback
    @app.callback(
        Output("exploration-summary-stats", "children"),
        [Input("exploration-data-store", "data")]
    )
    def update_summary_stats(data_json):
        if data_json is None:
            return no_update
        
        df = pd.read_json(data_json, orient='split')
        return create_summary_stats(df)
    
    # Time series charts callback
    @app.callback(
        Output("exploration-time-series-container", "children"),
        [Input("exploration-data-store", "data")]
    )
    def update_time_series(data_json):
        if data_json is None:
            return no_update
        
        df = pd.read_json(data_json, orient='split')
        
        # Check if there's a date/time column
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_dtype(df[col])]
        if not date_cols and df.index.dtype.kind not in 'M':
            # Try to convert potential date columns
            for col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
                except:
                    continue
        
        # If no date column found, return message
        if not date_cols and df.index.dtype.kind not in 'M':
            return html.Div("No date/time columns found for time series visualization.")
        
        # Use the first date column or index
        x_col = date_cols[0] if date_cols else None
        
        # Get numeric columns for y-axis
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Create charts for up to 3 numeric columns
        charts = []
        for i, col in enumerate(numeric_cols[:3]):
            if x_col:
                charts.append(create_time_series_chart(df, x_col, [col], f"{col} Over Time", f"exploration-{i}"))
            else:
                # Use index as x-axis
                df_copy = df.copy()
                df_copy['date_index'] = df_copy.index
                charts.append(create_time_series_chart(df_copy, 'date_index', [col], f"{col} Over Time", f"exploration-{i}"))
        
        return html.Div([
            html.H5("Time Series Visualizations", className="mb-3"),
            dbc.Row([
                dbc.Col(chart, width=12, className="mb-4")
                for chart in charts
            ]),
        ])
    
    # Correlation heatmap callback
    @app.callback(
        Output("exploration-correlation-container", "children"),
        [Input("exploration-data-store", "data")]
    )
    def update_correlation(data_json):
        if data_json is None:
            return no_update
        
        df = pd.read_json(data_json, orient='split')
        return create_correlation_heatmap(df)


def register_forecasting_callbacks(app: Dash):
    """
    Register callbacks for the forecasting page.
    
    Args:
        app: Dash application.
    """
    # Get available models
    @app.callback(
        Output("train-forecasting-model-selection", "children"),
        [Input("api-url", "data")]
    )
    def update_train_forecasting_model_selection(api_url):
        try:
            response = requests.get(f"{api_url}/models/")
            if response.status_code == 200:
                data = response.json()
                model_types = [model for model in data.get("available_models", [])
                              if model.endswith("Model") and model != "ModelBase"]
                return create_model_selection("train-forecasting", model_types)
            else:
                return html.Div(f"Error fetching model types: {response.status_code}")
        except Exception as e:
            return html.Div(f"Error connecting to API: {str(e)}")
    
    # Update model parameters based on selected model type
    @app.callback(
        Output("train-forecasting-model-params-container", "children"),
        [Input("train-forecasting-model-type", "value")]
    )
    def update_train_forecasting_model_params(model_type):
        if model_type is None:
            return no_update
        
        return create_model_parameters(model_type, "train-forecasting")
    
    # Data upload callback for training
    @app.callback(
        [Output("train-forecasting-upload-output", "children"),
         Output("train-forecasting-data-store", "data"),
         Output("train-forecasting-data-preview", "children"),
         Output("train-forecasting-feature-columns", "options"),
         Output("train-forecasting-target-column", "options"),
         Output("train-forecasting-date-column", "options")],
        [Input("train-forecasting-upload", "contents")],
        [State("train-forecasting-upload", "filename")]
    )
    def update_train_forecasting_output(contents, filename):
        if contents is None:
            return no_update, no_update, no_update, no_update, no_update, no_update
        
        data_json, preview = parse_contents(contents, filename)
        
        if data_json is None:
            return html.Div([
                html.Div(f"Error processing file: {filename}", className="text-danger"),
            ]), no_update, no_update, no_update, no_update, no_update
        
        df = pd.read_json(data_json, orient='split')
        
        # Create column options
        all_cols = [{"label": col, "value": col} for col in df.columns]
        
        # Try to identify date columns
        date_cols = []
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                date_cols.append({"label": col, "value": col})
            except:
                continue
        
        return html.Div([
            html.Div(f"Uploaded: {filename}", className="text-success"),
        ]), data_json, preview, all_cols, all_cols, date_cols
    
    # Train model callback
    @app.callback(
        Output("train-forecasting-results", "children"),
        [Input("train-forecasting-button", "n_clicks")],
        [State("api-url", "data"),
         State("train-forecasting-model-type", "value"),
         State("train-forecasting-model-name", "value"),
         State("train-forecasting-data-store", "data"),
         State("train-forecasting-feature-columns", "value"),
         State("train-forecasting-target-column", "value"),
         State("train-forecasting-date-column", "value"),
         # Add additional states for model parameters here
        ]
    )
    def train_forecasting_model(n_clicks, api_url, model_type, model_name, data_json, 
                               feature_columns, target_column, date_column, *args):
        if n_clicks is None or data_json is None or model_type is None or target_column is None:
            return no_update
        
        try:
            # Get model parameters based on model type
            model_params = {}
            # Fill model_params dictionary based on the model type and inputs
            
            # Create training parameters
            training_params = {
                "model_type": model_type,
                "model_name": model_name,
                "feature_columns": feature_columns,
                "target_column": target_column,
                "date_column": date_column,
                "model_params": model_params,
                "save_model": True
            }
            
            # Convert dataframe to file
            df = pd.read_json(data_json, orient='split')
            buffer = io.StringIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            files = {'file': ('data.csv', buffer.getvalue())}
            
            # Send request to API
            response = requests.post(
                f"{api_url}/forecasting/train",
                files=files,
                data={"params": json.dumps(training_params)}
            )
            
            if response.status_code == 200:
                result = response.json()
                return html.Div([
                    dbc.Alert(f"Model '{result['model_name']}' trained successfully!", color="success"),
                    dbc.Card([
                        dbc.CardHeader("Training Metrics"),
                        dbc.CardBody([
                            html.Pre(json.dumps(result['metrics'], indent=2))
                        ])
                    ])
                ])
            else:
                return dbc.Alert(f"Error training model: {response.text}", color="danger")
        
        except Exception as e:
            return dbc.Alert(f"Error training model: {str(e)}", color="danger")
    
    # Similar callbacks for forecast and cross-validation tabs
    # The implementation would follow a similar pattern


def register_anomaly_detection_callbacks(app: Dash):
    """
    Register callbacks for the anomaly detection page.
    
    Args:
        app: Dash application.
    """
    # Similar to forecasting callbacks, but for anomaly detection
    # Would include callbacks for getting model types, updating parameters,
    # uploading data, training models, and detecting anomalies


def register_model_management_callbacks(app: Dash):
    """
    Register callbacks for the model management page.
    
    Args:
        app: Dash application.
    """
    # Refresh trained models list
    @app.callback(
        Output("trained-models-list", "children"),
        [Input("refresh-trained-models-button", "n_clicks"),
         Input("url", "pathname")],
        [State("api-url", "data")]
    )
    def refresh_trained_models(n_clicks, pathname, api_url):
        if pathname != "/models" or api_url is None:
            return no_update
        
        try:
            response = requests.get(f"{api_url}/models/?trained=true")
            if response.status_code == 200:
                models_data = response.json().get("trained_models", {})
                return create_model_list_table(models_data, "trained-models")
            else:
                return html.Div(f"Error fetching trained models: {response.status_code}")
        except Exception as e:
            return html.Div(f"Error connecting to API: {str(e)}")
    
    # Refresh deployed models list
    @app.callback(
        Output("deployed-models-list", "children"),
        [Input("refresh-deployed-models-button", "n_clicks"),
         Input("url", "pathname")],
        [State("api-url", "data")]
    )
    def refresh_deployed_models(n_clicks, pathname, api_url):
        if pathname != "/models" or api_url is None:
            return no_update
        
        try:
            response = requests.get(f"{api_url}/models/?deployed=true")
            if response.status_code == 200:
                models_data = response.json().get("deployed_models", {})
                return create_model_list_table(models_data, "deployed-models")
            else:
                return html.Div(f"Error fetching deployed models: {response.status_code}")
        except Exception as e:
            return html.Div(f"Error connecting to API: {str(e)}")
    
    # Show trained model details
    @app.callback(
        Output("trained-model-details", "children"),
        [Input("trained-models-table", "selected_rows")],
        [State("trained-models-table", "data"),
         State("api-url", "data")]
    )
    def show_trained_model_details(selected_rows, data, api_url):
        if not selected_rows or not data:
            return no_update
        
        model_id = data[selected_rows[0]]["model_id"]
        model_type = data[selected_rows[0]]["model_type"]
        
        try:
            response = requests.get(f"{api_url}/models/{model_id}?model_type={model_type}")
            if response.status_code == 200:
                model_info = response.json()
                return create_model_details(model_info, False, "trained-model")
            else:
                return html.Div(f"Error fetching model details: {response.status_code}")
        except Exception as e:
            return html.Div(f"Error connecting to API: {str(e)}")
    
    # Show deployed model details
    @app.callback(
        Output("deployed-model-details", "children"),
        [Input("deployed-models-table", "selected_rows")],
        [State("deployed-models-table", "data"),
         State("api-url", "data")]
    )
    def show_deployed_model_details(selected_rows, data, api_url):
        if not selected_rows or not data:
            return no_update
        
        model_id = data[selected_rows[0]]["model_id"]
        model_type = data[selected_rows[0]]["model_type"]
        
        try:
            response = requests.get(f"{api_url}/models/{model_id}?model_type={model_type}&from_deployment=true")
            if response.status_code == 200:
                model_info = response.json()
                return create_model_details(model_info, True, "deployed-model")
            else:
                return html.Div(f"Error fetching model details: {response.status_code}")
        except Exception as e:
            return html.Div(f"Error connecting to API: {str(e)}")
    
    # Deploy model callback
    @app.callback(
        Output("trained-model-deploy-status", "children"),
        [Input("trained-model-deploy-button", "n_clicks")],
        [State("trained-model-selected-model", "data"),
         State("api-url", "data")]
    )
    def deploy_model(n_clicks, model_id, api_url):
        if n_clicks is None or model_id is None:
            return no_update
        
        try:
            response = requests.post(f"{api_url}/models/{model_id}/deploy")
            if response.status_code == 200:
                result = response.json()
                return dbc.Alert(f"Model deployed successfully!", color="success")
            else:
                return dbc.Alert(f"Error deploying model: {response.text}", color="danger")
        except Exception as e:
            return dbc.Alert(f"Error deploying model: {str(e)}", color="danger")
    
    # Delete model callbacks for both trained and deployed models