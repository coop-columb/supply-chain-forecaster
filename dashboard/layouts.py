"""Layouts for the supply chain forecaster dashboard."""

import dash_bootstrap_components as dbc
from dash import dcc, html

from dashboard.components.navbar import create_navbar
from dashboard.pages.anomaly_detection import create_anomaly_layout
from dashboard.pages.data_exploration import create_data_exploration_layout
from dashboard.pages.forecasting import create_forecasting_layout
from dashboard.pages.home import create_home_layout
from dashboard.pages.model_management import create_model_management_layout


def create_layout(api_url: str):
    """
    Create the main dashboard layout.
    
    Args:
        api_url: URL of the API.
    
    Returns:
        Dashboard layout.
    """
    return html.Div([
        # Store API URL in a dcc.Store for access in callbacks
        dcc.Store(id='api-url', data=api_url),
        
        # Create the navbar
        create_navbar(),
        
        # Main content area
        dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        html.Div(id='page-content', className='my-4'),
                        width=12,
                    )
                )
            ],
            fluid=True,
            className='mt-4',
        ),
        
        # URL routing
        dcc.Location(id='url', refresh=False),
    ])
