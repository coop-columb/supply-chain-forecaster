"""Dashboard application factory for the supply chain forecaster."""

import os
from typing import Optional

import dash
from dash import Dash, html
import dash_bootstrap_components as dbc

from config import config
from dashboard.layouts import create_layout
from utils import get_logger, setup_logger

logger = get_logger(__name__)


def create_dashboard(api_url: Optional[str] = None) -> Dash:
    """
    Create and configure the Dash application.
    
    Args:
        api_url: URL of the API.
    
    Returns:
        Configured Dash application.
    """
    # Configure logging
    setup_logger(config.LOG_LEVEL)
    
    # Default API URL if not provided
    if api_url is None:
        api_url = f"http://{config.API_HOST}:{config.API_PORT}"
    
    # Create Dash app
    app = Dash(
        __name__,
        title="Supply Chain Forecaster",
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )
    
    # Set app layout
    app.layout = create_layout(api_url)
    
    # Register callbacks
    from dashboard.callbacks import register_callbacks
    register_callbacks(app)
    
    # Log application startup
    logger.info(f"Dashboard initialized with API URL: {api_url}")
    
    return app


app = create_dashboard()

if __name__ == "__main__":
    app.run_server(
        host=config.DASHBOARD_HOST,
        port=config.DASHBOARD_PORT,
        debug=config.DASHBOARD_DEBUG,
    )
