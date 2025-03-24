"""Main entry point for the supply chain forecaster dashboard."""

import os
from dash import Dash
import dash_bootstrap_components as dbc

from config import config
from dashboard.app import create_dashboard
from utils import setup_logger

# Initialize logger
setup_logger(config.LOG_LEVEL)

# Set API URL from environment or default
api_url = os.getenv("API_URL", f"http://{config.API_HOST}:{config.API_PORT}")

# Create Dash app
app = create_dashboard(api_url)

if __name__ == "__main__":
    app.run_server(
        host=config.DASHBOARD_HOST,
        port=config.DASHBOARD_PORT,
        debug=config.DASHBOARD_DEBUG,
    )
