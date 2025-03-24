"""Dashboard application factory for the supply chain forecaster."""

import os
import time
import uuid
from typing import Optional

import dash
import dash_bootstrap_components as dbc
from dash import Dash, callback_context, html
from flask import Flask, request, Response

from config import config
from dashboard.layouts import create_layout
from utils import get_logger, get_request_id, log_request, reset_request_id, set_request_id, setup_logger

logger = get_logger(__name__)


def configure_flask_server() -> Flask:
    """
    Configure the Flask server for Dash with middleware and routes.
    
    Returns:
        Configured Flask server.
    """
    is_production = os.getenv("ENV", "development") == "production"
    server = Flask(__name__)
    
    # Add request tracking middleware
    @server.before_request
    def before_request():
        # Extract or generate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        set_request_id(request_id)
        
        # Store start time
        request.start_time = time.time()
    
    @server.after_request
    def after_request(response):
        # Add request ID to response
        request_id = get_request_id()
        response.headers["X-Request-ID"] = request_id
        
        # Add timing information
        if hasattr(request, "start_time"):
            duration_ms = (time.time() - request.start_time) * 1000
            response.headers["X-Process-Time-ms"] = str(round(duration_ms, 2))
            
            # Log request in production
            if is_production and hasattr(config, "COLLECT_API_METRICS") and config.COLLECT_API_METRICS:
                log_request(
                    request_id=request_id,
                    method=request.method,
                    url=request.path,
                    status_code=response.status_code,
                    duration_ms=duration_ms,
                )
        
        # Reset request ID
        reset_request_id()
        
        return response
    
    # Add health check endpoints
    @server.route("/health")
    def health_check():
        return {"status": "ok", "timestamp": time.time()}
    
    @server.route("/health/readiness")
    def readiness_check():
        return {"status": "ready", "timestamp": time.time()}
    
    @server.route("/health/liveness")
    def liveness_check():
        return {"status": "alive", "timestamp": time.time()}
    
    # Add metrics endpoint if in production
    if is_production and hasattr(config, "PROMETHEUS_METRICS") and config.PROMETHEUS_METRICS:
        try:
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            
            @server.route("/metrics")
            def metrics():
                return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
        except ImportError:
            logger.warning("prometheus_client not installed, metrics endpoint not available")
    
    return server


def create_dashboard(api_url: Optional[str] = None) -> Dash:
    """
    Create and configure the Dash application.
    
    Args:
        api_url: URL of the API.
    
    Returns:
        Configured Dash application.
    """
    # Configure logging
    is_production = os.getenv("ENV", "development") == "production"
    log_file = config.LOG_FILE if hasattr(config, "LOG_FILE") else None
    json_format = getattr(config, "LOG_JSON_FORMAT", False)
    env = getattr(config, "LOG_ENVIRONMENT", "development")
    
    setup_logger(
        log_level=config.LOG_LEVEL,
        log_file=log_file,
        rotation=getattr(config, "LOG_ROTATION", "10 MB"),
        retention=getattr(config, "LOG_RETENTION", "1 week"),
        json_format=json_format,
        env=env,
    )
    
    # Default API URL if not provided
    if api_url is None:
        api_url = f"http://{config.API_HOST}:{config.API_PORT}"
    
    # Configure Flask server with middleware and monitoring endpoints
    server = configure_flask_server()
    
    # Create Dash app
    app = Dash(
        __name__,
        server=server,
        title="Supply Chain Forecaster",
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        update_title="Updating...",
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        ],
    )
    
    # Set app layout
    app.layout = create_layout(api_url)
    
    # Register callbacks
    from dashboard.callbacks import register_callbacks
    register_callbacks(app)
    
    # Add performance monitoring for callbacks in production
    if is_production and hasattr(config, "COLLECT_API_METRICS") and config.COLLECT_API_METRICS:
        # Wrap the dispatch method to track callback performance
        original_dispatch = app.callback_map["dispatch"]
        
        def dispatch_with_tracking(body, outputs, inputs, state):
            start_time = time.time()
            result = original_dispatch(body, outputs, inputs, state)
            duration_ms = (time.time() - start_time) * 1000
            
            # Log the callback execution time
            callback_name = callback_context.triggered[0]["prop_id"] if callback_context.triggered else "unknown"
            logger.info(
                f"Callback {callback_name} completed in {duration_ms:.2f}ms",
                callback=callback_name,
                duration_ms=duration_ms,
                request_id=get_request_id(),
            )
            
            return result
        
        app.callback_map["dispatch"] = dispatch_with_tracking
    
    # Log application startup
    logger.info(
        f"Dashboard initialized with API URL: {api_url}",
        api_url=api_url,
        environment=os.getenv("ENV", "development"),
    )
    
    return app


app = create_dashboard()

if __name__ == "__main__":
    app.run_server(
        host=config.DASHBOARD_HOST,
        port=config.DASHBOARD_PORT,
        debug=config.DASHBOARD_DEBUG,
    )
