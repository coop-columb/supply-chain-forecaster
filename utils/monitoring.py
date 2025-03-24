"""Monitoring utilities for the supply chain forecaster."""

import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server

from config import config
from utils.logging import get_logger, get_request_id

logger = get_logger(__name__)

# Define Prometheus metrics
http_requests_total = Counter(
    'http_requests_total', 'Total HTTP Requests', ['method', 'endpoint', 'status']
)
request_duration_seconds = Histogram(
    'request_duration_seconds', 'HTTP Request Duration in seconds', ['method', 'endpoint']
)
prediction_duration_seconds = Histogram(
    'prediction_duration_seconds', 'Model Prediction Duration in seconds', ['model_name']
)
model_prediction_count = Counter(
    'model_prediction_count', 'Total Number of Model Predictions', ['model_name', 'status']
)
model_latency_seconds = Summary(
    'model_latency_seconds', 'Model Prediction Latency', ['model_name']
)
active_requests = Gauge(
    'active_requests', 'Number of active requests'
)
errors_total = Counter(
    'errors_total', 'Total number of errors', ['error_type']
)
memory_usage_bytes = Gauge(
    'memory_usage_bytes', 'Memory usage in bytes'
)
cpu_usage_percent = Gauge(
    'cpu_usage_percent', 'CPU usage in percent'
)


def setup_monitoring(export_metrics: bool = True, metrics_port: int = 8000) -> None:
    """
    Set up monitoring with Prometheus metrics.
    
    Args:
        export_metrics: Whether to export metrics via HTTP server
        metrics_port: Port to expose metrics on
    """
    if export_metrics:
        start_http_server(metrics_port)
        logger.info(f"Prometheus metrics server started on port {metrics_port}")


def track_request_duration(func: Callable) -> Callable:
    """
    Decorator to track request durations and count.
    
    Args:
        func: The FastAPI endpoint function to track
        
    Returns:
        Decorated function with metrics tracking
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        active_requests.inc()
        method = kwargs.get('request').method if 'request' in kwargs else 'UNKNOWN'
        endpoint = func.__name__
        
        start_time = time.time()
        try:
            response = await func(*args, **kwargs)
            duration = time.time() - start_time
            status = response.status_code
            
            # Record metrics
            http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
            request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
            
            # Log request if enabled
            if config.COLLECT_API_METRICS:
                from utils.logging import log_request
                request_id = get_request_id()
                log_request(
                    request_id=request_id,
                    method=method,
                    url=kwargs.get('request').url.path if 'request' in kwargs else endpoint,
                    status_code=status,
                    duration_ms=duration * 1000
                )
                
            return response
        except Exception as e:
            errors_total.labels(error_type=type(e).__name__).inc()
            raise
        finally:
            active_requests.dec()
    
    return wrapper


def track_model_prediction(func: Callable) -> Callable:
    """
    Decorator to track model prediction performance.
    
    Args:
        func: The model prediction function to track
        
    Returns:
        Decorated function with model metrics tracking
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        model_name = args[0].__class__.__name__ if args else 'unknown_model'
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Record metrics
            model_prediction_count.labels(model_name=model_name, status='success').inc()
            prediction_duration_seconds.labels(model_name=model_name).observe(duration)
            model_latency_seconds.labels(model_name=model_name).observe(duration)
            
            # Log model prediction if enabled
            if config.COLLECT_PREDICTION_METRICS:
                from utils.logging import log_model_prediction
                
                if hasattr(args[0], 'input_shape') and hasattr(args[0], 'output_shape'):
                    input_shape = args[0].input_shape
                    output_shape = args[0].output_shape
                else:
                    # Try to determine shapes from inputs and outputs
                    input_shape = tuple([len(arg) for arg in args[1:] if hasattr(arg, '__len__')])
                    output_shape = result.shape if hasattr(result, 'shape') else (1,)
                
                log_model_prediction(
                    model_name=model_name,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    duration_ms=duration * 1000,
                    request_id=get_request_id()
                )
                
            return result
        except Exception as e:
            model_prediction_count.labels(model_name=model_name, status='error').inc()
            errors_total.labels(error_type=type(e).__name__).inc()
            raise
    
    return wrapper


def update_resource_metrics() -> None:
    """Update resource usage metrics (CPU, memory)."""
    try:
        import psutil
        
        # Get memory usage
        memory_info = psutil.Process().memory_info()
        memory_usage_bytes.set(memory_info.rss)
        
        # Get CPU usage
        cpu_usage_percent.set(psutil.Process().cpu_percent(interval=0.1))
        
    except ImportError:
        logger.warning("psutil not installed, resource metrics will not be collected")


class PrometheusMiddleware:
    """
    Middleware for tracking request metrics with Prometheus.
    
    This middleware tracks HTTP request counts and durations for all routes.
    """
    
    def __init__(self, app):
        """
        Initialize the middleware.
        
        Args:
            app: The FastAPI application
        """
        self.app = app
    
    async def __call__(self, scope, receive, send):
        """Process a request and record metrics."""
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        # Extract request details
        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "UNKNOWN")
        
        active_requests.inc()
        start_time = time.time()
        
        # Modified send function to capture status code
        original_send = send
        
        async def send_with_metrics(message):
            if message["type"] == "http.response.start":
                status_code = message["status"]
                duration = time.time() - start_time
                
                # Record metrics
                http_requests_total.labels(method=method, endpoint=path, status=status_code).inc()
                request_duration_seconds.labels(method=method, endpoint=path).observe(duration)
                
                # Update resource metrics periodically
                if config.COLLECT_API_METRICS:
                    update_resource_metrics()
                
            return await original_send(message)
        
        try:
            return await self.app(scope, receive, send_with_metrics)
        except Exception as e:
            errors_total.labels(error_type=type(e).__name__).inc()
            raise
        finally:
            active_requests.dec()


def create_monitoring_endpoints(app):
    """
    Create monitoring endpoints for health checks and metrics.
    
    Args:
        app: The FastAPI application
    """
    from fastapi import FastAPI, Response
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    
    @app.get(config.HEALTH_CHECK_PATH)
    async def health_check():
        """Basic health check endpoint."""
        return {"status": "ok", "timestamp": time.time()}
    
    @app.get(config.READINESS_CHECK_PATH)
    async def readiness_check():
        """Check if the application is ready to receive traffic."""
        # In a real implementation, check database connections, model availability, etc.
        return {"status": "ready", "timestamp": time.time()}
    
    @app.get(config.LIVENESS_CHECK_PATH)
    async def liveness_check():
        """Check if the application is alive and functioning."""
        return {"status": "alive", "timestamp": time.time()}
    
    if config.PROMETHEUS_METRICS:
        @app.get(config.METRICS_ENDPOINT)
        async def metrics():
            """Endpoint to expose Prometheus metrics."""
            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)