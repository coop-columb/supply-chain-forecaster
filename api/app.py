"""API application factory for the supply chain forecaster."""

import logging
import os
import time
import uuid
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api.routes import anomaly, forecasting, health, model, prediction
from config import config
from utils import (
    ApplicationError,
    get_logger,
    get_request_id,
    log_request,
    reset_request_id,
    set_request_id,
    setup_logger,
)
from utils.monitoring import PrometheusMiddleware, create_monitoring_endpoints, setup_monitoring

logger = get_logger(__name__)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID and timing to each request."""
    
    async def dispatch(self, request: Request, call_next):
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        set_request_id(request_id)
        
        # Add request ID to response headers
        start_time = time.time()
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Add timing and request ID headers
            duration_ms = (time.time() - start_time) * 1000
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-ms"] = str(round(duration_ms, 2))
            
            # Log the request if API metrics collection is enabled
            if config.COLLECT_API_METRICS:
                log_request(
                    request_id=request_id,
                    method=request.method,
                    url=str(request.url),
                    status_code=response.status_code,
                    duration_ms=duration_ms,
                )
            
            return response
        finally:
            # Reset request ID at the end of processing
            reset_request_id()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application.
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
    
    # Create FastAPI app
    app = FastAPI(
        title="Supply Chain Forecaster API",
        description="API for supply chain forecasting and anomaly detection",
        version="0.1.0",
    )
    
    # Add monitoring if enabled in production
    if is_production and getattr(config, "PROMETHEUS_METRICS", False):
        # Set up monitoring and add Prometheus middleware
        setup_monitoring(export_metrics=True)
        app.add_middleware(PrometheusMiddleware)
        
        # Create monitoring endpoints
        create_monitoring_endpoints(app)
    
    # Add request tracing middleware
    app.add_middleware(RequestTracingMiddleware)
    
    # Add CORS middleware
    allowed_origins = getattr(config, "ALLOWED_ORIGINS", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add rate limiting if enabled
    if getattr(config, "ENABLE_RATE_LIMITING", False):
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded
        from slowapi.middleware import SlowAPIMiddleware
        from slowapi.util import get_remote_address
        
        limiter = Limiter(key_func=get_remote_address)
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        app.add_middleware(SlowAPIMiddleware)
        
        logger.info(f"Rate limiting enabled: {config.RATE_LIMIT_REQUESTS_PER_MINUTE} requests per minute")
    
    # Add exception handler
    @app.exception_handler(ApplicationError)
    async def application_error_handler(request: Request, exc: ApplicationError):
        # Log the error
        logger.error(
            f"Application error: {exc.message}",
            error=exc.message,
            status_code=exc.status_code,
            details=exc.details,
            request_path=request.url.path,
            request_id=get_request_id(),
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, "details": exc.details},
        )
    
    # Include routers
    app.include_router(health.router, tags=["health"])
    app.include_router(model.router, prefix="/models", tags=["models"])
    app.include_router(forecasting.router, prefix="/forecasting", tags=["forecasting"])
    app.include_router(prediction.router, prefix="/predictions", tags=["predictions"])
    app.include_router(anomaly.router, prefix="/anomalies", tags=["anomalies"])
    
    # Log application startup
    @app.on_event("startup")
    async def startup_event():
        logger.info(
            f"Starting Supply Chain Forecaster API (version: {app.version})",
            version=app.version,
            environment=os.getenv("ENV", "development"),
        )
    
    # Log application shutdown
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down Supply Chain Forecaster API")
    
    return app


app = create_app()
