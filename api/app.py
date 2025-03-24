"""API application factory for the supply chain forecaster."""

import logging
import os
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import anomaly, forecasting, health, model, prediction
from config import config
from utils import ApplicationError, get_logger, setup_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application.
    """
    # Configure logging
    setup_logger(config.LOG_LEVEL)
    
    # Create FastAPI app
    app = FastAPI(
        title="Supply Chain Forecaster API",
        description="API for supply chain forecasting and anomaly detection",
        version="0.1.0",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict this to specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add exception handler
    @app.exception_handler(ApplicationError)
    async def application_error_handler(request: Request, exc: ApplicationError):
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
        logger.info(f"Starting Supply Chain Forecaster API (version: {app.version})")
        logger.info(f"Environment: {os.getenv('ENV', 'development')}")
    
    # Log application shutdown
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down Supply Chain Forecaster API")
    
    return app


app = create_app()
