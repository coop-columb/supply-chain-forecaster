"""Main entry point for the supply chain forecaster API."""

import os
import uvicorn

from api.app import create_app
from config import config
from utils import setup_logger

# Initialize logger
setup_logger(config.LOG_LEVEL)

# Create FastAPI app
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_DEBUG,
    )
