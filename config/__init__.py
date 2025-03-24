"""Configuration module for the supply chain forecaster."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from config.base_config import BaseConfig
from config.dev_config import DevConfig
from config.prod_config import ProdConfig

# Determine which configuration to use based on environment
ENV = os.getenv("ENV", "development").lower()

if ENV == "production":
    config = ProdConfig()
elif ENV == "development":
    config = DevConfig()
else:
    raise ValueError(f"Unknown environment: {ENV}")

__all__ = ["config", "BaseConfig", "DevConfig", "ProdConfig"]
