"""Base configuration for the supply chain forecaster."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union


class BaseConfig:
    """Base configuration class with common settings."""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR = DATA_DIR / "external"
    MODELS_DIR = PROJECT_ROOT / "models"
    TRAINING_MODELS_DIR = MODELS_DIR / "training"
    EVALUATION_MODELS_DIR = MODELS_DIR / "evaluation"
    DEPLOYMENT_MODELS_DIR = MODELS_DIR / "deployment"

    # Ensure directories exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DEPLOYMENT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # API configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_DEBUG = os.getenv("API_DEBUG", "False").lower() == "true"

    # Dashboard configuration
    DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8050"))
    DASHBOARD_DEBUG = os.getenv("DASHBOARD_DEBUG", "False").lower() == "true"

    # Model configuration
    MODEL_TYPE = os.getenv("MODEL_TYPE", "prophet")
    MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
    MODEL_STORAGE_PATH = os.getenv("MODEL_STORAGE_PATH", str(DEPLOYMENT_MODELS_DIR))

    # Time series forecasting parameters
    FORECAST_HORIZON = 30  # Days to forecast
    FORECAST_FREQUENCY = "D"  # Daily frequency

    # Data preprocessing parameters
    OUTLIER_DETECTION_METHOD = "zscore"
    OUTLIER_THRESHOLD = 3.0
    IMPUTATION_METHOD = "linear"

    # Feature engineering parameters
    INCLUDE_HOLIDAYS = True
    INCLUDE_WEATHER = False
    LAG_FEATURES = [1, 7, 14, 30]
    ROLLING_WINDOW_SIZES = [7, 14, 30]

    # Cross-validation parameters
    CV_STRATEGY = "expanding"
    CV_INITIAL_WINDOW = 30
    CV_STEP_SIZE = 7
    CV_HORIZON = 14

    # Error metrics to compute
    ERROR_METRICS = ["mae", "rmse", "mape", "smape"]

    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Security configuration
    ENABLE_AUTH = os.getenv("ENABLE_AUTH", "False").lower() == "true"
    SECRET_KEY = os.getenv("SECRET_KEY", None)
    API_KEYS_FILE = os.getenv("API_KEYS_FILE", str(PROJECT_ROOT / "api_keys.json"))
    TOKEN_EXPIRE_MINUTES = int(os.getenv("TOKEN_EXPIRE_MINUTES", "60"))
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    # Performance profiling configuration
    ENABLE_PROFILING = os.getenv("ENABLE_PROFILING", "False").lower() == "true"
    PROFILING_SAMPLE_RATE = float(
        os.getenv("PROFILING_SAMPLE_RATE", "0.1")
    )  # Sample 10% of requests by default

    # Caching configuration
    ENABLE_MODEL_CACHING = os.getenv("ENABLE_MODEL_CACHING", "False").lower() == "true"
    MODEL_CACHE_SIZE = int(os.getenv("MODEL_CACHE_SIZE", "10"))
    ENABLE_RESPONSE_CACHING = (
        os.getenv("ENABLE_RESPONSE_CACHING", "False").lower() == "true"
    )
    RESPONSE_CACHE_TTL_SECONDS = int(
        os.getenv("RESPONSE_CACHE_TTL_SECONDS", "3600")
    )  # 1 hour

    # Dashboard optimization configuration
    ENABLE_DASHBOARD_CACHING = (
        os.getenv("ENABLE_DASHBOARD_CACHING", "False").lower() == "true"
    )
    DASHBOARD_CACHE_TTL_SECONDS = int(
        os.getenv("DASHBOARD_CACHE_TTL_SECONDS", "600")
    )  # 10 minutes
    DASHBOARD_MAX_POINTS = int(
        os.getenv("DASHBOARD_MAX_POINTS", "500")
    )  # Max data points in charts

    def as_dict(self) -> Dict[str, Union[str, int, bool, List[int]]]:
        """Return configuration as a dictionary."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("__") and not callable(value)
        }
