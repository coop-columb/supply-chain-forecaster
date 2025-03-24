"""Development configuration for the supply chain forecaster."""

from config.base_config import BaseConfig


class DevConfig(BaseConfig):
    """Development-specific configuration settings."""

    # API/dashboard configuration
    API_DEBUG = True
    DASHBOARD_DEBUG = True

    # Enable more verbose logging
    LOG_LEVEL = "DEBUG"

    # Smaller dataset for faster development
    DEV_SAMPLE_SIZE = 1000

    # Enable feature flags for testing new features
    FEATURE_FLAG_ANOMALY_DETECTION = True
    FEATURE_FLAG_INTERACTIVE_VISUALIZATIONS = True
    FEATURE_FLAG_SCENARIO_PLANNING = True

    # Development mode settings
    USE_SYNTHETIC_DATA = True
    SKIP_HEAVY_PREPROCESSING = True
    QUICK_TRAIN_ITERATIONS = 10
