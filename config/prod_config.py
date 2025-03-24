"""Production configuration for the supply chain forecaster."""

from config.base_config import BaseConfig


class ProdConfig(BaseConfig):
    """Production-specific configuration settings."""
    
    # API/dashboard configuration
    API_DEBUG = False
    DASHBOARD_DEBUG = False
    
    # Standard logging level for production
    LOG_LEVEL = "INFO"
    
    # Production mode settings
    USE_SYNTHETIC_DATA = False
    SKIP_HEAVY_PREPROCESSING = False
    
    # Enable caching for faster responses
    ENABLE_RESPONSE_CACHING = True
    CACHE_EXPIRY_SECONDS = 3600
    
    # Monitoring settings
    ENABLE_PERFORMANCE_MONITORING = True
    ENABLE_ERROR_TRACKING = True
    MONITORING_SAMPLE_RATE = 0.1
    
    # Feature flags for production
    FEATURE_FLAG_ANOMALY_DETECTION = True
    FEATURE_FLAG_INTERACTIVE_VISUALIZATIONS = True
    FEATURE_FLAG_SCENARIO_PLANNING = False  # Not ready for production yet