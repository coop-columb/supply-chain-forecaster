"""Production configuration for the supply chain forecaster."""

from pathlib import Path

from config.base_config import BaseConfig


class ProdConfig(BaseConfig):
    """Production-specific configuration settings."""

    # API/dashboard configuration
    API_DEBUG = False
    DASHBOARD_DEBUG = False

    # Standard logging level for production
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "json"
    LOG_FILE = "/var/log/supply-chain-forecaster/app.log"
    LOG_ROTATION = "100 MB"
    LOG_RETENTION = "14 days"
    LOG_JSON_FORMAT = True
    LOG_ENVIRONMENT = "production"

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
    PROMETHEUS_METRICS = True
    METRICS_ENDPOINT = "/metrics"
    
    # Logging and monitoring paths
    METRICS_EXPORT_PATH = "/var/log/supply-chain-forecaster/metrics"
    HEALTH_CHECK_PATH = "/health"
    READINESS_CHECK_PATH = "/health/readiness"
    LIVENESS_CHECK_PATH = "/health/liveness"
    
    # Request rate limiting
    ENABLE_RATE_LIMITING = True
    RATE_LIMIT_REQUESTS_PER_MINUTE = 60
    
    # Enable metrics collection
    COLLECT_PREDICTION_METRICS = True
    COLLECT_TRAINING_METRICS = True
    COLLECT_API_METRICS = True
    
    # Feature flags for production
    FEATURE_FLAG_ANOMALY_DETECTION = True
    FEATURE_FLAG_INTERACTIVE_VISUALIZATIONS = True
    FEATURE_FLAG_SCENARIO_PLANNING = False  # Not ready for production yet
    
    # Security settings for production
    ENABLE_AUTH = True
    ALLOWED_ORIGINS = ["https://app.example.com", "https://api.example.com"]
    CORS_ALLOW_CREDENTIALS = True
    SECURE_COOKIES = True
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    
    # Performance profiling - disabled by default in production
    # Enable only temporarily for debugging specific performance issues
    ENABLE_PROFILING = False
    PROFILING_SAMPLE_RATE = 0.01  # Only profile 1% of requests if enabled
    
    # Caching is enabled in production for better performance
    ENABLE_MODEL_CACHING = True
    MODEL_CACHE_SIZE = 20
    ENABLE_RESPONSE_CACHING = True
    RESPONSE_CACHE_TTL_SECONDS = 3600  # 1 hour
