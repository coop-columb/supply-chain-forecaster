"""Logging utility for the supply chain forecaster."""

import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger


class InterceptHandler(logging.Handler):
    """
    Intercept standard logging messages towards Loguru.

    This handler intercepts all standard library logging and redirects it
    through loguru, ensuring consistent logging throughout the application.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """
        Intercept log records and pass them to loguru.

        Args:
            record: The log record to intercept.
        """
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


class JSONSink:
    """
    Custom JSON sink for loguru that formats logs as JSON.
    
    This sink formats log records as JSON for better parsing by log
    management systems like ELK, Loki, or CloudWatch.
    """
    
    def __init__(self, sink, request_id_provider=None):
        """
        Initialize the JSON sink.
        
        Args:
            sink: The sink to write logs to (file or stream)
            request_id_provider: Function that returns the current request ID
        """
        self.sink = sink
        self.request_id_provider = request_id_provider
    
    def __call__(self, message):
        """
        Format and write the log message as JSON.
        
        Args:
            message: The loguru message object
        """
        record = message.record
        
        # Add request ID if available
        request_id = None
        if self.request_id_provider:
            try:
                request_id = self.request_id_provider()
            except Exception:
                pass
        
        # Create the JSON log entry
        log_entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "message": record["message"],
            "name": record["name"],
            "function": record["function"],
            "line": record["line"],
            "file": record["file"].path,
            "process": record["process"].id,
            "thread": record["thread"].id,
        }
        
        # Add request ID if available
        if request_id:
            log_entry["request_id"] = request_id
        
        # Add exception info if available
        if record["exception"]:
            log_entry["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": record["exception"].traceback,
            }
        
        # Add extra fields
        for key, value in record["extra"].items():
            log_entry[key] = value
        
        # Write the JSON log entry
        self.sink.write(json.dumps(log_entry) + "\n")
        self.sink.flush()


# Global request ID for distributed tracing
_request_id = None


def get_request_id() -> str:
    """
    Get the current request ID or generate a new one.
    
    Returns:
        The current request ID string.
    """
    global _request_id
    if _request_id is None:
        _request_id = str(uuid.uuid4())
    return _request_id


def set_request_id(request_id: str) -> None:
    """
    Set the current request ID for distributed tracing.
    
    Args:
        request_id: The request ID to set.
    """
    global _request_id
    _request_id = request_id


def reset_request_id() -> None:
    """Reset the request ID to None."""
    global _request_id
    _request_id = None


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
    json_format: bool = False,
    env: str = "development",
) -> None:
    """
    Set up the logger for consistent logging across the application.

    Args:
        log_level: The minimum log level to capture.
        log_file: Optional path to a log file. If None, logs to stdout only.
        rotation: When to rotate the log file (size or time).
        retention: How long to keep log files.
        json_format: Whether to output logs in JSON format (for production).
        env: The environment name (development, production, etc.)
    """
    # Remove any existing handlers
    logger.remove()
    
    # Add application metadata to all logs
    logger = logger.bind(
        application="supply-chain-forecaster",
        environment=env,
        version="0.1.0",
    )
    
    # Configure stderr handler
    if json_format:
        # Use JSON format for stderr in production
        logger.add(
            JSONSink(sys.stderr, request_id_provider=get_request_id),
            level=log_level,
        )
    else:
        # Use colorized format for development
        logger.add(
            sys.stderr,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            level=log_level,
            colorize=True,
        )

    # Add a file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if json_format:
            # JSON format for log files in production
            with open(log_path, "a") as log_file_handle:
                logger.add(
                    JSONSink(log_file_handle, request_id_provider=get_request_id),
                    level=log_level,
                    rotation=rotation,
                    retention=retention,
                    compression="zip",
                )
        else:
            # Standard format for log files in development
            logger.add(
                str(log_path),
                level=log_level,
                rotation=rotation,
                retention=retention,
                compression="zip",
            )

    # Intercept standard library logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Configure key libraries to use our logging
    for lib_logger in [
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "fastapi",
        "pandas",
        "matplotlib",
        "prophet",
        "tensorflow",
    ]:
        logging.getLogger(lib_logger).handlers = [InterceptHandler()]
        logging.getLogger(lib_logger).propagate = False

    logger.debug("Logging system initialized")


def get_logger(name: str = None):
    """
    Get a configured logger instance for a specific module.

    Args:
        name: The name of the module for which to get a logger.
              If None, returns the root logger.

    Returns:
        A configured logger instance.
    """
    if name:
        return logger.bind(name=name)
    return logger


def log_request(request_id: str, method: str, url: str, status_code: int, duration_ms: float, user_id: Optional[str] = None) -> None:
    """
    Log an API request with performance metrics.
    
    Args:
        request_id: The unique ID for the request
        method: HTTP method (GET, POST, etc.)
        url: The request URL
        status_code: The HTTP response status code
        duration_ms: Request duration in milliseconds
        user_id: Optional user ID
    """
    log_data = {
        "request_id": request_id,
        "method": method,
        "url": url,
        "status_code": status_code,
        "duration_ms": duration_ms,
    }
    
    if user_id:
        log_data["user_id"] = user_id
    
    get_logger("api.request").info(
        f"Request {method} {url} completed with status {status_code} in {duration_ms:.2f}ms",
        **log_data
    )


def log_model_prediction(model_name: str, input_shape: tuple, output_shape: tuple, duration_ms: float, request_id: Optional[str] = None) -> None:
    """
    Log a model prediction event with performance metrics.
    
    Args:
        model_name: The name of the model used
        input_shape: Shape of the input data
        output_shape: Shape of the output predictions
        duration_ms: Prediction duration in milliseconds
        request_id: Optional request ID for correlation
    """
    log_data = {
        "model_name": model_name,
        "input_shape": str(input_shape),
        "output_shape": str(output_shape),
        "duration_ms": duration_ms,
    }
    
    if request_id:
        log_data["request_id"] = request_id
    
    get_logger("models.prediction").info(
        f"Model {model_name} prediction completed in {duration_ms:.2f}ms",
        **log_data
    )