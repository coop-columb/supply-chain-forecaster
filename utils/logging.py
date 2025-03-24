"""Logging utility for the supply chain forecaster."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

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


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
) -> None:
    """
    Set up the logger for consistent logging across the application.

    Args:
        log_level: The minimum log level to capture.
        log_file: Optional path to a log file. If None, logs to stdout only.
        rotation: When to rotate the log file (size or time).
        retention: How long to keep log files.
    """
    # Remove any existing handlers
    logger.remove()

    # Configure loguru's logger
    logger.configure(
        handlers=[
            {"sink": sys.stderr, "level": log_level, "colorize": True},
        ]
    )

    # Add a file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    # Intercept standard library logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Explicitly configure loggers from libraries
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


# Convenience function to get logger from anywhere in the codebase
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
