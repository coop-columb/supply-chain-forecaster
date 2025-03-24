"""Error handling utilities for the supply chain forecaster."""

import functools
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

from loguru import logger

# Type variables for function decorators
F = TypeVar("F", bound=Callable[..., Any])
R = TypeVar("R")


class ApplicationError(Exception):
    """Base exception class for all application-specific errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the application error.

        Args:
            message: Human-readable error message.
            status_code: HTTP status code associated with this error.
            details: Additional error details for logging and debugging.
        """
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(ApplicationError):
    """Exception raised for validation errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the validation error."""
        super().__init__(message, status_code=400, details=details)


class NotFoundError(ApplicationError):
    """Exception raised when a requested resource is not found."""

    def __init__(self, resource_type: str, resource_id: Any):
        """
        Initialize the not found error.

        Args:
            resource_type: Type of resource that was not found.
            resource_id: Identifier of the resource.
        """
        message = f"{resource_type} with id '{resource_id}' not found"
        super().__init__(
            message,
            status_code=404,
            details={"resource_type": resource_type, "resource_id": resource_id},
        )


class ModelError(ApplicationError):
    """Exception raised for model-related errors."""

    def __init__(
        self, message: str, model_name: str, details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the model error.

        Args:
            message: Human-readable error message.
            model_name: Name of the model that caused the error.
            details: Additional error details.
        """
        error_details = details or {}
        error_details["model_name"] = model_name
        super().__init__(message, status_code=500, details=error_details)


def safe_execute(
    func: Callable[..., R],
    exception_map: Optional[Dict[Type[Exception], Type[ApplicationError]]] = None,
    default_error_cls: Type[ApplicationError] = ApplicationError,
    default_value: Optional[R] = None,
    log_exceptions: bool = True,
) -> R:
    """
    Execute a function safely, catching and mapping exceptions.

    Args:
        func: The function to execute.
        exception_map: A mapping from caught exception types to application error types.
        default_error_cls: The default error class to use for unmapped exceptions.
        default_value: The default value to return if an exception is caught and default_value is not None.
        log_exceptions: Whether to log caught exceptions.

    Returns:
        The result of the function call, or default_value if an exception is caught and default_value is not None.

    Raises:
        ApplicationError: If an exception is caught and default_value is None.
    """
    exception_map = exception_map or {}

    try:
        return func()
    except tuple(exception_map.keys()) as e:
        error_cls = exception_map[type(e)]
        if log_exceptions:
            logger.error(f"Caught exception {type(e).__name__}: {str(e)}")
            logger.debug(traceback.format_exc())

        if default_value is not None:
            return default_value

        raise error_cls(str(e))
    except Exception as e:
        if log_exceptions:
            logger.error(f"Caught unexpected exception {type(e).__name__}: {str(e)}")
            logger.debug(traceback.format_exc())

        if default_value is not None:
            return default_value

        raise default_error_cls(str(e))


def safe_decorator(
    exception_map: Optional[Dict[Type[Exception], Type[ApplicationError]]] = None,
    default_error_cls: Type[ApplicationError] = ApplicationError,
    default_value: Optional[Any] = None,
    log_exceptions: bool = True,
) -> Callable[[F], F]:
    """
    Decorator that wraps a function with safe_execute.

    Args:
        exception_map: A mapping from caught exception types to application error types.
        default_error_cls: The default error class to use for unmapped exceptions.
        default_value: The default value to return if an exception is caught and default_value is not None.
        log_exceptions: Whether to log caught exceptions.

    Returns:
        A decorator function.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return safe_execute(
                lambda: func(*args, **kwargs),
                exception_map=exception_map,
                default_error_cls=default_error_cls,
                default_value=default_value,
                log_exceptions=log_exceptions,
            )

        return cast(F, wrapper)

    return decorator
