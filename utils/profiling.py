"""Performance profiling utilities for the supply chain forecaster."""

import cProfile
import functools
import io
import pstats
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from utils import get_logger

logger = get_logger(__name__)

# Store profiling results
profiling_results = {
    "api": {},  # API route timing
    "model": {},  # Model training and inference timing
    "dashboard": {},  # Dashboard component loading timing
}


@contextmanager
def profile_time(name: str, category: str = "api"):
    """
    Context manager for timing code execution.

    Args:
        name: Name of the operation being timed.
        category: Category for grouping timings (api, model, dashboard).

    Yields:
        None
    """
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        # Store result
        if category not in profiling_results:
            profiling_results[category] = {}

        if name not in profiling_results[category]:
            profiling_results[category][name] = []

        profiling_results[category][name].append(duration_ms)

        # Log the timing
        logger.info(f"PROFILING: {category}.{name} took {duration_ms:.2f}ms")


def timeit(category: str = "api"):
    """
    Decorator for timing function execution.

    Args:
        category: Category for grouping timings.

    Returns:
        Decorated function.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with profile_time(func.__name__, category):
                return func(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def profile_memory(name: str, category: str = "api"):
    """
    Context manager for memory profiling.

    Args:
        name: Name of the operation being profiled.
        category: Category for grouping profiles.

    Yields:
        None
    """
    try:
        import psutil

        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        yield
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before

        logger.info(
            f"MEMORY: {category}.{name} used {mem_used:.2f}MB (Total: {mem_after:.2f}MB)"
        )
    except ImportError:
        logger.warning("psutil not installed, memory profiling unavailable")
        yield


@contextmanager
def profile_cpu(name: str, category: str = "api"):
    """
    Context manager for CPU profiling using cProfile.

    Args:
        name: Name of the operation being profiled.
        category: Category for grouping profiles.

    Yields:
        None
    """
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        yield
    finally:
        profiler.disable()

        # Get stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(20)  # Print top 20 functions

        # Log results
        logger.info(f"CPU PROFILE: {category}.{name}\n{s.getvalue()}")


def get_profiling_stats() -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Get statistical summary of profiling results.

    Returns:
        Dictionary with profiling statistics by category and operation.
    """
    stats = {}

    for category, operations in profiling_results.items():
        stats[category] = {}
        for operation, timings in operations.items():
            if timings:
                timings_array = pd.Series(timings)
                stats[category][operation] = {
                    "count": len(timings),
                    "mean_ms": timings_array.mean(),
                    "median_ms": timings_array.median(),
                    "min_ms": timings_array.min(),
                    "max_ms": timings_array.max(),
                    "p95_ms": timings_array.quantile(0.95),
                    "p99_ms": timings_array.quantile(0.99),
                    "std_ms": timings_array.std(),
                }

    return stats


def reset_profiling_stats():
    """Reset all profiling statistics."""
    for category in profiling_results:
        profiling_results[category] = {}
