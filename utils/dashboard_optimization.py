"""Dashboard optimization utilities for the supply chain forecaster."""

import datetime
import functools
import hashlib
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from config import config
from utils import get_logger
from utils.caching import generate_cache_key, hash_dataframe

logger = get_logger(__name__)

# Global dashboard component cache with expiration
_component_cache: Dict[str, Tuple[Any, datetime.datetime]] = {}
_DEFAULT_COMPONENT_CACHE_TTL = datetime.timedelta(minutes=10)


def memoize_component(ttl: Optional[datetime.timedelta] = None):
    """
    Decorator to memoize a dashboard component function with expiry.
    
    Args:
        ttl: Time to live for cache entries. If None, uses default TTL.
        
    Returns:
        Decorated function.
    """
    ttl = ttl or _DEFAULT_COMPONENT_CACHE_TTL
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not hasattr(config, "ENABLE_DASHBOARD_CACHING") or not config.ENABLE_DASHBOARD_CACHING:
                return func(*args, **kwargs)
            
            # Generate cache key
            cache_key = generate_cache_key(func, args, kwargs)
            
            # Check cache
            now = datetime.datetime.now()
            if cache_key in _component_cache:
                result, timestamp = _component_cache[cache_key]
                if now - timestamp < ttl:
                    logger.debug(f"Component cache hit for {func.__name__}")
                    return result
                else:
                    logger.debug(f"Component cache expired for {func.__name__}")
            
            # Compute and cache result
            result = func(*args, **kwargs)
            _component_cache[cache_key] = (result, now)
            
            # Clean up expired cache entries
            expired_keys = [
                k for k, (_, ts) in _component_cache.items() 
                if now - ts > ttl
            ]
            for k in expired_keys:
                del _component_cache[k]
            
            return result
        
        return wrapper
    
    return decorator


def clear_component_cache():
    """Clear the dashboard component cache."""
    global _component_cache
    _component_cache.clear()
    logger.info("Dashboard component cache cleared")


def get_component_cache_stats() -> Dict[str, int]:
    """
    Get statistics about the dashboard component cache.
    
    Returns:
        Dictionary of cache statistics.
    """
    now = datetime.datetime.now()
    total_entries = len(_component_cache)
    active_entries = sum(1 for _, ts in _component_cache.values() 
                         if now - ts < _DEFAULT_COMPONENT_CACHE_TTL)
    expired_entries = total_entries - active_entries
    
    return {
        "total_entries": total_entries,
        "active_entries": active_entries,
        "expired_entries": expired_entries,
    }


def downsample_timeseries(
    df: pd.DataFrame, 
    date_column: str, 
    value_columns: Union[str, List[str]],
    max_points: int = 500
) -> pd.DataFrame:
    """
    Downsample a time series DataFrame to a maximum number of points.
    
    Args:
        df: DataFrame containing the time series data.
        date_column: Name of the column containing date/time values.
        value_columns: Name of column(s) containing values to aggregate.
        max_points: Maximum number of points to include in the result.
        
    Returns:
        Downsampled DataFrame.
    """
    # Ensure date column is datetime type
    if not pd.api.types.is_datetime64_dtype(df[date_column]):
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
    
    # If DataFrame is already small enough, return it as is
    if len(df) <= max_points:
        return df
    
    # Ensure value_columns is a list
    if isinstance(value_columns, str):
        value_columns = [value_columns]
    
    # Calculate the appropriate frequency based on the date range and max_points
    date_range = df[date_column].max() - df[date_column].min()
    days = date_range.total_seconds() / (24 * 3600)
    
    # Choose appropriate frequency (hourly, daily, weekly, monthly)
    if days <= 2:  # Short range: hours
        freq = f"{max(1, int(48 / max_points))}H"
    elif days <= 60:  # Medium range: days
        freq = f"{max(1, int(days / max_points))}D"
    elif days <= 730:  # Long range: weeks
        freq = f"{max(1, int(days / (7 * max_points)))}W"
    else:  # Very long range: months
        freq = f"{max(1, int(days / (30 * max_points)))}M"
    
    # Set the date column as index
    temp_df = df.set_index(date_column)
    
    # Resample and aggregate
    resampled = temp_df[value_columns].resample(freq).mean()
    
    # Reset index to get the date column back
    result = resampled.reset_index()
    
    # If still too large, subsample (fallback method)
    if len(result) > max_points:
        indices = np.linspace(0, len(result) - 1, max_points, dtype=int)
        result = result.iloc[indices]
    
    return result


def optimize_plotly_figure(fig: go.Figure) -> go.Figure:
    """
    Optimize a Plotly figure for faster rendering.
    
    Args:
        fig: Plotly figure to optimize.
        
    Returns:
        Optimized figure.
    """
    # Apply performance optimizations
    for trace in fig.data:
        # For scatter traces, simplify if there are too many points
        if hasattr(trace, "x") and hasattr(trace, "y") and len(trace.x) > 1000:
            # Simplify by removing intermediate points that don't affect the visual trend
            indices = np.linspace(0, len(trace.x) - 1, 1000, dtype=int)
            trace.x = [trace.x[i] for i in indices]
            trace.y = [trace.y[i] for i in indices]
        
        # Set decimation for large datasets
        trace.hoverinfo = "none"  # Disable hover for better performance
    
    # Optimize layout
    fig.update_layout(
        # Disable animations
        transition_duration=0,
        # Optimize rendering
        autosize=True,
        # Disable unnecessary UI components
        modebar=dict(
            orientation='v',
            remove=['sendDataToCloud', 'lasso2d', 'select2d']
        ),
    )
    
    return fig


class ComponentTimer:
    """Context manager for timing component rendering."""
    
    def __init__(self, component_name: str):
        """
        Initialize the timer.
        
        Args:
            component_name: Name of the component being timed.
        """
        self.component_name = component_name
        self.start_time = None
    
    def __enter__(self):
        """Start the timer."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the timer and log the result."""
        if self.start_time is not None:
            elapsed_ms = (time.time() - self.start_time) * 1000
            logger.debug(
                f"Rendered {self.component_name} in {elapsed_ms:.2f}ms",
                component=self.component_name,
                duration_ms=elapsed_ms,
            )


def profile_component(component_name: str):
    """
    Decorator to profile a component's rendering time.
    
    Args:
        component_name: Name of the component being profiled.
        
    Returns:
        Decorated function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with ComponentTimer(component_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator