"""Caching utilities for the supply chain forecaster."""

import datetime
import functools
import hashlib
import json
import pickle
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config import config
from utils import get_logger

logger = get_logger(__name__)


def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Create a hash for a dataframe to use as a cache key.

    Args:
        df: Dataframe to hash.

    Returns:
        Hash string representing the dataframe state.
    """
    # Convert dataframe to a stable representation
    if isinstance(df, pd.DataFrame):
        # Include columns, index, and first/last few rows for efficiency
        sample_size = min(100, len(df))
        if len(df) > 0:
            sample = pd.concat([df.head(sample_size // 2), df.tail(sample_size // 2)])
            df_repr = {
                "columns": list(df.columns),
                "dtypes": {col: str(df[col].dtype) for col in df.columns},
                "shape": df.shape,
                "sample_values": sample.to_dict("records"),
                "index_sample": list(map(str, sample.index.tolist())),
            }
        else:
            df_repr = {"columns": list(df.columns), "shape": df.shape, "empty": True}
    elif isinstance(df, pd.Series):
        sample_size = min(100, len(df))
        if len(df) > 0:
            sample = pd.concat([df.head(sample_size // 2), df.tail(sample_size // 2)])
            df_repr = {
                "name": df.name,
                "dtype": str(df.dtype),
                "shape": df.shape,
                "sample_values": sample.tolist(),
                "index_sample": list(map(str, sample.index.tolist())),
            }
        else:
            df_repr = {"name": df.name, "shape": df.shape, "empty": True}
    else:
        df_repr = {"type": str(type(df)), "repr": str(df)[:1000]}

    # Convert to stable JSON string and hash
    stable_repr = json.dumps(df_repr, sort_keys=True)
    return hashlib.md5(stable_repr.encode()).hexdigest()


def generate_cache_key(func: Callable, args: Tuple, kwargs: Dict) -> str:
    """
    Generate a cache key for a function call.

    Args:
        func: Function being called.
        args: Positional arguments.
        kwargs: Keyword arguments.

    Returns:
        Cache key string.
    """
    # Start with function name and module
    key_parts = [func.__module__, func.__name__]

    # Add argument hashes
    for arg in args:
        if isinstance(arg, (pd.DataFrame, pd.Series)):
            key_parts.append(hash_dataframe(arg))
        elif isinstance(arg, np.ndarray):
            key_parts.append(hashlib.md5(arg.tobytes()).hexdigest())
        elif isinstance(arg, (str, int, float, bool, type(None))):
            key_parts.append(str(arg))
        else:
            # For other types, use their string representation
            try:
                arg_str = str(arg)[:100]  # Limit string length
                key_parts.append(hashlib.md5(arg_str.encode()).hexdigest())
            except:
                key_parts.append("unhashable")

    # Add keyword arguments
    kwargs_list = []
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (pd.DataFrame, pd.Series)):
            kwargs_list.append(f"{k}:{hash_dataframe(v)}")
        elif isinstance(v, np.ndarray):
            kwargs_list.append(f"{k}:{hashlib.md5(v.tobytes()).hexdigest()}")
        elif isinstance(v, (str, int, float, bool, type(None))):
            kwargs_list.append(f"{k}:{v}")
        else:
            # For other types, use their string representation
            try:
                v_str = str(v)[:100]  # Limit string length
                kwargs_list.append(f"{k}:{hashlib.md5(v_str.encode()).hexdigest()}")
            except:
                kwargs_list.append(f"{k}:unhashable")

    key_parts.extend(kwargs_list)

    # Join parts and hash
    return hashlib.md5("|".join(key_parts).encode()).hexdigest()


# Global prediction cache with expiration
_prediction_cache: Dict[str, Tuple[Any, datetime.datetime]] = {}
_DEFAULT_CACHE_TTL = datetime.timedelta(hours=1)


def memoize_with_expiry(ttl: Optional[datetime.timedelta] = None):
    """
    Decorator to memoize a function with expiry.

    Args:
        ttl: Time to live for cache entries. If None, uses default TTL.

    Returns:
        Decorated function.
    """
    ttl = ttl or _DEFAULT_CACHE_TTL

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if (
                not hasattr(config, "ENABLE_RESPONSE_CACHING")
                or not config.ENABLE_RESPONSE_CACHING
            ):
                return func(*args, **kwargs)

            # Generate cache key
            cache_key = generate_cache_key(func, args, kwargs)

            # Check cache
            now = datetime.datetime.now()
            if cache_key in _prediction_cache:
                result, timestamp = _prediction_cache[cache_key]
                if now - timestamp < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
                else:
                    logger.debug(f"Cache expired for {func.__name__}")

            # Compute and cache result
            result = func(*args, **kwargs)
            _prediction_cache[cache_key] = (result, now)

            # Clean up expired cache entries
            expired_keys = [
                k for k, (_, ts) in _prediction_cache.items() if now - ts > ttl
            ]
            for k in expired_keys:
                del _prediction_cache[k]

            return result

        return wrapper

    return decorator


def clear_prediction_cache():
    """Clear the prediction cache."""
    global _prediction_cache
    _prediction_cache.clear()
    logger.info("Prediction cache cleared")


def get_prediction_cache_stats() -> Dict[str, int]:
    """
    Get statistics about the prediction cache.

    Returns:
        Dictionary of cache statistics.
    """
    now = datetime.datetime.now()
    total_entries = len(_prediction_cache)
    active_entries = sum(
        1 for _, ts in _prediction_cache.values() if now - ts < _DEFAULT_CACHE_TTL
    )
    expired_entries = total_entries - active_entries

    return {
        "total_entries": total_entries,
        "active_entries": active_entries,
        "expired_entries": expired_entries,
    }


class ModelCache:
    """LRU cache for model instances."""

    def __init__(self, maxsize: int = 10):
        """
        Initialize the model cache.

        Args:
            maxsize: Maximum number of models to cache.
        """
        self.cache = {}  # Use a simple dict for now
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
        logger.info(f"Initialized model cache with maxsize={maxsize}")

    def get(self, model_id: str) -> Optional[Any]:
        """
        Get a model from the cache.

        Args:
            model_id: Model identifier.

        Returns:
            Model if found, None otherwise.
        """
        if (
            not hasattr(config, "ENABLE_MODEL_CACHING")
            or not config.ENABLE_MODEL_CACHING
        ):
            return None

        if model_id in self.cache:
            self.hits += 1
            # Move to end as recently used
            model = self.cache.pop(model_id)
            self.cache[model_id] = model
            logger.debug(f"Model cache hit for {model_id}")
            return model

        self.misses += 1
        logger.debug(f"Model cache miss for {model_id}")
        return None

    def put(self, model_id: str, model: Any) -> None:
        """
        Put a model into the cache.

        Args:
            model_id: Model identifier.
            model: Model instance.
        """
        if (
            not hasattr(config, "ENABLE_MODEL_CACHING")
            or not config.ENABLE_MODEL_CACHING
        ):
            return

        # If cache is full, remove the least recently used item
        if len(self.cache) >= self.maxsize:
            # Remove oldest (first) item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"Model cache eviction for {oldest_key}")

        # Add to cache
        self.cache[model_id] = model
        logger.debug(f"Model cached: {model_id}")

    def invalidate(self, model_id: str) -> None:
        """
        Invalidate a model in the cache.

        Args:
            model_id: Model identifier.
        """
        if model_id in self.cache:
            del self.cache[model_id]
            logger.debug(f"Model cache invalidated for {model_id}")

    def clear(self) -> None:
        """Clear the entire cache."""
        self.cache.clear()
        logger.info("Model cache cleared")

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the cache.

        Returns:
            Dictionary of cache statistics.
        """
        return {
            "size": len(self.cache),
            "maxsize": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": (
                self.hits / (self.hits + self.misses)
                if (self.hits + self.misses) > 0
                else 0
            ),
        }


# Global instance of ModelCache
model_cache = ModelCache(maxsize=getattr(config, "MODEL_CACHE_SIZE", 10))
