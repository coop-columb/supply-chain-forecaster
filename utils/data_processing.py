"""Data processing utilities for the supply chain forecaster."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from utils.logging import get_logger

logger = get_logger(__name__)


def detect_outliers(
    series: pd.Series, method: str = "zscore", threshold: float = 3.0
) -> pd.Series:
    """
    Detect outliers in a time series.

    Args:
        series: The time series data.
        method: The outlier detection method (zscore, iqr, or mad).
        threshold: The threshold for outlier detection.

    Returns:
        A boolean series where True indicates an outlier.
    """
    logger.debug(f"Detecting outliers using {method} method with threshold {threshold}")

    if method == "zscore":
        z_scores = np.abs(stats.zscore(series.fillna(series.mean())))
        return pd.Series(z_scores > threshold, index=series.index)

    elif method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (series < lower_bound) | (series > upper_bound)

    elif method == "mad":
        median = series.median()
        mad = np.median(np.abs(series - median))
        modified_z_scores = 0.6745 * np.abs(series - median) / mad
        return modified_z_scores > threshold

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def handle_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "zscore",
    threshold: float = 3.0,
    strategy: str = "clip",
) -> pd.DataFrame:
    """
    Handle outliers in a dataframe column.

    Args:
        df: The dataframe containing the data.
        column: The column to check for outliers.
        method: The outlier detection method.
        threshold: The threshold for outlier detection.
        strategy: The strategy to handle outliers (clip, remove, or replace).

    Returns:
        A dataframe with outliers handled.
    """
    logger.info(f"Handling outliers in column '{column}' with strategy '{strategy}'")

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Detect outliers
    outliers = detect_outliers(result_df[column], method, threshold)
    outlier_count = outliers.sum()

    logger.info(f"Detected {outlier_count} outliers out of {len(df)} data points")

    if outlier_count == 0:
        return result_df

    if strategy == "remove":
        result_df = result_df.loc[~outliers]
        logger.debug(f"Removed {outlier_count} outliers")

    elif strategy == "clip":
        if method == "zscore":
            z_scores = stats.zscore(result_df[column].fillna(result_df[column].mean()))
            result_df.loc[outliers, column] = (
                result_df[column].mean()
                + np.sign(z_scores[outliers]) * threshold * result_df[column].std()
            )

        elif method == "iqr":
            q1 = result_df[column].quantile(0.25)
            q3 = result_df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            result_df.loc[result_df[column] < lower_bound, column] = lower_bound
            result_df.loc[result_df[column] > upper_bound, column] = upper_bound

        elif method == "mad":
            median = result_df[column].median()
            mad = np.median(np.abs(result_df[column] - median))
            lower_bound = median - threshold * mad
            upper_bound = median + threshold * mad
            result_df.loc[result_df[column] < lower_bound, column] = lower_bound
            result_df.loc[result_df[column] > upper_bound, column] = upper_bound

        logger.debug(f"Clipped {outlier_count} outliers")

    elif strategy == "replace":
        # Replace with median for non-time series data
        if not isinstance(result_df.index, pd.DatetimeIndex):
            result_df.loc[outliers, column] = result_df[column].median()
            logger.debug(f"Replaced {outlier_count} outliers with median")
        else:
            # For time series, try to use linear interpolation
            original_values = result_df.loc[outliers, column].copy()
            result_df.loc[outliers, column] = np.nan
            result_df[column] = result_df[column].interpolate(method="linear")

            # If interpolation doesn't work (e.g., outliers at the edges), use ffill/bfill
            if result_df[column].isna().any():
                result_df[column] = (
                    result_df[column].fillna(method="ffill").fillna(method="bfill")
                )

            logger.debug(f"Replaced {outlier_count} outliers using interpolation")

    else:
        raise ValueError(f"Unknown outlier handling strategy: {strategy}")

    return result_df


def impute_missing_values(
    df: pd.DataFrame, column: str, method: str = "linear"
) -> pd.DataFrame:
    """
    Impute missing values in a dataframe column.

    Args:
        df: The dataframe containing the data.
        column: The column to impute.
        method: The imputation method (linear, mean, median, mode, ffill, bfill).

    Returns:
        A dataframe with missing values imputed.
    """
    logger.info(f"Imputing missing values in column '{column}' with method '{method}'")

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    missing_count = result_df[column].isna().sum()
    if missing_count == 0:
        logger.debug(f"No missing values found in column '{column}'")
        return result_df

    logger.info(f"Imputing {missing_count} missing values out of {len(df)} data points")

    if method == "linear":
        result_df[column] = result_df[column].interpolate(method="linear")
        # Handle edge cases
        result_df[column] = (
            result_df[column].fillna(method="ffill").fillna(method="bfill")
        )

    elif method == "time":
        if not isinstance(result_df.index, pd.DatetimeIndex):
            logger.warning(
                "Using time method for non-datetime index, falling back to linear"
            )
            result_df[column] = result_df[column].interpolate(method="linear")
        else:
            result_df[column] = result_df[column].interpolate(method="time")
        # Handle edge cases
        result_df[column] = (
            result_df[column].fillna(method="ffill").fillna(method="bfill")
        )

    elif method == "mean":
        result_df[column] = result_df[column].fillna(result_df[column].mean())

    elif method == "median":
        result_df[column] = result_df[column].fillna(result_df[column].median())

    elif method == "mode":
        result_df[column] = result_df[column].fillna(result_df[column].mode()[0])

    elif method == "ffill":
        result_df[column] = result_df[column].fillna(method="ffill")
        # In case there are NaNs at the beginning
        result_df[column] = result_df[column].fillna(method="bfill")

    elif method == "bfill":
        result_df[column] = result_df[column].fillna(method="bfill")
        # In case there are NaNs at the end
        result_df[column] = result_df[column].fillna(method="ffill")

    else:
        raise ValueError(f"Unknown imputation method: {method}")

    return result_df


def create_time_features(df: pd.DataFrame, date_column: str = None) -> pd.DataFrame:
    """
    Create time-based features from a date column or index.

    Args:
        df: The dataframe containing the data.
        date_column: The column containing dates. If None, uses the index.

    Returns:
        A dataframe with additional time-based features.
    """
    logger.info(f"Creating time features from '{date_column or 'index'}'")

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Get the date series
    if date_column is not None:
        if date_column not in result_df.columns:
            raise ValueError(f"Date column '{date_column}' not found in dataframe")
        date_series = pd.to_datetime(result_df[date_column])
    else:
        if not isinstance(result_df.index, pd.DatetimeIndex):
            raise ValueError(
                "Dataframe index is not a DatetimeIndex and no date column provided"
            )
        date_series = result_df.index

    # Extract basic time components
    result_df["year"] = date_series.year
    result_df["month"] = date_series.month
    result_df["day"] = date_series.day
    result_df["dayofweek"] = date_series.dayofweek
    result_df["quarter"] = date_series.quarter
    result_df["dayofyear"] = date_series.dayofyear
    result_df["weekofyear"] = date_series.isocalendar().week

    # Add cyclical features for day of week, month, and day of year
    result_df["dayofweek_sin"] = np.sin(2 * np.pi * date_series.dayofweek / 7)
    result_df["dayofweek_cos"] = np.cos(2 * np.pi * date_series.dayofweek / 7)
    result_df["month_sin"] = np.sin(2 * np.pi * date_series.month / 12)
    result_df["month_cos"] = np.cos(2 * np.pi * date_series.month / 12)
    result_df["dayofyear_sin"] = np.sin(2 * np.pi * date_series.dayofyear / 366)
    result_df["dayofyear_cos"] = np.cos(2 * np.pi * date_series.dayofyear / 366)

    # Add is_weekend flag
    result_df["is_weekend"] = (date_series.dayofweek >= 5).astype(int)

    # Add business quarter start/end flags
    result_df["is_quarter_start"] = date_series.is_quarter_start.astype(int)
    result_df["is_quarter_end"] = date_series.is_quarter_end.astype(int)
    result_df["is_month_start"] = date_series.is_month_start.astype(int)
    result_df["is_month_end"] = date_series.is_month_end.astype(int)

    return result_df


def create_lag_features(
    df: pd.DataFrame, column: str, lags: List[int], group_by: Optional[str] = None
) -> pd.DataFrame:
    """
    Create lag features for a column.

    Args:
        df: The dataframe containing the data.
        column: The column to create lags for.
        lags: A list of lag periods.
        group_by: An optional column to group by before creating lags.

    Returns:
        A dataframe with additional lag features.
    """
    logger.info(f"Creating lag features for column '{column}' with lags {lags}")

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    if group_by is not None:
        for lag in lags:
            result_df[f"{column}_lag_{lag}"] = result_df.groupby(group_by)[
                column
            ].shift(lag)
    else:
        for lag in lags:
            result_df[f"{column}_lag_{lag}"] = result_df[column].shift(lag)

    return result_df


def create_rolling_features(
    df: pd.DataFrame,
    column: str,
    windows: List[int],
    functions: List[str] = ["mean", "std", "min", "max"],
    group_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create rolling window features for a column.

    Args:
        df: The dataframe containing the data.
        column: The column to create rolling features for.
        windows: A list of window sizes.
        functions: A list of functions to apply (mean, std, min, max, median, sum).
        group_by: An optional column to group by before creating rolling features.

    Returns:
        A dataframe with additional rolling window features.
    """
    logger.info(
        f"Creating rolling features for column '{column}' with windows {windows}"
    )

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    for window in windows:
        if group_by is not None:
            grouped = result_df.groupby(group_by)[column]

            for func in functions:
                if func == "mean":
                    result_df[f"{column}_rolling_{window}_{func}"] = grouped.transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                elif func == "std":
                    result_df[f"{column}_rolling_{window}_{func}"] = grouped.transform(
                        lambda x: x.rolling(window, min_periods=1).std()
                    )
                elif func == "min":
                    result_df[f"{column}_rolling_{window}_{func}"] = grouped.transform(
                        lambda x: x.rolling(window, min_periods=1).min()
                    )
                elif func == "max":
                    result_df[f"{column}_rolling_{window}_{func}"] = grouped.transform(
                        lambda x: x.rolling(window, min_periods=1).max()
                    )
                elif func == "median":
                    result_df[f"{column}_rolling_{window}_{func}"] = grouped.transform(
                        lambda x: x.rolling(window, min_periods=1).median()
                    )
                elif func == "sum":
                    result_df[f"{column}_rolling_{window}_{func}"] = grouped.transform(
                        lambda x: x.rolling(window, min_periods=1).sum()
                    )
                else:
                    logger.warning(f"Unknown rolling function: {func}, skipping")
        else:
            for func in functions:
                if func == "mean":
                    result_df[f"{column}_rolling_{window}_{func}"] = (
                        result_df[column].rolling(window, min_periods=1).mean()
                    )
                elif func == "std":
                    result_df[f"{column}_rolling_{window}_{func}"] = (
                        result_df[column].rolling(window, min_periods=1).std()
                    )
                elif func == "min":
                    result_df[f"{column}_rolling_{window}_{func}"] = (
                        result_df[column].rolling(window, min_periods=1).min()
                    )
                elif func == "max":
                    result_df[f"{column}_rolling_{window}_{func}"] = (
                        result_df[column].rolling(window, min_periods=1).max()
                    )
                elif func == "median":
                    result_df[f"{column}_rolling_{window}_{func}"] = (
                        result_df[column].rolling(window, min_periods=1).median()
                    )
                elif func == "sum":
                    result_df[f"{column}_rolling_{window}_{func}"] = (
                        result_df[column].rolling(window, min_periods=1).sum()
                    )
                else:
                    logger.warning(f"Unknown rolling function: {func}, skipping")

    return result_df


def add_holiday_features(df: pd.DataFrame, country: str = "US") -> pd.DataFrame:
    """
    Add holiday features to a dataframe with a datetime index.

    Args:
        df: The dataframe containing the data (must have a DatetimeIndex).
        country: The country code for holidays.

    Returns:
        A dataframe with additional holiday features.
    """
    try:
        from pandas.tseries.holiday import USFederalHolidayCalendar
        from pandas.tseries.offsets import CustomBusinessDay
    except ImportError:
        logger.warning(
            "pandas.tseries.holiday not available, skipping holiday features"
        )
        return df

    logger.info(f"Adding holiday features for country '{country}'")

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    if not isinstance(result_df.index, pd.DatetimeIndex):
        logger.warning(
            "Dataframe index is not a DatetimeIndex, skipping holiday features"
        )
        return result_df

    if country == "US":
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=result_df.index.min(), end=result_df.index.max())
        result_df["is_holiday"] = result_df.index.isin(holidays).astype(int)

        # Add days before/after holiday
        result_df["is_day_before_holiday"] = result_df.index.isin(
            holidays - pd.Timedelta(days=1)
        ).astype(int)
        result_df["is_day_after_holiday"] = result_df.index.isin(
            holidays + pd.Timedelta(days=1)
        ).astype(int)

        # Add business day features
        bday = CustomBusinessDay(calendar=cal)
        result_df["is_business_day"] = result_df.index.map(
            lambda x: x.dayofweek < 5 and x not in holidays
        ).astype(int)

        # Add specific US holidays
        for holiday_date, holiday_name in zip(holidays, cal.holiday_names):
            mask = result_df.index == holiday_date
            if mask.any():
                holiday_col_name = (
                    f"is_{holiday_name.lower().replace(' ', '_').replace('-', '_')}"
                )
                result_df[holiday_col_name] = 0
                result_df.loc[mask, holiday_col_name] = 1
    else:
        logger.warning(
            f"Holiday features for country '{country}' not implemented, using US instead"
        )
        return add_holiday_features(df, country="US")

    return result_df
