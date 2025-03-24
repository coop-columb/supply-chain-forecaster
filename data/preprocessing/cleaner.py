"""Data cleaning utilities for the supply chain forecaster."""

from datetime import datetime
from typing import Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

from config import config
from data.preprocessing.base import DataPreprocessorBase
from utils import (
    add_holiday_features,
    detect_outliers,
    get_logger,
    handle_outliers,
    impute_missing_values,
)

logger = get_logger(__name__)


class DataCleaner(DataPreprocessorBase):
    """Class for cleaning and preprocessing supply chain data."""

    def __init__(
        self,
        input_dir=None,
        output_dir=None,
        file_format="parquet",
        date_column="date",
        target_column="demand",
    ):
        """
        Initialize the data cleaner.

        Args:
            input_dir: Directory containing input data.
            output_dir: Directory to save processed data.
            file_format: Format to save processed data.
            date_column: Column containing date information.
            target_column: Target column for forecasting.
        """
        super().__init__(input_dir, output_dir, file_format)
        self.date_column = date_column
        self.target_column = target_column
        logger.info(
            f"Data cleaner initialized with date column '{date_column}' "
            f"and target column '{target_column}'"
        )

    def process(
        self,
        data: pd.DataFrame,
        handle_duplicate_dates: bool = True,
        handle_missing_values: bool = True,
        handle_outliers_in_target: bool = True,
        convert_dates: bool = True,
        categorical_columns: Optional[List[str]] = None,
        drop_columns: Optional[List[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Clean and preprocess the input dataframe.

        Args:
            data: Input dataframe to clean.
            handle_duplicate_dates: Whether to handle duplicate dates.
            handle_missing_values: Whether to handle missing values.
            handle_outliers_in_target: Whether to handle outliers in the target column.
            convert_dates: Whether to convert date columns to datetime.
            categorical_columns: List of categorical columns to encode.
            drop_columns: List of columns to drop.
            **kwargs: Additional keyword arguments.

        Returns:
            Cleaned dataframe.
        """
        logger.info(
            f"Cleaning dataset with {len(data)} rows and {len(data.columns)} columns"
        )

        # Create a copy to avoid modifying the original
        df = data.copy()

        # Convert date columns
        if convert_dates and self.date_column in df.columns:
            logger.info(f"Converting date column '{self.date_column}' to datetime")
            try:
                df[self.date_column] = pd.to_datetime(df[self.date_column])
            except Exception as e:
                logger.error(f"Error converting date column: {str(e)}")

        # Handle missing values
        if handle_missing_values:
            missing_count = df.isna().sum().sum()
            if missing_count > 0:
                logger.info(f"Handling {missing_count} missing values")

                # Handle missing dates if date column is present
                if self.date_column in df.columns:
                    # If the date is the index or we're asked to handle duplicate dates
                    if df.index.name == self.date_column or handle_duplicate_dates:
                        self._handle_date_issues(df)

                # Impute missing values in numeric columns
                numeric_cols = df.select_dtypes(include=["number"]).columns
                for col in numeric_cols:
                    if df[col].isna().any():
                        logger.debug(f"Imputing missing values in column '{col}'")
                        df = impute_missing_values(
                            df, col, method=config.IMPUTATION_METHOD
                        )

                # Handle missing values in categorical columns
                if categorical_columns:
                    for col in categorical_columns:
                        if col in df.columns and df[col].isna().any():
                            logger.debug(
                                f"Filling missing values in categorical column '{col}'"
                            )
                            df[col] = df[col].fillna("Unknown")

        # Handle outliers in target column
        if handle_outliers_in_target and self.target_column in df.columns:
            logger.info(f"Handling outliers in target column '{self.target_column}'")
            df = handle_outliers(
                df,
                self.target_column,
                method=config.OUTLIER_DETECTION_METHOD,
                threshold=config.OUTLIER_THRESHOLD,
                strategy="clip",
            )

        # Encode categorical variables if specified
        if categorical_columns:
            logger.info(f"Encoding {len(categorical_columns)} categorical columns")
            for col in categorical_columns:
                if col in df.columns:
                    # Check if column is already numeric
                    if df[col].dtype.kind not in "ifu":
                        logger.debug(f"Label encoding column '{col}'")
                        # Label encoding
                        df[f"{col}_encoded"] = pd.factorize(df[col])[0]

        # Drop specified columns
        if drop_columns:
            valid_drop_cols = [col for col in drop_columns if col in df.columns]
            if valid_drop_cols:
                logger.info(f"Dropping columns: {valid_drop_cols}")
                df = df.drop(columns=valid_drop_cols)

        logger.info(f"Cleaning complete: {len(df)} rows, {len(df.columns)} columns")
        return df

    def _handle_date_issues(self, df: pd.DataFrame) -> None:
        """
        Handle issues with date columns, including duplicates and missing dates.

        Args:
            df: Dataframe to process, modified in-place.
        """
        if self.date_column not in df.columns:
            logger.warning(f"Date column '{self.date_column}' not found in dataframe")
            return

        # Ensure date column is datetime type
        if not pd.api.types.is_datetime64_dtype(df[self.date_column]):
            df[self.date_column] = pd.to_datetime(df[self.date_column])

        # Check for duplicate dates
        duplicates = df.duplicated(subset=[self.date_column], keep=False)
        if duplicates.any():
            n_duplicates = duplicates.sum()
            logger.warning(
                f"Found {n_duplicates} duplicate dates in {self.date_column}"
            )

            # If we have other grouping columns, check for duplicates within groups
            group_cols = [
                col for col in df.columns if col.endswith("_id") or col.endswith("_ID")
            ]

            if group_cols:
                logger.info(f"Checking for duplicates within groups: {group_cols}")
                group_duplicates = df.duplicated(
                    subset=[self.date_column] + group_cols, keep=False
                )

                if group_duplicates.any():
                    logger.warning(
                        f"Found {group_duplicates.sum()} duplicate dates within groups"
                    )
                    # Aggregate duplicate rows within groups
                    numeric_cols = df.select_dtypes(include=["number"]).columns
                    df = (
                        df.groupby([self.date_column] + group_cols)[numeric_cols]
                        .mean()
                        .reset_index()
                    )
                    logger.info(
                        f"Aggregated duplicate dates within groups, new shape: {df.shape}"
                    )
            else:
                # Aggregate duplicate dates
                numeric_cols = df.select_dtypes(include=["number"]).columns
                df = df.groupby(self.date_column)[numeric_cols].mean().reset_index()
                logger.info(f"Aggregated duplicate dates, new shape: {df.shape}")

        # Check for missing dates if time series
        min_date = df[self.date_column].min()
        max_date = df[self.date_column].max()
        ideal_range = pd.date_range(start=min_date, end=max_date, freq="D")

        if len(ideal_range) != len(df):
            logger.warning(
                f"Date range from {min_date} to {max_date} should have {len(ideal_range)} dates, "
                f"but found {len(df)}"
            )

            # If we have grouping columns, we need to handle missing dates within each group
            group_cols = [
                col for col in df.columns if col.endswith("_id") or col.endswith("_ID")
            ]

            if group_cols:
                logger.info(f"Checking for missing dates within groups: {group_cols}")

                # Create a complete date range for each group
                all_groups_data = []

                for group_values, group_df in df.groupby(group_cols):
                    # Convert to tuple if single group column
                    if not isinstance(group_values, tuple):
                        group_values = (group_values,)

                    # Create ideal date range for this group
                    group_min_date = group_df[self.date_column].min()
                    group_max_date = group_df[self.date_column].max()
                    group_date_range = pd.date_range(
                        start=group_min_date, end=group_max_date, freq="D"
                    )

                    # Create a complete dataframe for this group
                    group_ideal_df = pd.DataFrame({self.date_column: group_date_range})

                    # Add group columns
                    for i, col in enumerate(group_cols):
                        group_ideal_df[col] = group_values[i]

                    # Merge with actual data
                    group_complete = pd.merge(
                        group_ideal_df,
                        group_df,
                        on=[self.date_column] + group_cols,
                        how="left",
                    )
                    all_groups_data.append(group_complete)

                # Combine all groups
                df_complete = pd.concat(all_groups_data, ignore_index=True)

                logger.info(
                    f"Created complete date range for all groups, new shape: {df_complete.shape}"
                )

                # Impute missing values
                numeric_cols = df_complete.select_dtypes(include=["number"]).columns
                for col in numeric_cols:
                    if df_complete[col].isna().any():
                        logger.debug(f"Imputing missing values in column '{col}'")
                        df_complete = impute_missing_values(
                            df_complete, col, method=config.IMPUTATION_METHOD
                        )

                df = df_complete
            else:
                # Create a complete date range dataframe
                df_ideal = pd.DataFrame({self.date_column: ideal_range})

                # Merge with actual data
                df_complete = pd.merge(df_ideal, df, on=self.date_column, how="left")

                logger.info(
                    f"Created complete date range, new shape: {df_complete.shape}"
                )

                # Impute missing values
                numeric_cols = df_complete.select_dtypes(include=["number"]).columns
                for col in numeric_cols:
                    if df_complete[col].isna().any():
                        logger.debug(f"Imputing missing values in column '{col}'")
                        df_complete = impute_missing_values(
                            df_complete, col, method=config.IMPUTATION_METHOD
                        )

                df = df_complete
