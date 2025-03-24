"""Feature engineering for the supply chain forecaster."""

from typing import Dict, List, Optional, Union

import pandas as pd

from config import config
from data.preprocessing.base import DataPreprocessorBase
from utils import (
    add_holiday_features,
    create_lag_features,
    create_rolling_features,
    create_time_features,
    get_logger,
)

logger = get_logger(__name__)


class FeatureEngineer(DataPreprocessorBase):
    """Class for creating features from supply chain data."""

    def __init__(
        self,
        input_dir=None,
        output_dir=None,
        file_format="parquet",
        date_column="date",
        target_column="demand",
    ):
        """
        Initialize the feature engineer.
        
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
            f"Feature engineer initialized with date column '{date_column}' "
            f"and target column '{target_column}'"
        )

    def process(
        self,
        data: pd.DataFrame,
        create_time_based_features: bool = True,
        create_lags: bool = True,
        lag_periods: Optional[List[int]] = None,
        create_rolling: bool = True,
        rolling_windows: Optional[List[int]] = None,
        rolling_functions: Optional[List[str]] = None,
        add_holidays: bool = True,
        holidays_country: str = "US",
        group_by_columns: Optional[List[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Create features from the input dataframe.
        
        Args:
            data: Input dataframe to process.
            create_time_based_features: Whether to create time-based features.
            create_lags: Whether to create lag features.
            lag_periods: List of lag periods to create.
            create_rolling: Whether to create rolling window features.
            rolling_windows: List of rolling window sizes.
            rolling_functions: List of functions to apply to rolling windows.
            add_holidays: Whether to add holiday features.
            holidays_country: Country code for holidays.
            group_by_columns: Columns to group by when creating features.
            **kwargs: Additional keyword arguments.
        
        Returns:
            Dataframe with additional features.
        """
        logger.info(f"Creating features for dataset with {len(data)} rows")
        
        # Create a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure date column is datetime type
        if self.date_column in df.columns:
            if not pd.api.types.is_datetime64_dtype(df[self.date_column]):
                logger.info(f"Converting {self.date_column} to datetime")
                df[self.date_column] = pd.to_datetime(df[self.date_column])
            
            # Set date as index for time-based features if not already
            if df.index.name != self.date_column:
                logger.info(f"Setting {self.date_column} as index for feature creation")
                df = df.set_index(self.date_column)
                date_as_index = True
            else:
                date_as_index = False
        else:
            logger.warning(f"Date column '{self.date_column}' not found in dataframe")
            date_as_index = False
        
        # Create time-based features
        if create_time_based_features and (self.date_column in df.columns or date_as_index):
            logger.info("Creating time-based features")
            df = create_time_features(df)
        
        # Add holiday features
        if add_holidays and config.INCLUDE_HOLIDAYS and date_as_index:
            logger.info(f"Adding holiday features for country '{holidays_country}'")
            df = add_holiday_features(df, country=holidays_country)
        
        # Reset index if we set date as index
        if date_as_index:
            df = df.reset_index()
        
        # Create lag features for the target column
        if create_lags and self.target_column in df.columns:
            lag_periods = lag_periods or config.LAG_FEATURES
            logger.info(f"Creating lag features for '{self.target_column}' with periods {lag_periods}")
            
            if group_by_columns:
                for group_col in group_by_columns:
                    if group_col in df.columns:
                        logger.debug(f"Creating lag features grouped by '{group_col}'")
                        df = create_lag_features(
                            df, self.target_column, lag_periods, group_by=group_col
                        )
            else:
                df = create_lag_features(df, self.target_column, lag_periods)
        
        # Create rolling window features for the target column
        if create_rolling and self.target_column in df.columns:
            rolling_windows = rolling_windows or config.ROLLING_WINDOW_SIZES
            rolling_functions = rolling_functions or ["mean", "std", "min", "max"]
            
            logger.info(
                f"Creating rolling features for '{self.target_column}' with "
                f"windows {rolling_windows} and functions {rolling_functions}"
            )
            
            if group_by_columns:
                for group_col in group_by_columns:
                    if group_col in df.columns:
                        logger.debug(f"Creating rolling features grouped by '{group_col}'")
                        df = create_rolling_features(
                            df, self.target_column, rolling_windows, rolling_functions, group_by=group_col
                        )
            else:
                df = create_rolling_features(
                    df, self.target_column, rolling_windows, rolling_functions
                )
        
        # Create interaction features
        numeric_cols = df.select_dtypes(include=["number"]).columns
        target_related_cols = [
            col for col in numeric_cols
            if self.target_column in col and col != self.target_column
        ]
        
        logger.info(f"Found {len(target_related_cols)} target-related features")
        
        # Create simple ratio features for inventory management
        if "inventory" in df.columns and self.target_column in df.columns:
            logger.info("Creating inventory ratio features")
            df["inventory_to_demand_ratio"] = df["inventory"] / df[self.target_column].replace(0, 0.1)
            
            if "lead_time" in df.columns:
                df["coverage_days"] = df["inventory"] / df[self.target_column].replace(0, 0.1) * df["lead_time"]
        
        # Create economic features
        if "unit_cost" in df.columns and "selling_price" in df.columns:
            logger.info("Creating economic features")
            df["margin"] = df["selling_price"] - df["unit_cost"]
            df["margin_ratio"] = df["margin"] / df["selling_price"]
        
        logger.info(f"Feature engineering complete: {len(df)} rows, {len(df.columns)} columns")
        return df