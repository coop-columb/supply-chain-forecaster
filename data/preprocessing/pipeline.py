"""Preprocessing pipeline for the supply chain forecaster."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from config import config
from data.preprocessing.base import DataPreprocessorBase
from data.preprocessing.cleaner import DataCleaner
from data.preprocessing.feature_engineering import FeatureEngineer
from utils import get_logger, safe_execute

logger = get_logger(__name__)


class PreprocessingPipeline(DataPreprocessorBase):
    """Pipeline for preprocessing supply chain data."""

    def __init__(
        self,
        input_dir=None,
        output_dir=None,
        file_format="parquet",
        date_column="date",
        target_column="demand",
    ):
        """
        Initialize the preprocessing pipeline.

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

        # Initialize preprocessing steps
        self.cleaner = DataCleaner(
            input_dir=input_dir,
            output_dir=output_dir,
            file_format=file_format,
            date_column=date_column,
            target_column=target_column,
        )

        self.feature_engineer = FeatureEngineer(
            input_dir=input_dir,
            output_dir=output_dir,
            file_format=file_format,
            date_column=date_column,
            target_column=target_column,
        )

        logger.info("Preprocessing pipeline initialized")

    def process(
        self,
        data: pd.DataFrame,
        clean_data: bool = True,
        engineer_features: bool = True,
        clean_params: Optional[Dict] = None,
        feature_params: Optional[Dict] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Run the preprocessing pipeline on the input dataframe.

        Args:
            data: Input dataframe to process.
            clean_data: Whether to clean the data.
            engineer_features: Whether to engineer features.
            clean_params: Parameters for the data cleaning step.
            feature_params: Parameters for the feature engineering step.
            **kwargs: Additional keyword arguments.

        Returns:
            Processed dataframe.
        """
        logger.info(f"Running preprocessing pipeline on {len(data)} rows")

        # Create a copy to avoid modifying the original
        df = data.copy()

        # Data cleaning
        if clean_data:
            clean_params = clean_params or {}
            logger.info("Running data cleaning step")
            df = safe_execute(
                lambda: self.cleaner.process(df, **clean_params),
                default_value=df,
                log_exceptions=True,
            )

        # Feature engineering
        if engineer_features:
            feature_params = feature_params or {}
            logger.info("Running feature engineering step")
            df = safe_execute(
                lambda: self.feature_engineer.process(df, **feature_params),
                default_value=df,
                log_exceptions=True,
            )

        logger.info(
            f"Preprocessing complete: {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    def get_training_data(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = False,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """
        Split the data into training, validation, and test sets.

        Args:
            data: Input dataframe to split.
            train_ratio: Proportion of data to use for training.
            validation_ratio: Proportion of data to use for validation.
            test_ratio: Proportion of data to use for testing.
            shuffle: Whether to shuffle the data before splitting.
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary containing training, validation, and test sets.
        """
        logger.info(f"Splitting data into training, validation, and test sets")

        if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
            logger.warning(
                f"Split ratios do not sum to 1.0: "
                f"train={train_ratio}, validation={validation_ratio}, test={test_ratio}"
            )
            # Normalize ratios
            total = train_ratio + validation_ratio + test_ratio
            train_ratio /= total
            validation_ratio /= total
            test_ratio /= total
            logger.info(
                f"Normalized ratios: "
                f"train={train_ratio}, validation={validation_ratio}, test={test_ratio}"
            )

        # Create a copy to avoid modifying the original
        df = data.copy()

        # Check if date column exists for time-based split
        if self.date_column in df.columns and not shuffle:
            logger.info(f"Using time-based split with column '{self.date_column}'")

            # Ensure date column is datetime type
            if not pd.api.types.is_datetime64_dtype(df[self.date_column]):
                df[self.date_column] = pd.to_datetime(df[self.date_column])

            # Sort by date
            df = df.sort_values(by=self.date_column)

            # Calculate split indices
            n = len(df)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * validation_ratio)

            # Split data
            train_data = df.iloc[:train_end]
            val_data = df.iloc[train_end:val_end]
            test_data = df.iloc[val_end:]

            logger.info(
                f"Time-based split: "
                f"train={len(train_data)} rows ({train_data[self.date_column].min()} to {train_data[self.date_column].max()}), "
                f"validation={len(val_data)} rows ({val_data[self.date_column].min()} to {val_data[self.date_column].max()}), "
                f"test={len(test_data)} rows ({test_data[self.date_column].min()} to {test_data[self.date_column].max()})"
            )

        else:
            logger.info("Using random split")

            # Calculate split indices
            n = len(df)
            indices = np.arange(n)

            if shuffle:
                np.random.shuffle(indices)

            train_end = int(n * train_ratio)
            val_end = train_end + int(n * validation_ratio)

            # Split data
            train_data = df.iloc[indices[:train_end]]
            val_data = df.iloc[indices[train_end:val_end]]
            test_data = df.iloc[indices[val_end:]]

            logger.info(
                f"Random split: "
                f"train={len(train_data)} rows, "
                f"validation={len(val_data)} rows, "
                f"test={len(test_data)} rows"
            )

        return {
            "train": train_data,
            "validation": val_data,
            "test": test_data,
        }
