"""Base data preprocessing class for the supply chain forecaster."""

import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from config import config
from utils import get_logger

logger = get_logger(__name__)


class DataPreprocessorBase(ABC):
    """Base class for all data preprocessing operations."""

    def __init__(
        self,
        input_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        file_format: str = "parquet",
    ):
        """
        Initialize the data preprocessor.

        Args:
            input_dir: Directory containing input data. If None, uses raw data directory.
            output_dir: Directory to save processed data. If None, uses processed data directory.
            file_format: Format to save processed data (csv or parquet).
        """
        self.input_dir = Path(input_dir) if input_dir else config.RAW_DATA_DIR
        self.output_dir = Path(output_dir) if output_dir else config.PROCESSED_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_format = file_format.lower()

        if self.file_format not in ["csv", "parquet"]:
            logger.warning(
                f"Unsupported file format: {file_format}, defaulting to parquet"
            )
            self.file_format = "parquet"

        logger.info(
            f"Initialized {self.__class__.__name__} with "
            f"input directory '{self.input_dir}' and "
            f"output directory '{self.output_dir}'"
        )

    @abstractmethod
    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Process the input dataframe.

        Args:
            data: Input dataframe to process.
            **kwargs: Additional keyword arguments for processing.

        Returns:
            Processed dataframe.
        """
        pass

    def save(self, df: pd.DataFrame, name: str, suffix: Optional[str] = None) -> Path:
        """
        Save the processed dataframe.

        Args:
            df: Dataframe to save.
            name: Base name for the saved file.
            suffix: Optional suffix to add to the filename.

        Returns:
            Path to the saved file.
        """
        if df.empty:
            logger.warning("DataFrame is empty, nothing to save")
            return None

        # Create filename with optional suffix and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix_str = f"_{suffix}" if suffix else ""
        filename = f"{name}{suffix_str}_processed_{timestamp}"

        # Save file in the specified format
        if self.file_format == "csv":
            file_path = self.output_dir / f"{filename}.csv"
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {len(df)} rows to {file_path}")
        else:  # parquet
            file_path = self.output_dir / f"{filename}.parquet"
            df.to_parquet(file_path, index=False)
            logger.info(f"Saved {len(df)} rows to {file_path}")

        return file_path

    def load(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from a file.

        Args:
            file_path: Path to the file to load.
            **kwargs: Additional keyword arguments for loading.

        Returns:
            Loaded dataframe.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading data from {file_path}")

        if file_path.suffix.lower() == ".csv":
            return pd.read_csv(file_path, **kwargs)
        elif file_path.suffix.lower() == ".parquet":
            return pd.read_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def run(
        self,
        input_data: Union[pd.DataFrame, str, Path],
        save_output: bool = True,
        output_name: Optional[str] = None,
        suffix: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, Path]]:
        """
        Run the preprocessor on input data and optionally save the result.

        Args:
            input_data: Input dataframe or path to input file.
            save_output: Whether to save the processed data.
            output_name: Name to use for the output file.
            suffix: Optional suffix to add to the output filename.
            **kwargs: Additional keyword arguments for processing.

        Returns:
            Dictionary containing the processed dataframe and path to the saved file.
        """
        # Load data if input is a file path
        if isinstance(input_data, (str, Path)):
            df = self.load(input_data)
        else:
            df = input_data.copy()

        logger.info(f"Running {self.__class__.__name__} on {len(df)} rows")

        # Process the data
        processed_df = self.process(df, **kwargs)

        result = {"data": processed_df}

        # Save the processed data if requested
        if save_output and not processed_df.empty:
            if output_name is None:
                if isinstance(input_data, (str, Path)):
                    output_name = Path(input_data).stem
                else:
                    output_name = "processed_data"

            file_path = self.save(processed_df, output_name, suffix)
            result["file_path"] = file_path

        result["rows"] = len(processed_df)
        result["columns"] = list(processed_df.columns)

        return result
