"""Base data ingestion class for the supply chain forecaster."""

import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from config import config
from utils import get_logger

logger = get_logger(__name__)


class DataIngestionBase(ABC):
    """Base class for all data ingestion methods."""

    def __init__(
        self,
        source_name: str,
        target_dir: Optional[Union[str, Path]] = None,
        file_format: str = "parquet",
    ):
        """
        Initialize the data ingestion class.
        
        Args:
            source_name: Name of the data source.
            target_dir: Directory to save the ingested data. If None, uses the configured raw data directory.
            file_format: Format to save the ingested data (csv or parquet).
        """
        self.source_name = source_name
        self.target_dir = Path(target_dir) if target_dir else config.RAW_DATA_DIR
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.file_format = file_format.lower()
        
        if self.file_format not in ["csv", "parquet"]:
            logger.warning(
                f"Unsupported file format: {file_format}, defaulting to parquet"
            )
            self.file_format = "parquet"
        
        logger.info(
            f"Initialized {self.__class__.__name__} for source '{source_name}'"
            f" with target directory '{self.target_dir}'"
        )

    @abstractmethod
    def extract(self, **kwargs) -> pd.DataFrame:
        """
        Extract data from the source.
        
        Returns:
            DataFrame containing the extracted data.
        """
        pass

    def save(self, df: pd.DataFrame, suffix: Optional[str] = None) -> Path:
        """
        Save the extracted data to the target directory.
        
        Args:
            df: DataFrame to save.
            suffix: Optional suffix to add to the filename.
        
        Returns:
            Path to the saved file.
        """
        if df.empty:
            logger.warning("DataFrame is empty, nothing to save")
            return None
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix_str = f"_{suffix}" if suffix else ""
        filename = f"{self.source_name}{suffix_str}_{timestamp}"
        
        # Save file in the specified format
        if self.file_format == "csv":
            file_path = self.target_dir / f"{filename}.csv"
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {len(df)} rows to {file_path}")
        else:  # parquet
            file_path = self.target_dir / f"{filename}.parquet"
            df.to_parquet(file_path, index=False)
            logger.info(f"Saved {len(df)} rows to {file_path}")
        
        return file_path

    def ingest(self, **kwargs) -> Dict[str, Union[pd.DataFrame, Path]]:
        """
        Extract data and save it to the target directory.
        
        Returns:
            Dictionary containing the extracted DataFrame and path to the saved file.
        """
        logger.info(f"Starting data ingestion from {self.source_name}")
        
        try:
            # Extract data
            df = self.extract(**kwargs)
            
            # Save data
            file_path = self.save(df, kwargs.get("suffix"))
            
            return {
                "data": df,
                "file_path": file_path,
                "source_name": self.source_name,
                "timestamp": datetime.now(),
                "rows": len(df),
                "columns": list(df.columns),
            }
        
        except Exception as e:
            logger.error(f"Error during data ingestion: {str(e)}")
            raise