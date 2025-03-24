"""CSV data ingestion for the supply chain forecaster."""

import glob
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from data.ingestion.base import DataIngestionBase
from utils import get_logger

logger = get_logger(__name__)


class CSVDataIngestion(DataIngestionBase):
    """Class for ingesting data from CSV files."""

    def __init__(
        self,
        source_name: str,
        source_path: Union[str, Path],
        target_dir: Optional[Union[str, Path]] = None,
        file_format: str = "parquet",
    ):
        """
        Initialize the CSV data ingestion.

        Args:
            source_name: Name of the data source.
            source_path: Path to the CSV file or directory containing CSV files.
            target_dir: Directory to save the ingested data.
            file_format: Format to save the ingested data (csv or parquet).
        """
        super().__init__(source_name, target_dir, file_format)
        self.source_path = Path(source_path)

        if not self.source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {source_path}")

        logger.info(f"CSV data ingestion configured with source path: {source_path}")

    def extract(
        self,
        pattern: str = "*.csv",
        recursive: bool = False,
        read_options: Optional[Dict] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Extract data from CSV file(s).

        Args:
            pattern: File pattern to match if source_path is a directory.
            recursive: Whether to search recursively in subdirectories.
            read_options: Options to pass to pandas.read_csv.
            **kwargs: Additional keyword arguments.

        Returns:
            DataFrame containing the extracted data.
        """
        read_options = read_options or {}
        default_options = {
            "encoding": "utf-8",
            "low_memory": False,
        }
        # Merge default options with provided options
        options = {**default_options, **read_options}

        logger.info(
            f"Extracting CSV data with pattern: {pattern}, recursive: {recursive}"
        )

        if self.source_path.is_file():
            # Single file
            logger.info(f"Reading single CSV file: {self.source_path}")
            return pd.read_csv(self.source_path, **options)
        elif self.source_path.is_dir():
            # Directory of files
            if recursive:
                files = list(self.source_path.rglob(pattern))
            else:
                files = list(self.source_path.glob(pattern))

            if not files:
                logger.warning(f"No CSV files found matching pattern: {pattern}")
                return pd.DataFrame()

            logger.info(f"Found {len(files)} CSV files to process")

            # Read all files and concatenate
            dfs = []
            for file in files:
                logger.debug(f"Reading CSV file: {file}")
                try:
                    df = pd.read_csv(file, **options)
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Error reading file {file}: {str(e)}")

            if not dfs:
                logger.warning("No data extracted from any CSV files")
                return pd.DataFrame()

            return pd.concat(dfs, ignore_index=True)
        else:
            logger.error(
                f"Source path is neither a file nor a directory: {self.source_path}"
            )
            return pd.DataFrame()
