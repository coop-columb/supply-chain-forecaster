"""Database data ingestion for the supply chain forecaster."""

import os
from typing import Dict, List, Optional, Union

import pandas as pd

from data.ingestion.base import DataIngestionBase
from utils import get_logger

logger = get_logger(__name__)


class DatabaseDataIngestion(DataIngestionBase):
    """Class for ingesting data from databases."""

    def __init__(
        self,
        source_name: str,
        connection_string: str,
        target_dir: Optional[Union[str, os.PathLike]] = None,
        file_format: str = "parquet",
    ):
        """
        Initialize the database data ingestion.
        
        Args:
            source_name: Name of the data source.
            connection_string: Database connection string.
            target_dir: Directory to save the ingested data.
            file_format: Format to save the ingested data (csv or parquet).
        """
        super().__init__(source_name, target_dir, file_format)
        self.connection_string = connection_string
        logger.info(f"Database data ingestion configured for source: {source_name}")

    def extract(
        self, query: str, params: Optional[Dict] = None, **kwargs
    ) -> pd.DataFrame:
        """
        Extract data from a database using a SQL query.
        
        Args:
            query: SQL query to execute.
            params: Parameters for the SQL query.
            **kwargs: Additional keyword arguments.
        
        Returns:
            DataFrame containing the extracted data.
        """
        try:
            import sqlalchemy
            
            logger.info(f"Extracting data from database with query: {query[:100]}...")
            engine = sqlalchemy.create_engine(self.connection_string)
            
            with engine.connect() as connection:
                return pd.read_sql(query, connection, params=params)
        
        except ImportError:
            logger.error("SQLAlchemy not installed. Please install with pip install sqlalchemy")
            raise
        
        except Exception as e:
            logger.error(f"Error extracting data from database: {str(e)}")
            raise