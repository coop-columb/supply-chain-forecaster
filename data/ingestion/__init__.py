"""Data ingestion module for the supply chain forecaster."""

from data.ingestion.base import DataIngestionBase
from data.ingestion.csv_ingestion import CSVDataIngestion
from data.ingestion.db_ingestion import DatabaseDataIngestion
from data.ingestion.synthetic import SyntheticDataGenerator

__all__ = [
    "DataIngestionBase",
    "CSVDataIngestion",
    "DatabaseDataIngestion",
    "SyntheticDataGenerator",
]