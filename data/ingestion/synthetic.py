"""Synthetic data generator for the supply chain forecaster."""

import datetime
import random
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from data.ingestion.base import DataIngestionBase
from utils import get_logger

logger = get_logger(__name__)


class SyntheticDataGenerator(DataIngestionBase):
    """Class for generating synthetic supply chain data."""

    def __init__(
        self,
        source_name: str = "synthetic",
        target_dir: Optional[Union[str, datetime.datetime]] = None,
        file_format: str = "parquet",
    ):
        """
        Initialize the synthetic data generator.
        
        Args:
            source_name: Name of the data source.
            target_dir: Directory to save the generated data.
            file_format: Format to save the generated data (csv or parquet).
        """
        super().__init__(source_name, target_dir, file_format)
        logger.info("Synthetic data generator initialized")

    def extract(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
        frequency: str = "D",
        n_products: int = 10,
        n_locations: int = 5,
        seed: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate synthetic supply chain data.
        
        Args:
            start_date: Start date for the generated data.
            end_date: End date for the generated data.
            frequency: Frequency of the generated data (D for daily, W for weekly, etc.).
            n_products: Number of products to generate.
            n_locations: Number of locations to generate.
            seed: Random seed for reproducibility.
            **kwargs: Additional keyword arguments.
        
        Returns:
            DataFrame containing the generated data.
        """
        logger.info(
            f"Generating synthetic data from {start_date} to {end_date} "
            f"with {n_products} products and {n_locations} locations"
        )
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        n_dates = len(date_range)
        logger.debug(f"Generated {n_dates} dates from {start_date} to {end_date}")
        
        # Generate product and location IDs
        products = [f"Product_{i+1}" for i in range(n_products)]
        locations = [f"Location_{i+1}" for i in range(n_locations)]
        
        # Create combinations of dates, products, and locations
        dates = np.repeat(date_range, n_products * n_locations)
        product_ids = np.tile(np.repeat(products, n_locations), n_dates)
        location_ids = np.tile(locations, n_dates * n_products)
        
        # Generate demand with time-based patterns and randomness
        # Base demand with product-specific levels
        base_demand = np.random.uniform(50, 200, n_products)
        product_demand = np.repeat(base_demand, n_locations * n_dates)
        
        # Add location-specific factors
        location_factors = np.random.uniform(0.8, 1.2, n_locations)
        location_effect = np.tile(np.repeat(location_factors, 1), n_dates * n_products)
        
        # Add day-of-week effect (higher on weekends)
        day_of_week = dates.dayofweek
        weekend_effect = np.where(day_of_week >= 5, 1.3, 1.0)
        
        # Add monthly seasonality
        month = dates.month
        monthly_seasonality = 1.0 + 0.2 * np.sin(2 * np.pi * month / 12)
        
        # Add yearly trend (growing over time)
        days_since_start = (dates - pd.Timestamp(start_date)).days
        yearly_trend = 1.0 + days_since_start / 365 * 0.1
        
        # Add holidays effect (spikes around certain dates)
        # Simplified approach: increase demand near month ends
        day_of_month = dates.day
        days_in_month = pd.Series(dates).dt.days_in_month.values
        end_of_month_effect = 1.0 + 0.3 * np.exp(-0.5 * ((day_of_month - days_in_month) / 5) ** 2)
        
        # Combine all effects
        demand = (
            product_demand
            * location_effect
            * weekend_effect
            * monthly_seasonality
            * yearly_trend
            * end_of_month_effect
        )
        
        # Add random noise
        demand = demand * np.random.normal(1, 0.1, len(demand))
        
        # Ensure demand is non-negative and round to integers
        demand = np.maximum(0, np.round(demand)).astype(int)
        
        # Generate other supply chain metrics
        # Inventory levels (inversely related to demand)
        inventory = np.maximum(
            0,
            np.random.normal(
                200 - 0.5 * demand, 20, len(demand)
            )
        ).astype(int)
        
        # Lead times with variation by location
        base_lead_times = np.random.uniform(2, 7, n_locations)
        lead_times = np.maximum(
            1,
            np.random.normal(
                np.tile(np.repeat(base_lead_times, 1), n_dates * n_products),
                1,
                len(demand),
            ),
        ).astype(int)
        
        # Stock-outs (boolean indicator)
        stock_outs = (inventory < demand).astype(int)
        
        # Order quantities (based on demand and current inventory)
        reorder_point = demand * 1.5
        orders = np.where(inventory < reorder_point, 
                         np.maximum(reorder_point * 2 - inventory, 0), 
                         0).astype(int)
        
        # Cost per unit (varies by product)
        base_costs = np.random.uniform(5, 50, n_products)
        unit_costs = np.tile(np.repeat(base_costs, n_locations), n_dates)
        
        # Selling price (varies by product with markup)
        markups = np.random.uniform(1.3, 2.0, n_products)
        selling_prices = np.tile(np.repeat(base_costs * markups, n_locations), n_dates)
        
        # Total costs
        total_costs = orders * unit_costs
        
        # Revenue
        revenue = demand * selling_prices
        
        # Profit
        profit = revenue - total_costs
        
        # Create the dataframe
        df = pd.DataFrame({
            "date": dates,
            "product_id": product_ids,
            "location_id": location_ids,
            "demand": demand,
            "inventory": inventory,
            "lead_time": lead_times,
            "stock_out": stock_outs,
            "order_quantity": orders,
            "unit_cost": unit_costs,
            "selling_price": selling_prices,
            "total_cost": total_costs,
            "revenue": revenue,
            "profit": profit,
        })
        
        logger.info(f"Generated synthetic dataset with {len(df)} rows")
        
        return df