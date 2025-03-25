"""Statistical anomaly detection for the supply chain forecaster."""

import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from models.base import ModelBase, ModelRegistry
from utils import get_logger, safe_execute

logger = get_logger(__name__)


@ModelRegistry.register
class StatisticalDetector(ModelBase):
    """Statistical anomaly detection model."""

    def __init__(
        self,
        name: str = "StatisticalDetector",
        method: str = "zscore",
        threshold: float = 3.0,
        target_column: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the statistical anomaly detector.

        Args:
            name: Name of the model.
            method: Detection method (zscore, iqr, or mad).
            threshold: Threshold for anomaly detection.
            target_column: Target column for univariate detection.
            **kwargs: Additional parameters.
        """
        if method not in ["zscore", "iqr", "mad"]:
            raise ValueError(
                f"Unknown method: {method}. Must be one of: zscore, iqr, mad"
            )

        super().__init__(
            name=name,
            method=method,
            threshold=threshold,
            target_column=target_column,
            **kwargs,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kwargs,
    ) -> "StatisticalDetector":
        """
        Fit the statistical detector to the data.

        Args:
            X: Feature dataframe.
            y: Target series (optional).
            **kwargs: Additional fitting parameters.

        Returns:
            Self for method chaining.
        """
        logger.info(
            f"Fitting statistical detector {self.name} with method {self.params['method']}"
        )

        # Store feature names
        self.features = list(X.columns)

        # Set target column if provided
        if y is not None:
            self.target = y.name
        else:
            self.target = self.params["target_column"]

        # If no target column specified, we'll use all numeric columns
        if self.target is None:
            self.target_columns = X.select_dtypes(include=["number"]).columns.tolist()
            logger.info(
                f"No target column specified, using all numeric columns: {self.target_columns}"
            )
        else:
            if self.target not in X.columns and (y is None or y.name != self.target):
                raise ValueError(f"Target column '{self.target}' not found in data")
            self.target_columns = [self.target]

        # Calculate statistics for each column
        self.stats = {}

        for col in self.target_columns:
            if col in X.columns:
                series = X[col]
            elif y is not None and y.name == col:
                series = y
            else:
                continue

            col_stats = {
                "mean": series.mean(),
                "std": series.std(),
                "median": series.median(),
                "q1": series.quantile(0.25),
                "q3": series.quantile(0.75),
                "iqr": series.quantile(0.75) - series.quantile(0.25),
                "mad": stats.median_abs_deviation(series, scale=1),
            }

            self.stats[col] = col_stats

        self.model = self.stats  # Store stats as the model

        # Update metadata
        self.metadata.update(
            {
                "fitted_at": datetime.datetime.now().isoformat(),
                "data_shape": X.shape,
                "target_columns": self.target_columns,
                "method": self.params["method"],
                "threshold": self.params["threshold"],
            }
        )

        logger.info(f"Successfully fitted statistical detector {self.name}")

        return self

    def predict(
        self,
        X: pd.DataFrame,
        method: Optional[str] = None,
        threshold: Optional[float] = None,
        return_scores: bool = False,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict anomalies in the data.

        Args:
            X: Feature dataframe.
            method: Detection method (zscore, iqr, or mad).
            threshold: Threshold for anomaly detection.
            return_scores: Whether to return anomaly scores.
            **kwargs: Additional prediction parameters.

        Returns:
            Array of anomaly predictions (1 for normal, -1 for anomaly) and optionally scores.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted")

        method = method or self.params["method"]
        threshold = threshold or self.params["threshold"]

        logger.info(f"Predicting anomalies with statistical detector {self.name}")

        # Initialize arrays for results
        anomaly_flags = np.zeros(len(X), dtype=bool)
        anomaly_scores = np.zeros(len(X))

        # Detect anomalies for each column
        for col in self.target_columns:
            if col not in X.columns:
                logger.warning(f"Column '{col}' not found in data, skipping")
                continue

            if col not in self.stats:
                logger.warning(f"No statistics for column '{col}', skipping")
                continue

            series = X[col]
            stats = self.stats[col]

            if method == "zscore":
                # Z-score method
                scores = np.abs((series - stats["mean"]) / stats["std"])
                flags = scores > threshold

            elif method == "iqr":
                # IQR method
                lower_bound = stats["q1"] - threshold * stats["iqr"]
                upper_bound = stats["q3"] + threshold * stats["iqr"]
                flags = (series < lower_bound) | (series > upper_bound)

                # Calculate scores as distance from acceptable range, normalized by IQR
                scores = np.zeros(len(series))
                scores[series < lower_bound] = (
                    lower_bound - series[series < lower_bound]
                ) / stats["iqr"]
                scores[series > upper_bound] = (
                    series[series > upper_bound] - upper_bound
                ) / stats["iqr"]

            elif method == "mad":
                # MAD method
                scores = np.abs(series - stats["median"]) / stats["mad"]
                flags = scores > threshold

            else:
                raise ValueError(f"Unknown method: {method}")

            # Combine results (any column flagged as anomaly => entire row is anomaly)
            anomaly_flags = anomaly_flags | flags
            anomaly_scores = np.maximum(anomaly_scores, scores)

        # Convert boolean flags to 1/-1 format
        predictions = np.where(anomaly_flags, -1, 1)

        if return_scores:
            return predictions, anomaly_scores
        else:
            return predictions

    def get_anomalies(
        self,
        X: pd.DataFrame,
        method: Optional[str] = None,
        threshold: Optional[float] = None,
        include_scores: bool = True,
    ) -> pd.DataFrame:
        """
        Get anomalies with their scores.

        Args:
            X: Feature dataframe.
            method: Detection method (zscore, iqr, or mad).
            threshold: Threshold for anomaly detection.
            include_scores: Whether to include anomaly scores in the result.

        Returns:
            Dataframe with anomalies and their scores.
        """
        predictions, scores = self.predict(
            X, method=method, threshold=threshold, return_scores=True
        )

        # Create result dataframe
        result = X.copy()
        result["is_anomaly"] = predictions == -1

        if include_scores:
            result["anomaly_score"] = scores

        # Add individual column anomaly flags
        method = method or self.params["method"]
        threshold = threshold or self.params["threshold"]

        for col in self.target_columns:
            if col in X.columns and col in self.stats:
                series = X[col]
                stats = self.stats[col]

                col_anomaly = f"{col}_anomaly"
                col_score = f"{col}_score"

                if method == "zscore":
                    scores = np.abs((series - stats["mean"]) / stats["std"])
                    result[col_anomaly] = scores > threshold

                elif method == "iqr":
                    lower_bound = stats["q1"] - threshold * stats["iqr"]
                    upper_bound = stats["q3"] + threshold * stats["iqr"]
                    result[col_anomaly] = (series < lower_bound) | (
                        series > upper_bound
                    )

                    # Calculate scores
                    scores = np.zeros(len(series))
                    scores[series < lower_bound] = (
                        lower_bound - series[series < lower_bound]
                    ) / stats["iqr"]
                    scores[series > upper_bound] = (
                        series[series > upper_bound] - upper_bound
                    ) / stats["iqr"]

                elif method == "mad":
                    scores = np.abs(series - stats["median"]) / stats["mad"]
                    result[col_anomaly] = scores > threshold

                if include_scores:
                    result[col_score] = scores

        # Filter to only anomalies
        anomalies = result[result["is_anomaly"]].copy()

        logger.info(f"Found {len(anomalies)} anomalies out of {len(X)} points")

        return anomalies
