"""Isolation Forest anomaly detection for the supply chain forecaster."""

import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from models.base import ModelBase, ModelRegistry
from utils import get_logger, safe_execute

logger = get_logger(__name__)


@ModelRegistry.register
class IsolationForestDetector(ModelBase):
    """Isolation Forest anomaly detection model."""

    def __init__(
        self,
        name: str = "IsolationForestDetector",
        n_estimators: int = 100,
        max_samples: Union[str, int] = "auto",
        contamination: Union[str, float] = "auto",
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = False,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize the Isolation Forest anomaly detector.

        Args:
            name: Name of the model.
            n_estimators: Number of isolation trees.
            max_samples: Number of samples to draw for each tree.
            contamination: Expected proportion of anomalies.
            max_features: Number of features to consider for each split.
            bootstrap: Whether to bootstrap samples.
            n_jobs: Number of jobs to run in parallel.
            random_state: Random seed.
            **kwargs: Additional parameters.
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.error(
                "scikit-learn not installed. Please install with pip install scikit-learn"
            )
            raise

        super().__init__(
            name=name,
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kwargs,
    ) -> "IsolationForestDetector":
        """
        Fit the Isolation Forest model to the data.

        Args:
            X: Feature dataframe.
            y: Ignored (kept for API consistency).
            **kwargs: Additional fitting parameters.

        Returns:
            Self for method chaining.
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.error(
                "scikit-learn not installed. Please install with pip install scikit-learn"
            )
            raise

        logger.info(f"Fitting Isolation Forest model {self.name}")

        # Store feature names
        self.features = list(X.columns)

        # Initialize model
        model = IsolationForest(
            n_estimators=self.params["n_estimators"],
            max_samples=self.params["max_samples"],
            contamination=self.params["contamination"],
            max_features=self.params["max_features"],
            bootstrap=self.params["bootstrap"],
            n_jobs=self.params["n_jobs"],
            random_state=self.params["random_state"],
            **kwargs,
        )

        # Fit the model
        model.fit(X)

        self.model = model

        # Update metadata
        self.metadata.update(
            {
                "fitted_at": datetime.datetime.now().isoformat(),
                "data_shape": X.shape,
                "feature_names": self.features,
            }
        )

        logger.info(f"Successfully fitted Isolation Forest model {self.name}")

        return self

    def predict(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None,
        return_scores: bool = False,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict anomalies in the data.

        Args:
            X: Feature dataframe.
            threshold: Threshold for anomaly score (overrides model's threshold).
            return_scores: Whether to return anomaly scores.
            **kwargs: Additional prediction parameters.

        Returns:
            Array of anomaly predictions (1 for normal, -1 for anomaly) and optionally scores.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted")

        logger.info(f"Predicting anomalies with Isolation Forest model {self.name}")

        # Get anomaly scores
        scores = self.model.decision_function(X)

        # Apply custom threshold if provided
        if threshold is not None:
            predictions = np.where(scores <= threshold, -1, 1)
        else:
            predictions = self.model.predict(X)

        if return_scores:
            return predictions, scores
        else:
            return predictions

    def get_anomalies(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None,
        include_scores: bool = True,
    ) -> pd.DataFrame:
        """
        Get anomalies with their scores.

        Args:
            X: Feature dataframe.
            threshold: Threshold for anomaly score (overrides model's threshold).
            include_scores: Whether to include anomaly scores in the result.

        Returns:
            Dataframe with anomalies and their scores.
        """
        predictions, scores = self.predict(X, threshold=threshold, return_scores=True)

        # Create result dataframe
        result = X.copy()
        result["is_anomaly"] = predictions == -1

        if include_scores:
            result["anomaly_score"] = (
                -scores
            )  # Negate for easier interpretation (higher = more anomalous)

        # Filter to only anomalies
        anomalies = result[result["is_anomaly"]].copy()

        logger.info(f"Found {len(anomalies)} anomalies out of {len(X)} points")

        return anomalies
