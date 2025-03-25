"""Exponential smoothing model for the supply chain forecaster."""

import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from models.base import ModelBase, ModelRegistry
from utils import get_logger, safe_execute

logger = get_logger(__name__)


@ModelRegistry.register
class ExponentialSmoothingModel(ModelBase):
    """Exponential smoothing model for time series forecasting."""

    def __init__(
        self,
        name: str = "ExponentialSmoothingModel",
        trend: Optional[str] = None,
        damped_trend: bool = False,
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        initialization_method: str = "estimated",
        **kwargs,
    ):
        """
        Initialize the exponential smoothing model.

        Args:
            name: Name of the model.
            trend: Type of trend component (None, 'add', or 'mul').
            damped_trend: Whether to damp the trend.
            seasonal: Type of seasonal component (None, 'add', or 'mul').
            seasonal_periods: Number of periods in a seasonal cycle.
            initialization_method: Method to initialize the model ('estimated' or 'heuristic').
            **kwargs: Additional model parameters.
        """
        try:
            import statsmodels
        except ImportError:
            logger.error(
                "statsmodels not installed. Please install with pip install statsmodels"
            )
            raise

        super().__init__(
            name=name,
            trend=trend,
            damped_trend=damped_trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method=initialization_method,
            **kwargs,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        date_col: str = None,
        detect_seasonality: bool = True,
        **kwargs,
    ) -> "ExponentialSmoothingModel":
        """
        Fit the exponential smoothing model to the data.

        Args:
            X: Feature dataframe.
            y: Target series.
            date_col: Column containing dates. If None, uses the index.
            detect_seasonality: Whether to automatically detect seasonality.
            **kwargs: Additional fitting parameters.

        Returns:
            Self for method chaining.
        """
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError:
            logger.error(
                "statsmodels not installed. Please install with pip install statsmodels"
            )
            raise

        logger.info(f"Fitting exponential smoothing model {self.name}")

        # Store feature names and target name
        self.features = list(X.columns)
        self.target = y.name if y.name else "target"

        # Get date series
        if date_col is not None and date_col in X.columns:
            dates = X[date_col]
            y.index = pd.DatetimeIndex(dates)
        else:
            if not isinstance(y.index, pd.DatetimeIndex) and isinstance(
                X.index, pd.DatetimeIndex
            ):
                y.index = X.index

        # Detect seasonality if requested
        seasonal = self.params["seasonal"]
        seasonal_periods = self.params["seasonal_periods"]

        if detect_seasonality and seasonal is not None and seasonal_periods is None:
            # Try to detect seasonal periods from data frequency
            if isinstance(y.index, pd.DatetimeIndex):
                freq = pd.infer_freq(y.index)

                if freq is not None:
                    if freq.startswith("D"):
                        seasonal_periods = 7  # Weekly seasonality
                    elif freq.startswith("M"):
                        seasonal_periods = 12  # Monthly seasonality
                    elif freq.startswith("Q"):
                        seasonal_periods = 4  # Quarterly seasonality
                    elif freq.startswith("H"):
                        seasonal_periods = 24  # Hourly seasonality

                    logger.info(
                        f"Detected frequency {freq}, using seasonal_periods={seasonal_periods}"
                    )
                else:
                    logger.warning("Could not infer frequency from data")
            else:
                logger.warning("Cannot detect seasonality without datetime index")

        # Use detected seasonal_periods or fall back to parameter
        if seasonal_periods is None and seasonal is not None:
            logger.warning(
                "Seasonal component specified but no seasonal_periods provided"
            )
            seasonal = None

        # Create and fit the model
        model = ExponentialSmoothing(
            y,
            trend=self.params["trend"],
            damped_trend=self.params["damped_trend"],
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method=self.params["initialization_method"],
            **kwargs,
        )

        model_fit = model.fit()

        self.model = model_fit

        # Update metadata
        self.metadata.update(
            {
                "fitted_at": datetime.datetime.now().isoformat(),
                "data_shape": X.shape,
                "target_mean": float(y.mean()),
                "target_std": float(y.std()),
                "trend": self.params["trend"],
                "damped_trend": self.params["damped_trend"],
                "seasonal": seasonal,
                "seasonal_periods": seasonal_periods,
                "aic": float(model_fit.aic) if hasattr(model_fit, "aic") else None,
                "bic": float(model_fit.bic) if hasattr(model_fit, "bic") else None,
                "params": (
                    model_fit.params.tolist() if hasattr(model_fit, "params") else None
                ),
            }
        )

        logger.info(f"Successfully fitted exponential smoothing model {self.name}")

        return self

    def predict(
        self,
        X: pd.DataFrame,
        steps: int = None,
        return_conf_int: bool = False,
        alpha: float = 0.05,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Make predictions using the exponential smoothing model.

        Args:
            X: Feature dataframe.
            steps: Number of steps to forecast.
            return_conf_int: Whether to return confidence intervals.
            alpha: Significance level for confidence intervals.
            **kwargs: Additional prediction parameters.

        Returns:
            Predicted values, optionally with confidence intervals.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted")

        logger.info(f"Making predictions with exponential smoothing model {self.name}")

        # Determine the number of steps to forecast
        if steps is None:
            steps = len(X)

        # Make predictions
        if return_conf_int:
            forecast = self.model.forecast(steps=steps, **kwargs)
            pred_conf = self.model.get_prediction(
                start=len(self.model.fittedvalues),
                end=len(self.model.fittedvalues) + steps - 1,
            )
            conf_int = pred_conf.conf_int(alpha=alpha)

            lower = conf_int.iloc[:, 0].values
            upper = conf_int.iloc[:, 1].values

            return forecast.values, lower, upper
        else:
            forecast = self.model.forecast(steps=steps, **kwargs)
            return forecast.values
