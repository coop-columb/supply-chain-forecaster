"""Prophet forecasting model for the supply chain forecaster."""

import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from models.base import ModelBase, ModelRegistry
from utils import get_logger, safe_execute

logger = get_logger(__name__)


@ModelRegistry.register
class ProphetModel(ModelBase):
    """Facebook Prophet forecasting model."""

    def __init__(
        self,
        name: str = "ProphetModel",
        seasonality_mode: str = "additive",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        daily_seasonality: bool = False,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
        add_country_holidays: Optional[str] = None,
        uncertainty_samples: int = 1000,
        **kwargs,
    ):
        """
        Initialize the Prophet model.

        Args:
            name: Name of the model.
            seasonality_mode: Seasonality mode (additive or multiplicative).
            changepoint_prior_scale: Parameter controlling flexibility of trend changes.
            seasonality_prior_scale: Parameter controlling flexibility of seasonality.
            holidays_prior_scale: Parameter controlling flexibility of holidays effects.
            daily_seasonality: Whether to include daily seasonality.
            weekly_seasonality: Whether to include weekly seasonality.
            yearly_seasonality: Whether to include yearly seasonality.
            add_country_holidays: Country code to include holidays for.
            uncertainty_samples: Number of samples for uncertainty intervals.
            **kwargs: Additional Prophet model parameters.
        """
        try:
            import prophet
        except ImportError:
            logger.error(
                "Prophet not installed. Please install with pip install prophet"
            )
            raise

        super().__init__(
            name=name,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            add_country_holidays=add_country_holidays,
            uncertainty_samples=uncertainty_samples,
            **kwargs,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        date_col: str = None,
        additional_regressors: Optional[List[str]] = None,
        **kwargs,
    ) -> "ProphetModel":
        """
        Fit the Prophet model to the data.

        Args:
            X: Feature dataframe.
            y: Target series.
            date_col: Column containing dates. If None, uses the index.
            additional_regressors: List of columns to use as additional regressors.
            **kwargs: Additional fitting parameters.

        Returns:
            Self for method chaining.
        """
        try:
            from prophet import Prophet
        except ImportError:
            logger.error(
                "Prophet not installed. Please install with pip install prophet"
            )
            raise

        logger.info(f"Fitting Prophet model {self.name}")

        # Store feature names and target name
        self.features = list(X.columns)
        self.target = y.name if y.name else "target"

        # Get date series
        if date_col is not None and date_col in X.columns:
            dates = X[date_col]
        else:
            if isinstance(X.index, pd.DatetimeIndex):
                dates = X.index
            else:
                raise ValueError(
                    "Either date_col must be provided or X must have a DatetimeIndex"
                )

        # Prepare data for Prophet
        df = pd.DataFrame({"ds": dates, "y": y})

        # Initialize and configure Prophet model
        model = Prophet(
            seasonality_mode=self.params["seasonality_mode"],
            changepoint_prior_scale=self.params["changepoint_prior_scale"],
            seasonality_prior_scale=self.params["seasonality_prior_scale"],
            holidays_prior_scale=self.params["holidays_prior_scale"],
            daily_seasonality=self.params["daily_seasonality"],
            weekly_seasonality=self.params["weekly_seasonality"],
            yearly_seasonality=self.params["yearly_seasonality"],
        )

        # Add country holidays if specified
        if self.params["add_country_holidays"]:
            model.add_country_holidays(country_name=self.params["add_country_holidays"])

        # Add additional regressors if specified
        if additional_regressors:
            for regressor in additional_regressors:
                if regressor in X.columns:
                    df[regressor] = X[regressor]
                    model.add_regressor(regressor)
                else:
                    logger.warning(f"Regressor {regressor} not found in dataframe")

        # Fit the model
        model.fit(df)

        # Store the fitted model
        self.model = model

        # Update metadata
        self.metadata.update(
            {
                "fitted_at": datetime.datetime.now().isoformat(),
                "data_shape": X.shape,
                "target_mean": float(y.mean()),
                "target_std": float(y.std()),
                "additional_regressors": additional_regressors,
            }
        )

        logger.info(f"Successfully fitted Prophet model {self.name}")

        return self

    def predict(
        self,
        X: pd.DataFrame,
        date_col: str = None,
        periods: int = None,
        freq: str = "D",
        include_history: bool = False,
        additional_regressors: Optional[List[str]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Make predictions using the Prophet model.

        Args:
            X: Feature dataframe.
            date_col: Column containing dates. If None, uses the index.
            periods: Number of future periods to predict. Only needed if X doesn't have the future dates.
            freq: Frequency of predictions for future periods.
            include_history: Whether to include historical predictions.
            additional_regressors: List of columns to use as additional regressors.
            **kwargs: Additional prediction parameters.

        Returns:
            Predicted values.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted")

        logger.info(f"Making predictions with Prophet model {self.name}")

        # Get date series
        if date_col is not None and date_col in X.columns:
            dates = X[date_col]
        else:
            if isinstance(X.index, pd.DatetimeIndex):
                dates = X.index
            else:
                # If no dates provided and we need future predictions, use periods and freq
                if periods is not None:
                    logger.info(f"Creating future dataframe with {periods} periods")
                    future = self.model.make_future_dataframe(
                        periods=periods, freq=freq
                    )

                    # Add additional regressors if specified
                    if additional_regressors:
                        for regressor in additional_regressors:
                            if regressor in X.columns:
                                # Create a mapping from existing dates to regressor values
                                regressor_values = dict(zip(dates, X[regressor]))

                                # Fill future dataframe with regressor values
                                # For dates we don't have values for, use the latest available value
                                latest_value = X[regressor].iloc[-1]
                                future[regressor] = future["ds"].map(
                                    lambda x: regressor_values.get(x, latest_value)
                                )
                            else:
                                logger.warning(
                                    f"Regressor {regressor} not found in dataframe"
                                )

                    # Make predictions
                    forecast = self.model.predict(future)

                    # Return only future predictions if not include_history
                    if not include_history:
                        forecast = forecast.iloc[-periods:]

                    return forecast["yhat"].values

                else:
                    raise ValueError(
                        "Either date_col must be provided, X must have a DatetimeIndex, "
                        "or periods must be specified for future predictions"
                    )

        # Prepare data for prediction
        future = pd.DataFrame({"ds": dates})

        # Add additional regressors if specified
        if additional_regressors:
            for regressor in additional_regressors:
                if regressor in X.columns:
                    future[regressor] = X[regressor]
                else:
                    logger.warning(f"Regressor {regressor} not found in dataframe")

        # Make predictions
        forecast = self.model.predict(future)

        return forecast["yhat"].values

    def get_components(self, X: pd.DataFrame, date_col: str = None) -> pd.DataFrame:
        """
        Get the forecast components (trend, seasonality, etc.).

        Args:
            X: Feature dataframe.
            date_col: Column containing dates. If None, uses the index.

        Returns:
            Dataframe with forecast components.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted")

        # Get date series
        if date_col is not None and date_col in X.columns:
            dates = X[date_col]
        else:
            if isinstance(X.index, pd.DatetimeIndex):
                dates = X.index
            else:
                raise ValueError(
                    "Either date_col must be provided or X must have a DatetimeIndex"
                )

        # Prepare data for prediction
        future = pd.DataFrame({"ds": dates})

        # Make predictions with components
        forecast = self.model.predict(future)

        component_columns = [
            col
            for col in forecast.columns
            if col not in {"ds", "yhat", "yhat_lower", "yhat_upper"}
        ]

        return forecast[["ds"] + component_columns]
