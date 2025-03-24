"""ARIMA forecasting model for the supply chain forecaster."""

import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from models.base import ModelBase, ModelRegistry
from utils import get_logger, safe_execute

logger = get_logger(__name__)


@ModelRegistry.register
class ARIMAModel(ModelBase):
    """ARIMA forecasting model."""

    def __init__(
        self,
        name: str = "ARIMAModel",
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        trend: Optional[str] = None,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        auto_arima: bool = False,
        auto_arima_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initialize the ARIMA model.
        
        Args:
            name: Name of the model.
            order: ARIMA order (p, d, q).
            seasonal_order: Seasonal ARIMA order (P, D, Q, s).
            trend: Trend component.
            enforce_stationarity: Whether to enforce stationarity.
            enforce_invertibility: Whether to enforce invertibility.
            auto_arima: Whether to use auto_arima to find best parameters.
            auto_arima_kwargs: Additional parameters for auto_arima.
            **kwargs: Additional ARIMA model parameters.
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
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            auto_arima=auto_arima,
            auto_arima_kwargs=auto_arima_kwargs or {},
            **kwargs,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        date_col: str = None,
        exog_cols: Optional[List[str]] = None,
        **kwargs,
    ) -> "ARIMAModel":
        """
        Fit the ARIMA model to the data.
        
        Args:
            X: Feature dataframe.
            y: Target series.
            date_col: Column containing dates. If None, uses the index.
            exog_cols: List of exogenous variables to include.
            **kwargs: Additional fitting parameters.
        
        Returns:
            Self for method chaining.
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            logger.error(
                "statsmodels not installed. Please install with pip install statsmodels"
            )
            raise
        
        logger.info(f"Fitting ARIMA model {self.name}")
        
        # Store feature names and target name
        self.features = list(X.columns)
        self.target = y.name if y.name else "target"
        
        # Get date series
        if date_col is not None and date_col in X.columns:
            dates = X[date_col]
            y.index = pd.DatetimeIndex(dates)
        else:
            if not isinstance(y.index, pd.DatetimeIndex) and isinstance(X.index, pd.DatetimeIndex):
                y.index = X.index
        
        # Prepare exogenous variables if specified
        exog = None
        if exog_cols:
            exog_cols = [col for col in exog_cols if col in X.columns]
            if exog_cols:
                exog = X[exog_cols]
                logger.info(f"Using exogenous variables: {exog_cols}")
            else:
                logger.warning("No valid exogenous variables found")
        
        # Use auto_arima if specified
        if self.params["auto_arima"]:
            try:
                import pmdarima as pm
                
                logger.info("Using auto_arima to find best parameters")
                
                auto_arima_kwargs = self.params["auto_arima_kwargs"]
                auto_model = pm.auto_arima(
                    y,
                    exogenous=exog,
                    seasonal=True,
                    **auto_arima_kwargs,
                )
                
                # Update model parameters with auto_arima results
                self.params["order"] = auto_model.order
                self.params["seasonal_order"] = auto_model.seasonal_order
                
                logger.info(
                    f"Auto ARIMA selected order={self.params['order']}, "
                    f"seasonal_order={self.params['seasonal_order']}"
                )
                
                # Use the fitted model from auto_arima
                self.model = auto_model
                
            except ImportError:
                logger.error(
                    "pmdarima not installed. Please install with pip install pmdarima"
                )
                raise
        
        else:
            # Fit ARIMA or SARIMAX model
            if self.params["seasonal_order"]:
                # Use SARIMAX for seasonal models
                model = SARIMAX(
                    y,
                    exog=exog,
                    order=self.params["order"],
                    seasonal_order=self.params["seasonal_order"],
                    trend=self.params["trend"],
                    enforce_stationarity=self.params["enforce_stationarity"],
                    enforce_invertibility=self.params["enforce_invertibility"],
                )
            else:
                # Use regular ARIMA for non-seasonal models
                model = ARIMA(
                    y,
                    exog=exog,
                    order=self.params["order"],
                    trend=self.params["trend"],
                )
            
            # Fit the model
            model_fit = model.fit()
            self.model = model_fit
        
        # Update metadata
        self.metadata.update({
            "fitted_at": datetime.datetime.now().isoformat(),
            "data_shape": X.shape,
            "target_mean": float(y.mean()),
            "target_std": float(y.std()),
            "exog_cols": exog_cols,
            "order": self.params["order"],
            "seasonal_order": self.params["seasonal_order"],
            "aic": float(self.model.aic) if hasattr(self.model, "aic") else None,
            "bic": float(self.model.bic) if hasattr(self.model, "bic") else None,
        })
        
        logger.info(f"Successfully fitted ARIMA model {self.name}")
        
        return self

    def predict(
        self,
        X: pd.DataFrame,
        date_col: str = None,
        steps: int = None,
        exog_cols: Optional[List[str]] = None,
        return_conf_int: bool = False,
        alpha: float = 0.05,
        dynamic: bool = False,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Make predictions using the ARIMA model.
        
        Args:
            X: Feature dataframe.
            date_col: Column containing dates. If None, uses the index.
            steps: Number of steps to forecast.
            exog_cols: List of exogenous variables to include.
            return_conf_int: Whether to return confidence intervals.
            alpha: Significance level for confidence intervals.
            dynamic: Whether to do dynamic forecasting.
            **kwargs: Additional prediction parameters.
        
        Returns:
            Predicted values, optionally with confidence intervals.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted")
        
        logger.info(f"Making predictions with ARIMA model {self.name}")
        
        # Prepare exogenous variables if specified
        exog = None
        if exog_cols:
            exog_cols = [col for col in exog_cols if col in X.columns]
            if exog_cols:
                exog = X[exog_cols]
                logger.info(f"Using exogenous variables: {exog_cols}")
            else:
                logger.warning("No valid exogenous variables found")
        
        # Determine the number of steps to forecast
        if steps is None:
            steps = len(X)
        
        # Make predictions
        try:
            if hasattr(self.model, "get_forecast"):
                # SARIMAX or ARIMA from statsmodels.tsa.arima.model
                forecast = self.model.get_forecast(steps=steps, exog=exog, alpha=alpha)
                predictions = forecast.predicted_mean
                
                if return_conf_int:
                    conf_int = forecast.conf_int(alpha=alpha)
                    lower = conf_int.iloc[:, 0]
                    upper = conf_int.iloc[:, 1]
                    return predictions.values, lower.values, upper.values
                else:
                    return predictions.values
            
            elif hasattr(self.model, "predict"):
                # auto_arima model from pmdarima
                predictions = self.model.predict(n_periods=steps, exogenous=exog, return_conf_int=return_conf_int, alpha=alpha)
                
                if return_conf_int:
                    predictions, conf_int = predictions
                    lower = conf_int[:, 0]
                    upper = conf_int[:, 1]
                    return predictions, lower, upper
                else:
                    return predictions
            
            else:
                raise AttributeError("Model has no predict or get_forecast method")
        
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
