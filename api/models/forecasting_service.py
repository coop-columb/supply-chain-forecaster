"""Forecasting service for the supply chain forecaster API."""

import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from api.models.model_service import ModelService
from models.base import ModelBase
from models.forecasting import ARIMAModel, LSTMModel, ProphetModel, XGBoostModel
from utils import ModelError, get_logger, time_series_cross_validation
from utils.caching import memoize_with_expiry
from utils.profiling import profile_time

logger = get_logger(__name__)


class ForecastingService:
    """Service for forecasting operations in the API."""

    def __init__(self, model_service: Optional[ModelService] = None):
        """
        Initialize the forecasting service.
        
        Args:
            model_service: Model service instance.
        """
        self.model_service = model_service or ModelService()
        logger.info("Initialized forecasting service")

    def train_model(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_params: Optional[Dict] = None,
        model_name: Optional[str] = None,
        save_model: bool = True,
        **kwargs,
    ) -> Tuple[ModelBase, Dict]:
        """
        Train a forecasting model.
        
        Args:
            model_type: Type of model to train.
            X_train: Training feature dataframe.
            y_train: Training target series.
            model_params: Model parameters.
            model_name: Name for the model.
            save_model: Whether to save the trained model.
            **kwargs: Additional training parameters.
        
        Returns:
            Tuple of (trained model, training metrics).
        """
        logger.info(f"Training {model_type} model with {len(X_train)} samples")
        
        # Create model
        model_params = model_params or {}
        
        if not model_name:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_type}_{timestamp}"
        
        try:
            model = self.model_service.create_model(
                model_type, name=model_name, **model_params
            )
            
            # Train model
            model.fit(X_train, y_train, **kwargs)
            
            # Calculate training metrics
            train_predictions = model.predict(X_train)
            train_metrics = model.evaluate(X_train, y_train)
            
            # Save model if requested
            if save_model:
                self.model_service.save_model(model)
            
            logger.info(f"Successfully trained {model_type} model '{model_name}'")
            
            return model, train_metrics
        
        except Exception as e:
            raise ModelError(
                f"Error training {model_type} model: {str(e)}", model_name=model_name
            )

    def evaluate_model(
        self,
        model: ModelBase,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        metrics: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Evaluate a forecasting model.
        
        Args:
            model: Model to evaluate.
            X_test: Test feature dataframe.
            y_test: Test target series.
            metrics: List of metrics to compute.
            **kwargs: Additional evaluation parameters.
        
        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info(f"Evaluating model '{model.name}' with {len(X_test)} samples")
        
        try:
            # Make predictions
            y_pred = model.predict(X_test, **kwargs)
            
            # Calculate metrics
            from utils.evaluation import calculate_metrics
            
            metrics = metrics or ["mae", "rmse", "mape", "smape", "r2"]
            eval_metrics = calculate_metrics(y_test, y_pred, metrics)
            
            logger.info(f"Evaluation metrics for '{model.name}': {eval_metrics}")
            
            return eval_metrics
        
        except Exception as e:
            raise ModelError(
                f"Error evaluating model: {str(e)}", model_name=model.name
            )

    def cross_validate(
        self,
        model_type: str,
        data: pd.DataFrame,
        target_col: str,
        date_col: Optional[str] = None,
        model_params: Optional[Dict] = None,
        strategy: str = "expanding",
        initial_window: int = 30,
        step_size: int = 7,
        horizon: int = 7,
        max_train_size: Optional[int] = None,
        metrics: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict:
        """
        Perform time series cross-validation on a model.
        
        Args:
            model_type: Type of model to cross-validate.
            data: Dataframe containing features and target.
            target_col: Target column name.
            date_col: Date column name.
            model_params: Model parameters.
            strategy: Cross-validation strategy ('expanding' or 'sliding').
            initial_window: Initial window size.
            step_size: Step size for moving forward.
            horizon: Forecasting horizon.
            max_train_size: Maximum number of samples used for training.
            metrics: List of metrics to compute.
            **kwargs: Additional cross-validation parameters.
        
        Returns:
            Dictionary with cross-validation results.
        """
        logger.info(
            f"Performing time series cross-validation for {model_type} "
            f"with {len(data)} samples"
        )
        
        try:
            # Create model
            model_params = model_params or {}
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_type}_cv_{timestamp}"
            
            model = self.model_service.create_model(
                model_type, name=model_name, **model_params
            )
            
            # Perform cross-validation
            metrics = metrics or ["mae", "rmse", "mape", "smape", "r2"]
            metric_values, predictions_df = time_series_cross_validation(
                model,
                data,
                target_col,
                date_col=date_col,
                strategy=strategy,
                initial_window=initial_window,
                step_size=step_size,
                horizon=horizon,
                max_train_size=max_train_size,
                metrics=metrics,
            )
            
            # Calculate average metrics
            avg_metrics = {
                metric: float(np.mean(values))
                for metric, values in metric_values.items()
            }
            
            logger.info(f"Cross-validation average metrics: {avg_metrics}")
            
            # Return all results
            return {
                "model_type": model_type,
                "model_params": model_params,
                "cv_strategy": strategy,
                "initial_window": initial_window,
                "step_size": step_size,
                "horizon": horizon,
                "max_train_size": max_train_size,
                "num_folds": len(metric_values[list(metric_values.keys())[0]]),
                "avg_metrics": avg_metrics,
                "fold_metrics": {
                    metric: values.tolist() if isinstance(values, np.ndarray) else values
                    for metric, values in metric_values.items()
                },
            }
        
        except Exception as e:
            raise ModelError(
                f"Error during cross-validation: {str(e)}", model_name=model_type
            )

    @memoize_with_expiry()
    def forecast(
        self,
        model_name: str,
        model_type: str,
        X: pd.DataFrame,
        steps: int = 30,
        return_conf_int: bool = False,
        from_deployment: bool = True,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate a forecast using a trained model.
        
        Args:
            model_name: Name of the model to use.
            model_type: Type of the model.
            X: Feature dataframe.
            steps: Number of steps to forecast.
            return_conf_int: Whether to return confidence intervals.
            from_deployment: Whether to load model from deployment.
            **kwargs: Additional forecast parameters.
        
        Returns:
            Forecast values, optionally with confidence intervals.
        """
        logger.info(f"Generating forecast with model '{model_name}' for {steps} steps")
        
        with profile_time(f"forecast_{model_type}_{model_name}", "api"):
            try:
                # Load model
                model = self.model_service.load_model(
                    model_name, model_type, from_deployment=from_deployment
                )
                
                # Generate forecast
                if return_conf_int and hasattr(model, "predict") and "return_conf_int" in model.predict.__code__.co_varnames:
                    predictions, lower, upper = model.predict(
                        X, steps=steps, return_conf_int=True, **kwargs
                    )
                    return predictions, lower, upper
                else:
                    predictions = model.predict(X, steps=steps, **kwargs)
                    return predictions
            
            except Exception as e:
                raise ModelError(
                    f"Error generating forecast: {str(e)}", model_name=model_name
                )
