"""Forecasting model evaluator for the supply chain forecaster."""

import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from models.base import ModelBase
from utils import (
    calculate_metrics,
    create_feature_importance_plot,
    get_logger,
    plot_forecast_components,
    plot_forecast_vs_actual,
    time_series_cross_validation,
)

logger = get_logger(__name__)


class ForecastingEvaluator:
    """Class for evaluating forecasting models."""

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the forecasting evaluator.

        Args:
            metrics: List of metrics to compute.
            output_dir: Directory to save evaluation results.
        """
        self.metrics = metrics or ["mae", "rmse", "mape", "smape", "r2"]
        self.output_dir = Path(output_dir) if output_dir else Path("models/evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized forecasting evaluator with metrics: {self.metrics}")

    def evaluate_model(
        self,
        model: ModelBase,
        X: pd.DataFrame,
        y: pd.Series,
        prefix: str = "",
        **kwargs,
    ) -> Dict[str, float]:
        """
        Evaluate a model on test data.

        Args:
            model: Model to evaluate.
            X: Feature dataframe.
            y: Target series.
            prefix: Prefix for metric names in the result.
            **kwargs: Additional arguments for model.predict.

        Returns:
            Dictionary of metric values.
        """
        logger.info(f"Evaluating model {model.name} on {len(X)} data points")

        # Make predictions
        y_pred = model.predict(X, **kwargs)

        # Calculate metrics
        metrics = calculate_metrics(y.values, y_pred, self.metrics)

        # Add prefix to metric names if specified
        if prefix:
            metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}

        # Log metrics
        for name, value in metrics.items():
            logger.info(f"{name}: {value:.4f}")

        return metrics

    def plot_predictions(
        self,
        model: ModelBase,
        X: pd.DataFrame,
        y: pd.Series,
        date_col: str = None,
        title: str = None,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Plot model predictions against actual values.

        Args:
            model: Model to generate predictions.
            X: Feature dataframe.
            y: Target series.
            date_col: Column containing dates. If None, uses the index.
            title: Plot title.
            save_path: Path to save the plot. If None, uses default path.
            **kwargs: Additional arguments for model.predict.
        """
        logger.info(f"Plotting predictions for model {model.name}")

        # Make predictions
        y_pred = model.predict(X, **kwargs)

        # Get dates
        if date_col is not None and date_col in X.columns:
            dates = X[date_col]
        else:
            if isinstance(X.index, pd.DatetimeIndex):
                dates = X.index
            else:
                dates = pd.RangeIndex(len(y))

        # Create Series with dates as index
        y_actual = pd.Series(y.values, index=dates, name="Actual")
        y_predicted = pd.Series(y_pred, index=dates, name="Predicted")

        # Set title
        if title is None:
            title = f"Forecast vs Actual for {model.name}"

        # Plot
        plot_forecast_vs_actual(y_actual, y_predicted, title=title)

        # Save plot if requested
        if save_path:
            try:
                import matplotlib.pyplot as plt

                plt.savefig(save_path)
                logger.info(f"Saved plot to {save_path}")
            except Exception as e:
                logger.error(f"Error saving plot: {str(e)}")

    def cross_validate(
        self,
        model: ModelBase,
        X: pd.DataFrame,
        y: pd.Series,
        date_col: str = None,
        strategy: str = "expanding",
        initial_window: int = 30,
        step_size: int = 7,
        horizon: int = 7,
        max_train_size: Optional[int] = None,
        save_results: bool = True,
        **kwargs,
    ) -> Tuple[Dict[str, List[float]], pd.DataFrame]:
        """
        Perform time series cross-validation on a model.

        Args:
            model: Model to cross-validate.
            X: Feature dataframe.
            y: Target series.
            date_col: Column containing dates. If None, uses the index.
            strategy: Cross-validation strategy ('expanding' or 'sliding').
            initial_window: Initial window size.
            step_size: Step size for moving forward.
            horizon: Forecasting horizon.
            max_train_size: Maximum number of samples used for training.
            save_results: Whether to save cross-validation results.
            **kwargs: Additional arguments for time_series_cross_validation.

        Returns:
            Tuple containing:
            - Dictionary of metric values for each fold.
            - Dataframe with true and predicted values for each fold.
        """
        logger.info(f"Performing time series cross-validation for model {model.name}")

        # Clone model for cross-validation
        model_class = model.__class__
        cv_model = model_class(**model.params)

        # Perform cross-validation
        metric_values, predictions_df = time_series_cross_validation(
            cv_model,
            pd.concat([X, y], axis=1),
            y.name,
            date_col=date_col,
            strategy=strategy,
            initial_window=initial_window,
            step_size=step_size,
            horizon=horizon,
            max_train_size=max_train_size,
            metrics=self.metrics,
        )

        # Save results if requested
        if save_results:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save metrics
            metrics_path = self.output_dir / f"{model.name}_cv_metrics_{timestamp}.csv"
            metrics_df = pd.DataFrame(metric_values)
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Saved cross-validation metrics to {metrics_path}")

            # Save predictions
            predictions_path = (
                self.output_dir / f"{model.name}_cv_predictions_{timestamp}.csv"
            )
            predictions_df.to_csv(predictions_path, index=False)
            logger.info(f"Saved cross-validation predictions to {predictions_path}")

        return metric_values, predictions_df

    def compare_models(
        self,
        models: List[ModelBase],
        X: pd.DataFrame,
        y: pd.Series,
        save_results: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same test data.

        Args:
            models: List of models to compare.
            X: Feature dataframe.
            y: Target series.
            save_results: Whether to save comparison results.
            **kwargs: Additional arguments for model.predict.

        Returns:
            Dataframe with comparison results.
        """
        logger.info(f"Comparing {len(models)} models")

        results = []

        for model in models:
            try:
                metrics = self.evaluate_model(model, X, y, **kwargs)

                # Add model name
                metrics["model_name"] = model.name
                results.append(metrics)

            except Exception as e:
                logger.error(f"Error evaluating model {model.name}: {str(e)}")

        # Create comparison dataframe
        comparison_df = pd.DataFrame(results)

        # Set model_name as index
        if "model_name" in comparison_df.columns:
            comparison_df = comparison_df.set_index("model_name")

        # Save results if requested
        if save_results and not comparison_df.empty:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_path = self.output_dir / f"model_comparison_{timestamp}.csv"
            comparison_df.to_csv(comparison_path)
            logger.info(f"Saved model comparison to {comparison_path}")

        return comparison_df

    def generate_report(
        self,
        model: ModelBase,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv_results: Optional[Dict] = None,
        **kwargs,
    ) -> Dict:
        """
        Generate a comprehensive evaluation report for a model.

        Args:
            model: Model to evaluate.
            X_train: Training feature dataframe.
            y_train: Training target series.
            X_test: Test feature dataframe.
            y_test: Test target series.
            cv_results: Cross-validation results.
            **kwargs: Additional arguments for model.predict.

        Returns:
            Dictionary with evaluation report.
        """
        logger.info(f"Generating evaluation report for model {model.name}")

        report = {
            "model_name": model.name,
            "model_type": model.__class__.__name__,
            "parameters": model.params,
            "report_time": datetime.datetime.now().isoformat(),
            "train_data_shape": X_train.shape,
            "test_data_shape": X_test.shape,
        }

        # Evaluate on test data
        test_metrics = self.evaluate_model(
            model, X_test, y_test, prefix="test", **kwargs
        )
        report["test_metrics"] = test_metrics

        # Evaluate on training data
        train_metrics = self.evaluate_model(
            model, X_train, y_train, prefix="train", **kwargs
        )
        report["train_metrics"] = train_metrics

        # Add cross-validation results if provided
        if cv_results:
            report["cv_results"] = cv_results

        # Get feature importance if available
        if hasattr(model, "get_feature_importance"):
            try:
                feature_importance = model.get_feature_importance()
                report["feature_importance"] = feature_importance.to_dict("records")
            except Exception as e:
                logger.error(f"Error getting feature importance: {str(e)}")

        # Save report
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"{model.name}_report_{timestamp}.json"

        try:
            import json

            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"Saved evaluation report to {report_path}")
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")

        return report
