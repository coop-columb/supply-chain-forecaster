"""Model evaluation utilities for the supply chain forecaster."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from utils.logging import get_logger

logger = get_logger(__name__)


def calculate_metrics(
    y_true: Union[List[float], np.ndarray, pd.Series],
    y_pred: Union[List[float], np.ndarray, pd.Series],
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for regression problems.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        metrics: List of metrics to calculate. If None, calculates all.

    Returns:
        A dictionary of metric names and values.
    """
    # Convert inputs to numpy arrays
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    # Default metrics
    if metrics is None:
        metrics = ["mae", "rmse", "mape", "smape", "r2"]

    results = {}

    # Calculate requested metrics
    for metric in metrics:
        if metric.lower() == "mae":
            results["mae"] = mean_absolute_error(y_true_np, y_pred_np)

        elif metric.lower() == "rmse":
            results["rmse"] = np.sqrt(mean_squared_error(y_true_np, y_pred_np))

        elif metric.lower() == "mse":
            results["mse"] = mean_squared_error(y_true_np, y_pred_np)

        elif metric.lower() == "r2":
            results["r2"] = r2_score(y_true_np, y_pred_np)

        elif metric.lower() == "mape":
            # Handle zeros in y_true
            mask = y_true_np != 0
            if not np.any(mask):
                results["mape"] = np.nan
            else:
                try:
                    results["mape"] = (
                        mean_absolute_percentage_error(y_true_np[mask], y_pred_np[mask])
                        * 100
                    )  # Convert to percentage
                except:
                    # Calculate manually if sklearn version doesn't support it
                    results["mape"] = (
                        np.mean(
                            np.abs(
                                (y_true_np[mask] - y_pred_np[mask]) / y_true_np[mask]
                            )
                        )
                        * 100
                    )

        elif metric.lower() == "smape":
            # Symmetric Mean Absolute Percentage Error
            denominator = np.abs(y_true_np) + np.abs(y_pred_np)
            mask = denominator != 0  # Avoid division by zero
            if not np.any(mask):
                results["smape"] = np.nan
            else:
                smape = (
                    2.0
                    * np.mean(
                        np.abs(y_pred_np[mask] - y_true_np[mask]) / denominator[mask]
                    )
                    * 100
                )  # Convert to percentage
                results["smape"] = smape

        else:
            logger.warning(f"Unknown metric: {metric}, skipping")

    return results


def time_series_cross_validation(
    model,
    data: pd.DataFrame,
    target_col: str,
    date_col: str = None,
    strategy: str = "expanding",
    initial_window: int = 30,
    step_size: int = 7,
    horizon: int = 7,
    max_train_size: Optional[int] = None,
    metrics: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[float]], pd.DataFrame]:
    """
    Perform time series cross-validation with expanding or sliding window.

    Args:
        model: The model to train and evaluate.
        data: The dataframe containing the data.
        target_col: The target column name.
        date_col: The date column name. If None, uses the index.
        strategy: The cross-validation strategy ('expanding' or 'sliding').
        initial_window: The initial window size.
        step_size: The step size for moving forward.
        horizon: The forecasting horizon.
        max_train_size: Maximum number of samples used for training.
        metrics: List of metrics to calculate.

    Returns:
        A tuple containing:
        - Dictionary of metric values for each fold.
        - Dataframe with true and predicted values for each fold.
    """
    logger.info(
        f"Performing time series cross-validation with {strategy} window "
        f"(initial_window={initial_window}, step_size={step_size}, horizon={horizon})"
    )

    # Ensure data is sorted by date
    if date_col is not None:
        data = data.sort_values(date_col)
    else:
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()
        else:
            logger.warning(
                "Data is not sorted by date and no date column provided. "
                "Assuming data is already sorted chronologically."
            )

    # Initialize results
    metric_values = {
        metric: [] for metric in (metrics or ["mae", "rmse", "mape", "smape", "r2"])
    }
    predictions_df = []

    n_samples = len(data)

    # Cannot perform CV if we don't have enough data
    if n_samples <= initial_window + horizon:
        logger.error(
            f"Not enough data for cross-validation: {n_samples} samples, "
            f"need at least {initial_window + horizon + 1}"
        )
        return metric_values, pd.DataFrame()

    # Determine the number of folds
    n_folds = (n_samples - initial_window - horizon) // step_size + 1
    logger.info(f"Performing {n_folds} cross-validation folds")

    for fold in range(n_folds):
        # Calculate train/test indices for this fold
        train_start = 0

        if strategy == "expanding":
            train_end = initial_window + fold * step_size
        elif strategy == "sliding":
            if max_train_size is not None:
                train_start = max(0, initial_window + fold * step_size - max_train_size)
            train_end = initial_window + fold * step_size
        else:
            raise ValueError(f"Unknown cross-validation strategy: {strategy}")

        test_start = train_end
        test_end = min(test_start + horizon, n_samples)

        # Extract train/test sets
        train_data = data.iloc[train_start:train_end]
        test_data = data.iloc[test_start:test_end]

        logger.debug(
            f"Fold {fold + 1}/{n_folds}: train={train_start}:{train_end} "
            f"({len(train_data)} samples), test={test_start}:{test_end} "
            f"({len(test_data)} samples)"
        )

        if len(train_data) == 0 or len(test_data) == 0:
            logger.warning(f"Skipping fold {fold + 1} due to empty train or test set")
            continue

        # Get X and y for train and test
        X_train = train_data.drop(columns=[target_col])
        y_train = train_data[target_col]
        X_test = test_data.drop(columns=[target_col])
        y_test = test_data[target_col]

        # Train the model
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            fold_metrics = calculate_metrics(y_test, y_pred, metrics)
            for metric, value in fold_metrics.items():
                metric_values[metric].append(value)

            # Store predictions
            fold_df = test_data.copy()
            fold_df["fold"] = fold + 1
            fold_df["predicted"] = y_pred
            predictions_df.append(fold_df)

            logger.debug(f"Fold {fold + 1} metrics: {fold_metrics}")

        except Exception as e:
            logger.error(f"Error in fold {fold + 1}: {str(e)}")
            continue

    # Combine all predictions
    if predictions_df:
        all_predictions = pd.concat(predictions_df)
    else:
        all_predictions = pd.DataFrame()

    # Calculate average metrics
    for metric in metric_values:
        logger.info(f"Average {metric.upper()}: {np.mean(metric_values[metric])}")

    return metric_values, all_predictions


def plot_forecast_vs_actual(
    actual: pd.Series,
    forecast: pd.Series,
    title: str = "Forecast vs Actual",
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Plot forecast values against actual values.

    Args:
        actual: The actual values.
        forecast: The forecasted values.
        title: The plot title.
        figsize: The figure size (width, height).
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize)
        plt.plot(actual.index, actual, label="Actual", marker="o")
        plt.plot(forecast.index, forecast, label="Forecast", marker="x")
        plt.fill_between(
            forecast.index,
            forecast * 0.9,
            forecast * 1.1,
            alpha=0.2,
            label="±10% Range",
        )
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    except ImportError:
        logger.warning("matplotlib not available, skipping plot")


def plot_forecast_components(model, figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Plot the components of a time series forecast model.

    Args:
        model: The trained forecasting model with plot_components method.
        figsize: The figure size (width, height).
    """
    try:
        # Check if model has plot_components method (e.g., Prophet)
        if hasattr(model, "plot_components") and callable(
            getattr(model, "plot_components")
        ):
            import matplotlib.pyplot as plt

            fig = model.plot_components(figsize=figsize)
            plt.tight_layout()
            plt.show()
        else:
            logger.warning("Model does not have plot_components method, skipping plot")

    except ImportError:
        logger.warning("Required plotting libraries not available, skipping plot")
    except Exception as e:
        logger.error(f"Error plotting forecast components: {str(e)}")


def create_feature_importance_plot(
    model, feature_names, top_n: int = 20, figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot feature importances for tree-based models.

    Args:
        model: The trained model (must have feature_importances_ attribute).
        feature_names: List of feature names.
        top_n: Number of top features to display.
        figsize: The figure size (width, height).
    """
    try:
        import matplotlib.pyplot as plt

        # Check if model has feature_importances_
        if hasattr(model, "feature_importances_"):
            # Get feature importances
            importances = model.feature_importances_

            # Sort features by importance
            indices = np.argsort(importances)[::-1]

            # Limit to top_n features
            indices = indices[:top_n]
            top_features = [feature_names[i] for i in indices]
            top_importances = importances[indices]

            # Plot
            plt.figure(figsize=figsize)
            plt.barh(range(len(top_importances)), top_importances, align="center")
            plt.yticks(range(len(top_importances)), top_features)
            plt.title("Feature Importances")
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.show()

        # Check if model is a Pipeline with a final estimator that has feature_importances_
        elif hasattr(model, "steps") and hasattr(
            model.steps[-1][1], "feature_importances_"
        ):
            importances = model.steps[-1][1].feature_importances_

            # Sort features by importance
            indices = np.argsort(importances)[::-1]

            # Limit to top_n features
            indices = indices[:top_n]
            top_features = [feature_names[i] for i in indices]
            top_importances = importances[indices]

            # Plot
            plt.figure(figsize=figsize)
            plt.barh(range(len(top_importances)), top_importances, align="center")
            plt.yticks(range(len(top_importances)), top_features)
            plt.title("Feature Importances")
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.show()

        else:
            logger.warning(
                "Model does not have feature_importances_ attribute, skipping plot"
            )

    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
    except Exception as e:
        logger.error(f"Error plotting feature importances: {str(e)}")
