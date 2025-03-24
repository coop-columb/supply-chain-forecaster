"""Anomaly detection model evaluator for the supply chain forecaster."""

import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)

from models.base import ModelBase
from utils import get_logger

logger = get_logger(__name__)


class AnomalyEvaluator:
    """Class for evaluating anomaly detection models."""

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the anomaly evaluator.
        
        Args:
            metrics: List of metrics to compute.
            output_dir: Directory to save evaluation results.
        """
        self.metrics = metrics or ["precision", "recall", "f1", "accuracy", "roc_auc"]
        self.output_dir = Path(output_dir) if output_dir else Path("models/evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized anomaly evaluator with metrics: {self.metrics}")

    def evaluate_model(
        self,
        model: ModelBase,
        X: pd.DataFrame,
        y_true: Optional[pd.Series] = None,
        threshold: Optional[float] = None,
        prefix: str = "",
        **kwargs,
    ) -> Dict[str, float]:
        """
        Evaluate an anomaly detection model.
        
        Args:
            model: Model to evaluate.
            X: Feature dataframe.
            y_true: True anomaly labels (1 for normal, -1 for anomaly).
            threshold: Threshold for anomaly detection.
            prefix: Prefix for metric names in the result.
            **kwargs: Additional arguments for model.predict.
        
        Returns:
            Dictionary of metric values.
        """
        logger.info(f"Evaluating anomaly detection model {model.name} on {len(X)} data points")
        
        # Make predictions
        y_pred, scores = model.predict(X, threshold=threshold, return_scores=True, **kwargs)
        
        metrics = {}
        
        # If true labels are provided, calculate classification metrics
        if y_true is not None:
            # Convert to numpy arrays
            y_true_np = y_true.values if isinstance(y_true, pd.Series) else y_true
            y_pred_np = y_pred
            
            # Ensure binary labels (1 for normal, -1 for anomaly)
            # Convert to 0 (normal) and 1 (anomaly) for sklearn metrics
            y_true_binary = (y_true_np == -1).astype(int)
            y_pred_binary = (y_pred_np == -1).astype(int)
            
            # Calculate metrics
            if "precision" in self.metrics:
                metrics["precision"] = precision_score(y_true_binary, y_pred_binary)
            
            if "recall" in self.metrics:
                metrics["recall"] = recall_score(y_true_binary, y_pred_binary)
            
            if "f1" in self.metrics:
                metrics["f1"] = f1_score(y_true_binary, y_pred_binary)
            
            if "accuracy" in self.metrics:
                metrics["accuracy"] = accuracy_score(y_true_binary, y_pred_binary)
            
            if "roc_auc" in self.metrics:
                try:
                    # For ROC AUC, we need scores rather than binary predictions
                    # Invert scores since higher scores should indicate anomalies
                    metrics["roc_auc"] = roc_auc_score(y_true_binary, -scores)
                except Exception as e:
                    logger.warning(f"Error calculating ROC AUC: {str(e)}")
            
            # Calculate confusion matrix
            if "confusion_matrix" in self.metrics:
                cm = confusion_matrix(y_true_binary, y_pred_binary)
                metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = cm.ravel()
        
        # Calculate anomaly rate
        metrics["anomaly_rate"] = np.mean(y_pred == -1)
        
        # Add prefix to metric names if specified
        if prefix:
            metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
        
        # Log metrics
        for name, value in metrics.items():
            logger.info(f"{name}: {value:.4f}")
        
        return metrics

    def plot_anomalies(
        self,
        model: ModelBase,
        X: pd.DataFrame,
        date_col: str = None,
        target_col: str = None,
        threshold: Optional[float] = None,
        title: str = None,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Plot time series with detected anomalies.
        
        Args:
            model: Model to generate anomalies.
            X: Feature dataframe.
            date_col: Column containing dates. If None, uses the index.
            target_col: Column to plot (e.g., demand, inventory).
            threshold: Threshold for anomaly detection.
            title: Plot title.
            save_path: Path to save the plot. If None, uses default path.
            **kwargs: Additional arguments for model.predict.
        """
        logger.info(f"Plotting anomalies for model {model.name}")
        
        # If target_col is not specified, try to use model's target or the first numeric column
        if target_col is None:
            if hasattr(model, "target") and model.target is not None:
                target_col = model.target
            else:
                numeric_cols = X.select_dtypes(include=["number"]).columns
                if len(numeric_cols) > 0:
                    target_col = numeric_cols[0]
                else:
                    raise ValueError("No target column specified and no numeric columns found")
        
        if target_col not in X.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Get dates
        if date_col is not None and date_col in X.columns:
            dates = X[date_col]
        else:
            if isinstance(X.index, pd.DatetimeIndex):
                dates = X.index
            else:
                dates = pd.RangeIndex(len(X))
        
        # Get anomalies
        y_pred, scores = model.predict(X, threshold=threshold, return_scores=True, **kwargs)
        
        # Create plot
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            
            # Plot time series
            plt.plot(dates, X[target_col], label=target_col)
            
            # Highlight anomalies
            anomaly_mask = y_pred == -1
            plt.scatter(
                dates[anomaly_mask],
                X[target_col][anomaly_mask],
                color="red",
                marker="o",
                label="Anomalies",
                zorder=5,
            )
            
            # Set title
            if title is None:
                title = f"Anomalies detected by {model.name} in {target_col}"
            plt.title(title)
            
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Format dates on x-axis if datetime
            if isinstance(dates, pd.DatetimeIndex) or pd.api.types.is_datetime64_dtype(dates):
                plt.gcf().autofmt_xdate()
            
            plt.tight_layout()
            
            # Save plot if requested
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Saved plot to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating anomaly plot: {str(e)}")

    def compare_models(
        self,
        models: List[ModelBase],
        X: pd.DataFrame,
        y_true: Optional[pd.Series] = None,
        save_results: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Compare multiple anomaly detection models on the same data.
        
        Args:
            models: List of models to compare.
            X: Feature dataframe.
            y_true: True anomaly labels (if available).
            save_results: Whether to save comparison results.
            **kwargs: Additional arguments for model.predict.
        
        Returns:
            Dataframe with comparison results.
        """
        logger.info(f"Comparing {len(models)} anomaly detection models")
        
        results = []
        
        for model in models:
            try:
                metrics = self.evaluate_model(model, X, y_true, **kwargs)
                
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
            comparison_path = self.output_dir / f"anomaly_model_comparison_{timestamp}.csv"
            comparison_df.to_csv(comparison_path)
            logger.info(f"Saved model comparison to {comparison_path}")
        
        return comparison_df

    def generate_report(
        self,
        model: ModelBase,
        X: pd.DataFrame,
        y_true: Optional[pd.Series] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> Dict:
        """
        Generate a comprehensive evaluation report for an anomaly detection model.
        
        Args:
            model: Model to evaluate.
            X: Feature dataframe.
            y_true: True anomaly labels (if available).
            threshold: Threshold for anomaly detection.
            **kwargs: Additional arguments for model.predict.
        
        Returns:
            Dictionary with evaluation report.
        """
        logger.info(f"Generating evaluation report for anomaly model {model.name}")
        
        report = {
            "model_name": model.name,
            "model_type": model.__class__.__name__,
            "parameters": model.params,
            "report_time": datetime.datetime.now().isoformat(),
            "data_shape": X.shape,
        }
        
        # Evaluate model
        metrics = self.evaluate_model(model, X, y_true, threshold, **kwargs)
        report["metrics"] = metrics
        
        # Get anomalies
        try:
            anomalies = model.get_anomalies(X, threshold)
            report["anomaly_count"] = len(anomalies)
            report["anomaly_rate"] = len(anomalies) / len(X)
            
            # Get top anomalies
            if "anomaly_score" in anomalies.columns:
                top_anomalies = anomalies.sort_values("anomaly_score", ascending=False).head(10)
                
                # Convert to serializable format
                top_anomalies_list = []
                for idx, row in top_anomalies.iterrows():
                    anomaly_dict = {"index": str(idx)}
                    for col in row.index:
                        anomaly_dict[str(col)] = float(row[col]) if pd.api.types.is_numeric_dtype(row[col]) else str(row[col])
                    top_anomalies_list.append(anomaly_dict)
                
                report["top_anomalies"] = top_anomalies_list
        except Exception as e:
            logger.error(f"Error getting anomalies: {str(e)}")
        
        # Save report
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"{model.name}_anomaly_report_{timestamp}.json"
        
        try:
            import json
            
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Saved anomaly evaluation report to {report_path}")
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
        
        return report