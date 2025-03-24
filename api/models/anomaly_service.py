"""Anomaly detection service for the supply chain forecaster API."""

import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from api.models.model_service import ModelService
from models.base import ModelBase
from models.anomaly import IsolationForestDetector, StatisticalDetector, AutoencoderDetector
from utils import ModelError, get_logger

logger = get_logger(__name__)


class AnomalyService:
    """Service for anomaly detection operations in the API."""

    def __init__(self, model_service: Optional[ModelService] = None):
        """
        Initialize the anomaly detection service.
        
        Args:
            model_service: Model service instance.
        """
        self.model_service = model_service or ModelService()
        logger.info("Initialized anomaly detection service")

    def train_model(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
        model_params: Optional[Dict] = None,
        model_name: Optional[str] = None,
        save_model: bool = True,
        **kwargs,
    ) -> Tuple[ModelBase, Dict]:
        """
        Train an anomaly detection model.
        
        Args:
            model_type: Type of model to train.
            X_train: Training feature dataframe.
            y_train: Training target series (if supervised).
            model_params: Model parameters.
            model_name: Name for the model.
            save_model: Whether to save the trained model.
            **kwargs: Additional training parameters.
        
        Returns:
            Tuple of (trained model, training metrics).
        """
        logger.info(f"Training {model_type} anomaly model with {len(X_train)} samples")
        
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
            train_predictions, scores = model.predict(X_train, return_scores=True)
            
            # Basic metrics
            train_metrics = {
                "anomaly_rate": float(np.mean(train_predictions == -1)),
                "num_anomalies": int(np.sum(train_predictions == -1)),
            }
            
            # Save model if requested
            if save_model:
                self.model_service.save_model(model)
            
            logger.info(f"Successfully trained {model_type} anomaly model '{model_name}'")
            
            return model, train_metrics
        
        except Exception as e:
            raise ModelError(
                f"Error training {model_type} anomaly model: {str(e)}", model_name=model_name
            )

    def evaluate_model(
        self,
        model: ModelBase,
        X_test: pd.DataFrame,
        y_test: Optional[pd.Series] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Evaluate an anomaly detection model.
        
        Args:
            model: Model to evaluate.
            X_test: Test feature dataframe.
            y_test: True anomaly labels (if available).
            threshold: Threshold for anomaly detection.
            **kwargs: Additional evaluation parameters.
        
        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info(f"Evaluating anomaly model '{model.name}' with {len(X_test)} samples")
        
        try:
            # Make predictions
            y_pred, scores = model.predict(X_test, threshold=threshold, return_scores=True, **kwargs)
            
            # Calculate basic metrics
            metrics = {
                "anomaly_rate": float(np.mean(y_pred == -1)),
                "num_anomalies": int(np.sum(y_pred == -1)),
            }
            
            # Calculate classification metrics if true labels are provided
            if y_test is not None:
                from sklearn.metrics import (
                    precision_score,
                    recall_score,
                    f1_score,
                    accuracy_score,
                    roc_auc_score,
                )
                
                # Convert to binary format (0 for normal, 1 for anomaly)
                y_true_binary = (y_test == -1).astype(int)
                y_pred_binary = (y_pred == -1).astype(int)
                
                metrics.update({
                    "precision": float(precision_score(y_true_binary, y_pred_binary)),
                    "recall": float(recall_score(y_true_binary, y_pred_binary)),
                    "f1": float(f1_score(y_true_binary, y_pred_binary)),
                    "accuracy": float(accuracy_score(y_true_binary, y_pred_binary)),
                })
                
                # ROC AUC uses scores, not binary predictions
                try:
                    metrics["roc_auc"] = float(roc_auc_score(y_true_binary, -scores))
                except Exception as e:
                    logger.warning(f"Error calculating ROC AUC: {str(e)}")
            
            logger.info(f"Evaluation metrics for '{model.name}': {metrics}")
            
            return metrics
        
        except Exception as e:
            raise ModelError(
                f"Error evaluating anomaly model: {str(e)}", model_name=model.name
            )

    def detect_anomalies(
        self,
        model_name: str,
        model_type: str,
        X: pd.DataFrame,
        threshold: Optional[float] = None,
        from_deployment: bool = True,
        return_details: bool = True,
        **kwargs,
    ) -> Dict:
        """
        Detect anomalies in the data.
        
        Args:
            model_name: Name of the model to use.
            model_type: Type of the model.
            X: Feature dataframe.
            threshold: Threshold for anomaly detection.
            from_deployment: Whether to load model from deployment.
            return_details: Whether to return detailed anomaly information.
            **kwargs: Additional detection parameters.
        
        Returns:
            Dictionary with anomaly detection results.
        """
        logger.info(f"Detecting anomalies with model '{model_name}' in {len(X)} samples")
        
        try:
            # Load model
            model = self.model_service.load_model(
                model_name, model_type, from_deployment=from_deployment
            )
            
            # Get predictions and scores
            y_pred, scores = model.predict(X, threshold=threshold, return_scores=True, **kwargs)
            
            # Create basic results
            results = {
                "model_name": model_name,
                "model_type": model_type,
                "data_points": len(X),
                "anomaly_count": int(np.sum(y_pred == -1)),
                "anomaly_rate": float(np.mean(y_pred == -1)),
                "threshold": threshold or getattr(model, "threshold", None),
            }
            
            # Add detailed anomaly information if requested
            if return_details:
                anomalies = model.get_anomalies(X, threshold=threshold)
                
                if not anomalies.empty:
                    # Prepare anomaly data for JSON serialization
                    anomaly_data = []
                    
                    for idx, row in anomalies.iterrows():
                        anomaly_dict = {"index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx)}
                        
                        for col in row.index:
                            value = row[col]
                            
                            # Convert numpy and pandas types to Python types
                            if isinstance(value, (np.integer, np.floating)):
                                value = float(value)
                            elif isinstance(value, np.bool_):
                                value = bool(value)
                            elif pd.isna(value):
                                value = None
                            elif isinstance(value, (pd.Timestamp, datetime.datetime)):
                                value = value.isoformat()
                            
                            anomaly_dict[str(col)] = value
                        
                        anomaly_data.append(anomaly_dict)
                    
                    results["anomalies"] = anomaly_data
                    
                    # Add top features contributing to anomalies if available
                    if hasattr(model, "get_feature_importance"):
                        try:
                            feature_importance = model.get_feature_importance()
                            results["feature_importance"] = feature_importance.to_dict("records")
                        except Exception as e:
                            logger.warning(f"Error getting feature importance: {str(e)}")
            
            return results
        
        except Exception as e:
            raise ModelError(
                f"Error detecting anomalies: {str(e)}", model_name=model_name
            )
